/**
 * The MIT License (MIT)
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of this software
 * and associated documentation files (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge, publish, distribute,
 * sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all copies or
 * substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT
 * NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/**
 * @file cpu_device.hpp
 * @brief CpuDevice — naive single-core CPU implementation of IDevice.
 *
 * Execution model
 * ---------------
 * Nodes run sequentially on a worker thread in the topological order given
 * by DGraph::nodes. launch() returns immediately; wait() joins the worker.
 *
 * Buffer management
 * -----------------
 * Each GraphBuffer that appears as an output or RW-output in the graph is
 * backed by a heap-allocated byte array owned privately by this device.
 * Graph-level input buffers must be pre-populated by the user before
 * launch() via setInputBuffer().
 *
 * Kernel dispatch
 * ---------------
 * CPU kernels are plain C++ functions (or lambdas) registered before launch()
 * via registerKernel().
 *
 * Cross-device synchronisation and data movement
 * ----------------------------------------------
 * CpuDevice has no built-in notion of either: the compiler synthesises
 * `BridgeOpNode` entries (each carrying an opaque `std::function<void()>`
 * closure produced by an `IBridge`) directly into the device's per-device
 * `DGraph::nodes`. CpuDevice walks the node list with `std::visit` and
 * runs the closures inline. A typical CPU↔X bridge gives the CPU side a
 * closure that reads/writes the device's buffer storage via the public
 * setInputBuffer/getOutputBuffer/bufferSize accessors, capturing whatever
 * bridge-private staging and synchronisation primitives it owns.
 */

#ifndef VRT_GRAPH_DEVICE_CPU_DEVICE_HPP
#define VRT_GRAPH_DEVICE_CPU_DEVICE_HPP

#include <cstdint>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/node.hpp>
#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

// ---------------------------------------------------------------------------
// CpuBufferView — typed view into a buffer passed to kernel functions
// ---------------------------------------------------------------------------

/**
 * @brief Non-owning view of a buffer argument as seen by a CPU kernel function.
 */
struct CpuBufferView {
    void*      data       = nullptr;
    size_t     sizeBytes  = 0;
    BufferType elementType;

    template <typename T>
    T* as() const { return static_cast<T*>(data); }

    size_t elementCount() const;  // sizeBytes / sizeof(element); defined in .cpp
};

// ---------------------------------------------------------------------------
// CpuKernelArgs — context passed to every registered kernel function
// ---------------------------------------------------------------------------

/**
 * @brief Provides access to all bound arguments for a single kernel invocation.
 *
 * Buffer and scalar look-ups are by port name (matching the IOTypeMap
 * declaration).  Throws std::out_of_range on an unknown port name.
 */
class CpuKernelArgs {
   public:
    CpuKernelArgs(std::map<std::string, CpuBufferView> buffers,
                  std::map<std::string, uint64_t>       scalars)
        : buffers_(std::move(buffers)), scalars_(std::move(scalars)) {}

    /**
     * @brief Get a view of a buffer argument by port name.
     */
    const CpuBufferView& buffer(const std::string& portName) const {
        auto it = buffers_.find(portName);
        if (it == buffers_.end()) {
            throw std::out_of_range("CpuKernelArgs: unknown buffer port '" + portName + "'");
        }
        return it->second;
    }

    /**
     * @brief Get a scalar argument value by port name.
     */
    uint64_t scalar(const std::string& portName) const {
        auto it = scalars_.find(portName);
        if (it == scalars_.end()) {
            throw std::out_of_range("CpuKernelArgs: unknown scalar port '" + portName + "'");
        }
        return it->second;
    }

   private:
    std::map<std::string, CpuBufferView> buffers_;
    std::map<std::string, uint64_t>      scalars_;
};

using CpuKernelFn = std::function<void(const CpuKernelArgs&)>;

// ---------------------------------------------------------------------------
// CpuDevice
// ---------------------------------------------------------------------------

class CpuDevice : public IDevice {
   public:
    /**
     * @brief Construct a CpuDevice.
     *
     * @param id  Logical device id, e.g. "cpu" or "cpu:0".
     */
    explicit CpuDevice(std::string id);

    // --- Kernel registration (call before launch) ---

    /**
     * @brief Register a CPU kernel implementation.
     */
    void registerKernel(std::string kernelName, CpuKernelFn fn);

    // --- Buffer accessors (also used by bridges) ---

    /**
     * @brief Supply data for a graph-level input buffer (no producer node).
     *
     * Also used by bridges as the consumer-side write path on the CPU.
     */
    void setInputBuffer(const std::string& bufferName, const void* data, size_t sizeBytes);

    /**
     * @brief Read back an output buffer.
     *
     * Also used by bridges as the producer-side read path on the CPU.
     */
    void getOutputBuffer(const std::string& bufferName, void* data, size_t sizeBytes) const;

    /**
     * @brief Returns the current size of @p bufferName, or 0 if not present.
     */
    size_t bufferSize(const std::string& bufferName) const;

    // --- IDevice ---

    DeviceType  type() const override { return DeviceType::CPU; }
    std::string id()   const override { return id_; }

    void compile(const DGraph& dg) override;

    void launch() override;
    void wait() override;

   private:
    enum class NodeKind { Kernel, ProducerOp, ConsumerOp };

    struct NodeRuntime {
        std::string                id;
        NodeKind                   kind;
        size_t                     unmet = 0;
        std::vector<size_t>        successors;
        // Kernel payload (only meaningful when kind == Kernel)
        KernelNode                 kernel;
        // Op payloads (only meaningful when kind != Kernel)
        std::function<bool()>      tryReady;
        std::function<void()>      action;
    };

    void executeKernel(const KernelNode& node);

    CpuBufferView resolveBuffer(const std::string& name) const;
    std::vector<uint8_t>& ensureBuffer(const std::string& name, size_t sizeBytes);

    std::string                                  id_;
    std::map<std::string, CpuKernelFn>           kernels_;
    std::map<std::string, std::vector<uint8_t>>  buffers_;
    std::map<std::string, uint64_t>              scalarStore_;

    std::vector<NodeRuntime>                     runtime_;
    std::unordered_map<std::string, size_t>      idToIdx_;
    std::thread                                  worker_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_CPU_DEVICE_HPP
