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
 * Storage ownership
 * -----------------
 * `CpuDevice` is split between two concerns:
 *   - The device itself owns *device-resident* state: the registered
 *     `CpuKernel` table and the per-buffer-name byte storage. Both are
 *     *shared across every plan compiled by this device*. Per-scope buffer
 *     keys (`scope:N:name`) keep different graph regions from colliding;
 *     two plans for the same Graph naturally observe the same logical
 *     buffer when they reference the same scoped name, which mirrors
 *     accelerator hardware (a buffer "lives on" the device).
 *   - Each `CpuDevicePlan` owns its own *graph-level scalar map*, taken
 *     from `DGraph::scalarValues` at compile time. Scalar state is owned
 *     by the `Graph` and threaded into every device plan, so two plans
 *     for the same Graph see one shared scalar map and two plans for
 *     different Graphs see independent maps.
 *
 * Execution model
 * ---------------
 * Nodes run sequentially on a worker thread owned by the compiled CPU plan.
 * launch() returns immediately; wait() joins the worker. The plan also
 * runs kernel dispatch, boundary copies, and control-flow execution; the
 * device is consulted only as a kernel registry and a buffer store.
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
 * CPU kernels are CpuKernel objects registered before launch() via
 * registerKernel(). Each kernel owns its name, typed IO signature, and call
 * implementation.
 *
 * Cross-device synchronisation and data movement
 * ----------------------------------------------
 * CpuDevice has no built-in notion of either: the compiler synthesises
 * `CompiledBridgeOpNode` entries (each carrying an opaque `std::function<void()>`
 * closure produced by an `IBridge`) directly into the device's per-device
 * `DGraph::nodes`. CpuDevicePlan walks the node list with `std::visit` and
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
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/graph/node/compiled_node.hpp>
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
                                    std::map<std::string, uint64_t>       scalars,
                                    std::map<std::string, uint64_t*>      writableScalars = {})
                : buffers_(std::move(buffers)),
                    scalars_(std::move(scalars)),
                    writableScalars_(std::move(writableScalars)) {}

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
        if (it != scalars_.end()) {
            return it->second;
        }

        auto writableIt = writableScalars_.find(portName);
        if (writableIt != writableScalars_.end()) {
            return *writableIt->second;
        }

        throw std::out_of_range("CpuKernelArgs: unknown scalar port '" + portName + "'");
    }

    /**
     * @brief Write an output scalar argument by port name.
     */
    void setScalar(const std::string& portName, uint64_t value) const {
        auto it = writableScalars_.find(portName);
        if (it == writableScalars_.end()) {
            throw std::out_of_range(
                "CpuKernelArgs: unknown writable scalar port '" + portName + "'");
        }
        *it->second = value;
    }

   private:
    std::map<std::string, CpuBufferView> buffers_;
    std::map<std::string, uint64_t>      scalars_;
    std::map<std::string, uint64_t*>     writableScalars_;
};

// ---------------------------------------------------------------------------
// CpuKernel — polymorphic CPU kernel implementation
// ---------------------------------------------------------------------------

class CpuKernel {
   public:
    virtual ~CpuKernel() = default;

    /** @brief Logical kernel name used to match KernelDescriptor::name. */
    virtual const std::string& name() const = 0;

    /** @brief Typed I/O signature for this CPU kernel. */
    virtual const IOTypeMap& ioTypeMap() const = 0;

    /** @brief Execute one graph node invocation. */
    virtual void call(const CpuKernelArgs& args) = 0;

    /** @brief Convenience descriptor for Graph::addNode(). */
    KernelDescriptor descriptor() const {
        return KernelDescriptor{name(), DeviceType::CPU, std::nullopt, ioTypeMap()};
    }
};

// ---------------------------------------------------------------------------
// CpuDevice
// ---------------------------------------------------------------------------

class CpuDevicePlan;

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
    void registerKernel(std::shared_ptr<CpuKernel> kernel);

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

    std::unique_ptr<IDevicePlan> compilePlan(const DGraph& dg) override;

   private:
    friend class CpuDevicePlan;

    std::string                                  id_;
    std::map<std::string, std::shared_ptr<CpuKernel>> kernels_;
    std::map<std::string, std::vector<uint8_t>>  buffers_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_CPU_DEVICE_HPP
