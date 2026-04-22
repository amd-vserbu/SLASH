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
 * Nodes run sequentially on the calling thread in the topological order given
 * by DGraph::nodes.  launch() is synchronous and returns only after all nodes
 * have finished; wait() is therefore a no-op.
 *
 * Buffer management
 * -----------------
 * Each GraphBuffer token that appears as an output or RW-output in the graph is
 * backed by a heap-allocated byte array owned by the device.  Graph-level input
 * buffers must be pre-populated by the user before launch() is called (see
 * setInputBuffer()).
 *
 * Kernel dispatch
 * ---------------
 * CPU kernels are plain C++ functions (or lambdas) registered before launch()
 * via registerKernel().  When a node is executed the device:
 *  1. Builds a CpuKernelArgs from the node's IOMap (resolving buffer names to
 *     void* pointers and scalar names to uint64_t values).
 *  2. Looks up the registered function by KernelDescriptor::name.
 *  3. Calls it.
 *
 * Cross-device synchronisation
 * ----------------------------
 * Semaphores are represented as std::atomic<bool> stored in a shared
 * SemaphorePool.  The same pool must be passed to every device participating
 * in the graph so that signal/await pairs across devices work correctly.
 *
 * DMA transfers between devices are implemented as memcpy into/out of this
 * device's buffer store.  The source or destination buffer must have been
 * pre-registered or produced by a node on this device.
 */

#ifndef VRT_GRAPH_DEVICE_CPU_DEVICE_HPP
#define VRT_GRAPH_DEVICE_CPU_DEVICE_HPP

#include <atomic>
#include <cstdint>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/node.hpp>
#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

// ---------------------------------------------------------------------------
// SemaphorePool — shared cross-device semaphore store
// ---------------------------------------------------------------------------

/**
 * @brief Thread-safe pool of binary semaphores indexed by SemaphoreHandle::id.
 *
 * Pass a shared_ptr to the same SemaphorePool to every device that participates
 * in a graph.  The GraphCompiler allocates semaphore ids; the pool grows on
 * demand.
 *
 * await() busy-waits; this is intentionally naive.  A production implementation
 * would use a condition variable or OS primitive.
 */
class SemaphorePool {
   public:
    void signal(SemaphoreHandle sem) {
        getFlag(sem.id).store(true, std::memory_order_release);
    }

    void await(SemaphoreHandle sem) {
        auto& flag = getFlag(sem.id);
        while (!flag.load(std::memory_order_acquire)) {
            // busy-wait — acceptable for a naive single-core backend
        }
        flag.store(false, std::memory_order_relaxed);  // reset for reuse
    }

   private:
    std::atomic<bool>& getFlag(uint32_t id) {
        std::lock_guard<std::mutex> lk(mutex_);
        while (id >= flags_.size()) {
            flags_.emplace_back(std::make_unique<std::atomic<bool>>(false));
        }
        return *flags_[id];
    }

    std::mutex                                          mutex_;
    std::vector<std::unique_ptr<std::atomic<bool>>>    flags_;
};

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
     *
     * The raw uint64_t bits can be reinterpreted by the caller as needed
     * (e.g. bit_cast to float for ScalarType::F32).
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
     * @param id     Logical device id, e.g. "cpu" or "cpu:0".
     * @param pool   Shared semaphore pool; pass the same instance to all
     *               devices in the graph for cross-device sync to work.
     *               If nullptr a private pool is created (single-device graphs).
     * @param buffers Shared buffer store; pass the same instance to all
     *               devices in the graph for cross-device DMA to work.
     *               If nullptr a private store is created.
     */
    explicit CpuDevice(std::string id,
                       std::shared_ptr<SemaphorePool> pool = nullptr,
                       std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> buffers = nullptr);

    // --- Kernel registration (call before launch) ---

    /**
     * @brief Register a CPU kernel implementation.
     *
     * @param kernelName  Must match KernelDescriptor::name for the node(s) that
     *                    should execute this function.
     * @param fn          Function called with resolved args on each invocation.
     */
    void registerKernel(std::string kernelName, CpuKernelFn fn);

    // --- Pre-populated input buffers ---

    /**
     * @brief Supply data for a graph-level input buffer (no producer node).
     *
     * The backend copies @p sizeBytes bytes from @p data into its internal store
     * under the given buffer name.  Must be called before launch().
     *
     * @param bufferName  Matches GraphBuffer::name() for the token returned by
     *                    Graph::inputBuffer().
     * @param data        Source data pointer (host memory).
     * @param sizeBytes   Number of bytes to copy.
     */
    void setInputBuffer(const std::string& bufferName, const void* data, size_t sizeBytes);

    /**
     * @brief Read back an output buffer after launch() completes.
     *
     * @param bufferName  GraphBuffer::name() of an output or RW-output buffer.
     * @param data        Destination pointer (host memory); must hold at least
     *                    the number of bytes written by the producing kernel.
     * @param sizeBytes   Number of bytes to copy out.
     */
    void getOutputBuffer(const std::string& bufferName, void* data, size_t sizeBytes) const;

    // --- IDevice ---

    DeviceType  type() const override { return DeviceType::CPU; }
    std::string id()   const override { return id_; }

    void compile(const DGraph& dg) override;

    void insertSignal(SemaphoreHandle sem, const std::string& afterNodeId) override;
    void insertAwait(SemaphoreHandle sem, const std::string& beforeNodeId)  override;
    void insertDMA(DMADescriptor dma, const std::string& beforeNodeId)      override;

    /**
     * @brief Execute all nodes on a background thread.
     *
     * Returns immediately; call wait() to block until completion.
     */
    void launch() override;

    /**
     * @brief Block until the background thread completes.
     */
    void wait() override;

   private:
    // Internal step types injected by the compiler
    struct SignalStep { SemaphoreHandle sem; };
    struct AwaitStep  { SemaphoreHandle sem; };
    struct DMAStep    { DMADescriptor   dma; };
    struct KernelStep { Node node; };

    using Step = std::variant<SignalStep, AwaitStep, DMAStep, KernelStep>;

    void executeKernel(const Node& node);
    void executeDMA(const DMADescriptor& dma);

    CpuBufferView resolveBuffer(const std::string& name) const;

    std::vector<uint8_t>& ensureBuffer(const std::string& name, size_t sizeBytes);

    std::string                                  id_;
    std::shared_ptr<SemaphorePool>               semPool_;
    std::map<std::string, CpuKernelFn>           kernels_;
    std::shared_ptr<std::map<std::string, std::vector<uint8_t>>>  buffers_;     // name → heap storage
    std::map<std::string, uint64_t>              scalarStore_; // named global variables

    // Pending injected steps collected before compile() is called
    struct PendingSignal { SemaphoreHandle sem; std::string afterNodeId; };
    struct PendingAwait  { SemaphoreHandle sem; std::string beforeNodeId; };
    struct PendingDMA    { DMADescriptor   dma; std::string beforeNodeId; };

    std::vector<PendingSignal> pendingSignals_;
    std::vector<PendingAwait>  pendingAwaits_;
    std::vector<PendingDMA>    pendingDMAs_;

    // Compiled execution plan (nodes + injected sync/DMA steps in order)
    std::vector<Step> steps_;

    // Background execution thread
    std::thread worker_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_CPU_DEVICE_HPP
