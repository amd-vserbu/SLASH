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
 * @file device.hpp
 * @brief IDevice — abstract execution interface for a single device instance.
 *
 * The GraphCompiler partitions the user's Graph into one DGraph per device and
 * hands each DGraph to the corresponding IDevice for compilation and
 * execution.  Cross-device synchronisation primitives (semaphores, DMA) are
 * injected by the compiler via insertSignal / insertAwait / insertDMA before
 * compile() is called.
 *
 * Concrete implementations (FpgaDevice, GpuDevice, CpuDevice) are provided
 * separately and are not part of this abstract header.
 */

#ifndef VRT_GRAPH_DEVICE_DEVICE_HPP
#define VRT_GRAPH_DEVICE_DEVICE_HPP

#include <cstdint>
#include <string>

#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

struct DGraph;  // forward declaration; defined in dgraph.hpp

/**
 * @brief Opaque handle to a cross-device semaphore allocated by the compiler.
 */
struct SemaphoreHandle {
    uint32_t id;
};

/**
 * @brief Describes a DMA transfer between two GraphBuffers on different devices.
 *
 * The compiler allocates DMADescriptors for every cross-device buffer flow and
 * passes them to the appropriate device via insertDMA().  Which device
 * "owns" the transfer (initiates it) is a policy choice left to the device;
 * devices may expose a prefersDMAInitiation() hint to guide that decision.
 */
struct DMADescriptor {
    GraphBuffer src;        ///< Source buffer token
    GraphBuffer dst;        ///< Destination buffer token
    uint64_t    sizeBytes;  ///< Number of bytes to transfer
};

/**
 * @brief Abstract interface for a device execution backend.
 *
 * One instance per physical device (or logical device group, e.g. all CPUs).
 * Registered with Graph::registerDevice() before compilation.
 */
class IDevice {
   public:
    virtual ~IDevice() = default;

    // --- Identity ---

    /** @brief Returns the device type (CPU, GPU, FPGA). */
    virtual DeviceType type() const = 0;

    /**
     * @brief Returns the unique device identifier, e.g. "fpga:0", "gpu:1", "cpu".
     *
     * This string is matched against Node::deviceHint during compilation.
     */
    virtual std::string id() const = 0;

    // --- Compilation ---

    /**
     * @brief Compile the per-device subgraph.
     *
     * Called by GraphCompiler after all sync/DMA nodes have been injected via
     * insertSignal / insertAwait / insertDMA.  The device may decompose the
     * DGraph further (e.g. FPGA graphlets, HIP graph segments).
     *
     * @param dg  The per-device subgraph to compile.
     */
    virtual void compile(const DGraph& dg) = 0;

    // --- Cross-device sync primitives (called before compile()) ---

    /**
     * @brief Inject a semaphore signal after the node identified by @p afterNodeId.
     *
     * @param sem          Semaphore to signal.
     * @param afterNodeId  Node id on this device after which the signal occurs.
     */
    virtual void insertSignal(SemaphoreHandle sem, const std::string& afterNodeId) = 0;

    /**
     * @brief Inject a semaphore await before the node identified by @p beforeNodeId.
     *
     * @param sem           Semaphore to await.
     * @param beforeNodeId  Node id on this device before which the await occurs.
     */
    virtual void insertAwait(SemaphoreHandle sem, const std::string& beforeNodeId) = 0;

    /**
     * @brief Inject a DMA transfer before the node identified by @p beforeNodeId.
     *
     * @param dma           DMA descriptor.
     * @param beforeNodeId  Node id on this device before which the DMA executes.
     */
    virtual void insertDMA(DMADescriptor dma, const std::string& beforeNodeId) = 0;

    /**
     * @brief Hint used by the compiler to decide which device initiates DMA.
     *
     * Return true if this device prefers to push data to the remote device
     * (e.g. a GPU that has an efficient peer-write path).  Return false if the
     * remote device should pull.  Defaults to false.
     */
    virtual bool prefersDMAInitiation() const { return false; }

    // --- Execution ---

    /**
     * @brief Start asynchronous execution of the compiled subgraph.
     *
     * Called by Graph::launch() after all devices have been compiled.
     */
    virtual void launch() = 0;

    /**
     * @brief Block until this device's subgraph has completed.
     *
     * Called by Graph::wait() (or Graph::run() implicitly).
     */
    virtual void wait() = 0;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_DEVICE_HPP
