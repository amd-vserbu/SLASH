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
 * @file bridge.hpp
 * @brief IBridge — abstract interface for cross-device transfer and
 *        synchronisation between two devices.
 *
 * A bridge is instantiated for every device pair and is responsible for
 * injecting the semaphore and DMA operations needed to move data between
 * two devices.
 *
 * Concrete implementations: CpuFpgaBridge, FpgaFpgaBridge, CpuGpuBridge, …
 */

#ifndef VRT_GRAPH_CROSSDEVICE_BRIDGE_HPP
#define VRT_GRAPH_CROSSDEVICE_BRIDGE_HPP

#include <algorithm>
#include <cstdint>
#include <functional>
#include <utility>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

/**
 * @brief Abstract interface for cross-device transfer and synchronisation.
 *
 * Registered on Graph via registerBridge().  The compiler looks up the
 * appropriate IBridge for every cross-device buffer edge and
 * delegates the injection of semaphores and DMAs to it.
 *
 * A device pair {A, B} is always stored in canonical order (lower enum value
 * first) so that lookup is direction-independent.
 */
class IBridge {
   public:
    virtual ~IBridge() = default;

    /**
     * @brief Returns the canonical device-type pair handled by this bridge.
     *
     * The first element has the lower (or equal) enum value.
     */
    virtual std::pair<DeviceType, DeviceType> devicePair() const = 0;

    /**
     * @brief Inject all semaphore and DMA operations needed to transfer a
     *        buffer from @p src to @p dst.
     *
     * Called by the compiler (or BridgeRouter) for every cross-device
     * buffer edge.  The implementation should call insertSignal / insertAwait /
     * insertDMA on the appropriate devices, using the node-id parameters to
     * associate sync operations with the correct kernel nodes.
     *
     * @param src             The producing device.
     * @param dst             The consuming device.
     * @param buffer          The buffer token being transferred.
     * @param sizeHintBytes   Expected transfer size (0 = unknown).
     * @param allocSemaphore  Callback to allocate a fresh SemaphoreHandle;
     *                        ownership of id assignment stays with the compiler.
     * @param producerNodeId  Node id on src that produces the buffer.
     * @param consumerNodeId  Node id on dst that consumes the buffer.
     */
    virtual void injectTransfer(IDevice&                          src,
                                IDevice&                          dst,
                                const GraphBuffer&                buffer,
                                uint64_t                          sizeHintBytes,
                                std::function<SemaphoreHandle()>  allocSemaphore,
                                const std::string&                producerNodeId,
                                const std::string&                consumerNodeId) = 0;

    // --- Helpers ---

    /**
     * @brief Build the canonical key for a device-type pair (lower enum first).
     */
    static std::pair<DeviceType, DeviceType> makeKey(DeviceType a, DeviceType b) {
        if (static_cast<int>(a) <= static_cast<int>(b)) return {a, b};
        return {b, a};
    }
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CROSSDEVICE_BRIDGE_HPP
