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
 * @file bridge_router.hpp
 * @brief BridgeRouter — routes cross-device transfers through direct
 *        bridges or bounces via the CPU when no direct path exists.
 *
 * Used by the compiler's injectCrossDeviceSync to transparently handle the
 * case where two devices have no dedicated bridge.  In that case
 * the transfer is split into two hops: src → CPU, then CPU → dst, using the
 * mandatory {CPU, SrcType} and {CPU, DstType} bridges.
 */

#ifndef VRT_GRAPH_CROSSDEVICE_BRIDGE_ROUTER_HPP
#define VRT_GRAPH_CROSSDEVICE_BRIDGE_ROUTER_HPP

#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>

namespace vrt::graph {

class BridgeRouter {
   public:
    using BridgeMap = std::map<std::pair<DeviceType, DeviceType>,
                              std::shared_ptr<IBridge>>;

    /**
     * @brief Route a cross-device buffer transfer, bouncing via CPU if needed.
     *
     * If a direct IBridge exists for {src.type(), dst.type()}, it
     * is used.  Otherwise the transfer is split into two legs:
     *   1. src → cpuDevice  (using the {CPU, src.type()} bridge)
     *   2. cpuDevice → dst  (using the {CPU, dst.type()} bridge)
     *
     * The intermediate bounce buffer is named deterministically so the CPU
     * backend can lazily allocate it.
     *
     * @param src             Producing device.
     * @param dst             Consuming device.
     * @param buffer          Buffer token being transferred.
     * @param sizeHintBytes   Expected size (0 = unknown).
     * @param crossBackends   Registered bridges (canonical keys).
     * @param cpuDevice       CPU device (bounce target).
     * @param allocSemaphore  Callback to allocate fresh SemaphoreHandles.
     */
    static void routeTransfer(
        IDevice&                                 src,
        IDevice&                                 dst,
        const GraphBuffer&                       buffer,
        uint64_t                                 sizeHintBytes,
        const BridgeMap&                          bridges,
        IDevice&                                 cpuDevice,
        std::function<SemaphoreHandle()>          allocSemaphore,
        const std::string&                       producerNodeId,
        const std::string&                       consumerNodeId)
    {
        auto directKey = IBridge::makeKey(src.type(), dst.type());
        auto directIt  = bridges.find(directKey);

        if (directIt != bridges.end()) {
            directIt->second->injectTransfer(src, dst, buffer,
                                             sizeHintBytes, allocSemaphore,
                                             producerNodeId, consumerNodeId);
            return;
        }

        // Bounce via CPU: src → cpu, then cpu → dst.
        auto srcCpuKey = IBridge::makeKey(src.type(), DeviceType::CPU);
        auto srcCpuIt  = bridges.find(srcCpuKey);
        if (srcCpuIt == bridges.end()) {
            throw std::runtime_error(
                "BridgeRouter: no bridge for {" + src.id() +
                ", cpu} — cannot bounce transfer of buffer '" +
                buffer.name() + "'");
        }

        auto cpuDstKey = IBridge::makeKey(DeviceType::CPU, dst.type());
        auto cpuDstIt  = bridges.find(cpuDstKey);
        if (cpuDstIt == bridges.end()) {
            throw std::runtime_error(
                "BridgeRouter: no bridge for {cpu, " + dst.id() +
                "} — cannot bounce transfer of buffer '" +
                buffer.name() + "'");
        }

        // Intermediate bounce buffer lives on the CPU.
        GraphBuffer bounceBuffer("__bounce_" + buffer.name() + "_" +
                                 src.id() + "_" + dst.id());

        // Leg 1: src → cpu
        srcCpuIt->second->injectTransfer(src, cpuDevice, buffer,
                                         sizeHintBytes, allocSemaphore,
                                         producerNodeId, consumerNodeId);

        // Leg 2: cpu → dst
        cpuDstIt->second->injectTransfer(cpuDevice, dst, bounceBuffer,
                                         sizeHintBytes, allocSemaphore,
                                         producerNodeId, consumerNodeId);
    }
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CROSSDEVICE_BRIDGE_ROUTER_HPP
