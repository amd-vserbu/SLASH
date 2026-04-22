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
 * @file cpu_fpga_bridge.hpp
 * @brief CpuFpgaBridge — bridge for CPU ↔ FPGA transfers.
 *
 * Handles semaphore and DMA injection for buffer edges that cross the
 * CPU / FPGA boundary.  DMA ownership is determined by
 * IDevice::prefersDMAInitiation(); if neither side prefers, the
 * destination device initiates the transfer (pull model).
 */

#ifndef VRT_GRAPH_CROSSDEVICE_CPU_FPGA_BRIDGE_HPP
#define VRT_GRAPH_CROSSDEVICE_CPU_FPGA_BRIDGE_HPP

#include <vrt/graph/crossdevice/bridge.hpp>

namespace vrt::graph {

class CpuFpgaBridge : public IBridge {
   public:
    CpuFpgaBridge() = default;

    std::pair<DeviceType, DeviceType> devicePair() const override {
        return {DeviceType::CPU, DeviceType::FPGA};
    }

    void injectTransfer(IDevice&                          src,
                        IDevice&                          dst,
                        const GraphBuffer&                buffer,
                        uint64_t                          sizeHintBytes,
                        std::function<SemaphoreHandle()>  allocSemaphore,
                        const std::string&                producerNodeId,
                        const std::string&                consumerNodeId) override;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_CROSSDEVICE_CPU_FPGA_BRIDGE_HPP
