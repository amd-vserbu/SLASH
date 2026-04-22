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
 * @file dgraph.hpp
 * @brief DGraph — per-device compiled subgraph produced by GraphCompiler.
 *
 * A DGraph is the output of the compilation step for a single device.  It
 * contains:
 *  - The ordered list of Nodes assigned to that device (data-kernel nodes only;
 *    sync/DMA operations are tracked separately via the device's insertSignal /
 *    insertAwait / insertDMA interface).
 *  - A pointer to the IDevice responsible for executing the subgraph.
 *
 * DGraph is an internal compiler artifact; it is not part of the user-facing API.
 */

#ifndef VRT_GRAPH_DEVICE_DGRAPH_HPP
#define VRT_GRAPH_DEVICE_DGRAPH_HPP

#include <memory>
#include <string>
#include <vector>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/node/node.hpp>

namespace vrt::graph {

struct DGraph {
    /**
     * @brief ID of the device this subgraph targets (matches IDevice::id()).
     */
    std::string deviceId;

    /**
     * @brief Nodes assigned to this device, in topological order.
     */
    std::vector<Node> nodes;

    /**
     * @brief The device that will compile and execute this subgraph.
     */
    std::shared_ptr<IDevice> device;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_DGRAPH_HPP
