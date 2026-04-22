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
 * @file node.hpp
 * @brief Node — a kernel instantiation within a Graph.
 *
 * Each Node pairs a KernelDescriptor (what to run) with an IOMap (the actual
 * data it operates on), a device hint (where to run it), and optional explicit
 * ordering constraints (afterNodes).
 *
 * Most ordering edges are derived automatically from data dependencies in the
 * IOMap (if node B's input buffer was produced by node A, an edge A→B exists).
 * afterNodes adds additional edges without data flow.
 *
 * Nodes are created and owned by Graph; users receive their string id.
 */

#ifndef VRT_GRAPH_NODE_NODE_HPP
#define VRT_GRAPH_NODE_NODE_HPP

#include <string>
#include <vector>

#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>

namespace vrt::graph {

struct Node {
    /**
     * @brief Unique node identifier within the owning Graph.
     *
     * Auto-assigned by Graph::addNode() unless the user supplies one (future
     * extension).  Format: "<kernelName>_<monotonic index>".
     */
    std::string id;

    /**
     * @brief The kernel type/signature this node instantiates.
     */
    KernelDescriptor kernel;

    /**
     * @brief Device placement hint.
     *
     * Identifies the specific device instance, e.g. "fpga:0", "gpu:1", "cpu".
     * An empty string means "any device that satisfies kernel.type".
     * The GraphCompiler validates that a registered IDevice matches.
     */
    std::string deviceHint;

    /**
     * @brief Concrete data bindings for this instantiation.
     */
    IOMap ioMap;

    /**
     * @brief Explicit ordering constraints.
     *
     * This node will not start until every node listed here has completed,
     * even if there is no data dependency between them.
     */
    std::vector<std::string> afterNodes;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_NODE_NODE_HPP
