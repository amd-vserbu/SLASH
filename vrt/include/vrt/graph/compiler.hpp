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
 * @file compiler.hpp
 * @brief GraphCompiler — translates a Graph into a list of per-device DGraphs.
 *
 * GraphCompiler is an internal component; users interact with Graph::run() /
 * Graph::launch() which invoke the compiler transparently.
 *
 * Compilation steps:
 *  1. Topological sort of all Nodes using data-dependency edges (derived from
 *     IOMap buffer tokens) and explicit afterNodes edges.
 *  2. Group nodes by deviceHint → one DGraph per device.
 *  3. For each cross-device buffer edge: allocate a SemaphoreHandle, call
 *     insertSignal() on the producer backend and insertAwait() on the consumer
 *     backend; call insertDMA() on whichever backend prefersDMAInitiation().
 *  4. Call backend.compile(dg) for each DGraph.
 */

#ifndef VRT_GRAPH_COMPILER_HPP
#define VRT_GRAPH_COMPILER_HPP

#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/crossdevice/bridge_router.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/node/node.hpp>

namespace vrt::graph {

class Graph;  // forward; compiler reads graph internals via a friend accessor

class GraphCompiler {
   public:
    /**
     * @brief Compile a Graph given the registered backends.
     *
     * @param nodes     Nodes in insertion order (will be topologically sorted internally).
     * @param devices   Map of device-id → device, as registered with Graph.
     * @param bridges  Map of device-type pair → cross-device backend (optional;
     *                       empty map disables cross-device sync injection).
     * @return          One DGraph per device, each with its nodes in execution order.
     *
     * @throws std::runtime_error  If the graph contains a cycle, an unbound mandatory
     *                             port, or a deviceHint that has no matching device.
     */
    std::vector<DGraph> compile(
        const std::vector<Node>&                               nodes,
        const std::map<std::string, std::shared_ptr<IDevice>>& devices,
        const std::map<std::pair<DeviceType, DeviceType>,
                       std::shared_ptr<IBridge>>& bridges = {});

   private:
    // --- Topology ---

    /**
     * @brief Build the adjacency list (node-id → set of successor node-ids) from
     *        data-dependency edges and explicit afterNodes edges.
     */
    std::map<std::string, std::vector<std::string>> buildAdjacency(
        const std::vector<Node>& nodes) const;

    /**
     * @brief Kahn's algorithm topological sort.
     *
     * @throws std::runtime_error on cycle detection.
     */
    std::vector<std::string> topoSort(
        const std::vector<Node>&                                 nodes,
        const std::map<std::string, std::vector<std::string>>&   adj) const;

    // --- Buffer token → producer node mapping ---

    /**
     * @brief Build a map from GraphBuffer name → the node id that produced it.
     *
     * Graph-level input buffers (no producer) are absent from this map.
     */
    std::map<std::string, std::string> buildProducerMap(
        const std::vector<Node>& nodes) const;

    // --- Cross-device sync ---

    /**
     * @brief Allocate a fresh semaphore id (monotonically increasing).
     */
    SemaphoreHandle allocSemaphore() { return {nextSemId_++}; }

    /**
     * @brief For a cross-device edge producer→consumer, inject the appropriate
     *        semaphore and DMA nodes into both devices.
     *
     * Delegates to BridgeRouter::routeTransfer which uses the registered
     * cross-device backends or bounces via CPU if no direct path exists.
     */
    void injectCrossDeviceSync(IDevice& producer,
                               IDevice& consumer,
                               const GraphBuffer& buf,
                               uint64_t sizeHint,
                               const std::map<std::pair<DeviceType, DeviceType>,
                                              std::shared_ptr<IBridge>>& bridges,
                               IDevice& cpuDevice,
                               const std::string& producerNodeId,
                               const std::string& consumerNodeId);

    uint32_t nextSemId_ = 0;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_COMPILER_HPP
