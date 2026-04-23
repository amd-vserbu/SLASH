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
 *  1. Topological sort of all KernelNodes using data-dependency edges (derived
 *     from IOMap buffer tokens) and explicit afterNodes edges.
 *  2. Group kernels by deviceHint → one DGraph per device.
 *  3. For each cross-device buffer edge: ask BridgeRouter for a `RoutedLeg`
 *     (one direct hop or two via the CPU bounce). Each leg's `BridgeStepPair`
 *     is materialised as a producer-side and a consumer-side `BridgeOpNode`
 *     and spliced into the corresponding DGraphs.
 *  4. Call backend.compile(dg) for each DGraph.
 */

#ifndef VRT_GRAPH_COMPILER_HPP
#define VRT_GRAPH_COMPILER_HPP

#include <map>
#include <memory>
#include <functional>
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
     * @brief Lookup callback used by the compiler to obtain a bridge
     *        instance for a concrete (srcDeviceId, dstDeviceId) pair.
     *
     * Implementations are expected to lazily instantiate one bridge per
     * pair from registered factories and cache the result. `Graph` wraps
     * its `bridgeFor()` method as this callback.
     */
    using BridgeFor =
        std::function<IBridge&(const std::string& srcDevId,
                               const std::string& dstDevId)>;

    /**
     * @brief Compile a Graph given the registered backends.
     *
     * @param nodes     Kernel nodes in insertion order (will be topologically
     *                  sorted internally).
     * @param devices   Map of device-id → device, as registered with Graph.
     * @param bridgeFor Lookup that returns (lazily creating if needed) the
     *                  bridge instance handling a concrete pair of device ids.
     *                  The compiler invokes this once per cross-device edge it
     *                  materialises.
     * @return          One DGraph per device, each with its nodes (kernel +
     *                  bridge ops) in execution order.
     *
     * @throws std::runtime_error  If the graph contains a cycle, an unbound
     *                             mandatory port, a deviceHint with no
     *                             matching device, or `bridgeFor` rejects a
     *                             needed pair.
     */
    std::vector<DGraph> compile(
        const std::vector<KernelNode>&                         nodes,
        const std::map<std::string, std::shared_ptr<IDevice>>& devices,
        const BridgeFor&                                       bridgeFor,
        const std::shared_ptr<std::map<std::string, uint64_t>>& scalarValues);

   private:
    // --- Topology ---

    /**
     * @brief Build the adjacency list (node-id → set of successor node-ids) from
     *        data-dependency edges and explicit afterNodes edges.
     */
    std::map<std::string, std::vector<std::string>> buildAdjacency(
        const std::vector<KernelNode>& nodes) const;

    /**
     * @brief Kahn's algorithm topological sort.
     *
     * @throws std::runtime_error on cycle detection.
     */
    std::vector<std::string> topoSort(
        const std::vector<KernelNode>&                           nodes,
        const std::map<std::string, std::vector<std::string>>&   adj) const;

    // --- Buffer token → producer node mapping ---

    /**
     * @brief Build a map from GraphBuffer name → the kernel-node id that
     *        produced it.
     *
     * Graph-level input buffers (no producer) are absent from this map.
     */
    std::map<std::string, std::string> buildProducerMap(
        const std::vector<KernelNode>& nodes) const;

    /**
     * @brief Build a map from graph-global scalar name → the kernel-node id
     *        that writes it.
     *
     * Only typed output scalar ports participate.
     */
    std::map<std::string, std::string> buildScalarProducerMap(
        const std::vector<KernelNode>& nodes) const;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_COMPILER_HPP
