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

#include <vrt/graph/compiler.hpp>

#include <algorithm>
#include <limits>
#include <queue>
#include <set>
#include <utility>

namespace vrt::graph {

// ---------------------------------------------------------------------------
// buildProducerMap
// ---------------------------------------------------------------------------

std::map<std::string, std::string> GraphCompiler::buildProducerMap(
    const std::vector<KernelNode>& nodes) const {
    std::map<std::string, std::string> producers;
    for (const auto& node : nodes) {
        for (const auto& [port, buf] : node.ioMap.outputBuffers()) {
            producers[buf.name()] = node.id;
        }
        for (const auto& rw : node.ioMap.rwBuffers()) {
            producers[rw.out.name()] = node.id;
        }
    }
    return producers;
}

// ---------------------------------------------------------------------------
// buildAdjacency
// ---------------------------------------------------------------------------

std::map<std::string, std::vector<std::string>> GraphCompiler::buildAdjacency(
    const std::vector<KernelNode>& nodes) const {
    auto producers = buildProducerMap(nodes);

    std::map<std::string, std::vector<std::string>> adj;

    for (const auto& node : nodes) {
        adj.emplace(node.id, std::vector<std::string>{});
    }

    for (const auto& node : nodes) {
        for (const auto& [port, buf] : node.ioMap.inputBuffers()) {
            auto it = producers.find(buf.name());
            if (it != producers.end()) {
                adj[it->second].push_back(node.id);
            }
        }
        for (const auto& rw : node.ioMap.rwBuffers()) {
            auto it = producers.find(rw.in.name());
            if (it != producers.end()) {
                adj[it->second].push_back(node.id);
            }
        }
        for (const auto& after : node.afterNodes) {
            adj[after].push_back(node.id);
        }
    }

    return adj;
}

// ---------------------------------------------------------------------------
// topoSort — Kahn's algorithm
// ---------------------------------------------------------------------------

std::vector<std::string> GraphCompiler::topoSort(
    const std::vector<KernelNode>& nodes,
    const std::map<std::string, std::vector<std::string>>& adj) const {

    std::map<std::string, int> inDegree;
    for (const auto& node : nodes) {
        inDegree[node.id] = 0;
    }
    for (const auto& [src, successors] : adj) {
        for (const auto& dst : successors) {
            ++inDegree[dst];
        }
    }

    std::queue<std::string> ready;
    for (const auto& node : nodes) {
        if (inDegree[node.id] == 0) {
            ready.push(node.id);
        }
    }

    std::vector<std::string> sorted;
    sorted.reserve(nodes.size());

    while (!ready.empty()) {
        auto current = ready.front();
        ready.pop();
        sorted.push_back(current);

        auto it = adj.find(current);
        if (it != adj.end()) {
            for (const auto& succ : it->second) {
                if (--inDegree[succ] == 0) {
                    ready.push(succ);
                }
            }
        }
    }

    if (sorted.size() != nodes.size()) {
        throw std::runtime_error(
            "GraphCompiler: cycle detected in dependency graph");
    }

    return sorted;
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

std::vector<DGraph> GraphCompiler::compile(
    const std::vector<KernelNode>& nodes,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const BridgeFor&                                       bridgeFor) {

    // 1. Topological sort.
    auto adj       = buildAdjacency(nodes);
    auto sortedIds = topoSort(nodes, adj);

    // KernelNode lookup by id.
    std::map<std::string, const KernelNode*> nodeById;
    for (const auto& node : nodes) {
        nodeById[node.id] = &node;
    }

    // 2. Resolve device placement for each kernel.
    std::map<std::string, std::string> nodeDevice;  // node-id → device-id
    for (const auto& id : sortedIds) {
        const KernelNode* node = nodeById[id];
        if (!node->deviceHint.empty()) {
            if (devices.find(node->deviceHint) == devices.end()) {
                throw std::runtime_error(
                    "GraphCompiler: deviceHint '" + node->deviceHint +
                    "' for node '" + id + "' does not match any registered device");
            }
            nodeDevice[id] = node->deviceHint;
        } else {
            std::string found;
            for (const auto& [did, dev] : devices) {
                if (dev->type() == node->kernel.type) {
                    found = did;
                    break;
                }
            }
            if (found.empty()) {
                throw std::runtime_error(
                    "GraphCompiler: no device of type matching kernel '" +
                    node->kernel.name + "' for node '" + id + "'");
            }
            nodeDevice[id] = found;
        }
    }

    // 3. Group kernels per device in topological order.
    //
    // Build an ordered list of KernelNodes per device id; we splice
    // BridgeOpNodes around them in step 4.
    std::map<std::string, std::vector<KernelNode>> kernelsByDevice;
    for (const auto& [did, dev] : devices) {
        kernelsByDevice[did];  // ensure entry exists
    }
    for (const auto& id : sortedIds) {
        kernelsByDevice[nodeDevice[id]].push_back(*nodeById[id]);
    }

    // 4. Synthesise BridgeOpNodes for every cross-device buffer edge.
    auto producerMap = buildProducerMap(nodes);

    // Find a CPU device for bounce routing (if any).
    IDevice* cpuDevice = nullptr;
    for (const auto& [did, dev] : devices) {
        if (dev->type() == DeviceType::CPU) {
            cpuDevice = dev.get();
            break;
        }
    }

    // For each device, accumulate the bridge ops to insert before/after each
    // kernel index in kernelsByDevice[did].
    struct DeviceInsertions {
        // index into kernelsByDevice[did] → ordered list of ops to splice
        std::map<size_t, std::vector<BridgeOpNode>> beforeKernel;
        std::map<size_t, std::vector<BridgeOpNode>> afterKernel;
        // Bridge ops with no anchor kernel on this device (bounce-route
        // intermediaries living on the cpu when neither endpoint kernel
        // is on the cpu). Appended to the DGraph after the kernel
        // sequence in insertion order.
        std::vector<BridgeOpNode> trailing;
    };
    std::map<std::string, DeviceInsertions> insertions;

    // Helper: index of a kernel id within a device's kernel vector, or
    // SIZE_MAX if not found.
    auto kernelIndex = [&](const std::string& did, const std::string& kid) -> size_t {
        const auto& v = kernelsByDevice[did];
        for (size_t i = 0; i < v.size(); ++i) {
            if (v[i].id == kid) return i;
        }
        return std::numeric_limits<size_t>::max();
    };

    // Track already-handled (buffer, consumer-device) pairs.
    std::set<std::pair<std::string, std::string>> handledEdges;

    // Bridge-op id counter.
    uint32_t bridgeCounter = 0;

    auto materialiseLeg = [&](const RoutedLeg& leg,
                              const std::string& extraProducerDep,
                              const std::string& opIdPrefix = "_bridge_")
        -> std::pair<std::string, std::string> {
        auto pOpId = opIdPrefix + std::to_string(bridgeCounter) + "_p";
        auto cOpId = opIdPrefix + std::to_string(bridgeCounter) + "_c";
        ++bridgeCounter;

        BridgeOpNode pNode{
            pOpId, leg.srcDeviceId, leg.pair.op,
            leg.pair.producerAction,
            BridgeOpNode::Side::Producer,
            leg.producerKernelId};
        // Producer side: tryReady defaults to always-true (no wait).
        // Producer side runs after the kernel that produced its data.
        pNode.dependsOn.push_back(leg.producerKernelId);
        // Bounce-route chaining: leg2's producer (on the cpu) runs after
        // leg1's consumer (also on the cpu) finishes the first hop.
        if (!extraProducerDep.empty()) {
            pNode.dependsOn.push_back(extraProducerDep);
        }

        BridgeOpNode cNode{
            cOpId, leg.dstDeviceId, leg.pair.op,
            leg.pair.consumerAction,
            BridgeOpNode::Side::Consumer,
            leg.consumerKernelId};
        // Consumer side learns it is fire-able via the bridge's
        // tryReady probe; the executor calls it before action().
        cNode.tryReady = leg.pair.consumerTryReady;
        // Consumer side has no kernel-side predecessors here.
        // dependsOn intentionally empty.

        // Producer side: AFTER producerKernelId on src device, or trailed
        // if the producer kernel doesn't live on src (bounce intermediary).
        size_t pIdx = kernelIndex(leg.srcDeviceId, leg.producerKernelId);
        if (pIdx == std::numeric_limits<size_t>::max()) {
            insertions[leg.srcDeviceId].trailing.push_back(std::move(pNode));
        } else {
            insertions[leg.srcDeviceId].afterKernel[pIdx].push_back(std::move(pNode));
        }

        // Consumer side: BEFORE consumerKernelId on dst device, or trailed
        // similarly if the consumer kernel isn't on dst.
        size_t cIdx = kernelIndex(leg.dstDeviceId, leg.consumerKernelId);
        if (cIdx == std::numeric_limits<size_t>::max()) {
            insertions[leg.dstDeviceId].trailing.push_back(std::move(cNode));
        } else {
            insertions[leg.dstDeviceId].beforeKernel[cIdx].push_back(std::move(cNode));
        }

        return {pOpId, cOpId};
    };

    for (const auto& id : sortedIds) {
        const KernelNode* node = nodeById[id];
        const std::string& consumerDevId = nodeDevice[id];

        auto checkBuffer = [&](const std::string& bufName, const GraphBuffer& bufObj) {
            auto prodIt = producerMap.find(bufName);
            if (prodIt == producerMap.end()) return;  // graph-level input
            const std::string& producerNodeId = prodIt->second;
            const std::string& producerDevId  = nodeDevice[producerNodeId];
            if (producerDevId == consumerDevId) return;  // same device

            auto edgeKey = std::make_pair(bufName, consumerDevId);
            if (handledEdges.count(edgeKey)) return;
            handledEdges.insert(edgeKey);

            if (!cpuDevice) {
                throw std::runtime_error(
                    "GraphCompiler: cross-device transfer of buffer '" + bufName +
                    "' requires a CPU device but none is registered");
            }

            auto legs = BridgeRouter::routeTransfer(
                *devices.at(producerDevId),
                *devices.at(consumerDevId),
                bufObj, 0, bridgeFor, *cpuDevice,
                producerNodeId, id);
            std::string prevConsumerId;
            for (const auto& leg : legs) {
                auto idsPair = materialiseLeg(leg, prevConsumerId);
                prevConsumerId = idsPair.second;
            }
        };

        for (const auto& [port, buf] : node->ioMap.inputBuffers()) {
            checkBuffer(buf.name(), buf);
        }
        for (const auto& rw : node->ioMap.rwBuffers()) {
            checkBuffer(rw.in.name(), rw.in);
        }
    }

    // 5a. Materialise barrier op pairs for every cross-device afterNodes
    //     edge. The consumer-side barrier op lands in
    //     `insertions[k.dev].beforeKernel[i]` and is picked up by the
    //     dependsOn loop below as a regular consumer-side bridge dep.
    //
    //     If no direct (srcType,dstType) bridge factory exists, bounce
    //     the barrier through the cpu (one barrier per hop), chaining
    //     leg2.producer after leg1.consumer.
    for (auto& [did, kernels] : kernelsByDevice) {
        for (size_t i = 0; i < kernels.size(); ++i) {
            KernelNode& k = kernels[i];
            for (const auto& a : k.afterNodes) {
                auto ndIt = nodeDevice.find(a);
                if (ndIt == nodeDevice.end()) continue;
                if (ndIt->second == did) continue;  // same-device handled below

                IDevice& srcDev = *devices.at(ndIt->second);
                IDevice& dstDev = *devices.at(did);

                // Try direct.
                IBridge* directBr = nullptr;
                try {
                    directBr = &bridgeFor(srcDev.id(), dstDev.id());
                } catch (const std::runtime_error&) {
                    directBr = nullptr;
                }

                if (directBr) {
                    auto pair = directBr->makeBarrier(srcDev, dstDev, a, k.id);
                    RoutedLeg leg{
                        srcDev.id(), dstDev.id(),
                        a, k.id,
                        std::move(pair)};
                    materialiseLeg(leg, "", "_barrier_");
                    continue;
                }

                // Bounce through cpu.
                if (!cpuDevice) {
                    throw std::runtime_error(
                        "GraphCompiler: cross-device afterNodes from '" + a +
                        "' to '" + k.id + "' requires bouncing a barrier "
                        "through cpu but no CPU device is registered");
                }
                IBridge& srcCpu = bridgeFor(srcDev.id(), cpuDevice->id());
                IBridge& cpuDst = bridgeFor(cpuDevice->id(), dstDev.id());
                auto pair1 = srcCpu.makeBarrier(srcDev, *cpuDevice, a, k.id);
                RoutedLeg leg1{
                    srcDev.id(), cpuDevice->id(),
                    a, k.id,
                    std::move(pair1)};
                auto ids1 = materialiseLeg(leg1, "", "_barrier_");

                auto pair2 = cpuDst.makeBarrier(*cpuDevice, dstDev, a, k.id);
                RoutedLeg leg2{
                    cpuDevice->id(), dstDev.id(),
                    a, k.id,
                    std::move(pair2)};
                materialiseLeg(leg2, ids1.second, "_barrier_");
            }
        }
    }

    // 5b. Populate `dependsOn` on every KernelNode in `kernelsByDevice`.
    //
    // For each kernel k on device D:
    //   - Same-device data deps: producer kernel id (via producerMap).
    //   - Cross-device data deps: covered by adding all entries in
    //     insertions[D].beforeKernel[kIdx] (the consumer-side bridge or
    //     barrier ops synthesised for k's incoming buffers / cross-device
    //     afterNodes).
    //   - afterNodes: same-device → copy; cross-device → already reflected
    //     via the consumer-side barrier ops added in step 5a.
    for (auto& [did, kernels] : kernelsByDevice) {
        const auto& devIns = insertions[did];
        for (size_t i = 0; i < kernels.size(); ++i) {
            KernelNode& k = kernels[i];
            std::set<std::string> seen;

            auto addDep = [&](const std::string& depId) {
                if (depId.empty() || depId == k.id) return;
                if (seen.insert(depId).second) k.dependsOn.push_back(depId);
            };

            auto pushBufferDep = [&](const std::string& bufName) {
                auto pit = producerMap.find(bufName);
                if (pit == producerMap.end()) return;
                if (nodeDevice[pit->second] == did) addDep(pit->second);
            };
            for (const auto& [port, buf] : k.ioMap.inputBuffers()) {
                (void)port;
                pushBufferDep(buf.name());
            }
            for (const auto& rw : k.ioMap.rwBuffers()) {
                pushBufferDep(rw.in.name());
            }

            // Cross-device data + barrier deps via consumer-side ops.
            auto bIt = devIns.beforeKernel.find(i);
            if (bIt != devIns.beforeKernel.end()) {
                for (const auto& bop : bIt->second) addDep(bop.id);
            }

            // Same-device afterNodes only; cross-device afterNodes are
            // already represented as consumer-side barrier ops above.
            for (const auto& a : k.afterNodes) {
                auto ndIt = nodeDevice.find(a);
                if (ndIt == nodeDevice.end()) continue;
                if (ndIt->second == did) {
                    addDep(a);
                }
            }
        }
    }

    // 6. Build final DGraphs by interleaving bridge ops with kernels.
    //    Devices with no user kernels but with bounce-intermediary trailing
    //    ops still get a DGraph so the executor runs those ops.
    std::vector<DGraph> result;
    for (auto& [did, kernels] : kernelsByDevice) {
        const auto& ins = insertions[did];
        if (kernels.empty() && ins.trailing.empty()) continue;

        DGraph dg;
        dg.deviceId = did;
        dg.device   = devices.at(did);

        for (size_t i = 0; i < kernels.size(); ++i) {
            auto bIt = ins.beforeKernel.find(i);
            if (bIt != ins.beforeKernel.end()) {
                for (const auto& op : bIt->second) dg.nodes.emplace_back(op);
            }
            dg.nodes.emplace_back(std::move(kernels[i]));
            auto aIt = ins.afterKernel.find(i);
            if (aIt != ins.afterKernel.end()) {
                for (const auto& op : aIt->second) dg.nodes.emplace_back(op);
            }
        }
        for (const auto& op : ins.trailing) dg.nodes.emplace_back(op);

        dg.device->compile(dg);
        result.push_back(std::move(dg));
    }

    return result;
}

}  // namespace vrt::graph
