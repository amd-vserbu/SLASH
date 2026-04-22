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
#include <queue>
#include <set>

namespace vrt::graph {

// ---------------------------------------------------------------------------
// buildProducerMap
// ---------------------------------------------------------------------------

std::map<std::string, std::string> GraphCompiler::buildProducerMap(
    const std::vector<Node>& nodes) const {
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
    const std::vector<Node>& nodes) const {
    auto producers = buildProducerMap(nodes);

    std::map<std::string, std::vector<std::string>> adj;

    // Ensure every node has an entry even if it has no successors.
    for (const auto& node : nodes) {
        adj.emplace(node.id, std::vector<std::string>{});
    }

    for (const auto& node : nodes) {
        // Data-dependency edges: input buffer produced by another node.
        for (const auto& [port, buf] : node.ioMap.inputBuffers()) {
            auto it = producers.find(buf.name());
            if (it != producers.end()) {
                adj[it->second].push_back(node.id);
            }
        }

        // RW buffer input side.
        for (const auto& rw : node.ioMap.rwBuffers()) {
            auto it = producers.find(rw.in.name());
            if (it != producers.end()) {
                adj[it->second].push_back(node.id);
            }
        }

        // Explicit ordering constraints.
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
    const std::vector<Node>& nodes,
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
// injectCrossDeviceSync
// ---------------------------------------------------------------------------

void GraphCompiler::injectCrossDeviceSync(
    IDevice& producer,
    IDevice& consumer,
    const GraphBuffer& buf,
    uint64_t sizeHint,
    const std::map<std::pair<DeviceType, DeviceType>,
                   std::shared_ptr<IBridge>>& bridges,
    IDevice& cpuDevice,
    const std::string& producerNodeId,
    const std::string& consumerNodeId) {
    BridgeRouter::routeTransfer(
        producer, consumer, buf, sizeHint, bridges, cpuDevice,
        [this]() { return allocSemaphore(); },
        producerNodeId, consumerNodeId);
}

// ---------------------------------------------------------------------------
// compile
// ---------------------------------------------------------------------------

std::vector<DGraph> GraphCompiler::compile(
    const std::vector<Node>& nodes,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const std::map<std::pair<DeviceType, DeviceType>,
                   std::shared_ptr<IBridge>>& bridges) {

    // 1. Topological sort.
    auto adj       = buildAdjacency(nodes);
    auto sortedIds = topoSort(nodes, adj);

    // Build node lookup by id.
    std::map<std::string, const Node*> nodeById;
    for (const auto& node : nodes) {
        nodeById[node.id] = &node;
    }

    // 2. Resolve device placement for each node.
    std::map<std::string, std::string> nodeDevice;  // node-id → device-id
    for (const auto& id : sortedIds) {
        const Node* node = nodeById[id];
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

    // 3. Group nodes by device in topological order.
    std::map<std::string, DGraph> dgraphByDevice;
    for (const auto& [did, dev] : devices) {
        auto& dg = dgraphByDevice[did];
        dg.deviceId = did;
        dg.device   = dev;
    }
    for (const auto& id : sortedIds) {
        dgraphByDevice[nodeDevice[id]].nodes.push_back(*nodeById[id]);
    }

    // 4. Inject cross-device synchronisation via bridges.
    auto producerMap = buildProducerMap(nodes);

    // Find CPU device for bounce routing (if any).
    IDevice* cpuDevice = nullptr;
    for (const auto& [did, dev] : devices) {
        if (dev->type() == DeviceType::CPU) {
            cpuDevice = dev.get();
            break;
        }
    }

    // Track already-injected edges to avoid duplicates.
    std::set<std::pair<std::string, std::string>> injectedEdges;

    for (const auto& id : sortedIds) {
        const Node* node = nodeById[id];
        const std::string& consumerDevId = nodeDevice[id];

        auto checkBuffer = [&](const std::string& bufName, const GraphBuffer& bufObj) {
            auto prodIt = producerMap.find(bufName);
            if (prodIt == producerMap.end()) return;  // graph-level input
            const std::string& producerNodeId = prodIt->second;
            const std::string& producerDevId  = nodeDevice[producerNodeId];
            if (producerDevId == consumerDevId) return;  // same device

            auto edgeKey = std::make_pair(bufName, consumerDevId);
            if (injectedEdges.count(edgeKey)) return;
            injectedEdges.insert(edgeKey);

            injectCrossDeviceSync(
                *devices.at(producerDevId),
                *devices.at(consumerDevId),
                bufObj,
                0,  // sizeHint unknown at compile time
                bridges,
                *cpuDevice,
                producerNodeId,
                id);
        };

        for (const auto& [port, buf] : node->ioMap.inputBuffers()) {
            checkBuffer(buf.name(), buf);
        }
        for (const auto& rw : node->ioMap.rwBuffers()) {
            checkBuffer(rw.in.name(), rw.in);
        }
    }

    // 5. Build result and call device compile().
    std::vector<DGraph> result;
    for (auto& [did, dg] : dgraphByDevice) {
        if (dg.nodes.empty()) continue;  // skip devices with no work
        dg.device->compile(dg);
        result.push_back(std::move(dg));
    }

    return result;
}

}  // namespace vrt::graph
