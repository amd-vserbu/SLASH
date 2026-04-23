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

namespace {

bool hasDeclaredPorts(const IOTypeMap& ioType) {
    return !ioType.inputScalars.empty() ||
           !ioType.outputScalars.empty() ||
           !ioType.inputBuffers.empty() ||
           !ioType.outputBuffers.empty() ||
           !ioType.rwBuffers.empty();
}

const char* scalarTypeName(ScalarType type) {
    switch (type) {
        case ScalarType::U8:  return "U8";
        case ScalarType::U16: return "U16";
        case ScalarType::U32: return "U32";
        case ScalarType::U64: return "U64";
        case ScalarType::I8:  return "I8";
        case ScalarType::I16: return "I16";
        case ScalarType::I32: return "I32";
        case ScalarType::I64: return "I64";
        case ScalarType::F32: return "F32";
        case ScalarType::F64: return "F64";
    }
    return "unknown";
}

const char* bufferTypeName(BufferType type) {
    switch (type) {
        case BufferType::U8:  return "U8";
        case BufferType::U16: return "U16";
        case BufferType::U32: return "U32";
        case BufferType::U64: return "U64";
        case BufferType::I8:  return "I8";
        case BufferType::I16: return "I16";
        case BufferType::I32: return "I32";
        case BufferType::I64: return "I64";
        case BufferType::F32: return "F32";
        case BufferType::F64: return "F64";
    }
    return "unknown";
}

template <typename PortVec>
const typename PortVec::value_type* findPortByName(const PortVec& ports,
                                                   const std::string& name) {
    auto it = std::find_if(ports.begin(), ports.end(), [&](const auto& port) {
        return port.name == name;
    });
    return (it == ports.end()) ? nullptr : &*it;
}

const RWBufferPort* findRWPortByNames(const std::vector<RWBufferPort>& ports,
                                      const std::string& inName,
                                      const std::string& outName) {
    auto it = std::find_if(ports.begin(), ports.end(), [&](const RWBufferPort& port) {
        return port.in.name == inName && port.out.name == outName;
    });
    return (it == ports.end()) ? nullptr : &*it;
}

void validateDeclaredPorts(const KernelNode& node,
                           const std::map<std::string, const KernelNode*>& nodeById) {
    for (const auto& after : node.afterNodes) {
        if (after == node.id) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' cannot depend on itself via afterNodes");
        }
        if (!nodeById.count(after)) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' references unknown afterNodes id '" +
                after + "'");
        }
    }

    const IOTypeMap& ioType = node.kernel.ioType;
    if (!hasDeclaredPorts(ioType)) return;

    for (const auto& expected : ioType.inputScalars) {
        auto it = node.ioMap.scalars().find(expected.name);
        if (it == node.ioMap.scalars().end()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' missing mandatory input scalar port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' input scalar '" + expected.name +
                "' type mismatch: declared " + scalarTypeName(expected.type) +
                ", bound " + scalarTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.outputScalars) {
        auto it = node.ioMap.scalars().find(expected.name);
        if (it == node.ioMap.scalars().end()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' missing mandatory output scalar port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' output scalar '" + expected.name +
                "' type mismatch: declared " + scalarTypeName(expected.type) +
                ", bound " + scalarTypeName(it->second.type()));
        }
        if (it->second.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' output scalar '" + expected.name +
                "' must be bound to GraphScalar::globalVar()");
        }
    }

    for (const auto& expected : ioType.inputBuffers) {
        auto it = node.ioMap.inputBuffers().find(expected.name);
        if (it == node.ioMap.inputBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' missing mandatory input buffer port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' input buffer '" + expected.name +
                "' type mismatch: declared " + bufferTypeName(expected.type) +
                ", bound " + bufferTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.outputBuffers) {
        auto it = node.ioMap.outputBuffers().find(expected.name);
        if (it == node.ioMap.outputBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' missing mandatory output buffer port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' output buffer '" + expected.name +
                "' type mismatch: declared " + bufferTypeName(expected.type) +
                ", bound " + bufferTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.rwBuffers) {
        auto it = std::find_if(node.ioMap.rwBuffers().begin(), node.ioMap.rwBuffers().end(),
                               [&](const IOMap::RWBinding& binding) {
                                   return binding.inPort == expected.in.name &&
                                          binding.outPort == expected.out.name;
                               });
        if (it == node.ioMap.rwBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' missing mandatory RW buffer ports '" +
                expected.in.name + "'/'" + expected.out.name + "'");
        }
        if (it->in.type() != expected.in.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' RW input buffer '" + expected.in.name +
                "' type mismatch: declared " + bufferTypeName(expected.in.type) +
                ", bound " + bufferTypeName(it->in.type()));
        }
        if (it->out.type() != expected.out.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' RW output buffer '" + expected.out.name +
                "' type mismatch: declared " + bufferTypeName(expected.out.type) +
                ", bound " + bufferTypeName(it->out.type()));
        }
    }

    for (const auto& [name, scalar] : node.ioMap.scalars()) {
        const auto* input = findPortByName(ioType.inputScalars, name);
        const auto* output = findPortByName(ioType.outputScalars, name);
        if (!input && !output) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' binds unknown scalar port '" + name +
                "'");
        }
        if (input && scalar.type() != input->type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' scalar '" + name +
                "' type mismatch: declared " + scalarTypeName(input->type) +
                ", bound " + scalarTypeName(scalar.type()));
        }
        if (output && scalar.type() != output->type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' scalar '" + name +
                "' type mismatch: declared " + scalarTypeName(output->type) +
                ", bound " + scalarTypeName(scalar.type()));
        }
        if (output && scalar.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' output scalar '" + name +
                "' must be bound to GraphScalar::globalVar()");
        }
    }

    for (const auto& [name, buffer] : node.ioMap.inputBuffers()) {
        const auto* port = findPortByName(ioType.inputBuffers, name);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' binds unknown input buffer port '" +
                name + "'");
        }
        if (buffer.type() != port->type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' input buffer '" + name +
                "' type mismatch: declared " + bufferTypeName(port->type) +
                ", bound " + bufferTypeName(buffer.type()));
        }
    }

    for (const auto& [name, buffer] : node.ioMap.outputBuffers()) {
        const auto* port = findPortByName(ioType.outputBuffers, name);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' binds unknown output buffer port '" +
                name + "'");
        }
        if (buffer.type() != port->type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' output buffer '" + name +
                "' type mismatch: declared " + bufferTypeName(port->type) +
                ", bound " + bufferTypeName(buffer.type()));
        }
    }

    for (const auto& binding : node.ioMap.rwBuffers()) {
        const auto* port = findRWPortByNames(ioType.rwBuffers, binding.inPort, binding.outPort);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' binds unknown RW buffer ports '" +
                binding.inPort + "'/'" + binding.outPort + "'");
        }
        if (binding.in.type() != port->in.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' RW input buffer '" + binding.inPort +
                "' type mismatch: declared " + bufferTypeName(port->in.type) +
                ", bound " + bufferTypeName(binding.in.type()));
        }
        if (binding.out.type() != port->out.type) {
            throw std::runtime_error(
                "GraphCompiler: node '" + node.id + "' RW output buffer '" + binding.outPort +
                "' type mismatch: declared " + bufferTypeName(port->out.type) +
                ", bound " + bufferTypeName(binding.out.type()));
        }
    }
}

}  // namespace

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

std::map<std::string, std::string> GraphCompiler::buildScalarProducerMap(
    const std::vector<KernelNode>& nodes) const {
    std::map<std::string, std::string> producers;
    for (const auto& node : nodes) {
        for (const auto& port : node.kernel.ioType.outputScalars) {
            auto it = node.ioMap.scalars().find(port.name);
            if (it == node.ioMap.scalars().end() || it->second.isConstant()) continue;

            const std::string& varName = it->second.varName();
            auto [existing, inserted] = producers.emplace(varName, node.id);
            if (!inserted && existing->second != node.id) {
                throw std::runtime_error(
                    "GraphCompiler: multiple nodes write global scalar '" + varName + "'");
            }
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
    auto scalarProducers = buildScalarProducerMap(nodes);

    std::map<std::string, std::vector<std::string>> adj;

    for (const auto& node : nodes) {
        adj.emplace(node.id, std::vector<std::string>{});
    }

    for (const auto& node : nodes) {
        for (const auto& port : node.kernel.ioType.inputScalars) {
            auto scalarIt = node.ioMap.scalars().find(port.name);
            if (scalarIt == node.ioMap.scalars().end() || scalarIt->second.isConstant()) {
                continue;
            }

            auto producerIt = scalarProducers.find(scalarIt->second.varName());
            if (producerIt != scalarProducers.end() && producerIt->second != node.id) {
                adj[producerIt->second].push_back(node.id);
            }
        }
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
    const BridgeFor&                                       bridgeFor,
    const std::shared_ptr<std::map<std::string, uint64_t>>& scalarValues) {

    // KernelNode lookup by id.
    std::map<std::string, const KernelNode*> nodeById;
    for (const auto& node : nodes) {
        nodeById[node.id] = &node;
    }

    for (const auto& node : nodes) {
        validateDeclaredPorts(node, nodeById);
        if (node.kernel.type != DeviceType::CPU) {
            for (const auto& [portName, scalar] : node.ioMap.scalars()) {
                (void)portName;
                if (!scalar.isConstant()) {
                    throw std::runtime_error(
                        "GraphCompiler: global scalar bindings are currently supported only on CPU kernels");
                }
            }
            if (!node.kernel.ioType.outputScalars.empty()) {
                throw std::runtime_error(
                    "GraphCompiler: output scalar ports are currently supported only on CPU kernels");
            }
        }
    }

    // 1. Topological sort.
    auto adj       = buildAdjacency(nodes);
    auto sortedIds = topoSort(nodes, adj);

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
    auto scalarProducerMap = buildScalarProducerMap(nodes);

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

    // Track the terminal consumer-side bridge op for every remote
    // (buffer, consumer-device) transfer. The transfer itself is still
    // materialised at most once, but every consumer kernel on that device
    // must depend on the same consumer-side bridge id.
    std::map<std::pair<std::string, std::string>, std::string> remoteConsumerBridgeIds;

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
            if (remoteConsumerBridgeIds.count(edgeKey)) return;

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
            if (prevConsumerId.empty()) {
                throw std::runtime_error(
                    "GraphCompiler: remote transfer of buffer '" + bufName +
                    "' to device '" + consumerDevId +
                    "' did not produce a consumer-side bridge op");
            }
            remoteConsumerBridgeIds[edgeKey] = prevConsumerId;
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
                if (nodeDevice[pit->second] == did) {
                    addDep(pit->second);
                    return;
                }

                auto bridgeIt = remoteConsumerBridgeIds.find({bufName, did});
                if (bridgeIt == remoteConsumerBridgeIds.end()) {
                    throw std::runtime_error(
                        "GraphCompiler: missing consumer-side bridge for remote buffer '" +
                        bufName + "' on device '" + did + "'");
                }
                addDep(bridgeIt->second);
            };
            for (const auto& [port, buf] : k.ioMap.inputBuffers()) {
                (void)port;
                pushBufferDep(buf.name());
            }
            for (const auto& rw : k.ioMap.rwBuffers()) {
                pushBufferDep(rw.in.name());
            }
            for (const auto& port : k.kernel.ioType.inputScalars) {
                auto scalarIt = k.ioMap.scalars().find(port.name);
                if (scalarIt == k.ioMap.scalars().end() || scalarIt->second.isConstant()) {
                    continue;
                }

                auto producerIt = scalarProducerMap.find(scalarIt->second.varName());
                if (producerIt == scalarProducerMap.end()) continue;
                const std::string& producerNodeId = producerIt->second;
                const std::string& producerDeviceId = nodeDevice.at(producerNodeId);
                if (producerDeviceId != did) {
                    throw std::runtime_error(
                        "GraphCompiler: cross-device global scalar dependencies are not supported yet");
                }
                addDep(producerNodeId);
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
                if (ndIt == nodeDevice.end()) {
                    throw std::runtime_error(
                        "GraphCompiler: node '" + k.id +
                        "' references unknown afterNodes id '" + a + "'");
                }
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
        dg.scalarValues = scalarValues;

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
