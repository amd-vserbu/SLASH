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

#include <vrt/graph/render/dot.hpp>

#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/node.hpp>

namespace vrt::graph::render {

namespace {

/// Escape a string for inclusion inside a double-quoted DOT identifier or label.
std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 4);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': break;
            default:   out += c;      break;
        }
    }
    return out;
}

const char* deviceTypeShortName(DeviceType dt) {
    switch (dt) {
        case DeviceType::CPU:      return "CPU";
        case DeviceType::GPU:      return "GPU";
        case DeviceType::FPGA:     return "FPGA";
        case DeviceType::MOCK_CPU: return "MOCK_CPU";
    }
    return "?";
}

/// Build a map: producer-buffer-name → producing-kernel-node-id.
std::unordered_map<std::string, std::string> buildProducerMap(
    const std::vector<KernelNode>& nodes) {
    std::unordered_map<std::string, std::string> producers;
    for (const auto& n : nodes) {
        for (const auto& [port, buf] : n.ioMap.outputBuffers()) {
            (void)port;
            producers[buf.name()] = n.id;
        }
        for (const auto& rw : n.ioMap.rwBuffers()) {
            producers[rw.out.name()] = n.id;
        }
    }
    return producers;
}

/// Collect the unique consumer-buffer-names a kernel node depends on.
std::vector<std::string> consumedBuffers(const KernelNode& n) {
    std::vector<std::string> names;
    names.reserve(n.ioMap.inputBuffers().size() + n.ioMap.rwBuffers().size());
    for (const auto& [port, buf] : n.ioMap.inputBuffers()) {
        (void)port;
        names.push_back(buf.name());
    }
    for (const auto& rw : n.ioMap.rwBuffers()) {
        names.push_back(rw.in.name());
    }
    return names;
}

/// Emit a kernel node as a rounded box.
void emitKernelNode(std::ostringstream& os, const KernelNode& n,
                    const std::string& indent) {
    os << indent << "\"" << escape(n.id) << "\""
       << " [label=\"" << escape(n.id) << "\\n[" << escape(n.kernel.name) << "]\"];\n";
}

/// Emit a bridge-op node as a dashed blue ellipse.
void emitBridgeOpNode(std::ostringstream& os, const BridgeOpNode& b,
                      const std::string& indent) {
    const char* sideLabel = (b.side == BridgeOpNode::Side::Producer) ? "Producer" : "Consumer";
    std::string opLabel = b.op ? b.op->label() : std::string{"bridge_op"};
    os << indent << "\"" << escape(b.id) << "\""
       << " [shape=ellipse, style=dashed, color=blue, label=\""
       << escape(b.id) << "\\n[" << escape(opLabel) << "]\\n("
       << sideLabel << ")\"];\n";
}

void emitDataEdge(std::ostringstream& os,
                  const std::string&  from,
                  const std::string&  to,
                  const std::string&  bufName,
                  const std::string&  indent) {
    os << indent << "\"" << escape(from) << "\" -> \"" << escape(to) << "\""
       << " [label=\"" << escape(bufName) << "\"];\n";
}

void emitAfterEdge(std::ostringstream& os,
                   const std::string&  from,
                   const std::string&  to,
                   const std::string&  indent) {
    os << indent << "\"" << escape(from) << "\" -> \"" << escape(to) << "\""
       << " [style=dashed, label=\"after\"];\n";
}

void emitSeqEdge(std::ostringstream& os,
                 const std::string&  from,
                 const std::string&  to,
                 const std::string&  indent) {
    os << indent << "\"" << escape(from) << "\" -> \"" << escape(to) << "\""
       << " [style=dotted, color=gray];\n";
}

}  // namespace

std::string renderToDot(const Graph& graph) {
    std::ostringstream os;
    os << "digraph G {\n";
    os << "  rankdir=LR;\n";
    os << "  node [shape=box, style=rounded];\n";

    const auto& nodes   = graph.nodes();
    const auto& devices = graph.devices();

    // Group by deviceHint.
    std::map<std::string, std::vector<const KernelNode*>> byDevice;
    for (const auto& n : nodes) {
        const std::string key = n.deviceHint.empty() ? std::string{"_unassigned"} : n.deviceHint;
        byDevice[key].push_back(&n);
    }

    int clusterIdx = 0;
    for (const auto& [devId, nodePtrs] : byDevice) {
        os << "  subgraph cluster_" << clusterIdx++ << " {\n";
        std::string label = devId;
        auto it = devices.find(devId);
        if (it != devices.end()) {
            label += " [";
            label += deviceTypeShortName(it->second->type());
            label += "]";
        } else if (devId == "_unassigned") {
            label = "(unassigned)";
        }
        os << "    label=\"" << escape(label) << "\";\n";
        os << "    style=rounded;\n";
        os << "    color=gray;\n";
        for (const KernelNode* n : nodePtrs) {
            emitKernelNode(os, *n, "    ");
        }
        os << "  }\n";
    }

    auto producers = buildProducerMap(nodes);
    std::unordered_set<std::string> idSet;
    idSet.reserve(nodes.size());
    for (const auto& n : nodes) idSet.insert(n.id);

    for (const auto& n : nodes) {
        for (const auto& bufName : consumedBuffers(n)) {
            auto pit = producers.find(bufName);
            if (pit == producers.end()) continue;
            emitDataEdge(os, pit->second, n.id, bufName, "  ");
        }
        for (const auto& after : n.afterNodes) {
            if (idSet.count(after)) {
                emitAfterEdge(os, after, n.id, "  ");
            }
        }
    }

    os << "}\n";
    return os.str();
}

std::string renderToDot(const DGraph& dgraph) {
    std::ostringstream os;
    os << "digraph \"" << escape(dgraph.deviceId) << "\" {\n";
    os << "  rankdir=LR;\n";
    os << "  node [shape=box, style=rounded];\n";

    std::string typeLabel;
    if (dgraph.device) {
        typeLabel = std::string{" ["} + deviceTypeShortName(dgraph.device->type()) + "]";
    }
    os << "  label=\"" << escape(dgraph.deviceId) << escape(typeLabel) << "\";\n";

    // Collect kernels and bridge ops, and emit their visual nodes.
    std::vector<KernelNode>          kernels;
    std::vector<BridgeOpNode>        bridgeOps;
    std::unordered_set<std::string>  idSet;
    kernels.reserve(dgraph.nodes.size());
    bridgeOps.reserve(dgraph.nodes.size());
    idSet.reserve(dgraph.nodes.size());

    for (const Node& node : dgraph.nodes) {
        std::visit(
            [&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                idSet.insert(n.id);
                if constexpr (std::is_same_v<T, KernelNode>) {
                    emitKernelNode(os, n, "  ");
                    kernels.push_back(n);
                } else if constexpr (std::is_same_v<T, BridgeOpNode>) {
                    emitBridgeOpNode(os, n, "  ");
                    bridgeOps.push_back(n);
                }
            },
            node);
    }

    // Dependency edges, derived directly from each node's `dependsOn`
    // (populated by the compiler). Each entry produces one edge.
    //
    // Edge style:
    //   - Both endpoints are KernelNodes → solid; labelled with the
    //     shared buffer name when identifiable from IOMap, else
    //     unlabelled.
    //   - Otherwise (any bridge-op endpoint) → dotted gray.

    auto producers = buildProducerMap(kernels);

    std::unordered_set<std::string> kernelIdSet;
    kernelIdSet.reserve(kernels.size());
    for (const auto& k : kernels) kernelIdSet.insert(k.id);

    auto sharedBufferName = [&](const KernelNode& consumer,
                                const std::string& producerId) -> std::string {
        for (const auto& bufName : consumedBuffers(consumer)) {
            auto pit = producers.find(bufName);
            if (pit != producers.end() && pit->second == producerId) return bufName;
        }
        return {};
    };

    auto emitForNode = [&](const std::string&  toId,
                           const std::vector<std::string>& deps,
                           const KernelNode*    consumerKernel /*nullable*/) {
        for (const auto& depId : deps) {
            if (!idSet.count(depId)) continue;
            const bool bothKernels =
                kernelIdSet.count(depId) && kernelIdSet.count(toId);
            if (bothKernels) {
                std::string buf;
                if (consumerKernel) buf = sharedBufferName(*consumerKernel, depId);
                if (!buf.empty()) {
                    emitDataEdge(os, depId, toId, buf, "  ");
                } else {
                    os << "  \"" << escape(depId) << "\" -> \"" << escape(toId) << "\";\n";
                }
            } else {
                emitSeqEdge(os, depId, toId, "  ");
            }
        }
    };

    for (const auto& k : kernels) {
        emitForNode(k.id, k.dependsOn, &k);
    }
    for (const auto& b : bridgeOps) {
        emitForNode(b.id, b.dependsOn, nullptr);
    }

    os << "}\n";
    return os.str();
}

namespace {

void writeDotString(const std::string& dot, const std::string& path) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!ofs) {
        throw std::runtime_error("render::writeToDotFile: cannot open '" + path +
                                 "' for writing");
    }
    ofs.write(dot.data(), static_cast<std::streamsize>(dot.size()));
    if (!ofs) {
        throw std::runtime_error("render::writeToDotFile: write failed for '" + path + "'");
    }
}

}  // namespace

void writeToDotFile(const Graph& graph, const std::string& path) {
    writeDotString(renderToDot(graph), path);
}

void writeToDotFile(const DGraph& dgraph, const std::string& path) {
    writeDotString(renderToDot(dgraph), path);
}

}  // namespace vrt::graph::render
