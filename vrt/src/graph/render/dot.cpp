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
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/compiled_node.hpp>

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

/// Build a map: scoped-buffer-key → producing-kernel-node-id.
///
/// Keying on the scoped key (not just the buffer name) keeps the renderer
/// correct when a single DGraph contains buffers with the same name in
/// different scopes - which can happen after Phase 7C output-placement
/// materialisation, where a child DGraph hosts both its body's local
/// buffer and a parent-scope token bridged in for publication.
template <typename KernelT>
std::unordered_map<std::string, std::string> buildProducerMap(
    const std::vector<KernelT>& nodes) {
    std::unordered_map<std::string, std::string> producers;
    for (const auto& n : nodes) {
        for (const auto& [port, buf] : n.ioMap.outputBuffers()) {
            (void)port;
            producers[scopedBufferKey(buf.scopeId(), buf.name())] = n.id;
        }
        for (const auto& rw : n.ioMap.rwBuffers()) {
            producers[scopedBufferKey(rw.out.scopeId(), rw.out.name())] = n.id;
        }
    }
    return producers;
}

/// A buffer this kernel reads, with both the scoped lookup key (used to
/// match against `buildProducerMap`) and the bare buffer name (used for
/// human-readable edge labels).
struct ConsumedBufferRef {
    std::string key;
    std::string name;
};

/// Collect the unique consumer-buffer references a kernel node depends on.
template <typename KernelT>
std::vector<ConsumedBufferRef> consumedBuffers(const KernelT& n) {
    std::vector<ConsumedBufferRef> refs;
    refs.reserve(n.ioMap.inputBuffers().size() + n.ioMap.rwBuffers().size());
    for (const auto& [port, buf] : n.ioMap.inputBuffers()) {
        (void)port;
        refs.push_back({scopedBufferKey(buf.scopeId(), buf.name()), buf.name()});
    }
    for (const auto& rw : n.ioMap.rwBuffers()) {
        refs.push_back({scopedBufferKey(rw.in.scopeId(), rw.in.name()), rw.in.name()});
    }
    return refs;
}

/// Emit a kernel node as a rounded box.
template <typename KernelT>
void emitKernelNode(std::ostringstream& os, const KernelT& n,
                    const std::string& indent) {
    os << indent << "\"" << escape(n.id) << "\""
       << " [label=\"" << escape(n.id) << "\\n[" << escape(n.kernel.name) << "]\"];\n";
}

/// Emit a bridge-op node as a dashed blue ellipse.
void emitBridgeOpNode(std::ostringstream& os, const CompiledBridgeOpNode& b,
                      const std::string& indent) {
    const char* sideLabel =
        (b.side == CompiledBridgeOpNode::Side::Producer) ? "Producer" : "Consumer";
    std::string opLabel = b.op ? b.op->label() : std::string{"bridge_op"};
    os << indent << "\"" << escape(b.id) << "\""
       << " [shape=ellipse, style=dashed, color=blue, label=\""
       << escape(b.id) << "\\n[" << escape(opLabel) << "]\\n("
       << sideLabel << ")\"];\n";
}

const char* compiledLoopKindLabel(CompiledLoopKind kind) {
    switch (kind) {
        case CompiledLoopKind::FixedCount:     return "FixedCount";
        case CompiledLoopKind::WhileCondition: return "WhileCondition";
    }
    return "?";
}

const char* compareOpLabel(CompareOp op) {
    switch (op) {
        case CompareOp::AlwaysTrue:  return "always true";
        case CompareOp::AlwaysFalse: return "always false";
        case CompareOp::LT:          return "LT";
        case CompareOp::LE:          return "LE";
        case CompareOp::EQ:          return "EQ";
        case CompareOp::GT:          return "GT";
        case CompareOp::GE:          return "GE";
        case CompareOp::NE:          return "NE";
        case CompareOp::EQE:         return "EQE";
        case CompareOp::NEE:         return "NEE";
    }
    return "?";
}

std::string pluralized(size_t count, const char* singular) {
    std::ostringstream os;
    os << count << ' ' << singular;
    if (count != 1) os << 's';
    return os.str();
}

std::string bufferScalarCounts(const char* label, size_t bufferCount, size_t scalarCount) {
    std::ostringstream os;
    os << label << ": " << pluralized(bufferCount, "buffer")
       << ", " << pluralized(scalarCount, "scalar");
    return os.str();
}

std::string scopedName(uint64_t scopeId, const std::string& name) {
    if (scopeId == 0) return name;
    std::ostringstream os;
    os << "scope" << scopeId << ':' << name;
    return os.str();
}

std::string tripCountSummary(const LoopTripCount& tripCount) {
    switch (tripCount.kind()) {
        case LoopTripCount::Kind::Constant:
            return "trip: const";
        case LoopTripCount::Kind::Scalar:
            return "trip: scalar " + scopedName(tripCount.scopeId(), tripCount.name());
    }
    return "trip: ?";
}

std::string conditionSummary(const Condition& condition) {
    std::ostringstream os;
    os << "condition: " << compareOpLabel(condition.op());
    if (!condition.isAlways()) {
        size_t scalarCount = 0;
        auto countScalar = [&](const std::optional<ConditionOperand>& operand) {
            if (operand && operand->isScalar()) ++scalarCount;
        };
        countScalar(condition.lhs());
        countScalar(condition.rhs());
        countScalar(condition.epsilon());
        os << " (" << pluralized(scalarCount, "scalar") << ')';
    }
    return os.str();
}

void emitBoundaryNode(std::ostringstream& os, const CompiledBoundaryNode& b,
                      const std::string& indent) {
    const char* sideLabel = (b.side == CompiledBoundaryNode::Side::Start) ? "Start" : "End";
    os << indent << "\"" << escape(b.id) << "\""
       << " [shape=diamond, style=dashed, color=gray, label=\""
       << escape(b.id) << "\\n[Boundary]\\n(" << sideLabel << ")\\n"
       << escape(bufferScalarCounts("copies", b.bufferCopies.size(),
                                    b.scalarCopies.size()))
       << "\"];\n";
}

void emitLoopNode(std::ostringstream& os, const CompiledLoopNode& c,
                  const std::string& indent) {
    std::ostringstream label;
    label << c.id << "\n[Loop]";
    label << "\n(" << compiledLoopKindLabel(c.loopKind) << ')';
    if (c.tripCount) label << "\n" << tripCountSummary(*c.tripCount);
    if (c.condition) label << "\n" << conditionSummary(*c.condition);
    if (!c.outputBufferPublications.empty() || !c.outputScalarPublications.empty()) {
        label << "\n" << bufferScalarCounts("outputs", c.outputBufferPublications.size(),
                                             c.outputScalarPublications.size());
    }
    if (!c.outputBufferPlacements.empty() || !c.outputScalarPlacements.empty()) {
        label << "\n" << bufferScalarCounts("placements", c.outputBufferPlacements.size(),
                                             c.outputScalarPlacements.size());
    }

    os << indent << "\"" << escape(c.id) << "\""
       << " [shape=octagon, style=dashed, color=gray, label=\""
       << escape(label.str()) << "\"];\n";
}

void emitConditionalNode(std::ostringstream& os, const CompiledConditionalNode& c,
                         const std::string& indent) {
    std::ostringstream label;
    label << c.id << "\n[Conditional]";
    label << "\n" << conditionSummary(c.condition);
    if (!c.outputBufferPublications.empty() || !c.outputScalarPublications.empty()) {
        label << "\n" << bufferScalarCounts("outputs", c.outputBufferPublications.size(),
                                             c.outputScalarPublications.size());
    }
    if (!c.outputBufferPlacements.empty() || !c.outputScalarPlacements.empty()) {
        label << "\n" << bufferScalarCounts("placements", c.outputBufferPlacements.size(),
                                             c.outputScalarPlacements.size());
    }

    os << indent << "\"" << escape(c.id) << "\""
       << " [shape=octagon, style=dashed, color=gray, label=\""
       << escape(label.str()) << "\"];\n";
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

const char* boundarySideLabel(BoundarySide side) {
    switch (side) {
        case BoundarySide::Start: return "Start";
        case BoundarySide::End:   return "End";
    }
    return "?";
}

const char* loopKindLabel(LoopKind kind) {
    switch (kind) {
        case LoopKind::FixedCount:     return "FixedCount";
        case LoopKind::WhileCondition: return "WhileCondition";
    }
    return "?";
}

void emitAuthoredBoundaryNode(std::ostringstream& os, const SubgraphBoundaryOp& b,
                              const std::string& indent) {
    os << indent << "\"" << escape(b.id) << "\""
       << " [shape=diamond, style=dashed, color=gray, label=\""
       << escape(b.id) << "\\n[Boundary]\\n(" << boundarySideLabel(b.side) << ")\"];\n";
}

void emitAuthoredControlNode(std::ostringstream& os, const LoopOp& loop,
                             const std::string& indent) {
    os << indent << "\"" << escape(loop.id) << "\""
       << " [shape=octagon, style=dashed, color=gray, label=\""
       << escape(loop.id) << "\\n[Loop]\\n(" << loopKindLabel(loop.kind) << ")\"];\n";
}

void emitAuthoredControlNode(std::ostringstream& os, const ConditionalOp& conditional,
                             const std::string& indent) {
    os << indent << "\"" << escape(conditional.id) << "\""
       << " [shape=octagon, style=dashed, color=gray, label=\""
       << escape(conditional.id) << "\\n[Conditional]\"];\n";
}

struct RenderEdge {
    enum class Kind { Data, After };
    Kind kind = Kind::Data;
    std::string from;
    std::string to;
    std::string label;
};

struct AuthoredRenderContext {
    const std::map<std::string, std::shared_ptr<IDevice>>& devices;
    int clusterIdx = 0;
    std::vector<RenderEdge> edges;
};

std::string nextClusterName(AuthoredRenderContext& ctx) {
    return "cluster_" + std::to_string(ctx.clusterIdx++);
}

const IOMap& authoredIoMap(const RegionOp& op) {
    return std::visit(
        [](const auto& concrete) -> const IOMap& {
            return concrete.ioMap;
        },
        op);
}

const std::vector<std::string>& authoredAfterOps(const RegionOp& op) {
    return std::visit(
        [](const auto& concrete) -> const std::vector<std::string>& {
            return concrete.afterOps;
        },
        op);
}

std::string authoredBufferKey(const GraphBuffer& buffer) {
    return scopedBufferKey(buffer.scopeId(), buffer.name());
}

std::vector<GraphBuffer> authoredConsumedBuffers(const RegionOp& op) {
    const IOMap& ioMap = authoredIoMap(op);
    std::vector<GraphBuffer> buffers;
    buffers.reserve(ioMap.inputBuffers().size() + ioMap.rwBuffers().size());
    for (const auto& [port, buffer] : ioMap.inputBuffers()) {
        (void)port;
        buffers.push_back(buffer);
    }
    for (const auto& rw : ioMap.rwBuffers()) {
        buffers.push_back(rw.in);
    }
    return buffers;
}

void collectRegionEdges(const GraphRegion& region, AuthoredRenderContext& ctx) {
    struct Producer {
        std::string id;
        std::string label;
    };

    std::unordered_map<std::string, Producer> producers;
    std::unordered_set<std::string> idSet;
    idSet.reserve(region.ops().size());

    for (const RegionOp& op : region.ops()) {
        const std::string& id = regionOpId(op);
        idSet.insert(id);

        const IOMap& ioMap = authoredIoMap(op);
        for (const auto& [port, buffer] : ioMap.outputBuffers()) {
            (void)port;
            producers[authoredBufferKey(buffer)] = Producer{id, buffer.name()};
        }
        for (const auto& rw : ioMap.rwBuffers()) {
            producers[authoredBufferKey(rw.out)] = Producer{id, rw.out.name()};
        }
    }

    for (const RegionOp& op : region.ops()) {
        const std::string& toId = regionOpId(op);
        for (const GraphBuffer& buffer : authoredConsumedBuffers(op)) {
            auto pit = producers.find(authoredBufferKey(buffer));
            if (pit == producers.end()) continue;
            if (pit->second.id == toId) continue;
            ctx.edges.push_back(RenderEdge{RenderEdge::Kind::Data,
                                           pit->second.id, toId, buffer.name()});
        }
        for (const std::string& after : authoredAfterOps(op)) {
            if (idSet.count(after)) {
                ctx.edges.push_back(RenderEdge{RenderEdge::Kind::After, after, toId, {}});
            }
        }
    }
}

std::string deviceClusterLabel(
    const std::string& devId,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    std::string label = devId;
    auto it = devices.find(devId);
    if (it != devices.end()) {
        label += " [";
        label += deviceTypeShortName(it->second->type());
        label += "]";
    } else if (devId == "_unassigned") {
        label = "(unassigned)";
    }
    return label;
}

void emitAuthoredDeviceClusters(std::ostringstream& os,
                                const GraphRegion& region,
                                AuthoredRenderContext& ctx,
                                const std::string& indent) {
    std::map<std::string, std::vector<const KernelOp*>> byDevice;
    for (const RegionOp& op : region.ops()) {
        if (const auto* kernel = std::get_if<KernelOp>(&op)) {
            const std::string key = kernel->deviceHint.empty()
                ? std::string{"_unassigned"}
                : kernel->deviceHint;
            byDevice[key].push_back(kernel);
        }
    }

    for (const auto& [devId, nodePtrs] : byDevice) {
        os << indent << "subgraph " << nextClusterName(ctx) << " {\n";
        os << indent << "  label=\"" << escape(deviceClusterLabel(devId, ctx.devices)) << "\";\n";
        os << indent << "  style=rounded;\n";
        os << indent << "  color=gray;\n";
        for (const KernelOp* node : nodePtrs) {
            emitKernelNode(os, *node, indent + "  ");
        }
        os << indent << "}\n";
    }
}

void emitRegionCluster(std::ostringstream& os,
                       const GraphRegion& region,
                       const std::string& label,
                       AuthoredRenderContext& ctx,
                       const std::string& indent) {
    collectRegionEdges(region, ctx);

    os << indent << "subgraph " << nextClusterName(ctx) << " {\n";
    os << indent << "  label=\"" << escape(label) << "\";\n";
    os << indent << "  style=rounded;\n";
    os << indent << "  color=gray;\n";

    emitAuthoredDeviceClusters(os, region, ctx, indent + "  ");

    for (const RegionOp& op : region.ops()) {
        std::visit(
            [&](const auto& concrete) {
                using T = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<T, KernelOp>) {
                    return;
                } else if constexpr (std::is_same_v<T, SubgraphBoundaryOp>) {
                    emitAuthoredBoundaryNode(os, concrete, indent + "  ");
                } else if constexpr (std::is_same_v<T, LoopOp>) {
                    emitAuthoredControlNode(os, concrete, indent + "  ");
                    if (concrete.body) {
                        emitRegionCluster(os, *concrete.body, concrete.id + " loop body",
                                          ctx, indent + "  ");
                    }
                } else if constexpr (std::is_same_v<T, ConditionalOp>) {
                    emitAuthoredControlNode(os, concrete, indent + "  ");
                    if (concrete.thenRegion) {
                        emitRegionCluster(os, *concrete.thenRegion, concrete.id + " then",
                                          ctx, indent + "  ");
                    }
                    if (concrete.elseRegion) {
                        emitRegionCluster(os, *concrete.elseRegion, concrete.id + " else",
                                          ctx, indent + "  ");
                    }
                }
            },
            op);
    }

    os << indent << "}\n";
}

}  // namespace

std::string renderToDot(const Graph& graph) {
    std::ostringstream os;
    os << "digraph G {\n";
    os << "  rankdir=LR;\n";
    os << "  node [shape=box, style=rounded];\n";

    AuthoredRenderContext ctx{graph.devices()};
    emitRegionCluster(os, graph.rootRegion(), "root region", ctx, "  ");

    for (const RenderEdge& edge : ctx.edges) {
        if (edge.kind == RenderEdge::Kind::Data) {
            emitDataEdge(os, edge.from, edge.to, edge.label, "  ");
        } else {
            emitAfterEdge(os, edge.from, edge.to, "  ");
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
    std::vector<CompiledKernelNode>  kernels;
    std::unordered_set<std::string>  idSet;
    kernels.reserve(dgraph.nodes.size());
    idSet.reserve(dgraph.nodes.size());

    for (const CompiledNode& node : dgraph.nodes) {
        std::visit(
            [&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                idSet.insert(n.id);
                if constexpr (std::is_same_v<T, CompiledKernelNode>) {
                    emitKernelNode(os, n, "  ");
                    kernels.push_back(n);
                } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
                    emitBridgeOpNode(os, n, "  ");
                } else if constexpr (std::is_same_v<T, CompiledBoundaryNode>) {
                    emitBoundaryNode(os, n, "  ");
                } else if constexpr (std::is_same_v<T, CompiledLoopNode>) {
                    emitLoopNode(os, n, "  ");
                } else if constexpr (std::is_same_v<T, CompiledConditionalNode>) {
                    emitConditionalNode(os, n, "  ");
                }
            },
            node);
    }

    // Dependency edges, derived directly from each node's `dependsOn`
    // (populated by the compiler). Each entry produces one edge.
    //
    // Edge style:
    //   - Both endpoints are CompiledKernelNodes → solid; labelled with the
    //     shared buffer name when identifiable from IOMap, else
    //     unlabelled.
    //   - Otherwise (any bridge-op endpoint) → dotted gray.

    auto producers = buildProducerMap(kernels);

    std::unordered_set<std::string> kernelIdSet;
    kernelIdSet.reserve(kernels.size());
    for (const auto& k : kernels) kernelIdSet.insert(k.id);

    auto sharedBufferName = [&](const CompiledKernelNode& consumer,
                                const std::string& producerId) -> std::string {
        for (const auto& ref : consumedBuffers(consumer)) {
            auto pit = producers.find(ref.key);
            if (pit != producers.end() && pit->second == producerId) return ref.name;
        }
        return {};
    };

    auto emitForNode = [&](const std::string&  toId,
                           const std::vector<std::string>& deps,
                           const CompiledKernelNode* consumerKernel /*nullable*/) {
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

    for (const CompiledNode& node : dgraph.nodes) {
        const auto* kernel = std::get_if<CompiledKernelNode>(&node);
        emitForNode(compiledNodeId(node), compiledNodeDependsOn(node), kernel);
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
