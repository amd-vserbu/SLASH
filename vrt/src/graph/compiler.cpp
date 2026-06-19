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
#include <cstddef>
#include <functional>
#include <limits>
#include <queue>
#include <set>
#include <type_traits>
#include <utility>

namespace vrt::graph {

namespace {

/// Mints the producer / consumer node id pair for a compiler-synthesised
/// bridge. Centralised here so each kind of bridge has exactly one place
/// owning its id prefix - tests and tools that match by id rely on these
/// strings being stable across compilations.
struct BridgeIds {
    enum class Kind {
        DataBridge,           // Cross-device data buffer transfer (`_bridge_`).
        Barrier,              // afterOps cross-device barrier (`_barrier_`).
        ControlOutputBridge,  // Loop / conditional output placement transfer
                              // (`_control_output_bridge_`).
    };

    static const char* prefix(Kind kind) {
        switch (kind) {
            case Kind::DataBridge:          return "_bridge_";
            case Kind::Barrier:             return "_barrier_";
            case Kind::ControlOutputBridge: return "_control_output_bridge_";
        }
        return "_bridge_";
    }

    /// Returns @c {producerId, consumerId} for the given counter. The caller
    /// owns the counter and increments it after each call.
    static std::pair<std::string, std::string> mint(Kind kind, uint32_t counter) {
        const std::string stem = std::string(prefix(kind)) + std::to_string(counter);
        return {stem + "_p", stem + "_c"};
    }
};

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

void requireAllowedScope(const std::string& opId,
                         const std::string& tokenKind,
                         const std::string& tokenName,
                         uint64_t tokenScope,
                         uint64_t regionScope,
                         const std::set<uint64_t>& allowedScopes) {
    if (allowedScopes.count(tokenScope)) return;
    throw std::runtime_error(
        "GraphCompiler: op '" + opId + "' uses " + tokenKind + " '" + tokenName +
        "' from scope " + std::to_string(tokenScope) + " while compiling region scope " +
        std::to_string(regionScope) +
        "; import it through a boundary mapping before using it inside the region");
}

void validateIoMapScopes(const std::string& opId,
                         const IOMap& ioMap,
                         uint64_t regionScope,
                         const std::set<uint64_t>& allowedScopes) {
    for (const auto& [port, scalar] : ioMap.scalars()) {
        (void)port;
        if (scalar.isConstant()) continue;
        requireAllowedScope(opId, "scalar", scalar.varName(), scalar.scopeId(),
                            regionScope, allowedScopes);
    }

    for (const auto& [port, buffer] : ioMap.inputBuffers()) {
        (void)port;
        requireAllowedScope(opId, "input buffer", buffer.name(), buffer.scopeId(),
                            regionScope, allowedScopes);
    }
    for (const auto& [port, buffer] : ioMap.outputBuffers()) {
        (void)port;
        requireAllowedScope(opId, "output buffer", buffer.name(), buffer.scopeId(),
                            regionScope, allowedScopes);
    }
    for (const auto& rw : ioMap.rwBuffers()) {
        requireAllowedScope(opId, "RW input buffer", rw.in.name(), rw.in.scopeId(),
                            regionScope, allowedScopes);
        requireAllowedScope(opId, "RW output buffer", rw.out.name(), rw.out.scopeId(),
                            regionScope, allowedScopes);
    }
}

void validateConditionOperandScope(const std::string& opId,
                                   const ConditionOperand& operand,
                                   uint64_t regionScope) {
    if (!operand.isScalar()) return;
    requireAllowedScope(opId, "condition scalar", operand.name(), operand.scopeId(),
                        regionScope, {regionScope});
}

void validateConditionScopes(const std::string& opId,
                             const Condition& condition,
                             uint64_t regionScope) {
    if (condition.lhs()) validateConditionOperandScope(opId, *condition.lhs(), regionScope);
    if (condition.rhs()) validateConditionOperandScope(opId, *condition.rhs(), regionScope);
    if (condition.epsilon()) validateConditionOperandScope(opId, *condition.epsilon(), regionScope);
}

void validateTripCountScope(const std::string& opId,
                            const LoopTripCount& tripCount,
                            uint64_t regionScope) {
    if (tripCount.kind() != LoopTripCount::Kind::Scalar) return;
    requireAllowedScope(opId, "trip-count scalar", tripCount.name(), tripCount.scopeId(),
                        regionScope, {regionScope});
}

void validateScalarBoundaryMappings(const SubgraphBoundaryOp& boundary) {
    for (const auto& mapping : boundary.scalarMappings) {
        if (mapping.source.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' scalar source must be a scoped scalar variable");
        }
        if (mapping.target.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' scalar target must be a scoped scalar variable");
        }
        if (mapping.source.type() != mapping.target.type()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' scalar mapping type mismatch: source " +
                scalarTypeName(mapping.source.type()) + ", target " +
                scalarTypeName(mapping.target.type()));
        }

        const uint64_t expectedSourceScope =
            (boundary.side == BoundarySide::Start) ? boundary.parentScopeId
                                                   : boundary.localScopeId;
        const uint64_t expectedTargetScope =
            (boundary.side == BoundarySide::Start) ? boundary.localScopeId
                                                   : boundary.parentScopeId;
        if (mapping.source.scopeId() != expectedSourceScope) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' scalar source scope " + std::to_string(mapping.source.scopeId()) +
                " does not match expected scope " + std::to_string(expectedSourceScope));
        }
        if (mapping.target.scopeId() != expectedTargetScope) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' scalar target scope " + std::to_string(mapping.target.scopeId()) +
                " does not match expected scope " + std::to_string(expectedTargetScope));
        }
    }
}

void validateBufferBoundaryMappings(const SubgraphBoundaryOp& boundary) {
    for (const auto& mapping : boundary.bufferMappings) {
        if (!mapping.source.valid()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' buffer source must be a valid GraphBuffer");
        }
        if (!mapping.target.valid()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' buffer target must be a valid GraphBuffer");
        }
        if (mapping.source.type() != mapping.target.type()) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' buffer mapping type mismatch: source " +
                bufferTypeName(mapping.source.type()) + ", target " +
                bufferTypeName(mapping.target.type()));
        }

        const uint64_t expectedSourceScope =
            (boundary.side == BoundarySide::Start) ? boundary.parentScopeId
                                                   : boundary.localScopeId;
        const uint64_t expectedTargetScope =
            (boundary.side == BoundarySide::Start) ? boundary.localScopeId
                                                   : boundary.parentScopeId;
        if (mapping.source.scopeId() != expectedSourceScope) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' buffer source scope " + std::to_string(mapping.source.scopeId()) +
                " does not match expected scope " + std::to_string(expectedSourceScope));
        }
        if (mapping.target.scopeId() != expectedTargetScope) {
            throw std::runtime_error(
                "GraphCompiler: boundary op '" + boundary.id +
                "' buffer target scope " + std::to_string(mapping.target.scopeId()) +
                " does not match expected scope " + std::to_string(expectedTargetScope));
        }
    }
}

void validateChildRegion(const std::string& opId,
                         const std::shared_ptr<GraphRegion>& child,
                         uint64_t parentScope) {
    if (!child) {
        throw std::runtime_error(
            "GraphCompiler: control op '" + opId + "' has a null child region");
    }
    if (child->parentScopeId() != parentScope) {
        throw std::runtime_error(
            "GraphCompiler: control op '" + opId + "' references child region scope " +
            std::to_string(child->scopeId()) + " with parent scope " +
            std::to_string(child->parentScopeId()) + ", expected " +
            std::to_string(parentScope));
    }
}

const char* deviceTypeNameStr(DeviceType dt) {
    switch (dt) {
        case DeviceType::CPU:      return "CPU";
        case DeviceType::GPU:      return "GPU";
        case DeviceType::FPGA:     return "FPGA";
        case DeviceType::MOCK_CPU: return "MOCK_CPU";
    }
    return "unknown";
}

/// Returns the unique CPU-typed device pointer (or nullptr if none).
///
/// The CPU device is a singleton owned by `Graph` (typically registered via
/// `Graph::withDefaults()`); `Graph::registerDevice` rejects a second
/// `DeviceType::CPU` registration. The compiler relies on this invariant in
/// two places:
///   * control-flow ops (loops, conditionals) are pinned to the CPU device,
///     because the CPU backend is the only one that owns control-flow
///     execution today;
///   * cross-device transfers without a direct `(srcType, dstType)` bridge
///     factory are routed through the CPU as a bounce buffer.
///
/// If multiple CPU-typed devices appear (e.g. via direct manipulation of the
/// device map), this throws to surface the violated invariant.
IDevice* findSingletonCpuDevice(
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    IDevice* found = nullptr;
    for (const auto& [did, dev] : devices) {
        (void)did;
        if (!dev || dev->type() != DeviceType::CPU) continue;
        if (found) {
            throw std::runtime_error(
                std::string("GraphCompiler: multiple CPU-typed devices registered ('") +
                found->id() + "' and '" + dev->id() +
                "'); at most one is allowed");
        }
        found = dev.get();
    }
    return found;
}

void validateBridgeFactories(
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const std::map<std::pair<DeviceType, DeviceType>, BridgeFactory>& bridgeFactories) {
    (void)findSingletonCpuDevice(devices);  // defends the singleton invariant
    for (const auto& [id, device] : devices) {
        DeviceType dt = device->type();
        if (dt == DeviceType::CPU) continue;
        if (!bridgeFactories.count({DeviceType::CPU, dt}) ||
            !bridgeFactories.count({dt, DeviceType::CPU})) {
            throw std::runtime_error(
                std::string("GraphCompiler: device '") + id +
                "' requires {CPU, " + deviceTypeNameStr(dt) +
                "} and {" + deviceTypeNameStr(dt) +
                ", CPU} bridge factories, but at least one is missing");
        }
    }
}

void validateRootScopeScalarReferences(const GraphRegion& rootRegion) {
    const uint64_t rootScopeId = rootRegion.scopeId();
    const auto& declaredScalars = rootRegion.declaredScalars();

    auto requireDeclared = [&](const std::string& name, uint64_t scopeId) {
        if (scopeId != rootScopeId) return;
        if (declaredScalars.count(name)) return;
        throw std::runtime_error(
            "GraphCompiler: global scalar '" + name +
            "' is not declared; call Graph::globalScalar() first");
    };

    auto checkScalar = [&](const GraphScalar& scalar) {
        if (scalar.isConstant()) return;
        requireDeclared(scalar.varName(), scalar.scopeId());
    };

    auto checkOperand = [&](const std::optional<ConditionOperand>& operand) {
        if (!operand || !operand->isScalar()) return;
        requireDeclared(operand->name(), operand->scopeId());
    };

    auto checkCondition = [&](const Condition& condition) {
        checkOperand(condition.lhs());
        checkOperand(condition.rhs());
        checkOperand(condition.epsilon());
    };

    auto checkTripCount = [&](const LoopTripCount& tripCount) {
        if (tripCount.kind() != LoopTripCount::Kind::Scalar) return;
        requireDeclared(tripCount.name(), tripCount.scopeId());
    };

    auto checkIoMap = [&](const IOMap& ioMap) {
        for (const auto& [portName, scalar] : ioMap.scalars()) {
            (void)portName;
            checkScalar(scalar);
        }
    };

    std::function<void(const GraphRegion&)> walk = [&](const GraphRegion& region) {
        for (const RegionOp& op : region.ops()) {
            std::visit(
                [&](const auto& concrete) {
                    using T = std::decay_t<decltype(concrete)>;
                    if constexpr (std::is_same_v<T, KernelOp>) {
                        checkIoMap(concrete.ioMap);
                    } else if constexpr (std::is_same_v<T, SubgraphBoundaryOp>) {
                        checkIoMap(concrete.ioMap);
                        for (const auto& mapping : concrete.scalarMappings) {
                            checkScalar(mapping.source);
                            checkScalar(mapping.target);
                        }
                    } else if constexpr (std::is_same_v<T, LoopOp>) {
                        checkIoMap(concrete.ioMap);
                        if (concrete.tripCount) checkTripCount(*concrete.tripCount);
                        if (concrete.condition) checkCondition(*concrete.condition);
                        if (concrete.body) walk(*concrete.body);
                    } else if constexpr (std::is_same_v<T, ConditionalOp>) {
                        checkIoMap(concrete.ioMap);
                        checkCondition(concrete.condition);
                        if (concrete.thenRegion) walk(*concrete.thenRegion);
                        if (concrete.elseRegion) walk(*concrete.elseRegion);
                    }
                },
                op);
        }
    };

    walk(rootRegion);
}

void validateRootScopeBufferReferences(const GraphRegion& rootRegion) {
    const uint64_t rootScopeId = rootRegion.scopeId();
    const auto& declaredInputs = rootRegion.declaredInputBufferNames();

    // Collect every buffer name produced at root scope by any op of the root
    // region. Kernel ops contribute outputBuffers and rwBuffers.out. Loop and
    // conditional control ops publish their declared outputBuffers to the
    // parent scope, so they count as producers too. Boundary ops at root
    // produce only into a child scope, so we skip them here.
    std::set<std::string> producedAtRoot;
    auto recordProduced = [&](const IOMap& ioMap) {
        for (const auto& [port, buffer] : ioMap.outputBuffers()) {
            (void)port;
            if (buffer.scopeId() == rootScopeId) {
                producedAtRoot.insert(buffer.name());
            }
        }
        for (const auto& rw : ioMap.rwBuffers()) {
            if (rw.out.scopeId() == rootScopeId) {
                producedAtRoot.insert(rw.out.name());
            }
        }
    };
    // Loop / conditional control ops publish to the parent scope through their
    // body/branch end boundaries (the struct-literal authoring API binds
    // `.outputs` this way rather than via the control op's own IOMap), so scan
    // those end-boundary targets too.
    auto recordControlPublications = [&](const GraphRegion& child) {
        for (const RegionOp& childOp : child.ops()) {
            const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
            if (!boundary || boundary->side != BoundarySide::End) continue;
            for (const auto& mapping : boundary->bufferMappings) {
                if (mapping.target.scopeId() == rootScopeId) {
                    producedAtRoot.insert(mapping.target.name());
                }
            }
        }
    };
    for (const RegionOp& op : rootRegion.ops()) {
        std::visit(
            [&](const auto& concrete) {
                using T = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<T, KernelOp>) {
                    recordProduced(concrete.ioMap);
                } else if constexpr (std::is_same_v<T, LoopOp>) {
                    recordProduced(concrete.ioMap);
                    if (concrete.body) recordControlPublications(*concrete.body);
                } else if constexpr (std::is_same_v<T, ConditionalOp>) {
                    recordProduced(concrete.ioMap);
                    if (concrete.thenRegion) recordControlPublications(*concrete.thenRegion);
                    if (concrete.elseRegion) recordControlPublications(*concrete.elseRegion);
                } else if constexpr (std::is_same_v<T, SubgraphBoundaryOp>) {
                    for (const auto& mapping : concrete.bufferMappings) {
                        if (mapping.target.scopeId() == rootScopeId) {
                            producedAtRoot.insert(mapping.target.name());
                        }
                    }
                }
            },
            op);
    }

    auto requireDeclared = [&](const GraphBuffer& buffer) {
        if (buffer.scopeId() != rootScopeId) return;
        if (declaredInputs.count(buffer.name())) return;
        if (producedAtRoot.count(buffer.name())) return;
        throw std::runtime_error(
            "GraphCompiler: input buffer '" + buffer.name() +
            "' is not declared; call Graph::inputBuffer() first");
    };

    auto checkIoMap = [&](const IOMap& ioMap) {
        for (const auto& [port, buffer] : ioMap.inputBuffers()) {
            (void)port;
            requireDeclared(buffer);
        }
        for (const auto& rw : ioMap.rwBuffers()) {
            requireDeclared(rw.in);
        }
    };

    std::function<void(const GraphRegion&)> walk = [&](const GraphRegion& region) {
        for (const RegionOp& op : region.ops()) {
            std::visit(
                [&](const auto& concrete) {
                    using T = std::decay_t<decltype(concrete)>;
                    if constexpr (std::is_same_v<T, KernelOp>) {
                        checkIoMap(concrete.ioMap);
                    } else if constexpr (std::is_same_v<T, SubgraphBoundaryOp>) {
                        checkIoMap(concrete.ioMap);
                        for (const auto& mapping : concrete.bufferMappings) {
                            requireDeclared(mapping.source);
                        }
                    } else if constexpr (std::is_same_v<T, LoopOp>) {
                        checkIoMap(concrete.ioMap);
                        if (concrete.body) walk(*concrete.body);
                    } else if constexpr (std::is_same_v<T, ConditionalOp>) {
                        checkIoMap(concrete.ioMap);
                        if (concrete.thenRegion) walk(*concrete.thenRegion);
                        if (concrete.elseRegion) walk(*concrete.elseRegion);
                    }
                },
                op);
        }
    };

    walk(rootRegion);
}

// ---- RegionOp accessors -------------------------------------------------
//
// regionOpId() lives in control_node.hpp; the per-field accessors below are
// the std::visit-based equivalents for IOTypeMap / IOMap / afterOps. The
// compiler pattern-matches directly on the authored variant rather than
// going through a bespoke discriminator-and-pointer wrapper.

const IOTypeMap& regionOpIoType(const RegionOp& op) {
    return std::visit(
        [](const auto& concrete) -> const IOTypeMap& {
            using T = std::decay_t<decltype(concrete)>;
            if constexpr (std::is_same_v<T, KernelOp>) {
                return concrete.kernel.ioType;
            } else {
                return concrete.ioType;
            }
        },
        op);
}

const IOMap& regionOpIoMap(const RegionOp& op) {
    return std::visit(
        [](const auto& concrete) -> const IOMap& { return concrete.ioMap; },
        op);
}

const std::vector<std::string>& regionOpAfterOps(const RegionOp& op) {
    return std::visit(
        [](const auto& concrete) -> const std::vector<std::string>& {
            return concrete.afterOps;
        },
        op);
}

/// Build a stable index of pointers to the authored RegionOp variants in
/// insertion order. The pointers stay valid for the GraphRegion's lifetime
/// (its op list is append-only and never reallocates a previously-issued
/// op address).
std::vector<const RegionOp*> collectRegionOps(const GraphRegion& region) {
    std::vector<const RegionOp*> ops;
    ops.reserve(region.ops().size());
    for (const RegionOp& op : region.ops()) {
        ops.push_back(&op);
    }
    return ops;
}

void validateDeclaredRegionPorts(const RegionOp& op,
                                 const std::set<std::string>& opIds) {
    const std::string& opId = regionOpId(op);
    for (const auto& after : regionOpAfterOps(op)) {
        if (after == opId) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' cannot depend on itself via afterOps");
        }
        if (!opIds.count(after)) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' references unknown afterOps id '" +
                after + "'");
        }
    }

    const IOTypeMap& ioType = regionOpIoType(op);
    if (!hasDeclaredPorts(ioType)) return;

    const IOMap& ioMap = regionOpIoMap(op);

    for (const auto& expected : ioType.inputScalars) {
        auto it = ioMap.scalars().find(expected.name);
        if (it == ioMap.scalars().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing mandatory input scalar port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' input scalar '" + expected.name +
                "' type mismatch: declared " + scalarTypeName(expected.type) +
                ", bound " + scalarTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.outputScalars) {
        auto it = ioMap.scalars().find(expected.name);
        if (it == ioMap.scalars().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing mandatory output scalar port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' output scalar '" + expected.name +
                "' type mismatch: declared " + scalarTypeName(expected.type) +
                ", bound " + scalarTypeName(it->second.type()));
        }
        if (it->second.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' output scalar '" + expected.name +
                "' must be bound to GraphScalar::globalVar()");
        }
    }

    for (const auto& expected : ioType.inputBuffers) {
        auto it = ioMap.inputBuffers().find(expected.name);
        if (it == ioMap.inputBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing mandatory input buffer port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' input buffer '" + expected.name +
                "' type mismatch: declared " + bufferTypeName(expected.type) +
                ", bound " + bufferTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.outputBuffers) {
        auto it = ioMap.outputBuffers().find(expected.name);
        if (it == ioMap.outputBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing mandatory output buffer port '" +
                expected.name + "'");
        }
        if (it->second.type() != expected.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' output buffer '" + expected.name +
                "' type mismatch: declared " + bufferTypeName(expected.type) +
                ", bound " + bufferTypeName(it->second.type()));
        }
    }

    for (const auto& expected : ioType.rwBuffers) {
        auto it = std::find_if(ioMap.rwBuffers().begin(), ioMap.rwBuffers().end(),
                               [&](const IOMap::RWBinding& binding) {
                                   return binding.inPort == expected.in.name &&
                                          binding.outPort == expected.out.name;
                               });
        if (it == ioMap.rwBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing mandatory RW buffer ports '" +
                expected.in.name + "'/'" + expected.out.name + "'");
        }
        if (it->in.type() != expected.in.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' RW input buffer '" + expected.in.name +
                "' type mismatch: declared " + bufferTypeName(expected.in.type) +
                ", bound " + bufferTypeName(it->in.type()));
        }
        if (it->out.type() != expected.out.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' RW output buffer '" + expected.out.name +
                "' type mismatch: declared " + bufferTypeName(expected.out.type) +
                ", bound " + bufferTypeName(it->out.type()));
        }
    }

    for (const auto& [name, scalar] : ioMap.scalars()) {
        const auto* input = findPortByName(ioType.inputScalars, name);
        const auto* output = findPortByName(ioType.outputScalars, name);
        if (!input && !output) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' binds unknown scalar port '" + name + "'");
        }
        if (input && scalar.type() != input->type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' scalar '" + name +
                "' type mismatch: declared " + scalarTypeName(input->type) +
                ", bound " + scalarTypeName(scalar.type()));
        }
        if (output && scalar.type() != output->type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' scalar '" + name +
                "' type mismatch: declared " + scalarTypeName(output->type) +
                ", bound " + scalarTypeName(scalar.type()));
        }
        if (output && scalar.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' output scalar '" + name +
                "' must be bound to GraphScalar::globalVar()");
        }
    }

    for (const auto& [name, buffer] : ioMap.inputBuffers()) {
        const auto* port = findPortByName(ioType.inputBuffers, name);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' binds unknown input buffer port '" +
                name + "'");
        }
        if (buffer.type() != port->type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' input buffer '" + name +
                "' type mismatch: declared " + bufferTypeName(port->type) +
                ", bound " + bufferTypeName(buffer.type()));
        }
    }

    for (const auto& [name, buffer] : ioMap.outputBuffers()) {
        const auto* port = findPortByName(ioType.outputBuffers, name);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' binds unknown output buffer port '" +
                name + "'");
        }
        if (buffer.type() != port->type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' output buffer '" + name +
                "' type mismatch: declared " + bufferTypeName(port->type) +
                ", bound " + bufferTypeName(buffer.type()));
        }
    }

    for (const auto& binding : ioMap.rwBuffers()) {
        const auto* port = findRWPortByNames(ioType.rwBuffers, binding.inPort, binding.outPort);
        if (!port) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' binds unknown RW buffer ports '" +
                binding.inPort + "'/'" + binding.outPort + "'");
        }
        if (binding.in.type() != port->in.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' RW input buffer '" + binding.inPort +
                "' type mismatch: declared " + bufferTypeName(port->in.type) +
                ", bound " + bufferTypeName(binding.in.type()));
        }
        if (binding.out.type() != port->out.type) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' RW output buffer '" + binding.outPort +
                "' type mismatch: declared " + bufferTypeName(port->out.type) +
                ", bound " + bufferTypeName(binding.out.type()));
        }
    }
}

struct ConsumedScalarRef {
    std::string key;
    std::string name;
    uint64_t scopeId = 0;
    std::string kind;
};

struct ConsumedBufferRef {
    std::string key;
    std::string name;
    uint64_t scopeId = 0;
    std::string kind;
};

std::vector<ConsumedScalarRef> consumedScalarRefs(const RegionOp& op);
std::vector<std::string> consumedScalarKeys(const RegionOp& op);
std::vector<std::string> producedScalarKeys(const RegionOp& op);
std::vector<ConsumedBufferRef> consumedBufferRefs(const RegionOp& op);
std::vector<std::string> consumedBufferKeys(const RegionOp& op);
std::vector<std::string> producedBufferKeys(const RegionOp& op);

struct ProducerMapInfo {
    std::map<std::string, std::string> producers;
    std::map<std::pair<std::string, std::string>, std::string> loopCarriedInitialProducers;
};

std::set<std::string> loopCarriedBufferKeys(const LoopOp& loop) {
    std::set<std::string> imports;
    std::set<std::string> exports;

    for (const RegionOp& childOp : loop.body->ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary) continue;
        if (boundary->side == BoundarySide::Start) {
            for (const auto& mapping : boundary->bufferMappings) {
                if (mapping.source.scopeId() == boundary->parentScopeId) {
                    imports.insert(scopedBufferKey(mapping.source.scopeId(), mapping.source.name()));
                }
            }
        } else {
            for (const auto& mapping : boundary->bufferMappings) {
                if (mapping.target.scopeId() == boundary->parentScopeId) {
                    exports.insert(scopedBufferKey(mapping.target.scopeId(), mapping.target.name()));
                }
            }
        }
    }

    std::set<std::string> carried;
    std::set_intersection(imports.begin(), imports.end(),
                          exports.begin(), exports.end(),
                          std::inserter(carried, carried.begin()));
    return carried;
}

std::set<std::string> loopCarriedScalarKeys(const LoopOp& loop) {
    std::set<std::string> imports;
    std::set<std::string> exports;

    for (const RegionOp& childOp : loop.body->ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary) continue;
        if (boundary->side == BoundarySide::Start) {
            for (const auto& mapping : boundary->scalarMappings) {
                if (!mapping.source.isConstant() &&
                    mapping.source.scopeId() == boundary->parentScopeId) {
                    imports.insert(scopedScalarKey(mapping.source.scopeId(),
                                                   mapping.source.varName()));
                }
            }
        } else {
            for (const auto& mapping : boundary->scalarMappings) {
                if (!mapping.target.isConstant() &&
                    mapping.target.scopeId() == boundary->parentScopeId) {
                    exports.insert(scopedScalarKey(mapping.target.scopeId(),
                                                   mapping.target.varName()));
                }
            }
        }
    }

    std::set<std::string> carried;
    std::set_intersection(imports.begin(), imports.end(),
                          exports.begin(), exports.end(),
                          std::inserter(carried, carried.begin()));
    return carried;
}

/// Build a "string about producer X" hint that mentions the boundary-mutation
/// case explicitly. Helps authors spot stacked writes from a control op's
/// end-boundary on top of an explicit producer port.
std::string describeProducer(const RegionOp& op) {
    const std::string& opId = regionOpId(op);
    if (std::holds_alternative<LoopOp>(op)) {
        return "loop op '" + opId + "' (mutates via end-boundary in its body)";
    }
    if (std::holds_alternative<ConditionalOp>(op)) {
        return "conditional op '" + opId +
               "' (mutates via end-boundary in one or both branches)";
    }
    if (std::holds_alternative<SubgraphBoundaryOp>(op)) {
        return "boundary op '" + opId + "'";
    }
    if (std::holds_alternative<ReprogramOp>(op)) {
        return "reprogram op '" + opId + "'";
    }
    if (std::holds_alternative<KernelOp>(op)) {
        return "kernel op '" + opId + "'";
    }
    return "op '" + opId + "'";
}

std::string multipleProducersMessage(const std::string& kind, const std::string& key,
                                     const RegionOp& existing,
                                     const RegionOp& conflicting) {
    return "GraphCompiler: multiple ops write scoped " + kind + " '" + key +
           "': " + describeProducer(existing) + " and " + describeProducer(conflicting) +
           ". Either declare a single producer (e.g. an explicit output port) or "
           "sequence the writes via afterOps.";
}

bool isLoopCarriedKey(const RegionOp& op, const std::string& key, bool scalar) {
    const auto* loop = std::get_if<LoopOp>(&op);
    if (!loop) return false;
    const auto carried = scalar ? loopCarriedScalarKeys(*loop) : loopCarriedBufferKeys(*loop);
    return carried.count(key) != 0;
}

/// Verify FPGA dispatch image safety: every FPGA kernel that names an image
/// must be gated behind a reprogram of that same image via its afterOps. The
/// user region starts with no active image, so an ungated FPGA dispatch would
/// poke an absent kernel; reject it at compile() instead.
void validateFpgaImageSafety(const std::vector<const RegionOp*>& ops) {
    std::map<std::string, const RegionOp*> byId;
    for (const RegionOp* opPtr : ops) byId[regionOpId(*opPtr)] = opPtr;

    for (const RegionOp* opPtr : ops) {
        const auto* kernel = std::get_if<KernelOp>(opPtr);
        if (!kernel) continue;
        if (kernel->kernel.type != DeviceType::FPGA || !kernel->kernel.image) continue;

        const std::string& image = *kernel->kernel.image;
        bool gated = false;
        for (const auto& afterId : kernel->afterOps) {
            auto it = byId.find(afterId);
            if (it == byId.end()) continue;
            if (const auto* reprog = std::get_if<ReprogramOp>(it->second)) {
                if (reprog->imageId == image) {
                    gated = true;
                    break;
                }
            }
        }
        if (!gated) {
            throw std::runtime_error(
                "GraphCompiler: FPGA kernel '" + kernel->kernel.name + "' (op '" + kernel->id +
                "') requires image '" + image +
                "' but is not gated behind a reprogram of that image; declare "
                "`.after = {<reprogram of " + image + ">}` on the dispatch");
        }
    }
}

ProducerMapInfo buildRegionProducerMapInfo(const std::vector<const RegionOp*>& ops,
                                           bool scalar) {
    ProducerMapInfo info;
    std::map<std::string, const RegionOp*> producerOps;
    for (const RegionOp* opPtr : ops) {
        const RegionOp& op = *opPtr;
        const std::string& opId = regionOpId(op);
        const auto producedKeys = scalar ? producedScalarKeys(op) : producedBufferKeys(op);
        for (const auto& key : producedKeys) {
            auto [existing, inserted] = info.producers.emplace(key, opId);
            if (!inserted && existing->second != opId) {
                const RegionOp& existingOp = *producerOps[key];
                const bool existingIsCarriedLoop = isLoopCarriedKey(existingOp, key, scalar);
                const bool currentIsCarriedLoop = isLoopCarriedKey(op, key, scalar);

                if (existingIsCarriedLoop == currentIsCarriedLoop) {
                    throw std::runtime_error(multipleProducersMessage(
                        scalar ? "scalar" : "buffer", key, existingOp, op));
                }

                const std::string& loopId = existingIsCarriedLoop ? existing->second : opId;
                const std::string& initialId = existingIsCarriedLoop ? opId : existing->second;
                const auto initialKey = std::make_pair(loopId, key);
                auto [initialIt, initialInserted] =
                    info.loopCarriedInitialProducers.emplace(initialKey, initialId);
                if (!initialInserted && initialIt->second != initialId) {
                    throw std::runtime_error(
                        "GraphCompiler: loop op '" + loopId +
                        "' has multiple initial producers for carried " +
                        std::string(scalar ? "scalar" : "buffer") + " '" + key + "'");
                }

                if (currentIsCarriedLoop) {
                    existing->second = opId;
                    producerOps[key] = &op;
                }
                continue;
            }
            if (inserted) producerOps[key] = &op;
        }
    }
    return info;
}

std::map<std::string, std::string> buildRegionProducerMap(
    const std::vector<const RegionOp*>& ops) {
    return buildRegionProducerMapInfo(ops, /*scalar=*/false).producers;
}

std::map<std::string, std::string> buildRegionScalarProducerMap(
    const std::vector<const RegionOp*>& ops) {
    return buildRegionProducerMapInfo(ops, /*scalar=*/true).producers;
}

/**
 * Derived side-effect ordering edges, returned as `successor -> [predecessors]`.
 *
 * These are NOT data dependencies and are never author-listed:
 *  (a) Reprogram drain: a reprogram R chaining to a prior reprogram P (via
 *      afterOps) must wait on every op gated behind P (every op listing P in
 *      its afterOps), so the old image fully drains before reconfiguration.
 *  (b) Readers-before-mutator: an in-place (inout) op consuming token K must
 *      run after every pure-input reader of K.
 *
 * They must feed BOTH the topological sort (buildRegionAdjacency) AND the
 * runtime dependsOn barriers (populateDependsOn). The latter is essential: on
 * the asynchronous FPGA scheduler only dependsOn barriers gate execution, so a
 * drain edge that lives solely in the topo order does not actually keep a
 * reprogram from reconfiguring before the kernels of the old image have drained.
 */
std::map<std::string, std::vector<std::string>> computeSideEffectOrderingEdges(
    const std::vector<const RegionOp*>& ops) {
    std::map<std::string, std::vector<std::string>> edges;  // succ -> preds

    std::map<std::string, const RegionOp*> byId;
    for (const RegionOp* opPtr : ops) byId[regionOpId(*opPtr)] = opPtr;

    auto isReprogram = [&](const std::string& id) {
        auto it = byId.find(id);
        return it != byId.end() && std::holds_alternative<ReprogramOp>(*it->second);
    };

    // (a) Reprogram drain.
    for (const RegionOp* opPtr : ops) {
        const auto* reprog = std::get_if<ReprogramOp>(opPtr);
        if (!reprog) continue;
        for (const auto& prior : reprog->afterOps) {
            if (!isReprogram(prior)) continue;
            for (const RegionOp* otherPtr : ops) {
                const std::string& otherId = regionOpId(*otherPtr);
                if (otherId == reprog->id) continue;
                const auto& otherAfter = regionOpAfterOps(*otherPtr);
                if (std::find(otherAfter.begin(), otherAfter.end(), prior) !=
                    otherAfter.end()) {
                    edges[reprog->id].push_back(otherId);
                }
            }
        }
    }

    // (b) Readers-before-mutator.
    for (const RegionOp* opPtr : ops) {
        const auto* mutator = std::get_if<KernelOp>(opPtr);
        if (!mutator || mutator->ioMap.rwBuffers().empty()) continue;
        for (const auto& rw : mutator->ioMap.rwBuffers()) {
            const std::string inKey = scopedBufferKey(rw.in.scopeId(), rw.in.name());
            for (const RegionOp* readerPtr : ops) {
                const std::string& readerId = regionOpId(*readerPtr);
                if (readerId == mutator->id) continue;
                bool readsKey = false;
                for (const auto& [port, buf] : regionOpIoMap(*readerPtr).inputBuffers()) {
                    (void)port;
                    if (scopedBufferKey(buf.scopeId(), buf.name()) == inKey) {
                        readsKey = true;
                        break;
                    }
                }
                if (readsKey) edges[mutator->id].push_back(readerId);
            }
        }
    }

    return edges;
}

/**
 * Build the producer/consumer adjacency map for a region.
 *
 * Synthesised bridge node ids and DGraph ordering are stable for a given
 * input as long as the topological order is. Changes to producer-map
 * iteration order or topo tiebreaking will renumber bridges and may shift
 * any test fixture that matches by id.
 */
std::map<std::string, std::vector<std::string>> buildRegionAdjacency(
    const std::vector<const RegionOp*>& ops,
    const ProducerMapInfo& bufferProducers,
    const ProducerMapInfo& scalarProducers) {

    std::map<std::string, std::vector<std::string>> adj;
    for (const RegionOp* opPtr : ops) {
        adj.emplace(regionOpId(*opPtr), std::vector<std::string>{});
    }

    for (const RegionOp* opPtr : ops) {
        const RegionOp& op = *opPtr;
        const std::string& opId = regionOpId(op);
        for (const auto& key : consumedBufferKeys(op)) {
            auto carriedIt = bufferProducers.loopCarriedInitialProducers.find({opId, key});
            std::string producerId;
            if (carriedIt != bufferProducers.loopCarriedInitialProducers.end()) {
                producerId = carriedIt->second;
            } else if (auto producerIt = bufferProducers.producers.find(key);
                       producerIt != bufferProducers.producers.end()) {
                producerId = producerIt->second;
            }
            if (!producerId.empty() && producerId != opId) {
                adj[producerId].push_back(opId);
            }
        }
        for (const auto& key : consumedScalarKeys(op)) {
            auto carriedIt = scalarProducers.loopCarriedInitialProducers.find({opId, key});
            std::string producerId;
            if (carriedIt != scalarProducers.loopCarriedInitialProducers.end()) {
                producerId = carriedIt->second;
            } else if (auto producerIt = scalarProducers.producers.find(key);
                       producerIt != scalarProducers.producers.end()) {
                producerId = producerIt->second;
            }
            if (!producerId.empty() && producerId != opId) {
                adj[producerId].push_back(opId);
            }
        }
        for (const auto& after : regionOpAfterOps(op)) {
            adj[after].push_back(opId);
        }
    }

    // Side-effect ordering edges (reprogram drain, readers-before-mutator) feed
    // the topological sort here and the runtime dependsOn barriers in
    // populateDependsOn (see computeSideEffectOrderingEdges).
    for (const auto& [succ, preds] : computeSideEffectOrderingEdges(ops)) {
        for (const auto& pred : preds) {
            adj[pred].push_back(succ);
        }
    }

    return adj;
}

/**
 * Topologically sort the ops in @p ops by the adjacency map @p adj.
 *
 * @see buildRegionAdjacency for the stability note that also applies here:
 * the Kahn-style sweep is deterministic given a deterministic @p adj, so
 * downstream device-list / bridge-id ordering only changes when this
 * function's tie-breaking does.
 *
 * STABILITY INVARIANT (matters for test fixtures): synthesised bridge node
 * ids - those minted via BridgeIds::mint() with prefixes _bridge_,
 * _barrier_, _control_output_bridge_ - and the resulting DGraph node order
 * are stable for a given input as long as this topological order is stable.
 * Changes to producer-map iteration order, this function's tie-breaking,
 * or BridgeIds::mint()'s counter scheme will renumber bridges and may
 * shift any test fixture that matches by id.
 */
std::vector<std::string> topoSortRegion(
    const std::vector<const RegionOp*>& ops,
    const std::map<std::string, std::vector<std::string>>& adj) {
    std::map<std::string, int> inDegree;
    for (const RegionOp* opPtr : ops) {
        inDegree[regionOpId(*opPtr)] = 0;
    }
    for (const auto& [src, successors] : adj) {
        (void)src;
        for (const auto& dst : successors) {
            ++inDegree[dst];
        }
    }

    std::queue<std::string> ready;
    for (const RegionOp* opPtr : ops) {
        const std::string& opId = regionOpId(*opPtr);
        if (inDegree[opId] == 0) ready.push(opId);
    }

    std::vector<std::string> sorted;
    sorted.reserve(ops.size());
    while (!ready.empty()) {
        auto current = ready.front();
        ready.pop();
        sorted.push_back(current);

        auto it = adj.find(current);
        if (it != adj.end()) {
            for (const auto& succ : it->second) {
                if (--inDegree[succ] == 0) ready.push(succ);
            }
        }
    }

    if (sorted.size() != ops.size()) {
        throw std::runtime_error("GraphCompiler: cycle detected in region dependency graph");
    }

    return sorted;
}

std::vector<std::string>& mutableCompiledNodeDependsOn(CompiledNode& node) {
    return std::visit([](auto& concrete) -> std::vector<std::string>& {
        return concrete.dependsOn;
    }, node);
}

struct OutputBufferBinding {
    std::string portName;
    std::string tokenName;
    uint64_t scopeId = 0;
    BufferType type = BufferType::U8;
};

struct OutputScalarBinding {
    std::string portName;
    std::string tokenName;
    uint64_t scopeId = 0;
    ScalarType type = ScalarType::U8;
};

struct RegionOutputBindings {
    std::vector<OutputBufferBinding> buffers;
    std::vector<OutputScalarBinding> scalars;
};

struct OutputBufferProducer {
    OutputBufferBinding binding;
    std::string opId;
    std::string deviceId;
};

struct OutputScalarProducer {
    OutputScalarBinding binding;
    std::string opId;
    std::string deviceId;
};

struct RegionExitProducers {
    std::map<std::string, OutputBufferProducer> buffersByPort;
    std::map<std::string, OutputScalarProducer> scalarsByPort;
};

struct ControlBufferMaterialization {
    DGraphChildRole role = DGraphChildRole::LoopBody;
    size_t publicationIndex = 0;
    OutputBufferBinding sourceBinding;
    std::string producerOpId;
    std::string sourceDeviceId;
    std::string placementDeviceId;
};

/// Compiler-internal aggregate describing the device placement and
/// per-publication metadata for a single LoopOp's parent-facing outputs.
/// One instance is built per LoopOp during resolveControlOutputPlacements()
/// and consumed by makeCompiledRegionNode() to populate a CompiledLoopNode.
struct CompiledLoopOutputPlacement {
    std::map<std::string, std::string> buffers;
    std::map<std::string, std::string> scalars;
    std::vector<CompiledLoopBufferPublication> bufferPublications;
    std::vector<CompiledLoopScalarPublication> scalarPublications;
    std::vector<ControlBufferMaterialization> bufferMaterializations;
};

/// Conditional counterpart to CompiledLoopOutputPlacement.
struct CompiledConditionalOutputPlacement {
    std::map<std::string, std::string> buffers;
    std::map<std::string, std::string> scalars;
    std::vector<CompiledConditionalBufferPublication> bufferPublications;
    std::vector<CompiledConditionalScalarPublication> scalarPublications;
    std::vector<ControlBufferMaterialization> bufferMaterializations;
};

CompiledLoopKind compiledLoopKind(LoopKind kind) {
    switch (kind) {
        case LoopKind::FixedCount: return CompiledLoopKind::FixedCount;
        case LoopKind::WhileCondition: return CompiledLoopKind::WhileCondition;
    }
    return CompiledLoopKind::FixedCount;
}

const IOMap::RWBinding* findRWBindingByNames(const IOMap& ioMap,
                                             const std::string& inName,
                                             const std::string& outName) {
    auto it = std::find_if(ioMap.rwBuffers().begin(), ioMap.rwBuffers().end(),
                           [&](const IOMap::RWBinding& binding) {
                               return binding.inPort == inName && binding.outPort == outName;
                           });
    return (it == ioMap.rwBuffers().end()) ? nullptr : &*it;
}

RegionOutputBindings collectOutputBindings(const IOTypeMap& ioType,
                                           const IOMap& ioMap,
                                           const std::string& opId) {
    RegionOutputBindings outputs;

    for (const auto& port : ioType.outputBuffers) {
        auto bindingIt = ioMap.outputBuffers().find(port.name);
        if (bindingIt == ioMap.outputBuffers().end()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing output buffer binding for port '" +
                port.name + "'");
        }
        outputs.buffers.push_back(
            OutputBufferBinding{port.name, bindingIt->second.name(),
                                bindingIt->second.scopeId(), port.type});
    }

    for (const auto& port : ioType.rwBuffers) {
        const IOMap::RWBinding* binding =
            findRWBindingByNames(ioMap, port.in.name, port.out.name);
        if (!binding) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing RW output buffer binding for port '" +
                port.out.name + "'");
        }
        outputs.buffers.push_back(
            OutputBufferBinding{port.out.name, binding->out.name(),
                                binding->out.scopeId(), port.out.type});
    }

    for (const auto& port : ioType.outputScalars) {
        auto bindingIt = ioMap.scalars().find(port.name);
        if (bindingIt == ioMap.scalars().end() || bindingIt->second.isConstant()) {
            throw std::runtime_error(
                "GraphCompiler: op '" + opId + "' missing output scalar binding for port '" +
                port.name + "'");
        }
        outputs.scalars.push_back(OutputScalarBinding{port.name,
                                                      bindingIt->second.varName(),
                                                      bindingIt->second.scopeId(),
                                                      port.type});
    }

    return outputs;
}

void appendConsumedScalar(std::vector<ConsumedScalarRef>& refs,
                          std::string name,
                          uint64_t scopeId,
                          std::string kind) {
    refs.push_back(ConsumedScalarRef{scopedScalarKey(scopeId, name),
                                     std::move(name),
                                     scopeId,
                                     std::move(kind)});
}

void appendConsumedBuffer(std::vector<ConsumedBufferRef>& refs,
                          const GraphBuffer& buffer,
                          std::string kind) {
    refs.push_back(ConsumedBufferRef{scopedBufferKey(buffer.scopeId(), buffer.name()),
                                     buffer.name(),
                                     buffer.scopeId(),
                                     std::move(kind)});
}

void appendConditionScalarRefs(std::vector<ConsumedScalarRef>& refs,
                               const Condition& condition,
                               const std::string& kind) {
    auto appendOperand = [&](const std::optional<ConditionOperand>& operand) {
        if (!operand || !operand->isScalar()) return;
        appendConsumedScalar(refs, operand->name(), operand->scopeId(), kind);
    };
    appendOperand(condition.lhs());
    appendOperand(condition.rhs());
    appendOperand(condition.epsilon());
}

void appendChildStartBoundaryScalarRefs(std::vector<ConsumedScalarRef>& refs,
                                        const GraphRegion& child,
                                        uint64_t parentScopeId,
                                        const std::string& kind) {
    for (const RegionOp& childOp : child.ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary || boundary->side != BoundarySide::Start) continue;
        for (const auto& mapping : boundary->scalarMappings) {
            if (mapping.source.isConstant()) continue;
            if (mapping.source.scopeId() != parentScopeId) continue;
            appendConsumedScalar(refs, mapping.source.varName(), mapping.source.scopeId(), kind);
        }
    }
}

void appendChildStartBoundaryBufferRefs(std::vector<ConsumedBufferRef>& refs,
                                        const GraphRegion& child,
                                        uint64_t parentScopeId,
                                        const std::string& kind) {
    for (const RegionOp& childOp : child.ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary || boundary->side != BoundarySide::Start) continue;
        for (const auto& mapping : boundary->bufferMappings) {
            if (mapping.source.scopeId() != parentScopeId) continue;
            appendConsumedBuffer(refs, mapping.source, kind);
        }
    }
}

/// Collect parent-scope scalar targets written by end boundaries inside @p
/// child. The owning loop / conditional op acts as the parent-region producer
/// of these scalars so subsequent parent-scope readers gain a `dependsOn`
/// edge to it.
void appendChildEndBoundaryScalarTargets(std::vector<std::string>& keys,
                                         const GraphRegion& child,
                                         uint64_t parentScopeId) {
    for (const RegionOp& childOp : child.ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary || boundary->side != BoundarySide::End) continue;
        for (const auto& mapping : boundary->scalarMappings) {
            if (mapping.target.isConstant()) continue;
            if (mapping.target.scopeId() != parentScopeId) continue;
            keys.push_back(scopedScalarKey(mapping.target.scopeId(),
                                           mapping.target.varName()));
        }
    }
}

/// Collect parent-scope buffer targets written by end boundaries inside @p
/// child. Counterpart of appendChildEndBoundaryScalarTargets() for buffers.
void appendChildEndBoundaryBufferTargets(std::vector<std::string>& keys,
                                         const GraphRegion& child,
                                         uint64_t parentScopeId) {
    for (const RegionOp& childOp : child.ops()) {
        const auto* boundary = std::get_if<SubgraphBoundaryOp>(&childOp);
        if (!boundary || boundary->side != BoundarySide::End) continue;
        for (const auto& mapping : boundary->bufferMappings) {
            if (mapping.target.scopeId() != parentScopeId) continue;
            keys.push_back(scopedBufferKey(mapping.target.scopeId(),
                                           mapping.target.name()));
        }
    }
}

std::vector<ConsumedBufferRef> consumedBufferRefs(const RegionOp& op) {
    std::vector<ConsumedBufferRef> refs;
    const IOMap& ioMap = regionOpIoMap(op);
    for (const auto& [port, buffer] : ioMap.inputBuffers()) {
        (void)port;
        appendConsumedBuffer(refs, buffer, "input buffer");
    }
    for (const auto& rw : ioMap.rwBuffers()) {
        appendConsumedBuffer(refs, rw.in, "RW input buffer");
    }
    if (const auto* boundary = std::get_if<SubgraphBoundaryOp>(&op)) {
        for (const auto& mapping : boundary->bufferMappings) {
            appendConsumedBuffer(refs, mapping.source, "boundary buffer source");
        }
    } else if (const auto* loop = std::get_if<LoopOp>(&op)) {
        appendChildStartBoundaryBufferRefs(refs, *loop->body,
                                           loop->body->parentScopeId(),
                                           "child boundary buffer source");
    } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
        appendChildStartBoundaryBufferRefs(refs, *cond->thenRegion,
                                           cond->thenRegion->parentScopeId(),
                                           "child boundary buffer source");
        appendChildStartBoundaryBufferRefs(refs, *cond->elseRegion,
                                           cond->elseRegion->parentScopeId(),
                                           "child boundary buffer source");
    }
    // Dedup by scoped key. The same parent token can be reached through both
    // the parent op's IOMap inputs and a child start-boundary import, and we
    // do not want callers to see duplicate adjacency / dependency edges.
    std::set<std::string> seen;
    std::vector<ConsumedBufferRef> dedup;
    dedup.reserve(refs.size());
    for (auto& ref : refs) {
        if (seen.insert(ref.key).second) dedup.push_back(std::move(ref));
    }
    return dedup;
}

std::vector<std::string> consumedBufferKeys(const RegionOp& op) {
    std::vector<std::string> keys;
    for (const auto& ref : consumedBufferRefs(op)) {
        keys.push_back(ref.key);
    }
    return keys;
}

std::vector<std::string> producedBufferKeys(const RegionOp& op) {
    std::vector<std::string> keys;
    const IOMap& ioMap = regionOpIoMap(op);
    for (const auto& [port, buffer] : ioMap.outputBuffers()) {
        (void)port;
        keys.push_back(scopedBufferKey(buffer.scopeId(), buffer.name()));
    }
    for (const auto& rw : ioMap.rwBuffers()) {
        keys.push_back(scopedBufferKey(rw.out.scopeId(), rw.out.name()));
    }
    if (const auto* boundary = std::get_if<SubgraphBoundaryOp>(&op)) {
        for (const auto& mapping : boundary->bufferMappings) {
            if (mapping.target.scopeId() != boundary->localScopeId) continue;
            keys.push_back(scopedBufferKey(mapping.target.scopeId(), mapping.target.name()));
        }
    } else if (const auto* loop = std::get_if<LoopOp>(&op)) {
        appendChildEndBoundaryBufferTargets(keys, *loop->body,
                                            loop->body->parentScopeId());
    } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
        appendChildEndBoundaryBufferTargets(keys, *cond->thenRegion,
                                            cond->thenRegion->parentScopeId());
        appendChildEndBoundaryBufferTargets(keys, *cond->elseRegion,
                                            cond->elseRegion->parentScopeId());
    }
    return keys;
}

std::vector<ConsumedScalarRef> consumedScalarRefs(const RegionOp& op) {
    std::vector<ConsumedScalarRef> refs;
    const IOTypeMap& ioType = regionOpIoType(op);
    const IOMap& ioMap = regionOpIoMap(op);
    for (const auto& port : ioType.inputScalars) {
        auto scalarIt = ioMap.scalars().find(port.name);
        if (scalarIt == ioMap.scalars().end() || scalarIt->second.isConstant()) continue;
        appendConsumedScalar(refs, scalarIt->second.varName(), scalarIt->second.scopeId(),
                             "input scalar");
    }
    if (const auto* boundary = std::get_if<SubgraphBoundaryOp>(&op)) {
        for (const auto& mapping : boundary->scalarMappings) {
            if (mapping.source.isConstant()) continue;
            appendConsumedScalar(refs, mapping.source.varName(), mapping.source.scopeId(),
                                 "boundary scalar source");
        }
    } else if (const auto* loop = std::get_if<LoopOp>(&op)) {
        if (loop->tripCount && loop->tripCount->kind() == LoopTripCount::Kind::Scalar) {
            appendConsumedScalar(refs, loop->tripCount->name(), loop->tripCount->scopeId(),
                                 "trip-count scalar");
        }
        if (loop->condition) {
            appendConditionScalarRefs(refs, *loop->condition, "condition scalar");
        }
        appendChildStartBoundaryScalarRefs(refs, *loop->body,
                                           loop->body->parentScopeId(),
                                           "child boundary scalar source");
    } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
        appendConditionScalarRefs(refs, cond->condition, "condition scalar");
        appendChildStartBoundaryScalarRefs(refs, *cond->thenRegion,
                                           cond->thenRegion->parentScopeId(),
                                           "child boundary scalar source");
        appendChildStartBoundaryScalarRefs(refs, *cond->elseRegion,
                                           cond->elseRegion->parentScopeId(),
                                           "child boundary scalar source");
    }
    // Dedup by scoped key so the same parent scalar reached via multiple
    // routes (kernel input + boundary import, condition + boundary import,
    // both branches importing the same parent token, ...) only contributes
    // a single dependency edge.
    std::set<std::string> seen;
    std::vector<ConsumedScalarRef> dedup;
    dedup.reserve(refs.size());
    for (auto& ref : refs) {
        if (seen.insert(ref.key).second) dedup.push_back(std::move(ref));
    }
    return dedup;
}

std::vector<std::string> consumedScalarKeys(const RegionOp& op) {
    std::vector<std::string> keys;
    for (const auto& ref : consumedScalarRefs(op)) {
        keys.push_back(ref.key);
    }
    return keys;
}

std::vector<std::string> producedScalarKeys(const RegionOp& op) {
    std::vector<std::string> keys;
    const IOTypeMap& ioType = regionOpIoType(op);
    const IOMap& ioMap = regionOpIoMap(op);
    for (const auto& port : ioType.outputScalars) {
        auto scalarIt = ioMap.scalars().find(port.name);
        if (scalarIt == ioMap.scalars().end() || scalarIt->second.isConstant()) continue;
        keys.push_back(scopedScalarKey(scalarIt->second.scopeId(), scalarIt->second.varName()));
    }
    if (const auto* boundary = std::get_if<SubgraphBoundaryOp>(&op)) {
        for (const auto& mapping : boundary->scalarMappings) {
            if (mapping.target.isConstant()) continue;
            if (mapping.target.scopeId() != boundary->localScopeId) continue;
            keys.push_back(scopedScalarKey(mapping.target.scopeId(), mapping.target.varName()));
        }
    } else if (const auto* loop = std::get_if<LoopOp>(&op)) {
        appendChildEndBoundaryScalarTargets(keys, *loop->body,
                                            loop->body->parentScopeId());
    } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
        appendChildEndBoundaryScalarTargets(keys, *cond->thenRegion,
                                            cond->thenRegion->parentScopeId());
        appendChildEndBoundaryScalarTargets(keys, *cond->elseRegion,
                                            cond->elseRegion->parentScopeId());
    }
    return keys;
}

void validateRegionScalarProvenance(
    const GraphRegion& region,
    const std::vector<const RegionOp*>& ops,
    const std::map<std::string, std::string>& scalarProducerMap) {
    const uint64_t regionScope = region.scopeId();
    for (const RegionOp* opPtr : ops) {
        const RegionOp& op = *opPtr;
        for (const auto& ref : consumedScalarRefs(op)) {
            if (ref.scopeId != regionScope) continue;
            if (scalarProducerMap.count(ref.key)) continue;
            // Root scope (0) is validated separately by Graph::validateDeclaredScalars,
            // which knows the user-declared globals; nested regions enforce strict
            // start-boundary imports so we keep this check active for them.
            if (regionScope == 0) continue;
            throw std::runtime_error(
                "GraphCompiler: op '" + regionOpId(op) + "' reads " + ref.kind + " '" +
                ref.name + "' in region scope " + std::to_string(regionScope) +
                " before it is produced locally or imported through a start boundary");
        }
    }
}

void validateRegionBufferProvenance(
    const GraphRegion& region,
    const std::vector<const RegionOp*>& ops,
    const std::map<std::string, std::string>& bufferProducerMap) {
    const uint64_t regionScope = region.scopeId();
    for (const RegionOp* opPtr : ops) {
        const RegionOp& op = *opPtr;
        for (const auto& ref : consumedBufferRefs(op)) {
            if (ref.scopeId != regionScope) continue;
            if (bufferProducerMap.count(ref.key)) continue;
            // Root scope (0) is validated separately by Graph::validateDeclaredBuffers,
            // which knows the user-declared inputs; nested regions enforce strict
            // start-boundary imports so we keep this check active for them.
            if (regionScope == 0) continue;
            throw std::runtime_error(
                "GraphCompiler: op '" + regionOpId(op) + "' reads " + ref.kind + " '" +
                ref.name + "' in region scope " + std::to_string(regionScope) +
                " before it is produced locally or imported through a start boundary");
        }
    }
}

RegionOutputBindings collectOutputBindings(const RegionOp& op) {
    return collectOutputBindings(regionOpIoType(op), regionOpIoMap(op), regionOpId(op));
}

const CompiledNode* findCompiledNodeInChildDGraphs(const DGraphChild& child,
                                                   const std::string& nodeId,
                                                   std::string& deviceId) {
    for (const auto& dgraph : child.dgraphs) {
        if (!dgraph) continue;
        for (const CompiledNode& node : dgraph->nodes) {
            if (compiledNodeId(node) != nodeId) continue;
            deviceId = dgraph->deviceId;
            return &node;
        }
    }
    return nullptr;
}

std::string compiledOutputBufferPlacement(const CompiledNode& node,
                                          const OutputBufferBinding& binding,
                                          const std::string& fallbackDeviceId) {
    const std::map<std::string, std::string>* placements = nullptr;
    if (const auto* loop = std::get_if<CompiledLoopNode>(&node)) {
        placements = &loop->outputBufferPlacements;
    } else if (const auto* cond = std::get_if<CompiledConditionalNode>(&node)) {
        placements = &cond->outputBufferPlacements;
    }
    if (placements != nullptr) {
        auto it = placements->find(scopedBufferKey(binding.scopeId, binding.tokenName));
        if (it != placements->end()) return it->second;
    }
    return fallbackDeviceId;
}

std::string compiledOutputScalarPlacement(const CompiledNode& node,
                                          const OutputScalarBinding& binding,
                                          const std::string& fallbackDeviceId) {
    const std::map<std::string, std::string>* placements = nullptr;
    if (const auto* loop = std::get_if<CompiledLoopNode>(&node)) {
        placements = &loop->outputScalarPlacements;
    } else if (const auto* cond = std::get_if<CompiledConditionalNode>(&node)) {
        placements = &cond->outputScalarPlacements;
    }
    if (placements != nullptr) {
        auto it = placements->find(scopedScalarKey(binding.scopeId, binding.tokenName));
        if (it != placements->end()) return it->second;
    }
    return fallbackDeviceId;
}

std::string formatDeviceSet(const std::set<std::string>& devices) {
    std::string result;
    for (const auto& deviceId : devices) {
        if (!result.empty()) result += ", ";
        result += deviceId;
    }
    return result;
}

std::string resolveOutputPlacement(
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const std::string& controlKind,
    const std::string& controlId,
    const std::string& outputKind,
    const std::string& portName,
    const std::set<std::string>& producerDevices,
    const std::map<std::string, std::string>& placementHints) {
    auto hintIt = placementHints.find(portName);
    if (hintIt != placementHints.end()) {
        if (devices.find(hintIt->second) == devices.end()) {
            throw std::runtime_error(
                "GraphCompiler: " + controlKind + " '" + controlId + "' output " +
                outputKind + " port '" + portName + "' declares unknown placement device '" +
                hintIt->second + "'");
        }
        return hintIt->second;
    }

    if (producerDevices.size() == 1) return *producerDevices.begin();

    throw std::runtime_error(
        "GraphCompiler: " + controlKind + " '" + controlId + "' output " + outputKind +
        " port '" + portName + "' has ambiguous placement across devices {" +
        formatDeviceSet(producerDevices) + "}; declare an explicit output placement");
}

RegionExitProducers collectRegionExitProducers(const GraphRegion& region,
                                               const DGraphChild& child) {
    RegionExitProducers producers;

    for (const RegionOp& op : region.ops()) {
        RegionOutputBindings outputs = collectOutputBindings(op);
        if (outputs.buffers.empty() && outputs.scalars.empty()) continue;

        const std::string& opId = regionOpId(op);
        std::string opDeviceId;
        const CompiledNode* compiledNode =
            findCompiledNodeInChildDGraphs(child, opId, opDeviceId);
        if (!compiledNode) {
            throw std::runtime_error(
                "GraphCompiler: missing compiled child node for output-producing op '" +
                opId + "'");
        }

        for (const auto& binding : outputs.buffers) {
            OutputBufferProducer producer{
                binding,
                opId,
                compiledOutputBufferPlacement(*compiledNode, binding, opDeviceId)};
            auto [existing, inserted] = producers.buffersByPort.emplace(binding.portName,
                                                                        producer);
            if (!inserted) {
                throw std::runtime_error(
                    "GraphCompiler: output buffer port '" + binding.portName +
                    "' has multiple exit producers ('" + existing->second.opId + "' and '" +
                    opId + "') in region scope " + std::to_string(region.scopeId()));
            }
        }

        for (const auto& binding : outputs.scalars) {
            OutputScalarProducer producer{
                binding,
                opId,
                compiledOutputScalarPlacement(*compiledNode, binding, opDeviceId)};
            auto [existing, inserted] = producers.scalarsByPort.emplace(binding.portName,
                                                                        producer);
            if (!inserted) {
                throw std::runtime_error(
                    "GraphCompiler: output scalar port '" + binding.portName +
                    "' has multiple exit producers ('" + existing->second.opId + "' and '" +
                    opId + "') in region scope " + std::to_string(region.scopeId()));
            }
        }
    }

    return producers;
}

const DGraphChild& requireChildDGraphs(const std::vector<DGraphChild>& children,
                                      const std::string& controlId,
                                      DGraphChildRole role) {
    for (const auto& child : children) {
        if (child.parentNodeId == controlId && child.role == role) return child;
    }
    throw std::runtime_error(
        "GraphCompiler: control op '" + controlId + "' is missing compiled child DGraphs");
}

DGraphChild& requireMutableChildDGraphs(std::vector<DGraphChild>& children,
                                        const std::string& controlId,
                                        DGraphChildRole role) {
    for (auto& child : children) {
        if (child.parentNodeId == controlId && child.role == role) return child;
    }
    throw std::runtime_error(
        "GraphCompiler: control op '" + controlId + "' is missing compiled child DGraphs");
}

DGraph& ensureChildDGraph(
    DGraphChild& child,
    const std::string& deviceId,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const std::shared_ptr<std::map<std::string, uint64_t>>& scalarValues) {
    for (auto& dgraph : child.dgraphs) {
        if (dgraph && dgraph->deviceId == deviceId) return *dgraph;
    }

    auto deviceIt = devices.find(deviceId);
    if (deviceIt == devices.end()) {
        throw std::runtime_error(
            "GraphCompiler: output placement transfer targets unknown device '" +
            deviceId + "'");
    }

    DGraph dgraph;
    dgraph.deviceId = deviceId;
    dgraph.device = deviceIt->second;
    dgraph.scalarValues = scalarValues;
    child.dgraphs.push_back(std::make_shared<DGraph>(std::move(dgraph)));
    return *child.dgraphs.back();
}

size_t childNodeIndex(const DGraph& dgraph, const std::string& nodeId) {
    for (size_t i = 0; i < dgraph.nodes.size(); ++i) {
        if (compiledNodeId(dgraph.nodes[i]) == nodeId) return i;
    }
    return std::numeric_limits<size_t>::max();
}

void retargetLoopBufferPublication(CompiledLoopOutputPlacement& placement,
                                   const ControlBufferMaterialization& materialization) {
    if (materialization.publicationIndex >= placement.bufferPublications.size()) {
        throw std::runtime_error(
            "GraphCompiler: output placement publication index is out of range");
    }
    if (materialization.role != DGraphChildRole::LoopBody) {
        throw std::runtime_error(
            "GraphCompiler: loop output publication has unexpected non-LoopBody role");
    }
    placement.bufferPublications[materialization.publicationIndex].sourceDeviceId =
        materialization.placementDeviceId;
}

void retargetConditionalBufferPublication(CompiledConditionalOutputPlacement& placement,
                                          const ControlBufferMaterialization& materialization) {
    if (materialization.publicationIndex >= placement.bufferPublications.size()) {
        throw std::runtime_error(
            "GraphCompiler: output placement publication index is out of range");
    }
    auto& publication = placement.bufferPublications[materialization.publicationIndex];
    switch (materialization.role) {
        case DGraphChildRole::LoopBody:
            throw std::runtime_error(
                "GraphCompiler: conditional output publication has unexpected LoopBody role");
        case DGraphChildRole::ConditionalThen:
            publication.thenSourceDeviceId = materialization.placementDeviceId;
            break;
        case DGraphChildRole::ConditionalElse:
            publication.elseSourceDeviceId = materialization.placementDeviceId;
            break;
    }
}

void materializeControlOutputBufferTransfer(
    DGraphChild& child,
    const std::string& controlId,
    const ControlBufferMaterialization& materialization,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const GraphCompiler::BridgeFor& bridgeFor,
    IDevice& cpuDevice,
    const std::shared_ptr<std::map<std::string, uint64_t>>& scalarValues,
    uint32_t& bridgeCounter) {
    if (materialization.sourceDeviceId == materialization.placementDeviceId) return;

    auto sourceDeviceIt = devices.find(materialization.sourceDeviceId);
    auto placementDeviceIt = devices.find(materialization.placementDeviceId);
    if (sourceDeviceIt == devices.end() || placementDeviceIt == devices.end()) {
        throw std::runtime_error(
            "GraphCompiler: output placement transfer references an unknown device");
    }

    const GraphBuffer sourceBuffer = GraphBuffer::make(
        materialization.sourceBinding.type,
        materialization.sourceBinding.tokenName,
        materialization.sourceBinding.scopeId);
    auto legs = BridgeRouter::routeTransfer(
        *sourceDeviceIt->second,
        *placementDeviceIt->second,
        sourceBuffer,
        0,
        bridgeFor,
        cpuDevice,
        materialization.producerOpId,
        controlId);

    std::string previousConsumerId;
    for (const auto& leg : legs) {
        auto [producerId, consumerId] =
            BridgeIds::mint(BridgeIds::Kind::ControlOutputBridge, bridgeCounter);
        ++bridgeCounter;

        CompiledBridgeOpNode producerNode{
            producerId,
            leg.srcDeviceId,
            leg.pair.op,
            leg.pair.producerAction,
            CompiledBridgeOpNode::Side::Producer,
            leg.producerKernelId};
        producerNode.dependsOn.push_back(leg.producerKernelId);
        if (!previousConsumerId.empty()) producerNode.dependsOn.push_back(previousConsumerId);

        CompiledBridgeOpNode consumerNode{
            consumerId,
            leg.dstDeviceId,
            leg.pair.op,
            leg.pair.consumerAction,
            CompiledBridgeOpNode::Side::Consumer,
            leg.consumerKernelId};
        consumerNode.tryReady = leg.pair.consumerTryReady;

        DGraph& producerDGraph = ensureChildDGraph(child, leg.srcDeviceId, devices, scalarValues);
        const size_t producerIndex = childNodeIndex(producerDGraph, leg.producerKernelId);
        if (producerIndex == std::numeric_limits<size_t>::max()) {
            producerDGraph.nodes.emplace_back(std::move(producerNode));
        } else {
            producerDGraph.nodes.insert(producerDGraph.nodes.begin() +
                                            static_cast<std::ptrdiff_t>(producerIndex + 1),
                                        std::move(producerNode));
        }

        DGraph& consumerDGraph = ensureChildDGraph(child, leg.dstDeviceId, devices, scalarValues);
        consumerDGraph.nodes.emplace_back(std::move(consumerNode));
        previousConsumerId = consumerId;
    }

    if (previousConsumerId.empty()) {
        throw std::runtime_error(
            "GraphCompiler: output placement transfer for control op '" + controlId +
            "' did not produce a consumer-side bridge op");
    }
}

CompiledLoopOutputPlacement validateLoopOutputPlacements(
    const LoopOp& loop,
    const DGraphChild& bodyChild,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    CompiledLoopOutputPlacement result;
    RegionOutputBindings declaredOutputs = collectOutputBindings(loop.ioType, loop.ioMap, loop.id);
    if (declaredOutputs.buffers.empty() && declaredOutputs.scalars.empty()) return result;

    // A constant trip count of zero would never enter the body, so the
    // declared outputs would never be materialised. Reject this at compile
    // time; scalar trip counts can still be zero at runtime and remain a
    // dynamic check inside CpuDevice::executeLoop.
    if (loop.kind == LoopKind::FixedCount && loop.tripCount &&
        loop.tripCount->kind() == LoopTripCount::Kind::Constant &&
        loop.tripCount->constantBits() == 0) {
        throw std::runtime_error(
            "GraphCompiler: loop '" + loop.id +
            "' has a constant trip count of 0 but declares output ports; "
            "zero-iteration loops cannot materialize outputs");
    }

    RegionExitProducers bodyOutputs = collectRegionExitProducers(*loop.body, bodyChild);
    for (const auto& declared : declaredOutputs.buffers) {
        auto producerIt = bodyOutputs.buffersByPort.find(declared.portName);
        if (producerIt == bodyOutputs.buffersByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: loop '" + loop.id + "' output buffer port '" +
                declared.portName + "' is not produced by its body");
        }
        const auto& producer = producerIt->second;
        if (producer.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: loop '" + loop.id + "' output buffer port '" +
                declared.portName + "' type mismatch: declared " + bufferTypeName(declared.type) +
                ", body produced " + bufferTypeName(producer.binding.type));
        }
        const std::string placementDevice = resolveOutputPlacement(
            devices, "loop", loop.id, "buffer", declared.portName, {producer.deviceId},
            loop.outputPlacement.buffers);
        result.buffers[scopedBufferKey(declared.scopeId, declared.tokenName)] = placementDevice;
        CompiledLoopBufferPublication publication;
        publication.portName = declared.portName;
        publication.parentTokenName = declared.tokenName;
        publication.parentScopeId = declared.scopeId;
        publication.sourceTokenName = producer.binding.tokenName;
        publication.sourceScopeId = producer.binding.scopeId;
        publication.sourceDeviceId = producer.deviceId;
        const size_t publicationIndex = result.bufferPublications.size();
        result.bufferPublications.push_back(std::move(publication));
        if (producer.deviceId != placementDevice) {
            result.bufferMaterializations.push_back(ControlBufferMaterialization{
                DGraphChildRole::LoopBody,
                publicationIndex,
                producer.binding,
                producer.opId,
                producer.deviceId,
                placementDevice});
        }
    }

    for (const auto& declared : declaredOutputs.scalars) {
        auto producerIt = bodyOutputs.scalarsByPort.find(declared.portName);
        if (producerIt == bodyOutputs.scalarsByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: loop '" + loop.id + "' output scalar port '" +
                declared.portName + "' is not produced by its body");
        }
        const auto& producer = producerIt->second;
        if (producer.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: loop '" + loop.id + "' output scalar port '" +
                declared.portName + "' type mismatch: declared " + scalarTypeName(declared.type) +
                ", body produced " + scalarTypeName(producer.binding.type));
        }
        result.scalars[scopedScalarKey(declared.scopeId, declared.tokenName)] = resolveOutputPlacement(
            devices, "loop", loop.id, "scalar", declared.portName, {producer.deviceId},
            loop.outputPlacement.scalars);
        CompiledLoopScalarPublication publication;
        publication.portName = declared.portName;
        publication.parentTokenName = declared.tokenName;
        publication.parentScopeId = declared.scopeId;
        publication.sourceTokenName = producer.binding.tokenName;
        publication.sourceScopeId = producer.binding.scopeId;
        publication.sourceDeviceId = producer.deviceId;
        result.scalarPublications.push_back(std::move(publication));
    }

    return result;
}

CompiledConditionalOutputPlacement validateConditionalOutputPlacements(
    const ConditionalOp& conditional,
    const DGraphChild& thenChild,
    const DGraphChild& elseChild,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    CompiledConditionalOutputPlacement result;
    RegionOutputBindings declaredOutputs =
        collectOutputBindings(conditional.ioType, conditional.ioMap, conditional.id);
    if (declaredOutputs.buffers.empty() && declaredOutputs.scalars.empty()) return result;

    RegionExitProducers thenOutputs =
        collectRegionExitProducers(*conditional.thenRegion, thenChild);
    RegionExitProducers elseOutputs =
        collectRegionExitProducers(*conditional.elseRegion, elseChild);

    for (const auto& declared : declaredOutputs.buffers) {
        auto thenIt = thenOutputs.buffersByPort.find(declared.portName);
        auto elseIt = elseOutputs.buffersByPort.find(declared.portName);
        if (thenIt == thenOutputs.buffersByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' output buffer port '" +
                declared.portName + "' is not produced by the then branch");
        }
        if (elseIt == elseOutputs.buffersByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' output buffer port '" +
                declared.portName + "' is not produced by the else branch");
        }
        if (thenIt->second.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' then output buffer port '" +
                declared.portName + "' type mismatch: declared " + bufferTypeName(declared.type) +
                ", branch produced " + bufferTypeName(thenIt->second.binding.type));
        }
        if (elseIt->second.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' else output buffer port '" +
                declared.portName + "' type mismatch: declared " + bufferTypeName(declared.type) +
                ", branch produced " + bufferTypeName(elseIt->second.binding.type));
        }
        const std::string placementDevice = resolveOutputPlacement(
            devices, "conditional", conditional.id, "buffer", declared.portName,
            {thenIt->second.deviceId, elseIt->second.deviceId},
            conditional.outputPlacement.buffers);
        result.buffers[scopedBufferKey(declared.scopeId, declared.tokenName)] = placementDevice;
        CompiledConditionalBufferPublication publication;
        publication.portName = declared.portName;
        publication.parentTokenName = declared.tokenName;
        publication.parentScopeId = declared.scopeId;
        publication.thenSourceTokenName = thenIt->second.binding.tokenName;
        publication.thenSourceScopeId = thenIt->second.binding.scopeId;
        publication.thenSourceDeviceId = thenIt->second.deviceId;
        publication.elseSourceTokenName = elseIt->second.binding.tokenName;
        publication.elseSourceScopeId = elseIt->second.binding.scopeId;
        publication.elseSourceDeviceId = elseIt->second.deviceId;
        const size_t publicationIndex = result.bufferPublications.size();
        result.bufferPublications.push_back(std::move(publication));
        if (thenIt->second.deviceId != placementDevice) {
            result.bufferMaterializations.push_back(ControlBufferMaterialization{
                DGraphChildRole::ConditionalThen,
                publicationIndex,
                thenIt->second.binding,
                thenIt->second.opId,
                thenIt->second.deviceId,
                placementDevice});
        }
        if (elseIt->second.deviceId != placementDevice) {
            result.bufferMaterializations.push_back(ControlBufferMaterialization{
                DGraphChildRole::ConditionalElse,
                publicationIndex,
                elseIt->second.binding,
                elseIt->second.opId,
                elseIt->second.deviceId,
                placementDevice});
        }
    }

    for (const auto& declared : declaredOutputs.scalars) {
        auto thenIt = thenOutputs.scalarsByPort.find(declared.portName);
        auto elseIt = elseOutputs.scalarsByPort.find(declared.portName);
        if (thenIt == thenOutputs.scalarsByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' output scalar port '" +
                declared.portName + "' is not produced by the then branch");
        }
        if (elseIt == elseOutputs.scalarsByPort.end()) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' output scalar port '" +
                declared.portName + "' is not produced by the else branch");
        }
        if (thenIt->second.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' then output scalar port '" +
                declared.portName + "' type mismatch: declared " + scalarTypeName(declared.type) +
                ", branch produced " + scalarTypeName(thenIt->second.binding.type));
        }
        if (elseIt->second.binding.type != declared.type) {
            throw std::runtime_error(
                "GraphCompiler: conditional '" + conditional.id + "' else output scalar port '" +
                declared.portName + "' type mismatch: declared " + scalarTypeName(declared.type) +
                ", branch produced " + scalarTypeName(elseIt->second.binding.type));
        }
        result.scalars[scopedScalarKey(declared.scopeId, declared.tokenName)] = resolveOutputPlacement(
            devices, "conditional", conditional.id, "scalar", declared.portName,
            {thenIt->second.deviceId, elseIt->second.deviceId},
            conditional.outputPlacement.scalars);
        CompiledConditionalScalarPublication publication;
        publication.portName = declared.portName;
        publication.parentTokenName = declared.tokenName;
        publication.parentScopeId = declared.scopeId;
        publication.thenSourceTokenName = thenIt->second.binding.tokenName;
        publication.thenSourceScopeId = thenIt->second.binding.scopeId;
        publication.thenSourceDeviceId = thenIt->second.deviceId;
        publication.elseSourceTokenName = elseIt->second.binding.tokenName;
        publication.elseSourceScopeId = elseIt->second.binding.scopeId;
        publication.elseSourceDeviceId = elseIt->second.deviceId;
        result.scalarPublications.push_back(std::move(publication));
    }

    return result;
}

std::string resolveKernelDevice(
    const KernelOp& node,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    if (!node.deviceHint.empty()) {
        if (devices.find(node.deviceHint) == devices.end()) {
            throw std::runtime_error(
                "GraphCompiler: deviceHint '" + node.deviceHint +
                "' for node '" + node.id + "' does not match any registered device");
        }
        return node.deviceHint;
    }

    for (const auto& [did, dev] : devices) {
        if (dev->type() == node.kernel.type) return did;
    }

    throw std::runtime_error(
        "GraphCompiler: no device of type matching kernel '" + node.kernel.name +
        "' for node '" + node.id + "'");
}

std::string resolveReprogramDevice(
    const ReprogramOp& node,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices) {
    if (!node.deviceHint.empty()) {
        auto it = devices.find(node.deviceHint);
        if (it == devices.end()) {
            throw std::runtime_error(
                "GraphCompiler: deviceHint '" + node.deviceHint +
                "' for reprogram node '" + node.id + "' does not match any registered device");
        }
        if (it->second->type() != DeviceType::FPGA) {
            throw std::runtime_error(
                "GraphCompiler: reprogram node '" + node.id +
                "' must target an FPGA device");
        }
        return node.deviceHint;
    }

    for (const auto& [did, dev] : devices) {
        if (dev->type() == DeviceType::FPGA) return did;
    }

    throw std::runtime_error(
        "GraphCompiler: no FPGA device registered for reprogram node '" + node.id + "'");
}

DGraphChild makeDGraphChild(std::string parentNodeId,
                            DGraphChildRole role,
                            std::vector<DGraph> dgraphs) {
    DGraphChild child;
    child.parentNodeId = std::move(parentNodeId);
    child.role = role;
    child.dgraphs.reserve(dgraphs.size());
    for (auto& dg : dgraphs) {
        child.dgraphs.push_back(std::make_shared<DGraph>(std::move(dg)));
    }
    return child;
}

CompiledNode makeCompiledRegionNode(const RegionOp& op,
                                    const std::string& deviceId,
                                    const CompiledLoopOutputPlacement* loopOutputPlacement,
                                    const CompiledConditionalOutputPlacement* condOutputPlacement) {
    if (const auto* kernel = std::get_if<KernelOp>(&op)) {
        return CompiledKernelNode{
            kernel->id, kernel->kernel, deviceId, kernel->ioMap, {}};
    }
    if (const auto* reprogram = std::get_if<ReprogramOp>(&op)) {
        CompiledReprogramNode node;
        node.id = reprogram->id;
        node.deviceId = deviceId;
        node.imageId = reprogram->imageId;
        node.pdiPath = reprogram->pdiPath;
        node.timeoutCycles = reprogram->timeoutCycles;
        return node;
    }
    if (const auto* boundary = std::get_if<SubgraphBoundaryOp>(&op)) {
        const auto side = (boundary->side == BoundarySide::Start)
            ? CompiledBoundaryNode::Side::Start
            : CompiledBoundaryNode::Side::End;
        CompiledBoundaryNode node{boundary->id, deviceId, side};
        node.scalarCopies.reserve(boundary->scalarMappings.size());
        for (const auto& mapping : boundary->scalarMappings) {
            node.scalarCopies.push_back(CompiledScalarBoundaryCopy{
                mapping.source.varName(),
                mapping.source.scopeId(),
                mapping.target.varName(),
                mapping.target.scopeId()});
        }
        node.bufferCopies.reserve(boundary->bufferMappings.size());
        for (const auto& mapping : boundary->bufferMappings) {
            node.bufferCopies.push_back(CompiledBufferBoundaryCopy{
                mapping.source.name(),
                mapping.source.scopeId(),
                mapping.target.name(),
                mapping.target.scopeId()});
        }
        return node;
    }
    if (const auto* loop = std::get_if<LoopOp>(&op)) {
        CompiledLoopNode node;
        node.id = loop->id;
        node.deviceId = deviceId;
        node.loopKind = compiledLoopKind(loop->kind);
        node.tripCount = loop->tripCount;
        node.condition = loop->condition;
        if (loopOutputPlacement) {
            node.outputBufferPlacements = loopOutputPlacement->buffers;
            node.outputScalarPlacements = loopOutputPlacement->scalars;
            node.outputBufferPublications = loopOutputPlacement->bufferPublications;
            node.outputScalarPublications = loopOutputPlacement->scalarPublications;
        }
        return node;
    }
    if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
        CompiledConditionalNode node;
        node.id = cond->id;
        node.deviceId = deviceId;
        node.condition = cond->condition;
        if (condOutputPlacement) {
            node.outputBufferPlacements = condOutputPlacement->buffers;
            node.outputScalarPlacements = condOutputPlacement->scalars;
            node.outputBufferPublications = condOutputPlacement->bufferPublications;
            node.outputScalarPublications = condOutputPlacement->scalarPublications;
        }
        return node;
    }
    throw std::runtime_error(
        "GraphCompiler: makeCompiledRegionNode encountered an unsupported region op");
}

/**
 * Splits an authored GraphRegion into one DGraph per device.
 *
 * compileRegion() runs a fixed pipeline of phases (validate -> resolve
 * placements -> assign devices -> materialise bridges -> populate dependsOn
 * -> assemble); each phase mutates a per-call RegionCompilation state struct
 * but does not touch the compiler's own members. The compiler is therefore
 * trivially reentrant for nested regions: every recursive compileRegion()
 * call gets a fresh RegionCompilation.
 *
 * Note on output stability: synthesised bridge node ids and the resulting
 * DGraph ordering are stable for a given input as long as the topological
 * order is. Changes to producer-map iteration order, topo tiebreaking, or
 * the bridge id counter scheme will renumber bridges and may shift any test
 * fixture that matches by id.
 */
class RegionCompiler {
   public:
    RegionCompiler(const std::map<std::string, std::shared_ptr<IDevice>>& devices,
                   const GraphCompiler::BridgeFor& bridgeFor,
                   std::shared_ptr<std::map<std::string, uint64_t>> scalarValues)
        : devices_(devices), bridgeFor_(bridgeFor), scalarValues_(std::move(scalarValues)) {}

    std::vector<DGraph> compileRegion(const GraphRegion& region) {
        RegionCompilation rc;
        rc.ops = collectRegionOps(region);
        if (rc.ops.empty()) return {};

        rc.cpuDevice = findSingletonCpuDevice(devices_);
        indexOps(rc);
        compileChildRegions(rc);
        validateOpsAndPortBindings(rc);
        validateFpgaImageSafety(rc.ops);
        validateProvenance(region, rc);
        resolveControlOutputPlacements(rc);
        materializeControlOutputBridges(rc);
        assignDevices(rc);
        buildPerDeviceCompiledNodes(rc);
        insertCrossDeviceBridges(rc);
        insertAfterOpsBarriers(rc);
        populateDependsOn(rc);
        return assembleDGraphs(rc);
    }

   private:
    struct DeviceInsertions {
        std::map<size_t, std::vector<CompiledBridgeOpNode>> beforeNode;
        std::map<size_t, std::vector<CompiledBridgeOpNode>> afterNode;
        std::vector<CompiledBridgeOpNode>                    trailing;
    };

    /// Per-call state for a single compileRegion() invocation. Lives on the
    /// stack so recursive compilations do not interfere.
    struct RegionCompilation {
        std::vector<const RegionOp*> ops;
        IDevice* cpuDevice = nullptr;
        std::map<std::string, const RegionOp*>          opById;
        std::set<std::string>                            opIds;
        std::map<std::string, std::vector<DGraphChild>> childrenByControlId;
        std::map<std::string, std::string>              scalarProducerMap;
        std::map<std::string, std::string>              bufferProducerMap;
        std::map<std::pair<std::string, std::string>, std::string>
            loopCarriedInitialScalarProducers;
        std::map<std::pair<std::string, std::string>, std::string>
            loopCarriedInitialBufferProducers;
        std::map<std::string, CompiledLoopOutputPlacement>        loopOutputPlacements;
        std::map<std::string, CompiledConditionalOutputPlacement> conditionalOutputPlacements;
        uint32_t                                                  controlOutputBridgeCounter = 0;
        std::vector<std::string>                              sortedIds;
        std::map<std::string, std::string>                    nodeDevice;
        std::map<std::string, std::vector<CompiledNode>>      nodesByDevice;
        std::map<std::string, DeviceInsertions>               insertions;
        std::map<std::pair<std::string, std::string>, std::string> remoteConsumerBridgeIds;
        uint32_t bridgeCounter = 0;
    };

    // ---- Phases ---------------------------------------------------------

    /// Index every op by id so later phases can resolve references in O(log n).
    void indexOps(RegionCompilation& rc) const {
        for (const RegionOp* opPtr : rc.ops) {
            const std::string& opId = regionOpId(*opPtr);
            rc.opById[opId] = opPtr;
            rc.opIds.insert(opId);
        }
    }

    /// Recurse into loop bodies / conditional branches, attaching their
    /// compiled per-device DGraphs as DGraphChild entries on the parent
    /// control op's id.
    void compileChildRegions(RegionCompilation& rc) {
        for (const RegionOp* opPtr : rc.ops) {
            const RegionOp& op = *opPtr;
            const std::string& opId = regionOpId(op);
            if (const auto* loop = std::get_if<LoopOp>(&op)) {
                rc.childrenByControlId[opId].push_back(
                    makeDGraphChild(opId, DGraphChildRole::LoopBody,
                                    compileRegion(*loop->body)));
            } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
                rc.childrenByControlId[opId].push_back(
                    makeDGraphChild(opId, DGraphChildRole::ConditionalThen,
                                    compileRegion(*cond->thenRegion)));
                rc.childrenByControlId[opId].push_back(
                    makeDGraphChild(opId, DGraphChildRole::ConditionalElse,
                                    compileRegion(*cond->elseRegion)));
            }
        }
    }

    /// Verify per-op port bindings against the kernel descriptor and
    /// enforce the current "global scalars are CPU-only" backend
    /// restriction.
    void validateOpsAndPortBindings(const RegionCompilation& rc) const {
        for (const RegionOp* opPtr : rc.ops) {
            const RegionOp& op = *opPtr;
            validateDeclaredRegionPorts(op, rc.opIds);
            const auto* kernel = std::get_if<KernelOp>(&op);
            if (!kernel) continue;
            if (kernel->kernel.type != DeviceType::CPU) {
                for (const auto& [portName, scalar] : kernel->ioMap.scalars()) {
                    (void)portName;
                    if (!scalar.isConstant()) {
                        throw std::runtime_error(
                            "GraphCompiler: global scalar bindings are currently supported only on CPU kernels");
                    }
                }
                if (!kernel->kernel.ioType.outputScalars.empty()) {
                    throw std::runtime_error(
                        "GraphCompiler: output scalar ports are currently supported only on CPU kernels");
                }
            }
        }
    }

    /// Build scalar / buffer producer maps and verify that every consumed
    /// token has a producer in scope (or is a graph-global).
    void validateProvenance(const GraphRegion& region, RegionCompilation& rc) const {
        ProducerMapInfo scalarProducers = buildRegionProducerMapInfo(rc.ops, /*scalar=*/true);
        ProducerMapInfo bufferProducers = buildRegionProducerMapInfo(rc.ops, /*scalar=*/false);
        rc.scalarProducerMap = std::move(scalarProducers.producers);
        rc.bufferProducerMap = std::move(bufferProducers.producers);
        rc.loopCarriedInitialScalarProducers =
            std::move(scalarProducers.loopCarriedInitialProducers);
        rc.loopCarriedInitialBufferProducers =
            std::move(bufferProducers.loopCarriedInitialProducers);
        validateRegionScalarProvenance(region, rc.ops, rc.scalarProducerMap);
        validateRegionBufferProvenance(region, rc.ops, rc.bufferProducerMap);
    }

    /// Resolve the device placement of every loop/conditional output,
    /// using the already-compiled child DGraphs as hints.
    void resolveControlOutputPlacements(RegionCompilation& rc) const {
        for (const RegionOp* opPtr : rc.ops) {
            const RegionOp& op = *opPtr;
            const std::string& opId = regionOpId(op);
            auto childIt = rc.childrenByControlId.find(opId);
            if (const auto* loop = std::get_if<LoopOp>(&op)) {
                const DGraphChild& bodyChild = requireChildDGraphs(
                    childIt->second, opId, DGraphChildRole::LoopBody);
                rc.loopOutputPlacements[opId] = validateLoopOutputPlacements(
                    *loop, bodyChild, devices_);
            } else if (const auto* cond = std::get_if<ConditionalOp>(&op)) {
                const DGraphChild& thenChild = requireChildDGraphs(
                    childIt->second, opId, DGraphChildRole::ConditionalThen);
                const DGraphChild& elseChild = requireChildDGraphs(
                    childIt->second, opId, DGraphChildRole::ConditionalElse);
                rc.conditionalOutputPlacements[opId] = validateConditionalOutputPlacements(
                    *cond, thenChild, elseChild, devices_);
            }
        }
    }

    /// Insert bridge ops *inside* the child DGraphs so loop / conditional
    /// outputs that need to land on a device different from where they were
    /// produced are physically transferred.
    void materializeControlOutputBridges(RegionCompilation& rc) {
        auto materializeFor = [&](const std::string& controlId,
                                  const std::vector<ControlBufferMaterialization>& materializations,
                                  auto retargetOne) {
            if (materializations.empty()) return;
            if (!rc.cpuDevice) {
                throw std::runtime_error(
                    "GraphCompiler: cross-device control-flow output buffer publication "
                    "requires a CPU device but none is registered");
            }
            auto childIt = rc.childrenByControlId.find(controlId);
            if (childIt == rc.childrenByControlId.end()) {
                throw std::runtime_error(
                    "GraphCompiler: control op '" + controlId +
                    "' is missing output-placement child DGraphs");
            }
            for (const auto& materialization : materializations) {
                DGraphChild& child = requireMutableChildDGraphs(
                    childIt->second, controlId, materialization.role);
                materializeControlOutputBufferTransfer(
                    child, controlId, materialization, devices_, bridgeFor_,
                    *rc.cpuDevice, scalarValues_, rc.controlOutputBridgeCounter);
                retargetOne(materialization);
            }
        };

        for (auto& [controlId, placement] : rc.loopOutputPlacements) {
            materializeFor(controlId, placement.bufferMaterializations,
                           [&](const ControlBufferMaterialization& m) {
                               retargetLoopBufferPublication(placement, m);
                           });
        }
        for (auto& [controlId, placement] : rc.conditionalOutputPlacements) {
            materializeFor(controlId, placement.bufferMaterializations,
                           [&](const ControlBufferMaterialization& m) {
                               retargetConditionalBufferPublication(placement, m);
                           });
        }
    }

    /// Topologically sort the region's ops and pin each one to a device
    /// (kernels via deviceHint, control / boundary ops to the singleton CPU).
    void assignDevices(RegionCompilation& rc) const {
        ProducerMapInfo bufferProducers;
        bufferProducers.producers = rc.bufferProducerMap;
        bufferProducers.loopCarriedInitialProducers = rc.loopCarriedInitialBufferProducers;
        ProducerMapInfo scalarProducers;
        scalarProducers.producers = rc.scalarProducerMap;
        scalarProducers.loopCarriedInitialProducers = rc.loopCarriedInitialScalarProducers;
        const auto adj = buildRegionAdjacency(rc.ops, bufferProducers, scalarProducers);
        rc.sortedIds = topoSortRegion(rc.ops, adj);

        for (const auto& id : rc.sortedIds) {
            const RegionOp& op = *rc.opById.at(id);
            if (const auto* kernel = std::get_if<KernelOp>(&op)) {
                rc.nodeDevice[id] = resolveKernelDevice(*kernel, devices_);
            } else if (const auto* reprogram = std::get_if<ReprogramOp>(&op)) {
                rc.nodeDevice[id] = resolveReprogramDevice(*reprogram, devices_);
            } else if (std::holds_alternative<LoopOp>(op) ||
                       std::holds_alternative<ConditionalOp>(op)) {
                if (!rc.cpuDevice) {
                    throw std::runtime_error(
                        "GraphCompiler: control op '" + id +
                        "' requires a CPU device for execution but none is registered");
                }
                rc.nodeDevice[id] = rc.cpuDevice->id();
            } else {
                if (!rc.cpuDevice) {
                    throw std::runtime_error(
                        "GraphCompiler: boundary op '" + id +
                        "' requires a CPU device but none is registered");
                }
                rc.nodeDevice[id] = rc.cpuDevice->id();
            }
        }
    }

    /// Lower each authored op into its CompiledNode form on the device it
    /// was assigned to, preserving topological order.
    void buildPerDeviceCompiledNodes(RegionCompilation& rc) const {
        for (const auto& [did, dev] : devices_) {
            (void)dev;
            rc.nodesByDevice[did];
        }
        for (const auto& id : rc.sortedIds) {
            const RegionOp& op = *rc.opById.at(id);
            const CompiledLoopOutputPlacement* loopPlacement = nullptr;
            const CompiledConditionalOutputPlacement* condPlacement = nullptr;
            if (auto loopIt = rc.loopOutputPlacements.find(id);
                loopIt != rc.loopOutputPlacements.end()) {
                loopPlacement = &loopIt->second;
            } else if (auto condIt = rc.conditionalOutputPlacements.find(id);
                       condIt != rc.conditionalOutputPlacements.end()) {
                condPlacement = &condIt->second;
            }
            rc.nodesByDevice[rc.nodeDevice.at(id)].push_back(
                makeCompiledRegionNode(op, rc.nodeDevice.at(id),
                                       loopPlacement, condPlacement));
        }
    }

    /// For each cross-device buffer dependency at an input port, route a
    /// (possibly CPU-bounced) chain of bridge ops, queuing them up next to
    /// their producer / consumer kernels.
    void insertCrossDeviceBridges(RegionCompilation& rc) {
        for (const auto& id : rc.sortedIds) {
            const RegionOp& op = *rc.opById.at(id);
            const std::string& consumerDevId = rc.nodeDevice.at(id);
            const IOMap& ioMap = regionOpIoMap(op);
            for (const auto& [port, buf] : ioMap.inputBuffers()) {
                (void)port;
                routeBufferTransferIfNeeded(rc, op, consumerDevId, buf);
            }
            for (const auto& rw : ioMap.rwBuffers()) {
                routeBufferTransferIfNeeded(rc, op, consumerDevId, rw.in);
            }
        }
    }

    /// For every cross-device afterOps edge, materialise a barrier-only
    /// bridge pair (CPU-bounced when no direct factory exists).
    void insertAfterOpsBarriers(RegionCompilation& rc) {
        for (auto& [did, nodes] : rc.nodesByDevice) {
            for (size_t i = 0; i < nodes.size(); ++i) {
                const std::string id = compiledNodeId(nodes[i]);
                auto sourceIt = rc.opById.find(id);
                if (sourceIt == rc.opById.end()) continue;
                const auto& afterOps = regionOpAfterOps(*sourceIt->second);
                for (const auto& a : afterOps) {
                    auto ndIt = rc.nodeDevice.find(a);
                    if (ndIt == rc.nodeDevice.end()) continue;
                    if (ndIt->second == did) continue;
                    materialiseAfterOpsBarrier(rc, ndIt->second, did, a, id);
                }
            }
        }
    }

    /// Final dependsOn pass: turn buffer / scalar producer relationships and
    /// same-device afterOps into edges on the corresponding compiled nodes.
    void populateDependsOn(RegionCompilation& rc) {
        // Derived side-effect ordering (reprogram drain, readers-before-mutator)
        // must become real runtime barriers, not just topo-sort hints, so the
        // async FPGA scheduler actually drains the old image before a reprogram.
        const auto sideEdges = computeSideEffectOrderingEdges(rc.ops);
        for (auto& [did, nodes] : rc.nodesByDevice) {
            const auto& devIns = rc.insertions[did];
            for (size_t i = 0; i < nodes.size(); ++i) {
                CompiledNode& node = nodes[i];
                auto sourceIt = rc.opById.find(compiledNodeId(node));
                if (sourceIt == rc.opById.end()) continue;
                const RegionOp& source = *sourceIt->second;
                std::set<std::string> seen;
                for (const auto& existing : compiledNodeDependsOn(node)) {
                    seen.insert(existing);
                }

                for (const auto& ref : consumedBufferRefs(source)) {
                    pushBufferDependency(rc, did, node, seen, ref);
                }
                for (const auto& scalarKey : consumedScalarKeys(source)) {
                    std::string producerNodeId;
                    auto carriedIt = rc.loopCarriedInitialScalarProducers.find(
                        {compiledNodeId(node), scalarKey});
                    if (carriedIt != rc.loopCarriedInitialScalarProducers.end()) {
                        producerNodeId = carriedIt->second;
                    } else if (auto producerIt = rc.scalarProducerMap.find(scalarKey);
                               producerIt != rc.scalarProducerMap.end()) {
                        producerNodeId = producerIt->second;
                    }
                    if (producerNodeId.empty()) continue;
                    const std::string& producerDeviceId = rc.nodeDevice.at(producerNodeId);
                    if (producerDeviceId != did) {
                        throw std::runtime_error(
                            "GraphCompiler: cross-device global scalar dependencies are not supported yet");
                    }
                    addDep(node, seen, producerNodeId);
                }

                auto bIt = devIns.beforeNode.find(i);
                if (bIt != devIns.beforeNode.end()) {
                    for (const auto& bop : bIt->second) addDep(node, seen, bop.id);
                }

                for (const auto& a : regionOpAfterOps(source)) {
                    auto ndIt = rc.nodeDevice.find(a);
                    if (ndIt == rc.nodeDevice.end()) {
                        throw std::runtime_error(
                            "GraphCompiler: op '" + compiledNodeId(node) +
                            "' references unknown afterOps id '" + a + "'");
                    }
                    if (ndIt->second == did) addDep(node, seen, a);
                }

                // Derived drain / readers-before-mutator predecessors. Same
                // device => a real barrier edge (this is what enforces the
                // old-image drain on the FPGA scheduler).
                if (auto seIt = sideEdges.find(compiledNodeId(node));
                    seIt != sideEdges.end()) {
                    for (const auto& pred : seIt->second) {
                        auto ndIt = rc.nodeDevice.find(pred);
                        if (ndIt != rc.nodeDevice.end() && ndIt->second == did) {
                            addDep(node, seen, pred);
                        }
                    }
                }
            }
        }
    }

    /// Stitch per-device node lists, queued bridge insertions, and child
    /// DGraphs into the final per-device DGraph vector.
    std::vector<DGraph> assembleDGraphs(RegionCompilation& rc) const {
        std::vector<DGraph> result;
        for (auto& [did, nodes] : rc.nodesByDevice) {
            const auto& ins = rc.insertions[did];
            if (nodes.empty() && ins.trailing.empty()) continue;

            DGraph dg;
            dg.deviceId = did;
            dg.device = devices_.at(did);
            dg.scalarValues = scalarValues_;

            for (size_t i = 0; i < nodes.size(); ++i) {
                auto bIt = ins.beforeNode.find(i);
                if (bIt != ins.beforeNode.end()) {
                    for (const auto& op : bIt->second) dg.nodes.emplace_back(op);
                }
                dg.nodes.emplace_back(std::move(nodes[i]));
                auto aIt = ins.afterNode.find(i);
                if (aIt != ins.afterNode.end()) {
                    for (const auto& op : aIt->second) dg.nodes.emplace_back(op);
                }
            }
            for (const auto& op : ins.trailing) dg.nodes.emplace_back(op);

            for (const CompiledNode& node : dg.nodes) {
                auto childIt = rc.childrenByControlId.find(compiledNodeId(node));
                if (childIt == rc.childrenByControlId.end()) continue;
                dg.childDGraphs.insert(dg.childDGraphs.end(),
                                       childIt->second.begin(), childIt->second.end());
            }

            result.push_back(std::move(dg));
        }

        return result;
    }

    // ---- Helpers (extracted from former in-place lambdas) ---------------

    /// Find the position of the compiled node @p nodeId on device @p did,
    /// or a sentinel if it was already filtered out / placed via trailing.
    size_t nodeIndex(const RegionCompilation& rc, const std::string& did,
                     const std::string& nodeId) const {
        auto it = rc.nodesByDevice.find(did);
        if (it == rc.nodesByDevice.end()) return std::numeric_limits<size_t>::max();
        for (size_t i = 0; i < it->second.size(); ++i) {
            if (compiledNodeId(it->second[i]) == nodeId) return i;
        }
        return std::numeric_limits<size_t>::max();
    }

    /// Materialise a producer/consumer bridge pair next to the kernels that
    /// own them, mint stable producer/consumer ids using @p kind and
    /// rc.bridgeCounter, and attach optional extra producer-side dep.
    std::pair<std::string, std::string> materialiseLeg(
        RegionCompilation& rc, const RoutedLeg& leg,
        const std::string& extraProducerDep,
        BridgeIds::Kind kind = BridgeIds::Kind::DataBridge) {
        auto [pOpId, cOpId] = BridgeIds::mint(kind, rc.bridgeCounter);
        ++rc.bridgeCounter;

        CompiledBridgeOpNode pNode{
            pOpId, leg.srcDeviceId, leg.pair.op,
            leg.pair.producerAction,
            CompiledBridgeOpNode::Side::Producer,
            leg.producerKernelId};
        pNode.dependsOn.push_back(leg.producerKernelId);
        if (!extraProducerDep.empty()) {
            pNode.dependsOn.push_back(extraProducerDep);
        }

        CompiledBridgeOpNode cNode{
            cOpId, leg.dstDeviceId, leg.pair.op,
            leg.pair.consumerAction,
            CompiledBridgeOpNode::Side::Consumer,
            leg.consumerKernelId};
        cNode.tryReady = leg.pair.consumerTryReady;

        size_t pIdx = nodeIndex(rc, leg.srcDeviceId, leg.producerKernelId);
        if (pIdx == std::numeric_limits<size_t>::max()) {
            rc.insertions[leg.srcDeviceId].trailing.push_back(std::move(pNode));
        } else {
            rc.insertions[leg.srcDeviceId].afterNode[pIdx].push_back(std::move(pNode));
        }

        size_t cIdx = nodeIndex(rc, leg.dstDeviceId, leg.consumerKernelId);
        if (cIdx == std::numeric_limits<size_t>::max()) {
            rc.insertions[leg.dstDeviceId].trailing.push_back(std::move(cNode));
        } else {
            rc.insertions[leg.dstDeviceId].beforeNode[cIdx].push_back(std::move(cNode));
        }

        return {pOpId, cOpId};
    }

    /// Inspect a single consumed buffer; if the producer lives on a different
    /// device, route a bridge chain (CPU-bounced when needed).
    void routeBufferTransferIfNeeded(RegionCompilation& rc,
                                     const RegionOp& consumerOp,
                                     const std::string& consumerDevId,
                                     const GraphBuffer& bufObj) {
        const std::string bufKey = scopedBufferKey(bufObj.scopeId(), bufObj.name());
        std::string producerNodeId;
        auto carriedIt = rc.loopCarriedInitialBufferProducers.find(
            {regionOpId(consumerOp), bufKey});
        if (carriedIt != rc.loopCarriedInitialBufferProducers.end()) {
            producerNodeId = carriedIt->second;
        } else if (auto prodIt = rc.bufferProducerMap.find(bufKey);
                   prodIt != rc.bufferProducerMap.end()) {
            producerNodeId = prodIt->second;
        }
        if (producerNodeId.empty()) return;
        const std::string& producerDevId = rc.nodeDevice.at(producerNodeId);
        if (producerDevId == consumerDevId) return;

        const RegionOp& producerOp = *rc.opById.at(producerNodeId);
        // Boundary ops are CPU-pinned and move data through their compiled
        // CompiledBoundaryNode actions, not through bridges; refuse to route
        // bridges that involve them on either side. Kernel and control op
        // (loop / conditional) producers are both fine: control op outputs
        // are already published on the resolved placement device after Phase
        // 7C materialisation, so a parent-level cross-device consumer is
        // treated like any other cross-device kernel-to-kernel transfer.
        const bool producerOk = std::holds_alternative<KernelOp>(producerOp) ||
                                std::holds_alternative<LoopOp>(producerOp) ||
                                std::holds_alternative<ConditionalOp>(producerOp);
        if (!producerOk) {
            throw std::runtime_error(
                "GraphCompiler: cross-device transfer from boundary op '" +
                producerNodeId + "' on '" + producerDevId +
                "' to '" + regionOpId(consumerOp) + "' on '" + consumerDevId +
                "' is not supported; boundary data movement happens through "
                "CompiledBoundaryNode actions, not bridges");
        }
        if (std::holds_alternative<SubgraphBoundaryOp>(consumerOp)) {
            throw std::runtime_error(
                "GraphCompiler: cross-device transfer to boundary op '" +
                regionOpId(consumerOp) + "' on '" + consumerDevId +
                "' from '" + producerNodeId + "' on '" + producerDevId +
                "' is not supported; boundary data movement happens through "
                "CompiledBoundaryNode actions, not bridges");
        }

        auto edgeKey = std::make_pair(bufKey, consumerDevId);
        if (rc.remoteConsumerBridgeIds.count(edgeKey)) return;

        if (!rc.cpuDevice) {
            throw std::runtime_error(
                "GraphCompiler: cross-device transfer of buffer '" + bufObj.name() +
                "' requires a CPU device but none is registered");
        }

        auto legs = BridgeRouter::routeTransfer(
            *devices_.at(producerDevId),
            *devices_.at(consumerDevId),
            bufObj, 0, bridgeFor_, *rc.cpuDevice,
            producerNodeId, regionOpId(consumerOp));
        std::string prevConsumerId;
        for (const auto& leg : legs) {
            auto idsPair = materialiseLeg(rc, leg, prevConsumerId);
            prevConsumerId = idsPair.second;
        }
        if (prevConsumerId.empty()) {
            throw std::runtime_error(
                "GraphCompiler: remote transfer of buffer '" + bufObj.name() +
                "' to device '" + consumerDevId +
                "' did not produce a consumer-side bridge op");
        }
        rc.remoteConsumerBridgeIds[edgeKey] = prevConsumerId;
    }

    /// Materialise a barrier-only bridge pair for an afterOps edge from
    /// `srcDeviceId.afterNodeId` to `dstDeviceId.toNodeId`. Routes through
    /// the CPU bounce when no direct factory exists.
    void materialiseAfterOpsBarrier(RegionCompilation& rc,
                                    const std::string& srcDeviceId,
                                    const std::string& dstDeviceId,
                                    const std::string& afterNodeId,
                                    const std::string& toNodeId) {
        IDevice& srcDev = *devices_.at(srcDeviceId);
        IDevice& dstDev = *devices_.at(dstDeviceId);

        if (IBridge* directBr = bridgeFor_(srcDev.id(), dstDev.id())) {
            auto pair = directBr->makeBarrier(srcDev, dstDev, afterNodeId, toNodeId);
            RoutedLeg leg{srcDev.id(), dstDev.id(), afterNodeId, toNodeId, std::move(pair)};
            materialiseLeg(rc, leg, "", BridgeIds::Kind::Barrier);
            return;
        }

        if (!rc.cpuDevice) {
            throw std::runtime_error(
                "GraphCompiler: cross-device afterOps from '" + afterNodeId +
                "' to '" + toNodeId + "' requires bouncing a barrier "
                "through cpu but no CPU device is registered");
        }
        IBridge* srcCpu = bridgeFor_(srcDev.id(), rc.cpuDevice->id());
        IBridge* cpuDst = bridgeFor_(rc.cpuDevice->id(), dstDev.id());
        if (!srcCpu || !cpuDst) {
            throw std::runtime_error(
                "GraphCompiler: cross-device afterOps from '" + afterNodeId +
                "' to '" + toNodeId + "' is missing a CPU bounce factory");
        }
        auto pair1 = srcCpu->makeBarrier(srcDev, *rc.cpuDevice, afterNodeId, toNodeId);
        RoutedLeg leg1{srcDev.id(), rc.cpuDevice->id(), afterNodeId, toNodeId, std::move(pair1)};
        auto ids1 = materialiseLeg(rc, leg1, "", BridgeIds::Kind::Barrier);

        auto pair2 = cpuDst->makeBarrier(*rc.cpuDevice, dstDev, afterNodeId, toNodeId);
        RoutedLeg leg2{rc.cpuDevice->id(), dstDev.id(), afterNodeId, toNodeId, std::move(pair2)};
        materialiseLeg(rc, leg2, ids1.second, BridgeIds::Kind::Barrier);
    }

    /// Append @p depId to @p node's dependsOn list, deduplicating against
    /// previously-seen deps and refusing to add a self-edge.
    static void addDep(CompiledNode& node, std::set<std::string>& seen,
                       const std::string& depId) {
        if (depId.empty() || depId == compiledNodeId(node)) return;
        if (seen.insert(depId).second) {
            mutableCompiledNodeDependsOn(node).push_back(depId);
        }
    }

    /// Wire a buffer-consumer dep: same-device producer becomes a direct
    /// edge; cross-device producer becomes an edge to the consumer-side
    /// bridge op materialised earlier.
    void pushBufferDependency(const RegionCompilation& rc,
                              const std::string& did,
                              CompiledNode& node,
                              std::set<std::string>& seen,
                              const ConsumedBufferRef& ref) const {
        auto pit = rc.bufferProducerMap.find(ref.key);
        std::string producerNodeId;
        auto carriedIt = rc.loopCarriedInitialBufferProducers.find({compiledNodeId(node), ref.key});
        if (carriedIt != rc.loopCarriedInitialBufferProducers.end()) {
            producerNodeId = carriedIt->second;
        } else if (pit != rc.bufferProducerMap.end()) {
            producerNodeId = pit->second;
        }
        if (producerNodeId.empty()) return;
        if (rc.nodeDevice.at(producerNodeId) == did) {
            addDep(node, seen, producerNodeId);
            return;
        }
        auto bridgeIt = rc.remoteConsumerBridgeIds.find({ref.key, did});
        if (bridgeIt == rc.remoteConsumerBridgeIds.end()) {
            throw std::runtime_error(
                "GraphCompiler: missing consumer-side bridge for remote buffer '" +
                ref.name + "' on device '" + did + "'");
        }
        addDep(node, seen, bridgeIt->second);
    }

    const std::map<std::string, std::shared_ptr<IDevice>>& devices_;
    const GraphCompiler::BridgeFor&                        bridgeFor_;
    std::shared_ptr<std::map<std::string, uint64_t>>       scalarValues_;
};

}  // namespace

// ---------------------------------------------------------------------------
// validateRegionScopes
// ---------------------------------------------------------------------------

void GraphCompiler::validateRegionScopes(const GraphRegion& region) const {
    const uint64_t regionScope = region.scopeId();
    const std::set<uint64_t> localScope{regionScope};

    for (const RegionOp& op : region.ops()) {
        std::visit(
            [&](const auto& concrete) {
                using T = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<T, KernelOp>) {
                    validateIoMapScopes(concrete.id, concrete.ioMap, regionScope, localScope);
                } else if constexpr (std::is_same_v<T, SubgraphBoundaryOp>) {
                    validateIoMapScopes(concrete.id, concrete.ioMap, regionScope,
                                        {concrete.parentScopeId, concrete.localScopeId});
                    validateScalarBoundaryMappings(concrete);
                    validateBufferBoundaryMappings(concrete);
                } else if constexpr (std::is_same_v<T, LoopOp>) {
                    validateIoMapScopes(concrete.id, concrete.ioMap, regionScope, localScope);
                    if (concrete.tripCount) {
                        validateTripCountScope(concrete.id, *concrete.tripCount, regionScope);
                    }
                    if (concrete.condition) {
                        validateConditionScopes(concrete.id, *concrete.condition, regionScope);
                    }
                    validateChildRegion(concrete.id, concrete.body, regionScope);
                    validateRegionScopes(*concrete.body);
                } else if constexpr (std::is_same_v<T, ConditionalOp>) {
                    validateIoMapScopes(concrete.id, concrete.ioMap, regionScope, localScope);
                    validateConditionScopes(concrete.id, concrete.condition, regionScope);
                    validateChildRegion(concrete.id, concrete.thenRegion, regionScope);
                    validateChildRegion(concrete.id, concrete.elseRegion, regionScope);
                    validateRegionScopes(*concrete.thenRegion);
                    validateRegionScopes(*concrete.elseRegion);
                }
            },
            op);
    }
}

std::vector<DGraph> GraphCompiler::compile(
    const GraphRegion& rootRegion,
    const std::map<std::string, std::shared_ptr<IDevice>>& devices,
    const std::map<std::pair<DeviceType, DeviceType>, BridgeFactory>& bridgeFactories,
    const BridgeFor&                                       bridgeFor,
    const std::shared_ptr<std::map<std::string, uint64_t>>& scalarValues) {
    if (rootRegion.ops().empty()) {
        throw std::runtime_error("GraphCompiler::compile: graph has no ops");
    }
    if (devices.empty()) {
        throw std::runtime_error("GraphCompiler::compile: no devices registered");
    }
    validateBridgeFactories(devices, bridgeFactories);
    validateRegionScopes(rootRegion);
    validateRootScopeScalarReferences(rootRegion);
    validateRootScopeBufferReferences(rootRegion);
    RegionCompiler compiler(devices, bridgeFor, scalarValues);
    return compiler.compileRegion(rootRegion);
}

}  // namespace vrt::graph
