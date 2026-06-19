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
 * @file cpu_device.cpp
 * @brief CpuDevice implementation — naive single-core CPU device.
 */

#include <vrt/graph/device/cpu_device.hpp>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <deque>
#include <exception>
#include <limits>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

namespace vrt::graph {

namespace {

constexpr std::chrono::seconds kBridgeWaitTimeout{35};

template <typename T>
T bitsAs(uint64_t bits) {
    T value{};
    std::memcpy(&value, &bits, sizeof(T));
    return value;
}

int64_t signedScalarValue(ScalarType type, uint64_t bits) {
    switch (type) {
        case ScalarType::I8:  return bitsAs<int8_t>(bits);
        case ScalarType::I16: return bitsAs<int16_t>(bits);
        case ScalarType::I32: return bitsAs<int32_t>(bits);
        case ScalarType::I64: return bitsAs<int64_t>(bits);
        default:
            throw std::invalid_argument("CpuDevice: scalar type is not signed integer");
    }
}

uint64_t unsignedScalarValue(ScalarType type, uint64_t bits) {
    switch (type) {
        case ScalarType::U8:  return bitsAs<uint8_t>(bits);
        case ScalarType::U16: return bitsAs<uint16_t>(bits);
        case ScalarType::U32: return bitsAs<uint32_t>(bits);
        case ScalarType::U64: return bitsAs<uint64_t>(bits);
        default:
            throw std::invalid_argument("CpuDevice: scalar type is not unsigned integer");
    }
}

long double floatingScalarValue(ScalarType type, uint64_t bits) {
    switch (type) {
        case ScalarType::F32: return bitsAs<float>(bits);
        case ScalarType::F64: return bitsAs<double>(bits);
        default:
            throw std::invalid_argument("CpuDevice: scalar type is not floating point");
    }
}

template <typename T>
bool compareValues(CompareOp op, T lhs, T rhs) {
    switch (op) {
        case CompareOp::LT: return lhs < rhs;
        case CompareOp::LE: return lhs <= rhs;
        case CompareOp::EQ: return lhs == rhs;
        case CompareOp::GT: return lhs > rhs;
        case CompareOp::GE: return lhs >= rhs;
        case CompareOp::NE: return lhs != rhs;
        default:
            throw std::runtime_error("CpuDevice: unsupported comparison operator");
    }
}

}  // namespace

class CpuDevicePlan : public IDevicePlan {
   public:
    CpuDevicePlan(CpuDevice& device, const DGraph& dg)
        : device_(device),
          scalarValues_(dg.scalarValues
                            ? dg.scalarValues
                            : std::make_shared<std::map<std::string, uint64_t>>()) {
        runtime_.reserve(dg.nodes.size());
        idToIdx_.reserve(dg.nodes.size());

        // First pass: build per-node runtime records, keyed by id.
        for (const CompiledNode& node : dg.nodes) {
            NodeRuntime rt;
            std::visit(
                [&](const auto& n) {
                    using T = std::decay_t<decltype(n)>;
                    rt.id = n.id;
                    if constexpr (std::is_same_v<T, CompiledKernelNode>) {
                        rt.kind = NodeKind::Kernel;
                        rt.kernel = n;
                    } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
                        rt.kind = (n.side == CompiledBridgeOpNode::Side::Producer)
                                      ? NodeKind::ProducerOp
                                      : NodeKind::ConsumerOp;
                        rt.tryReady = n.tryReady;
                        rt.action = n.action;
                    } else if constexpr (std::is_same_v<T, CompiledBoundaryNode>) {
                        rt.kind = NodeKind::Boundary;
                        rt.boundary = n;
                    } else if constexpr (std::is_same_v<T, CompiledLoopNode>) {
                        rt.kind = NodeKind::Loop;
                        rt.loop = n;
                    } else if constexpr (std::is_same_v<T, CompiledConditionalNode>) {
                        rt.kind = NodeKind::Conditional;
                        rt.conditional = n;
                    } else if constexpr (std::is_same_v<T, CompiledReprogramNode>) {
                        throw std::runtime_error(
                            "CpuDevice: reprogram nodes must execute on an FPGA device");
                    } else {
                        static_assert(sizeof(T) == 0, "Unhandled compiled node type");
                    }
                },
                node);
            idToIdx_[rt.id] = runtime_.size();
            runtime_.push_back(std::move(rt));
        }

        // Second pass: convert dependsOn ids to indices, build successors and
        // the immutable initial unmet counts used to seed each launch.
        // dependsOn may legitimately reference ids from other DGraphs. Ids not
        // local to this DGraph are ignored; bridge readiness handles cross-device
        // synchronization.
        for (size_t i = 0; i < dg.nodes.size(); ++i) {
            const auto& deps = compiledNodeDependsOn(dg.nodes[i]);
            for (const std::string& depId : deps) {
                auto it = idToIdx_.find(depId);
                if (it == idToIdx_.end()) continue;
                runtime_[it->second].successors.push_back(i);
                ++runtime_[i].initialUnmet;
            }
        }

        for (const DGraphChild& child : dg.childDGraphs) {
            auto& plans = childPlans_[child.parentNodeId][child.role];
            plans.reserve(child.dgraphs.size());
            for (const auto& childDGraph : child.dgraphs) {
                if (!childDGraph) continue;
                if (!childDGraph->device) {
                    throw std::runtime_error(
                        "CpuDevice: control-flow child DGraph is missing its target device");
                }
                auto plan = childDGraph->device->compilePlan(*childDGraph);
                if (!plan) {
                    throw std::runtime_error(
                        "CpuDevice: target device returned a null control-flow child plan");
                }
                plans.push_back(std::move(plan));
            }
        }
    }

    ~CpuDevicePlan() override {
        try {
            wait();
        } catch (...) {
        }
    }

    void launch() override {
        wait();
        workerException_ = nullptr;
        worker_ = std::thread([this] {
            try {
                runOnce();
            } catch (...) {
                workerException_ = std::current_exception();
            }
        });
    }

    void wait() override {
        if (worker_.joinable()) worker_.join();
        if (workerException_) {
            std::exception_ptr ex = workerException_;
            workerException_ = nullptr;
            std::rethrow_exception(ex);
        }
    }

   private:
    enum class NodeKind { Kernel, ProducerOp, ConsumerOp, Boundary, Loop, Conditional };

    struct NodeRuntime {
        std::string id;
        NodeKind kind = NodeKind::Boundary;
        size_t initialUnmet = 0;
        std::vector<size_t> successors;
        CompiledKernelNode kernel;
        CompiledBoundaryNode boundary;
        CompiledLoopNode loop;
        CompiledConditionalNode conditional;
        std::function<bool()> tryReady;
        std::function<void()> action;
    };

    void runOnce() {
        std::vector<size_t> unmetCounts;
        unmetCounts.reserve(runtime_.size());
        for (const auto& rt : runtime_) {
            unmetCounts.push_back(rt.initialUnmet);
        }

        std::deque<size_t> readyKP;
        std::vector<size_t> pendingCons;

        auto promote = [&](size_t idx) {
            if (runtime_[idx].kind == NodeKind::ConsumerOp) {
                pendingCons.push_back(idx);
            } else {
                readyKP.push_back(idx);
            }
        };

        for (size_t i = 0; i < runtime_.size(); ++i) {
            if (unmetCounts[i] == 0) promote(i);
        }

        auto runIndex = [&](size_t idx) {
            NodeRuntime& rt = runtime_[idx];
            switch (rt.kind) {
                case NodeKind::Kernel:
                    executeKernel(rt.kernel);
                    break;
                case NodeKind::ProducerOp:
                case NodeKind::ConsumerOp:
                    rt.action();
                    break;
                case NodeKind::Boundary:
                    executeBoundary(rt.boundary);
                    break;
                case NodeKind::Loop:
                    executeLoop(rt.loop);
                    break;
                case NodeKind::Conditional:
                    executeConditional(rt.conditional);
                    break;
            }
            for (size_t successor : rt.successors) {
                if (--unmetCounts[successor] == 0) promote(successor);
            }
        };

        size_t rrCursor = 0;
        auto idleSince = std::chrono::steady_clock::now();
        for (;;) {
            while (!readyKP.empty()) {
                size_t idx = readyKP.front();
                readyKP.pop_front();
                runIndex(idx);
                idleSince = std::chrono::steady_clock::now();
            }
            if (pendingCons.empty()) break;

            bool fired = false;
            for (size_t step = 0; step < pendingCons.size(); ++step) {
                if (rrCursor >= pendingCons.size()) rrCursor = 0;
                size_t idx = pendingCons[rrCursor];
                if (runtime_[idx].tryReady && runtime_[idx].tryReady()) {
                    pendingCons.erase(pendingCons.begin() +
                                      static_cast<std::ptrdiff_t>(rrCursor));
                    runIndex(idx);
                    fired = true;
                    idleSince = std::chrono::steady_clock::now();
                    break;
                }
                ++rrCursor;
            }
            if (!fired) {
                if (std::chrono::steady_clock::now() - idleSince > kBridgeWaitTimeout) {
                    std::string pending;
                    for (size_t idx : pendingCons) {
                        if (!pending.empty()) pending += ", ";
                        pending += runtime_[idx].id;
                    }
                    throw std::runtime_error(
                        "CpuDevice: timed out waiting for bridge consumer(s): " + pending);
                }
                std::this_thread::yield();
            }
        }
    }

    uint64_t scalarBits(const std::string& name, uint64_t scopeId) const {
        auto it = scalarValues_->find(scopedScalarKey(scopeId, name));
        if (it == scalarValues_->end() && scopeId == 0) {
            it = scalarValues_->find(name);
        }
        if (it == scalarValues_->end()) {
            throw std::runtime_error(
                "CpuDevice: scalar '" + name + "' is not set before control-flow evaluation");
        }
        return it->second;
    }

    uint64_t operandBits(const ConditionOperand& operand) const {
        if (operand.isConstant()) return operand.constantBits();
        return scalarBits(operand.name(), operand.scopeId());
    }

    uint64_t evaluateTripCount(const LoopTripCount& tripCount) const {
        const uint64_t bits = (tripCount.kind() == LoopTripCount::Kind::Constant)
                                  ? tripCount.constantBits()
                                  : scalarBits(tripCount.name(), tripCount.scopeId());
        if (isSignedIntegerScalarType(tripCount.type())) {
            const int64_t value = signedScalarValue(tripCount.type(), bits);
            if (value < 0) {
                throw std::runtime_error("CpuDevice: loop trip count cannot be negative");
            }
            return static_cast<uint64_t>(value);
        }
        return unsignedScalarValue(tripCount.type(), bits);
    }

    bool evaluateCondition(const Condition& condition) const {
        switch (condition.op()) {
            case CompareOp::AlwaysTrue:
                return true;
            case CompareOp::AlwaysFalse:
                return false;
            default:
                break;
        }

        if (!condition.lhs() || !condition.rhs()) {
            throw std::runtime_error("CpuDevice: condition is missing comparison operands");
        }

        const ScalarType type = condition.lhs()->type();
        const uint64_t lhsBits = operandBits(*condition.lhs());
        const uint64_t rhsBits = operandBits(*condition.rhs());

        if (condition.isEpsilonCompare()) {
            if (!condition.epsilon()) {
                throw std::runtime_error("CpuDevice: epsilon comparison is missing epsilon");
            }
            const long double lhs = floatingScalarValue(type, lhsBits);
            const long double rhs = floatingScalarValue(type, rhsBits);
            const long double epsilon = floatingScalarValue(type, operandBits(*condition.epsilon()));
            const bool equal = std::fabs(lhs - rhs) <= epsilon;
            return condition.op() == CompareOp::EQE ? equal : !equal;
        }

        if (isFloatingScalarType(type)) {
            return compareValues(condition.op(), floatingScalarValue(type, lhsBits),
                                 floatingScalarValue(type, rhsBits));
        }
        if (isSignedIntegerScalarType(type)) {
            return compareValues(condition.op(), signedScalarValue(type, lhsBits),
                                 signedScalarValue(type, rhsBits));
        }
        return compareValues(condition.op(), unsignedScalarValue(type, lhsBits),
                             unsignedScalarValue(type, rhsBits));
    }

    template <typename ControlNodeT>
    bool hasMaterializedOutputs(const ControlNodeT& control) const {
        return !control.outputBufferPublications.empty() ||
               !control.outputScalarPublications.empty() ||
               !control.outputBufferPlacements.empty() ||
               !control.outputScalarPlacements.empty();
    }

    std::vector<std::unique_ptr<IDevicePlan>>& childPlansFor(
        const std::string& controlId,
        DGraphChildRole role) {
        auto parentIt = childPlans_.find(controlId);
        if (parentIt == childPlans_.end()) {
            throw std::runtime_error(
                "CpuDevice: control node '" + controlId + "' is missing child DGraphs");
        }
        auto roleIt = parentIt->second.find(role);
        if (roleIt == parentIt->second.end()) {
            throw std::runtime_error(
                "CpuDevice: control node '" + controlId + "' is missing child DGraphs");
        }
        return roleIt->second;
    }

    void runChildPlans(const std::string& controlId, DGraphChildRole role) {
        auto& plans = childPlansFor(controlId, role);
        std::exception_ptr firstException;
        size_t launched = 0;

        for (auto& plan : plans) {
            try {
                plan->launch();
                ++launched;
            } catch (...) {
                firstException = std::current_exception();
                break;
            }
        }

        for (size_t i = 0; i < launched; ++i) {
            try {
                plans[i]->wait();
            } catch (...) {
                if (!firstException) firstException = std::current_exception();
            }
        }

        if (firstException) {
            std::rethrow_exception(firstException);
        }
    }

    template <typename ControlNodeT>
    void requireParentBufferPlacement(const ControlNodeT& control,
                                      const std::string& tokenName) const {
        auto placementIt = control.outputBufferPlacements.find(tokenName);
        if (placementIt != control.outputBufferPlacements.end() &&
            placementIt->second != device_.id()) {
            throw std::runtime_error(
                "CpuDevice: cross-device control-flow output buffer publication is not executable yet");
        }
    }

    template <typename ControlNodeT>
    void requireParentScalarPlacement(const ControlNodeT& control,
                                      const std::string& tokenName) const {
        auto placementIt = control.outputScalarPlacements.find(tokenName);
        if (placementIt != control.outputScalarPlacements.end() &&
            placementIt->second != device_.id()) {
            throw std::runtime_error(
                "CpuDevice: cross-device control-flow output scalar publication is not executable yet");
        }
    }

    void executeBoundary(const CompiledBoundaryNode& boundary) {
        for (const auto& copy : boundary.scalarCopies) {
            const std::string sourceKey = scopedScalarKey(copy.sourceScopeId, copy.sourceName);
            auto sourceIt = scalarValues_->find(sourceKey);
            if (sourceIt == scalarValues_->end() && copy.sourceScopeId == 0) {
                sourceIt = scalarValues_->find(copy.sourceName);
            }
            if (sourceIt == scalarValues_->end()) {
                throw std::runtime_error(
                    "CpuDevice: boundary scalar source '" + copy.sourceName +
                    "' was not set before boundary execution");
            }
            (*scalarValues_)[scopedScalarKey(copy.targetScopeId, copy.targetName)] =
                sourceIt->second;
        }
        for (const auto& copy : boundary.bufferCopies) {
            const std::string sourceKey = scopedBufferKey(copy.sourceScopeId, copy.sourceName);
            auto sourceIt = device_.buffers_.find(sourceKey);
            if (sourceIt == device_.buffers_.end() && copy.sourceScopeId == 0) {
                sourceIt = device_.buffers_.find(copy.sourceName);
            }
            if (sourceIt == device_.buffers_.end()) {
                throw std::runtime_error(
                    "CpuDevice: boundary buffer source '" + copy.sourceName +
                    "' was not produced before boundary execution");
            }
            device_.buffers_[scopedBufferKey(copy.targetScopeId, copy.targetName)] =
                sourceIt->second;
        }
    }

    template <typename ControlNodeT, typename PublicationT>
    void publishBuffer(const ControlNodeT& control,
                       const PublicationT& publication,
                       const std::string& sourceTokenName,
                       uint64_t sourceScopeId,
                       const std::string& sourceDeviceId) {
        if (!sourceDeviceId.empty() && sourceDeviceId != device_.id()) {
            throw std::runtime_error(
                "CpuDevice: cross-device control-flow output buffer publication is not executable yet");
        }
        const std::string parentKey = scopedBufferKey(publication.parentScopeId,
                                                      publication.parentTokenName);
        requireParentBufferPlacement(control, parentKey);
        const std::string sourceKey = scopedBufferKey(sourceScopeId, sourceTokenName);
        auto sourceIt = device_.buffers_.find(sourceKey);
        if (sourceIt == device_.buffers_.end() && sourceScopeId == 0) {
            sourceIt = device_.buffers_.find(sourceTokenName);
        }
        if (sourceIt == device_.buffers_.end()) {
            throw std::runtime_error(
                "CpuDevice: control-flow output buffer source '" + sourceTokenName +
                "' was not produced");
        }
        device_.buffers_[parentKey] = sourceIt->second;
    }

    template <typename ControlNodeT, typename PublicationT>
    void publishScalar(const ControlNodeT& control,
                       const PublicationT& publication,
                       const std::string& sourceTokenName,
                       uint64_t sourceScopeId,
                       const std::string& sourceDeviceId) {
        if (!sourceDeviceId.empty() && sourceDeviceId != device_.id()) {
            throw std::runtime_error(
                "CpuDevice: cross-device control-flow output scalar publication is not executable yet");
        }
        requireParentScalarPlacement(
            control, scopedScalarKey(publication.parentScopeId, publication.parentTokenName));
        auto sourceIt = scalarValues_->find(scopedScalarKey(sourceScopeId, sourceTokenName));
        if (sourceIt == scalarValues_->end()) {
            throw std::runtime_error(
                "CpuDevice: control-flow output scalar source '" + sourceTokenName +
                "' was not produced");
        }
        (*scalarValues_)[scopedScalarKey(publication.parentScopeId,
                                         publication.parentTokenName)] = sourceIt->second;
    }

    void publishLoopOutputs(const CompiledLoopNode& loop) {
        for (const auto& publication : loop.outputBufferPublications) {
            publishBuffer(loop, publication, publication.sourceTokenName,
                          publication.sourceScopeId,
                          publication.sourceDeviceId);
        }
        for (const auto& publication : loop.outputScalarPublications) {
            publishScalar(loop, publication, publication.sourceTokenName,
                          publication.sourceScopeId,
                          publication.sourceDeviceId);
        }
    }

    void publishConditionalOutputs(const CompiledConditionalNode& cond, bool thenBranch) {
        for (const auto& publication : cond.outputBufferPublications) {
            publishBuffer(cond, publication,
                          thenBranch ? publication.thenSourceTokenName
                                     : publication.elseSourceTokenName,
                          thenBranch ? publication.thenSourceScopeId
                                     : publication.elseSourceScopeId,
                          thenBranch ? publication.thenSourceDeviceId
                                     : publication.elseSourceDeviceId);
        }
        for (const auto& publication : cond.outputScalarPublications) {
            publishScalar(cond, publication,
                          thenBranch ? publication.thenSourceTokenName
                                     : publication.elseSourceTokenName,
                          thenBranch ? publication.thenSourceScopeId
                                     : publication.elseSourceScopeId,
                          thenBranch ? publication.thenSourceDeviceId
                                     : publication.elseSourceDeviceId);
        }
    }

    void executeLoop(const CompiledLoopNode& loop) {
        bool completedIteration = false;
        if (loop.loopKind == CompiledLoopKind::FixedCount) {
            if (!loop.tripCount) {
                throw std::runtime_error("CpuDevice: compiled fixed-count loop is missing trip count");
            }
            const uint64_t count = evaluateTripCount(*loop.tripCount);
            if (count == 0) {
                if (hasMaterializedOutputs(loop)) {
                    throw std::runtime_error(
                        "CpuDevice: zero-iteration loop cannot materialize outputs");
                }
                return;
            }
            for (uint64_t i = 0; i < count; ++i) {
                runChildPlans(loop.id, DGraphChildRole::LoopBody);
                completedIteration = true;
            }
        } else {
            if (!loop.condition) {
                throw std::runtime_error("CpuDevice: compiled while loop is missing condition");
            }
            while (evaluateCondition(*loop.condition)) {
                runChildPlans(loop.id, DGraphChildRole::LoopBody);
                completedIteration = true;
            }
        }

        if (!completedIteration && hasMaterializedOutputs(loop)) {
            throw std::runtime_error("CpuDevice: zero-iteration loop cannot materialize outputs");
        }
        if (completedIteration) publishLoopOutputs(loop);
    }

    void executeConditional(const CompiledConditionalNode& cond) {
        const bool thenBranch = evaluateCondition(cond.condition);
        runChildPlans(cond.id, thenBranch ? DGraphChildRole::ConditionalThen
                                          : DGraphChildRole::ConditionalElse);
        publishConditionalOutputs(cond, thenBranch);
    }

    void executeKernel(const CompiledKernelNode& node);
    CpuBufferView resolveBuffer(const GraphBuffer& buffer) const;
    std::vector<uint8_t>& ensureBuffer(const GraphBuffer& buffer, size_t sizeBytes);

    CpuDevice& device_;
    std::shared_ptr<std::map<std::string, uint64_t>> scalarValues_;
    std::vector<NodeRuntime> runtime_;
    std::unordered_map<std::string, size_t> idToIdx_;
    std::map<std::string, std::map<DGraphChildRole, std::vector<std::unique_ptr<IDevicePlan>>>>
        childPlans_;
    std::thread worker_;
    std::exception_ptr workerException_;
};

// ---------------------------------------------------------------------------
// CpuBufferView helpers
// ---------------------------------------------------------------------------

namespace {

size_t elementSize(BufferType t) {
    switch (t) {
        case BufferType::U8:  case BufferType::I8:  return 1;
        case BufferType::U16: case BufferType::I16: return 2;
        case BufferType::U32: case BufferType::I32: case BufferType::F32: return 4;
        case BufferType::U64: case BufferType::I64: case BufferType::F64: return 8;
    }
    return 1;
}

}  // namespace

size_t CpuBufferView::elementCount() const {
    size_t es = elementSize(elementType);
    return (es > 0) ? (sizeBytes / es) : 0;
}

// ---------------------------------------------------------------------------
// CpuDevice
// ---------------------------------------------------------------------------

CpuDevice::CpuDevice(std::string id) : id_(std::move(id)) {}

void CpuDevice::registerKernel(std::shared_ptr<CpuKernel> kernel) {
    if (!kernel) {
        throw std::invalid_argument("CpuDevice::registerKernel: kernel must not be null");
    }
    std::string kernelName = kernel->name();
    if (kernelName.empty()) {
        throw std::invalid_argument("CpuDevice::registerKernel: kernel name must not be empty");
    }
    kernels_[std::move(kernelName)] = std::move(kernel);
}

// Buffer keys are always scoped ("scope:N:name"). The user-facing setters and
// getters accept a plain buffer name — the declared root-scope name — and
// internally normalize to the scope-0 storage key. Cross-device bridges that
// already pass a scoped key (e.g. "scope:1:state" for a buffer published from
// a loop body) continue to work because we leave keys with the "scope:"
// prefix untouched.
namespace {
inline std::string normalizeUserBufferKey(const std::string& bufferName) {
    if (bufferName.rfind("scope:", 0) == 0) return bufferName;
    return scopedBufferKey(0, bufferName);
}
}  // namespace

void CpuDevice::setInputBuffer(const std::string& bufferName,
                                const void*        data,
                                size_t             sizeBytes) {
    auto& buf = buffers_[normalizeUserBufferKey(bufferName)];
    buf.resize(sizeBytes);
    if (data && sizeBytes > 0) {
        std::memcpy(buf.data(), data, sizeBytes);
    }
}

void CpuDevice::getOutputBuffer(const std::string& bufferName,
                                void*              data,
                                size_t             sizeBytes) const {
    auto it = buffers_.find(normalizeUserBufferKey(bufferName));
    if (it == buffers_.end()) {
        throw std::runtime_error("CpuDevice::getOutputBuffer: unknown buffer '" + bufferName + "'");
    }
    const auto& buf = it->second;
    if (sizeBytes > buf.size()) {
        throw std::out_of_range("CpuDevice::getOutputBuffer: requested " +
                                std::to_string(sizeBytes) + " bytes but buffer '" +
                                bufferName + "' holds " + std::to_string(buf.size()));
    }
    std::memcpy(data, buf.data(), sizeBytes);
}

size_t CpuDevice::bufferSize(const std::string& bufferName) const {
    auto it = buffers_.find(normalizeUserBufferKey(bufferName));
    return (it == buffers_.end()) ? 0 : it->second.size();
}

std::unique_ptr<IDevicePlan> CpuDevice::compilePlan(const DGraph& dg) {
    return std::make_unique<CpuDevicePlan>(*this, dg);
}

// --- CpuDevicePlan kernel execution helpers ---

void CpuDevicePlan::executeKernel(const CompiledKernelNode& node) {
    const std::string& kname = node.kernel.name;
    auto it = device_.kernels_.find(kname);
    if (it == device_.kernels_.end()) {
        throw std::runtime_error(
            "CpuDevice: no kernel registered for '" + kname + "'");
    }

    std::map<std::string, CpuBufferView> bufViews;

    for (const auto& [portName, gbuf] : node.ioMap.inputBuffers()) {
        CpuBufferView v = resolveBuffer(gbuf);
        v.elementType   = gbuf.type();
        bufViews[portName] = v;
    }

    const auto& inBufs = node.ioMap.inputBuffers();
    size_t defaultOutputSize = 0;
    if (!inBufs.empty()) {
        const GraphBuffer& firstBuffer = inBufs.begin()->second;
        const std::string firstKey = scopedBufferKey(firstBuffer.scopeId(), firstBuffer.name());
        auto fit = device_.buffers_.find(firstKey);
        if (fit != device_.buffers_.end()) {
            defaultOutputSize = fit->second.size();
        }
    }

    for (const auto& [portName, gbuf] : node.ioMap.outputBuffers()) {
        auto& storage = ensureBuffer(gbuf, defaultOutputSize);
        bufViews[portName] = CpuBufferView{storage.data(), storage.size(), gbuf.type()};
    }

    for (const auto& rwb : node.ioMap.rwBuffers()) {
        bufViews[rwb.inPort] = resolveBuffer(rwb.in);
        const std::string inKey = scopedBufferKey(rwb.in.scopeId(), rwb.in.name());
        const std::string outKey = scopedBufferKey(rwb.out.scopeId(), rwb.out.name());
        auto& inStorage = device_.buffers_.at(inKey);
        device_.buffers_[outKey] = inStorage;
        bufViews[rwb.outPort] = CpuBufferView{
            device_.buffers_[outKey].data(),
            device_.buffers_[outKey].size(),
            rwb.out.type()
        };
    }

    std::map<std::string, uint64_t> scalars;
    std::map<std::string, uint64_t*> writableScalars;
    std::set<std::string> outputScalarPorts;
    for (const auto& port : node.kernel.ioType.outputScalars) {
        outputScalarPorts.insert(port.name);
    }

    for (const auto& [portName, gs] : node.ioMap.scalars()) {
        if (outputScalarPorts.count(portName)) {
            if (gs.isConstant()) {
                throw std::runtime_error(
                    "CpuDevice: output scalar port '" + portName +
                    "' cannot be bound to a constant");
            }
            writableScalars[portName] =
                &(*scalarValues_)[scopedScalarKey(gs.scopeId(), gs.varName())];
            continue;
        }

        if (gs.isConstant()) {
            scalars[portName] = gs.constantBits();
        } else {
            auto sit = scalarValues_->find(scopedScalarKey(gs.scopeId(), gs.varName()));
            if (sit == scalarValues_->end() && gs.scopeId() == 0) {
                sit = scalarValues_->find(gs.varName());
            }
            if (sit == scalarValues_->end()) {
                throw std::runtime_error(
                    "CpuDevice: global scalar '" + gs.varName() + "' not set before launch");
            }
            scalars[portName] = sit->second;
        }
    }

    CpuKernelArgs args(std::move(bufViews), std::move(scalars), std::move(writableScalars));
    it->second->run(args);
}

CpuBufferView CpuDevicePlan::resolveBuffer(const GraphBuffer& buffer) const {
    const std::string key = scopedBufferKey(buffer.scopeId(), buffer.name());
    auto it = device_.buffers_.find(key);
    if (it == device_.buffers_.end()) {
        throw std::runtime_error(
            "CpuDevice: buffer '" + buffer.name() + "' not found; "
            "did you forget to call setInputBuffer()?");
    }
    return CpuBufferView{
        const_cast<void*>(static_cast<const void*>(it->second.data())),
        it->second.size(),
        BufferType::U8
    };
}

std::vector<uint8_t>& CpuDevicePlan::ensureBuffer(const GraphBuffer& buffer, size_t sizeBytes) {
    auto& buf = device_.buffers_[scopedBufferKey(buffer.scopeId(), buffer.name())];
    if (buf.size() < sizeBytes) {
        buf.resize(sizeBytes);
    }
    return buf;
}

}  // namespace vrt::graph
