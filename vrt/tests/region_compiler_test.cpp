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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <vrt/graph/compiler.hpp>
#include <vrt/graph/control/condition.hpp>
#include <vrt/graph/control/graph_region.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/crossdevice/bridge_op.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>

#include "test_support/control_specs.hpp"

using namespace vrt::graph;
using namespace vrt::graph::test_support;

namespace {

class NoopDevicePlan : public IDevicePlan {
   public:
    void launch() override {}
    void wait() override {}
};

class StubDevice : public IDevice {
   public:
    StubDevice(std::string id, DeviceType type)
        : id_(std::move(id)), type_(type) {}

    DeviceType type() const override { return type_; }
    std::string id() const override { return id_; }

    std::unique_ptr<IDevicePlan> compilePlan(const DGraph& dgraph) override {
        (void)dgraph;
        return std::make_unique<NoopDevicePlan>();
    }

   private:
    std::string id_;
    DeviceType type_ = DeviceType::CPU;
};

struct InspectionBridgeOp : IBridgeOp {
    explicit InspectionBridgeOp(std::string labelValue)
        : labelValue(std::move(labelValue)) {}

    std::string label() const override { return labelValue; }

    std::string labelValue;
};

class InspectionBridge : public IBridge {
   public:
    BridgeStepPair makeTransfer(IDevice& /*src*/, IDevice& /*dst*/,
                                 const GraphBuffer& /*buffer*/, uint64_t /*sizeHintBytes*/,
                                 const std::string& /*producerNodeId*/,
                                 const std::string& /*consumerNodeId*/) override {
        return BridgeStepPair{
            std::make_shared<InspectionBridgeOp>("inspection_xfer"),
            []() {},
            []() { return true; },
            []() {}};
    }

    BridgeStepPair makeBarrier(IDevice& /*src*/, IDevice& /*dst*/,
                                const std::string& /*producerNodeId*/,
                                const std::string& /*consumerNodeId*/) override {
        return BridgeStepPair{
            std::make_shared<InspectionBridgeOp>("inspection_barrier"),
            []() {},
            []() { return true; },
            []() {}};
    }
};

IOTypeMap singleOutputType(BufferType bufferType = BufferType::I32) {
    IOTypeMap ioType;
    ioType.outputBuffers.push_back({"out", bufferType});
    return ioType;
}

IOTypeMap singleInputOutputType(BufferType bufferType = BufferType::I32) {
    IOTypeMap ioType;
    ioType.inputBuffers.push_back({"in", bufferType});
    ioType.outputBuffers.push_back({"out", bufferType});
    return ioType;
}

IOTypeMap singleInputScalarType(ScalarType scalarType = ScalarType::I32) {
    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", scalarType});
    return ioType;
}

IOTypeMap singleOutputScalarType(ScalarType scalarType = ScalarType::I32) {
    IOTypeMap ioType;
    ioType.outputScalars.push_back({"out", scalarType});
    return ioType;
}

std::string addOutputKernel(GraphRegion& region,
                            KernelDescriptor kernel,
                            BufferType bufferType,
                            const std::string& deviceHint) {
    IOMap kernelIo;
    GraphBuffer output;
    kernelIo.bindOutputBuffer("out", bufferType, output, region.scopeId());
    return region.addKernel(std::move(kernel), std::move(kernelIo), deviceHint);
}

class AddI32BufferKernel : public CpuKernel {
   public:
    AddI32BufferKernel(std::string name, std::int32_t delta)
        : name_(std::move(name)), delta_(delta) {
        ioType_.inputBuffers.push_back({"in", BufferType::I32});
        ioType_.outputBuffers.push_back({"out", BufferType::I32});
    }

    const std::string& name() const override { return name_; }
    const IOTypeMap& ioTypeMap() const override { return ioType_; }

    void call(const CpuKernelArgs& args) override {
        const auto& in = args.buffer("in");
        const auto& out = args.buffer("out");
        const auto* src = in.as<const std::int32_t>();
        auto* dst = out.as<std::int32_t>();
        const std::size_t count = std::min(in.sizeBytes, out.sizeBytes) / sizeof(std::int32_t);
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = src[i] + delta_;
        }
    }

   private:
    std::string name_;
    std::int32_t delta_ = 0;
    IOTypeMap ioType_;
};

GraphBuffer bindControlOutput(IOMap& ioMap,
                              const GraphRegion& region,
                              BufferType bufferType = BufferType::I32) {
    GraphBuffer output;
    ioMap.bindOutputBuffer("out", bufferType, output, region.scopeId());
    return output;
}

// Auto-register stub factories for every (CPU<->non-CPU) and (non-CPU<->non-CPU)
// device-type pair seen in @p graph so the unified bridge-factory check inside
// GraphCompiler::compile can pass. Inspection tests don't actually exercise
// the factories — bridge resolution goes through a custom `bridgeFor` lambda —
// but the validator still requires them to be registered.
void ensureInspectionBridgeFactories(Graph& graph) {
    auto stubFactory = [](IDevice& /*src*/, IDevice& /*dst*/) -> std::shared_ptr<IBridge> {
        return std::make_shared<InspectionBridge>();
    };
    std::set<DeviceType> seen;
    for (const auto& [id, dev] : graph.devices()) {
        (void)id;
        if (dev) seen.insert(dev->type());
    }
    auto registerIfMissing = [&](DeviceType s, DeviceType d) {
        if (s == DeviceType::CPU && d == DeviceType::CPU) return;
        const auto key = std::make_pair(s, d);
        if (graph.bridgeFactories().count(key)) return;
        graph.registerBridgeFactory(s, d, stubFactory);
    };
    for (DeviceType s : seen) {
        for (DeviceType d : seen) {
            registerIfMissing(s, d);
        }
    }
}

std::vector<DGraph> compileForInspection(Graph& graph) {
    ensureInspectionBridgeFactories(graph);
    GraphCompiler compiler;
    auto bridgeFor = [](const std::string& src, const std::string& dst) -> IBridge* {
        throw std::runtime_error(
            "unexpected bridge request from '" + src + "' to '" + dst + "'");
    };
    return compiler.compile(graph.rootRegion(), graph.devices(),
                            graph.bridgeFactories(), bridgeFor,
                            std::make_shared<std::map<std::string, uint64_t>>());
}

std::vector<DGraph> compileForInspection(Graph& graph, IBridge& bridge) {
    ensureInspectionBridgeFactories(graph);
    GraphCompiler compiler;
    auto bridgeFor = [&bridge](const std::string& /*src*/, const std::string& /*dst*/)
        -> IBridge* { return &bridge; };
    return compiler.compile(graph.rootRegion(), graph.devices(),
                            graph.bridgeFactories(), bridgeFor,
                            std::make_shared<std::map<std::string, uint64_t>>());
}

const DGraph* findDGraph(const std::vector<DGraph>& dgraphs, const std::string& deviceId) {
    for (const auto& dgraph : dgraphs) {
        if (dgraph.deviceId == deviceId) return &dgraph;
    }
    return nullptr;
}

const CompiledNode* findCompiledNode(const DGraph& dgraph, const std::string& nodeId) {
    for (const auto& node : dgraph.nodes) {
        if (compiledNodeId(node) == nodeId) return &node;
    }
    return nullptr;
}

const DGraphChild* findChildDGraphs(const DGraph& dgraph,
                                    const std::string& parentNodeId,
                                    DGraphChildRole role) {
    for (const auto& child : dgraph.childDGraphs) {
        if (child.parentNodeId == parentNodeId && child.role == role) return &child;
    }
    return nullptr;
}

const DGraph* findChildDGraph(const DGraphChild& child, const std::string& deviceId) {
    for (const auto& dgraph : child.dgraphs) {
        if (dgraph && dgraph->deviceId == deviceId) return dgraph.get();
    }
    return nullptr;
}

const CompiledBridgeOpNode* findBridgeNode(const DGraph& dgraph,
                                           CompiledBridgeOpNode::Side side,
                                           const std::string& pairedKernelId) {
    for (const auto& node : dgraph.nodes) {
        const auto* bridge = std::get_if<CompiledBridgeOpNode>(&node);
        if (!bridge) continue;
        if (bridge->side == side && bridge->pairedKernelId == pairedKernelId) return bridge;
    }
    return nullptr;
}

bool dependsOn(const CompiledNode& node, const std::string& dependencyId) {
    const auto& dependencies = compiledNodeDependsOn(node);
    return std::find(dependencies.begin(), dependencies.end(), dependencyId) !=
           dependencies.end();
}

bool dependsOn(const CompiledBridgeOpNode& node, const std::string& dependencyId) {
    return std::find(node.dependsOn.begin(), node.dependsOn.end(), dependencyId) !=
           node.dependsOn.end();
}

}  // namespace

TEST(RegionCompilerTest, ConditionValidatesOperandTypes) {
    auto lhs = ConditionOperand::scalar(ScalarType::I32, "lhs", 7);
    auto rhs = ConditionOperand::constant<int32_t>(10);

    auto cond = Condition::compare(CompareOp::LT, lhs, rhs);
    EXPECT_EQ(cond.op(), CompareOp::LT);
    ASSERT_TRUE(cond.lhs());
    EXPECT_EQ(cond.lhs()->scopeId(), 7);

    EXPECT_THROW(
        Condition::compare(CompareOp::EQ,
                           ConditionOperand::scalar(ScalarType::I32, "a"),
                           ConditionOperand::constant<uint32_t>(1)),
        std::invalid_argument);
}

TEST(RegionCompilerTest, EpsilonConditionsAreFloatOnly) {
    EXPECT_NO_THROW(
        Condition::compareWithEpsilon(CompareOp::EQE,
                                      ConditionOperand::scalar(ScalarType::F32, "a"),
                                      ConditionOperand::constant<float>(1.0f),
                                      ConditionOperand::constant<float>(0.001f)));

    EXPECT_THROW(
        Condition::compareWithEpsilon(CompareOp::EQE,
                                      ConditionOperand::scalar(ScalarType::I32, "a"),
                                      ConditionOperand::constant<int32_t>(1),
                                      ConditionOperand::constant<int32_t>(0)),
        std::invalid_argument);

    EXPECT_THROW(
        Condition::compare(CompareOp::EQE,
                           ConditionOperand::scalar(ScalarType::F64, "a"),
                           ConditionOperand::constant<double>(1.0)),
        std::invalid_argument);
}

TEST(RegionCompilerTest, LoopTripCountRequiresIntegerType) {
    auto scalarCount = LoopTripCount::scalar(ScalarType::I64, "n", 9);
    EXPECT_EQ(scalarCount.kind(), LoopTripCount::Kind::Scalar);
    EXPECT_EQ(scalarCount.scopeId(), 9);

    auto constantCount = LoopTripCount::constant<int32_t>(3);
    EXPECT_EQ(constantCount.kind(), LoopTripCount::Kind::Constant);
    EXPECT_EQ(constantCount.type(), ScalarType::I32);

    EXPECT_THROW(LoopTripCount::scalar(ScalarType::F32, "n"), std::invalid_argument);
    EXPECT_THROW(LoopTripCount::constant<int32_t>(-3), std::invalid_argument);
}

TEST(RegionCompilerTest, ScopedTokensCarryRegionIdentity) {
    auto root = GraphRegion::createRoot();
    auto body = root->createChild();

    GraphBuffer rootInput = root->inputBuffer(BufferType::I32, "raw");
    GraphScalar bodyScalar = body->scalar(ScalarType::I32, "limit");

    EXPECT_EQ(rootInput.scopeId(), root->scopeId());
    EXPECT_EQ(bodyScalar.scopeId(), body->scopeId());
    EXPECT_NE(root->scopeId(), body->scopeId());
}

TEST(RegionCompilerTest, GraphExposesRootRegion) {
    Graph graph;

    GraphBuffer input = graph.inputBuffer(BufferType::U8, "raw");
    GraphScalar scalar = graph.globalScalar(ScalarType::I32, "n");

    EXPECT_EQ(graph.rootRegion().scopeId(), 0u);
    EXPECT_EQ(input.scopeId(), graph.rootRegion().scopeId());
    EXPECT_EQ(scalar.scopeId(), graph.rootRegion().scopeId());
}

TEST(RegionCompilerTest, GraphAddNodeAuthorsKernelInRootRegion) {
    Graph graph;

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    kernelType.outputBuffers.push_back({"out", BufferType::I32});

    GraphBuffer input = graph.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    GraphBuffer output;
    io.bindInputBuffer("in", input)
      .bindOutputBuffer("out", BufferType::I32, output, graph.rootRegion().scopeId());

    KernelDescriptor kernel{"copy", DeviceType::CPU, std::nullopt, kernelType};
    std::string nodeId = graph.addNode(std::move(kernel), std::move(io), "cpu");

    ASSERT_EQ(graph.rootRegion().ops().size(), 1u);
    const auto& op = graph.rootRegion().ops().front();
    ASSERT_TRUE(std::holds_alternative<KernelOp>(op));
    EXPECT_EQ(std::get<KernelOp>(op).id, nodeId);

    const auto rootKernels = graph.rootKernels();
    ASSERT_EQ(rootKernels.size(), 1u);
    EXPECT_EQ(rootKernels.front().get().id, nodeId);
    EXPECT_EQ(&rootKernels.front().get(),
              &std::get<KernelOp>(graph.rootRegion().ops().front()));
}

TEST(RegionCompilerTest, GraphRootControlHelpersDelegateToRootRegion) {
    Graph graph;
    auto loopBody = graph.rootRegion().createChild();
    auto whileBody = graph.rootRegion().createChild();
    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();

    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(2), loopBody));
    std::string whileId = graph.addLoop(
        whileLoopSpec(Condition::alwaysTrue(), whileBody));
    std::string ifId = graph.addConditional(
        ifElseSpec(Condition::alwaysFalse(), thenRegion, elseRegion));

    ASSERT_EQ(graph.rootRegion().ops().size(), 3u);
    EXPECT_EQ(regionOpId(graph.rootRegion().ops()[0]), loopId);
    EXPECT_EQ(regionOpId(graph.rootRegion().ops()[1]), whileId);
    EXPECT_EQ(regionOpId(graph.rootRegion().ops()[2]), ifId);
    EXPECT_TRUE(std::holds_alternative<LoopOp>(graph.rootRegion().ops()[0]));
    EXPECT_TRUE(std::holds_alternative<LoopOp>(graph.rootRegion().ops()[1]));
    EXPECT_TRUE(std::holds_alternative<ConditionalOp>(graph.rootRegion().ops()[2]));
}

TEST(RegionCompilerTest, RegionStoresKernelAndControlOps) {
    auto root = GraphRegion::createRoot();
    auto body = root->createChild();

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    kernelType.outputBuffers.push_back({"out", BufferType::I32});

    GraphBuffer bodyInput = body->inputBuffer(BufferType::I32, "in_buf");
    IOMap bodyIo;
    GraphBuffer bodyOutput;
    bodyIo.bindInputBuffer("in", bodyInput)
          .bindOutputBuffer("out", BufferType::I32, bodyOutput, body->scopeId());

    KernelDescriptor kernel{"copy", DeviceType::CPU, std::nullopt, kernelType};
    std::string kernelId = body->addKernel(std::move(kernel), std::move(bodyIo), "cpu");

    EXPECT_EQ(body->ops().size(), 1u);
    EXPECT_EQ(regionOpId(body->ops().front()), kernelId);
    EXPECT_EQ(bodyOutput.scopeId(), body->scopeId());

    std::string loopId = root->addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(3), body));
    ASSERT_EQ(root->ops().size(), 1u);
    EXPECT_EQ(regionOpId(root->ops().front()), loopId);

    const auto& loop = std::get<LoopOp>(root->ops().front());
    EXPECT_EQ(loop.kind, LoopKind::FixedCount);
    ASSERT_TRUE(loop.tripCount);
    EXPECT_EQ(loop.body, body);
}

TEST(RegionCompilerTest, CompilerBuildsLoopControlNodeAndChildDGraph) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    std::string bodyKernelId = body->addKernel(cpuKernel("body"), IOMap{}, "cpu");
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledLoopNode>(*loopNode));
    const auto& compiledLoop = std::get<CompiledLoopNode>(*loopNode);
    EXPECT_EQ(compiledLoop.loopKind, CompiledLoopKind::FixedCount);
    ASSERT_TRUE(compiledLoop.tripCount);
    EXPECT_EQ(compiledLoop.tripCount->kind(), LoopTripCount::Kind::Constant);
    EXPECT_EQ(compiledLoop.tripCount->type(), ScalarType::I32);

    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    ASSERT_EQ(bodyChild->dgraphs.size(), 1u);
    ASSERT_NE(bodyChild->dgraphs.front(), nullptr);
    EXPECT_EQ(bodyChild->dgraphs.front()->deviceId, "cpu");
    EXPECT_NE(findCompiledNode(*bodyChild->dgraphs.front(), bodyKernelId), nullptr);
}

TEST(RegionCompilerTest, CompilerBuildsNestedCrossDeviceBridgesInLoopBody) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    IOTypeMap outputType = singleOutputType();
    IOTypeMap inOutType = singleInputOutputType();

    IOMap producerIo;
    GraphBuffer cpuProduced;
    producerIo.bindOutputBuffer("out", BufferType::I32, cpuProduced, body->scopeId());
    std::string cpuProducerId = body->addKernel(cpuKernel("produce", outputType),
                                                std::move(producerIo), "cpu");

    IOMap mockIo;
    GraphBuffer mockProduced;
    mockIo.bindInputBuffer("in", cpuProduced)
          .bindOutputBuffer("out", BufferType::I32, mockProduced, body->scopeId());
    std::string mockKernelId = body->addKernel(mockCpuKernel("mock", inOutType),
                                               std::move(mockIo), "mcpu:0");

    IOMap cpuConsumerIo;
    GraphBuffer cpuConsumed;
    cpuConsumerIo.bindInputBuffer("in", mockProduced)
                 .bindOutputBuffer("out", BufferType::I32, cpuConsumed, body->scopeId());
    std::string cpuConsumerId = body->addKernel(cpuKernel("consume", inOutType),
                                                std::move(cpuConsumerIo), "cpu");

    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    ASSERT_EQ(bodyChild->dgraphs.size(), 2u);

    const DGraph* cpuChild = findChildDGraph(*bodyChild, "cpu");
    const DGraph* mockChild = findChildDGraph(*bodyChild, "mcpu:0");
    ASSERT_NE(cpuChild, nullptr);
    ASSERT_NE(mockChild, nullptr);

    EXPECT_NE(findCompiledNode(*cpuChild, cpuProducerId), nullptr);
    EXPECT_NE(findCompiledNode(*mockChild, mockKernelId), nullptr);
    EXPECT_NE(findCompiledNode(*cpuChild, cpuConsumerId), nullptr);

    const auto* cpuToMockProducer = findBridgeNode(
        *cpuChild, CompiledBridgeOpNode::Side::Producer, cpuProducerId);
    const auto* cpuToMockConsumer = findBridgeNode(
        *mockChild, CompiledBridgeOpNode::Side::Consumer, mockKernelId);
    const auto* mockToCpuProducer = findBridgeNode(
        *mockChild, CompiledBridgeOpNode::Side::Producer, mockKernelId);
    const auto* mockToCpuConsumer = findBridgeNode(
        *cpuChild, CompiledBridgeOpNode::Side::Consumer, cpuConsumerId);

    ASSERT_NE(cpuToMockProducer, nullptr);
    ASSERT_NE(cpuToMockConsumer, nullptr);
    ASSERT_NE(mockToCpuProducer, nullptr);
    ASSERT_NE(mockToCpuConsumer, nullptr);

    EXPECT_TRUE(dependsOn(*cpuToMockProducer, cpuProducerId));
    EXPECT_TRUE(dependsOn(*mockToCpuProducer, mockKernelId));

    const CompiledNode* mockKernel = findCompiledNode(*mockChild, mockKernelId);
    const CompiledNode* cpuConsumer = findCompiledNode(*cpuChild, cpuConsumerId);
    ASSERT_NE(mockKernel, nullptr);
    ASSERT_NE(cpuConsumer, nullptr);
    EXPECT_TRUE(dependsOn(*mockKernel, cpuToMockConsumer->id));
    EXPECT_TRUE(dependsOn(*cpuConsumer, mockToCpuConsumer->id));
}

TEST(RegionCompilerTest, CompilerBuildsNestedCrossDeviceBridgesInConditionalBranch) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    IOTypeMap outputType = singleOutputType();
    IOTypeMap inOutType = singleInputOutputType();

    IOMap producerIo;
    GraphBuffer cpuProduced;
    producerIo.bindOutputBuffer("out", BufferType::I32, cpuProduced,
                                thenRegion->scopeId());
    std::string cpuProducerId = thenRegion->addKernel(
        cpuKernel("produce", outputType), std::move(producerIo), "cpu");

    IOMap mockIo;
    GraphBuffer mockProduced;
    mockIo.bindInputBuffer("in", cpuProduced)
          .bindOutputBuffer("out", BufferType::I32, mockProduced,
                            thenRegion->scopeId());
    std::string mockKernelId = thenRegion->addKernel(
        mockCpuKernel("mock", inOutType), std::move(mockIo), "mcpu:0");

    IOMap cpuConsumerIo;
    GraphBuffer cpuConsumed;
    cpuConsumerIo.bindInputBuffer("in", mockProduced)
                 .bindOutputBuffer("out", BufferType::I32, cpuConsumed,
                                   thenRegion->scopeId());
    std::string cpuConsumerId = thenRegion->addKernel(
        cpuKernel("consume", inOutType), std::move(cpuConsumerIo), "cpu");

    std::string elseKernelId = elseRegion->addKernel(cpuKernel("else"), IOMap{}, "cpu");
    std::string conditionalId = graph.addConditional(
        ifElseSpec(Condition::alwaysTrue(), thenRegion, elseRegion));

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    ASSERT_NE(conditionalNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledConditionalNode>(*conditionalNode));

    const DGraphChild* thenChild = findChildDGraphs(
        *cpuDGraph, conditionalId, DGraphChildRole::ConditionalThen);
    const DGraphChild* elseChild = findChildDGraphs(
        *cpuDGraph, conditionalId, DGraphChildRole::ConditionalElse);
    ASSERT_NE(thenChild, nullptr);
    ASSERT_NE(elseChild, nullptr);
    ASSERT_EQ(thenChild->dgraphs.size(), 2u);
    ASSERT_EQ(elseChild->dgraphs.size(), 1u);

    const DGraph* cpuThenChild = findChildDGraph(*thenChild, "cpu");
    const DGraph* mockThenChild = findChildDGraph(*thenChild, "mcpu:0");
    const DGraph* cpuElseChild = findChildDGraph(*elseChild, "cpu");
    ASSERT_NE(cpuThenChild, nullptr);
    ASSERT_NE(mockThenChild, nullptr);
    ASSERT_NE(cpuElseChild, nullptr);

    EXPECT_NE(findCompiledNode(*cpuThenChild, cpuProducerId), nullptr);
    EXPECT_NE(findCompiledNode(*mockThenChild, mockKernelId), nullptr);
    EXPECT_NE(findCompiledNode(*cpuThenChild, cpuConsumerId), nullptr);
    EXPECT_NE(findCompiledNode(*cpuElseChild, elseKernelId), nullptr);

    const auto* cpuToMockProducer = findBridgeNode(
        *cpuThenChild, CompiledBridgeOpNode::Side::Producer, cpuProducerId);
    const auto* cpuToMockConsumer = findBridgeNode(
        *mockThenChild, CompiledBridgeOpNode::Side::Consumer, mockKernelId);
    const auto* mockToCpuProducer = findBridgeNode(
        *mockThenChild, CompiledBridgeOpNode::Side::Producer, mockKernelId);
    const auto* mockToCpuConsumer = findBridgeNode(
        *cpuThenChild, CompiledBridgeOpNode::Side::Consumer, cpuConsumerId);

    ASSERT_NE(cpuToMockProducer, nullptr);
    ASSERT_NE(cpuToMockConsumer, nullptr);
    ASSERT_NE(mockToCpuProducer, nullptr);
    ASSERT_NE(mockToCpuConsumer, nullptr);

    EXPECT_TRUE(dependsOn(*cpuToMockProducer, cpuProducerId));
    EXPECT_TRUE(dependsOn(*mockToCpuProducer, mockKernelId));

    const CompiledNode* mockKernel = findCompiledNode(*mockThenChild, mockKernelId);
    const CompiledNode* cpuConsumer = findCompiledNode(*cpuThenChild, cpuConsumerId);
    ASSERT_NE(mockKernel, nullptr);
    ASSERT_NE(cpuConsumer, nullptr);
    EXPECT_TRUE(dependsOn(*mockKernel, cpuToMockConsumer->id));
    EXPECT_TRUE(dependsOn(*cpuConsumer, mockToCpuConsumer->id));
}

TEST(RegionCompilerTest, CompilerBuildsConditionalControlNodeAndBranchDGraphs) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    std::string thenKernelId = thenRegion->addKernel(cpuKernel("then"), IOMap{}, "cpu");
    std::string elseKernelId = elseRegion->addKernel(cpuKernel("else"), IOMap{}, "cpu");
    std::string conditionalId = graph.addConditional(
        ifElseSpec(Condition::alwaysTrue(), thenRegion, elseRegion));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    ASSERT_NE(conditionalNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledConditionalNode>(*conditionalNode));
    const auto& compiledConditional = std::get<CompiledConditionalNode>(*conditionalNode);
    EXPECT_EQ(compiledConditional.condition.op(), CompareOp::AlwaysTrue);

    const DGraphChild* thenChild = findChildDGraphs(*cpuDGraph, conditionalId,
                                                   DGraphChildRole::ConditionalThen);
    const DGraphChild* elseChild = findChildDGraphs(*cpuDGraph, conditionalId,
                                                   DGraphChildRole::ConditionalElse);
    ASSERT_NE(thenChild, nullptr);
    ASSERT_NE(elseChild, nullptr);
    ASSERT_EQ(thenChild->dgraphs.size(), 1u);
    ASSERT_EQ(elseChild->dgraphs.size(), 1u);
    ASSERT_NE(thenChild->dgraphs.front(), nullptr);
    ASSERT_NE(elseChild->dgraphs.front(), nullptr);
    EXPECT_NE(findCompiledNode(*thenChild->dgraphs.front(), thenKernelId), nullptr);
    EXPECT_NE(findCompiledNode(*elseChild->dgraphs.front(), elseKernelId), nullptr);
}

TEST(RegionCompilerTest, CompilerLowersScalarBoundaryMappingsInChildDGraph) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");
    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");

    std::string startId = body->importFromParent({{parentCounter, localCounter}});
    std::string endId = body->exportToParent({{localCounter, parentCounter}}, {startId});
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    ASSERT_EQ(bodyChild->dgraphs.size(), 1u);
    ASSERT_NE(bodyChild->dgraphs.front(), nullptr);

    const CompiledNode* startNode = findCompiledNode(*bodyChild->dgraphs.front(), startId);
    const CompiledNode* endNode = findCompiledNode(*bodyChild->dgraphs.front(), endId);
    ASSERT_NE(startNode, nullptr);
    ASSERT_NE(endNode, nullptr);

    const auto& compiledStart = std::get<CompiledBoundaryNode>(*startNode);
    ASSERT_EQ(compiledStart.scalarCopies.size(), 1u);
    EXPECT_EQ(compiledStart.scalarCopies.front().sourceName, parentCounter.varName());
    EXPECT_EQ(compiledStart.scalarCopies.front().sourceScopeId, parentCounter.scopeId());
    EXPECT_EQ(compiledStart.scalarCopies.front().targetName, localCounter.varName());
    EXPECT_EQ(compiledStart.scalarCopies.front().targetScopeId, localCounter.scopeId());

    const auto& compiledEnd = std::get<CompiledBoundaryNode>(*endNode);
    ASSERT_EQ(compiledEnd.scalarCopies.size(), 1u);
    EXPECT_EQ(compiledEnd.scalarCopies.front().sourceName, localCounter.varName());
    EXPECT_EQ(compiledEnd.scalarCopies.front().sourceScopeId, localCounter.scopeId());
    EXPECT_EQ(compiledEnd.scalarCopies.front().targetName, parentCounter.varName());
    EXPECT_EQ(compiledEnd.scalarCopies.front().targetScopeId, parentCounter.scopeId());
}

TEST(RegionCompilerTest, CompilerOrdersScalarBoundaryDependencies) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    IOTypeMap incrementType;
    incrementType.inputScalars.push_back({"in", ScalarType::I32});
    incrementType.outputScalars.push_back({"out", ScalarType::I32});

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");
    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    GraphScalar localNext = body->scalar(ScalarType::I32, "next");

    std::string startId = body->importFromParent({{parentCounter, localCounter}});
    IOMap bodyIo;
    bodyIo.bindScalar("in", localCounter)
          .bindScalar("out", localNext);
    std::string kernelId = body->addKernel(cpuKernel("increment", incrementType),
                                           std::move(bodyIo), "cpu");
    std::string endId = body->exportToParent({{localNext, parentCounter}});
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    ASSERT_EQ(bodyChild->dgraphs.size(), 1u);

    const DGraph& bodyDGraph = *bodyChild->dgraphs.front();
    const CompiledNode* kernelNode = findCompiledNode(bodyDGraph, kernelId);
    const CompiledNode* endNode = findCompiledNode(bodyDGraph, endId);
    ASSERT_NE(kernelNode, nullptr);
    ASSERT_NE(endNode, nullptr);
    EXPECT_TRUE(dependsOn(*kernelNode, startId));
    EXPECT_TRUE(dependsOn(*endNode, kernelId));
}

TEST(RegionCompilerTest, CompilerAllowsLoopCarriedScalarWithInitialProducer) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");

    IOTypeMap initType;
    initType.outputScalars.push_back({"out", ScalarType::I32});
    IOMap initIo;
    initIo.bindScalar("out", parentCounter);
    const std::string initId = graph.addNode(cpuKernel("init_counter", initType),
                                             std::move(initIo), "cpu");

    IOTypeMap incrementType;
    incrementType.inputScalars.push_back({"in", ScalarType::I32});
    incrementType.outputScalars.push_back({"out", ScalarType::I32});

    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    GraphScalar localNext = body->scalar(ScalarType::I32, "next");

    const std::string startId = body->importFromParent({{parentCounter, localCounter}});
    IOMap bodyIo;
    bodyIo.bindScalar("in", localCounter)
          .bindScalar("out", localNext);
    const std::string bodyId = body->addKernel(cpuKernel("increment", incrementType),
                                               std::move(bodyIo), "cpu", {startId});
    body->exportToParent({{localNext, parentCounter}}, {bodyId});

    const std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(3), body));

    IOTypeMap consumerType;
    consumerType.inputScalars.push_back({"in", ScalarType::I32});
    IOMap consumerIo;
    consumerIo.bindScalar("in", parentCounter);
    const std::string consumerId = graph.addNode(cpuKernel("consume_counter", consumerType),
                                                 std::move(consumerIo), "cpu");

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_NE(consumerNode, nullptr);

    EXPECT_TRUE(dependsOn(*loopNode, initId));
    EXPECT_TRUE(dependsOn(*consumerNode, loopId));
    EXPECT_FALSE(dependsOn(*consumerNode, initId));
}

TEST(RegionCompilerTest, CompilerRejectsScalarBoundaryTypeMismatch) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");
    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::U32, "counter");

    body->importFromParent({{parentCounter, localCounter}});
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(graph.compile(), std::runtime_error);
}

// Helper that adds a kernel whose only output is a single I32 scalar bound to
// the given GraphScalar in the given region. Used by the end-boundary tests
// below to ensure the local scalar has a producer before being exported.
std::string addI32ScalarProducerKernel(GraphRegion& region,
                                       const std::string& kernelName,
                                       const GraphScalar& target,
                                       const std::string& deviceHint) {
    IOTypeMap kernelType;
    kernelType.outputScalars.push_back({"out", ScalarType::I32});
    IOMap kernelIo;
    kernelIo.bindScalar("out", target);
    return region.addKernel(cpuKernel(kernelName, std::move(kernelType)),
                            std::move(kernelIo), deviceHint);
}

TEST(RegionCompilerTest, RootReaderDependsOnLoopForScalarExportedToParent) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");
    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    addI32ScalarProducerKernel(*body, "body_produces_counter", localCounter, "cpu");
    body->exportToParent({{localCounter, parentCounter}});
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    IOTypeMap consumerType;
    consumerType.inputScalars.push_back({"in", ScalarType::I32});
    IOMap consumerIo;
    consumerIo.bindScalar("in", parentCounter);
    std::string consumerId = graph.addNode(cpuKernel("consume_counter", consumerType),
                                           std::move(consumerIo), "cpu");

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(consumerNode, nullptr);
    EXPECT_TRUE(dependsOn(*consumerNode, loopId));
}

TEST(RegionCompilerTest, RootReaderDependsOnConditionalWhenSingleBranchExports) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");

    auto thenRegion = graph.rootRegion().createChild();
    GraphScalar thenLocal = thenRegion->scalar(ScalarType::I32, "counter");
    addI32ScalarProducerKernel(*thenRegion, "then_produces_counter", thenLocal, "cpu");
    thenRegion->exportToParent({{thenLocal, parentCounter}});

    auto elseRegion = graph.rootRegion().createChild();

    std::string condId = graph.addConditional(
        ifElseSpec(Condition::alwaysFalse(), thenRegion, elseRegion));

    IOTypeMap consumerType;
    consumerType.inputScalars.push_back({"in", ScalarType::I32});
    IOMap consumerIo;
    consumerIo.bindScalar("in", parentCounter);
    std::string consumerId = graph.addNode(cpuKernel("consume_counter", consumerType),
                                           std::move(consumerIo), "cpu");

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(consumerNode, nullptr);
    EXPECT_TRUE(dependsOn(*consumerNode, condId));
}

TEST(RegionCompilerTest, ExplicitProducerCollidingWithControlEndBoundaryThrows) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");

    IOTypeMap producerType;
    producerType.outputScalars.push_back({"out", ScalarType::I32});
    IOMap producerIo;
    producerIo.bindScalar("out", parentCounter);
    graph.addNode(cpuKernel("explicit_producer", producerType),
                  std::move(producerIo), "cpu");

    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    addI32ScalarProducerKernel(*body, "body_produces_counter", localCounter, "cpu");
    body->exportToParent({{localCounter, parentCounter}});
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    try {
        graph.compile();
        FAIL() << "expected compile() to throw on multi-producer collision";
    } catch (const std::runtime_error& ex) {
        const std::string what = ex.what();
        EXPECT_NE(what.find("multiple ops write scoped scalar"), std::string::npos)
            << "actual: " << what;
        EXPECT_NE(what.find("end-boundary"), std::string::npos) << "actual: " << what;
        EXPECT_NE(what.find("afterOps"), std::string::npos) << "actual: " << what;
    }
}

TEST(RegionCompilerTest, CompilerLowersBufferBoundaryMappingsInChildDGraph) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphBuffer parentInput = graph.inputBuffer(BufferType::I32, "parent_input");
    GraphBuffer parentOutput = GraphBuffer::make(BufferType::I32, "parent_output",
                                                 graph.rootRegion().scopeId());
    auto body = graph.rootRegion().createChild();
    GraphBuffer localInput = body->inputBuffer(BufferType::I32, "input");

    std::string startId = body->importFromParent(
        std::vector<BufferBoundaryMapping>{{parentInput, localInput}});

    IOMap producerIo;
    GraphBuffer localOutput;
    producerIo.bindOutputBuffer("out", BufferType::I32, localOutput, body->scopeId());
    std::string producerId = body->addKernel(cpuKernel("produce", singleOutputType()),
                                             std::move(producerIo), "cpu");
    std::string endId = body->exportToParent(
        std::vector<BufferBoundaryMapping>{{localOutput, parentOutput}}, {producerId});
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    ASSERT_EQ(bodyChild->dgraphs.size(), 1u);

    const CompiledNode* startNode = findCompiledNode(*bodyChild->dgraphs.front(), startId);
    const CompiledNode* endNode = findCompiledNode(*bodyChild->dgraphs.front(), endId);
    ASSERT_NE(startNode, nullptr);
    ASSERT_NE(endNode, nullptr);

    const auto& compiledStart = std::get<CompiledBoundaryNode>(*startNode);
    ASSERT_EQ(compiledStart.bufferCopies.size(), 1u);
    EXPECT_EQ(compiledStart.bufferCopies.front().sourceName, parentInput.name());
    EXPECT_EQ(compiledStart.bufferCopies.front().sourceScopeId, parentInput.scopeId());
    EXPECT_EQ(compiledStart.bufferCopies.front().targetName, localInput.name());
    EXPECT_EQ(compiledStart.bufferCopies.front().targetScopeId, localInput.scopeId());

    const auto& compiledEnd = std::get<CompiledBoundaryNode>(*endNode);
    ASSERT_EQ(compiledEnd.bufferCopies.size(), 1u);
    EXPECT_EQ(compiledEnd.bufferCopies.front().sourceName, localOutput.name());
    EXPECT_EQ(compiledEnd.bufferCopies.front().sourceScopeId, localOutput.scopeId());
    EXPECT_EQ(compiledEnd.bufferCopies.front().targetName, parentOutput.name());
    EXPECT_EQ(compiledEnd.bufferCopies.front().targetScopeId, parentOutput.scopeId());
}

TEST(RegionCompilerTest, CompilerOrdersBufferBoundaryDependencies) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphBuffer parentState = graph.inputBuffer(BufferType::I32, "state");
    auto body = graph.rootRegion().createChild();
    GraphBuffer localState = body->inputBuffer(BufferType::I32, "state");

    std::string startId = body->importFromParent(
        std::vector<BufferBoundaryMapping>{{parentState, localState}});

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    kernelType.outputBuffers.push_back({"out", BufferType::I32});
    IOMap kernelIo;
    GraphBuffer localNext;
    kernelIo.bindInputBuffer("in", localState)
            .bindOutputBuffer("out", BufferType::I32, localNext, body->scopeId());
    std::string kernelId = body->addKernel(cpuKernel("advance", kernelType),
                                           std::move(kernelIo), "cpu");
    std::string endId = body->exportToParent(
        std::vector<BufferBoundaryMapping>{{localNext, parentState}});
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    const DGraph& bodyDGraph = *bodyChild->dgraphs.front();

    const CompiledNode* kernelNode = findCompiledNode(bodyDGraph, kernelId);
    const CompiledNode* endNode = findCompiledNode(bodyDGraph, endId);
    ASSERT_NE(kernelNode, nullptr);
    ASSERT_NE(endNode, nullptr);
    EXPECT_TRUE(dependsOn(*kernelNode, startId));
    EXPECT_TRUE(dependsOn(*endNode, kernelId));
}

TEST(RegionCompilerTest, CompilerAllowsLoopCarriedBufferWithInitialProducer) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphBuffer raw = graph.inputBuffer(BufferType::I32, "raw");
    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    kernelType.outputBuffers.push_back({"out", BufferType::I32});

    IOMap initIo;
    GraphBuffer parentState;
    initIo.bindInputBuffer("in", raw)
          .bindOutputBuffer("out", BufferType::I32, parentState);
    const std::string initId = graph.addNode(cpuKernel("init", kernelType),
                                             std::move(initIo), "cpu");

    auto body = graph.rootRegion().createChild();
    GraphBuffer localState = body->inputBuffer(BufferType::I32, "state");
    const std::string startId = body->importFromParent(
        std::vector<BufferBoundaryMapping>{{parentState, localState}});

    IOMap bodyIo;
    GraphBuffer localNext;
    bodyIo.bindInputBuffer("in", localState)
          .bindOutputBuffer("out", BufferType::I32, localNext, body->scopeId());
    const std::string bodyId = body->addKernel(cpuKernel("advance", kernelType),
                                               std::move(bodyIo), "cpu", {startId});
    body->exportToParent(std::vector<BufferBoundaryMapping>{{localNext, parentState}},
                         {bodyId});

    const std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(3), body));

    IOMap consumeIo;
    GraphBuffer finalOut;
    consumeIo.bindInputBuffer("in", parentState)
             .bindOutputBuffer("out", BufferType::I32, finalOut);
    const std::string consumeId = graph.addNode(cpuKernel("consume", kernelType),
                                                std::move(consumeIo), "cpu");

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    const CompiledNode* consumeNode = findCompiledNode(*cpuDGraph, consumeId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_NE(consumeNode, nullptr);

    EXPECT_TRUE(dependsOn(*loopNode, initId));
    EXPECT_TRUE(dependsOn(*consumeNode, loopId));
    EXPECT_FALSE(dependsOn(*consumeNode, initId));
}

TEST(RegionCompilerTest, CompilerRejectsBufferBoundaryTypeMismatch) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphBuffer parentInput = graph.inputBuffer(BufferType::I32, "raw");
    auto body = graph.rootRegion().createChild();
    GraphBuffer localInput = body->inputBuffer(BufferType::U32, "raw");

    body->importFromParent(std::vector<BufferBoundaryMapping>{{parentInput, localInput}});
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsUnimportedChildBufferInput) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphBuffer localInput = body->inputBuffer(BufferType::I32, "raw");

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    IOMap kernelIo;
    kernelIo.bindInputBuffer("in", localInput);
    body->addKernel(cpuKernel("consume", kernelType), std::move(kernelIo), "cpu");
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsWrongDirectionBufferBoundaryMapping) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphBuffer parentInput = graph.inputBuffer(BufferType::I32, "raw");
    auto body = graph.rootRegion().createChild();
    GraphBuffer localInput = body->inputBuffer(BufferType::I32, "raw");

    body->importFromParent(std::vector<BufferBoundaryMapping>{{localInput, parentInput}});
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsUnimportedChildScalarInput) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");

    IOMap bodyIo;
    bodyIo.bindScalar("in", localCounter);
    body->addKernel(cpuKernel("consume_scalar", singleInputScalarType()),
                    std::move(bodyIo), "cpu");
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsUnimportedChildConditionScalar) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    auto nestedBody = body->createChild();

    Condition condition = Condition::compare(
        CompareOp::LT,
        ConditionOperand::scalar(ScalarType::I32, localCounter.varName(), localCounter.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    body->addLoop(whileLoopSpec(std::move(condition), nestedBody));
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerAcceptsImportedChildConditionScalar) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar parentCounter = graph.globalScalar(ScalarType::I32, "counter");
    auto body = graph.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    body->importFromParent({{parentCounter, localCounter}});
    auto nestedBody = body->createChild();

    Condition condition = Condition::compare(
        CompareOp::LT,
        ConditionOperand::scalar(ScalarType::I32, localCounter.varName(), localCounter.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    body->addLoop(whileLoopSpec(std::move(condition), nestedBody));
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_NO_THROW(compileForInspection(graph));
}

TEST(RegionCompilerTest, CompilerRejectsUnimportedChildScalarTripCount) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphScalar localTripCount = body->scalar(ScalarType::I32, "trip_count");
    auto nestedBody = body->createChild();

    body->addLoop(fixedLoopSpec(
        LoopTripCount::scalar(ScalarType::I32,
                              localTripCount.varName(),
                              localTripCount.scopeId()),
        nestedBody));
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerOrdersConditionalAfterScalarConditionProducer) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar flag = graph.globalScalar(ScalarType::I32, "flag");
    IOMap producerIo;
    producerIo.bindScalar("out", flag);
    std::string producerId = graph.addNode(cpuKernel("produce_flag", singleOutputScalarType()),
                                           std::move(producerIo), "cpu");

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    std::string conditionalId = graph.addConditional(
        ifElseSpec(std::move(condition), thenRegion, elseRegion));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    ASSERT_NE(conditionalNode, nullptr);
    EXPECT_TRUE(dependsOn(*conditionalNode, producerId));
}

TEST(RegionCompilerTest, CompilerOrdersLoopAfterScalarTripCountProducer) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    GraphScalar tripCount = graph.globalScalar(ScalarType::I32, "trip_count");
    IOMap producerIo;
    producerIo.bindScalar("out", tripCount);
    std::string producerId = graph.addNode(cpuKernel("produce_trip_count",
                                                     singleOutputScalarType()),
                                           std::move(producerIo), "cpu");

    auto body = graph.rootRegion().createChild();
    std::string loopId = graph.addLoop(fixedLoopSpec(
        LoopTripCount::scalar(ScalarType::I32, tripCount.varName(), tripCount.scopeId()),
        body));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    ASSERT_NE(loopNode, nullptr);
    EXPECT_TRUE(dependsOn(*loopNode, producerId));
}

TEST(RegionCompilerTest, CompilerInfersLoopOutputPlacementAndParentDependency) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    std::string bodyKernelId = addOutputKernel(
        *body, cpuKernel("body", singleOutputType()), BufferType::I32, "cpu");

    IOTypeMap loopType = singleOutputType();
    IOMap loopIo;
    GraphBuffer loopOutput = bindControlOutput(loopIo, graph.rootRegion());
    std::string loopId = graph.addLoop(fixedLoopSpec(
        std::move(loopType), std::move(loopIo),
        LoopTripCount::constant<int32_t>(1), body));

    IOTypeMap consumerType;
    consumerType.inputBuffers.push_back({"in", BufferType::I32});
    IOMap consumerIo;
    consumerIo.bindInputBuffer("in", loopOutput);
    std::string consumerId = graph.addNode(cpuKernel("consume", consumerType),
                                           std::move(consumerIo), "cpu");

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_NE(consumerNode, nullptr);

    const auto& compiledLoop = std::get<CompiledLoopNode>(*loopNode);
    const std::string scopedLoopOutput = scopedBufferKey(loopOutput.scopeId(),
                                                          loopOutput.name());
    ASSERT_EQ(compiledLoop.outputBufferPlacements.count(scopedLoopOutput), 1u);
    EXPECT_EQ(compiledLoop.outputBufferPlacements.at(scopedLoopOutput), "cpu");
    ASSERT_EQ(compiledLoop.outputBufferPublications.size(), 1u);
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().portName, "out");
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().parentTokenName,
              loopOutput.name());
    EXPECT_FALSE(compiledLoop.outputBufferPublications.front().sourceTokenName.empty());
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().sourceDeviceId, "cpu");
    EXPECT_TRUE(dependsOn(*consumerNode, loopId));
    EXPECT_FALSE(dependsOn(*consumerNode, bodyKernelId));
}

TEST(RegionCompilerTest, CompilerInfersConditionalOutputPlacementWhenBranchesAgree) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    addOutputKernel(*thenRegion, cpuKernel("then", singleOutputType()), BufferType::I32,
                    "cpu");
    addOutputKernel(*elseRegion, cpuKernel("else", singleOutputType()), BufferType::I32,
                    "cpu");

    IOTypeMap conditionalType = singleOutputType();
    IOMap conditionalIo;
    GraphBuffer conditionalOutput = bindControlOutput(conditionalIo, graph.rootRegion());
    std::string conditionalId = graph.addConditional(
        ifElseSpec(std::move(conditionalType), std::move(conditionalIo),
                   Condition::alwaysTrue(), thenRegion, elseRegion));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    ASSERT_NE(conditionalNode, nullptr);
    const auto& compiledConditional = std::get<CompiledConditionalNode>(*conditionalNode);
    const std::string scopedCondOutput = scopedBufferKey(conditionalOutput.scopeId(),
                                                          conditionalOutput.name());
    ASSERT_EQ(compiledConditional.outputBufferPlacements.count(scopedCondOutput), 1u);
    EXPECT_EQ(compiledConditional.outputBufferPlacements.at(scopedCondOutput), "cpu");
    ASSERT_EQ(compiledConditional.outputBufferPublications.size(), 1u);
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().portName, "out");
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().parentTokenName,
              conditionalOutput.name());
    EXPECT_FALSE(compiledConditional.outputBufferPublications.front().thenSourceTokenName.empty());
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().thenSourceDeviceId, "cpu");
    EXPECT_FALSE(compiledConditional.outputBufferPublications.front().elseSourceTokenName.empty());
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().elseSourceDeviceId, "cpu");
}

TEST(RegionCompilerTest, CompilerRejectsConditionalMissingBranchOutput) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    addOutputKernel(*thenRegion, cpuKernel("then", singleOutputType()), BufferType::I32,
                    "cpu");

    IOTypeMap conditionalType = singleOutputType();
    IOMap conditionalIo;
    bindControlOutput(conditionalIo, graph.rootRegion());
    graph.addConditional(ifElseSpec(std::move(conditionalType), std::move(conditionalIo),
                                    Condition::alwaysTrue(), thenRegion, elseRegion));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsConditionalBranchOutputTypeMismatch) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    addOutputKernel(*thenRegion, cpuKernel("then", singleOutputType(BufferType::I32)),
                    BufferType::I32, "cpu");
    addOutputKernel(*elseRegion, cpuKernel("else", singleOutputType(BufferType::F32)),
                    BufferType::F32, "cpu");

    IOTypeMap conditionalType = singleOutputType(BufferType::I32);
    IOMap conditionalIo;
    bindControlOutput(conditionalIo, graph.rootRegion(), BufferType::I32);
    graph.addConditional(ifElseSpec(std::move(conditionalType), std::move(conditionalIo),
                                    Condition::alwaysTrue(), thenRegion, elseRegion));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsMixedBranchOutputPlacementWithoutHint) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));
    graph.registerDevice(std::make_shared<StubDevice>("mock", DeviceType::MOCK_CPU));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    addOutputKernel(*thenRegion, cpuKernel("then", singleOutputType()), BufferType::I32,
                    "cpu");
    addOutputKernel(*elseRegion, mockCpuKernel("else", singleOutputType()), BufferType::I32,
                    "mock");

    IOTypeMap conditionalType = singleOutputType();
    IOMap conditionalIo;
    bindControlOutput(conditionalIo, graph.rootRegion());
    graph.addConditional(ifElseSpec(std::move(conditionalType), std::move(conditionalIo),
                                    Condition::alwaysTrue(), thenRegion, elseRegion));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerAcceptsExplicitMixedBranchOutputPlacement) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));
    graph.registerDevice(std::make_shared<StubDevice>("mock", DeviceType::MOCK_CPU));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    addOutputKernel(*thenRegion, cpuKernel("then", singleOutputType()), BufferType::I32,
                    "cpu");
    addOutputKernel(*elseRegion, mockCpuKernel("else", singleOutputType()), BufferType::I32,
                    "mock");

    ConditionalSpec spec;
    spec.ioType = singleOutputType();
    GraphBuffer conditionalOutput = bindControlOutput(spec.ioMap, graph.rootRegion());
    spec.condition = Condition::alwaysTrue();
    spec.thenRegion = thenRegion;
    spec.elseRegion = elseRegion;
    spec.outputPlacement.buffers["out"] = "cpu";
    std::string conditionalId = graph.addConditional(std::move(spec));

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    ASSERT_NE(conditionalNode, nullptr);
    const auto& compiledConditional = std::get<CompiledConditionalNode>(*conditionalNode);
    const std::string scopedCondOutput = scopedBufferKey(conditionalOutput.scopeId(),
                                                          conditionalOutput.name());
    ASSERT_EQ(compiledConditional.outputBufferPlacements.count(scopedCondOutput), 1u);
    EXPECT_EQ(compiledConditional.outputBufferPlacements.at(scopedCondOutput), "cpu");
}

TEST(RegionCompilerTest, CompilerBuildsOutputPlacementBridgeForLoopBodyBuffer) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    IOMap producerIo;
    GraphBuffer bodyOutput;
    producerIo.bindOutputBuffer("out", BufferType::I32, bodyOutput, body->scopeId());
    std::string mockProducerId = body->addKernel(
        mockCpuKernel("remote_output", singleOutputType()), std::move(producerIo), "mcpu:0");

    LoopSpec spec;
    spec.ioType = singleOutputType();
    GraphBuffer loopOutput = bindControlOutput(spec.ioMap, graph.rootRegion());
    spec.tripCount = LoopTripCount::constant<int32_t>(1);
    spec.body = body;
    spec.outputPlacement.buffers["out"] = "cpu";
    std::string loopId = graph.addLoop(std::move(spec));

    IOTypeMap consumerType;
    consumerType.inputBuffers.push_back({"in", BufferType::I32});
    IOMap consumerIo;
    consumerIo.bindInputBuffer("in", loopOutput);
    std::string consumerId = graph.addNode(cpuKernel("consume", consumerType),
                                           std::move(consumerIo), "cpu");

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_NE(consumerNode, nullptr);

    const auto& compiledLoop = std::get<CompiledLoopNode>(*loopNode);
    const std::string scopedLoopOutput = scopedBufferKey(loopOutput.scopeId(),
                                                          loopOutput.name());
    ASSERT_EQ(compiledLoop.outputBufferPlacements.count(scopedLoopOutput), 1u);
    EXPECT_EQ(compiledLoop.outputBufferPlacements.at(scopedLoopOutput), "cpu");
    ASSERT_EQ(compiledLoop.outputBufferPublications.size(), 1u);
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().sourceTokenName,
              bodyOutput.name());
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().sourceScopeId,
              bodyOutput.scopeId());
    EXPECT_EQ(compiledLoop.outputBufferPublications.front().sourceDeviceId, "cpu");
    EXPECT_TRUE(dependsOn(*consumerNode, loopId));
    EXPECT_FALSE(dependsOn(*consumerNode, mockProducerId));

    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    const DGraph* cpuChild = findChildDGraph(*bodyChild, "cpu");
    const DGraph* mockChild = findChildDGraph(*bodyChild, "mcpu:0");
    ASSERT_NE(cpuChild, nullptr);
    ASSERT_NE(mockChild, nullptr);

    const auto* producerBridge = findBridgeNode(
        *mockChild, CompiledBridgeOpNode::Side::Producer, mockProducerId);
    const auto* consumerBridge = findBridgeNode(
        *cpuChild, CompiledBridgeOpNode::Side::Consumer, loopId);
    ASSERT_NE(producerBridge, nullptr);
    ASSERT_NE(consumerBridge, nullptr);
    EXPECT_TRUE(dependsOn(*producerBridge, mockProducerId));
}

TEST(RegionCompilerTest, CompilerBuildsOutputPlacementBridgeForConditionalBranchBuffer) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto thenRegion = graph.rootRegion().createChild();
    IOMap thenIo;
    GraphBuffer thenOutput;
    thenIo.bindOutputBuffer("out", BufferType::I32, thenOutput, thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("then_output", singleOutputType()),
                          std::move(thenIo), "cpu");

    auto elseRegion = graph.rootRegion().createChild();
    IOMap elseIo;
    GraphBuffer elseOutput;
    elseIo.bindOutputBuffer("out", BufferType::I32, elseOutput, elseRegion->scopeId());
    std::string elseProducerId = elseRegion->addKernel(
        mockCpuKernel("else_remote_output", singleOutputType()), std::move(elseIo), "mcpu:0");

    ConditionalSpec spec;
    spec.ioType = singleOutputType();
    GraphBuffer conditionalOutput = bindControlOutput(spec.ioMap, graph.rootRegion());
    spec.condition = Condition::alwaysTrue();
    spec.thenRegion = thenRegion;
    spec.elseRegion = elseRegion;
    spec.outputPlacement.buffers["out"] = "cpu";
    std::string conditionalId = graph.addConditional(std::move(spec));

    IOTypeMap consumerType;
    consumerType.inputBuffers.push_back({"in", BufferType::I32});
    IOMap consumerIo;
    consumerIo.bindInputBuffer("in", conditionalOutput);
    std::string consumerId = graph.addNode(cpuKernel("consume", consumerType),
                                           std::move(consumerIo), "cpu");

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* conditionalNode = findCompiledNode(*cpuDGraph, conditionalId);
    const CompiledNode* consumerNode = findCompiledNode(*cpuDGraph, consumerId);
    ASSERT_NE(conditionalNode, nullptr);
    ASSERT_NE(consumerNode, nullptr);

    const auto& compiledConditional = std::get<CompiledConditionalNode>(*conditionalNode);
    const std::string scopedCondOutput = scopedBufferKey(conditionalOutput.scopeId(),
                                                          conditionalOutput.name());
    ASSERT_EQ(compiledConditional.outputBufferPlacements.count(scopedCondOutput), 1u);
    EXPECT_EQ(compiledConditional.outputBufferPlacements.at(scopedCondOutput), "cpu");
    ASSERT_EQ(compiledConditional.outputBufferPublications.size(), 1u);
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().thenSourceDeviceId, "cpu");
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().elseSourceTokenName,
              elseOutput.name());
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().elseSourceScopeId,
              elseOutput.scopeId());
    EXPECT_EQ(compiledConditional.outputBufferPublications.front().elseSourceDeviceId, "cpu");
    EXPECT_TRUE(dependsOn(*consumerNode, conditionalId));
    EXPECT_FALSE(dependsOn(*consumerNode, elseProducerId));

    const DGraphChild* thenChild = findChildDGraphs(
        *cpuDGraph, conditionalId, DGraphChildRole::ConditionalThen);
    const DGraphChild* elseChild = findChildDGraphs(
        *cpuDGraph, conditionalId, DGraphChildRole::ConditionalElse);
    ASSERT_NE(thenChild, nullptr);
    ASSERT_NE(elseChild, nullptr);
    const DGraph* cpuThenChild = findChildDGraph(*thenChild, "cpu");
    const DGraph* cpuElseChild = findChildDGraph(*elseChild, "cpu");
    const DGraph* mockElseChild = findChildDGraph(*elseChild, "mcpu:0");
    ASSERT_NE(cpuThenChild, nullptr);
    ASSERT_NE(cpuElseChild, nullptr);
    ASSERT_NE(mockElseChild, nullptr);

    EXPECT_EQ(findBridgeNode(*cpuThenChild, CompiledBridgeOpNode::Side::Consumer,
                             conditionalId), nullptr);
    const auto* producerBridge = findBridgeNode(
        *mockElseChild, CompiledBridgeOpNode::Side::Producer, elseProducerId);
    const auto* consumerBridge = findBridgeNode(
        *cpuElseChild, CompiledBridgeOpNode::Side::Consumer, conditionalId);
    ASSERT_NE(producerBridge, nullptr);
    ASSERT_NE(consumerBridge, nullptr);
    EXPECT_TRUE(dependsOn(*producerBridge, elseProducerId));
}

// A loop publishes a buffer at its CPU placement device; a parent kernel on
// `mcpu:0` reads that published buffer. The compiler must route a `cpu ->
// mcpu:0` bridge whose producer-side hangs after the control op in the CPU
// DGraph and whose consumer-side hangs before the kernel in the MOCK_CPU
// DGraph. The previous kernel-only producer check rejected this with a
// generic "not implemented yet" error.
TEST(RegionCompilerTest, ParentKernelReadsControlOutputAcrossDevices) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    addOutputKernel(*body, cpuKernel("body", singleOutputType()), BufferType::I32, "cpu");

    LoopSpec loopSpec;
    loopSpec.ioType = singleOutputType();
    GraphBuffer loopOutput = bindControlOutput(loopSpec.ioMap, graph.rootRegion());
    loopSpec.tripCount = LoopTripCount::constant<int32_t>(1);
    loopSpec.body = body;
    std::string loopId = graph.addLoop(std::move(loopSpec));

    IOTypeMap consumerType;
    consumerType.inputBuffers.push_back({"in", BufferType::I32});
    IOMap consumerIo;
    consumerIo.bindInputBuffer("in", loopOutput);
    std::string consumerId = graph.addNode(mockCpuKernel("remote_consume", consumerType),
                                           std::move(consumerIo), "mcpu:0");

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);

    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    const DGraph* mockDGraph = findDGraph(dgraphs, "mcpu:0");
    ASSERT_NE(cpuDGraph, nullptr);
    ASSERT_NE(mockDGraph, nullptr);

    const auto* producerBridge = findBridgeNode(
        *cpuDGraph, CompiledBridgeOpNode::Side::Producer, loopId);
    const auto* consumerBridge = findBridgeNode(
        *mockDGraph, CompiledBridgeOpNode::Side::Consumer, consumerId);
    ASSERT_NE(producerBridge, nullptr)
        << "no producer-side bridge after control op '" << loopId << "' on cpu";
    ASSERT_NE(consumerBridge, nullptr)
        << "no consumer-side bridge before kernel '" << consumerId << "' on mcpu:0";
    EXPECT_TRUE(dependsOn(*producerBridge, loopId));

    const CompiledNode* consumerNode = findCompiledNode(*mockDGraph, consumerId);
    ASSERT_NE(consumerNode, nullptr);
    EXPECT_TRUE(dependsOn(*consumerNode, consumerBridge->id));
}

TEST(RegionCompilerTest, GraphRunExecutesEmptyStructuredControlOnCpu) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    graph.compile(); EXPECT_NO_THROW(graph.run());
}

TEST(RegionCompilerTest, GraphRunCarriesLoopBufferStateAcrossIterations) {
    Graph graph;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    graph.registerDevice(cpu);

    auto initKernel = std::make_shared<AddI32BufferKernel>("init", 10);
    auto advanceKernel = std::make_shared<AddI32BufferKernel>("advance", 1);
    auto reportKernel = std::make_shared<AddI32BufferKernel>("report", 100);
    cpu->registerKernel(initKernel);
    cpu->registerKernel(advanceKernel);
    cpu->registerKernel(reportKernel);

    GraphBuffer raw = graph.inputBuffer(BufferType::I32, "raw");

    IOMap initIo;
    GraphBuffer state;
    initIo.bindInputBuffer("in", raw)
          .bindOutputBuffer("out", BufferType::I32, state);
    graph.addNode(initKernel->descriptor(), std::move(initIo), "cpu");

    auto body = graph.rootRegion().createChild();
    GraphBuffer localState = body->inputBuffer(BufferType::I32, "state");
    const std::string startId = body->importFromParent(
        std::vector<BufferBoundaryMapping>{{state, localState}});

    IOMap advanceIo;
    GraphBuffer localNext;
    advanceIo.bindInputBuffer("in", localState)
             .bindOutputBuffer("out", BufferType::I32, localNext, body->scopeId());
    const std::string advanceId = body->addKernel(advanceKernel->descriptor(),
                                                  std::move(advanceIo), "cpu", {startId});
    body->exportToParent(std::vector<BufferBoundaryMapping>{{localNext, state}},
                         {advanceId});

    const std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(3), body));

    IOMap reportIo;
    GraphBuffer finalOut;
    reportIo.bindInputBuffer("in", state)
            .bindOutputBuffer("out", BufferType::I32, finalOut);
    graph.addNode(reportKernel->descriptor(), std::move(reportIo), "cpu", {loopId});

    const std::vector<std::int32_t> input = {0, 1, 2, 3};
    cpu->setInputBuffer(raw.name(), input.data(), input.size() * sizeof(input[0]));

    ASSERT_NO_THROW(graph.compile());
    ASSERT_NO_THROW(graph.run());

    std::vector<std::int32_t> output(input.size(), 0);
    cpu->getOutputBuffer(finalOut.name(), output.data(), output.size() * sizeof(output[0]));
    EXPECT_EQ(output, (std::vector<std::int32_t>{113, 114, 115, 116}));
}

TEST(RegionCompilerTest, CompilerRejectsDirectParentTokenUseInsideNestedRegion) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphBuffer rootInput = graph.inputBuffer(BufferType::I32, "raw");

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});
    kernelType.outputBuffers.push_back({"out", BufferType::I32});

    IOMap bodyIo;
    GraphBuffer bodyOutput;
    bodyIo.bindInputBuffer("in", rootInput)
          .bindOutputBuffer("out", BufferType::I32, bodyOutput, body->scopeId());

    KernelDescriptor kernel{"copy", DeviceType::CPU, std::nullopt, kernelType};
    body->addKernel(std::move(kernel), std::move(bodyIo), "cpu");
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(graph.compile(), std::runtime_error);
}

TEST(RegionCompilerTest, GraphValidationRejectsUndeclaredRootScalarInCondition) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();
    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, "missing", graph.rootRegion().scopeId()),
        ConditionOperand::constant<int32_t>(1));
    graph.addConditional(ifElseSpec(std::move(condition), thenRegion, elseRegion));

    EXPECT_THROW(graph.compile(), std::runtime_error);
}

TEST(RegionCompilerTest, GraphValidationRejectsUndeclaredRootScalarInBoundaryMapping) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    GraphScalar missingParent = GraphScalar::globalVar(ScalarType::I32, "missing",
                                                       graph.rootRegion().scopeId());
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    body->importFromParent({{missingParent, localCounter}});
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    EXPECT_THROW(graph.compile(), std::runtime_error);
}

TEST(RegionCompilerTest, ControlNodeRoutedToCpuDeviceWhenBodyIsRemote) {
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("cpu", DeviceType::CPU));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    std::string mockKernelId = body->addKernel(mockCpuKernel("body_remote"), IOMap{}, "mcpu:0");
    std::string loopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    InspectionBridge bridge;
    auto dgraphs = compileForInspection(graph, bridge);

    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const CompiledNode* loopNode = findCompiledNode(*cpuDGraph, loopId);
    ASSERT_NE(loopNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledLoopNode>(*loopNode));
    EXPECT_EQ(std::get<CompiledLoopNode>(*loopNode).deviceId, "cpu");

    const DGraphChild* bodyChild = findChildDGraphs(*cpuDGraph, loopId,
                                                   DGraphChildRole::LoopBody);
    ASSERT_NE(bodyChild, nullptr);
    const DGraph* mockChild = findChildDGraph(*bodyChild, "mcpu:0");
    ASSERT_NE(mockChild, nullptr);
    EXPECT_NE(findCompiledNode(*mockChild, mockKernelId), nullptr);

    const DGraph* mockTopDGraph = findDGraph(dgraphs, "mcpu:0");
    if (mockTopDGraph) {
        EXPECT_EQ(findCompiledNode(*mockTopDGraph, loopId), nullptr);
    }
}

TEST(RegionCompilerTest, ControlNodeWithoutCpuDeviceFails) {
    // A graph with only a non-CPU device cannot compile: the bridge-factory
    // check fires because every non-CPU device requires {CPU, T}/{T, CPU}
    // factories, and those factories cannot exist without a CPU device.
    // This subsumes the original "control node requires a CPU device" error.
    Graph graph;
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    graph.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(1), body));

    try {
        compileForInspection(graph);
        FAIL() << "expected compileRegion to throw because no CPU device is registered";
    } catch (const std::runtime_error& ex) {
        EXPECT_NE(std::string(ex.what()).find("bridge factories"), std::string::npos)
            << "actual: " << ex.what();
    }
}

TEST(RegionCompilerTest, GraphValidationRejectsUndeclaredRootInputBuffer) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    IOTypeMap kernelType;
    kernelType.inputBuffers.push_back({"in", BufferType::I32});

    GraphBuffer undeclared = GraphBuffer::make(BufferType::I32, "missing_input",
                                               graph.rootRegion().scopeId());
    IOMap io;
    io.bindInputBuffer("in", undeclared);
    graph.addNode(cpuKernel("consume", kernelType), std::move(io), "cpu");

    EXPECT_THROW(graph.compile(), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerRejectsZeroConstantTripCountWithOutputs) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    addOutputKernel(*body, cpuKernel("body_out", singleOutputType()), BufferType::I32, "cpu");

    IOTypeMap loopType = singleOutputType();
    IOMap loopIo;
    bindControlOutput(loopIo, graph.rootRegion());
    graph.addLoop(fixedLoopSpec(std::move(loopType), std::move(loopIo),
                                LoopTripCount::constant<int32_t>(0), body));

    EXPECT_THROW(compileForInspection(graph), std::runtime_error);
}

TEST(RegionCompilerTest, CompilerLowersNestedLoops) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto outerBody = graph.rootRegion().createChild();
    auto innerBody = outerBody->createChild();
    std::string innerKernelId = innerBody->addKernel(cpuKernel("inner"), IOMap{}, "cpu");
    std::string innerLoopId = outerBody->addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), innerBody));
    std::string outerLoopId = graph.addLoop(
        fixedLoopSpec(LoopTripCount::constant<int32_t>(1), outerBody));

    auto dgraphs = compileForInspection(graph);
    const DGraph* cpuDGraph = findDGraph(dgraphs, "cpu");
    ASSERT_NE(cpuDGraph, nullptr);

    const CompiledNode* outerLoopNode = findCompiledNode(*cpuDGraph, outerLoopId);
    ASSERT_NE(outerLoopNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledLoopNode>(*outerLoopNode));

    const DGraphChild* outerBodyChild = findChildDGraphs(*cpuDGraph, outerLoopId,
                                                         DGraphChildRole::LoopBody);
    ASSERT_NE(outerBodyChild, nullptr);
    ASSERT_EQ(outerBodyChild->dgraphs.size(), 1u);
    const DGraph* outerBodyDGraph = outerBodyChild->dgraphs.front().get();
    ASSERT_NE(outerBodyDGraph, nullptr);

    const CompiledNode* innerLoopNode = findCompiledNode(*outerBodyDGraph, innerLoopId);
    ASSERT_NE(innerLoopNode, nullptr);
    ASSERT_TRUE(std::holds_alternative<CompiledLoopNode>(*innerLoopNode));

    const DGraphChild* innerBodyChild = findChildDGraphs(*outerBodyDGraph, innerLoopId,
                                                         DGraphChildRole::LoopBody);
    ASSERT_NE(innerBodyChild, nullptr);
    ASSERT_EQ(innerBodyChild->dgraphs.size(), 1u);
    const DGraph* innerBodyDGraph = innerBodyChild->dgraphs.front().get();
    ASSERT_NE(innerBodyDGraph, nullptr);
    EXPECT_NE(findCompiledNode(*innerBodyDGraph, innerKernelId), nullptr);
}

TEST(RegionCompilerTest, GraphRunExecutesEmptyWhileLoopOnCpu) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto body = graph.rootRegion().createChild();
    graph.addLoop(whileLoopSpec(Condition::alwaysFalse(), body));

    graph.compile(); EXPECT_NO_THROW(graph.run());
}

TEST(RegionCompilerTest, GraphLaunchWithoutCompileThrows) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));
    auto body = graph.rootRegion().createChild();
    graph.addLoop(whileLoopSpec(Condition::alwaysFalse(), body));

    EXPECT_THROW(graph.launch(), std::runtime_error);
    EXPECT_THROW(graph.run(), std::runtime_error);
    EXPECT_THROW(graph.wait(), std::runtime_error);
}

TEST(RegionCompilerTest, GraphLaunchAfterStructuralMutationThrowsUntilRecompiled) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));
    auto body = graph.rootRegion().createChild();
    graph.addLoop(whileLoopSpec(Condition::alwaysFalse(), body));

    graph.compile();
    EXPECT_NO_THROW(graph.run());

    auto otherBody = graph.rootRegion().createChild();
    graph.addLoop(whileLoopSpec(Condition::alwaysFalse(), otherBody));
    EXPECT_THROW(graph.launch(), std::runtime_error);

    graph.compile();
    EXPECT_NO_THROW(graph.run());
}

TEST(RegionCompilerTest, GraphRegisterDeviceRejectsSecondCpu) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    EXPECT_THROW(
        graph.registerDevice(std::make_shared<CpuDevice>("cpu_2")),
        std::invalid_argument);

    Graph withDefaults = Graph::withDefaults();
    EXPECT_THROW(
        withDefaults.registerDevice(std::make_shared<CpuDevice>("another_cpu")),
        std::invalid_argument);
}

TEST(RegionCompilerTest, AddConditionalSpecWithoutConditionThrows) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));

    auto thenRegion = graph.rootRegion().createChild();
    auto elseRegion = graph.rootRegion().createChild();

    ConditionalSpec spec;
    spec.thenRegion = thenRegion;
    spec.elseRegion = elseRegion;

    EXPECT_THROW(graph.addConditional(std::move(spec)), std::invalid_argument);
}

TEST(RegionCompilerTest, CompileRejectsMissingBridgeFactoryForNonCpuDevice) {
    Graph graph;
    graph.registerDevice(std::make_shared<CpuDevice>("cpu"));
    graph.registerDevice(std::make_shared<StubDevice>("mcpu:0", DeviceType::MOCK_CPU));

    auto body = graph.rootRegion().createChild();
    graph.addLoop(whileLoopSpec(Condition::alwaysFalse(), body));

    try {
        graph.compile();
        FAIL() << "expected compile() to throw because the MOCK_CPU device has no bridge factories";
    } catch (const std::runtime_error& ex) {
        const std::string what = ex.what();
        EXPECT_NE(what.find("bridge factories"), std::string::npos) << "actual: " << what;
        EXPECT_NE(what.find("mcpu:0"), std::string::npos) << "actual: " << what;
    }
}

TEST(RegionCompilerTest, ScopedBufferKeyAlwaysIncludesScopePrefix) {
    EXPECT_EQ(scopedBufferKey(0, "x"), "scope:0:x");
    EXPECT_EQ(scopedBufferKey(1, "x"), "scope:1:x");
    EXPECT_EQ(scopedBufferKey(42, "buffer_anonymous"), "scope:42:buffer_anonymous");
}
