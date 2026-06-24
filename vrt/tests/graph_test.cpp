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
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <variant>
#include <vector>

#include <vrt/graph/graph.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/crossdevice/bridge_op.hpp>
#include <vrt/graph/crossdevice/semaphore_pool.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/graph_scalar.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/graph/node/compiled_node.hpp>

#include "test_support/control_specs.hpp"

using namespace vrt::graph;
using namespace vrt::graph::test_support;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

using TestKernelFn = std::function<void(const CpuKernelArgs&)>;

class TestCpuKernel : public CpuKernel {
   public:
    TestCpuKernel(std::string name, TestKernelFn fn, IOTypeMap ioType = {})
        : CpuKernel(std::move(name)), fn_(std::move(fn)), ioType_(std::move(ioType)) {}

    IOTypeMap ioTypeMap() const override { return ioType_; }
    void run(Args& args) override { fn_(args); }

   private:
    TestKernelFn fn_;
    IOTypeMap    ioType_;
};

static std::shared_ptr<CpuKernel> makeCpuKernel(std::string name,
                                                TestKernelFn fn,
                                                IOTypeMap ioType = {}) {
    return std::make_shared<TestCpuKernel>(std::move(name), std::move(fn), std::move(ioType));
}

TEST(GraphTest, WithDefaultsRegistersCpuAndKnownBridgeTypes) {
    Graph graph = Graph::withDefaults();

    auto cpu = graph.cpuDevice();
    ASSERT_NE(cpu, nullptr);
    EXPECT_EQ(cpu->id(), "cpu");

    EXPECT_TRUE(graph.hasBridgeFactory(DeviceType::CPU, DeviceType::FPGA));
    EXPECT_TRUE(graph.hasBridgeFactory(DeviceType::FPGA, DeviceType::CPU));
#if defined(VRT_HAS_GPU) && (VRT_HAS_GPU == 1)
    EXPECT_TRUE(graph.hasBridgeFactory(DeviceType::CPU, DeviceType::GPU));
    EXPECT_TRUE(graph.hasBridgeFactory(DeviceType::GPU, DeviceType::CPU));
#else
    EXPECT_FALSE(graph.hasBridgeFactory(DeviceType::CPU, DeviceType::GPU));
    EXPECT_FALSE(graph.hasBridgeFactory(DeviceType::GPU, DeviceType::CPU));
#endif

    cpu->registerKernel(makeCpuKernel("copy", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i];
        }
    }));

    GraphBuffer raw = graph.inputBuffer(BufferType::I32, "raw");

    IOMap io;
    GraphBuffer copied;
    io.bindInput("in", raw)
      .bindOutput("out", BufferType::I32, copied);
    graph.addNode(cpuKernel("copy"), std::move(io), "cpu");

    std::vector<int32_t> input = {7, 11, 13};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    ASSERT_NO_THROW(graph.compile().run());

    std::vector<int32_t> output(input.size(), 0);
    cpu->getOutputBuffer(copied.name(), output.data(), output.size() * sizeof(int32_t));
    EXPECT_EQ(output, input);
}

// ===========================================================================
// MockCpuDevice — threaded IDevice for cross-device testing
// ===========================================================================

class MockCpuDevice : public IDevice {
    class Plan;

   public:
    explicit MockCpuDevice(std::string id)
        : id_(std::move(id)) {}

    ~MockCpuDevice() override = default;

    void registerKernel(std::shared_ptr<CpuKernel> kernel) {
        if (!kernel) throw std::invalid_argument("MockCpuDevice: kernel must not be null");
        kernels_[kernel->name()] = std::move(kernel);
    }

    void setInputBuffer(const std::string& bufferName, const void* data, size_t sizeBytes) {
        auto& buf = buffers_[normalizeKey(bufferName)];
        buf.resize(sizeBytes);
        if (data && sizeBytes > 0) std::memcpy(buf.data(), data, sizeBytes);
    }

    void getOutputBuffer(const std::string& bufferName, void* data, size_t sizeBytes) const {
        auto it = buffers_.find(normalizeKey(bufferName));
        if (it == buffers_.end())
            throw std::runtime_error("MockCpuDevice: unknown buffer '" + bufferName + "'");
        std::memcpy(data, it->second.data(), std::min(sizeBytes, it->second.size()));
    }

    size_t bufferSize(const std::string& bufferName) const {
        auto it = buffers_.find(normalizeKey(bufferName));
        return (it == buffers_.end()) ? 0 : it->second.size();
    }

    // --- IDevice ---

    DeviceType  type() const override { return DeviceType::MOCK_CPU; }
    std::string id()   const override { return id_; }

    std::unique_ptr<IDevicePlan> compilePlan(const DGraph& dg) override;

   private:
    static std::string normalizeKey(const std::string& bufferName) {
        if (bufferName.rfind("scope:", 0) == 0) return bufferName;
        return scopedBufferKey(0, bufferName);
    }

    struct OpStep {
        std::function<bool()> tryReady;
        std::function<void()> action;
    };
    struct KernelStep { CompiledKernelNode node; };
    using Step = std::variant<OpStep, KernelStep>;

    class Plan : public IDevicePlan {
       public:
        Plan(MockCpuDevice& device, const DGraph& dg)
            : device_(device) {
            steps_.reserve(dg.nodes.size());
            for (const CompiledNode& node : dg.nodes) {
                std::visit(
                    [&](const auto& n) {
                        using T = std::decay_t<decltype(n)>;
                        if constexpr (std::is_same_v<T, CompiledKernelNode>) {
                            steps_.push_back(KernelStep{n});
                        } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
                            steps_.push_back(OpStep{n.tryReady, n.action});
                        } else {
                            throw std::runtime_error(
                                "MockCpuDevice: compiled control/boundary nodes are not executable yet");
                        }
                    },
                    node);
            }
        }

        ~Plan() override {
            try {
                wait();
            } catch (...) {
            }
        }

        void launch() override {
            if (worker_.joinable()) worker_.join();
            workerException_ = nullptr;
            worker_ = std::thread([this] {
                try {
                    for (const Step& step : steps_) {
                        if (std::holds_alternative<KernelStep>(step)) {
                            device_.executeKernel(std::get<KernelStep>(step).node);
                        } else {
                            const auto& op = std::get<OpStep>(step);
                            while (!op.tryReady()) {}
                            op.action();
                        }
                    }
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
        MockCpuDevice& device_;
        std::vector<Step> steps_;
        std::thread worker_;
        std::exception_ptr workerException_;
    };

    void executeKernel(const CompiledKernelNode& node) {
        auto it = kernels_.find(node.kernel.name);
        if (it == kernels_.end())
            throw std::runtime_error("MockCpuDevice: no kernel '" + node.kernel.name + "'");

        std::map<std::string, CpuBufferView> bufViews;

        for (const auto& [port, buf] : node.ioMap.inputs()) {
            CpuBufferView v = resolveBuffer(buf);
            v.elementType   = buf.type();
            bufViews[port]  = v;
        }

        size_t defaultSize = 0;
        if (!node.ioMap.inputs().empty()) {
            const GraphBuffer& firstBuffer = node.ioMap.inputs().begin()->second;
            auto fit = buffers_.find(bufferStorageKey(firstBuffer));
            if (fit != buffers_.end()) defaultSize = fit->second.size();
        }

        for (const auto& [port, buf] : node.ioMap.outputs()) {
            auto& storage = ensureBuffer(buf, defaultSize);
            bufViews[port] = CpuBufferView{storage.data(), storage.size(), buf.type()};
        }

        for (const auto& rw : node.ioMap.inouts()) {
            bufViews[rw.inPort] = resolveBuffer(rw.in);
            bufViews[rw.inPort].elementType = rw.in.type();
            auto& inStorage = buffers_.at(bufferStorageKey(rw.in));
            const std::string outKey = scopedBufferKey(rw.out.scopeId(), rw.out.name());
            buffers_[outKey] = inStorage;
            bufViews[rw.outPort] = CpuBufferView{
                buffers_[outKey].data(), buffers_[outKey].size(), rw.out.type()};
        }

        std::map<std::string, uint64_t> scalars;
        std::map<std::string, uint64_t*> writableScalars;
        for (const auto& [port, gs] : node.ioMap.outputScalars()) {
            if (gs.isConstant()) {
                throw std::runtime_error(
                    "MockCpuDevice: output scalar port '" + port +
                    "' cannot be bound to a constant");
            }
            writableScalars[port] =
                &scalarStore_[scopedScalarKey(gs.scopeId(), gs.varName())];
        }
        for (const auto& [port, gs] : node.ioMap.inputScalars()) {
            if (gs.isConstant()) {
                scalars[port] = gs.constantBits();
            } else {
                const std::string key = scopedScalarKey(gs.scopeId(), gs.varName());
                auto scalarIt = scalarStore_.find(key);
                if (scalarIt == scalarStore_.end() && gs.scopeId() == 0) {
                    scalarIt = scalarStore_.find(gs.varName());
                }
                if (scalarIt == scalarStore_.end()) {
                    throw std::runtime_error(
                        "MockCpuDevice: scalar '" + gs.varName() + "' not found");
                }
                scalars[port] = scalarIt->second;
            }
        }

        CpuKernelArgs args(std::move(bufViews), std::move(scalars),
                           std::move(writableScalars));
        it->second->run(args);
    }

    std::string bufferStorageKey(const GraphBuffer& buffer) const {
        const std::string key = scopedBufferKey(buffer.scopeId(), buffer.name());
        if (buffers_.count(key)) return key;
        if (buffer.scopeId() == 0 && buffers_.count(buffer.name())) return buffer.name();
        return key;
    }

    CpuBufferView resolveBuffer(const GraphBuffer& buffer) const {
        const std::string key = bufferStorageKey(buffer);
        auto it = buffers_.find(key);
        if (it == buffers_.end())
            throw std::runtime_error("MockCpuDevice: buffer '" + buffer.name() + "' not found");
        return CpuBufferView{
            const_cast<void*>(static_cast<const void*>(it->second.data())),
            it->second.size(), BufferType::U8};
    }

    std::vector<uint8_t>& ensureBuffer(const GraphBuffer& buffer, size_t sizeBytes) {
        auto& buf = buffers_[scopedBufferKey(buffer.scopeId(), buffer.name())];
        if (buf.size() < sizeBytes) buf.resize(sizeBytes);
        return buf;
    }

    std::string                                  id_;
    std::map<std::string, std::shared_ptr<CpuKernel>> kernels_;
    std::map<std::string, std::vector<uint8_t>>  buffers_;
    std::map<std::string, uint64_t>              scalarStore_;
};

std::unique_ptr<IDevicePlan> MockCpuDevice::compilePlan(const DGraph& dg) {
    return std::make_unique<Plan>(*this, dg);
}

// ===========================================================================
// Bridges — use SemaphorePool + host-staging, return BridgeStepPair so the
// compiler can splice them into the per-device DGraphs as CompiledBridgeOpNodes.
// ===========================================================================

namespace {

struct TestBridgeOp : IBridgeOp {
    SemaphorePool*       pool;
    SemaphoreHandle      sem;
    std::vector<uint8_t> staging;

    std::string label() const override { return "test_xfer"; }
};

struct TestBarrierOp : IBridgeOp {
    SemaphorePool*  pool;
    SemaphoreHandle sem;
    std::string     label() const override { return "barrier"; }
};

// Generic factory for any pair of "cpu-like" devices (CpuDevice or MockCpuDevice).
BridgeStepPair makeCpuLikeTransfer(SemaphorePool&     pool,
                                    IDevice&            src,
                                    IDevice&            dst,
                                    const GraphBuffer&  buffer) {
    auto op  = std::make_shared<TestBridgeOp>();
    op->pool = &pool;
    op->sem  = pool.allocate();

    const std::string bufName = scopedBufferKey(buffer.scopeId(), buffer.name());

    auto* srcCpu  = dynamic_cast<CpuDevice*>(&src);
    auto* srcMock = dynamic_cast<MockCpuDevice*>(&src);
    auto* dstCpu  = dynamic_cast<CpuDevice*>(&dst);
    auto* dstMock = dynamic_cast<MockCpuDevice*>(&dst);

    auto producerClosure = [op, srcCpu, srcMock, bufName]() {
        size_t sz = srcCpu  ? srcCpu->bufferSize(bufName)
                   : srcMock ? srcMock->bufferSize(bufName)
                   : 0;
        op->staging.resize(sz);
        if (sz > 0) {
            if (srcCpu)       srcCpu->getOutputBuffer(bufName, op->staging.data(), sz);
            else if (srcMock) srcMock->getOutputBuffer(bufName, op->staging.data(), sz);
        }
        op->pool->signal(op->sem);
    };

    auto tryReady = [op]() { return op->pool->tryAwait(op->sem); };
    auto consumerAction = [op, dstCpu, dstMock, bufName]() {
        if (dstCpu)       dstCpu->setInputBuffer(bufName, op->staging.data(), op->staging.size());
        else if (dstMock) dstMock->setInputBuffer(bufName, op->staging.data(), op->staging.size());
    };

    return BridgeStepPair{op,
                          std::move(producerClosure),
                          std::move(tryReady),
                          std::move(consumerAction)};
}

BridgeStepPair makeCpuLikeBarrier(SemaphorePool& pool) {
    auto op  = std::make_shared<TestBarrierOp>();
    op->pool = &pool;
    op->sem  = pool.allocate();
    auto producer = [op]() { op->pool->signal(op->sem); };
    auto tryReady = [op]() { return op->pool->tryAwait(op->sem); };
    auto consumer = []() {};
    return BridgeStepPair{op,
                          std::move(producer),
                          std::move(tryReady),
                          std::move(consumer)};
}

}  // namespace

class CpuMockCpuBridge : public IBridge {
   public:
    CpuMockCpuBridge(IDevice& /*src*/, IDevice& /*dst*/) {}

    BridgeStepPair makeTransfer(IDevice& src, IDevice& dst,
                                 const GraphBuffer& buffer, uint64_t /*sizeHintBytes*/,
                                 const std::string& /*producerNodeId*/,
                                 const std::string& /*consumerNodeId*/) override {
        return makeCpuLikeTransfer(pool_, src, dst, buffer);
    }

    BridgeStepPair makeBarrier(IDevice& /*src*/, IDevice& /*dst*/,
                                const std::string& /*producerNodeId*/,
                                const std::string& /*consumerNodeId*/) override {
        return makeCpuLikeBarrier(pool_);
    }

   private:
    SemaphorePool pool_;
};

class MockCpuMockCpuBridge : public IBridge {
   public:
    MockCpuMockCpuBridge(IDevice& /*src*/, IDevice& /*dst*/) {}

    BridgeStepPair makeTransfer(IDevice& src, IDevice& dst,
                                 const GraphBuffer& buffer, uint64_t /*sizeHintBytes*/,
                                 const std::string& /*producerNodeId*/,
                                 const std::string& /*consumerNodeId*/) override {
        return makeCpuLikeTransfer(pool_, src, dst, buffer);
    }

    BridgeStepPair makeBarrier(IDevice& /*src*/, IDevice& /*dst*/,
                                const std::string& /*producerNodeId*/,
                                const std::string& /*consumerNodeId*/) override {
        return makeCpuLikeBarrier(pool_);
    }

   private:
    SemaphorePool pool_;
};

// Helper: register both directions of a cpu-like bridge factory.
template <typename BridgeT>
static void registerCpuLikeFactory(Graph& g, DeviceType a, DeviceType b) {
    auto factory = [](IDevice& s, IDevice& d) -> std::shared_ptr<IBridge> {
        return std::make_shared<BridgeT>(s, d);
    };
    g.registerBridgeFactory(a, b, factory);
    if (a != b) g.registerBridgeFactory(b, a, factory);
}

// ---------------------------------------------------------------------------
// 3-node pipeline: add → double → negate
// ---------------------------------------------------------------------------

TEST(GraphTest, ThreeNodePipeline) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    // Register kernel implementations.
    cpu->registerKernel(makeCpuKernel("add", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        auto offset = static_cast<int32_t>(args.scalar("offset"));
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] + offset;
        }
    }));

    cpu->registerKernel(makeCpuKernel("dbl", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] * 2;
        }
    }));

    cpu->registerKernel(makeCpuKernel("neg", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = -in[i];
        }
    }));

    // Build graph.
    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // Node A: add offset=10
    IOMap ioA;
    GraphBuffer afterAdd;
    ioA.bindInput("in", raw)
       .bindOutput("out", BufferType::I32, afterAdd)
       .bindInputScalar("offset", GraphScalar::constant<int32_t>(10));
    g.addNode(cpuKernel("add"), std::move(ioA), "cpu");

    // Node B: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInput("in", afterAdd)
       .bindOutput("out", BufferType::I32, afterDbl);
    g.addNode(cpuKernel("dbl"), std::move(ioB), "cpu");

    // Node C: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInput("in", afterDbl)
       .bindOutput("out", BufferType::I32, afterNeg);
    g.addNode(cpuKernel("neg"), std::move(ioC), "cpu");

    // Provide input data.
    std::vector<int32_t> input = {1, 2, 3, 4};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    // Run.
    g.compile().run();

    // Read output: (x + 10) * 2 * (-1)
    std::vector<int32_t> output(4);
    cpu->getOutputBuffer(afterNeg.name(), output.data(), output.size() * sizeof(int32_t));

    EXPECT_EQ(output[0], -22);
    EXPECT_EQ(output[1], -24);
    EXPECT_EQ(output[2], -26);
    EXPECT_EQ(output[3], -28);
}

// ---------------------------------------------------------------------------
// Diamond dependency: A → B, A → C, B+C → D
// ---------------------------------------------------------------------------

TEST(GraphTest, DiamondDependency) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel(makeCpuKernel("split", [](const CpuKernelArgs& args) {
        auto in   = args.buffer("in").as<const int32_t>();
        auto outL = args.buffer("left").as<int32_t>();
        auto outR = args.buffer("right").as<int32_t>();
        auto n    = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            outL[i] = in[i] + 1;
            outR[i] = in[i] * 10;
        }
    }));

    cpu->registerKernel(makeCpuKernel("passL", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i];
    }));

    cpu->registerKernel(makeCpuKernel("passR", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i];
    }));

    cpu->registerKernel(makeCpuKernel("merge", [](const CpuKernelArgs& args) {
        auto left  = args.buffer("left").as<const int32_t>();
        auto right = args.buffer("right").as<const int32_t>();
        auto out   = args.buffer("out").as<int32_t>();
        auto n     = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = left[i] + right[i];
        }
    }));

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // A: split
    IOMap ioA;
    GraphBuffer leftBuf, rightBuf;
    ioA.bindInput("in", raw)
       .bindOutput("left", BufferType::I32, leftBuf)
       .bindOutput("right", BufferType::I32, rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B: pass left
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInput("in", leftBuf)
       .bindOutput("out", BufferType::I32, leftOut);
    g.addNode(cpuKernel("passL"), std::move(ioB), "cpu");

    // C: pass right
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInput("in", rightBuf)
       .bindOutput("out", BufferType::I32, rightOut);
    g.addNode(cpuKernel("passR"), std::move(ioC), "cpu");

    // D: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInput("left", leftOut)
       .bindInput("right", rightOut)
       .bindOutput("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("merge"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.compile().run();

    // left = x+1, right = x*10, merge = left+right = x+1+x*10 = 11x+1
    std::vector<int32_t> output(3);
    cpu->getOutputBuffer(finalBuf.name(), output.data(), output.size() * sizeof(int32_t));

    EXPECT_EQ(output[0], 12);   // 11*1+1
    EXPECT_EQ(output[1], 23);   // 11*2+1
    EXPECT_EQ(output[2], 34);   // 11*3+1
}

// ---------------------------------------------------------------------------
// Error cases
// ---------------------------------------------------------------------------

TEST(GraphTest, EmptyGraphThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);
    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, NoDeviceThrows) {
    Graph g;
    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    GraphBuffer out;
    io.bindInput("in", raw).bindOutput("out", BufferType::I32, out);
    g.addNode(cpuKernel("k"), std::move(io));
    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, MissingDeviceHintThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    GraphBuffer out;
    io.bindInput("in", raw).bindOutput("out", BufferType::I32, out);
    g.addNode(cpuKernel("k"), std::move(io), "nonexistent");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, DuplicateInputBufferBindThrows) {
    Graph g;
    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    IOMap io;
    io.bindInput("in", raw);
    EXPECT_THROW(io.bindInput("in", raw), std::invalid_argument);
}

TEST(GraphTest, MissingMandatoryInputBufferPortThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputs.push_back({"in", BufferType::I32});

    IOMap io;
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, UnknownInputBufferPortThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputs.push_back({"in", BufferType::I32});

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    io.bindInput("in", raw)
      .bindInput("extra", raw);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, InputBufferTypeMismatchThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputs.push_back({"in", BufferType::I32});

    GraphBuffer raw = g.inputBuffer(BufferType::U8, "raw");
    IOMap io;
    io.bindInput("in", raw);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, OutputScalarMustUseGlobalVar) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.outputScalars.push_back({"out", ScalarType::I32});

    IOMap io;
    io.bindOutputScalar("out", GraphScalar::constant<int32_t>(1));
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, InputScalarMustBeBoundAsInputScalar) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"n", ScalarType::I32});

    IOMap io;
    io.bindOutputScalar("n", GraphScalar::constant<int32_t>(3));
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, OutputScalarMustBeBoundAsOutputScalar) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.outputScalars.push_back({"out", ScalarType::I32});
    GraphScalar out = g.globalScalar(ScalarType::I32, "wrong_direction_out");

    IOMap io;
    io.bindInputScalar("out", out);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, MissingMandatoryInoutPortThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inouts.push_back(RWBufferPort{BufferPort{"data", BufferType::I32},
                                         BufferPort{"data", BufferType::I32}});

    IOMap io;
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, InoutBufferTypeMismatchThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inouts.push_back(RWBufferPort{BufferPort{"data", BufferType::I32},
                                         BufferPort{"data", BufferType::I32}});

    GraphBuffer raw = g.inputBuffer(BufferType::U8, "raw_inout_mismatch");
    IOMap io;
    GraphBuffer out;
    io.bindInout("data", "data", raw, out);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, InvalidAfterNodesReferenceThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOMap io;
    g.addNode(cpuKernel("typed"), std::move(io), "cpu", {"missing_node"});

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, CpuGlobalScalarRoundTrip) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", ScalarType::I32});
    ioType.outputScalars.push_back({"out", ScalarType::I32});

    cpu->registerKernel(makeCpuKernel("scalar_copy", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 1));
    }, ioType));

    GraphScalar input = g.globalScalar(ScalarType::I32, "input");
    GraphScalar output = g.globalScalar(ScalarType::I32, "output");

    IOMap io;
    io.bindInputScalar("in", input)
      .bindOutputScalar("out", output);
    g.addNode(cpuKernel("scalar_copy", ioType), std::move(io), "cpu");

    g.setScalar<int32_t>("input", 41);
    auto exec = g.compile();
    ASSERT_NO_THROW(exec.run());

    EXPECT_EQ(exec.getScalar<int32_t>("output"), 42);
}

TEST(GraphTest, CpuGlobalScalarUpdatesStayLiveAfterCompile) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", ScalarType::I32});
    ioType.outputScalars.push_back({"out", ScalarType::I32});

    cpu->registerKernel(makeCpuKernel("live_scalar_copy", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 1));
    }, ioType));

    GraphScalar input = g.globalScalar(ScalarType::I32, "live_input");
    GraphScalar output = g.globalScalar(ScalarType::I32, "live_output");

    IOMap io;
    io.bindInputScalar("in", input)
      .bindOutputScalar("out", output);
    g.addNode(cpuKernel("live_scalar_copy", ioType), std::move(io), "cpu");

    g.setScalar<int32_t>("live_input", 41);
    auto exec = g.compile();
    ASSERT_FALSE(exec.dgraphs().empty());

    g.setScalar<int32_t>("live_input", 200);
    ASSERT_NO_THROW(exec.run());
    EXPECT_EQ(exec.getScalar<int32_t>("live_output"), 42);

    exec.setScalar<int32_t>("live_input", 100);
    ASSERT_NO_THROW(exec.run());
    EXPECT_EQ(exec.getScalar<int32_t>("live_output"), 101);
}

TEST(GraphTest, CpuScalarDependencyOrdersNodes) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap producerType;
    producerType.outputScalars.push_back({"value", ScalarType::I32});
    IOTypeMap consumerType;
    consumerType.inputScalars.push_back({"value", ScalarType::I32});
    consumerType.outputScalars.push_back({"result", ScalarType::I32});

    cpu->registerKernel(makeCpuKernel("produce_scalar", [](const CpuKernelArgs& args) {
        args.setScalar("value", static_cast<uint64_t>(41));
    }, producerType));
    cpu->registerKernel(makeCpuKernel("consume_scalar", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("value"));
        args.setScalar("result", static_cast<uint64_t>(value + 1));
    }, consumerType));

    GraphScalar value = g.globalScalar(ScalarType::I32, "value");
    GraphScalar result = g.globalScalar(ScalarType::I32, "result");

    IOMap consumeIo;
    consumeIo.bindInputScalar("value", value)
             .bindOutputScalar("result", result);
    g.addNode(cpuKernel("consume_scalar", consumerType), std::move(consumeIo), "cpu");

    IOMap produceIo;
    produceIo.bindOutputScalar("value", value);
    g.addNode(cpuKernel("produce_scalar", producerType), std::move(produceIo), "cpu");

    auto exec = g.compile();
    ASSERT_NO_THROW(exec.run());
    EXPECT_EQ(exec.getScalar<int32_t>("result"), 42);
}

TEST(GraphTest, UndeclaredGlobalScalarThrows) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", ScalarType::I32});

    GraphScalar undeclared = GraphScalar::globalVar(ScalarType::I32, "missing");
    IOMap io;
    io.bindInputScalar("in", undeclared);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

// ===========================================================================
// Cross-device tests
// ===========================================================================

// Helper lambda factories (reused across devices)
static IOTypeMap i32BufferInOutType() {
    IOTypeMap ioType;
    ioType.inputs.push_back({"in", BufferType::I32});
    ioType.outputs.push_back({"out", BufferType::I32});
    return ioType;
}

static IOTypeMap i32BufferOutType() {
    IOTypeMap ioType;
    ioType.outputs.push_back({"out", BufferType::I32});
    return ioType;
}

static std::shared_ptr<CpuKernel> makeAddKernel(std::string name, int32_t offset) {
    return makeCpuKernel(std::move(name), [offset](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + offset;
    });
}

static std::shared_ptr<CpuKernel> makeDblKernel(std::string name) {
    return makeCpuKernel(std::move(name), [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * 2;
    });
}

static std::shared_ptr<CpuKernel> makeNegKernel(std::string name) {
    return makeCpuKernel(std::move(name), [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = -in[i];
    });
}

// ---------------------------------------------------------------------------
// Cross-device pipeline: cpu → mcpu:0 → mcpu:1 → cpu
// ---------------------------------------------------------------------------

TEST(GraphTest, CrossDevicePipeline) {
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    cpu->registerKernel(makeAddKernel("add10", 10));
    mcpu0->registerKernel(makeDblKernel("dbl"));
    mcpu1->registerKernel(makeNegKernel("neg"));
    cpu->registerKernel(makeAddKernel("add1", 1));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);
    registerCpuLikeFactory<MockCpuMockCpuBridge>(g, DeviceType::MOCK_CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // A on cpu: add 10
    IOMap ioA;
    GraphBuffer afterAdd;
    ioA.bindInput("in", raw).bindOutput("out", BufferType::I32, afterAdd);
    g.addNode(cpuKernel("add10"), std::move(ioA), "cpu");

    // B on mcpu:0: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInput("in", afterAdd).bindOutput("out", BufferType::I32, afterDbl);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInput("in", afterDbl).bindOutput("out", BufferType::I32, afterNeg);
    g.addNode(mockCpuKernel("neg"), std::move(ioC), "mcpu:1");

    // D on cpu: add 1
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInput("in", afterNeg).bindOutput("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("add1"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3, 4};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.compile().run();

    // ((x + 10) * 2 * (-1)) + 1
    std::vector<int32_t> output(4);
    cpu->getOutputBuffer(finalBuf.name(), output.data(), output.size() * sizeof(int32_t));

    EXPECT_EQ(output[0], -21);  // ((1+10)*2*-1)+1
    EXPECT_EQ(output[1], -23);  // ((2+10)*2*-1)+1
    EXPECT_EQ(output[2], -25);  // ((3+10)*2*-1)+1
    EXPECT_EQ(output[3], -27);  // ((4+10)*2*-1)+1
}

// ---------------------------------------------------------------------------
// Cross-device diamond: cpu splits, mcpu:0 + mcpu:1 process, cpu merges
// ---------------------------------------------------------------------------

TEST(GraphTest, CrossDeviceDiamond) {
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    cpu->registerKernel(makeCpuKernel("split", [](const CpuKernelArgs& args) {
        auto in = args.buffer("in").as<const int32_t>();
        auto l  = args.buffer("left").as<int32_t>();
        auto r  = args.buffer("right").as<int32_t>();
        auto n  = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) { l[i] = in[i] + 1; r[i] = in[i] * 10; }
    }));
    mcpu0->registerKernel(makeDblKernel("dbl"));
    mcpu1->registerKernel(makeDblKernel("dbl"));
    cpu->registerKernel(makeCpuKernel("merge", [](const CpuKernelArgs& args) {
        auto l   = args.buffer("left").as<const int32_t>();
        auto r   = args.buffer("right").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = l[i] + r[i];
    }));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);
    registerCpuLikeFactory<MockCpuMockCpuBridge>(g, DeviceType::MOCK_CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // A on cpu: split
    IOMap ioA;
    GraphBuffer leftBuf, rightBuf;
    ioA.bindInput("in", raw)
       .bindOutput("left", BufferType::I32, leftBuf)
       .bindOutput("right", BufferType::I32, rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B on mcpu:0: double the left branch
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInput("in", leftBuf).bindOutput("out", BufferType::I32, leftOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: double the right branch
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInput("in", rightBuf).bindOutput("out", BufferType::I32, rightOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioC), "mcpu:1");

    // D on cpu: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInput("left", leftOut)
       .bindInput("right", rightOut)
       .bindOutput("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("merge"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.compile().run();

    // left = (x+1)*2, right = (x*10)*2, merge = 2(x+1) + 2(10x) = 22x + 2
    std::vector<int32_t> output(3);
    cpu->getOutputBuffer(finalBuf.name(), output.data(), output.size() * sizeof(int32_t));

    EXPECT_EQ(output[0], 24);   // 22*1+2
    EXPECT_EQ(output[1], 46);   // 22*2+2
    EXPECT_EQ(output[2], 68);   // 22*3+2
}

TEST(GraphTest, CpuLoopRunsCrossDeviceChildBody) {
    auto cpu = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    cpu->registerKernel(makeAddKernel("loop_add1", 1));
    mcpu->registerKernel(makeDblKernel("loop_dbl"));
    cpu->registerKernel(makeAddKernel("loop_add3", 3));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    IOTypeMap ioType = i32BufferInOutType();
    GraphBuffer state = g.inputBuffer(BufferType::I32, "cross_device_loop_state");
    auto body = g.rootRegion().createChild();
    GraphBuffer localState = body->inputBuffer(BufferType::I32, "state");
    body->importFromParent(std::vector<BufferBoundaryMapping>{{state, localState}});

    IOMap addIo;
    GraphBuffer cpuStage;
    addIo.bindInput("in", localState)
         .bindOutput("out", BufferType::I32, cpuStage, body->scopeId());
    body->addKernel(cpuKernel("loop_add1", ioType), std::move(addIo), "cpu");

    IOMap doubleIo;
    GraphBuffer mockStage;
    doubleIo.bindInput("in", cpuStage)
            .bindOutput("out", BufferType::I32, mockStage, body->scopeId());
    body->addKernel(mockCpuKernel("loop_dbl", ioType), std::move(doubleIo), "mcpu:0");

    IOMap finishIo;
    GraphBuffer localNext;
    finishIo.bindInput("in", mockStage)
            .bindOutput("out", BufferType::I32, localNext, body->scopeId());
    body->addKernel(cpuKernel("loop_add3", ioType), std::move(finishIo), "cpu");
    body->exportToParent(std::vector<BufferBoundaryMapping>{{localNext, state}});

    g.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(3), body));

    std::vector<int32_t> input = {1, 2, 4};
    cpu->setInputBuffer(state.name(), input.data(), input.size() * sizeof(int32_t));

    auto exec = g.compile();
    ASSERT_NO_THROW(exec.run());

    std::vector<int32_t> output(input.size(), 0);
    cpu->getOutputBuffer(state.name(), output.data(), output.size() * sizeof(int32_t));
    EXPECT_EQ(output, (std::vector<int32_t>{43, 51, 67}));
}

TEST(GraphTest, CpuConditionalRunsSelectedCrossDeviceChildBranch) {
    auto cpu = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    auto thenCpuCalls = std::make_shared<int32_t>(0);
    auto thenMockCalls = std::make_shared<int32_t>(0);
    auto thenFinishCalls = std::make_shared<int32_t>(0);
    auto elseCalls = std::make_shared<int32_t>(0);
    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeCpuKernel("cond_then_add1", [thenCpuCalls](const CpuKernelArgs& args) {
        ++*thenCpuCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1;
    }, ioType));
    mcpu->registerKernel(makeCpuKernel("cond_then_dbl", [thenMockCalls](const CpuKernelArgs& args) {
        ++*thenMockCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * 2;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("cond_then_add3", [thenFinishCalls](const CpuKernelArgs& args) {
        ++*thenFinishCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 3;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("cond_else_sub5", [elseCalls](const CpuKernelArgs& args) {
        ++*elseCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] - 5;
    }, ioType));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphScalar flag = g.globalScalar(ScalarType::I32, "cross_cond_flag");
    GraphBuffer source = g.inputBuffer(BufferType::I32, "cross_cond_source");
    GraphBuffer result = g.inputBuffer(BufferType::I32, "cross_cond_result");

    auto thenRegion = g.rootRegion().createChild();
    GraphBuffer thenInput = thenRegion->inputBuffer(BufferType::I32, "input");
    thenRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, thenInput}});

    IOMap thenAddIo;
    GraphBuffer thenCpuStage;
    thenAddIo.bindInput("in", thenInput)
             .bindOutput("out", BufferType::I32, thenCpuStage,
                               thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("cond_then_add1", ioType), std::move(thenAddIo), "cpu");

    IOMap thenDoubleIo;
    GraphBuffer thenMockStage;
    thenDoubleIo.bindInput("in", thenCpuStage)
                .bindOutput("out", BufferType::I32, thenMockStage,
                                  thenRegion->scopeId());
    thenRegion->addKernel(mockCpuKernel("cond_then_dbl", ioType),
                          std::move(thenDoubleIo), "mcpu:0");

    IOMap thenFinishIo;
    GraphBuffer thenOutput;
    thenFinishIo.bindInput("in", thenMockStage)
                .bindOutput("out", BufferType::I32, thenOutput,
                                  thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("cond_then_add3", ioType),
                          std::move(thenFinishIo), "cpu");
    thenRegion->exportToParent(std::vector<BufferBoundaryMapping>{{thenOutput, result}});

    auto elseRegion = g.rootRegion().createChild();
    GraphBuffer elseInput = elseRegion->inputBuffer(BufferType::I32, "input");
    elseRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, elseInput}});
    IOMap elseIo;
    GraphBuffer elseOutput;
    elseIo.bindInput("in", elseInput)
          .bindOutput("out", BufferType::I32, elseOutput, elseRegion->scopeId());
    elseRegion->addKernel(cpuKernel("cond_else_sub5", ioType), std::move(elseIo), "cpu");
    elseRegion->exportToParent(std::vector<BufferBoundaryMapping>{{elseOutput, result}});

    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    g.addConditional(ifElseSpec(std::move(condition), thenRegion, elseRegion));

    auto runBranch = [&](int32_t branchFlag, std::vector<int32_t> input,
                         std::vector<int32_t> expected) {
        cpu->setInputBuffer(source.name(), input.data(), input.size() * sizeof(int32_t));
        g.setScalar<int32_t>(flag.varName(), branchFlag);
        ASSERT_NO_THROW(g.compile().run());
        std::vector<int32_t> output(input.size(), 0);
        cpu->getOutputBuffer(result.name(), output.data(), output.size() * sizeof(int32_t));
        EXPECT_EQ(output, expected);
    };

    runBranch(1, {1, 2}, {7, 9});
    EXPECT_EQ(*thenCpuCalls, 1);
    EXPECT_EQ(*thenMockCalls, 1);
    EXPECT_EQ(*thenFinishCalls, 1);
    EXPECT_EQ(*elseCalls, 0);

    runBranch(0, {10, 20}, {5, 15});
    EXPECT_EQ(*thenCpuCalls, 1);
    EXPECT_EQ(*thenMockCalls, 1);
    EXPECT_EQ(*thenFinishCalls, 1);
    EXPECT_EQ(*elseCalls, 1);

    runBranch(1, {3, 4}, {11, 13});
    EXPECT_EQ(*thenCpuCalls, 2);
    EXPECT_EQ(*thenMockCalls, 2);
    EXPECT_EQ(*thenFinishCalls, 2);
    EXPECT_EQ(*elseCalls, 1);
}

TEST(GraphTest, CpuLoopPublishesRemoteBufferOutputToCpuParentConsumer) {
    auto cpu = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    IOTypeMap stageType;
    stageType.inputs.push_back({"in", BufferType::I32});
    stageType.outputs.push_back({"stage", BufferType::I32});
    cpu->registerKernel(makeCpuKernel("phase7c_loop_add1", [](const CpuKernelArgs& args) {
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("stage").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1;
    }, stageType));
    mcpu->registerKernel(makeDblKernel("phase7c_loop_dbl"));
    cpu->registerKernel(makeAddKernel("phase7c_loop_sink", 3));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    IOTypeMap ioType = i32BufferInOutType();
    GraphBuffer source = g.inputBuffer(BufferType::I32, "phase7c_loop_source");
    auto body = g.rootRegion().createChild();
    GraphBuffer bodyInput = body->inputBuffer(BufferType::I32, "input");
    body->importFromParent(std::vector<BufferBoundaryMapping>{{source, bodyInput}});

    IOMap cpuStageIo;
    GraphBuffer cpuStage;
    cpuStageIo.bindInput("in", bodyInput)
              .bindOutput("stage", BufferType::I32, cpuStage, body->scopeId());
    body->addKernel(cpuKernel("phase7c_loop_add1", stageType), std::move(cpuStageIo), "cpu");

    IOMap remoteIo;
    GraphBuffer remoteOutput;
    remoteIo.bindInput("in", cpuStage)
            .bindOutput("out", BufferType::I32, remoteOutput, body->scopeId());
    body->addKernel(mockCpuKernel("phase7c_loop_dbl", ioType), std::move(remoteIo), "mcpu:0");

    LoopSpec loopSpec;
    loopSpec.ioType = i32BufferOutType();
    GraphBuffer loopOutput;
    loopSpec.ioMap.bindOutput("out", BufferType::I32, loopOutput,
                                    g.rootRegion().scopeId());
    loopSpec.tripCount = LoopTripCount::constant<int32_t>(1);
    loopSpec.body = body;
    loopSpec.outputPlacement.buffers["out"] = "cpu";
    g.addLoop(std::move(loopSpec));

    IOMap sinkIo;
    GraphBuffer finalOutput;
    sinkIo.bindInput("in", loopOutput)
          .bindOutput("out", BufferType::I32, finalOutput,
                            g.rootRegion().scopeId());
    g.addNode(cpuKernel("phase7c_loop_sink", ioType), std::move(sinkIo), "cpu");

    auto runCase = [&](std::vector<int32_t> input, std::vector<int32_t> expected) {
        cpu->setInputBuffer(source.name(), input.data(), input.size() * sizeof(int32_t));
        ASSERT_NO_THROW(g.compile().run());
        std::vector<int32_t> output(input.size(), 0);
        cpu->getOutputBuffer(finalOutput.name(), output.data(),
                             output.size() * sizeof(int32_t));
        EXPECT_EQ(output, expected);
    };

    runCase({1, 2}, {7, 9});
    runCase({10, 20}, {25, 45});
    runCase({3, 5, 8}, {11, 15, 21});
}

TEST(GraphTest, CpuConditionalPublishesRemoteBufferOutputToCpuParentConsumer) {
    auto cpu = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    auto thenCalls = std::make_shared<int32_t>(0);
    auto elseCpuCalls = std::make_shared<int32_t>(0);
    auto elseMockCalls = std::make_shared<int32_t>(0);
    auto sinkCalls = std::make_shared<int32_t>(0);
    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeCpuKernel("phase7c_then_add10", [thenCalls](const CpuKernelArgs& args) {
        ++*thenCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 10;
    }, ioType));
    IOTypeMap elseStageType;
    elseStageType.inputs.push_back({"in", BufferType::I32});
    elseStageType.outputs.push_back({"stage", BufferType::I32});
    cpu->registerKernel(makeCpuKernel("phase7c_else_add1", [elseCpuCalls](const CpuKernelArgs& args) {
        ++*elseCpuCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("stage").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1;
    }, elseStageType));
    mcpu->registerKernel(makeCpuKernel("phase7c_else_dbl", [elseMockCalls](const CpuKernelArgs& args) {
        ++*elseMockCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * 2;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("phase7c_cond_sink", [sinkCalls](const CpuKernelArgs& args) {
        ++*sinkCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 3;
    }, ioType));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphScalar flag = g.globalScalar(ScalarType::I32, "phase7c_cond_flag");
    GraphBuffer source = g.inputBuffer(BufferType::I32, "phase7c_cond_source");

    auto thenRegion = g.rootRegion().createChild();
    GraphBuffer thenInput = thenRegion->inputBuffer(BufferType::I32, "input");
    thenRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, thenInput}});
    IOMap thenIo;
    GraphBuffer thenOutput;
    thenIo.bindInput("in", thenInput)
          .bindOutput("out", BufferType::I32, thenOutput,
                            thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("phase7c_then_add10", ioType),
                          std::move(thenIo), "cpu");

    auto elseRegion = g.rootRegion().createChild();
    GraphBuffer elseInput = elseRegion->inputBuffer(BufferType::I32, "input");
    elseRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, elseInput}});
    IOMap elseCpuIo;
    GraphBuffer elseStage;
    elseCpuIo.bindInput("in", elseInput)
             .bindOutput("stage", BufferType::I32, elseStage,
                               elseRegion->scopeId());
    elseRegion->addKernel(cpuKernel("phase7c_else_add1", elseStageType),
                          std::move(elseCpuIo), "cpu");
    IOMap elseMockIo;
    GraphBuffer elseOutput;
    elseMockIo.bindInput("in", elseStage)
              .bindOutput("out", BufferType::I32, elseOutput,
                                elseRegion->scopeId());
    elseRegion->addKernel(mockCpuKernel("phase7c_else_dbl", ioType),
                          std::move(elseMockIo), "mcpu:0");

    ConditionalSpec conditionalSpec;
    conditionalSpec.ioType = i32BufferOutType();
    GraphBuffer conditionalOutput;
    conditionalSpec.ioMap.bindOutput("out", BufferType::I32, conditionalOutput,
                                           g.rootRegion().scopeId());
    conditionalSpec.condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    conditionalSpec.thenRegion = thenRegion;
    conditionalSpec.elseRegion = elseRegion;
    conditionalSpec.outputPlacement.buffers["out"] = "cpu";
    g.addConditional(std::move(conditionalSpec));

    IOMap sinkIo;
    GraphBuffer finalOutput;
    sinkIo.bindInput("in", conditionalOutput)
          .bindOutput("out", BufferType::I32, finalOutput,
                            g.rootRegion().scopeId());
    g.addNode(cpuKernel("phase7c_cond_sink", ioType), std::move(sinkIo), "cpu");

    auto runCase = [&](int32_t branchFlag,
                       std::vector<int32_t> input,
                       std::vector<int32_t> expected) {
        cpu->setInputBuffer(source.name(), input.data(), input.size() * sizeof(int32_t));
        g.setScalar<int32_t>(flag.varName(), branchFlag);
        ASSERT_NO_THROW(g.compile().run());
        std::vector<int32_t> output(input.size(), 0);
        cpu->getOutputBuffer(finalOutput.name(), output.data(),
                             output.size() * sizeof(int32_t));
        EXPECT_EQ(output, expected);
    };

    runCase(1, {1, 2}, {14, 15});
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCpuCalls, 0);
    EXPECT_EQ(*elseMockCalls, 0);
    EXPECT_EQ(*sinkCalls, 1);

    runCase(0, {2, 4}, {9, 13});
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCpuCalls, 1);
    EXPECT_EQ(*elseMockCalls, 1);
    EXPECT_EQ(*sinkCalls, 2);

    runCase(1, {5}, {18});
    EXPECT_EQ(*thenCalls, 2);
    EXPECT_EQ(*elseCpuCalls, 1);
    EXPECT_EQ(*elseMockCalls, 1);
    EXPECT_EQ(*sinkCalls, 3);
}

// ===========================================================================
// Phase-1: compiler populates Node::dependsOn on every DGraph node.
// ===========================================================================

namespace {

const DGraph* findDg(const std::vector<DGraph>& dgs, const std::string& id) {
    for (const auto& dg : dgs) if (dg.deviceId == id) return &dg;
    return nullptr;
}

const CompiledNode* findNode(const DGraph& dg, const std::string& id) {
    for (const auto& n : dg.nodes) if (compiledNodeId(n) == id) return &n;
    return nullptr;
}

bool depsContain(const CompiledNode& n, const std::string& id) {
    const auto& d = compiledNodeDependsOn(n);
    return std::find(d.begin(), d.end(), id) != d.end();
}

const DGraphChild* findChildDGraphs(const DGraph& dg,
                                    const std::string& parentNodeId,
                                    DGraphChildRole role) {
    for (const auto& child : dg.childDGraphs) {
        if (child.parentNodeId == parentNodeId && child.role == role) return &child;
    }
    return nullptr;
}

const DGraph* findChildDg(const DGraphChild& child, const std::string& deviceId) {
    for (const auto& dg : child.dgraphs) {
        if (dg && dg->deviceId == deviceId) return dg.get();
    }
    return nullptr;
}

}  // namespace

TEST(GraphTest, CompilerPopulatesDependsOnAcrossDevices) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    cpu ->registerKernel(makeAddKernel("add10", 10));
    mcpu->registerKernel(makeDblKernel("dbl"));
    cpu ->registerKernel(makeAddKernel("add1", 1));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2, b3;

    IOMap m1; m1.bindInput("in", raw).bindOutput("out", BufferType::I32, b1);
    auto idA = g.addNode(cpuKernel("add10"), std::move(m1), "cpu");

    IOMap m2; m2.bindInput("in", b1).bindOutput("out", BufferType::I32, b2);
    auto idB = g.addNode(mockCpuKernel("dbl"), std::move(m2), "mcpu:0");

    IOMap m3; m3.bindInput("in", b2).bindOutput("out", BufferType::I32, b3);
    auto idC = g.addNode(cpuKernel("add1"), std::move(m3), "cpu");

    std::vector<int32_t> in = {1};
    cpu->setInputBuffer("raw", in.data(), sizeof(int32_t));
    auto exec = g.compile();
    exec.run();

    const auto* dgCpu = findDg(exec.dgraphs(), "cpu");
    const auto* dgM   = findDg(exec.dgraphs(), "mcpu:0");
    ASSERT_NE(dgCpu, nullptr);
    ASSERT_NE(dgM,   nullptr);

    // Kernel A (cpu) is a graph-level input → no predecessors.
    const CompiledNode* nA = findNode(*dgCpu, idA);
    ASSERT_NE(nA, nullptr);
    EXPECT_TRUE(compiledNodeDependsOn(*nA).empty());

    // Kernel B (mcpu:0) consumes b1 produced by A (cpu) → its dep must be the
    // consumer-side bridge op anchored to B, NOT idA itself.
    const CompiledNode* nB = findNode(*dgM, idB);
    ASSERT_NE(nB, nullptr);
    EXPECT_FALSE(depsContain(*nB, idA));
    bool foundConsumerForB = false;
    for (const auto& depId : compiledNodeDependsOn(*nB)) {
        const CompiledNode* dep = findNode(*dgM, depId);
        ASSERT_NE(dep, nullptr) << "dependsOn references unknown id " << depId;
        if (std::holds_alternative<CompiledBridgeOpNode>(*dep)) {
            const auto& b = std::get<CompiledBridgeOpNode>(*dep);
            EXPECT_EQ(b.side, CompiledBridgeOpNode::Side::Consumer);
            EXPECT_EQ(b.pairedKernelId, idB);
            foundConsumerForB = true;
        }
    }
    EXPECT_TRUE(foundConsumerForB);

    // Producer-side bridge op on cpu (anchored to A) → dependsOn must
    // contain exactly { idA }.
    bool foundProducerForA = false;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<CompiledBridgeOpNode>(n)) continue;
        const auto& b = std::get<CompiledBridgeOpNode>(n);
        if (b.side == CompiledBridgeOpNode::Side::Producer && b.pairedKernelId == idA) {
            foundProducerForA = true;
            ASSERT_EQ(b.dependsOn.size(), 1u);
            EXPECT_EQ(b.dependsOn[0], idA);
        }
    }
    EXPECT_TRUE(foundProducerForA);

    // Kernel C (cpu) consumes b2 produced by B (mcpu:0) → depends on the
    // consumer-side bridge op on cpu anchored to C.
    const CompiledNode* nC = findNode(*dgCpu, idC);
    ASSERT_NE(nC, nullptr);
    bool foundConsumerForC = false;
    for (const auto& depId : compiledNodeDependsOn(*nC)) {
        const CompiledNode* dep = findNode(*dgCpu, depId);
        ASSERT_NE(dep, nullptr);
        if (std::holds_alternative<CompiledBridgeOpNode>(*dep) &&
            std::get<CompiledBridgeOpNode>(*dep).side == CompiledBridgeOpNode::Side::Consumer &&
            std::get<CompiledBridgeOpNode>(*dep).pairedKernelId == idC) {
            foundConsumerForC = true;
        }
    }
    EXPECT_TRUE(foundConsumerForC);
}

TEST(GraphTest, CompilerHonoursCrossDeviceAfterNodesViaBarrier) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    cpu ->registerKernel(makeAddKernel("a", 0));
    mcpu->registerKernel(makeDblKernel("b"));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer rawB = g.inputBuffer(BufferType::I32, "rawB");
    GraphBuffer outA, outB;

    IOMap mA; mA.bindInput("in", raw).bindOutput("out", BufferType::I32, outA);
    auto idA = g.addNode(cpuKernel("a"), std::move(mA), "cpu");

    // B has its own input from rawB and a CROSS-DEVICE afterNodes={idA}.
    IOMap mB; mB.bindInput("in", rawB).bindOutput("out", BufferType::I32, outB);
    auto idB = g.addNode(mockCpuKernel("b"), std::move(mB), "mcpu:0",
                         /*afterNodes=*/{idA});

    std::vector<int32_t> in = {1};
    cpu->setInputBuffer("raw",  in.data(), sizeof(int32_t));
    mcpu->setInputBuffer("rawB", in.data(), sizeof(int32_t));
    auto exec = g.compile();
    exec.run();

    // Phase 2: cross-device afterNodes materialises a barrier op pair.
    // - On 'cpu' (source side): a Producer-side barrier with
    //   pairedKernelId == idA, dependsOn == {idA}.
    // - On 'mcpu:0' (dest side): a Consumer-side barrier with
    //   pairedKernelId == idB; B.dependsOn must contain its id.
    const auto* dgCpu = findDg(exec.dgraphs(), "cpu");
    const auto* dgM   = findDg(exec.dgraphs(), "mcpu:0");
    ASSERT_NE(dgCpu, nullptr);
    ASSERT_NE(dgM, nullptr);

    const CompiledBridgeOpNode* prodBarrier = nullptr;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<CompiledBridgeOpNode>(n)) continue;
        const auto& b = std::get<CompiledBridgeOpNode>(n);
        if (b.side == CompiledBridgeOpNode::Side::Producer &&
            b.op && b.op->label() == "barrier" &&
            b.pairedKernelId == idA) {
            prodBarrier = &b;
            break;
        }
    }
    ASSERT_NE(prodBarrier, nullptr);
    EXPECT_EQ(prodBarrier->dependsOn, std::vector<std::string>{idA});

    const CompiledBridgeOpNode* consBarrier = nullptr;
    for (const auto& n : dgM->nodes) {
        if (!std::holds_alternative<CompiledBridgeOpNode>(n)) continue;
        const auto& b = std::get<CompiledBridgeOpNode>(n);
        if (b.side == CompiledBridgeOpNode::Side::Consumer &&
            b.op && b.op->label() == "barrier" &&
            b.pairedKernelId == idB) {
            consBarrier = &b;
            break;
        }
    }
    ASSERT_NE(consBarrier, nullptr);

    const CompiledNode* nB = findNode(*dgM, idB);
    ASSERT_NE(nB, nullptr);
    EXPECT_TRUE(depsContain(*nB, consBarrier->id));
}

TEST(GraphTest, CompilerChainsBounceLegsViaDependsOn) {
    // No direct mock<->mock bridge → router bounces via cpu, producing two
    // legs. The compiler must wire leg2.producer.dependsOn += leg1.consumer.id.
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    mcpu0->registerKernel(makeDblKernel("dbl"));
    mcpu1->registerKernel(makeNegKernel("neg"));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);
    // NOTE: no MockCpuMockCpuBridge → forces bounce through cpu.

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2;

    IOMap m1; m1.bindInput("in", raw).bindOutput("out", BufferType::I32, b1);
    g.addNode(mockCpuKernel("dbl"), std::move(m1), "mcpu:0");

    IOMap m2; m2.bindInput("in", b1).bindOutput("out", BufferType::I32, b2);
    g.addNode(mockCpuKernel("neg"), std::move(m2), "mcpu:1");

    std::vector<int32_t> in = {1};
    mcpu0->setInputBuffer("raw", in.data(), sizeof(int32_t));
    auto exec = g.compile();
    exec.run();

    // The bounce intermediary lives on the cpu DGraph: there must be one
    // CompiledBridgeOpNode pair (consumer of leg1, producer of leg2) where the
    // producer's dependsOn includes the consumer's id.
    const auto* dgCpu = findDg(exec.dgraphs(), "cpu");
    ASSERT_NE(dgCpu, nullptr);

    bool chained = false;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<CompiledBridgeOpNode>(n)) continue;
        const auto& b = std::get<CompiledBridgeOpNode>(n);
        if (b.side != CompiledBridgeOpNode::Side::Producer) continue;
        for (const auto& depId : b.dependsOn) {
            const CompiledNode* dep = findNode(*dgCpu, depId);
            if (!dep) continue;
            if (std::holds_alternative<CompiledBridgeOpNode>(*dep) &&
                std::get<CompiledBridgeOpNode>(*dep).side == CompiledBridgeOpNode::Side::Consumer) {
                chained = true;
            }
        }
    }
    EXPECT_TRUE(chained);
}

TEST(GraphTest, CompilerInstantiatesBridgesPerDevicePair) {
    // With a registered factory, each (srcDeviceId, dstDeviceId) pair
    // should yield its own cached bridge instance.
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);
    registerCpuLikeFactory<MockCpuMockCpuBridge>(g, DeviceType::MOCK_CPU, DeviceType::MOCK_CPU);

    IBridge* a = g.bridgeFor("mcpu:0", "mcpu:1");
    IBridge* b = g.bridgeFor("mcpu:1", "mcpu:0");
    IBridge* c = g.bridgeFor("mcpu:0", "mcpu:1");  // cached
    ASSERT_NE(a, nullptr);
    ASSERT_NE(b, nullptr);
    ASSERT_NE(c, nullptr);
    EXPECT_NE(a, b);
    EXPECT_EQ(a, c);
}

TEST(GraphTest, CompilerErrorsWhenNoBridgeFactoryRegistered) {
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    mcpu0->registerKernel(makeDblKernel("dbl"));
    mcpu1->registerKernel(makeNegKernel("neg"));

    Graph g;
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    // No bridge factories registered at all → compile() rejects the missing factory.

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2;

    IOMap m1; m1.bindInput("in", raw).bindOutput("out", BufferType::I32, b1);
    g.addNode(mockCpuKernel("dbl"), std::move(m1), "mcpu:0");
    IOMap m2; m2.bindInput("in", b1).bindOutput("out", BufferType::I32, b2);
    g.addNode(mockCpuKernel("neg"), std::move(m2), "mcpu:1");

    std::vector<int32_t> in = {1};
    mcpu0->setInputBuffer("raw", in.data(), sizeof(int32_t));
    EXPECT_THROW(g.compile(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Phase 3 — CpuDevice dep-driven executor tests
// ---------------------------------------------------------------------------

// Pure-CPU fan-out / fan-in: root → {A, B} → join.
// Verifies the dep-driven scheduler executes both branches and that the
// join kernel observes both inputs.
TEST(GraphTest, CpuExecutorRunsDiamondAcrossTwoBranches) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel(makeAddKernel("root", 0));      // identity copy
    cpu->registerKernel(makeAddKernel("addA", 100));
    cpu->registerKernel(makeAddKernel("addB", 200));
    cpu->registerKernel(makeCpuKernel("join", [](const CpuKernelArgs& args) {
        auto a   = args.buffer("a").as<const int32_t>();
        auto b   = args.buffer("b").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("a").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    }));

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw  = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer rOut, aOut, bOut, joined;

    IOMap ioR; ioR.bindInput("in", raw).bindOutput("out", BufferType::I32, rOut);
    g.addNode(cpuKernel("root"), std::move(ioR), "cpu");
    IOMap ioA; ioA.bindInput("in", rOut).bindOutput("out", BufferType::I32, aOut);
    g.addNode(cpuKernel("addA"), std::move(ioA), "cpu");
    IOMap ioB; ioB.bindInput("in", rOut).bindOutput("out", BufferType::I32, bOut);
    g.addNode(cpuKernel("addB"), std::move(ioB), "cpu");
    IOMap ioJ; ioJ.bindInput("a", aOut)
                  .bindInput("b", bOut)
                  .bindOutput("out", BufferType::I32, joined);
    g.addNode(cpuKernel("join"), std::move(ioJ), "cpu");

    std::vector<int32_t> in = {1, 2, 3};
    cpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    g.compile().run();

    std::vector<int32_t> out(3);
    cpu->getOutputBuffer(joined.name(), out.data(), out.size() * sizeof(int32_t));
    // (x+100) + (x+200) = 2x + 300
    EXPECT_EQ(out[0], 302);
    EXPECT_EQ(out[1], 304);
    EXPECT_EQ(out[2], 306);
}

TEST(GraphTest, CpuExecutorResetsDependencyStateAcrossRuns) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel(makeAddKernel("copy", 0));
    cpu->registerKernel(makeAddKernel("add1", 1));

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer mid, finalBuf;

    IOMap ioA; ioA.bindInput("in", raw).bindOutput("out", BufferType::I32, mid);
    g.addNode(cpuKernel("copy"), std::move(ioA), "cpu");

    IOMap ioB; ioB.bindInput("in", mid).bindOutput("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("add1"), std::move(ioB), "cpu");

    auto runOnce = [&](int32_t value) {
        cpu->setInputBuffer("raw", &value, sizeof(value));
        g.compile().run();
        int32_t out = 0;
        cpu->getOutputBuffer(finalBuf.name(), &out, sizeof(out));
        return out;
    };

    EXPECT_EQ(runOnce(1), 2);
    EXPECT_EQ(runOnce(5), 6);
    EXPECT_EQ(runOnce(9), 10);
}

TEST(GraphTest, CpuDevicePlansRemainIndependentAfterAnotherCompile) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel(makeAddKernel("add1", 1));
    cpu->registerKernel(makeAddKernel("add10", 10));

    GraphBuffer raw = GraphBuffer::make(BufferType::I32, "raw");

    IOMap io1;
    GraphBuffer out1;
    io1.bindInput("in", raw).bindOutput("out", BufferType::I32, out1);

    DGraph dg1;
    dg1.deviceId = "cpu";
    dg1.device = cpu;
    dg1.scalarValues = std::make_shared<std::map<std::string, uint64_t>>();
    dg1.nodes.emplace_back(CompiledKernelNode{"plan1", cpuKernel("add1"), "cpu", io1, {}});

    IOMap io2;
    GraphBuffer out2;
    io2.bindInput("in", raw).bindOutput("out", BufferType::I32, out2);

    DGraph dg2;
    dg2.deviceId = "cpu";
    dg2.device = cpu;
    dg2.scalarValues = std::make_shared<std::map<std::string, uint64_t>>();
    dg2.nodes.emplace_back(CompiledKernelNode{"plan2", cpuKernel("add10"), "cpu", io2, {}});

    auto plan1 = cpu->compilePlan(dg1);
    auto plan2 = cpu->compilePlan(dg2);

    int32_t input = 5;
    cpu->setInputBuffer("raw", &input, sizeof(input));

    plan1->launch();
    plan1->wait();
    int32_t firstOutput = 0;
    cpu->getOutputBuffer(out1.name(), &firstOutput, sizeof(firstOutput));
    EXPECT_EQ(firstOutput, 6);

    plan2->launch();
    plan2->wait();
    int32_t secondOutput = 0;
    cpu->getOutputBuffer(out2.name(), &secondOutput, sizeof(secondOutput));
    EXPECT_EQ(secondOutput, 15);
}

// Two plans compiled on the same CpuDevice with distinct scalar maps must
// each read and write only its own map. The previous architecture cached the
// scalar pointer on the device itself, so compiling plan B silently rebound
// plan A's kernel scalar reads to plan B's map.
TEST(GraphTest, CpuDevicePlansHaveIndependentScalarMaps) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    IOTypeMap addOneType;
    addOneType.inputScalars.push_back({"in", ScalarType::I32});
    addOneType.outputScalars.push_back({"out", ScalarType::I32});
    cpu->registerKernel(makeCpuKernel("add_one", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 1));
    }, addOneType));

    auto buildDGraph = [&](const std::string& nodeId,
                           const std::shared_ptr<std::map<std::string, uint64_t>>&
                               scalarValues) {
        IOMap io;
        io.bindInputScalar("in", GraphScalar::globalVar(ScalarType::I32, "input"))
          .bindOutputScalar("out", GraphScalar::globalVar(ScalarType::I32, "output"));

        DGraph dg;
        dg.deviceId = "cpu";
        dg.device = cpu;
        dg.scalarValues = scalarValues;
        dg.nodes.emplace_back(CompiledKernelNode{nodeId, cpuKernel("add_one", addOneType),
                                                 "cpu", std::move(io), {}});
        return dg;
    };

    auto scalars1 = std::make_shared<std::map<std::string, uint64_t>>();
    auto scalars2 = std::make_shared<std::map<std::string, uint64_t>>();

    auto plan1 = cpu->compilePlan(buildDGraph("plan1", scalars1));
    auto plan2 = cpu->compilePlan(buildDGraph("plan2", scalars2));

    (*scalars1)[scopedScalarKey(0, "input")] = 7;
    (*scalars2)[scopedScalarKey(0, "input")] = 1000;

    plan1->launch();
    plan1->wait();

    EXPECT_EQ((*scalars1)[scopedScalarKey(0, "output")], 8u);
    EXPECT_EQ(scalars2->count(scopedScalarKey(0, "output")), 0u)
        << "plan1 leaked into plan2's scalar map";

    plan2->launch();
    plan2->wait();

    EXPECT_EQ((*scalars2)[scopedScalarKey(0, "output")], 1001u);
    EXPECT_EQ((*scalars1)[scopedScalarKey(0, "output")], 8u)
        << "plan2 overwrote plan1's scalar output";
}

TEST(GraphTest, CpuFixedCountLoopWithZeroIterationsAndNoOutputsNoops) {
    Graph g = Graph::withDefaults();

    auto body = g.rootRegion().createChild();
    g.addLoop(fixedLoopSpec(LoopTripCount::constant<int32_t>(0), body));

    EXPECT_NO_THROW(g.compile().run());
}

TEST(GraphTest, CpuFixedCountLoopWithZeroIterationsAndOutputsThrows) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeAddKernel("copy", 0));

    GraphBuffer rootInput = g.inputBuffer(BufferType::I32, "zero_loop_input");
    auto body = g.rootRegion().createChild();
    GraphBuffer bodyInput = body->inputBuffer(BufferType::I32, "zero_loop_body_input");
    body->importFromParent(std::vector<BufferBoundaryMapping>{{rootInput, bodyInput}});
    GraphBuffer bodyOutput;
    IOMap bodyIo;
    bodyIo.bindInput("in", bodyInput)
          .bindOutput("out", BufferType::I32, bodyOutput, body->scopeId());
    body->addKernel(cpuKernel("copy", ioType), std::move(bodyIo), "cpu");

    IOMap loopIo;
    GraphBuffer loopOutput;
    loopIo.bindOutput("out", BufferType::I32, loopOutput, g.rootRegion().scopeId());
    g.addLoop(fixedLoopSpec(i32BufferOutType(), std::move(loopIo),
                            LoopTripCount::constant<int32_t>(0), body));

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST(GraphTest, CpuFixedCountLoopPublishesFinalBufferOutput) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    auto calls = std::make_shared<int32_t>(0);
    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeCpuKernel("loop_body", [calls](const CpuKernelArgs& args) {
        ++*calls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + *calls;
    }, ioType));

    GraphBuffer loopInput = g.inputBuffer(BufferType::I32, "loop_input");
    auto body = g.rootRegion().createChild();
    GraphBuffer bodyInput = body->inputBuffer(BufferType::I32, "loop_body_input");
    body->importFromParent(std::vector<BufferBoundaryMapping>{{loopInput, bodyInput}});
    GraphBuffer bodyOutput;
    IOMap bodyIo;
    bodyIo.bindInput("in", bodyInput)
          .bindOutput("out", BufferType::I32, bodyOutput, body->scopeId());
    body->addKernel(cpuKernel("loop_body", ioType), std::move(bodyIo), "cpu");

    IOMap loopIo;
    GraphBuffer loopOutput;
    loopIo.bindOutput("out", BufferType::I32, loopOutput, g.rootRegion().scopeId());
    g.addLoop(fixedLoopSpec(i32BufferOutType(), std::move(loopIo),
                            LoopTripCount::constant<int32_t>(3), body));

    std::vector<int32_t> input = {10, 20};
    cpu->setInputBuffer(loopInput.name(), input.data(), input.size() * sizeof(int32_t));

    auto exec = g.compile();
    ASSERT_NO_THROW(exec.run());

    std::vector<int32_t> output(2);
    cpu->getOutputBuffer(loopOutput.name(), output.data(), output.size() * sizeof(int32_t));
    EXPECT_EQ(output[0], 13);
    EXPECT_EQ(output[1], 23);
    EXPECT_EQ(*calls, 3);
}

TEST(GraphTest, CpuScalarTripCountLoopResetsAcrossRuns) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    auto calls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("tick", [calls](const CpuKernelArgs&) {
        ++*calls;
    }));

    GraphScalar tripCount = g.globalScalar(ScalarType::I32, "trip_count");
    auto body = g.rootRegion().createChild();
    body->addKernel(cpuKernel("tick"), IOMap{}, "cpu");
    g.addLoop(fixedLoopSpec(
        LoopTripCount::scalar(ScalarType::I32, tripCount.varName(), tripCount.scopeId()),
        body));

    g.setScalar<int32_t>("trip_count", 1);
    ASSERT_NO_THROW(g.compile().run());
    EXPECT_EQ(*calls, 1);

    g.setScalar<int32_t>("trip_count", 3);
    ASSERT_NO_THROW(g.compile().run());
    EXPECT_EQ(*calls, 4);

    g.setScalar<int32_t>("trip_count", 2);
    ASSERT_NO_THROW(g.compile().run());
    EXPECT_EQ(*calls, 6);
}

TEST(GraphTest, CpuConditionalUsesScalarConditionAndPublishesSelectedBranch) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    auto thenCalls = std::make_shared<int32_t>(0);
    auto elseCalls = std::make_shared<int32_t>(0);
    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeCpuKernel("then_branch", [thenCalls](const CpuKernelArgs& args) {
        ++*thenCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 100;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("else_branch", [elseCalls](const CpuKernelArgs& args) {
        ++*elseCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 200;
    }, ioType));

    GraphScalar flag = g.globalScalar(ScalarType::I32, "branch_flag");
    GraphBuffer thenSource = g.inputBuffer(BufferType::I32, "then_source");
    GraphBuffer elseSource = g.inputBuffer(BufferType::I32, "else_source");
    auto thenRegion = g.rootRegion().createChild();
    auto elseRegion = g.rootRegion().createChild();

    GraphBuffer thenInput = thenRegion->inputBuffer(BufferType::I32, "then_input");
    thenRegion->importFromParent(std::vector<BufferBoundaryMapping>{{thenSource, thenInput}});
    GraphBuffer thenOutput;
    IOMap thenIo;
    thenIo.bindInput("in", thenInput)
          .bindOutput("out", BufferType::I32, thenOutput, thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("then_branch", ioType), std::move(thenIo), "cpu");

    GraphBuffer elseInput = elseRegion->inputBuffer(BufferType::I32, "else_input");
    elseRegion->importFromParent(std::vector<BufferBoundaryMapping>{{elseSource, elseInput}});
    GraphBuffer elseOutput;
    IOMap elseIo;
    elseIo.bindInput("in", elseInput)
          .bindOutput("out", BufferType::I32, elseOutput, elseRegion->scopeId());
    elseRegion->addKernel(cpuKernel("else_branch", ioType), std::move(elseIo), "cpu");

    IOMap conditionalIo;
    GraphBuffer conditionalOutput;
    conditionalIo.bindOutput("out", BufferType::I32, conditionalOutput,
                                   g.rootRegion().scopeId());
    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    g.addConditional(ifElseSpec(i32BufferOutType(), std::move(conditionalIo),
                                std::move(condition), thenRegion, elseRegion));

    int32_t thenValue = 1;
    int32_t elseValue = 2;
    cpu->setInputBuffer(thenSource.name(), &thenValue, sizeof(thenValue));
    cpu->setInputBuffer(elseSource.name(), &elseValue, sizeof(elseValue));

    g.setScalar<int32_t>("branch_flag", 1);
    ASSERT_NO_THROW(g.compile().run());
    int32_t output = 0;
    cpu->getOutputBuffer(conditionalOutput.name(), &output, sizeof(output));
    EXPECT_EQ(output, 101);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 0);

    g.setScalar<int32_t>("branch_flag", 0);
    ASSERT_NO_THROW(g.compile().run());
    cpu->getOutputBuffer(conditionalOutput.name(), &output, sizeof(output));
    EXPECT_EQ(output, 202);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 1);
}

TEST(GraphTest, CpuFixedCountLoopBufferBoundaryCarriesUpdatedStateAcrossRuns) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType = i32BufferInOutType();
    auto calls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("buffer_boundary_increment",
                                      [calls](const CpuKernelArgs& args) {
        ++*calls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1;
    }, ioType));

    GraphScalar tripCount = g.globalScalar(ScalarType::I32, "buffer_boundary_trip_count");
    GraphBuffer state = g.inputBuffer(BufferType::I32, "buffer_boundary_state");
    auto body = g.rootRegion().createChild();
    GraphBuffer localState = body->inputBuffer(BufferType::I32, "state");
    body->importFromParent(std::vector<BufferBoundaryMapping>{{state, localState}});

    IOMap bodyIo;
    GraphBuffer localNext;
    bodyIo.bindInput("in", localState)
          .bindOutput("out", BufferType::I32, localNext, body->scopeId());
    body->addKernel(cpuKernel("buffer_boundary_increment", ioType), std::move(bodyIo), "cpu");
    body->exportToParent(std::vector<BufferBoundaryMapping>{{localNext, state}});

    g.addLoop(fixedLoopSpec(
        LoopTripCount::scalar(ScalarType::I32, tripCount.varName(), tripCount.scopeId()),
        body));

    auto runWithCount = [&](int32_t count, std::vector<int32_t> input) {
        cpu->setInputBuffer(state.name(), input.data(), input.size() * sizeof(int32_t));
        g.setScalar<int32_t>("buffer_boundary_trip_count", count);
        ASSERT_NO_THROW(g.compile().run());
        std::vector<int32_t> output(input.size(), 0);
        cpu->getOutputBuffer(state.name(), output.data(), output.size() * sizeof(int32_t));
        for (size_t i = 0; i < input.size(); ++i) {
            EXPECT_EQ(output[i], input[i] + count);
        }
    };

    runWithCount(1, {10, 20});
    EXPECT_EQ(*calls, 1);
    runWithCount(3, {1, 2});
    EXPECT_EQ(*calls, 4);
    runWithCount(2, {7, 8});
    EXPECT_EQ(*calls, 6);
}

TEST(GraphTest, CpuConditionalBufferBoundaryExportsOnlySelectedBranch) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType = i32BufferInOutType();
    auto thenCalls = std::make_shared<int32_t>(0);
    auto elseCalls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("then_buffer_boundary", [thenCalls](const CpuKernelArgs& args) {
        ++*thenCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 100;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("else_buffer_boundary", [elseCalls](const CpuKernelArgs& args) {
        ++*elseCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 200;
    }, ioType));

    GraphScalar flag = g.globalScalar(ScalarType::I32, "buffer_branch_flag");
    GraphBuffer source = g.inputBuffer(BufferType::I32, "buffer_branch_source");
    GraphBuffer result = g.inputBuffer(BufferType::I32, "buffer_branch_result");

    auto thenRegion = g.rootRegion().createChild();
    GraphBuffer thenInput = thenRegion->inputBuffer(BufferType::I32, "input");
    thenRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, thenInput}});
    IOMap thenIo;
    GraphBuffer thenOutput;
    thenIo.bindInput("in", thenInput)
          .bindOutput("out", BufferType::I32, thenOutput, thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("then_buffer_boundary", ioType), std::move(thenIo), "cpu");
    thenRegion->exportToParent(std::vector<BufferBoundaryMapping>{{thenOutput, result}});

    auto elseRegion = g.rootRegion().createChild();
    GraphBuffer elseInput = elseRegion->inputBuffer(BufferType::I32, "input");
    elseRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, elseInput}});
    IOMap elseIo;
    GraphBuffer elseOutput;
    elseIo.bindInput("in", elseInput)
          .bindOutput("out", BufferType::I32, elseOutput, elseRegion->scopeId());
    elseRegion->addKernel(cpuKernel("else_buffer_boundary", ioType), std::move(elseIo), "cpu");
    elseRegion->exportToParent(std::vector<BufferBoundaryMapping>{{elseOutput, result}});

    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    g.addConditional(ifElseSpec(std::move(condition), thenRegion, elseRegion));

    auto runBranch = [&](int32_t branchFlag, int32_t input, int32_t expected) {
        cpu->setInputBuffer(source.name(), &input, sizeof(input));
        g.setScalar<int32_t>("buffer_branch_flag", branchFlag);
        ASSERT_NO_THROW(g.compile().run());
        int32_t output = 0;
        cpu->getOutputBuffer(result.name(), &output, sizeof(output));
        EXPECT_EQ(output, expected);
    };

    runBranch(1, 7, 107);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 0);
    runBranch(0, 11, 211);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 1);
    runBranch(1, 3, 103);
    EXPECT_EQ(*thenCalls, 2);
    EXPECT_EQ(*elseCalls, 1);
}

TEST(GraphTest, CpuConditionalScalarBoundaryExportsOnlySelectedBranch) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap scalarInOutType;
    scalarInOutType.inputScalars.push_back({"in", ScalarType::I32});
    scalarInOutType.outputScalars.push_back({"out", ScalarType::I32});

    auto thenCalls = std::make_shared<int32_t>(0);
    auto elseCalls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("then_scalar_boundary", [thenCalls](const CpuKernelArgs& args) {
        ++*thenCalls;
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 100));
    }, scalarInOutType));
    cpu->registerKernel(makeCpuKernel("else_scalar_boundary", [elseCalls](const CpuKernelArgs& args) {
        ++*elseCalls;
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 200));
    }, scalarInOutType));

    GraphScalar flag = g.globalScalar(ScalarType::I32, "scalar_branch_flag");
    GraphScalar source = g.globalScalar(ScalarType::I32, "scalar_branch_source");
    GraphScalar result = g.globalScalar(ScalarType::I32, "scalar_branch_result");

    auto thenRegion = g.rootRegion().createChild();
    GraphScalar thenInput = thenRegion->scalar(ScalarType::I32, "input");
    GraphScalar thenOutput = thenRegion->scalar(ScalarType::I32, "output");
    thenRegion->importFromParent({{source, thenInput}});
    IOMap thenIo;
    thenIo.bindInputScalar("in", thenInput)
          .bindOutputScalar("out", thenOutput);
    thenRegion->addKernel(cpuKernel("then_scalar_boundary", scalarInOutType),
                          std::move(thenIo), "cpu");
    thenRegion->exportToParent({{thenOutput, result}});

    auto elseRegion = g.rootRegion().createChild();
    GraphScalar elseInput = elseRegion->scalar(ScalarType::I32, "input");
    GraphScalar elseOutput = elseRegion->scalar(ScalarType::I32, "output");
    elseRegion->importFromParent({{source, elseInput}});
    IOMap elseIo;
    elseIo.bindInputScalar("in", elseInput)
          .bindOutputScalar("out", elseOutput);
    elseRegion->addKernel(cpuKernel("else_scalar_boundary", scalarInOutType),
                          std::move(elseIo), "cpu");
    elseRegion->exportToParent({{elseOutput, result}});

    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    g.addConditional(ifElseSpec(std::move(condition), thenRegion, elseRegion));

    g.setScalar<int32_t>("scalar_branch_flag", 1);
    g.setScalar<int32_t>("scalar_branch_source", 7);
    auto execThen = g.compile();
    ASSERT_NO_THROW(execThen.run());
    EXPECT_EQ(execThen.getScalar<int32_t>("scalar_branch_result"), 107);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 0);

    g.setScalar<int32_t>("scalar_branch_flag", 0);
    g.setScalar<int32_t>("scalar_branch_source", 11);
    auto execElse = g.compile();
    ASSERT_NO_THROW(execElse.run());
    EXPECT_EQ(execElse.getScalar<int32_t>("scalar_branch_result"), 211);
    EXPECT_EQ(*thenCalls, 1);
    EXPECT_EQ(*elseCalls, 1);

    g.setScalar<int32_t>("scalar_branch_flag", 1);
    g.setScalar<int32_t>("scalar_branch_source", 3);
    auto execThenAgain = g.compile();
    ASSERT_NO_THROW(execThenAgain.run());
    EXPECT_EQ(execThenAgain.getScalar<int32_t>("scalar_branch_result"), 103);
    EXPECT_EQ(*thenCalls, 2);
    EXPECT_EQ(*elseCalls, 1);
}

TEST(GraphTest, CpuWhileLoopFalseConditionNoops) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    auto calls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("while_tick", [calls](const CpuKernelArgs&) {
        ++*calls;
    }));

    auto body = g.rootRegion().createChild();
    body->addKernel(cpuKernel("while_tick"), IOMap{}, "cpu");
    g.addLoop(whileLoopSpec(Condition::alwaysFalse(), body));

    EXPECT_NO_THROW(g.compile().run());
    EXPECT_EQ(*calls, 0);
}

TEST(GraphTest, CpuWhileLoopScalarBoundaryCarriesUpdatedCondition) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap incrementType;
    incrementType.inputScalars.push_back({"in", ScalarType::I32});
    incrementType.outputScalars.push_back({"out", ScalarType::I32});

    auto calls = std::make_shared<int32_t>(0);
    cpu->registerKernel(makeCpuKernel("increment_scalar", [calls](const CpuKernelArgs& args) {
        ++*calls;
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 1));
    }, incrementType));

    GraphScalar counter = g.globalScalar(ScalarType::I32, "boundary_counter");
    GraphScalar limit = g.globalScalar(ScalarType::I32, "boundary_limit");

    auto body = g.rootRegion().createChild();
    GraphScalar localCounter = body->scalar(ScalarType::I32, "counter");
    GraphScalar localNext = body->scalar(ScalarType::I32, "next");

    std::string startId = body->importFromParent({{counter, localCounter}});

    IOMap bodyIo;
    bodyIo.bindInputScalar("in", localCounter)
          .bindOutputScalar("out", localNext);
    std::string kernelId = body->addKernel(cpuKernel("increment_scalar", incrementType),
                                           std::move(bodyIo), "cpu", {startId});
    body->exportToParent({{localNext, counter}}, {kernelId});

    Condition condition = Condition::compare(
        CompareOp::LT,
        ConditionOperand::scalar(ScalarType::I32, counter.varName(), counter.scopeId()),
        ConditionOperand::scalar(ScalarType::I32, limit.varName(), limit.scopeId()));
    g.addLoop(whileLoopSpec(std::move(condition), body));

    g.setScalar<int32_t>("boundary_counter", 0);
    g.setScalar<int32_t>("boundary_limit", 1);
    auto execOne = g.compile();
    ASSERT_NO_THROW(execOne.run());
    EXPECT_EQ(execOne.getScalar<int32_t>("boundary_counter"), 1);
    EXPECT_EQ(*calls, 1);

    g.setScalar<int32_t>("boundary_counter", 0);
    g.setScalar<int32_t>("boundary_limit", 3);
    auto execThree = g.compile();
    ASSERT_NO_THROW(execThree.run());
    EXPECT_EQ(execThree.getScalar<int32_t>("boundary_counter"), 3);
    EXPECT_EQ(*calls, 4);

    g.setScalar<int32_t>("boundary_counter", 0);
    g.setScalar<int32_t>("boundary_limit", 2);
    auto execTwo = g.compile();
    ASSERT_NO_THROW(execTwo.run());
    EXPECT_EQ(execTwo.getScalar<int32_t>("boundary_counter"), 2);
    EXPECT_EQ(*calls, 6);
}

// Cross-device data flow: CPU consumer kernel must wait on a producer
// running on MockCpuDevice. If the consumer fires before the producer
// signals, the read-back will be wrong (zero-init / stale).
TEST(GraphTest, CpuExecutorBlocksConsumerUntilProducerSignals) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    mcpu->registerKernel(makeDblKernel("dbl"));
    cpu->registerKernel(makeAddKernel("sink", 7));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer dbld, finalBuf;

    IOMap ioM; ioM.bindInput("in", raw).bindOutput("out", BufferType::I32, dbld);
    g.addNode(mockCpuKernel("dbl"), std::move(ioM), "mcpu:0");
    IOMap ioS; ioS.bindInput("in", dbld).bindOutput("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("sink"), std::move(ioS), "cpu");

    std::vector<int32_t> in = {3, 5, 9};
    mcpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    g.compile().run();

    std::vector<int32_t> out(3);
    cpu->getOutputBuffer(finalBuf.name(), out.data(), out.size() * sizeof(int32_t));
    EXPECT_EQ(out[0], 13);   // 3*2 + 7
    EXPECT_EQ(out[1], 17);   // 5*2 + 7
    EXPECT_EQ(out[2], 25);   // 9*2 + 7
}

// One remote producer feeds two CpuDevice kernels on the same consumer
// device. Both consumers must depend on the same consumer-side bridge op;
// otherwise the dep-driven executor may run one consumer before the bridge
// has materialised the buffer locally.
TEST(GraphTest, CpuExecutorSharedRemoteBufferFanoutUsesSameConsumerBridge) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    mcpu->registerKernel(makeDblKernel("dbl"));
    cpu->registerKernel(makeAddKernel("add1", 1));
    cpu->registerKernel(makeAddKernel("add2", 2));
    cpu->registerKernel(makeCpuKernel("sum", [](const CpuKernelArgs& args) {
        auto left  = args.buffer("left").as<const int32_t>();
        auto right = args.buffer("right").as<const int32_t>();
        auto out   = args.buffer("out").as<int32_t>();
        auto n     = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = left[i] + right[i];
    }));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer sharedBuf, leftBuf, rightBuf, sumBuf;

    IOMap ioP; ioP.bindInput("in", raw).bindOutput("out", BufferType::I32, sharedBuf);
    g.addNode(mockCpuKernel("dbl"), std::move(ioP), "mcpu:0");

    IOMap ioL; ioL.bindInput("in", sharedBuf).bindOutput("out", BufferType::I32, leftBuf);
    auto idL = g.addNode(cpuKernel("add1"), std::move(ioL), "cpu");

    IOMap ioR; ioR.bindInput("in", sharedBuf).bindOutput("out", BufferType::I32, rightBuf);
    auto idR = g.addNode(cpuKernel("add2"), std::move(ioR), "cpu");

    IOMap ioS; ioS.bindInput("left", leftBuf)
                  .bindInput("right", rightBuf)
                  .bindOutput("out", BufferType::I32, sumBuf);
    g.addNode(cpuKernel("sum"), std::move(ioS), "cpu");

    std::vector<int32_t> in = {3, 5};
    mcpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    auto exec = g.compile();
    ASSERT_NO_THROW(exec.run());

    std::vector<int32_t> out(2);
    cpu->getOutputBuffer(sumBuf.name(), out.data(), out.size() * sizeof(int32_t));
    EXPECT_EQ(out[0], 15);  // (3*2 + 1) + (3*2 + 2)
    EXPECT_EQ(out[1], 23);  // (5*2 + 1) + (5*2 + 2)

    const auto* dgCpu = findDg(exec.dgraphs(), "cpu");
    ASSERT_NE(dgCpu, nullptr);

    const CompiledNode* nL = findNode(*dgCpu, idL);
    const CompiledNode* nR = findNode(*dgCpu, idR);
    ASSERT_NE(nL, nullptr);
    ASSERT_NE(nR, nullptr);

    std::string sharedBridgeId;
    for (const auto& depId : compiledNodeDependsOn(*nL)) {
        const CompiledNode* dep = findNode(*dgCpu, depId);
        if (!dep || !std::holds_alternative<CompiledBridgeOpNode>(*dep)) continue;
        const auto& bridge = std::get<CompiledBridgeOpNode>(*dep);
        if (bridge.side == CompiledBridgeOpNode::Side::Consumer) {
            sharedBridgeId = bridge.id;
            break;
        }
    }

    ASSERT_FALSE(sharedBridgeId.empty());
    EXPECT_TRUE(depsContain(*nR, sharedBridgeId));
}

TEST(GraphTest, CpuConditionalSelectedBranchSharedRemoteFanoutUsesSameConsumerBridge) {
    auto cpu = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    auto seedCalls = std::make_shared<int32_t>(0);
    auto remoteCalls = std::make_shared<int32_t>(0);
    auto leftCalls = std::make_shared<int32_t>(0);
    auto rightCalls = std::make_shared<int32_t>(0);
    auto elseCalls = std::make_shared<int32_t>(0);
    IOTypeMap ioType = i32BufferInOutType();
    cpu->registerKernel(makeCpuKernel("fanout_seed", [seedCalls](const CpuKernelArgs& args) {
        ++*seedCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i];
    }, ioType));
    mcpu->registerKernel(makeCpuKernel("fanout_dbl", [remoteCalls](const CpuKernelArgs& args) {
        ++*remoteCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * 2;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("fanout_add1", [leftCalls](const CpuKernelArgs& args) {
        ++*leftCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 1;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("fanout_add2", [rightCalls](const CpuKernelArgs& args) {
        ++*rightCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + 2;
    }, ioType));
    cpu->registerKernel(makeCpuKernel("fanout_sum", [](const CpuKernelArgs& args) {
        auto left = args.buffer("left").as<const int32_t>();
        auto right = args.buffer("right").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = left[i] + right[i];
    }));
    cpu->registerKernel(makeCpuKernel("fanout_else", [elseCalls](const CpuKernelArgs& args) {
        ++*elseCalls;
        auto in = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] - 100;
    }, ioType));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphScalar flag = g.globalScalar(ScalarType::I32, "fanout_branch_flag");
    GraphBuffer source = g.inputBuffer(BufferType::I32, "fanout_branch_source");
    GraphBuffer result = g.inputBuffer(BufferType::I32, "fanout_branch_result");

    auto thenRegion = g.rootRegion().createChild();
    GraphBuffer thenInput = thenRegion->inputBuffer(BufferType::I32, "input");
    thenRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, thenInput}});

    IOMap seedIo;
    GraphBuffer seeded;
    seedIo.bindInput("in", thenInput)
          .bindOutput("out", BufferType::I32, seeded, thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("fanout_seed", ioType), std::move(seedIo), "cpu");

    IOMap remoteIo;
    GraphBuffer sharedRemote;
    remoteIo.bindInput("in", seeded)
            .bindOutput("out", BufferType::I32, sharedRemote,
                              thenRegion->scopeId());
    thenRegion->addKernel(mockCpuKernel("fanout_dbl", ioType), std::move(remoteIo),
                          "mcpu:0");

    IOMap leftIo;
    GraphBuffer leftOut;
    leftIo.bindInput("in", sharedRemote)
          .bindOutput("out", BufferType::I32, leftOut, thenRegion->scopeId());
    std::string leftId = thenRegion->addKernel(cpuKernel("fanout_add1", ioType),
                                               std::move(leftIo), "cpu");

    IOMap rightIo;
    GraphBuffer rightOut;
    rightIo.bindInput("in", sharedRemote)
           .bindOutput("out", BufferType::I32, rightOut, thenRegion->scopeId());
    std::string rightId = thenRegion->addKernel(cpuKernel("fanout_add2", ioType),
                                                std::move(rightIo), "cpu");

    IOMap sumIo;
    GraphBuffer thenOutput;
    sumIo.bindInput("left", leftOut)
         .bindInput("right", rightOut)
         .bindOutput("out", BufferType::I32, thenOutput, thenRegion->scopeId());
    thenRegion->addKernel(cpuKernel("fanout_sum"), std::move(sumIo), "cpu");
    thenRegion->exportToParent(std::vector<BufferBoundaryMapping>{{thenOutput, result}});

    auto elseRegion = g.rootRegion().createChild();
    GraphBuffer elseInput = elseRegion->inputBuffer(BufferType::I32, "input");
    elseRegion->importFromParent(std::vector<BufferBoundaryMapping>{{source, elseInput}});
    IOMap elseIo;
    GraphBuffer elseOutput;
    elseIo.bindInput("in", elseInput)
          .bindOutput("out", BufferType::I32, elseOutput, elseRegion->scopeId());
    elseRegion->addKernel(cpuKernel("fanout_else", ioType), std::move(elseIo), "cpu");
    elseRegion->exportToParent(std::vector<BufferBoundaryMapping>{{elseOutput, result}});

    Condition condition = Condition::compare(
        CompareOp::EQ,
        ConditionOperand::scalar(ScalarType::I32, flag.varName(), flag.scopeId()),
        ConditionOperand::constant<int32_t>(1));
    std::string conditionalId = g.addConditional(
        ifElseSpec(std::move(condition), thenRegion, elseRegion));

    auto runBranch = [&](int32_t branchFlag, std::vector<int32_t> input,
                         std::vector<int32_t> expected) {
        cpu->setInputBuffer(source.name(), input.data(), input.size() * sizeof(int32_t));
        g.setScalar<int32_t>(flag.varName(), branchFlag);
        ASSERT_NO_THROW(g.compile().run());
        std::vector<int32_t> output(input.size(), 0);
        cpu->getOutputBuffer(result.name(), output.data(), output.size() * sizeof(int32_t));
        EXPECT_EQ(output, expected);
    };

    runBranch(1, {3, 5}, {15, 23});
    EXPECT_EQ(*seedCalls, 1);
    EXPECT_EQ(*remoteCalls, 1);
    EXPECT_EQ(*leftCalls, 1);
    EXPECT_EQ(*rightCalls, 1);
    EXPECT_EQ(*elseCalls, 0);

    runBranch(0, {101, 105}, {1, 5});
    EXPECT_EQ(*seedCalls, 1);
    EXPECT_EQ(*remoteCalls, 1);
    EXPECT_EQ(*leftCalls, 1);
    EXPECT_EQ(*rightCalls, 1);
    EXPECT_EQ(*elseCalls, 1);

    runBranch(1, {2, 4}, {11, 19});
    EXPECT_EQ(*seedCalls, 2);
    EXPECT_EQ(*remoteCalls, 2);
    EXPECT_EQ(*leftCalls, 2);
    EXPECT_EQ(*rightCalls, 2);
    EXPECT_EQ(*elseCalls, 1);

    auto structureExec = g.compile();
    const DGraph* cpuDGraph = findDg(structureExec.dgraphs(), "cpu");
    ASSERT_NE(cpuDGraph, nullptr);
    const DGraphChild* thenChild = findChildDGraphs(
        *cpuDGraph, conditionalId, DGraphChildRole::ConditionalThen);
    ASSERT_NE(thenChild, nullptr);
    const DGraph* cpuThenChild = findChildDg(*thenChild, "cpu");
    ASSERT_NE(cpuThenChild, nullptr);

    const CompiledNode* leftNode = findNode(*cpuThenChild, leftId);
    const CompiledNode* rightNode = findNode(*cpuThenChild, rightId);
    ASSERT_NE(leftNode, nullptr);
    ASSERT_NE(rightNode, nullptr);

    std::string sharedBridgeId;
    for (const auto& depId : compiledNodeDependsOn(*leftNode)) {
        const CompiledNode* dep = findNode(*cpuThenChild, depId);
        if (!dep || !std::holds_alternative<CompiledBridgeOpNode>(*dep)) continue;
        const auto& bridge = std::get<CompiledBridgeOpNode>(*dep);
        if (bridge.side == CompiledBridgeOpNode::Side::Consumer) {
            sharedBridgeId = bridge.id;
            break;
        }
    }

    ASSERT_FALSE(sharedBridgeId.empty());
    EXPECT_TRUE(depsContain(*rightNode, sharedBridgeId));
}

// Three independent producers on three mock devices feed a single CPU
// fan-in kernel. The CPU executor must service all three pending
// consumer-side ops regardless of the order their semaphores fire.
TEST(GraphTest, CpuExecutorMultipleConsumersOutOfOrderSignals) {
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");
    auto mcpu2 = std::make_shared<MockCpuDevice>("mcpu:2");

    mcpu0->registerKernel(makeAddKernel("k", 1));
    mcpu1->registerKernel(makeAddKernel("k", 10));
    mcpu2->registerKernel(makeAddKernel("k", 100));
    cpu->registerKernel(makeCpuKernel("fanin", [](const CpuKernelArgs& args) {
        auto a = args.buffer("a").as<const int32_t>();
        auto b = args.buffer("b").as<const int32_t>();
        auto c = args.buffer("c").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("a").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i] + c[i];
    }));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    g.registerDevice(mcpu2);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer rawA = g.inputBuffer(BufferType::I32, "rawA");
    GraphBuffer rawB = g.inputBuffer(BufferType::I32, "rawB");
    GraphBuffer rawC = g.inputBuffer(BufferType::I32, "rawC");
    GraphBuffer aOut, bOut, cOut, sumBuf;

    IOMap ioA; ioA.bindInput("in", rawA).bindOutput("out", BufferType::I32, aOut);
    g.addNode(mockCpuKernel("k"), std::move(ioA), "mcpu:0");
    IOMap ioB; ioB.bindInput("in", rawB).bindOutput("out", BufferType::I32, bOut);
    g.addNode(mockCpuKernel("k"), std::move(ioB), "mcpu:1");
    IOMap ioC; ioC.bindInput("in", rawC).bindOutput("out", BufferType::I32, cOut);
    g.addNode(mockCpuKernel("k"), std::move(ioC), "mcpu:2");
    IOMap ioJ; ioJ.bindInput("a", aOut)
                  .bindInput("b", bOut)
                  .bindInput("c", cOut)
                  .bindOutput("out", BufferType::I32, sumBuf);
    g.addNode(cpuKernel("fanin"), std::move(ioJ), "cpu");

    std::vector<int32_t> a = {1, 2}, b = {3, 4}, c = {5, 6};
    mcpu0->setInputBuffer("rawA", a.data(), a.size() * sizeof(int32_t));
    mcpu1->setInputBuffer("rawB", b.data(), b.size() * sizeof(int32_t));
    mcpu2->setInputBuffer("rawC", c.data(), c.size() * sizeof(int32_t));

    g.compile().run();

    std::vector<int32_t> out(2);
    cpu->getOutputBuffer(sumBuf.name(), out.data(), out.size() * sizeof(int32_t));
    // (1+1) + (3+10) + (5+100) = 120
    // (2+1) + (4+10) + (6+100) = 123
    EXPECT_EQ(out[0], 120);
    EXPECT_EQ(out[1], 123);
}
