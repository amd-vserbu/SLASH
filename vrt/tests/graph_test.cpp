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
#include <vrt/graph/node/node.hpp>

using namespace vrt::graph;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static KernelDescriptor cpuKernel(std::string name, IOTypeMap ioType = {}) {
    return KernelDescriptor{std::move(name), DeviceType::CPU, std::nullopt, std::move(ioType)};
}

static KernelDescriptor mockCpuKernel(std::string name, IOTypeMap ioType = {}) {
    return KernelDescriptor{std::move(name), DeviceType::MOCK_CPU, std::nullopt, std::move(ioType)};
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

    cpu->registerKernel("copy", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i];
        }
    });

    GraphBuffer raw = graph.inputBuffer(BufferType::I32, "raw");

    IOMap io;
    GraphBuffer copied;
    io.bindInputBuffer("in", raw)
      .bindOutputBuffer("out", BufferType::I32, copied);
    graph.addNode(cpuKernel("copy"), std::move(io), "cpu");

    std::vector<int32_t> input = {7, 11, 13};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    ASSERT_NO_THROW(graph.run());

    std::vector<int32_t> output(input.size(), 0);
    cpu->getOutputBuffer(copied.name(), output.data(), output.size() * sizeof(int32_t));
    EXPECT_EQ(output, input);
}

// ===========================================================================
// MockCpuDevice — threaded IDevice for cross-device testing
// ===========================================================================

class MockCpuDevice : public IDevice {
   public:
    using KernelFn = std::function<void(const CpuKernelArgs&)>;

    explicit MockCpuDevice(std::string id)
        : id_(std::move(id)) {}

    ~MockCpuDevice() override {
        if (worker_.joinable()) worker_.join();
    }

    void registerKernel(std::string kernelName, KernelFn fn) {
        kernels_[std::move(kernelName)] = std::move(fn);
    }

    void setInputBuffer(const std::string& bufferName, const void* data, size_t sizeBytes) {
        auto& buf = buffers_[bufferName];
        buf.resize(sizeBytes);
        if (data && sizeBytes > 0) std::memcpy(buf.data(), data, sizeBytes);
    }

    void getOutputBuffer(const std::string& bufferName, void* data, size_t sizeBytes) const {
        auto it = buffers_.find(bufferName);
        if (it == buffers_.end())
            throw std::runtime_error("MockCpuDevice: unknown buffer '" + bufferName + "'");
        std::memcpy(data, it->second.data(), std::min(sizeBytes, it->second.size()));
    }

    size_t bufferSize(const std::string& bufferName) const {
        auto it = buffers_.find(bufferName);
        return (it == buffers_.end()) ? 0 : it->second.size();
    }

    // --- IDevice ---

    DeviceType  type() const override { return DeviceType::MOCK_CPU; }
    std::string id()   const override { return id_; }

    void compile(const DGraph& dg) override {
        steps_.clear();
        steps_.reserve(dg.nodes.size());
        for (const Node& node : dg.nodes) {
            std::visit(
                [&](const auto& n) {
                    using T = std::decay_t<decltype(n)>;
                    if constexpr (std::is_same_v<T, KernelNode>) {
                        steps_.push_back(KernelStep{n});
                    } else if constexpr (std::is_same_v<T, BridgeOpNode>) {
                        steps_.push_back(OpStep{n.tryReady, n.action});
                    }
                },
                node);
        }
    }

    void launch() override {
        if (worker_.joinable()) worker_.join();
        worker_ = std::thread([this] {
            for (const Step& step : steps_) {
                if (std::holds_alternative<KernelStep>(step)) {
                    executeKernel(std::get<KernelStep>(step).node);
                } else {
                    const auto& op = std::get<OpStep>(step);
                    while (!op.tryReady()) {}
                    op.action();
                }
            }
        });
    }

    void wait() override {
        if (worker_.joinable()) worker_.join();
    }

   private:
    struct OpStep {
        std::function<bool()> tryReady;
        std::function<void()> action;
    };
    struct KernelStep { KernelNode node; };
    using Step = std::variant<OpStep, KernelStep>;

    void executeKernel(const KernelNode& node) {
        auto it = kernels_.find(node.kernel.name);
        if (it == kernels_.end())
            throw std::runtime_error("MockCpuDevice: no kernel '" + node.kernel.name + "'");

        std::map<std::string, CpuBufferView> bufViews;

        for (const auto& [port, buf] : node.ioMap.inputBuffers()) {
            CpuBufferView v = resolveBuffer(buf.name());
            v.elementType   = buf.type();
            bufViews[port]  = v;
        }

        size_t defaultSize = 0;
        if (!node.ioMap.inputBuffers().empty()) {
            auto fit = buffers_.find(node.ioMap.inputBuffers().begin()->second.name());
            if (fit != buffers_.end()) defaultSize = fit->second.size();
        }

        for (const auto& [port, buf] : node.ioMap.outputBuffers()) {
            auto& storage = ensureBuffer(buf.name(), defaultSize);
            bufViews[port] = CpuBufferView{storage.data(), storage.size(), buf.type()};
        }

        for (const auto& rw : node.ioMap.rwBuffers()) {
            bufViews[rw.inPort] = resolveBuffer(rw.in.name());
            bufViews[rw.inPort].elementType = rw.in.type();
            auto& inStorage = buffers_.at(rw.in.name());
            buffers_[rw.out.name()] = inStorage;
            bufViews[rw.outPort] = CpuBufferView{
                buffers_[rw.out.name()].data(), buffers_[rw.out.name()].size(), rw.out.type()};
        }

        std::map<std::string, uint64_t> scalars;
        for (const auto& [port, gs] : node.ioMap.scalars()) {
            scalars[port] = gs.isConstant() ? gs.constantBits() : scalarStore_.at(gs.varName());
        }

        CpuKernelArgs args(std::move(bufViews), std::move(scalars));
        it->second(args);
    }

    CpuBufferView resolveBuffer(const std::string& name) const {
        auto it = buffers_.find(name);
        if (it == buffers_.end())
            throw std::runtime_error("MockCpuDevice: buffer '" + name + "' not found");
        return CpuBufferView{
            const_cast<void*>(static_cast<const void*>(it->second.data())),
            it->second.size(), BufferType::U8};
    }

    std::vector<uint8_t>& ensureBuffer(const std::string& name, size_t sizeBytes) {
        auto& buf = buffers_[name];
        if (buf.size() < sizeBytes) buf.resize(sizeBytes);
        return buf;
    }

    std::string                                  id_;
    std::map<std::string, KernelFn>              kernels_;
    std::map<std::string, std::vector<uint8_t>>  buffers_;
    std::map<std::string, uint64_t>              scalarStore_;

    std::vector<Step> steps_;
    std::thread       worker_;
};

// ===========================================================================
// Bridges — use SemaphorePool + host-staging, return BridgeStepPair so the
// compiler can splice them into the per-device DGraphs as BridgeOpNodes.
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

    const std::string bufName = buffer.name();

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
    cpu->registerKernel("add", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        auto offset = static_cast<int32_t>(args.scalar("offset"));
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] + offset;
        }
    });

    cpu->registerKernel("dbl", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = in[i] * 2;
        }
    });

    cpu->registerKernel("neg", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = -in[i];
        }
    });

    // Build graph.
    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // Node A: add offset=10
    IOMap ioA;
    GraphBuffer afterAdd;
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("out", BufferType::I32, afterAdd)
       .bindScalar("offset", GraphScalar::constant<int32_t>(10));
    g.addNode(cpuKernel("add"), std::move(ioA), "cpu");

    // Node B: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInputBuffer("in", afterAdd)
       .bindOutputBuffer("out", BufferType::I32, afterDbl);
    g.addNode(cpuKernel("dbl"), std::move(ioB), "cpu");

    // Node C: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInputBuffer("in", afterDbl)
       .bindOutputBuffer("out", BufferType::I32, afterNeg);
    g.addNode(cpuKernel("neg"), std::move(ioC), "cpu");

    // Provide input data.
    std::vector<int32_t> input = {1, 2, 3, 4};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    // Run.
    g.run();

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

    cpu->registerKernel("split", [](const CpuKernelArgs& args) {
        auto in   = args.buffer("in").as<const int32_t>();
        auto outL = args.buffer("left").as<int32_t>();
        auto outR = args.buffer("right").as<int32_t>();
        auto n    = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            outL[i] = in[i] + 1;
            outR[i] = in[i] * 10;
        }
    });

    cpu->registerKernel("passL", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i];
    });

    cpu->registerKernel("passR", [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i];
    });

    cpu->registerKernel("merge", [](const CpuKernelArgs& args) {
        auto left  = args.buffer("left").as<const int32_t>();
        auto right = args.buffer("right").as<const int32_t>();
        auto out   = args.buffer("out").as<int32_t>();
        auto n     = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) {
            out[i] = left[i] + right[i];
        }
    });

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    // A: split
    IOMap ioA;
    GraphBuffer leftBuf, rightBuf;
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("left", BufferType::I32, leftBuf)
       .bindOutputBuffer("right", BufferType::I32, rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B: pass left
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInputBuffer("in", leftBuf)
       .bindOutputBuffer("out", BufferType::I32, leftOut);
    g.addNode(cpuKernel("passL"), std::move(ioB), "cpu");

    // C: pass right
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInputBuffer("in", rightBuf)
       .bindOutputBuffer("out", BufferType::I32, rightOut);
    g.addNode(cpuKernel("passR"), std::move(ioC), "cpu");

    // D: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("left", leftOut)
       .bindInputBuffer("right", rightOut)
       .bindOutputBuffer("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("merge"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.run();

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
    EXPECT_THROW(g.run(), std::runtime_error);
}

TEST(GraphTest, NoDeviceThrows) {
    Graph g;
    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    GraphBuffer out;
    io.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, out);
    g.addNode(cpuKernel("k"), std::move(io));
    EXPECT_THROW(g.run(), std::runtime_error);
}

TEST(GraphTest, MissingDeviceHintThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    GraphBuffer out;
    io.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, out);
    g.addNode(cpuKernel("k"), std::move(io), "nonexistent");

    EXPECT_THROW(g.run(), std::runtime_error);
}

TEST(GraphTest, DuplicateInputBufferBindThrows) {
    Graph g;
    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");

    IOMap io;
    io.bindInputBuffer("in", raw);
    EXPECT_THROW(io.bindInputBuffer("in", raw), std::invalid_argument);
}

TEST(GraphTest, MissingMandatoryInputBufferPortThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputBuffers.push_back({"in", BufferType::I32});

    IOMap io;
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.validate(), std::runtime_error);
}

TEST(GraphTest, UnknownInputBufferPortThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputBuffers.push_back({"in", BufferType::I32});

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOMap io;
    io.bindInputBuffer("in", raw)
      .bindInputBuffer("extra", raw);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.validate(), std::runtime_error);
}

TEST(GraphTest, InputBufferTypeMismatchThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.inputBuffers.push_back({"in", BufferType::I32});

    GraphBuffer raw = g.inputBuffer(BufferType::U8, "raw");
    IOMap io;
    io.bindInputBuffer("in", raw);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.validate(), std::runtime_error);
}

TEST(GraphTest, OutputScalarMustUseGlobalVar) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOTypeMap ioType;
    ioType.outputScalars.push_back({"out", ScalarType::I32});

    IOMap io;
    io.bindScalar("out", GraphScalar::constant<int32_t>(1));
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.validate(), std::runtime_error);
}

TEST(GraphTest, InvalidAfterNodesReferenceThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    IOMap io;
    g.addNode(cpuKernel("typed"), std::move(io), "cpu", {"missing_node"});

    EXPECT_THROW(g.validate(), std::runtime_error);
}

TEST(GraphTest, CpuGlobalScalarRoundTrip) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", ScalarType::I32});
    ioType.outputScalars.push_back({"out", ScalarType::I32});

    cpu->registerKernel("scalar_copy", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("in"));
        args.setScalar("out", static_cast<uint64_t>(value + 1));
    });

    GraphScalar input = g.globalScalar(ScalarType::I32, "input");
    GraphScalar output = g.globalScalar(ScalarType::I32, "output");

    IOMap io;
    io.bindScalar("in", input)
      .bindScalar("out", output);
    g.addNode(cpuKernel("scalar_copy", ioType), std::move(io), "cpu");

    g.setScalar<int32_t>("input", 41);
    ASSERT_NO_THROW(g.run());

    EXPECT_EQ(g.getScalar<int32_t>("output"), 42);
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

    cpu->registerKernel("produce_scalar", [](const CpuKernelArgs& args) {
        args.setScalar("value", static_cast<uint64_t>(41));
    });
    cpu->registerKernel("consume_scalar", [](const CpuKernelArgs& args) {
        auto value = static_cast<int32_t>(args.scalar("value"));
        args.setScalar("result", static_cast<uint64_t>(value + 1));
    });

    GraphScalar value = g.globalScalar(ScalarType::I32, "value");
    GraphScalar result = g.globalScalar(ScalarType::I32, "result");

    IOMap consumeIo;
    consumeIo.bindScalar("value", value)
             .bindScalar("result", result);
    g.addNode(cpuKernel("consume_scalar", consumerType), std::move(consumeIo), "cpu");

    IOMap produceIo;
    produceIo.bindScalar("value", value);
    g.addNode(cpuKernel("produce_scalar", producerType), std::move(produceIo), "cpu");

    ASSERT_NO_THROW(g.run());
    EXPECT_EQ(g.getScalar<int32_t>("result"), 42);
}

TEST(GraphTest, UndeclaredGlobalScalarThrows) {
    Graph g = Graph::withDefaults();
    auto cpu = g.cpuDevice();
    ASSERT_NE(cpu, nullptr);

    IOTypeMap ioType;
    ioType.inputScalars.push_back({"in", ScalarType::I32});

    GraphScalar undeclared = GraphScalar::globalVar(ScalarType::I32, "missing");
    IOMap io;
    io.bindScalar("in", undeclared);
    g.addNode(cpuKernel("typed", ioType), std::move(io), "cpu");

    EXPECT_THROW(g.validate(), std::runtime_error);
}

// ===========================================================================
// Cross-device tests
// ===========================================================================

// Helper lambda factories (reused across devices)
static auto makeAddKernel(int32_t offset) {
    return [offset](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] + offset;
    };
}

static auto makeDblKernel() {
    return [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = in[i] * 2;
    };
}

static auto makeNegKernel() {
    return [](const CpuKernelArgs& args) {
        auto in  = args.buffer("in").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = -in[i];
    };
}

// ---------------------------------------------------------------------------
// Cross-device pipeline: cpu → mcpu:0 → mcpu:1 → cpu
// ---------------------------------------------------------------------------

TEST(GraphTest, CrossDevicePipeline) {
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    cpu->registerKernel("add10", makeAddKernel(10));
    mcpu0->registerKernel("dbl", makeDblKernel());
    mcpu1->registerKernel("neg", makeNegKernel());
    cpu->registerKernel("add1", makeAddKernel(1));

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
    ioA.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, afterAdd);
    g.addNode(cpuKernel("add10"), std::move(ioA), "cpu");

    // B on mcpu:0: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInputBuffer("in", afterAdd).bindOutputBuffer("out", BufferType::I32, afterDbl);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInputBuffer("in", afterDbl).bindOutputBuffer("out", BufferType::I32, afterNeg);
    g.addNode(mockCpuKernel("neg"), std::move(ioC), "mcpu:1");

    // D on cpu: add 1
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("in", afterNeg).bindOutputBuffer("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("add1"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3, 4};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.run();

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

    cpu->registerKernel("split", [](const CpuKernelArgs& args) {
        auto in = args.buffer("in").as<const int32_t>();
        auto l  = args.buffer("left").as<int32_t>();
        auto r  = args.buffer("right").as<int32_t>();
        auto n  = args.buffer("in").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) { l[i] = in[i] + 1; r[i] = in[i] * 10; }
    });
    mcpu0->registerKernel("dbl", makeDblKernel());
    mcpu1->registerKernel("dbl", makeDblKernel());
    cpu->registerKernel("merge", [](const CpuKernelArgs& args) {
        auto l   = args.buffer("left").as<const int32_t>();
        auto r   = args.buffer("right").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = l[i] + r[i];
    });

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
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("left", BufferType::I32, leftBuf)
       .bindOutputBuffer("right", BufferType::I32, rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B on mcpu:0: double the left branch
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInputBuffer("in", leftBuf).bindOutputBuffer("out", BufferType::I32, leftOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: double the right branch
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInputBuffer("in", rightBuf).bindOutputBuffer("out", BufferType::I32, rightOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioC), "mcpu:1");

    // D on cpu: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("left", leftOut)
       .bindInputBuffer("right", rightOut)
       .bindOutputBuffer("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("merge"), std::move(ioD), "cpu");

    std::vector<int32_t> input = {1, 2, 3};
    cpu->setInputBuffer("raw", input.data(), input.size() * sizeof(int32_t));

    g.run();

    // left = (x+1)*2, right = (x*10)*2, merge = 2(x+1) + 2(10x) = 22x + 2
    std::vector<int32_t> output(3);
    cpu->getOutputBuffer(finalBuf.name(), output.data(), output.size() * sizeof(int32_t));

    EXPECT_EQ(output[0], 24);   // 22*1+2
    EXPECT_EQ(output[1], 46);   // 22*2+2
    EXPECT_EQ(output[2], 68);   // 22*3+2
}

// ===========================================================================
// Phase-1: compiler populates Node::dependsOn on every DGraph node.
// ===========================================================================

namespace {

const DGraph* findDg(const std::vector<DGraph>& dgs, const std::string& id) {
    for (const auto& dg : dgs) if (dg.deviceId == id) return &dg;
    return nullptr;
}

const Node* findNode(const DGraph& dg, const std::string& id) {
    for (const auto& n : dg.nodes) if (nodeId(n) == id) return &n;
    return nullptr;
}

bool depsContain(const Node& n, const std::string& id) {
    const auto& d = nodeDependsOn(n);
    return std::find(d.begin(), d.end(), id) != d.end();
}

}  // namespace

TEST(GraphTest, CompilerPopulatesDependsOnAcrossDevices) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    cpu ->registerKernel("add10", makeAddKernel(10));
    mcpu->registerKernel("dbl",   makeDblKernel());
    cpu ->registerKernel("add1",  makeAddKernel(1));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2, b3;

    IOMap m1; m1.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, b1);
    auto idA = g.addNode(cpuKernel("add10"), std::move(m1), "cpu");

    IOMap m2; m2.bindInputBuffer("in", b1).bindOutputBuffer("out", BufferType::I32, b2);
    auto idB = g.addNode(mockCpuKernel("dbl"), std::move(m2), "mcpu:0");

    IOMap m3; m3.bindInputBuffer("in", b2).bindOutputBuffer("out", BufferType::I32, b3);
    auto idC = g.addNode(cpuKernel("add1"), std::move(m3), "cpu");

    std::vector<int32_t> in = {1};
    cpu->setInputBuffer("raw", in.data(), sizeof(int32_t));
    g.run();

    const auto* dgCpu = findDg(g.dgraphs(), "cpu");
    const auto* dgM   = findDg(g.dgraphs(), "mcpu:0");
    ASSERT_NE(dgCpu, nullptr);
    ASSERT_NE(dgM,   nullptr);

    // Kernel A (cpu) is a graph-level input → no predecessors.
    const Node* nA = findNode(*dgCpu, idA);
    ASSERT_NE(nA, nullptr);
    EXPECT_TRUE(nodeDependsOn(*nA).empty());

    // Kernel B (mcpu:0) consumes b1 produced by A (cpu) → its dep must be the
    // consumer-side bridge op anchored to B, NOT idA itself.
    const Node* nB = findNode(*dgM, idB);
    ASSERT_NE(nB, nullptr);
    EXPECT_FALSE(depsContain(*nB, idA));
    bool foundConsumerForB = false;
    for (const auto& depId : nodeDependsOn(*nB)) {
        const Node* dep = findNode(*dgM, depId);
        ASSERT_NE(dep, nullptr) << "dependsOn references unknown id " << depId;
        if (std::holds_alternative<BridgeOpNode>(*dep)) {
            const auto& b = std::get<BridgeOpNode>(*dep);
            EXPECT_EQ(b.side, BridgeOpNode::Side::Consumer);
            EXPECT_EQ(b.pairedKernelId, idB);
            foundConsumerForB = true;
        }
    }
    EXPECT_TRUE(foundConsumerForB);

    // Producer-side bridge op on cpu (anchored to A) → dependsOn must
    // contain exactly { idA }.
    bool foundProducerForA = false;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<BridgeOpNode>(n)) continue;
        const auto& b = std::get<BridgeOpNode>(n);
        if (b.side == BridgeOpNode::Side::Producer && b.pairedKernelId == idA) {
            foundProducerForA = true;
            ASSERT_EQ(b.dependsOn.size(), 1u);
            EXPECT_EQ(b.dependsOn[0], idA);
        }
    }
    EXPECT_TRUE(foundProducerForA);

    // Kernel C (cpu) consumes b2 produced by B (mcpu:0) → depends on the
    // consumer-side bridge op on cpu anchored to C.
    const Node* nC = findNode(*dgCpu, idC);
    ASSERT_NE(nC, nullptr);
    bool foundConsumerForC = false;
    for (const auto& depId : nodeDependsOn(*nC)) {
        const Node* dep = findNode(*dgCpu, depId);
        ASSERT_NE(dep, nullptr);
        if (std::holds_alternative<BridgeOpNode>(*dep) &&
            std::get<BridgeOpNode>(*dep).side == BridgeOpNode::Side::Consumer &&
            std::get<BridgeOpNode>(*dep).pairedKernelId == idC) {
            foundConsumerForC = true;
        }
    }
    EXPECT_TRUE(foundConsumerForC);
}

TEST(GraphTest, CompilerHonoursCrossDeviceAfterNodesViaBarrier) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    cpu ->registerKernel("a", makeAddKernel(0));
    mcpu->registerKernel("b", makeDblKernel());

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer rawB = g.inputBuffer(BufferType::I32, "rawB");
    GraphBuffer outA, outB;

    IOMap mA; mA.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, outA);
    auto idA = g.addNode(cpuKernel("a"), std::move(mA), "cpu");

    // B has its own input from rawB and a CROSS-DEVICE afterNodes={idA}.
    IOMap mB; mB.bindInputBuffer("in", rawB).bindOutputBuffer("out", BufferType::I32, outB);
    auto idB = g.addNode(mockCpuKernel("b"), std::move(mB), "mcpu:0",
                         /*afterNodes=*/{idA});

    std::vector<int32_t> in = {1};
    cpu->setInputBuffer("raw",  in.data(), sizeof(int32_t));
    mcpu->setInputBuffer("rawB", in.data(), sizeof(int32_t));
    g.run();

    // Phase 2: cross-device afterNodes materialises a barrier op pair.
    // - On 'cpu' (source side): a Producer-side barrier with
    //   pairedKernelId == idA, dependsOn == {idA}.
    // - On 'mcpu:0' (dest side): a Consumer-side barrier with
    //   pairedKernelId == idB; B.dependsOn must contain its id.
    const auto* dgCpu = findDg(g.dgraphs(), "cpu");
    const auto* dgM   = findDg(g.dgraphs(), "mcpu:0");
    ASSERT_NE(dgCpu, nullptr);
    ASSERT_NE(dgM, nullptr);

    const BridgeOpNode* prodBarrier = nullptr;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<BridgeOpNode>(n)) continue;
        const auto& b = std::get<BridgeOpNode>(n);
        if (b.side == BridgeOpNode::Side::Producer &&
            b.op && b.op->label() == "barrier" &&
            b.pairedKernelId == idA) {
            prodBarrier = &b;
            break;
        }
    }
    ASSERT_NE(prodBarrier, nullptr);
    EXPECT_EQ(prodBarrier->dependsOn, std::vector<std::string>{idA});

    const BridgeOpNode* consBarrier = nullptr;
    for (const auto& n : dgM->nodes) {
        if (!std::holds_alternative<BridgeOpNode>(n)) continue;
        const auto& b = std::get<BridgeOpNode>(n);
        if (b.side == BridgeOpNode::Side::Consumer &&
            b.op && b.op->label() == "barrier" &&
            b.pairedKernelId == idB) {
            consBarrier = &b;
            break;
        }
    }
    ASSERT_NE(consBarrier, nullptr);

    const Node* nB = findNode(*dgM, idB);
    ASSERT_NE(nB, nullptr);
    EXPECT_TRUE(depsContain(*nB, consBarrier->id));
}

TEST(GraphTest, CompilerChainsBounceLegsViaDependsOn) {
    // No direct mock<->mock bridge → router bounces via cpu, producing two
    // legs. The compiler must wire leg2.producer.dependsOn += leg1.consumer.id.
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    mcpu0->registerKernel("dbl", makeDblKernel());
    mcpu1->registerKernel("neg", makeNegKernel());

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);
    // NOTE: no MockCpuMockCpuBridge → forces bounce through cpu.

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2;

    IOMap m1; m1.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, b1);
    g.addNode(mockCpuKernel("dbl"), std::move(m1), "mcpu:0");

    IOMap m2; m2.bindInputBuffer("in", b1).bindOutputBuffer("out", BufferType::I32, b2);
    g.addNode(mockCpuKernel("neg"), std::move(m2), "mcpu:1");

    std::vector<int32_t> in = {1};
    mcpu0->setInputBuffer("raw", in.data(), sizeof(int32_t));
    g.run();

    // The bounce intermediary lives on the cpu DGraph: there must be one
    // BridgeOpNode pair (consumer of leg1, producer of leg2) where the
    // producer's dependsOn includes the consumer's id.
    const auto* dgCpu = findDg(g.dgraphs(), "cpu");
    ASSERT_NE(dgCpu, nullptr);

    bool chained = false;
    for (const auto& n : dgCpu->nodes) {
        if (!std::holds_alternative<BridgeOpNode>(n)) continue;
        const auto& b = std::get<BridgeOpNode>(n);
        if (b.side != BridgeOpNode::Side::Producer) continue;
        for (const auto& depId : b.dependsOn) {
            const Node* dep = findNode(*dgCpu, depId);
            if (!dep) continue;
            if (std::holds_alternative<BridgeOpNode>(*dep) &&
                std::get<BridgeOpNode>(*dep).side == BridgeOpNode::Side::Consumer) {
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

    IBridge& a = g.bridgeFor("mcpu:0", "mcpu:1");
    IBridge& b = g.bridgeFor("mcpu:1", "mcpu:0");
    IBridge& c = g.bridgeFor("mcpu:0", "mcpu:1");  // cached
    EXPECT_NE(&a, &b);
    EXPECT_EQ(&a, &c);
}

TEST(GraphTest, CompilerErrorsWhenNoBridgeFactoryRegistered) {
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");

    mcpu0->registerKernel("dbl", makeDblKernel());
    mcpu1->registerKernel("neg", makeNegKernel());

    Graph g;
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    // No bridge factories registered at all → validateBridges() fails first.

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer b1, b2;

    IOMap m1; m1.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, b1);
    g.addNode(mockCpuKernel("dbl"), std::move(m1), "mcpu:0");
    IOMap m2; m2.bindInputBuffer("in", b1).bindOutputBuffer("out", BufferType::I32, b2);
    g.addNode(mockCpuKernel("neg"), std::move(m2), "mcpu:1");

    std::vector<int32_t> in = {1};
    mcpu0->setInputBuffer("raw", in.data(), sizeof(int32_t));
    EXPECT_THROW(g.run(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Phase 3 — CpuDevice dep-driven executor tests
// ---------------------------------------------------------------------------

// Pure-CPU fan-out / fan-in: root → {A, B} → join.
// Verifies the dep-driven scheduler executes both branches and that the
// join kernel observes both inputs.
TEST(GraphTest, CpuExecutorRunsDiamondAcrossTwoBranches) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel("root", makeAddKernel(0));      // identity copy
    cpu->registerKernel("addA", makeAddKernel(100));
    cpu->registerKernel("addB", makeAddKernel(200));
    cpu->registerKernel("join", [](const CpuKernelArgs& args) {
        auto a   = args.buffer("a").as<const int32_t>();
        auto b   = args.buffer("b").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n   = args.buffer("a").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
    });

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw  = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer rOut, aOut, bOut, joined;

    IOMap ioR; ioR.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, rOut);
    g.addNode(cpuKernel("root"), std::move(ioR), "cpu");
    IOMap ioA; ioA.bindInputBuffer("in", rOut).bindOutputBuffer("out", BufferType::I32, aOut);
    g.addNode(cpuKernel("addA"), std::move(ioA), "cpu");
    IOMap ioB; ioB.bindInputBuffer("in", rOut).bindOutputBuffer("out", BufferType::I32, bOut);
    g.addNode(cpuKernel("addB"), std::move(ioB), "cpu");
    IOMap ioJ; ioJ.bindInputBuffer("a", aOut)
                  .bindInputBuffer("b", bOut)
                  .bindOutputBuffer("out", BufferType::I32, joined);
    g.addNode(cpuKernel("join"), std::move(ioJ), "cpu");

    std::vector<int32_t> in = {1, 2, 3};
    cpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    g.run();

    std::vector<int32_t> out(3);
    cpu->getOutputBuffer(joined.name(), out.data(), out.size() * sizeof(int32_t));
    // (x+100) + (x+200) = 2x + 300
    EXPECT_EQ(out[0], 302);
    EXPECT_EQ(out[1], 304);
    EXPECT_EQ(out[2], 306);
}

TEST(GraphTest, CpuExecutorResetsDependencyStateAcrossRuns) {
    auto cpu = std::make_shared<CpuDevice>("cpu");

    cpu->registerKernel("copy", makeAddKernel(0));
    cpu->registerKernel("add1", makeAddKernel(1));

    Graph g;
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer mid, finalBuf;

    IOMap ioA; ioA.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, mid);
    g.addNode(cpuKernel("copy"), std::move(ioA), "cpu");

    IOMap ioB; ioB.bindInputBuffer("in", mid).bindOutputBuffer("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("add1"), std::move(ioB), "cpu");

    auto runOnce = [&](int32_t value) {
        cpu->setInputBuffer("raw", &value, sizeof(value));
        g.run();
        int32_t out = 0;
        cpu->getOutputBuffer(finalBuf.name(), &out, sizeof(out));
        return out;
    };

    EXPECT_EQ(runOnce(1), 2);
    EXPECT_EQ(runOnce(5), 6);
    EXPECT_EQ(runOnce(9), 10);
}

// Cross-device data flow: CPU consumer kernel must wait on a producer
// running on MockCpuDevice. If the consumer fires before the producer
// signals, the read-back will be wrong (zero-init / stale).
TEST(GraphTest, CpuExecutorBlocksConsumerUntilProducerSignals) {
    auto cpu  = std::make_shared<CpuDevice>("cpu");
    auto mcpu = std::make_shared<MockCpuDevice>("mcpu:0");

    mcpu->registerKernel("dbl", makeDblKernel());
    cpu->registerKernel("sink", makeAddKernel(7));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer dbld, finalBuf;

    IOMap ioM; ioM.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, dbld);
    g.addNode(mockCpuKernel("dbl"), std::move(ioM), "mcpu:0");
    IOMap ioS; ioS.bindInputBuffer("in", dbld).bindOutputBuffer("out", BufferType::I32, finalBuf);
    g.addNode(cpuKernel("sink"), std::move(ioS), "cpu");

    std::vector<int32_t> in = {3, 5, 9};
    mcpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    g.run();

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

    mcpu->registerKernel("dbl", makeDblKernel());
    cpu->registerKernel("add1", makeAddKernel(1));
    cpu->registerKernel("add2", makeAddKernel(2));
    cpu->registerKernel("sum", [](const CpuKernelArgs& args) {
        auto left  = args.buffer("left").as<const int32_t>();
        auto right = args.buffer("right").as<const int32_t>();
        auto out   = args.buffer("out").as<int32_t>();
        auto n     = args.buffer("left").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = left[i] + right[i];
    });

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu);
    registerCpuLikeFactory<CpuMockCpuBridge>(g, DeviceType::CPU, DeviceType::MOCK_CPU);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    GraphBuffer sharedBuf, leftBuf, rightBuf, sumBuf;

    IOMap ioP; ioP.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::I32, sharedBuf);
    g.addNode(mockCpuKernel("dbl"), std::move(ioP), "mcpu:0");

    IOMap ioL; ioL.bindInputBuffer("in", sharedBuf).bindOutputBuffer("out", BufferType::I32, leftBuf);
    auto idL = g.addNode(cpuKernel("add1"), std::move(ioL), "cpu");

    IOMap ioR; ioR.bindInputBuffer("in", sharedBuf).bindOutputBuffer("out", BufferType::I32, rightBuf);
    auto idR = g.addNode(cpuKernel("add2"), std::move(ioR), "cpu");

    IOMap ioS; ioS.bindInputBuffer("left", leftBuf)
                  .bindInputBuffer("right", rightBuf)
                  .bindOutputBuffer("out", BufferType::I32, sumBuf);
    g.addNode(cpuKernel("sum"), std::move(ioS), "cpu");

    std::vector<int32_t> in = {3, 5};
    mcpu->setInputBuffer("raw", in.data(), in.size() * sizeof(int32_t));

    ASSERT_NO_THROW(g.run());

    std::vector<int32_t> out(2);
    cpu->getOutputBuffer(sumBuf.name(), out.data(), out.size() * sizeof(int32_t));
    EXPECT_EQ(out[0], 15);  // (3*2 + 1) + (3*2 + 2)
    EXPECT_EQ(out[1], 23);  // (5*2 + 1) + (5*2 + 2)

    const auto* dgCpu = findDg(g.dgraphs(), "cpu");
    ASSERT_NE(dgCpu, nullptr);

    const Node* nL = findNode(*dgCpu, idL);
    const Node* nR = findNode(*dgCpu, idR);
    ASSERT_NE(nL, nullptr);
    ASSERT_NE(nR, nullptr);

    std::string sharedBridgeId;
    for (const auto& depId : nodeDependsOn(*nL)) {
        const Node* dep = findNode(*dgCpu, depId);
        if (!dep || !std::holds_alternative<BridgeOpNode>(*dep)) continue;
        const auto& bridge = std::get<BridgeOpNode>(*dep);
        if (bridge.side == BridgeOpNode::Side::Consumer) {
            sharedBridgeId = bridge.id;
            break;
        }
    }

    ASSERT_FALSE(sharedBridgeId.empty());
    EXPECT_TRUE(depsContain(*nR, sharedBridgeId));
}

// Three independent producers on three mock devices feed a single CPU
// fan-in kernel. The CPU executor must service all three pending
// consumer-side ops regardless of the order their semaphores fire.
TEST(GraphTest, CpuExecutorMultipleConsumersOutOfOrderSignals) {
    auto cpu   = std::make_shared<CpuDevice>("cpu");
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0");
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1");
    auto mcpu2 = std::make_shared<MockCpuDevice>("mcpu:2");

    mcpu0->registerKernel("k", makeAddKernel(1));
    mcpu1->registerKernel("k", makeAddKernel(10));
    mcpu2->registerKernel("k", makeAddKernel(100));
    cpu->registerKernel("fanin", [](const CpuKernelArgs& args) {
        auto a = args.buffer("a").as<const int32_t>();
        auto b = args.buffer("b").as<const int32_t>();
        auto c = args.buffer("c").as<const int32_t>();
        auto out = args.buffer("out").as<int32_t>();
        auto n = args.buffer("a").sizeBytes / sizeof(int32_t);
        for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i] + c[i];
    });

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

    IOMap ioA; ioA.bindInputBuffer("in", rawA).bindOutputBuffer("out", BufferType::I32, aOut);
    g.addNode(mockCpuKernel("k"), std::move(ioA), "mcpu:0");
    IOMap ioB; ioB.bindInputBuffer("in", rawB).bindOutputBuffer("out", BufferType::I32, bOut);
    g.addNode(mockCpuKernel("k"), std::move(ioB), "mcpu:1");
    IOMap ioC; ioC.bindInputBuffer("in", rawC).bindOutputBuffer("out", BufferType::I32, cOut);
    g.addNode(mockCpuKernel("k"), std::move(ioC), "mcpu:2");
    IOMap ioJ; ioJ.bindInputBuffer("a", aOut)
                  .bindInputBuffer("b", bOut)
                  .bindInputBuffer("c", cOut)
                  .bindOutputBuffer("out", BufferType::I32, sumBuf);
    g.addNode(cpuKernel("fanin"), std::move(ioJ), "cpu");

    std::vector<int32_t> a = {1, 2}, b = {3, 4}, c = {5, 6};
    mcpu0->setInputBuffer("rawA", a.data(), a.size() * sizeof(int32_t));
    mcpu1->setInputBuffer("rawB", b.data(), b.size() * sizeof(int32_t));
    mcpu2->setInputBuffer("rawC", c.data(), c.size() * sizeof(int32_t));

    g.run();

    std::vector<int32_t> out(2);
    cpu->getOutputBuffer(sumBuf.name(), out.data(), out.size() * sizeof(int32_t));
    // (1+1) + (3+10) + (5+100) = 120
    // (2+1) + (4+10) + (6+100) = 123
    EXPECT_EQ(out[0], 120);
    EXPECT_EQ(out[1], 123);
}
