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

// ===========================================================================
// MockCpuDevice — threaded IDevice for cross-device testing
// ===========================================================================

class MockCpuDevice : public IDevice {
   public:
    using KernelFn = std::function<void(const CpuKernelArgs&)>;

    explicit MockCpuDevice(std::string id, std::shared_ptr<SemaphorePool> pool,
                           std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> buffers = nullptr)
        : id_(std::move(id)),
          semPool_(pool ? std::move(pool) : std::make_shared<SemaphorePool>()),
          buffers_(buffers ? std::move(buffers)
                          : std::make_shared<std::map<std::string, std::vector<uint8_t>>>()) {}

    ~MockCpuDevice() override {
        if (worker_.joinable()) worker_.join();
    }

    void registerKernel(std::string kernelName, KernelFn fn) {
        kernels_[std::move(kernelName)] = std::move(fn);
    }

    void setInputBuffer(const std::string& bufferName, const void* data, size_t sizeBytes) {
        auto& buf = (*buffers_)[bufferName];
        buf.resize(sizeBytes);
        if (data && sizeBytes > 0) std::memcpy(buf.data(), data, sizeBytes);
    }

    void getOutputBuffer(const std::string& bufferName, void* data, size_t sizeBytes) const {
        auto it = buffers_->find(bufferName);
        if (it == buffers_->end())
            throw std::runtime_error("MockCpuDevice: unknown buffer '" + bufferName + "'");
        std::memcpy(data, it->second.data(), std::min(sizeBytes, it->second.size()));
    }

    // --- IDevice ---

    DeviceType  type() const override { return DeviceType::MOCK_CPU; }
    std::string id()   const override { return id_; }

    void compile(const DGraph& dg) override {
        steps_.clear();
        std::vector<Step> ordered;

        for (const Node& node : dg.nodes) {
            for (const auto& a : pendingAwaits_) {
                if (a.beforeNodeId == node.id)
                    ordered.push_back(AwaitStep{a.sem});
            }
            for (const auto& d : pendingDMAs_) {
                if (d.beforeNodeId == node.id)
                    ordered.push_back(DMAStep{d.dma});
            }

            ordered.push_back(KernelStep{node});

            for (const auto& s : pendingSignals_) {
                if (s.afterNodeId == node.id)
                    ordered.push_back(SignalStep{s.sem});
            }
        }

        steps_ = std::move(ordered);
        pendingSignals_.clear();
        pendingAwaits_.clear();
        pendingDMAs_.clear();
    }

    void insertSignal(SemaphoreHandle sem, const std::string& afterNodeId) override {
        pendingSignals_.push_back({sem, afterNodeId});
    }
    void insertAwait(SemaphoreHandle sem, const std::string& beforeNodeId)  override {
        pendingAwaits_.push_back({sem, beforeNodeId});
    }
    void insertDMA(DMADescriptor dma, const std::string& beforeNodeId)      override {
        pendingDMAs_.push_back({std::move(dma), beforeNodeId});
    }

    void launch() override {
        if (worker_.joinable()) worker_.join();
        worker_ = std::thread([this] {
            for (const Step& step : steps_) {
                if (std::holds_alternative<KernelStep>(step)) {
                    executeKernel(std::get<KernelStep>(step).node);
                } else if (std::holds_alternative<SignalStep>(step)) {
                    semPool_->signal(std::get<SignalStep>(step).sem);
                } else if (std::holds_alternative<AwaitStep>(step)) {
                    semPool_->await(std::get<AwaitStep>(step).sem);
                } else if (std::holds_alternative<DMAStep>(step)) {
                    executeDMA(std::get<DMAStep>(step).dma);
                }
            }
        });
    }

    void wait() override {
        if (worker_.joinable()) worker_.join();
    }

   private:
    struct SignalStep { SemaphoreHandle sem; };
    struct AwaitStep  { SemaphoreHandle sem; };
    struct DMAStep    { DMADescriptor   dma; };
    struct KernelStep { Node node; };
    using Step = std::variant<SignalStep, AwaitStep, DMAStep, KernelStep>;

    void executeKernel(const Node& node) {
        auto it = kernels_.find(node.kernel.name);
        if (it == kernels_.end())
            throw std::runtime_error("MockCpuDevice: no kernel '" + node.kernel.name + "'");

        std::map<std::string, CpuBufferView> bufViews;

        for (const auto& [port, buf] : node.ioMap.inputBuffers()) {
            bufViews[port] = resolveBuffer(buf.name());
        }

        size_t defaultSize = 0;
        if (!node.ioMap.inputBuffers().empty()) {
            auto fit = buffers_->find(node.ioMap.inputBuffers().begin()->second.name());
            if (fit != buffers_->end()) defaultSize = fit->second.size();
        }

        for (const auto& [port, buf] : node.ioMap.outputBuffers()) {
            auto& storage = ensureBuffer(buf.name(), defaultSize);
            bufViews[port] = CpuBufferView{storage.data(), storage.size(), BufferType::U8};
        }

        for (const auto& rw : node.ioMap.rwBuffers()) {
            bufViews[rw.inPort] = resolveBuffer(rw.in.name());
            auto& inStorage = buffers_->at(rw.in.name());
            (*buffers_)[rw.out.name()] = inStorage;
            bufViews[rw.outPort] = CpuBufferView{
                (*buffers_)[rw.out.name()].data(), (*buffers_)[rw.out.name()].size(), BufferType::U8};
        }

        std::map<std::string, uint64_t> scalars;
        for (const auto& [port, gs] : node.ioMap.scalars()) {
            scalars[port] = gs.isConstant() ? gs.constantBits() : scalarStore_.at(gs.varName());
        }

        CpuKernelArgs args(std::move(bufViews), std::move(scalars));
        it->second(args);
    }

    void executeDMA(const DMADescriptor& dma) {
        auto srcIt = buffers_->find(dma.src.name());
        if (srcIt == buffers_->end())
            throw std::runtime_error("MockCpuDevice: DMA src '" + dma.src.name() + "' not found");
        auto& dst = ensureBuffer(dma.dst.name(), dma.sizeBytes);
        std::memcpy(dst.data(), srcIt->second.data(),
                    std::min(dma.sizeBytes, srcIt->second.size()));
    }

    CpuBufferView resolveBuffer(const std::string& name) const {
        auto it = buffers_->find(name);
        if (it == buffers_->end())
            throw std::runtime_error("MockCpuDevice: buffer '" + name + "' not found");
        return CpuBufferView{
            const_cast<void*>(static_cast<const void*>(it->second.data())),
            it->second.size(), BufferType::U8};
    }

    std::vector<uint8_t>& ensureBuffer(const std::string& name, size_t sizeBytes) {
        auto& buf = (*buffers_)[name];
        if (buf.size() < sizeBytes) buf.resize(sizeBytes);
        return buf;
    }

    std::string                                  id_;
    std::shared_ptr<SemaphorePool>               semPool_;
    std::map<std::string, KernelFn>              kernels_;
    std::shared_ptr<std::map<std::string, std::vector<uint8_t>>>  buffers_;
    std::map<std::string, uint64_t>              scalarStore_;

    struct PendingSignal { SemaphoreHandle sem; std::string afterNodeId; };
    struct PendingAwait  { SemaphoreHandle sem; std::string beforeNodeId; };
    struct PendingDMA    { DMADescriptor   dma; std::string beforeNodeId; };

    std::vector<PendingSignal> pendingSignals_;
    std::vector<PendingAwait>  pendingAwaits_;
    std::vector<PendingDMA>    pendingDMAs_;

    std::vector<Step> steps_;
    std::thread       worker_;
};

// ===========================================================================
// Bridges
// ===========================================================================

class CpuMockCpuBridge : public IBridge {
   public:
    std::pair<DeviceType, DeviceType> devicePair() const override {
        return IBridge::makeKey(DeviceType::CPU, DeviceType::MOCK_CPU);
    }

    void injectTransfer(IDevice& src, IDevice& dst,
                        const GraphBuffer& buffer, uint64_t sizeHintBytes,
                        std::function<SemaphoreHandle()> allocSemaphore,
                        const std::string& producerNodeId,
                        const std::string& consumerNodeId) override {
        SemaphoreHandle sem = allocSemaphore();
        src.insertSignal(sem, producerNodeId);
        dst.insertAwait(sem, consumerNodeId);
        DMADescriptor dma{buffer, buffer, sizeHintBytes};
        if (src.prefersDMAInitiation()) src.insertDMA(dma, producerNodeId);
        else                            dst.insertDMA(dma, consumerNodeId);
    }
};

class MockCpuMockCpuBridge : public IBridge {
   public:
    std::pair<DeviceType, DeviceType> devicePair() const override {
        return IBridge::makeKey(DeviceType::MOCK_CPU, DeviceType::MOCK_CPU);
    }

    void injectTransfer(IDevice& src, IDevice& dst,
                        const GraphBuffer& buffer, uint64_t sizeHintBytes,
                        std::function<SemaphoreHandle()> allocSemaphore,
                        const std::string& producerNodeId,
                        const std::string& consumerNodeId) override {
        SemaphoreHandle sem = allocSemaphore();
        src.insertSignal(sem, producerNodeId);
        dst.insertAwait(sem, consumerNodeId);
        DMADescriptor dma{buffer, buffer, sizeHintBytes};
        if (src.prefersDMAInitiation()) src.insertDMA(dma, producerNodeId);
        else                            dst.insertDMA(dma, consumerNodeId);
    }
};

// ---------------------------------------------------------------------------
// 3-node pipeline: add → double → negate
// ---------------------------------------------------------------------------

TEST(GraphTest, ThreeNodePipeline) {
    auto pool = std::make_shared<SemaphorePool>();
    auto cpu  = std::make_shared<CpuDevice>("cpu", pool);

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

    GraphBuffer raw = g.inputBuffer("raw");

    // Node A: add offset=10
    IOMap ioA;
    GraphBuffer afterAdd;
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("out", afterAdd)
       .bindScalar("offset", GraphScalar::constant(ScalarType::I32, 10));
    g.addNode(cpuKernel("add"), std::move(ioA), "cpu");

    // Node B: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInputBuffer("in", afterAdd)
       .bindOutputBuffer("out", afterDbl);
    g.addNode(cpuKernel("dbl"), std::move(ioB), "cpu");

    // Node C: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInputBuffer("in", afterDbl)
       .bindOutputBuffer("out", afterNeg);
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
    auto pool = std::make_shared<SemaphorePool>();
    auto cpu  = std::make_shared<CpuDevice>("cpu", pool);

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

    GraphBuffer raw = g.inputBuffer("raw");

    // A: split
    IOMap ioA;
    GraphBuffer leftBuf, rightBuf;
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("left", leftBuf)
       .bindOutputBuffer("right", rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B: pass left
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInputBuffer("in", leftBuf)
       .bindOutputBuffer("out", leftOut);
    g.addNode(cpuKernel("passL"), std::move(ioB), "cpu");

    // C: pass right
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInputBuffer("in", rightBuf)
       .bindOutputBuffer("out", rightOut);
    g.addNode(cpuKernel("passR"), std::move(ioC), "cpu");

    // D: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("left", leftOut)
       .bindInputBuffer("right", rightOut)
       .bindOutputBuffer("out", finalBuf);
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
    GraphBuffer raw = g.inputBuffer("raw");
    IOMap io;
    GraphBuffer out;
    io.bindInputBuffer("in", raw).bindOutputBuffer("out", out);
    g.addNode(cpuKernel("k"), std::move(io));
    EXPECT_THROW(g.run(), std::runtime_error);
}

TEST(GraphTest, MissingDeviceHintThrows) {
    Graph g;
    auto cpu = std::make_shared<CpuDevice>("cpu");
    g.registerDevice(cpu);

    GraphBuffer raw = g.inputBuffer("raw");
    IOMap io;
    GraphBuffer out;
    io.bindInputBuffer("in", raw).bindOutputBuffer("out", out);
    g.addNode(cpuKernel("k"), std::move(io), "nonexistent");

    EXPECT_THROW(g.run(), std::runtime_error);
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
    auto pool  = std::make_shared<SemaphorePool>();
    auto bufs  = std::make_shared<std::map<std::string, std::vector<uint8_t>>>();
    auto cpu   = std::make_shared<CpuDevice>("cpu", pool, bufs);
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0", pool, bufs);
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1", pool, bufs);

    cpu->registerKernel("add10", makeAddKernel(10));
    mcpu0->registerKernel("dbl", makeDblKernel());
    mcpu1->registerKernel("neg", makeNegKernel());
    cpu->registerKernel("add1", makeAddKernel(1));

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mcpu0);
    g.registerDevice(mcpu1);
    g.registerBridge(std::make_shared<CpuMockCpuBridge>());
    g.registerBridge(std::make_shared<MockCpuMockCpuBridge>());

    GraphBuffer raw = g.inputBuffer("raw");

    // A on cpu: add 10
    IOMap ioA;
    GraphBuffer afterAdd;
    ioA.bindInputBuffer("in", raw).bindOutputBuffer("out", afterAdd);
    g.addNode(cpuKernel("add10"), std::move(ioA), "cpu");

    // B on mcpu:0: double
    IOMap ioB;
    GraphBuffer afterDbl;
    ioB.bindInputBuffer("in", afterAdd).bindOutputBuffer("out", afterDbl);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: negate
    IOMap ioC;
    GraphBuffer afterNeg;
    ioC.bindInputBuffer("in", afterDbl).bindOutputBuffer("out", afterNeg);
    g.addNode(mockCpuKernel("neg"), std::move(ioC), "mcpu:1");

    // D on cpu: add 1
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("in", afterNeg).bindOutputBuffer("out", finalBuf);
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
    auto pool  = std::make_shared<SemaphorePool>();
    auto bufs  = std::make_shared<std::map<std::string, std::vector<uint8_t>>>();
    auto cpu   = std::make_shared<CpuDevice>("cpu", pool, bufs);
    auto mcpu0 = std::make_shared<MockCpuDevice>("mcpu:0", pool, bufs);
    auto mcpu1 = std::make_shared<MockCpuDevice>("mcpu:1", pool, bufs);

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
    g.registerBridge(std::make_shared<CpuMockCpuBridge>());
    g.registerBridge(std::make_shared<MockCpuMockCpuBridge>());

    GraphBuffer raw = g.inputBuffer("raw");

    // A on cpu: split
    IOMap ioA;
    GraphBuffer leftBuf, rightBuf;
    ioA.bindInputBuffer("in", raw)
       .bindOutputBuffer("left", leftBuf)
       .bindOutputBuffer("right", rightBuf);
    g.addNode(cpuKernel("split"), std::move(ioA), "cpu");

    // B on mcpu:0: double the left branch
    IOMap ioB;
    GraphBuffer leftOut;
    ioB.bindInputBuffer("in", leftBuf).bindOutputBuffer("out", leftOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioB), "mcpu:0");

    // C on mcpu:1: double the right branch
    IOMap ioC;
    GraphBuffer rightOut;
    ioC.bindInputBuffer("in", rightBuf).bindOutputBuffer("out", rightOut);
    g.addNode(mockCpuKernel("dbl"), std::move(ioC), "mcpu:1");

    // D on cpu: merge
    IOMap ioD;
    GraphBuffer finalBuf;
    ioD.bindInputBuffer("left", leftOut)
       .bindInputBuffer("right", rightOut)
       .bindOutputBuffer("out", finalBuf);
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
