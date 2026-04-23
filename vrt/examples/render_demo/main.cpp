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
 * @file render_demo/main.cpp
 * @brief Demo: build a multi-device pipeline (CPU + 2× MockCpu), run it, and
 *        write the rendered Graph + per-device DGraphs as Graphviz `.dot`
 *        files. Visualise the produced files with e.g. `dot`, `xdot`, or any
 *        Graphviz-compatible viewer.
 */

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/crossdevice/bridge_op.hpp>
#include <vrt/graph/crossdevice/semaphore_pool.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/graph/node/node.hpp>
#include <vrt/graph/render/dot.hpp>

using namespace vrt::graph;

// ============================================================================
// MockCpuDevice — same shape as the test mock; runs kernels on a worker thread
// ============================================================================

class MockCpuDevice : public IDevice {
   public:
    using KernelFn = std::function<void(const CpuKernelArgs&)>;

    explicit MockCpuDevice(std::string id) : id_(std::move(id)) {}

    ~MockCpuDevice() override {
        if (worker_.joinable()) worker_.join();
    }

    void registerKernel(std::string name, KernelFn fn) {
        kernels_[std::move(name)] = std::move(fn);
    }

    void setInputBuffer(const std::string& n, const void* d, size_t s) {
        auto& b = buffers_[n];
        b.resize(s);
        if (d && s) std::memcpy(b.data(), d, s);
    }
    void getOutputBuffer(const std::string& n, void* d, size_t s) const {
        auto it = buffers_.find(n);
        if (it == buffers_.end()) throw std::runtime_error("MockCpu: no buf " + n);
        std::memcpy(d, it->second.data(), std::min(s, it->second.size()));
    }
    size_t bufferSize(const std::string& n) const {
        auto it = buffers_.find(n);
        return it == buffers_.end() ? 0 : it->second.size();
    }

    DeviceType  type() const override { return DeviceType::MOCK_CPU; }
    std::string id()   const override { return id_; }

    void compile(const DGraph& dg) override {
        steps_.clear();
        for (const Node& n : dg.nodes) {
            std::visit(
                [&](const auto& x) {
                    using T = std::decay_t<decltype(x)>;
                    if constexpr (std::is_same_v<T, KernelNode>) {
                        steps_.push_back(KernelStep{x});
                    } else if constexpr (std::is_same_v<T, BridgeOpNode>) {
                        steps_.push_back(OpStep{x.tryReady, x.action});
                    }
                },
                n);
        }
    }

    void launch() override {
        if (worker_.joinable()) worker_.join();
        worker_ = std::thread([this] {
            for (auto& s : steps_) {
                if (std::holds_alternative<KernelStep>(s)) {
                    execKernel(std::get<KernelStep>(s).node);
                } else {
                    const auto& op = std::get<OpStep>(s);
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

    void execKernel(const KernelNode& node) {
        auto it = kernels_.find(node.kernel.name);
        if (it == kernels_.end()) throw std::runtime_error("MockCpu: no kernel " + node.kernel.name);

        std::map<std::string, CpuBufferView> bv;
        size_t defSize = 0;
        for (const auto& [p, b] : node.ioMap.inputBuffers()) {
            auto fit = buffers_.find(b.name());
            if (fit == buffers_.end()) throw std::runtime_error("MockCpu: missing input " + b.name());
            if (defSize == 0) defSize = fit->second.size();
            bv[p] = CpuBufferView{fit->second.data(), fit->second.size(), b.type()};
        }
        for (const auto& [p, b] : node.ioMap.outputBuffers()) {
            auto& s = buffers_[b.name()];
            if (s.size() < defSize) s.resize(defSize);
            bv[p] = CpuBufferView{s.data(), s.size(), b.type()};
        }
        CpuKernelArgs args(std::move(bv), {});
        it->second(args);
    }

    std::string                                  id_;
    std::map<std::string, KernelFn>              kernels_;
    std::map<std::string, std::vector<uint8_t>>  buffers_;
    std::vector<Step>                            steps_;
    std::thread                                  worker_;
};

// ============================================================================
// Bridge between any pair of cpu-like devices (CpuDevice + MockCpuDevice).
// Returns a BridgeStepPair so the compiler can splice it into the DGraphs.
// ============================================================================

namespace {

struct DemoBridgeOp : IBridgeOp {
    SemaphorePool*       pool;
    SemaphoreHandle      sem;
    std::vector<uint8_t> staging;

    std::string label() const override { return "demo_xfer"; }
};

BridgeStepPair makeCpuLikeTransfer(SemaphorePool&     pool,
                                    IDevice&            src,
                                    IDevice&            dst,
                                    const GraphBuffer&  buffer) {
    auto op  = std::make_shared<DemoBridgeOp>();
    op->pool = &pool;
    op->sem  = pool.allocate();

    const std::string n = buffer.name();
    auto* sc = dynamic_cast<CpuDevice*>(&src);
    auto* sm = dynamic_cast<MockCpuDevice*>(&src);
    auto* dc = dynamic_cast<CpuDevice*>(&dst);
    auto* dm = dynamic_cast<MockCpuDevice*>(&dst);

    auto producerClosure = [op, sc, sm, n] {
        size_t sz = sc ? sc->bufferSize(n) : sm ? sm->bufferSize(n) : 0;
        op->staging.resize(sz);
        if (sz) {
            if      (sc) sc->getOutputBuffer(n, op->staging.data(), sz);
            else if (sm) sm->getOutputBuffer(n, op->staging.data(), sz);
        }
        op->pool->signal(op->sem);
    };
    auto tryReady = [op]() { return op->pool->tryAwait(op->sem); };
    auto consumerAction = [op, dc, dm, n] {
        if      (dc) dc->setInputBuffer(n, op->staging.data(), op->staging.size());
        else if (dm) dm->setInputBuffer(n, op->staging.data(), op->staging.size());
    };

    return BridgeStepPair{op,
                          std::move(producerClosure),
                          std::move(tryReady),
                          std::move(consumerAction)};
}

struct DemoBarrierOp : IBridgeOp {
    SemaphorePool*  pool;
    SemaphoreHandle sem;
    std::string     label() const override { return "barrier"; }
};

BridgeStepPair makeCpuLikeBarrier(SemaphorePool& pool) {
    auto op  = std::make_shared<DemoBarrierOp>();
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

class CpuMockBridge : public IBridge {
   public:
    CpuMockBridge(IDevice& /*src*/, IDevice& /*dst*/) {}
    BridgeStepPair makeTransfer(IDevice& s, IDevice& d, const GraphBuffer& b,
                                 uint64_t, const std::string&, const std::string&) override {
        return makeCpuLikeTransfer(pool_, s, d, b);
    }
    BridgeStepPair makeBarrier(IDevice&, IDevice&,
                                const std::string&, const std::string&) override {
        return makeCpuLikeBarrier(pool_);
    }

   private:
    SemaphorePool pool_;
};

class MockMockBridge : public IBridge {
   public:
    MockMockBridge(IDevice& /*src*/, IDevice& /*dst*/) {}
    BridgeStepPair makeTransfer(IDevice& s, IDevice& d, const GraphBuffer& b,
                                 uint64_t, const std::string&, const std::string&) override {
        return makeCpuLikeTransfer(pool_, s, d, b);
    }
    BridgeStepPair makeBarrier(IDevice&, IDevice&,
                                const std::string&, const std::string&) override {
        return makeCpuLikeBarrier(pool_);
    }

   private:
    SemaphorePool pool_;
};

// ============================================================================
// Kernel functions
// ============================================================================

// 1-in, 1-out: byte-copy
static void copyKernel(const CpuKernelArgs& a) {
    const auto& in  = a.buffer("in");
    const auto& out = a.buffer("out");
    std::memcpy(out.data, in.data, std::min(in.sizeBytes, out.sizeBytes));
}

// 2-in, 1-out: XOR-merge (just to exercise multi-input wiring)
static void mergeKernel(const CpuKernelArgs& a) {
    const auto& a_buf = a.buffer("in_a");
    const auto& b_buf = a.buffer("in_b");
    const auto& out   = a.buffer("out");
    auto* ap = static_cast<const uint8_t*>(a_buf.data);
    auto* bp = static_cast<const uint8_t*>(b_buf.data);
    auto* op = static_cast<uint8_t*>(out.data);
    size_t n = std::min({a_buf.sizeBytes, b_buf.sizeBytes, out.sizeBytes});
    for (size_t i = 0; i < n; ++i) op[i] = ap[i] ^ bp[i];
}

// ============================================================================
// Helpers to build kernel descriptors with fixed I/O signatures
// ============================================================================

static IOTypeMap io1in1out() {
    IOTypeMap io;
    io.inputBuffers.push_back({"in",  BufferType::U8});
    io.outputBuffers.push_back({"out", BufferType::U8});
    return io;
}

static IOTypeMap io2in1out() {
    IOTypeMap io;
    io.inputBuffers.push_back({"in_a", BufferType::U8});
    io.inputBuffers.push_back({"in_b", BufferType::U8});
    io.outputBuffers.push_back({"out", BufferType::U8});
    return io;
}

static KernelDescriptor kd(std::string name, DeviceType t, IOTypeMap io = io1in1out()) {
    return KernelDescriptor{std::move(name), t, std::nullopt, std::move(io)};
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    const fs::path outDir = (argc > 1) ? fs::path(argv[1]) : fs::path("render_demo_out");

    auto cpu     = std::make_shared<CpuDevice>("cpu");
    auto mock_a  = std::make_shared<MockCpuDevice>("mock_a");
    auto mock_b  = std::make_shared<MockCpuDevice>("mock_b");

    // --- Register kernels (all use copyKernel except merge) ---

    auto regCpu  = [&](const std::string& n, auto fn) { cpu   ->registerKernel(n, fn); };
    auto regA    = [&](const std::string& n, auto fn) { mock_a->registerKernel(n, fn); };
    auto regB    = [&](const std::string& n, auto fn) { mock_b->registerKernel(n, fn); };

    regCpu("ingest",     copyKernel);
    regCpu("normalize",  copyKernel);
    regCpu("enhanceA",   copyKernel);
    regCpu("postA",      copyKernel);
    regCpu("postB",      copyKernel);
    regCpu("merge",      mergeKernel);
    regCpu("encode",     copyKernel);
    regCpu("finalize",   copyKernel);

    regA  ("filterA1",   copyKernel);
    regA  ("filterA2",   copyKernel);
    regA  ("sharpenA",   copyKernel);
    regA  ("featuresB",  copyKernel);  // bounces from mock_b to mock_a

    regB  ("filterB1",   copyKernel);
    regB  ("detectB",    copyKernel);
    regB  ("denoiseA",   copyKernel);  // bounces from cpu to mock_b
    regB  ("classifyB",  copyKernel);

    // --- Build the graph ---

    Graph g;
    g.registerDevice(cpu);
    g.registerDevice(mock_a);
    g.registerDevice(mock_b);
    g.registerBridgeFactory(DeviceType::CPU, DeviceType::MOCK_CPU,
        [](IDevice& s, IDevice& d){ return std::make_shared<CpuMockBridge>(s, d); });
    g.registerBridgeFactory(DeviceType::MOCK_CPU, DeviceType::CPU,
        [](IDevice& s, IDevice& d){ return std::make_shared<CpuMockBridge>(s, d); });
    g.registerBridgeFactory(DeviceType::MOCK_CPU, DeviceType::MOCK_CPU,
        [](IDevice& s, IDevice& d){ return std::make_shared<MockMockBridge>(s, d); });

    GraphBuffer raw = g.inputBuffer(BufferType::U8, "raw");

    GraphBuffer bIngest, bNorm,
                bA1, bA2, bA3, bEnhA, bDenA, bPostA,
                bB1, bDetB, bFeatB, bClsB, bPostB,
                bMerged, bEnc, bFinal;

    auto add1 = [&](std::string name, DeviceType dt, const std::string& did,
                    const GraphBuffer& in, GraphBuffer& out,
                    std::vector<std::string> after = {}) {
        IOMap m; m.bindInputBuffer("in", in).bindOutputBuffer("out", BufferType::U8, out);
        return g.addNode(kd(std::move(name), dt), std::move(m), did, std::move(after));
    };

    auto add2 = [&](std::string name, DeviceType dt, const std::string& did,
                    const GraphBuffer& inA, const GraphBuffer& inB, GraphBuffer& out) {
        IOMap m;
        m.bindInputBuffer("in_a", inA)
         .bindInputBuffer("in_b", inB)
         .bindOutputBuffer("out", BufferType::U8, out);
        return g.addNode(kd(std::move(name), dt, io2in1out()), std::move(m), did);
    };

    // 1–2: cpu ingestion
    /*nIngest =*/ add1("ingest",    DeviceType::CPU,      "cpu",    raw,     bIngest);
    /*nNorm   =*/ add1("normalize", DeviceType::CPU,      "cpu",    bIngest, bNorm);

    // Branch A: cpu → mock_a (×3) → cpu → mock_b → cpu
    /*nA1 =*/ add1("filterA1", DeviceType::MOCK_CPU, "mock_a", bNorm, bA1);
    /*nA2 =*/ add1("filterA2", DeviceType::MOCK_CPU, "mock_a", bA1,   bA2);
    /*nA3 =*/ add1("sharpenA", DeviceType::MOCK_CPU, "mock_a", bA2,   bA3);
    /*nEh =*/ add1("enhanceA", DeviceType::CPU,      "cpu",    bA3,   bEnhA);
    /*nDn =*/ add1("denoiseA", DeviceType::MOCK_CPU, "mock_b", bEnhA, bDenA);
    auto nPostA = add1("postA",    DeviceType::CPU,      "cpu",    bDenA, bPostA);

    // Branch B: cpu → mock_b → mock_b → mock_a → mock_b → cpu
    /*nB1 =*/ add1("filterB1",  DeviceType::MOCK_CPU, "mock_b", bNorm,  bB1);
    /*nDt =*/ add1("detectB",   DeviceType::MOCK_CPU, "mock_b", bB1,    bDetB);
    /*nFe =*/ add1("featuresB", DeviceType::MOCK_CPU, "mock_a", bDetB,  bFeatB);
    /*nCl =*/ add1("classifyB", DeviceType::MOCK_CPU, "mock_b", bFeatB, bClsB);
    auto nPostB = add1("postB",     DeviceType::CPU,      "cpu",    bClsB,  bPostB);

    // Merge + tail (cpu only); merge depends on both branches
    /*nMg =*/ add2("merge",    DeviceType::CPU, "cpu", bPostA, bPostB, bMerged);
    /*nEn =*/ add1("encode",   DeviceType::CPU, "cpu", bMerged, bEnc);
    /*nFi =*/ add1("finalize", DeviceType::CPU, "cpu", bEnc,    bFinal,
                    /*after=*/{nPostA, nPostB});

    // --- Run the pipeline (so the renderer can show the populated DGraphs) ---

    std::vector<uint8_t> data(64, 0xAA);
    cpu->setInputBuffer("raw", data.data(), data.size());
    g.run();

    // --- Render: write full Graph + every per-device DGraph as .dot files ---

    std::error_code ec;
    fs::create_directories(outDir, ec);
    if (ec) {
        std::cerr << "render_demo: failed to create output dir '" << outDir
                  << "': " << ec.message() << "\n";
        return 1;
    }

    const fs::path graphPath = outDir / "graph.dot";
    render::writeToDotFile(g, graphPath.string());
    std::cout << "wrote " << graphPath << "\n";

    std::vector<fs::path> dotFiles{graphPath};
    for (const auto& dg : g.dgraphs()) {
        const fs::path p = outDir / ("dgraph_" + dg.deviceId + ".dot");
        render::writeToDotFile(dg, p.string());
        std::cout << "wrote " << p << "\n";
        dotFiles.push_back(p);
    }

    // If `dot` (Graphviz) is on PATH, also render PNGs alongside the .dot files.
    const bool hasDot = (std::system("command -v dot >/dev/null 2>&1") == 0);
    if (hasDot) {
        for (const auto& dotPath : dotFiles) {
            const fs::path pngPath = dotPath.string() + ".png";
            const std::string cmd  = "dot -Tpng " + dotPath.string() +
                                     " -o " + pngPath.string();
            const int rc = std::system(cmd.c_str());
            if (rc == 0) std::cout << "wrote " << pngPath << "\n";
            else         std::cerr << "dot failed (rc=" << rc << ") for " << dotPath << "\n";
        }
    } else {
        std::cout << "\n[dot not found on PATH — skipping PNG rendering]\n"
                  << "Visualise manually with e.g.:\n"
                  << "  dot -Tpng " << graphPath << " -o " << graphPath.string() << ".png\n"
                  << "  xdot " << graphPath << "\n";
    }
    return 0;
}
