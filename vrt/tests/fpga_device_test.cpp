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
 * @file fpga_device_test.cpp
 *
 * End-to-end unit tests for vrt::graph::FpgaDevice driven by a raw,
 * heap-backed BAR window and the same fake-RP1 worker thread used by
 * rp1_submitter_test.  No daemon, no hardware.
 *
 * The tests build small vrt::graph::Graph instances, register an
 * FpgaDevice, compile, and run().  Assertions cover:
 *
 *  - The sentinel slot gets the expected magic written once the graph
 *    finishes (matches the existing `rp1_bringup diamond` contract).
 *  - The CQ contains one entry per kernel + the sentinel signal node.
 *  - Barrier masks and arg packing are correct for the diamond DAG.
 *  - Non-kernel CompiledNode variants (e.g. CompiledBridgeOpNode that
 *    the compiler splices for cross-device buffers) cause compilePlan
 *    to throw a descriptive diagnostic.
 *  - Deferred (global-variable) scalar resolution picks up values set
 *    on the Graph between compile() and launch().
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <slash/uapi/rp1_protocol.h>

#include <vrt/graph/core/graph_scalar.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/fpga/rp1_bar_window.hpp>
#include <vrt/graph/device/fpga/rp1_submitter.hpp>
#include <vrt/graph/device/fpga_device.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>

#include "test_support/control_specs.hpp"

using namespace vrt::graph;

namespace {

constexpr std::size_t   kBarSize   = 128ULL << 20;
constexpr std::uint64_t kWindowOff = 64ULL << 20;

// Diamond test addresses match examples/rp1_bringup/rp1_bringup.c.
constexpr std::uint32_t kKernelA_R5 = 0x88010000u;
constexpr std::uint32_t kKernelB_R5 = 0x88020000u;
constexpr std::uint32_t kKernelC_R5 = 0x88030000u;
constexpr std::uint32_t kKernelD_R5 = 0x88040000u;

KernelDescriptor fpgaKernel(std::string name, IOTypeMap ioType = {}) {
    return KernelDescriptor{std::move(name), DeviceType::FPGA, std::nullopt,
                            std::move(ioType)};
}

struct DdrView {
    std::byte* base;
    rp1_ctrl_t&        ctrl()      { return *reinterpret_cast<rp1_ctrl_t*>(base + kWindowOff); }
    rp1_node_t*        nodes()     { return reinterpret_cast<rp1_node_t*>(
                                         base + kWindowOff + RP1_DEFAULT_NODE_ARRAY_OFFSET); }
    rp1_cq_entry_t*    cq()        { return reinterpret_cast<rp1_cq_entry_t*>(
                                         base + kWindowOff + RP1_DEFAULT_CQ_OFFSET); }
    std::uint32_t*     args()      { return reinterpret_cast<std::uint32_t*>(
                                         base + kWindowOff + RP1_DEFAULT_ARG_BUF_OFFSET); }
    rp1_signal_slot_t* signals()   { return reinterpret_cast<rp1_signal_slot_t*>(
                                         base + kWindowOff + RP1_DEFAULT_SIG_ARRAY_OFFSET); }
};

class FakeRp1 {
   public:
    explicit FakeRp1(DdrView ddr) : ddr_(ddr) {
        thread_ = std::thread([this] { run(); });
    }
    ~FakeRp1() {
        stop_.store(true, std::memory_order_relaxed);
        if (thread_.joinable()) thread_.join();
    }

   private:
    void run() {
        while (!stop_.load(std::memory_order_relaxed)) {
            auto& c = ddr_.ctrl();
            if (c.graph_seq != c.graph_done_seq) {
                c.rp1_state = RP1_STATE_RUNNING;
                processGraph();
                std::atomic_thread_fence(std::memory_order_seq_cst);
                c.rp1_state      = RP1_STATE_READY;
                c.graph_done_seq = c.graph_seq;
                std::atomic_thread_fence(std::memory_order_seq_cst);
            }
            c.heartbeat = c.heartbeat + 1;
            std::this_thread::sleep_for(std::chrono::microseconds(200));
        }
    }

    void processGraph() {
        auto& c = ddr_.ctrl();
        const std::uint32_t count   = c.node_count;
        const std::uint32_t cq_size = c.cq_size;
        for (std::uint32_t i = 0; i < count; ++i) {
            rp1_node_t& n = ddr_.nodes()[i];
            if (n.opcode == RP1_OP_SIGNAL) {
                const auto& pl = n.payload.signal;
                ddr_.signals()[pl.target_slot].value = pl.value;
                ddr_.signals()[pl.target_slot].last_writer_node = i;
            }
            if ((n.flags & RP1_FLAG_SILENT) == 0u) {
                const std::uint32_t idx = c.cq_write_idx & (cq_size - 1u);
                rp1_cq_entry_t& e = ddr_.cq()[idx];
                e.node_index   = i;
                e.status       = RP1_CQ_OK;
                e.error_detail = 0;
                e.timestamp    = 0;
                ++c.cq_write_idx;
            }
            n.status = RP1_NODE_DONE;
        }
    }

    DdrView           ddr_;
    std::atomic<bool> stop_{false};
    std::thread       thread_;
};

void primeAsReady(DdrView ddr) {
    auto& c = ddr.ctrl();
    c.magic     = RP1_CTRL_MAGIC;
    c.version   = RP1_PROTOCOL_VERSION;
    c.rp1_state = RP1_STATE_READY;
    c.heartbeat = 1;
}

FpgaKernelLocationLookup makeDiamondLookup() {
    return [](const std::string& name) -> FpgaKernelLocation {
        if (name == "kA") return {kKernelA_R5, 0};
        if (name == "kB") return {kKernelB_R5, 0};
        if (name == "kC") return {kKernelC_R5, 0};
        if (name == "kD") return {kKernelD_R5, 0};
        throw std::runtime_error("unknown kernel '" + name + "'");
    };
}

class FpgaDeviceFixture : public ::testing::Test {
   protected:
    void SetUp() override {
        backing_.assign(kBarSize, std::byte{0});
        ddr_ = DdrView{backing_.data()};
        primeAsReady(ddr_);
        window_ = std::make_shared<fpga::Rp1BarWindow>(backing_.data(), backing_.size(), kWindowOff);
        rp1_    = std::make_unique<FakeRp1>(ddr_);
    }
    void TearDown() override {
        // device is closed before rp1_ exits to avoid use-after-free on
        // the shared submitter.
        rp1_.reset();
    }

    std::vector<std::byte>              backing_;
    DdrView                             ddr_{};
    std::shared_ptr<fpga::Rp1BarWindow> window_;
    std::unique_ptr<FakeRp1>            rp1_;
};

}  // namespace

// ---------------------------------------------------------------------------
// Construction validation
// ---------------------------------------------------------------------------

TEST_F(FpgaDeviceFixture, ConstructorRejectsNullWindow) {
    EXPECT_THROW(FpgaDevice("fpga:0", nullptr, makeDiamondLookup()),
                 std::invalid_argument);
}

TEST_F(FpgaDeviceFixture, ConstructorRejectsNullLookup) {
    EXPECT_THROW(FpgaDevice("fpga:0", window_, {}), std::invalid_argument);
}

TEST_F(FpgaDeviceFixture, TypeAndIdMatchIDeviceContract) {
    FpgaDevice dev("fpga:0", window_, makeDiamondLookup());
    EXPECT_EQ(dev.type(), DeviceType::FPGA);
    EXPECT_EQ(dev.id(), "fpga:0");
}

// ---------------------------------------------------------------------------
// compilePlan: rejection paths
// ---------------------------------------------------------------------------

namespace {

// CPU kernel that produces a buffer (so a CPU -> FPGA edge needs a bridge).
class CopyKernel : public CpuKernel {
   public:
    CopyKernel() {
        ioType_.inputBuffers.push_back({"in", BufferType::I32});
        ioType_.outputBuffers.push_back({"out", BufferType::I32});
    }
    const std::string& name() const override { return name_; }
    const IOTypeMap& ioTypeMap() const override { return ioType_; }
    void call(const CpuKernelArgs& args) override {
        const auto& in  = args.buffer("in");
        const auto& out = args.buffer("out");
        std::memcpy(out.data, in.data, std::min(in.sizeBytes, out.sizeBytes));
    }
   private:
    std::string name_ = "copy";
    IOTypeMap   ioType_;
};

}  // namespace

TEST_F(FpgaDeviceFixture, CrossDeviceBufferEdgesAreRejected) {
    // CPU kernel produces a buffer; FPGA kernel consumes it. The compiler
    // splices a CompiledBridgeOpNode into the FPGA DGraph (consumer-side
    // closure) which FpgaDevice's compilePlan refuses in phase 1.
    Graph g = Graph::withDefaults();
    g.cpuDevice()->registerKernel(std::make_shared<CopyKernel>());

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    g.registerDevice(dev);

    GraphBuffer raw = g.inputBuffer(BufferType::I32, "raw");
    IOTypeMap cpuIo;
    cpuIo.inputBuffers.push_back({"in", BufferType::I32});
    cpuIo.outputBuffers.push_back({"out", BufferType::I32});
    KernelDescriptor cpu{"copy", DeviceType::CPU, std::nullopt, cpuIo};

    IOMap io1;
    GraphBuffer staged;
    io1.bindInputBuffer("in", raw)
       .bindOutputBuffer("out", BufferType::I32, staged);
    g.addNode(cpu, std::move(io1), "cpu");

    IOTypeMap fpgaIo;
    fpgaIo.inputBuffers.push_back({"in", BufferType::I32});
    IOMap io2;
    io2.bindInputBuffer("in", staged);
    g.addNode(fpgaKernel("kA", fpgaIo), std::move(io2), "fpga:0");

    EXPECT_THROW(g.compile(), std::logic_error);
}

// ---------------------------------------------------------------------------
// Diamond happy-path
// ---------------------------------------------------------------------------

TEST_F(FpgaDeviceFixture, DiamondGraphCompletesAndSentinelFires) {
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());

    Graph g = Graph::withDefaults();
    g.registerDevice(dev);

    IOMap ioA, ioB, ioC, ioD;
    std::string a = g.addNode(fpgaKernel("kA"), std::move(ioA), "fpga:0");
    std::string b = g.addNode(fpgaKernel("kB"), std::move(ioB), "fpga:0", {a});
    std::string c = g.addNode(fpgaKernel("kC"), std::move(ioC), "fpga:0", {a});
    g.addNode(fpgaKernel("kD"), std::move(ioD), "fpga:0", {b, c});

    g.compile();
    ASSERT_NO_THROW(g.run());

    EXPECT_EQ(ddr_.signals()[kDefaultSentinelSlot].value, kDefaultSentinelValue);
    // 4 kernels + 1 sentinel signal = 5 CQ entries.
    EXPECT_EQ(ddr_.ctrl().cq_write_idx, 5u);
}

TEST_F(FpgaDeviceFixture, DiamondBarrierMasksAreCorrect) {
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());

    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    IOMap ioA, ioB, ioC, ioD;
    std::string a = g.addNode(fpgaKernel("kA"), std::move(ioA), "fpga:0");
    std::string b = g.addNode(fpgaKernel("kB"), std::move(ioB), "fpga:0", {a});
    std::string c_id = g.addNode(fpgaKernel("kC"), std::move(ioC), "fpga:0", {a});
    g.addNode(fpgaKernel("kD"), std::move(ioD), "fpga:0", {b, c_id});

    g.compile();
    g.launch();  // submits the graph
    g.wait();    // joins; firmware has by now processed the nodes

    // Inspect the node array we wrote to DDR.
    const rp1_node_t* n = ddr_.nodes();

    // The compiler may reorder topologically; locate nodes by R5 addr.
    auto find = [&](std::uint32_t r5) -> const rp1_node_t* {
        for (std::size_t i = 0; i < 4; ++i) {
            if (n[i].opcode == RP1_OP_KERNEL_DISPATCH &&
                n[i].payload.kernel_dispatch.kernel_base_addr == r5) {
                return &n[i];
            }
        }
        return nullptr;
    };
    const rp1_node_t* na = find(kKernelA_R5);
    const rp1_node_t* nb = find(kKernelB_R5);
    const rp1_node_t* nc = find(kKernelC_R5);
    const rp1_node_t* nd = find(kKernelD_R5);
    ASSERT_NE(na, nullptr);
    ASSERT_NE(nb, nullptr);
    ASSERT_NE(nc, nullptr);
    ASSERT_NE(nd, nullptr);

    EXPECT_EQ(na->barrier_await_mask, 0u);
    EXPECT_EQ(nb->barrier_await_mask, na->barrier_set_mask);
    EXPECT_EQ(nc->barrier_await_mask, na->barrier_set_mask);
    EXPECT_EQ(nd->barrier_await_mask, nb->barrier_set_mask | nc->barrier_set_mask);

    // Sentinel awaits only D (the unique leaf).
    const rp1_node_t& sentinel = n[4];
    EXPECT_EQ(sentinel.opcode, RP1_OP_SIGNAL);
    EXPECT_EQ(sentinel.barrier_await_mask, nd->barrier_set_mask);
    EXPECT_EQ(sentinel.payload.signal.value, kDefaultSentinelValue);
    EXPECT_EQ(sentinel.payload.signal.target_slot, kDefaultSentinelSlot);
}

TEST_F(FpgaDeviceFixture, ScalarArgsAreConstantsBakedAtCompileTime) {
    IOTypeMap iot;
    iot.inputScalars.push_back({"size", ScalarType::U32});
    iot.inputScalars.push_back({"flags", ScalarType::U8});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);

    IOMap io;
    io.bindScalar("size",  GraphScalar::constant<std::uint32_t>(123));
    io.bindScalar("flags", GraphScalar::constant<std::uint8_t>(7));
    g.addNode(fpgaKernel("kA", iot), std::move(io), "fpga:0");

    g.compile();
    g.launch();
    g.wait();

    // Both args staged at arg_buf[0..1] in IOTypeMap declaration order:
    // size=123 (1 word), flags=7 (1 word zero-extended).
    EXPECT_EQ(ddr_.args()[0], 123u);
    EXPECT_EQ(ddr_.args()[1], 7u);
    EXPECT_EQ(ddr_.nodes()[0].payload.kernel_dispatch.arg_count, 2u);
}

TEST_F(FpgaDeviceFixture, U64ScalarArgsConsumeTwoArgWords) {
    IOTypeMap iot;
    iot.inputScalars.push_back({"addr", ScalarType::U64});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);

    IOMap io;
    io.bindScalar("addr",
                  GraphScalar::constant<std::uint64_t>(0xDEAD'BEEF'CAFE'BABEull));
    g.addNode(fpgaKernel("kA", iot), std::move(io), "fpga:0");

    g.compile();
    g.launch();
    g.wait();

    EXPECT_EQ(ddr_.args()[0], 0xCAFEBABEu);
    EXPECT_EQ(ddr_.args()[1], 0xDEADBEEFu);
    EXPECT_EQ(ddr_.nodes()[0].payload.kernel_dispatch.arg_count, 2u);
}

// Note: as of phase 1, the GraphCompiler rejects global-variable scalar
// bindings on non-CPU kernels with "global scalar bindings are currently
// supported only on CPU kernels". The FpgaDevice's deferred-scalar
// resolution path is therefore unreachable through the public Graph API
// today, but the code is kept (and exercised by direct DGraph
// construction in DeferredScalarsResolvedAtLaunch) so a future phase
// that relaxes the compiler restriction Just Works.
TEST_F(FpgaDeviceFixture, GlobalScalarOnFpgaKernelIsRejectedByCompiler) {
    IOTypeMap iot;
    iot.inputScalars.push_back({"size", ScalarType::U32});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);

    GraphScalar var = g.globalScalar(ScalarType::U32, "size");

    IOMap io;
    io.bindScalar("size", var);
    g.addNode(fpgaKernel("kA", iot), std::move(io), "fpga:0");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST_F(FpgaDeviceFixture, DeferredScalarsResolvedAtLaunch) {
    // Build a DGraph by hand to bypass the compiler's "globals only on
    // CPU kernels" restriction.  Verifies that FpgaDevice's deferred
    // scalar code patches arg_buf right before submission.
    auto scalarValues = std::make_shared<std::map<std::string, std::uint64_t>>();
    (*scalarValues)["scope:0:size"] = 0xAAAAu;

    DGraph dg;
    dg.deviceId     = "fpga:0";
    dg.scalarValues = scalarValues;

    CompiledKernelNode k;
    k.id        = "kA";
    k.deviceId  = "fpga:0";
    k.kernel    = fpgaKernel("kA");
    k.kernel.ioType.inputScalars.push_back({"size", ScalarType::U32});
    k.ioMap.bindScalar("size", GraphScalar::globalVar(ScalarType::U32, "size", 0));
    dg.nodes.push_back(k);

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    dg.device = dev;
    auto plan = dev->compilePlan(dg);

    plan->launch();
    plan->wait();
    EXPECT_EQ(ddr_.args()[0], 0xAAAAu);

    (*scalarValues)["scope:0:size"] = 0xBBBBu;
    plan->launch();
    plan->wait();
    EXPECT_EQ(ddr_.args()[0], 0xBBBBu);
}

TEST_F(FpgaDeviceFixture, ArgBufferIsContiguousAcrossMultipleKernels) {
    IOTypeMap iot;
    iot.inputScalars.push_back({"s0", ScalarType::U32});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);

    auto bind = [&](std::uint32_t v) {
        IOMap io;
        io.bindScalar("s0", GraphScalar::constant<std::uint32_t>(v));
        return io;
    };
    std::string a = g.addNode(fpgaKernel("kA", iot), bind(0x11), "fpga:0");
    std::string b = g.addNode(fpgaKernel("kB", iot), bind(0x22), "fpga:0", {a});
    g.addNode(fpgaKernel("kC", iot), bind(0x33), "fpga:0", {b});

    g.compile();
    g.launch();
    g.wait();

    EXPECT_EQ(ddr_.args()[0], 0x11u);
    EXPECT_EQ(ddr_.args()[1], 0x22u);
    EXPECT_EQ(ddr_.args()[2], 0x33u);

    // Each kernel's arg_buffer_offset must point at its own slot.
    auto findOffsetFor = [&](std::uint32_t r5) -> std::uint32_t {
        for (std::size_t i = 0; i < 3; ++i) {
            const auto& kd = ddr_.nodes()[i].payload.kernel_dispatch;
            if (kd.kernel_base_addr == r5) return kd.arg_buffer_offset;
        }
        return UINT32_MAX;
    };
    EXPECT_EQ(findOffsetFor(kKernelA_R5), 0u * sizeof(std::uint32_t));
    EXPECT_EQ(findOffsetFor(kKernelB_R5), 1u * sizeof(std::uint32_t));
    EXPECT_EQ(findOffsetFor(kKernelC_R5), 2u * sizeof(std::uint32_t));
}

TEST_F(FpgaDeviceFixture, LookupReturningZeroAddressIsRejected) {
    auto bad_lookup = [](const std::string&) {
        return FpgaKernelLocation{0u, 0u};
    };
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, bad_lookup);
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    g.addNode(fpgaKernel("kA"), IOMap{}, "fpga:0");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST_F(FpgaDeviceFixture, UnboundInputScalarIsRejected) {
    IOTypeMap iot;
    iot.inputScalars.push_back({"missing", ScalarType::U32});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    g.addNode(fpgaKernel("kA", iot), IOMap{}, "fpga:0");

    EXPECT_THROW(g.compile(), std::runtime_error);
}

TEST_F(FpgaDeviceFixture, OutputScalarPortsAreRejectedInPhase1) {
    // The compiler rejects output scalar ports on non-CPU kernels at the
    // front end with "output scalar ports are currently supported only
    // on CPU kernels".  FpgaDevice's own rejection in compilePlan
    // provides defense in depth for direct DGraph construction; we
    // verify both layers complain here.
    IOTypeMap iot;
    iot.outputScalars.push_back({"result", ScalarType::U32});

    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    g.addNode(fpgaKernel("kA", iot), IOMap{}, "fpga:0");

    EXPECT_THROW(g.compile(), std::runtime_error);

    // Verify the deeper-layer rejection too.
    DGraph dg;
    dg.deviceId = "fpga:0";
    CompiledKernelNode k;
    k.id        = "kA";
    k.deviceId  = "fpga:0";
    k.kernel    = fpgaKernel("kA", iot);
    dg.nodes.push_back(k);
    EXPECT_THROW(dev->compilePlan(dg), std::logic_error);
}

TEST_F(FpgaDeviceFixture, SentinelSlotAndValueAreCustomisable) {
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    dev->setSentinelSlot(42);
    dev->setSentinelValue(0xC0FFEE00u);

    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    g.addNode(fpgaKernel("kA"), IOMap{}, "fpga:0");

    g.compile();
    g.run();

    EXPECT_EQ(ddr_.signals()[42].value, 0xC0FFEE00u);
}

TEST_F(FpgaDeviceFixture, KernelLocationLookupIsCalledOncePerKernel) {
    int kAcalls = 0;
    int kBcalls = 0;
    auto counting = [&](const std::string& n) -> FpgaKernelLocation {
        if (n == "kA") { ++kAcalls; return {kKernelA_R5, 0}; }
        if (n == "kB") { ++kBcalls; return {kKernelB_R5, 0}; }
        throw std::runtime_error("unknown kernel '" + n + "'");
    };
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, counting);

    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    std::string a = g.addNode(fpgaKernel("kA"), IOMap{}, "fpga:0");
    g.addNode(fpgaKernel("kB"), IOMap{}, "fpga:0", {a});

    g.compile();
    EXPECT_EQ(kAcalls, 1);
    EXPECT_EQ(kBcalls, 1);

    g.run();
    EXPECT_EQ(kAcalls, 1) << "lookup should not be re-called at launch";
    EXPECT_EQ(kBcalls, 1);
}

TEST_F(FpgaDeviceFixture, TooManyKernelsIsRejected) {
    auto dev = std::make_shared<FpgaDevice>("fpga:0", window_, makeDiamondLookup());
    Graph g = Graph::withDefaults();
    g.registerDevice(dev);
    // We only have 4 distinct names mapped; cycle them — that's fine,
    // the same name maps to the same R5 addr for this test.  We just
    // need >31 KERNEL_DISPATCH nodes to trip the per-bucket cap.
    std::string prev;
    const char* names[] = {"kA", "kB", "kC", "kD"};
    for (int i = 0; i < 32; ++i) {
        const std::vector<std::string> after = prev.empty()
            ? std::vector<std::string>{}
            : std::vector<std::string>{prev};
        prev = g.addNode(fpgaKernel(names[i % 4]), IOMap{}, "fpga:0", after);
    }
    EXPECT_THROW(g.compile(), std::logic_error);
}
