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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <vrtd/bar.hpp>
#include <vrtd/device.hpp>
#include <vrtd/session.hpp>

#include <vrt/graph/control/control_node.hpp>
#include <vrt/graph/control/graph_region.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/fpga/vbin_spec.hpp>
#include <vrt/graph/device/fpga/rp1_bar_window.hpp>
#include <vrt/graph/device/fpga/rp1_submitter.hpp>
#include <vrt/graph/device/fpga_device.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/device.hpp>

using namespace vrt::graph;

namespace {

constexpr const char* kImageA = "imageA";
constexpr const char* kImageB = "imageB";
constexpr const char* kFpgaDeviceId = "fpga:0";
constexpr const char* kStageOutPort = "stage_out";
constexpr const char* kImageAOutPort = "image_a_out";
constexpr const char* kMixOutPort = "mix_out";
constexpr const char* kImageBOutPort = "image_b_out";

struct Cli {
    std::string socket = "/run/vrtd.sock";
    std::string bdf;
    std::string vbinA;
    std::string vbinB;
    std::uint32_t iterations = 2;
    std::uint32_t elements = 16;
};

std::filesystem::path executableDir(const char* argv0) {
    std::filesystem::path p(argv0);
    if (p.has_parent_path()) return p.parent_path();
    return std::filesystem::current_path();
}

void usage(const char* argv0) {
    std::cout
        << "Usage: " << argv0 << " --bdf <PCI_BDF> [options]\n"
        << "\n"
        << "Options:\n"
        << "  --socket PATH       vrtd socket (default: /run/vrtd.sock)\n"
        << "  --vbin-a PATH       image A vbin (default: next to executable)\n"
        << "  --vbin-b PATH       image B vbin (default: next to executable)\n"
        << "  --iterations N      loop iterations (default: 2)\n"
        << "  --elements N        int32 elements (default: 16)\n"
        << "  --help, -h          show this help\n";
}

Cli parseArgs(int argc, char** argv) {
    Cli cli;
    const auto binDir = executableDir(argv[0]);
    cli.vbinA = (binDir / "rp1_graph_vbin_full_a_hw.vbin").string();
    cli.vbinB = (binDir / "rp1_graph_vbin_full_b_hw.vbin").string();

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto need = [&](const char* flag) -> std::string {
            if (++i >= argc) throw std::runtime_error(std::string("missing argument to ") + flag);
            return argv[i];
        };
        if (arg == "--socket") cli.socket = need("--socket");
        else if (arg == "--bdf") cli.bdf = need("--bdf");
        else if (arg == "--vbin-a") cli.vbinA = need("--vbin-a");
        else if (arg == "--vbin-b") cli.vbinB = need("--vbin-b");
        else if (arg == "--iterations") cli.iterations = static_cast<std::uint32_t>(
            std::stoul(need("--iterations")));
        else if (arg == "--elements") cli.elements = static_cast<std::uint32_t>(
            std::stoul(need("--elements")));
        else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(0);
        } else {
            throw std::runtime_error("unknown argument: " + arg);
        }
    }

    if (cli.bdf.empty()) {
        usage(argv[0]);
        throw std::runtime_error("--bdf is required");
    }
    if (cli.iterations == 0 || cli.elements == 0) {
        throw std::runtime_error("--iterations and --elements must be non-zero");
    }
    return cli;
}

vrtd::Device openDevice(vrtd::Session& session, const std::string& bdf) {
    return session.getDeviceByBdf(bdf);
}

using VectorOp = std::function<std::int32_t(std::int32_t)>;

class VectorCpuKernel : public CpuKernel {
   public:
    VectorCpuKernel(std::string name,
                    VectorOp op,
                    std::string inputPort = "in",
                    std::string outputPort = "out")
        : name_(std::move(name)),
          inputPort_(std::move(inputPort)),
          outputPort_(std::move(outputPort)),
          op_(std::move(op)) {
        ioType_.inputBuffers.push_back({inputPort_, BufferType::I32});
        ioType_.outputBuffers.push_back({outputPort_, BufferType::I32});
    }

    const std::string& name() const override { return name_; }
    const IOTypeMap& ioTypeMap() const override { return ioType_; }

    void call(const CpuKernelArgs& args) override {
        const auto& in = args.buffer(inputPort_);
        const auto& out = args.buffer(outputPort_);
        const auto* src = in.as<const std::int32_t>();
        auto* dst = out.as<std::int32_t>();
        const std::size_t count = std::min(in.sizeBytes, out.sizeBytes) / sizeof(std::int32_t);
        for (std::size_t i = 0; i < count; ++i) {
            dst[i] = op_(src[i]);
        }
    }

   private:
    std::string name_;
    std::string inputPort_;
    std::string outputPort_;
    VectorOp op_;
    IOTypeMap ioType_;
};

IOTypeMap cpuVectorIo(const std::string& inputPort = "in",
                      const std::string& outputPort = "out") {
    IOTypeMap io;
    io.inputBuffers.push_back({inputPort, BufferType::I32});
    io.outputBuffers.push_back({outputPort, BufferType::I32});
    return io;
}

IOTypeMap fpgaVectorIo(const std::string& outputPort = "out") {
    IOTypeMap io;
    io.inputScalars.push_back({"n", ScalarType::U64});
    io.inputBuffers.push_back({"in", BufferType::I32});
    io.outputBuffers.push_back({outputPort, BufferType::I32});
    return io;
}

KernelDescriptor refinedGraphKernel(const fpga::FpgaKernelSpec& spec,
                                    std::string imageId,
                                    const std::string& outputPort) {
    KernelDescriptor desc = spec.descriptor(imageId);
    desc.ioType = fpgaVectorIo(outputPort);
    return desc;
}

IOMap bindCpuVector(GraphBuffer in,
                    BufferType outType,
                    GraphBuffer& out,
                    uint64_t scopeId,
                    const std::string& inputPort = "in",
                    const std::string& outputPort = "out") {
    IOMap io;
    io.bindInputBuffer(inputPort, in)
      .bindOutputBuffer(outputPort, outType, out, scopeId);
    return io;
}

IOMap bindFpgaVector(std::uint64_t elements,
                     GraphBuffer in,
                     BufferType outType,
                     GraphBuffer& out,
                     uint64_t scopeId,
                     const std::string& outputPort = "out") {
    IOMap io;
    io.bindScalar("n", GraphScalar::constant<std::uint64_t>(elements))
      .bindInputBuffer("in", in)
      .bindOutputBuffer(outputPort, outType, out, scopeId);
    return io;
}

std::string addReprogram(GraphRegion& region,
                         const fpga::FpgaVbinSpec& specs,
                         const std::string& imageId,
                         std::vector<std::string> afterOps) {
    ReprogramSpec rp;
    rp.imageId = imageId;
    rp.pdiPath = specs.image(imageId).pdiPath;
    rp.deviceHint = kFpgaDeviceId;
    rp.afterOps = std::move(afterOps);
    return region.addReprogram(std::move(rp));
}

std::vector<std::int32_t> expectedOutput(std::uint32_t elements, std::uint32_t iterations) {
    std::vector<std::int32_t> out(elements);
    for (std::uint32_t i = 0; i < elements; ++i) {
        std::int32_t v = static_cast<std::int32_t>(i);
        v = v + 10;  // cpu_preprocess
        for (std::uint32_t iter = 0; iter < iterations; ++iter) {
            v = v + 1;      // cpu_stage
            v = v + 1;      // image A FPGA kernel
            v = v + 3;      // cpu_mix
            v = v * 2;      // image B FPGA kernel
            v = v - 4;      // cpu_finalize
        }
        out[i] = v + 100;  // root cpu_report
    }
    return out;
}

const char* rp1StateName(std::uint32_t state) {
    switch (state) {
    case RP1_STATE_INIT:    return "INIT";
    case RP1_STATE_READY:   return "READY";
    case RP1_STATE_RUNNING: return "RUNNING";
    case RP1_STATE_ERROR:   return "ERROR";
    case RP1_STATE_HALTED:  return "HALTED";
    default:                return "?";
    }
}

void dumpRp1Ctrl(fpga::Rp1BarWindow& window) {
    rp1_ctrl_t ctrl{};
    window.readCtrl(ctrl);
    std::cerr << "[rp1_graph_vbin_full] RP1 control block:\n"
              << "  state           = " << ctrl.rp1_state
              << " (" << rp1StateName(ctrl.rp1_state) << ")\n"
              << "  error_code      = " << ctrl.rp1_error_code << "\n"
              << "  current_node    = " << ctrl.rp1_current_node << "\n"
              << "  graph_seq       = " << ctrl.graph_seq << "\n"
              << "  graph_done_seq  = " << ctrl.graph_done_seq << "\n"
              << "  cq_write_idx    = " << ctrl.cq_write_idx << "\n"
              << "  heartbeat       = " << ctrl.heartbeat << std::endl;
}

const char* childRoleName(DGraphChildRole role) {
    switch (role) {
    case DGraphChildRole::LoopBody:         return "LoopBody";
    case DGraphChildRole::ConditionalThen:  return "ConditionalThen";
    case DGraphChildRole::ConditionalElse:  return "ConditionalElse";
    default:                                return "?";
    }
}

void dumpNode(const CompiledNode& node, const std::string& indent) {
    std::visit([&](const auto& n) {
        using T = std::decay_t<decltype(n)>;
        std::cerr << indent << "- " << n.id;
        if constexpr (std::is_same_v<T, CompiledKernelNode>) {
            std::cerr << " kernel=" << n.kernel.name;
        } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
            std::cerr << " bridge="
                      << (n.side == CompiledBridgeOpNode::Side::Producer ? "producer" : "consumer")
                      << " label=" << (n.op ? n.op->label() : "?")
                      << " paired=" << n.pairedKernelId;
        } else if constexpr (std::is_same_v<T, CompiledReprogramNode>) {
            std::cerr << " reprogram=" << n.imageId;
        } else if constexpr (std::is_same_v<T, CompiledBoundaryNode>) {
            std::cerr << " boundary="
                      << (n.side == CompiledBoundaryNode::Side::Start ? "start" : "end");
        } else if constexpr (std::is_same_v<T, CompiledLoopNode>) {
            std::cerr << " loop";
        } else if constexpr (std::is_same_v<T, CompiledConditionalNode>) {
            std::cerr << " conditional";
        }

        if (!n.dependsOn.empty()) {
            std::cerr << " deps=[";
            for (std::size_t i = 0; i < n.dependsOn.size(); ++i) {
                if (i != 0) std::cerr << ", ";
                std::cerr << n.dependsOn[i];
            }
            std::cerr << "]";
        }
        std::cerr << '\n';
    }, node);
}

void dumpDGraph(const DGraph& dg, const std::string& indent = "") {
    std::cerr << indent << "DGraph device=" << dg.deviceId
              << " nodes=" << dg.nodes.size()
              << " children=" << dg.childDGraphs.size() << '\n';
    for (const auto& node : dg.nodes) {
        dumpNode(node, indent + "  ");
    }
    for (const auto& child : dg.childDGraphs) {
        std::cerr << indent << "  child parent=" << child.parentNodeId
                  << " role=" << childRoleName(child.role)
                  << " dgraphs=" << child.dgraphs.size() << '\n';
        for (const auto& childDg : child.dgraphs) {
            if (childDg) dumpDGraph(*childDg, indent + "    ");
        }
    }
}

void dumpCompiledGraph(const Graph& graph) {
    std::cerr << "[rp1_graph_vbin_full] compiled DGraphs:" << std::endl;
    for (const auto& dg : graph.dgraphs()) {
        dumpDGraph(dg);
    }
}

}  // namespace

int main(int argc, char** argv) try {
    const Cli cli = parseArgs(argc, argv);

    std::cout << "[rp1_graph_vbin_full] opening VRT device for QDMA PDI staging..."
              << std::endl;
    vrt::Device stagingDevice(cli.bdf, cli.vbinA, /*program=*/false);

    auto specs = std::make_shared<fpga::FpgaVbinSpec>();
    specs->addImage(fpga::FpgaVbinSpec::loadImage(kImageA, cli.vbinA, cli.bdf));
    specs->addImage(fpga::FpgaVbinSpec::loadImage(kImageB, cli.vbinB, cli.bdf));

    const auto& kernelA = specs->kernel(kImageA, "graph_kernel_0");
    const auto& kernelB = specs->kernel(kImageB, "graph_kernel_0");
    KernelDescriptor fpgaA = refinedGraphKernel(kernelA, kImageA, kImageAOutPort);
    KernelDescriptor fpgaB = refinedGraphKernel(kernelB, kImageB, kImageBOutPort);

    vrtd::Session session(cli.socket.c_str());
    vrtd::Device dev = openDevice(session, cli.bdf);
    vrtd::BarFile barFile = dev.getBar(4).openBarFile();
    auto window = std::make_shared<fpga::Rp1BarWindow>(std::move(barFile));

    std::cout << "[rp1_graph_vbin_full] checking RP1 firmware readiness..." << std::endl;
    fpga::Rp1Submitter preflight(*window);
    preflight.ensureReady(std::chrono::milliseconds{5000});
    std::cout << "[rp1_graph_vbin_full] RP1 firmware is READY" << std::endl;

    auto fpgaDev = std::make_shared<FpgaDevice>(kFpgaDeviceId, window, specs, kImageA);
    fpgaDev->setWaitTimeout(std::chrono::milliseconds{30000});
    fpgaDev->setPdiStagingDevice(stagingDevice);

    Graph graph = Graph::withDefaults();
    graph.registerDevice(fpgaDev);
    auto cpu = graph.cpuDevice();
    cpu->registerKernel(std::make_shared<VectorCpuKernel>("cpu_preprocess",
                                                          [](std::int32_t v) { return v + 10; }));
    cpu->registerKernel(std::make_shared<VectorCpuKernel>("cpu_stage",
                                                          [](std::int32_t v) { return v + 1; },
                                                          "in", kStageOutPort));
    cpu->registerKernel(std::make_shared<VectorCpuKernel>("cpu_mix",
                                                          [](std::int32_t v) { return v + 3; },
                                                          "in", kMixOutPort));
    cpu->registerKernel(std::make_shared<VectorCpuKernel>("cpu_finalize",
                                                          [](std::int32_t v) { return v - 4; }));
    cpu->registerKernel(std::make_shared<VectorCpuKernel>("cpu_report",
                                                          [](std::int32_t v) { return v + 100; }));

    GraphBuffer raw = graph.inputBuffer(BufferType::I32, "raw");
    GraphBuffer preState;

    const std::string preprocess = graph.addNode(
        KernelDescriptor{"cpu_preprocess", DeviceType::CPU, std::nullopt, cpuVectorIo()},
        bindCpuVector(raw, BufferType::I32, preState, graph.rootRegion().scopeId()),
        "cpu");

    auto body = graph.rootRegion().createChild();
    GraphBuffer loopIn = body->inputBuffer(BufferType::I32, "loop_in");
    const std::string importId = body->importFromParent(
        std::vector<BufferBoundaryMapping>{{preState, loopIn}});

    GraphBuffer staged;
    const std::string stage = body->addKernel(
        KernelDescriptor{"cpu_stage", DeviceType::CPU, std::nullopt,
                         cpuVectorIo("in", kStageOutPort)},
        bindCpuVector(loopIn, BufferType::I32, staged, body->scopeId(),
                      "in", kStageOutPort),
        "cpu",
        {importId});

    const std::string reprogA = addReprogram(*body, *specs, kImageA, {});

    GraphBuffer afterA;
    const std::string fpgaNodeA = body->addKernel(
        fpgaA,
        bindFpgaVector(cli.elements, staged, BufferType::I32, afterA, body->scopeId(),
                       kImageAOutPort),
        kFpgaDeviceId,
        {reprogA});

    GraphBuffer mixed;
    const std::string mix = body->addKernel(
        KernelDescriptor{"cpu_mix", DeviceType::CPU, std::nullopt,
                         cpuVectorIo("in", kMixOutPort)},
        bindCpuVector(afterA, BufferType::I32, mixed, body->scopeId(),
                      "in", kMixOutPort),
        "cpu",
        {fpgaNodeA});

    const std::string reprogB = addReprogram(*body, *specs, kImageB, {fpgaNodeA});

    GraphBuffer afterB;
    const std::string fpgaNodeB = body->addKernel(
        fpgaB,
        bindFpgaVector(cli.elements, mixed, BufferType::I32, afterB, body->scopeId(),
                       kImageBOutPort),
        kFpgaDeviceId,
        {reprogB});

    GraphBuffer finalized;
    const std::string finalize = body->addKernel(
        KernelDescriptor{"cpu_finalize", DeviceType::CPU, std::nullopt, cpuVectorIo()},
        bindCpuVector(afterB, BufferType::I32, finalized, body->scopeId()),
        "cpu",
        {fpgaNodeB});

    IOTypeMap loopIoType;
    loopIoType.outputBuffers.push_back({"out", BufferType::I32});
    IOMap loopIo;
    GraphBuffer loopOut;
    loopIo.bindOutputBuffer("out", BufferType::I32, loopOut, graph.rootRegion().scopeId());

    body->exportToParent(std::vector<BufferBoundaryMapping>{{finalized, loopOut}}, {finalize});

    LoopSpec loop;
    loop.ioType = std::move(loopIoType);
    loop.ioMap = std::move(loopIo);
    loop.tripCount = LoopTripCount::constant<std::uint32_t>(cli.iterations);
    loop.body = body;
    loop.afterOps = {preprocess};
    const std::string loopId = graph.addLoop(std::move(loop));

    GraphBuffer finalOut;
    graph.addNode(KernelDescriptor{"cpu_report", DeviceType::CPU, std::nullopt, cpuVectorIo()},
                  bindCpuVector(loopOut, BufferType::I32, finalOut, graph.rootRegion().scopeId()),
                  "cpu",
                  {loopId});

    std::vector<std::int32_t> input(cli.elements);
    for (std::uint32_t i = 0; i < cli.elements; ++i) {
        input[i] = static_cast<std::int32_t>(i);
    }
    cpu->setInputBuffer(raw.name(), input.data(), input.size() * sizeof(input[0]));

    std::cout << "[rp1_graph_vbin_full] compiling graph with "
              << cli.iterations << " loop iteration(s), "
              << cli.elements << " element(s)" << std::endl;
    graph.compile();
    dumpCompiledGraph(graph);
    std::cout << "[rp1_graph_vbin_full] graph compile complete; running graph..." << std::endl;

    std::atomic<bool> runDone{false};
    std::thread watchdog([&] {
        while (!runDone.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::seconds{5});
            if (runDone.load(std::memory_order_acquire)) break;
            std::cerr << "[rp1_graph_vbin_full] graph.run() still active after watchdog tick"
                      << std::endl;
            dumpRp1Ctrl(*window);
        }
    });

    try {
        graph.run();
    } catch (...) {
        runDone.store(true, std::memory_order_release);
        if (watchdog.joinable()) watchdog.join();
        dumpRp1Ctrl(*window);
        throw;
    }
    runDone.store(true, std::memory_order_release);
    if (watchdog.joinable()) watchdog.join();
    std::cout << "[rp1_graph_vbin_full] graph run complete; checking output..." << std::endl;

    std::vector<std::int32_t> output(cli.elements, 0);
    cpu->getOutputBuffer(finalOut.name(), output.data(), output.size() * sizeof(output[0]));
    const auto expected = expectedOutput(cli.elements, cli.iterations);

    bool ok = (output == expected);
    std::cout << "[rp1_graph_vbin_full] output:";
    for (std::size_t i = 0; i < std::min<std::size_t>(output.size(), 8); ++i) {
        std::cout << ' ' << output[i];
    }
    if (output.size() > 8) std::cout << " ...";
    std::cout << std::endl;

    if (!ok) {
        std::cerr << "FAIL: output mismatch\nexpected:";
        for (std::size_t i = 0; i < std::min<std::size_t>(expected.size(), 8); ++i) {
            std::cerr << ' ' << expected[i];
        }
        if (expected.size() > 8) std::cerr << " ...";
        std::cerr << std::endl;
        return 1;
    }

    std::cout << "PASS: CPU + FPGA graph with two vbins and explicit reprogram nodes completed."
              << std::endl;
    return 0;
} catch (const std::exception& e) {
    std::cerr << "rp1_graph_vbin_full: " << e.what() << std::endl;
    return 1;
}
