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
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/core/types.hpp>
#include <vrt/graph/crossdevice/bridge_op.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/graph.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/io_type_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/graph/node/node.hpp>
#include <vrt/graph/render/dot.hpp>

using namespace vrt::graph;

namespace {

KernelDescriptor cpuKernel(std::string name, IOTypeMap io = {}) {
    return KernelDescriptor{std::move(name), DeviceType::CPU, std::nullopt, std::move(io)};
}

// Small CPU kernel that copies its input into its output (1:1 byte copy).
void copyKernel(const CpuKernelArgs& args) {
    const auto& in  = args.buffer("in");
    const auto& out = args.buffer("out");
    auto bytes      = std::min(in.sizeBytes, out.sizeBytes);
    std::memcpy(out.data, in.data, bytes);
}

// Build a 3-node chain on a single CpuDevice and return the Graph.
// Node IDs (auto): kA_0, kB_1, kC_2
struct ChainGraph {
    Graph                       g;
    std::shared_ptr<CpuDevice>  cpu;
    std::string                 nodeA, nodeB, nodeC;
};

ChainGraph buildChain() {
    ChainGraph c;
    c.cpu = std::make_shared<CpuDevice>("cpu");
    c.cpu->registerKernel("kA", copyKernel);
    c.cpu->registerKernel("kB", copyKernel);
    c.cpu->registerKernel("kC", copyKernel);
    c.g.registerDevice(c.cpu);

    IOTypeMap io;
    io.inputBuffers.push_back({"in", BufferType::U8});
    io.outputBuffers.push_back({"out", BufferType::U8});

    GraphBuffer raw = c.g.inputBuffer(BufferType::U8, "raw");

    GraphBuffer outA, outB, outC;

    IOMap mA;
    mA.bindInputBuffer("in", raw).bindOutputBuffer("out", BufferType::U8, outA);
    c.nodeA = c.g.addNode(cpuKernel("kA", io), std::move(mA), "cpu");

    IOMap mB;
    mB.bindInputBuffer("in", outA).bindOutputBuffer("out", BufferType::U8, outB);
    c.nodeB = c.g.addNode(cpuKernel("kB", io), std::move(mB), "cpu");

    IOMap mC;
    mC.bindInputBuffer("in", outB).bindOutputBuffer("out", BufferType::U8, outC);
    c.nodeC = c.g.addNode(cpuKernel("kC", io), std::move(mC), "cpu",
                          /*afterNodes=*/{c.nodeA});  // explicit ordering edge

    return c;
}

bool contains(const std::string& haystack, const std::string& needle) {
    return haystack.find(needle) != std::string::npos;
}

// Set VRT_RENDER_PRINT=1 (or any non-empty value) in the environment to make
// the render tests dump their DOT / ASCII output to stdout.  When unset they
// stay quiet so normal CI logs aren't cluttered.
//
//   VRT_RENDER_PRINT=1 ./tests/render_test
//   VRT_RENDER_PRINT=1 ctest -R render_test --output-on-failure -V
bool verbosePrint() {
    const char* v = std::getenv("VRT_RENDER_PRINT");
    return v && v[0] != '\0';
}

void dumpSection(const std::string& title, const std::string& body) {
    if (!verbosePrint()) return;
    std::cout << "\n--- " << title << " ---\n" << body;
    if (body.empty() || body.back() != '\n') std::cout << '\n';
    std::cout.flush();
}

}  // namespace

// ---------------------------------------------------------------------------
// renderToDot(Graph)
// ---------------------------------------------------------------------------

TEST(RenderDotTest, GraphContainsHeaderNodesAndEdges) {
    auto c   = buildChain();
    auto dot = render::renderToDot(c.g);
    dumpSection("Graph DOT", dot);

    EXPECT_TRUE(contains(dot, "digraph G"));
    EXPECT_TRUE(contains(dot, "subgraph cluster_"));
    EXPECT_TRUE(contains(dot, c.nodeA));
    EXPECT_TRUE(contains(dot, c.nodeB));
    EXPECT_TRUE(contains(dot, c.nodeC));
    EXPECT_TRUE(contains(dot, "->"));
    // Solid (data) edge from A -> B is unconditional; "after" edge for A -> C
    EXPECT_TRUE(contains(dot, "style=dashed"));
}

TEST(RenderDotTest, GraphLabelsCpuCluster) {
    auto c   = buildChain();
    auto dot = render::renderToDot(c.g);
    EXPECT_TRUE(contains(dot, "cpu [CPU]"));
}

// ---------------------------------------------------------------------------
// renderToDot(DGraph)
// ---------------------------------------------------------------------------

TEST(RenderDotTest, DGraphRendersAfterCompile) {
    auto c = buildChain();
    std::vector<uint8_t> data(16, 0xAB);
    c.cpu->setInputBuffer("raw", data.data(), data.size());
    c.g.run();  // compile + execute

    ASSERT_FALSE(c.g.dgraphs().empty());
    bool sawCpuDg = false;
    for (const auto& dg : c.g.dgraphs()) {
        if (dg.deviceId == "cpu") {
            sawCpuDg = true;
            auto dot = render::renderToDot(dg);
            dumpSection("DGraph DOT [" + dg.deviceId + "]", dot);
            EXPECT_TRUE(contains(dot, "digraph"));
            EXPECT_TRUE(contains(dot, c.nodeA));
            EXPECT_TRUE(contains(dot, c.nodeB));
            EXPECT_TRUE(contains(dot, c.nodeC));
            EXPECT_TRUE(contains(dot, "->"));
        }
    }
    EXPECT_TRUE(sawCpuDg);
}

// ---------------------------------------------------------------------------
// writeToDotFile: writes the DOT source to disk verbatim.
// ---------------------------------------------------------------------------

TEST(RenderDotTest, WriteToDotFileWritesGraphAndDGraph) {
    auto c = buildChain();
    std::vector<uint8_t> data(8, 0xCD);
    c.cpu->setInputBuffer("raw", data.data(), data.size());
    c.g.run();

    char gpath[]  = "/tmp/vrt_render_test_graph_XXXXXX.dot";
    char dgpath[] = "/tmp/vrt_render_test_dgraph_XXXXXX.dot";
    int gfd  = mkstemps(gpath,  4);
    int dgfd = mkstemps(dgpath, 4);
    ASSERT_GE(gfd,  0);
    ASSERT_GE(dgfd, 0);
    ::close(gfd);
    ::close(dgfd);

    ASSERT_NO_THROW(render::writeToDotFile(c.g, gpath));
    ASSERT_FALSE(c.g.dgraphs().empty());
    ASSERT_NO_THROW(render::writeToDotFile(c.g.dgraphs().front(), dgpath));

    auto slurp = [](const std::string& p) {
        std::ifstream     ifs(p);
        std::stringstream ss;
        ss << ifs.rdbuf();
        return ss.str();
    };
    auto gtxt  = slurp(gpath);
    auto dgtxt = slurp(dgpath);
    EXPECT_EQ(gtxt,  render::renderToDot(c.g));
    EXPECT_EQ(dgtxt, render::renderToDot(c.g.dgraphs().front()));
    EXPECT_EQ(gtxt.rfind("digraph", 0), 0u);
    EXPECT_EQ(dgtxt.rfind("digraph", 0), 0u);

    std::remove(gpath);
    std::remove(dgpath);
}

TEST(RenderDotTest, WriteToDotFileThrowsOnBadPath) {
    auto c = buildChain();
    EXPECT_THROW(render::writeToDotFile(c.g, "/no/such/dir/out.dot"),
                 std::runtime_error);
}

// ---------------------------------------------------------------------------
// DGraph rendering: BridgeOpNodes appear as dashed ellipses with the
// bridge-supplied label.
// ---------------------------------------------------------------------------

TEST(RenderDotTest, DGraphIncludesBridgeOpNodes) {
    // Build a DGraph by hand containing one KernelNode and one BridgeOpNode
    // (Producer side). This bypasses the compiler so we don't need a full
    // cross-device pipeline just to exercise the renderer.
    struct StubBridgeOp : IBridgeOp {
        std::string label() const override { return "stub_xfer"; }
    };

    KernelNode k;
    k.id     = "kA_0";
    k.kernel = cpuKernel("kA");

    BridgeOpNode b;
    b.id              = "_bridge_0_p";
    b.deviceHint      = "cpu";
    b.op              = std::make_shared<StubBridgeOp>();
    b.action          = []{};
    b.side            = BridgeOpNode::Side::Producer;
    b.pairedKernelId  = "kA_0";
    b.dependsOn       = {"kA_0"};  // Phase 1: explicit predecessor.

    DGraph dg;
    dg.deviceId = "cpu";
    dg.nodes.emplace_back(std::move(k));
    dg.nodes.emplace_back(std::move(b));

    auto dot = render::renderToDot(dg);
    dumpSection("DGraph DOT (with bridge op)", dot);

    EXPECT_TRUE(contains(dot, "kA_0"));
    EXPECT_TRUE(contains(dot, "_bridge_0_p"));
    EXPECT_TRUE(contains(dot, "shape=ellipse"));
    EXPECT_TRUE(contains(dot, "stub_xfer"));
    EXPECT_TRUE(contains(dot, "Producer"));
    // The dependsOn entry must materialise as an edge.
    EXPECT_TRUE(contains(dot, "\"kA_0\" -> \"_bridge_0_p\""));
}

// ---------------------------------------------------------------------------
// Every dependsOn entry must produce exactly one edge in the rendered DOT.
// ---------------------------------------------------------------------------

TEST(RenderDotTest, DGraphRendersEveryDependsOnAsEdge) {
    struct StubBridgeOp : IBridgeOp {
        std::string label() const override { return "stub"; }
    };

    KernelNode kA; kA.id = "kA"; kA.kernel = cpuKernel("kA");
    KernelNode kB; kB.id = "kB"; kB.kernel = cpuKernel("kB");
    kB.dependsOn = {"kA"};

    BridgeOpNode bp;
    bp.id = "_bridge_p"; bp.deviceHint = "cpu";
    bp.op = std::make_shared<StubBridgeOp>(); bp.action = []{};
    bp.side = BridgeOpNode::Side::Producer; bp.pairedKernelId = "kB";
    bp.dependsOn = {"kB"};

    BridgeOpNode bc;
    bc.id = "_bridge_c"; bc.deviceHint = "cpu";
    bc.op = std::make_shared<StubBridgeOp>(); bc.action = []{};
    bc.side = BridgeOpNode::Side::Consumer; bc.pairedKernelId = "kA";
    bc.dependsOn = {"_bridge_p"};  // arbitrary cross-bridge dep, e.g. bounce chain

    DGraph dg;
    dg.deviceId = "cpu";
    dg.nodes.emplace_back(std::move(kA));
    dg.nodes.emplace_back(std::move(kB));
    dg.nodes.emplace_back(std::move(bp));
    dg.nodes.emplace_back(std::move(bc));

    auto dot = render::renderToDot(dg);
    EXPECT_TRUE(contains(dot, "\"kA\" -> \"kB\""));
    EXPECT_TRUE(contains(dot, "\"kB\" -> \"_bridge_p\""));
    EXPECT_TRUE(contains(dot, "\"_bridge_p\" -> \"_bridge_c\""));
}

// ---------------------------------------------------------------------------
// Barrier op renders with its label and produces edges.
// ---------------------------------------------------------------------------

TEST(RenderDotTest, BarrierOpRendersInDot) {
    struct BarrierStub : IBridgeOp {
        std::string label() const override { return "barrier"; }
    };

    KernelNode kT; kT.id = "kTarget"; kT.kernel = cpuKernel("kT");
    BridgeOpNode bc;
    bc.id = "_barrier_0_c"; bc.deviceHint = "cpu";
    bc.op = std::make_shared<BarrierStub>(); bc.action = []{};
    bc.side = BridgeOpNode::Side::Consumer; bc.pairedKernelId = "kTarget";
    kT.dependsOn = {"_barrier_0_c"};

    DGraph dg;
    dg.deviceId = "cpu";
    dg.nodes.emplace_back(std::move(bc));
    dg.nodes.emplace_back(std::move(kT));

    auto dot = render::renderToDot(dg);
    EXPECT_TRUE(contains(dot, "_barrier_0_c"));
    EXPECT_TRUE(contains(dot, "[barrier]"));
    EXPECT_TRUE(contains(dot, "\"_barrier_0_c\" -> \"kTarget\""));
}
