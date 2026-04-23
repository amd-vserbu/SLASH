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
 * @file node.hpp
 * @brief Node — a step in a Graph or DGraph (variant of KernelNode + BridgeOpNode).
 */

#ifndef VRT_GRAPH_NODE_NODE_HPP
#define VRT_GRAPH_NODE_NODE_HPP

#include <functional>
#include <memory>
#include <string>
#include <variant>
#include <vector>

#include <vrt/graph/crossdevice/bridge_op.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>

namespace vrt::graph {

/**
 * @brief A user-authored kernel instantiation.
 */
struct KernelNode {
    /**
     * @brief Unique node identifier within the owning Graph.
     *
     * Auto-assigned by `Graph::addNode()`; format: `"<kernelName>_<idx>"`.
     */
    std::string id;

    /** @brief The kernel type/signature this node instantiates. */
    KernelDescriptor kernel;

    /**
     * @brief Device placement hint, e.g. `"fpga:0"` / `"gpu:1"` / `"cpu"`.
     *
     * An empty string means "any device that satisfies kernel.type"; the
     * compiler validates that a registered IDevice matches.
     */
    std::string deviceHint;

    /** @brief Concrete data bindings for this instantiation. */
    IOMap ioMap;

    /**
     * @brief Explicit ordering constraints (other KernelNode ids).
     */
    std::vector<std::string> afterNodes;

    /**
     * @brief Compiler-populated full predecessor list inside the owning
     *        DGraph (set by `GraphCompiler::compile`).
     *
     * Empty in the user-facing `Graph::nodes()` view; populated only on
     * the per-DGraph copies. Each entry is the id of another `Node`
     * (kernel or bridge op) on the same device that must complete before
     * this kernel may execute.
     */
    std::vector<std::string> dependsOn;
};

/**
 * @brief A compiler-synthesised bridge step — runs a bridge-supplied closure
 *        at a specific position in a single device's plan.
 *
 * Each transfer typically produces a pair of BridgeOpNodes (one Producer on
 * the source device, one Consumer on the destination device) that share the
 * same `shared_ptr<IBridgeOp>`. Tooling can pair the two sides via pointer
 * identity of `op`.
 */
struct BridgeOpNode {
    enum class Side { Producer, Consumer };

    /** @brief Synthetic identifier, e.g. `"_bridge_3_p"` or `"_bridge_3_c"`. */
    std::string id;

    /** @brief Device id this side runs on (matches IDevice::id()). */
    std::string deviceHint;

    /** @brief Bridge-owned primitive shared with the paired side. */
    std::shared_ptr<IBridgeOp> op;

    /** @brief The closure to run when this step is reached. */
    std::function<void()> action;

    /** @brief Whether this is the Producer or Consumer side of the transfer. */
    Side side;

    /**
     * @brief Id of the user kernel this op is anchored to (for tooling).
     */
    std::string pairedKernelId;

    /**
     * @brief Compiler-populated full predecessor list inside the owning
     *        DGraph.
     *
     * Same semantics as `KernelNode::dependsOn`.
     */
    std::vector<std::string> dependsOn;

    /**
     * @brief Non-blocking readiness probe. Returns `true` when this op may
     *        fire; the executor must call `action()` exactly once after
     *        observing `true`.
     *
     * Defaulted to "always ready" so producer-side ops (and any future
     * non-bridge OpStep paths) work without further wiring. The compiler
     * populates this with the bridge's `consumerTryReady` for consumer-side
     * ops.
     */
    std::function<bool()> tryReady = []{ return true; };
};

/**
 * @brief Variant alias used throughout the graph runtime.
 */
using Node = std::variant<KernelNode, BridgeOpNode>;

// ---------------------------------------------------------------------------
// Free helpers — uniform access to common Node fields
// ---------------------------------------------------------------------------

inline const std::string& nodeId(const Node& n) {
    return std::visit([](const auto& x) -> const std::string& { return x.id; }, n);
}

inline const std::string& nodeDeviceHint(const Node& n) {
    return std::visit([](const auto& x) -> const std::string& { return x.deviceHint; }, n);
}

inline bool isBridgeOp(const Node& n) {
    return std::holds_alternative<BridgeOpNode>(n);
}

inline const std::vector<std::string>& nodeDependsOn(const Node& n) {
    return std::visit(
        [](const auto& x) -> const std::vector<std::string>& { return x.dependsOn; }, n);
}

}  // namespace vrt::graph

#endif  // VRT_GRAPH_NODE_NODE_HPP
