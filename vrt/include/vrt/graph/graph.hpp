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
 * @file graph.hpp
 * @brief Graph — user-facing heterogeneous work graph builder and executor.
 *
 * Usage overview:
 *
 *  1. Register one IDevice per physical device.
 *  2. Declare graph-level input buffers with inputBuffer().
 *  3. Add kernel nodes with addNode(), capturing output buffer tokens from IOMap.
 *  4. Call validate() to catch structural errors early (optional but recommended).
 *  5. Call run() (blocking) or launch() + wait() (async).
 *
 * The Graph is compiled lazily on the first call to launch() or run().
 * Re-running a Graph after structural changes (addNode / registerBackend) resets
 * the compiled state and recompiles on the next launch.
 *
 * @example
 * @code
 *   Graph g;
 *   g.registerDevice(std::make_shared<FpgaDevice>("fpga:0", device));
 *
 *   GraphBuffer raw = g.inputBuffer("raw");
 *
 *   IOMap io;
 *   GraphBuffer result;
 *   io.bindInputBuffer("in", raw).bindOutputBuffer("out", result);
 *
 *   g.addNode(myKernel, io, "fpga:0");
 *   g.run();
 * @endcode
 */

#ifndef VRT_GRAPH_GRAPH_HPP
#define VRT_GRAPH_GRAPH_HPP

#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <vrt/graph/compiler.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>
#include <vrt/graph/node/node.hpp>

namespace vrt::graph {

class Graph {
   public:
    Graph() = default;

    // Non-copyable; move is fine.
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    // --- Setup ---

    /**
     * @brief Register a device.
     *
     * Must be called before compile/run.  Multiple devices of different types
     * and ids may be registered.  Registering a new device after compilation
     * resets the compiled state.
     *
     * @param device  Shared-ownership device instance.
     * @throws std::invalid_argument  If a device with the same id is already registered.
     */
    void registerDevice(std::shared_ptr<IDevice> device) {
        const std::string did = device->id();
        if (devices_.count(did)) {
            throw std::invalid_argument("Graph::registerDevice: duplicate device id '" + did + "'");
        }
        devices_[did] = std::move(device);
        compiled_ = false;
    }

    /**
     * @brief Register a cross-device bridge **factory** for an ordered
     *        device-type pair.
     *
     * The compiler lazily invokes the factory once per concrete
     * `(srcDeviceId, dstDeviceId)` pair it encounters, constructing one
     * bridge instance bound to that specific pair. The instance is owned
     * by the Graph, so any closures it returns via `BridgeStepPair`
     * remain valid for the Graph's lifetime.
     *
     * Each non-CPU device type SHOULD register at least a `(CPU, T)` and
     * `(T, CPU)` factory; missing factories surface as runtime errors at
     * `compile()` time when a transfer needs them.
     *
     * @throws std::invalid_argument  If a factory for the same ordered
     *         pair is already registered, or both types are CPU.
     */
    void registerBridgeFactory(DeviceType    srcType,
                                DeviceType    dstType,
                                BridgeFactory factory) {
        if (srcType == DeviceType::CPU && dstType == DeviceType::CPU) {
            throw std::invalid_argument(
                "Graph::registerBridgeFactory: {CPU, CPU} factory is not allowed");
        }
        auto key = std::make_pair(srcType, dstType);
        if (bridgeFactories_.count(key)) {
            throw std::invalid_argument(
                "Graph::registerBridgeFactory: duplicate factory for device-type pair");
        }
        bridgeFactories_[key] = std::move(factory);
        compiled_ = false;
    }

    /**
     * @brief Look up (or lazily create) the bridge instance handling
     *        transfers from device id @p srcDevId to @p dstDevId.
     *
     * The instance is cached; subsequent calls with the same pair return
     * the same `IBridge`.
     *
     * @throws std::runtime_error If no factory is registered for the
     *         underlying device-type pair, or if either device id is
     *         unknown.
     */
    IBridge& bridgeFor(const std::string& srcDevId,
                       const std::string& dstDevId) {
        auto cacheKey = std::make_pair(srcDevId, dstDevId);
        auto it = bridgeInstances_.find(cacheKey);
        if (it != bridgeInstances_.end()) return *it->second;

        auto sIt = devices_.find(srcDevId);
        auto dIt = devices_.find(dstDevId);
        if (sIt == devices_.end()) {
            throw std::runtime_error(
                "Graph::bridgeFor: unknown source device id '" + srcDevId + "'");
        }
        if (dIt == devices_.end()) {
            throw std::runtime_error(
                "Graph::bridgeFor: unknown destination device id '" + dstDevId + "'");
        }

        auto typeKey = std::make_pair(sIt->second->type(), dIt->second->type());
        auto fIt = bridgeFactories_.find(typeKey);
        if (fIt == bridgeFactories_.end()) {
            throw std::runtime_error(
                "Graph::bridgeFor: no bridge factory registered for {" +
                std::string(deviceTypeName(typeKey.first)) + ", " +
                std::string(deviceTypeName(typeKey.second)) +
                "} (needed for transfer from '" + srcDevId + "' to '" + dstDevId + "')");
        }

        auto inst = fIt->second(*sIt->second, *dIt->second);
        if (!inst) {
            throw std::runtime_error(
                "Graph::bridgeFor: factory returned null for '" + srcDevId +
                "' -> '" + dstDevId + "'");
        }
        auto& ref = *inst;
        bridgeInstances_[cacheKey] = std::move(inst);
        return ref;
    }

    /**
     * @brief Returns whether a factory is registered for an ordered
     *        device-type pair.
     */
    bool hasBridgeFactory(DeviceType srcType, DeviceType dstType) const {
        return bridgeFactories_.count({srcType, dstType}) != 0;
    }

    /**
     * @brief Returns the registered bridge factories.
     */
    const std::map<std::pair<DeviceType, DeviceType>, BridgeFactory>&
    bridgeFactories() const {
        return bridgeFactories_;
    }

    /**
     * @brief Returns the lazily-instantiated per-pair bridge cache.
     */
    const std::map<std::pair<std::string, std::string>,
                   std::shared_ptr<IBridge>>& bridgeInstances() const {
        return bridgeInstances_;
    }

    /**
     * @brief Returns the user-added kernel nodes in insertion order.
     *
     * Intended for inspection and visualisation; mutating the graph must go
     * through addNode(). Users only ever construct KernelNodes; BridgeOpNodes
     * are synthesised by the compiler into the per-device DGraphs (see
     * dgraphs()).
     */
    const std::vector<KernelNode>& nodes() const { return nodes_; }

    /**
     * @brief Returns the registered devices keyed by id().
     */
    const std::map<std::string, std::shared_ptr<IDevice>>& devices() const {
        return devices_;
    }

    /**
     * @brief Returns the per-device subgraphs produced by the last compile.
     *
     * Empty until launch()/run() (or any operation that triggers
     * ensureCompiled()) has been called.
     */
    const std::vector<DGraph>& dgraphs() const { return dgraphs_; }

    /**
     * @brief Declare a graph-level input buffer (no producer node).
     *
     * @param type  Element type of the buffer.
     * @param name  Logical name; must be unique among all graph buffers.
     * @return      A GraphBuffer token that may be passed to IOMap::bindInputBuffer().
     * @throws std::invalid_argument  If the name is already taken or empty.
     */
    GraphBuffer inputBuffer(BufferType type, std::string name) {
        if (bufferNames_.count(name)) {
            throw std::invalid_argument("Graph::inputBuffer: name '" + name + "' already used");
        }
        bufferNames_.insert(name);
        compiled_ = false;
        return GraphBuffer::make(type, std::move(name));
    }

    /**
     * @brief Add a kernel node to the graph.
     *
     * @param kernel      Kernel type and I/O signature.
     * @param ioMap       Concrete data bindings for this instantiation.
     * @param deviceHint  Target device id (e.g. "fpga:0").  Empty = any device
     *                    matching kernel.type.
     * @param afterNodes  Additional ordering constraints (node ids).
     * @return            The new node's id; stable for the lifetime of the Graph.
     *
     * @throws std::invalid_argument  If a node id collision occurs (should not
     *                                happen with auto-assigned ids).
     */
    std::string addNode(KernelDescriptor         kernel,
                        IOMap                    ioMap,
                        std::string              deviceHint = "",
                        std::vector<std::string> afterNodes = {}) {
        std::string id = kernel.name + "_" + std::to_string(nodeCounter_++);
        if (nodeById_.count(id)) {
            throw std::invalid_argument("Graph::addNode: node id collision for '" + id + "'");
        }
        KernelNode n{std::move(id), std::move(kernel), std::move(deviceHint),
                     std::move(ioMap), std::move(afterNodes)};
        const std::string nodeId = n.id;
        nodeById_[nodeId] = nodes_.size();
        nodes_.push_back(std::move(n));
        compiled_ = false;
        return nodeId;
    }

    // --- Validation & execution ---

    /**
     * @brief Validate the graph structure without compiling or executing.
     *
     * Checks:
     *  - No cycles in the dependency graph.
     *  - Every input buffer token references a declared buffer name.
     *  - Every deviceHint matches a registered device (or is empty).
     *  - No mandatory port in any IOTypeMap is left unbound (when IOTypeMap is
     *    provided; binding completeness is enforced by the device at compile()).
     *
     * @throws std::runtime_error  On any structural violation, with a descriptive message.
     */
    void validate() {
        // Cycle detection and basic consistency — delegate to compiler logic.
        // Full port-binding validation is backend-specific and deferred to compile().
        GraphCompiler compiler;
        auto lookup = [this](const std::string& s, const std::string& d) -> IBridge& {
            return this->bridgeFor(s, d);
        };
        compiler.compile(nodes_, devices_, lookup);  // throws on structural errors
        // If compile() succeeds we discard the result; this is a dry-run.
    }

    /**
     * @brief Compile (if needed) and execute synchronously.
     *
     * Equivalent to calling launch() followed immediately by wait().
     */
    void run() {
        launch();
        wait();
    }

    /**
     * @brief Compile (if needed) and start asynchronous execution.
     *
     * Returns as soon as all devices have been started.  Call wait() to block
     * until all devices complete.
     *
     * @throws std::runtime_error  If the graph is empty or has no registered devices.
     */
    void launch() {
        ensureCompiled();
        for (auto& dg : dgraphs_) {
            dg.device->launch();
        }
    }

    /**
     * @brief Wait for all devices to complete.
     *
     * Must be called after launch().  Blocks until every IDevice::wait()
     * returns.
     */
    void wait() {
        for (auto& dg : dgraphs_) {
            dg.device->wait();
        }
    }

   private:
    void ensureCompiled() {
        if (compiled_) return;
        if (nodes_.empty()) {
            throw std::runtime_error("Graph::launch: graph has no nodes");
        }
        if (devices_.empty()) {
            throw std::runtime_error("Graph::launch: no devices registered");
        }
        validateBridges();
        GraphCompiler compiler;
        auto lookup = [this](const std::string& s, const std::string& d) -> IBridge& {
            return this->bridgeFor(s, d);
        };
        dgraphs_ = compiler.compile(nodes_, devices_, lookup);
        compiled_ = true;
    }

    /**
     * @brief Verify that every non-CPU device type has a {CPU, Type} factory
     *        in both directions.
     */
    void validateBridges() const {
        for (const auto& [id, device] : devices_) {
            DeviceType dt = device->type();
            if (dt == DeviceType::CPU) continue;
            if (!bridgeFactories_.count({DeviceType::CPU, dt}) ||
                !bridgeFactories_.count({dt, DeviceType::CPU})) {
                throw std::runtime_error(
                    "Graph: device '" + id +
                    "' requires {CPU, " + deviceTypeName(dt) +
                    "} and {" + deviceTypeName(dt) +
                    ", CPU} bridge factories, but at least one is missing");
            }
        }
    }

    static const char* deviceTypeName(DeviceType dt) {
        switch (dt) {
            case DeviceType::CPU:      return "CPU";
            case DeviceType::GPU:      return "GPU";
            case DeviceType::FPGA:     return "FPGA";
            case DeviceType::MOCK_CPU: return "MOCK_CPU";
        }
        return "unknown";
    }

    std::vector<KernelNode>                                   nodes_;
    std::map<std::string, size_t>                             nodeById_;    // id → index in nodes_
    std::map<std::string, std::shared_ptr<IDevice>>           devices_;    // device-id → device
    std::map<std::pair<DeviceType, DeviceType>, BridgeFactory> bridgeFactories_;
    std::map<std::pair<std::string, std::string>,
             std::shared_ptr<IBridge>>                        bridgeInstances_;
    std::set<std::string>                                     bufferNames_; // declared buffer names

    std::vector<DGraph> dgraphs_;   // populated by ensureCompiled()
    bool                compiled_ = false;
    uint32_t            nodeCounter_ = 0;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_GRAPH_HPP
