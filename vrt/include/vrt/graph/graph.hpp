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
 *  2. Declare graph-level input buffers with inputBuffer() and graph-global
 *     mutable scalars with globalScalar().
 *  3. Add kernel nodes with addNode(), capturing output buffer tokens from IOMap.
 *  4. Call compile() to validate the structure and lower the graph into
 *     per-device DGraphs and IDevicePlans.
 *  5. Call run() (blocking) or launch() + wait() (async).
 *
 * Compilation is explicit and required before execution: launch(), wait(), and
 * run() throw if the graph has not been compiled. Authoring helpers (addNode,
 * addLoop, addConditional, registerDevice, registerBridgeFactory, …)
 * invalidate the compiled state, so re-running a Graph after structural
 * changes requires another compile().
 *
 * Nested authoring (kernels, scalars, and buffers inside a loop body or a
 * conditional branch) does NOT go through the `Graph` itself. Use the
 * `GraphRegion` returned by `rootRegion().createChild()` (or by an inner
 * region's `createChild()`) and call `addKernel()`, `inputBuffer()`,
 * `scalar()`, `addLoop()`, `addConditional()`, etc. on it directly. The
 * region pointer is then handed to `Graph::addLoop` / `Graph::addConditional`
 * (or `GraphRegion::addLoop` / `GraphRegion::addConditional` for deeper
 * nesting) so the compiler can wire it up as a child of the surrounding
 * control-flow op.
 *
 * # CPU device invariant
 *
 * A Graph hosts at most one CPU-typed `IDevice` (see `DeviceType::CPU`).
 * Production code obtains it from `Graph::withDefaults()`, which registers a
 * canonical `CpuDevice` under the id `"cpu"`; users should not register their
 * own `CpuDevice` on top of that. Heterogeneous host-side compute that wants
 * its own placement domain (e.g. NUMA-affine workers, an accelerator-side
 * helper CPU) should declare a separate `DeviceType` rather than a second
 * CPU device.
 *
 * The compiler relies on this invariant to:
 *   - host every control-flow op (loop / conditional) on the CPU, since the
 *     CPU device is the only backend that owns control-flow execution today;
 *   - route cross-device transfers through a CPU bounce buffer when no
 *     direct `(srcType, dstType)` bridge factory is registered.
 *
 * @example
 * @code
 *   Graph g = Graph::withDefaults();
 *   g.registerDevice(std::make_shared<FpgaDevice>("fpga:0", device));
 *
 *   GraphBuffer raw = g.inputBuffer(BufferType::F32, "raw");
 *
 *   IOMap rootIo;
 *   GraphBuffer rootOut;
 *   rootIo.bindInputBuffer("in", raw)
 *         .bindOutputBuffer("out", BufferType::F32, rootOut);
 *   g.addNode(rootKernel, std::move(rootIo), "fpga:0");
 *
 *   // Nested authoring lives on the child region. The body cannot reference
 *   // a parent-scope token directly; route it through an explicit start
 *   // boundary so the compiler can prove the import.
 *   auto body = g.rootRegion().createChild();
 *   GraphBuffer bodyImported = body->inputBuffer(BufferType::F32, "imported");
 *   body->importFromParent(
 *       std::vector<BufferBoundaryMapping>{{rootOut, bodyImported}});
 *
 *   IOMap bodyIo;
 *   GraphBuffer bodyOut;
 *   bodyIo.bindInputBuffer("in", bodyImported)
 *         .bindOutputBuffer("out", BufferType::F32, bodyOut, body->scopeId());
 *   body->addKernel(bodyKernel, std::move(bodyIo), "fpga:0");
 *
 *   LoopSpec loop;
 *   loop.tripCount = LoopTripCount::constant<int32_t>(4);
 *   loop.body = body;
 *   g.addLoop(std::move(loop));
 *
 *   g.compile();
 *   g.run();
 * @endcode
 */

#ifndef VRT_GRAPH_GRAPH_HPP
#define VRT_GRAPH_GRAPH_HPP

#include <cstddef>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include <vrt/graph/authoring/calls.hpp>
#include <vrt/graph/authoring/fpga.hpp>
#include <vrt/graph/authoring/region_builder.hpp>
#include <vrt/graph/compiler.hpp>
#include <vrt/graph/control/graph_region.hpp>
#include <vrt/graph/crossdevice/bridge.hpp>
#include <vrt/graph/device/cpu_device.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/fpga_device.hpp>
#include <vrt/graph/core/graph_buffer.hpp>
#include <vrt/graph/node/io_map.hpp>
#include <vrt/graph/node/kernel_descriptor.hpp>

namespace vrt::graph {

class CpuDevice;

/**
 * @brief Facade returned by Graph::cpu() for declaring CPU kernels.
 *
 * `add<K>(args...)` registers a CpuKernel subclass instance and returns a
 * handle; `elementwise<T>(name, fn)` is the map-each-element shorthand.
 */
class CpuKernels {
   public:
    explicit CpuKernels(std::shared_ptr<CpuDevice> device) : device_(std::move(device)) {}

    template <class K, class... Args>
    KernelHandle add(Args&&... args) {
        auto kernel = std::make_shared<K>(std::forward<Args>(args)...);
        KernelHandle handle{kernel->name(), DeviceType::CPU, std::nullopt,
                            kernel->ioTypeMap(), device_->id()};
        device_->registerKernel(std::move(kernel));
        return handle;
    }

    template <class T, class Fn>
    KernelHandle elementwise(std::string name, Fn fn) {
        auto kernel = std::make_shared<ElementwiseCpuKernel<T>>(
            name, std::function<T(T)>(std::move(fn)));
        KernelHandle handle{name, DeviceType::CPU, std::nullopt, kernel->ioTypeMap(),
                            device_->id()};
        device_->registerKernel(std::move(kernel));
        return handle;
    }

   private:
    std::shared_ptr<CpuDevice> device_;
};

class Graph {
   public:
    Graph() = default;

    // Non-copyable; move is fine.
    Graph(const Graph&) = delete;
    Graph& operator=(const Graph&) = delete;
    Graph(Graph&&) = default;
    Graph& operator=(Graph&&) = default;

    /**
     * @brief Build a graph preloaded with the canonical host CPU device and
     *        all production bridge factories available in the current build.
     *
     * The returned graph contains a CpuDevice registered under the canonical
     * id `"cpu"`. CPU↔FPGA bridges are always registered. CPU↔GPU bridges are
     * registered only when the library is built with GPU support.
     */
    static Graph withDefaults();

    // --- Setup ---

    /**
     * @brief Register a device.
     *
     * Must be called before compile/run.  Multiple devices of different types
     * and ids may be registered, but a Graph may host **at most one** CPU-typed
     * device. The CPU device is effectively a singleton owned by `Graph`:
     * production code should obtain it via `Graph::withDefaults()` (which
     * registers a canonical `CpuDevice` under the id `"cpu"`) and not call
     * `registerDevice` with another `CpuDevice` afterwards. Host-side compute
     * that wants its own placement domain (e.g. NUMA-affine workers) should
     * declare a separate `DeviceType` instead of a second CPU device.
     *
     * Registering a new device after compilation resets the compiled state.
     *
     * @param device  Shared-ownership device instance.
     * @throws std::invalid_argument  If a device with the same id is already
     *                                registered, or if a second CPU-typed
     *                                device is registered.
     */
    void registerDevice(std::shared_ptr<IDevice> device) {
        const std::string did = device->id();
        if (devices_.count(did)) {
            throw std::invalid_argument("Graph::registerDevice: duplicate device id '" + did + "'");
        }
        if (device->type() == DeviceType::CPU) {
            for (const auto& [existingId, existing] : devices_) {
                if (existing->type() == DeviceType::CPU) {
                    throw std::invalid_argument(
                        "Graph::registerDevice: a CPU device is already registered ('" +
                        existingId + "'); only one CPU-typed device is allowed");
                }
            }
        }
        devices_[did] = std::move(device);
        invalidateCompiledState();
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
        invalidateCompiledState();
    }

    /**
     * @brief Look up (or lazily create) the bridge instance handling
     *        transfers from device id @p srcDevId to @p dstDevId.
     *
     * Returns @c nullptr when no factory is registered for the underlying
     * device-type pair (callers that need a fall-back path inspect the
     * pointer rather than catching an exception). The instance is cached;
     * subsequent calls with the same pair return the same `IBridge*`.
     *
     * @throws std::runtime_error If either device id is unknown, or the
     *         registered factory itself returns null.
     */
    IBridge* bridgeFor(const std::string& srcDevId,
                       const std::string& dstDevId) {
        auto cacheKey = std::make_pair(srcDevId, dstDevId);
        auto it = bridgeInstances_.find(cacheKey);
        if (it != bridgeInstances_.end()) return it->second.get();

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
            return nullptr;  // No factory for this device-type pair.
        }

        auto inst = fIt->second(*sIt->second, *dIt->second);
        if (!inst) {
            throw std::runtime_error(
                "Graph::bridgeFor: factory returned null for '" + srcDevId +
                "' -> '" + dstDevId + "'");
        }
        auto* ptr = inst.get();
        bridgeInstances_[cacheKey] = std::move(inst);
        return ptr;
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
     * @brief Returns a non-owning view of the root-region kernel operations
     *        in insertion order.
     *
     * Intended for inspection and visualisation; structured control-flow ops
     * are available through rootRegion(). The returned vector references
     * `KernelOp`s held by `rootRegion()`'s op list, which is stable for the
     * region's lifetime; the view is invalidated by any subsequent op append
     * on the root region (`Graph::addNode`, `Graph::addLoop`,
     * `Graph::addConditional`, …). Callers should grab the view once and
     * stop using it before authoring more root-level ops.
     */
    std::vector<std::reference_wrapper<const KernelOp>> rootKernels() const {
        std::vector<std::reference_wrapper<const KernelOp>> out;
        for (const RegionOp& op : rootRegion_->ops()) {
            if (const auto* kernel = std::get_if<KernelOp>(&op)) {
                out.emplace_back(std::cref(*kernel));
            }
        }
        return out;
    }

    /**
     * @brief Returns the registered devices keyed by id().
     */
    const std::map<std::string, std::shared_ptr<IDevice>>& devices() const {
        return devices_;
    }

    /**
     * @brief Returns the root authored region of the structured graph.
     */
    GraphRegion& rootRegion() { return *rootRegion_; }
    const GraphRegion& rootRegion() const { return *rootRegion_; }

    /**
     * @brief Returns the per-device subgraphs produced by the last compile.
     *
     * Empty until compile() has been called and reset whenever the graph
     * is re-authored or new devices/bridge factories are registered.
     */
    const std::vector<DGraph>& dgraphs() const { return dgraphs_; }

    /**
     * @brief Returns the registered CPU-typed device as a CpuDevice.
     *
     * registerDevice() enforces that a Graph hosts at most one CPU-typed
     * device, so this returns the singleton when present. Returns nullptr if
     * no CPU device is registered or the registered CPU is not a CpuDevice
     * instance.
     */
    std::shared_ptr<CpuDevice> cpuDevice() const;

    /**
     * @brief Declare a graph-global mutable scalar variable.
     *
     * The returned GraphScalar token may be bound as an input scalar or,
     * when the kernel IOTypeMap declares it as an output scalar, as a result
     * location written by the kernel.
     *
     * Delegates fully to the root region: the type is recorded on
     * `rootRegion()` (via `GraphRegion::scalar()`) and the runtime value is
     * initialised to zero in the shared scalar store. `globalScalar()` and a
     * direct `rootRegion().scalar()` call are therefore interchangeable, and
     * both populate the type table queried by the typed accessors below.
     *
     * Graph::setScalar / Graph::getScalar / Graph::setScalarBits /
     * Graph::scalarBits address the **root scope only**: the `name` parameter
     * is interpreted as a root-scope identifier and resolved against the same
     * `scopedScalarKey` namespace used internally for runtime values, so
     * scalars created on a non-root `GraphRegion::scalar()` are not reachable
     * through these accessors.
     */
    GraphScalar globalScalar(ScalarType type, std::string name) {
        GraphScalar token = rootRegion_->scalar(type, name);
        const std::string key = scopedScalarKey(rootRegion_->scopeId(), name);
        (*scalarValues_)[key] = 0;
        invalidateCompiledState();
        return token;
    }

    /**
     * @brief Set a declared graph-global scalar from raw bits.
     */
    void setScalarBits(const std::string& name, uint64_t bits) {
        if (!rootRegion_->scalarType(name)) {
            throw std::out_of_range("Graph::setScalarBits: unknown scalar '" + name + "'");
        }
        const std::string key = scopedScalarKey(rootRegion_->scopeId(), name);
        (*scalarValues_)[key] = bits;
    }

    /**
     * @brief Read a declared graph-global scalar as raw bits.
     */
    uint64_t scalarBits(const std::string& name) const {
        if (!rootRegion_->scalarType(name)) {
            throw std::out_of_range("Graph::scalarBits: unknown scalar '" + name + "'");
        }
        const std::string key = scopedScalarKey(rootRegion_->scopeId(), name);
        auto valueIt = scalarValues_->find(key);
        if (valueIt == scalarValues_->end()) {
            throw std::out_of_range("Graph::scalarBits: scalar '" + name + "' has no value");
        }
        return valueIt->second;
    }

    template <class T>
    void setScalar(const std::string& name, T value) {
        static_assert(std::is_arithmetic_v<T>, "Graph::setScalar only supports arithmetic types");
        auto declaredType = rootRegion_->scalarType(name);
        if (!declaredType) {
            throw std::out_of_range("Graph::setScalar: unknown scalar '" + name + "'");
        }
        if (*declaredType != typeToScalarType<T>()) {
            throw std::invalid_argument(
                "Graph::setScalar: type mismatch for scalar '" + name + "'");
        }
        setScalarBits(name, detail::valueToBits(value));
    }

    template <class T>
    T getScalar() const = delete;

    template <class T>
    T getScalar(const std::string& name) const {
        static_assert(std::is_arithmetic_v<T>, "Graph::getScalar only supports arithmetic types");
        auto declaredType = rootRegion_->scalarType(name);
        if (!declaredType) {
            throw std::out_of_range("Graph::getScalar: unknown scalar '" + name + "'");
        }
        if (*declaredType != typeToScalarType<T>()) {
            throw std::invalid_argument(
                "Graph::getScalar: type mismatch for scalar '" + name + "'");
        }

        uint64_t bits = scalarBits(name);
        T value{};
        std::memcpy(&value, &bits, sizeof(T));
        return value;
    }

    /**
     * @brief Declare a graph-level input buffer (no producer node).
     *
     * @param type  Element type of the buffer.
     * @param name  Logical name; must be unique among all graph buffers.
     * @return      A GraphBuffer token that may be passed to IOMap::bindInputBuffer().
     * @throws std::invalid_argument  If the name is already taken.
     *
     * Delegates name registration to the root region so that ::inputBuffer()
     * calls directly on rootRegion() and ::inputBuffer() calls on Graph share
     * the same name set.
     */
    GraphBuffer inputBuffer(BufferType type, std::string name) {
        GraphBuffer token = rootRegion_->inputBuffer(type, std::move(name));
        invalidateCompiledState();
        return token;
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
        const std::string nodeId = rootRegion_->addKernel(
            std::move(kernel), std::move(ioMap), std::move(deviceHint), std::move(afterNodes));
        invalidateCompiledState();
        return nodeId;
    }

    std::string addReprogram(ReprogramSpec spec) {
        const std::string id = rootRegion_->addReprogram(std::move(spec));
        invalidateCompiledState();
        return id;
    }

    std::string addLoop(LoopSpec spec) {
        const std::string id = rootRegion_->addLoop(std::move(spec));
        invalidateCompiledState();
        return id;
    }

    std::string addConditional(ConditionalSpec spec) {
        const std::string id = rootRegion_->addConditional(std::move(spec));
        invalidateCompiledState();
        return id;
    }

    // --- Struct-literal authoring API ------------------------------------

    /**
     * @brief Access the CPU kernel registry facade (declare CPU kernels).
     */
    CpuKernels cpu() { return CpuKernels(cpuDevice()); }

    /**
     * @brief Bring up an FPGA device in one call and register it.
     *
     * Folds QDMA PDI staging, vbin/image loading, the vrtd session + BAR
     * window, the RP1 readiness preflight, and FpgaDevice construction. The
     * returned handle owns those resources for the graph's lifetime. Defined
     * in graph.cpp (pulls in vrtd/vrt device plumbing).
     */
    FpgaHandle addFpga(const FpgaSpec& spec);

    /**
     * @brief Declare a graph-level typed input buffer token.
     */
    template <class T>
    GraphBuffer input(std::string name, std::size_t count) {
        GraphBuffer token = rootRegion_->inputBuffer(typeToBufferType<T>(), std::move(name),
                                                      count);
        invalidateCompiledState();
        return token;
    }

    /**
     * @brief Mint a typed, single-assignment buffer token at root scope.
     */
    template <class T>
    GraphBuffer buffer(std::string name, std::size_t count) {
        return GraphBuffer::make(typeToBufferType<T>(), std::move(name),
                                 rootRegion_->scopeId(), count);
    }

    /**
     * @brief A named constant scalar input (buffer-symmetric constant form).
     */
    template <class T>
    GraphScalar scalarInput(std::string /*name*/, T value) {
        return GraphScalar::constant<T>(value);
    }

    /**
     * @brief Declare a named scalar written once by a kernel (output scalar).
     */
    template <class T>
    GraphScalar scalar(std::string name) {
        return globalScalar(typeToScalarType<T>(), std::move(name));
    }

    /** @brief Author a kernel dispatch at root scope. */
    GraphNode addKernelCall(const KernelCallSpec& spec) {
        GraphNode node = rootBuilder_.addKernelCall(spec);
        invalidateCompiledState();
        return node;
    }

    /** @brief Author an explicit reprogram (PDI_LOAD) node at root scope. */
    GraphNode addReprogram(const ReprogramCallSpec& spec) {
        GraphNode node = rootBuilder_.addReprogram(spec);
        invalidateCompiledState();
        return node;
    }

    /** @brief Author a loop region at root scope. */
    RegionBuilder addLoop(const LoopBuildSpec& spec) {
        RegionBuilder loop = rootBuilder_.addLoop(spec);
        invalidateCompiledState();
        return loop;
    }

    /** @brief Author a conditional at root scope; returns [then, else]. */
    std::pair<RegionBuilder, RegionBuilder> addConditional(const ConditionalBuildSpec& spec) {
        auto branches = rootBuilder_.addConditional(spec);
        invalidateCompiledState();
        return branches;
    }

    /**
     * @brief Provide host data for a graph-level input buffer token.
     */
    template <class T>
    void write(const GraphBuffer& token, const std::vector<T>& data) {
        auto cpu = cpuDevice();
        if (!cpu) {
            throw std::runtime_error("Graph::write: no CPU device registered");
        }
        cpu->setInputBuffer(scopedBufferKey(token.scopeId(), token.name()),
                            data.data(), data.size() * sizeof(T));
    }

    /**
     * @brief Read back a buffer token after run(), resolving its placement.
     */
    template <class T>
    void read(const GraphBuffer& token, std::vector<T>& out) {
        const std::string key = scopedBufferKey(token.scopeId(), token.name());
        const std::size_t bytes = out.size() * sizeof(T);
        if (auto cpu = cpuDevice(); cpu && cpu->bufferSize(key) > 0) {
            cpu->getOutputBuffer(key, out.data(), bytes);
            return;
        }
        for (const auto& [id, device] : devices_) {
            (void)id;
            if (auto fpga = std::dynamic_pointer_cast<FpgaDevice>(device);
                fpga && fpga->bufferSize(key) > 0) {
                fpga->getOutputBuffer(key, out.data(), bytes);
                return;
            }
        }
        throw std::runtime_error(
            "Graph::read: token '" + token.name() + "' has no readable storage; "
            "did the graph run and produce it?");
    }

    // --- Compilation & execution ---

    /**
     * @brief Compile the authored graph into per-device DGraphs and IDevicePlans.
     *
     * Compilation is the single, explicit validation step. After authoring
     * (registerDevice, registerBridgeFactory, addNode, addLoop, addConditional,
     * inputBuffer, globalScalar, …) the user must call compile() exactly once
     * before launch() / wait() / run(). compile() validates the graph
     * structure (root-scope scalar/buffer references, port bindings, scopes,
     * cycles, bridge factory coverage) and lowers it into per-device DGraphs
     * and IDevicePlans.
     *
     * Calling compile() again rebuilds from scratch. Authoring helpers and
     * registerDevice / registerBridgeFactory invalidate the compiled state, so
     * a fresh compile() is required after any structural mutation.
     *
     * @throws std::runtime_error  On any structural violation; see
     *                             GraphCompiler::compile for details.
     */
    void compile() {
        GraphCompiler compiler;
        auto lookup = [this](const std::string& s, const std::string& d) -> IBridge* {
            return this->bridgeFor(s, d);
        };
        plans_.clear();
        dgraphs_ = compiler.compile(rootRegion(), devices_, bridgeFactories_,
                                    lookup, scalarValues_);
        wireRendezvousAccessors();
        plans_.reserve(dgraphs_.size());
        for (const auto& dg : dgraphs_) {
            plans_.push_back(dg.device->compilePlan(dg));
        }
        compiled_ = true;
    }

    /**
     * @brief Execute synchronously. Equivalent to launch() followed by wait().
     *
     * Requires a prior call to compile(). Throws if the graph is not compiled.
     */
    void run() {
        launch();
        wait();
    }

    /**
     * @brief Start asynchronous execution on every device.
     *
     * Requires a prior call to compile(). Throws if the graph is not compiled.
     * Returns as soon as all device plans have been started; call wait() to
     * block until they all complete.
     */
    void launch() {
        requireCompiled("launch");
        for (auto& plan : plans_) {
            plan->launch();
        }
    }

    /**
     * @brief Wait for every device plan started by launch() to complete.
     *
     * Requires a prior call to compile(). Throws if the graph is not compiled.
     */
    void wait() {
        requireCompiled("wait");
        for (auto& plan : plans_) {
            plan->wait();
        }
    }

   private:
    void invalidateCompiledState() {
        plans_.clear();
        dgraphs_.clear();
        compiled_ = false;
    }

    /**
     * @brief Give the CPU device read/write access to the FPGA's signal array.
     *
     * A split cross-device loop runs its CPU body slice concurrently with the
     * FPGA queue, rendezvousing per iteration through host-visible signal slots
     * that live in the FPGA's BAR window. The CPU device polls and SETs those
     * slots while executing its CompiledWaitNode / CompiledSignalNode halves;
     * this wires it to the FPGA window so the two queues share the same slots.
     *
     * The Phase D compiler restricts a split loop to exactly one FPGA and one
     * CPU device, so a single accessor pair (to the first FPGA window) suffices.
     */
    void wireRendezvousAccessors() {
        std::shared_ptr<CpuDevice> cpu = cpuDevice();
        if (!cpu) return;
        std::shared_ptr<FpgaDevice> fpga;
        for (const auto& [id, device] : devices_) {
            (void)id;
            if (auto f = std::dynamic_pointer_cast<FpgaDevice>(device)) {
                fpga = f;
                break;
            }
        }
        if (!fpga) return;
        std::shared_ptr<fpga::Rp1BarWindow> win = fpga->window();
        if (!win) return;
        cpu->setSignalAccessors(
            [win](std::uint32_t slot) -> std::uint32_t {
                rp1_signal_slot_t s{};
                win->readSignal(slot, s);
                return s.value;
            },
            [win](std::uint32_t slot, std::uint32_t value) {
                win->writeU32(static_cast<std::uint32_t>(
                                  RP1_DEFAULT_SIG_ARRAY_OFFSET +
                                  slot * sizeof(rp1_signal_slot_t) +
                                  offsetof(rp1_signal_slot_t, value)),
                              value);
            });
    }

    void requireCompiled(const char* method) const {
        if (!compiled_) {
            throw std::runtime_error(
                std::string("Graph::") + method +
                ": graph has not been compiled; call compile() first");
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

    std::shared_ptr<GraphRegion>                              rootRegion_ =
        GraphRegion::createRoot();
    RegionBuilder                                             rootBuilder_{rootRegion_};
    std::map<std::string, std::shared_ptr<IDevice>>           devices_;    // device-id → device
    std::map<std::pair<DeviceType, DeviceType>, BridgeFactory> bridgeFactories_;
    std::map<std::pair<std::string, std::string>,
             std::shared_ptr<IBridge>>                        bridgeInstances_;
    std::shared_ptr<std::map<std::string, uint64_t>>          scalarValues_ =
        std::make_shared<std::map<std::string, uint64_t>>();

    std::vector<DGraph> dgraphs_;   // populated by compile()
    std::vector<std::unique_ptr<IDevicePlan>> plans_;
    bool                compiled_ = false;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_GRAPH_HPP
