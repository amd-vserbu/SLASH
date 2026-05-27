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

#include <vrt/graph/device/fpga_device.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <variant>
#include <vector>

#include <vrt/graph/core/graph_scalar.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/fpga/vbin_spec.hpp>
#include <vrt/graph/node/compiled_node.hpp>
#include <vrt/graph/node/io_type_map.hpp>

namespace vrt::graph {

namespace {

constexpr std::chrono::seconds kBridgeWaitTimeout{35};

// One bucket = 32 bits.  Bit 31 is reserved for the sentinel.
constexpr std::uint32_t kBarrierBitsPerBucket = 32u;
constexpr std::uint8_t  kSentinelBucket       = 0u;
constexpr std::uint32_t kSentinelBit          = 1u << 31;
constexpr std::uint32_t kKernelBitsPerBucket  = 31u;
constexpr std::uint32_t kArgBufferWords =
    (RP1_DEFAULT_SIG_ARRAY_OFFSET - RP1_DEFAULT_ARG_BUF_OFFSET) / sizeof(std::uint32_t);

constexpr std::uint32_t alignUp(std::uint32_t value, std::uint32_t alignment) {
    return (value + alignment - 1u) & ~(alignment - 1u);
}

constexpr std::uint32_t kBufferArenaStart =
    alignUp(RP1_DEFAULT_SIG_ARRAY_OFFSET +
                RP1_MAX_SIGNALS * sizeof(rp1_signal_slot_t),
            4096u);

/// How many 32-bit words does a scalar bit pattern occupy in the
/// argument buffer?  HLS AXI-Lite arg registers are 32-bit; wider
/// values use consecutive registers in little-endian order.
std::uint32_t scalarWidthInWords(ScalarType t) {
    switch (t) {
        case ScalarType::U8: case ScalarType::U16: case ScalarType::U32:
        case ScalarType::I8: case ScalarType::I16: case ScalarType::I32:
        case ScalarType::F32:
            return 1u;
        case ScalarType::U64: case ScalarType::I64: case ScalarType::F64:
            return 2u;
    }
    return 1u;
}

/// Zero-/sign-extend a value's raw bits to a sequence of 32-bit words.
/// Output @p dst must point to at least scalarWidthInWords(type) entries.
void writeScalarToArgWords(ScalarType type, std::uint64_t bits, std::uint32_t* dst) {
    switch (type) {
        case ScalarType::U8: {
            std::uint8_t v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[0] = static_cast<std::uint32_t>(v);
            return;
        }
        case ScalarType::U16: {
            std::uint16_t v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[0] = static_cast<std::uint32_t>(v);
            return;
        }
        case ScalarType::U32:
        case ScalarType::F32: {
            std::uint32_t v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[0] = v;
            return;
        }
        case ScalarType::I8: {
            std::int8_t v;
            std::memcpy(&v, &bits, sizeof(v));
            // Sign-extend to int32, then reinterpret to uint32 word.
            const std::int32_t e = v;
            std::memcpy(&dst[0], &e, sizeof(e));
            return;
        }
        case ScalarType::I16: {
            std::int16_t v;
            std::memcpy(&v, &bits, sizeof(v));
            const std::int32_t e = v;
            std::memcpy(&dst[0], &e, sizeof(e));
            return;
        }
        case ScalarType::I32: {
            std::int32_t v;
            std::memcpy(&v, &bits, sizeof(v));
            std::memcpy(&dst[0], &v, sizeof(v));
            return;
        }
        case ScalarType::U64:
        case ScalarType::I64:
        case ScalarType::F64: {
            std::uint64_t v;
            std::memcpy(&v, &bits, sizeof(v));
            dst[0] = static_cast<std::uint32_t>(v & 0xFFFFFFFFu);
            dst[1] = static_cast<std::uint32_t>((v >> 32) & 0xFFFFFFFFu);
            return;
        }
    }
}

void writeU64ToArgWords(std::uint64_t value, std::uint32_t* dst) {
    dst[0] = static_cast<std::uint32_t>(value & 0xFFFFFFFFu);
    dst[1] = static_cast<std::uint32_t>((value >> 32) & 0xFFFFFFFFu);
}

void ensureArgCapacity(std::uint32_t cursor_words,
                       std::uint32_t width,
                       const std::string& kernelName,
                       const std::string& portName) {
    const std::uint64_t total = static_cast<std::uint64_t>(cursor_words) + width;
    if (total > kArgBufferWords) {
        throw std::logic_error(
            "FpgaDevice: argument buffer overflow while packing kernel '" +
            kernelName + "' port '" + portName + "'");
    }
}

/// Bookkeeping for a single global-scalar binding that must be
/// re-read from the per-graph scalar map at launch time.
struct DeferredScalar {
    std::string  scopedKey;          // scope:N:varName, with N=scopeId
    std::string  fallbackKey;        // bare varName (used iff scopeId==0)
    bool         hasFallback;
    ScalarType   type;
    std::uint32_t arg_word_offset;   // absolute index in arg_buf
    std::string  diagnostic;         // "<kernelId>.<portName>"
};

const char* deviceTypeName(DeviceType t) {
    switch (t) {
        case DeviceType::CPU:      return "CPU";
        case DeviceType::GPU:      return "GPU";
        case DeviceType::FPGA:     return "FPGA";
        case DeviceType::MOCK_CPU: return "MOCK_CPU";
    }
    return "?";
}

}  // namespace

// =========================================================================
// FpgaDevicePlan
// =========================================================================

class FpgaDevicePlan : public IDevicePlan {
   public:
    FpgaDevicePlan(std::shared_ptr<fpga::Rp1Submitter>           submitter,
                   fpga::Rp1GraphImage                            image,
                   std::vector<DeferredScalar>                    deferred,
                   std::shared_ptr<std::map<std::string, std::uint64_t>> scalarValues,
                   std::uint32_t                                  sentinelSlot,
                   std::uint32_t                                  sentinelValue,
                   std::chrono::milliseconds                      timeout)
        : submitter_(std::move(submitter)),
          image_(std::move(image)),
          deferred_(std::move(deferred)),
          scalarValues_(std::move(scalarValues)),
          sentinelSlot_(sentinelSlot),
          sentinelValue_(sentinelValue),
          timeout_(timeout) {}

    FpgaDevicePlan(FpgaDevice&                                     device,
                   const DGraph&                                   dg,
                   std::shared_ptr<std::map<std::string, std::uint64_t>> scalarValues,
                   std::uint32_t                                   sentinelSlot,
                   std::uint32_t                                   sentinelValue,
                   std::chrono::milliseconds                       timeout)
        : device_(&device),
          submitter_(device.submitter_),
          scalarValues_(std::move(scalarValues)),
          sentinelSlot_(sentinelSlot),
          sentinelValue_(sentinelValue),
          timeout_(timeout) {
        buildRuntime(dg);
    }

    ~FpgaDevicePlan() override {
        try { wait(); } catch (...) { /* swallow */ }
    }

    void launch() override {
        wait();
        workerEx_ = nullptr;
        worker_ = std::thread([this] {
            try {
                lastCq_.clear();
                if (runtime_.empty()) {
                    resolveDeferredScalars();
                    submitter_->submitAndWait(image_, timeout_);
                    lastCq_ = submitter_->drainCq();
                } else {
                    runScheduled();
                }
            } catch (...) {
                workerEx_ = std::current_exception();
            }
        });
    }

    void wait() override {
        if (worker_.joinable()) worker_.join();
        if (workerEx_) {
            std::exception_ptr ex = workerEx_;
            workerEx_ = nullptr;
            std::rethrow_exception(ex);
        }
    }

    const std::vector<rp1_cq_entry_t>& lastCq() const noexcept { return lastCq_; }
    std::uint32_t sentinelSlot()  const noexcept { return sentinelSlot_; }
    std::uint32_t sentinelValue() const noexcept { return sentinelValue_; }
    const fpga::Rp1GraphImage& image() const noexcept { return image_; }

   private:
    enum class NodeKind { Kernel, Reprogram, ProducerOp, ConsumerOp };

    struct KernelRuntime {
        CompiledKernelNode node;
    };

    struct ReprogramRuntime {
        CompiledReprogramNode node;
    };

    struct NodeRuntime {
        std::string id;
        NodeKind kind = NodeKind::Kernel;
        std::size_t initialUnmet = 0;
        std::vector<std::size_t> successors;
        KernelRuntime kernel;
        ReprogramRuntime reprogram;
        std::function<bool()> tryReady;
        std::function<void()> action;
    };

    void buildRuntime(const DGraph& dg) {
        runtime_.reserve(dg.nodes.size());
        idToIdx_.reserve(dg.nodes.size());

        for (const CompiledNode& node : dg.nodes) {
            NodeRuntime rt;
            std::visit(
                [&](const auto& concrete) {
                    using T = std::decay_t<decltype(concrete)>;
                    rt.id = concrete.id;
                    if constexpr (std::is_same_v<T, CompiledKernelNode>) {
                        if (concrete.kernel.type != DeviceType::FPGA) {
                            throw std::logic_error(
                                std::string("FpgaDevice: kernel '") +
                                concrete.kernel.name +
                                "' has DeviceType::" + deviceTypeName(concrete.kernel.type) +
                                "; expected FPGA");
                        }
                        rt.kind = NodeKind::Kernel;
                        rt.kernel = KernelRuntime{concrete};
                    } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
                        rt.kind = (concrete.side == CompiledBridgeOpNode::Side::Producer)
                                      ? NodeKind::ProducerOp
                                      : NodeKind::ConsumerOp;
                        rt.tryReady = concrete.tryReady;
                        rt.action = concrete.action;
                    } else if constexpr (std::is_same_v<T, CompiledReprogramNode>) {
                        rt.kind = NodeKind::Reprogram;
                        rt.reprogram = ReprogramRuntime{concrete};
                    } else if constexpr (std::is_same_v<T, CompiledBoundaryNode>) {
                        throw std::logic_error(
                            std::string("FpgaDevice: graph-region boundaries are not yet "
                                        "supported, got '") + concrete.id + "'");
                    } else if constexpr (std::is_same_v<T, CompiledLoopNode>) {
                        throw std::logic_error(
                            std::string("FpgaDevice: LOOP nodes are not yet supported, got '") +
                            concrete.id + "' (will lower to RP1_OP_LOOP in a future phase)");
                    } else if constexpr (std::is_same_v<T, CompiledConditionalNode>) {
                        throw std::logic_error(
                            std::string("FpgaDevice: COND nodes are not yet supported, got '") +
                            concrete.id + "' (will lower to RP1_OP_COND in a future phase)");
                    } else {
                        static_assert(sizeof(T) == 0, "Unhandled CompiledNode variant");
                    }
                },
                node);
            idToIdx_[rt.id] = runtime_.size();
            runtime_.push_back(std::move(rt));
        }

        for (std::size_t i = 0; i < dg.nodes.size(); ++i) {
            for (const std::string& depId : compiledNodeDependsOn(dg.nodes[i])) {
                auto it = idToIdx_.find(depId);
                if (it == idToIdx_.end()) continue;
                runtime_[it->second].successors.push_back(i);
                ++runtime_[i].initialUnmet;
            }
        }
        validateImageOrdering();
    }

    void validateImageOrdering() const {
        std::string active = device_->activeImageId();
        for (const NodeRuntime& rt : runtime_) {
            if (rt.kind == NodeKind::Reprogram) {
                active = rt.reprogram.node.imageId;
                continue;
            }
            if (rt.kind != NodeKind::Kernel || !rt.kernel.node.kernel.image) continue;
            if (active.empty()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + rt.kernel.node.kernel.name +
                    "' requires image '" + *rt.kernel.node.kernel.image +
                    "' but the device has no active image; add a reprogram node first");
            }
            if (*rt.kernel.node.kernel.image != active) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + rt.kernel.node.kernel.name +
                    "' requires image '" + *rt.kernel.node.kernel.image +
                    "' but active image is '" + active + "'");
            }
        }
    }

    void resolveDeferredScalars() {
        if (deferred_.empty()) return;
        if (!scalarValues_) {
            throw std::runtime_error(
                "FpgaDevicePlan: deferred scalar resolution requires a scalar map "
                "(DGraph::scalarValues was null at compile time)");
        }
        for (const DeferredScalar& d : deferred_) {
            auto it = scalarValues_->find(d.scopedKey);
            if (it == scalarValues_->end() && d.hasFallback) {
                it = scalarValues_->find(d.fallbackKey);
            }
            if (it == scalarValues_->end()) {
                throw std::runtime_error(
                    "FpgaDevicePlan: global scalar bound to port '" + d.diagnostic +
                    "' is not set in the graph scalar map (key='" + d.scopedKey + "')");
            }
            writeScalarToArgWords(d.type, it->second,
                                  image_.arg_buf.data() + d.arg_word_offset);
        }
    }

    std::uint64_t scalarBits(const GraphScalar& gs, const std::string& diagnostic) const {
        if (gs.isConstant()) return gs.constantBits();
        if (!scalarValues_) {
            throw std::runtime_error(
                "FpgaDevicePlan: global scalar bound to port '" + diagnostic +
                "' requires a scalar map");
        }
        const std::string scopedKey = scopedScalarKey(gs.scopeId(), gs.varName());
        auto it = scalarValues_->find(scopedKey);
        if (it == scalarValues_->end() && gs.scopeId() == 0) {
            it = scalarValues_->find(gs.varName());
        }
        if (it == scalarValues_->end()) {
            throw std::runtime_error(
                "FpgaDevicePlan: global scalar bound to port '" + diagnostic +
                "' is not set in the graph scalar map (key='" + scopedKey + "')");
        }
        return it->second;
    }

    std::size_t currentBufferSize(const GraphBuffer& buffer) const {
        return device_->bufferSize(scopedBufferKey(buffer.scopeId(), buffer.name()));
    }

    std::size_t defaultOutputSize(const CompiledKernelNode& node) const {
        for (const auto& port : node.kernel.ioType.inputBuffers) {
            auto it = node.ioMap.inputBuffers().find(port.name);
            if (it == node.ioMap.inputBuffers().end()) continue;
            const std::size_t size = currentBufferSize(it->second);
            if (size != 0) return size;
        }
        for (const auto& rw : node.kernel.ioType.rwBuffers) {
            for (const auto& binding : node.ioMap.rwBuffers()) {
                if (binding.inPort == rw.in.name && binding.outPort == rw.out.name) {
                    const std::size_t size = currentBufferSize(binding.in);
                    if (size != 0) return size;
                }
            }
        }
        return 0;
    }

    void appendBufferAddress(fpga::Rp1GraphImage& image,
                             const CompiledKernelNode& node,
                             const std::string& portName,
                             const GraphBuffer& buffer,
                             std::size_t sizeBytes,
                             std::uint32_t& cursor_words,
                             std::uint32_t& arg_count) {
        ensureArgCapacity(cursor_words, 2u, node.kernel.name, portName);
        image.arg_buf.resize(cursor_words + 2u, 0u);
        const std::uint64_t addr = device_->bufferDeviceAddress(buffer, sizeBytes);
        writeU64ToArgWords(addr, image.arg_buf.data() + cursor_words);
        cursor_words += 2u;
        arg_count += 2u;
    }

    std::uint32_t packKernelArgs(fpga::Rp1GraphImage& image,
                                 const CompiledKernelNode& node) {
        std::uint32_t cursor_words = static_cast<std::uint32_t>(image.arg_buf.size());
        std::uint32_t arg_count = 0;

        const auto& boundScalars = node.ioMap.scalars();
        for (const ScalarPort& port : node.kernel.ioType.inputScalars) {
            auto it = boundScalars.find(port.name);
            if (it == boundScalars.end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + node.kernel.name +
                    "' input scalar port '" + port.name + "' has no IOMap binding");
            }
            const std::uint32_t width = scalarWidthInWords(port.type);
            ensureArgCapacity(cursor_words, width, node.kernel.name, port.name);
            image.arg_buf.resize(cursor_words + width, 0u);
            writeScalarToArgWords(port.type,
                                  scalarBits(it->second, node.id + "." + port.name),
                                  image.arg_buf.data() + cursor_words);
            cursor_words += width;
            arg_count += width;
        }

        if (!node.kernel.ioType.outputScalars.empty()) {
            throw std::logic_error(
                "FpgaDevice phase 1: output scalar ports are not yet supported "
                "(kernel '" + node.kernel.name + "' declares " +
                std::to_string(node.kernel.ioType.outputScalars.size()) + "); "
                "RP1_OP_SCALAR_READ lowering lands in a future phase");
        }

        const std::size_t defaultSize = defaultOutputSize(node);

        for (const BufferPort& port : node.kernel.ioType.inputBuffers) {
            auto it = node.ioMap.inputBuffers().find(port.name);
            if (it == node.ioMap.inputBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + node.kernel.name +
                    "' input buffer port '" + port.name + "' has no IOMap binding");
            }
            appendBufferAddress(image, node, port.name, it->second,
                                currentBufferSize(it->second), cursor_words, arg_count);
        }

        for (const BufferPort& port : node.kernel.ioType.outputBuffers) {
            auto it = node.ioMap.outputBuffers().find(port.name);
            if (it == node.ioMap.outputBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + node.kernel.name +
                    "' output buffer port '" + port.name + "' has no IOMap binding");
            }
            const std::size_t existing = currentBufferSize(it->second);
            appendBufferAddress(image, node, port.name, it->second,
                                std::max(defaultSize, existing), cursor_words, arg_count);
        }

        for (const RWBufferPort& port : node.kernel.ioType.rwBuffers) {
            auto it = std::find_if(node.ioMap.rwBuffers().begin(),
                                   node.ioMap.rwBuffers().end(),
                                   [&](const IOMap::RWBinding& binding) {
                                       return binding.inPort == port.in.name &&
                                              binding.outPort == port.out.name;
                                   });
            if (it == node.ioMap.rwBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + node.kernel.name +
                    "' RW buffer ports '" + port.in.name + "'/'" +
                    port.out.name + "' have no IOMap binding");
            }
            const std::size_t inSize = currentBufferSize(it->in);
            appendBufferAddress(image, node, port.in.name, it->in,
                                inSize, cursor_words, arg_count);
            appendBufferAddress(image, node, port.out.name, it->out,
                                std::max(inSize, currentBufferSize(it->out)),
                                cursor_words, arg_count);
        }

        if (arg_count > UINT16_MAX) {
            throw std::logic_error(
                "FpgaDevice: kernel '" + node.kernel.name + "' has " +
                std::to_string(arg_count) + " arg words, exceeds uint16_t cap");
        }
        return arg_count;
    }

    void executeKernel(const KernelRuntime& kernel) {
        fpga::Rp1GraphImage image;
        image.nodes.resize(2);
        const FpgaKernelLocation location =
            device_->resolveKernelLocation(kernel.node.kernel);

        rp1_node_t& dispatch = image.nodes[0];
        dispatch.opcode = RP1_OP_KERNEL_DISPATCH;
        dispatch.flags = 0;
        dispatch.status = RP1_NODE_PENDING;
        dispatch.barrier_await_bucket = 0;
        dispatch.barrier_await_mask = 0;
        dispatch.barrier_set_bucket = 0;
        dispatch.barrier_set_mask = 1u;

        const std::uint32_t argOffset = static_cast<std::uint32_t>(image.arg_buf.size()) *
                                        sizeof(std::uint32_t);
        const std::uint32_t argCount = packKernelArgs(image, kernel.node);

        auto& kd = dispatch.payload.kernel_dispatch;
        kd.kernel_base_addr = location.r5_base_addr;
        kd.arg_buffer_offset = argOffset;
        kd.arg_count = static_cast<std::uint16_t>(argCount);
        kd.ctrl_flags = 0;
        kd.timeout_cycles = location.timeout_cycles;

        rp1_node_t& sentinel = image.nodes[1];
        sentinel.opcode = RP1_OP_SIGNAL;
        sentinel.flags = 0;
        sentinel.status = RP1_NODE_PENDING;
        sentinel.barrier_await_bucket = 0;
        sentinel.barrier_await_mask = 1u;
        sentinel.barrier_set_bucket = kSentinelBucket;
        sentinel.barrier_set_mask = kSentinelBit;
        sentinel.payload.signal.target_slot = sentinelSlot_;
        sentinel.payload.signal.value = sentinelValue_;
        sentinel.payload.signal.operation = RP1_SIGOP_SET;
        image.clear_signal_slots.push_back(sentinelSlot_);

        submitter_->submitAndWait(image, timeout_);
        auto cq = submitter_->drainCq();
        lastCq_.insert(lastCq_.end(), cq.begin(), cq.end());
    }

    void executeReprogram(const ReprogramRuntime& reprogram) {
        fpga::Rp1GraphImage image;
        image.nodes.resize(2);

        std::cerr << "[FpgaDevice] reprogram " << reprogram.node.imageId
                  << ": staging PDI " << reprogram.node.pdiPath << std::endl;
        std::uint64_t pdiAddr = 0;
        if (device_->vbinSpec_ && device_->vbinSpec_->hasImage(reprogram.node.imageId)) {
            const auto& imageSpec = device_->vbinSpec_->image(reprogram.node.imageId);
            pdiAddr = device_->stagePdiBytes(imageSpec.id, imageSpec.pdiBytes);
        } else {
            pdiAddr = device_->stagePdiFile(reprogram.node.pdiPath);
        }
        std::cerr << "[FpgaDevice] reprogram " << reprogram.node.imageId
                  << ": staged PDI at 0x" << std::hex << pdiAddr << std::dec
                  << "; submitting RP1 PDI_LOAD" << std::endl;

        rp1_node_t& pdi = image.nodes[0];
        pdi.opcode = RP1_OP_PDI_LOAD;
        pdi.flags = RP1_FLAG_HALT_ON_ERROR;
        pdi.status = RP1_NODE_PENDING;
        pdi.barrier_await_bucket = 0;
        pdi.barrier_await_mask = 0;
        pdi.barrier_set_bucket = 0;
        pdi.barrier_set_mask = 1u;
        pdi.payload.pdi_load.pdi_addr_lo =
            static_cast<std::uint32_t>(pdiAddr & 0xFFFFFFFFull);
        pdi.payload.pdi_load.pdi_addr_hi =
            static_cast<std::uint32_t>((pdiAddr >> 32) & 0xFFFFFFFFull);
        pdi.payload.pdi_load.timeout_cycles = reprogram.node.timeoutCycles;

        rp1_node_t& sentinel = image.nodes[1];
        sentinel.opcode = RP1_OP_SIGNAL;
        sentinel.flags = 0;
        sentinel.status = RP1_NODE_PENDING;
        sentinel.barrier_await_bucket = 0;
        sentinel.barrier_await_mask = 1u;
        sentinel.barrier_set_bucket = kSentinelBucket;
        sentinel.barrier_set_mask = kSentinelBit;
        sentinel.payload.signal.target_slot = sentinelSlot_;
        sentinel.payload.signal.value = sentinelValue_;
        sentinel.payload.signal.operation = RP1_SIGOP_SET;
        image.clear_signal_slots.push_back(sentinelSlot_);

        submitter_->submitAndWait(image, timeout_);
        std::cerr << "[FpgaDevice] reprogram " << reprogram.node.imageId
                  << ": RP1 PDI_LOAD completed" << std::endl;
        auto cq = submitter_->drainCq();
        lastCq_.insert(lastCq_.end(), cq.begin(), cq.end());
        device_->setActiveImage(reprogram.node.imageId);
    }

    void runScheduled() {
        std::vector<std::size_t> unmetCounts;
        unmetCounts.reserve(runtime_.size());
        for (const auto& rt : runtime_) unmetCounts.push_back(rt.initialUnmet);

        std::deque<std::size_t> ready;
        std::vector<std::size_t> pendingConsumers;

        auto promote = [&](std::size_t idx) {
            if (runtime_[idx].kind == NodeKind::ConsumerOp) {
                pendingConsumers.push_back(idx);
            } else {
                ready.push_back(idx);
            }
        };

        for (std::size_t i = 0; i < runtime_.size(); ++i) {
            if (unmetCounts[i] == 0) promote(i);
        }

        auto runIndex = [&](std::size_t idx) {
            NodeRuntime& rt = runtime_[idx];
            switch (rt.kind) {
                case NodeKind::Kernel:
                    executeKernel(rt.kernel);
                    break;
                case NodeKind::Reprogram:
                    executeReprogram(rt.reprogram);
                    break;
                case NodeKind::ProducerOp:
                case NodeKind::ConsumerOp:
                    if (rt.action) rt.action();
                    break;
            }
            for (std::size_t successor : rt.successors) {
                if (--unmetCounts[successor] == 0) promote(successor);
            }
        };

        std::size_t rrCursor = 0;
        auto idleSince = std::chrono::steady_clock::now();
        for (;;) {
            while (!ready.empty()) {
                const std::size_t idx = ready.front();
                ready.pop_front();
                runIndex(idx);
                idleSince = std::chrono::steady_clock::now();
            }
            if (pendingConsumers.empty()) break;

            bool fired = false;
            for (std::size_t step = 0; step < pendingConsumers.size(); ++step) {
                if (rrCursor >= pendingConsumers.size()) rrCursor = 0;
                const std::size_t idx = pendingConsumers[rrCursor];
                NodeRuntime& rt = runtime_[idx];
                if (!rt.tryReady || rt.tryReady()) {
                    pendingConsumers.erase(pendingConsumers.begin() +
                                           static_cast<std::ptrdiff_t>(rrCursor));
                    runIndex(idx);
                    fired = true;
                    idleSince = std::chrono::steady_clock::now();
                    break;
                }
                ++rrCursor;
            }
            if (!fired) {
                if (std::chrono::steady_clock::now() - idleSince > kBridgeWaitTimeout) {
                    std::string pending;
                    for (std::size_t idx : pendingConsumers) {
                        if (!pending.empty()) pending += ", ";
                        pending += runtime_[idx].id;
                    }
                    throw std::runtime_error(
                        "FpgaDevice: timed out waiting for bridge consumer(s): " + pending);
                }
                std::this_thread::yield();
            }
        }
    }

    FpgaDevice*                                                device_ = nullptr;
    std::shared_ptr<fpga::Rp1Submitter>                       submitter_;
    fpga::Rp1GraphImage                                        image_;
    std::vector<DeferredScalar>                                deferred_;
    std::shared_ptr<std::map<std::string, std::uint64_t>>      scalarValues_;
    std::uint32_t                                              sentinelSlot_;
    std::uint32_t                                              sentinelValue_;
    std::chrono::milliseconds                                  timeout_;
    std::thread                                                worker_;
    std::exception_ptr                                         workerEx_;
    std::vector<rp1_cq_entry_t>                                lastCq_;
    std::vector<NodeRuntime>                                   runtime_;
    std::unordered_map<std::string, std::size_t>                idToIdx_;
};

// =========================================================================
// FpgaDevice
// =========================================================================

FpgaDevice::FpgaDevice(std::string                       id,
                        std::shared_ptr<fpga::Rp1BarWindow> window,
                        FpgaKernelLocationLookup           lookup,
                        std::uint32_t                      cq_size)
    : id_(std::move(id)),
      window_(std::move(window)),
      lookup_(std::move(lookup)),
      nextBufferOffset_(kBufferArenaStart) {
    if (!window_) {
        throw std::invalid_argument("FpgaDevice: window must not be null");
    }
    if (!lookup_) {
        throw std::invalid_argument("FpgaDevice: kernel-location lookup must not be null");
    }
    submitter_ = std::make_shared<fpga::Rp1Submitter>(*window_, cq_size);
}

FpgaDevice::FpgaDevice(std::string                       id,
                       std::shared_ptr<fpga::Rp1BarWindow> window,
                       std::shared_ptr<fpga::FpgaVbinSpec> vbinSpec,
                       std::string                       initialImageId,
                       std::uint32_t                     cq_size)
    : id_(std::move(id)),
      window_(std::move(window)),
      vbinSpec_(std::move(vbinSpec)),
      nextBufferOffset_(kBufferArenaStart),
      activeImageId_(std::move(initialImageId)) {
    if (!window_) {
        throw std::invalid_argument("FpgaDevice: window must not be null");
    }
    if (!vbinSpec_ || vbinSpec_->empty()) {
        throw std::invalid_argument("FpgaDevice: vbin spec must contain at least one image");
    }
    if (activeImageId_.empty()) {
        activeImageId_ = vbinSpec_->defaultImageId();
    }
    if (!vbinSpec_->hasImage(activeImageId_)) {
        throw std::invalid_argument(
            "FpgaDevice: initial image '" + activeImageId_ + "' is not in the vbin spec");
    }
    submitter_ = std::make_shared<fpga::Rp1Submitter>(*window_, cq_size);
}

FpgaDevice::~FpgaDevice() = default;

void FpgaDevice::setSentinelSlot(std::uint32_t slot) {
    if (slot >= RP1_MAX_SIGNALS) {
        throw std::invalid_argument(
            "FpgaDevice::setSentinelSlot: slot " + std::to_string(slot) +
            " out of range [0, " + std::to_string(RP1_MAX_SIGNALS) + ")");
    }
    sentinelSlot_ = slot;
}

void FpgaDevice::setSentinelValue(std::uint32_t value) {
    sentinelValue_ = value;
}

void FpgaDevice::setWaitTimeout(std::chrono::milliseconds t) {
    waitTimeout_ = t;
}

void FpgaDevice::setPdiStagingDevice(::vrt::Device device) {
    std::lock_guard<std::mutex> lk(pdiMutex_);
    pdiStagingDevice_ = std::make_shared<::vrt::Device>(std::move(device));
    stagedPdis_.clear();
}

std::string FpgaDevice::normalizeBufferKey(const std::string& bufferName) {
    if (bufferName.rfind("scope:", 0) == 0) return bufferName;
    return scopedBufferKey(0, bufferName);
}

FpgaDevice::BufferRecord FpgaDevice::ensureBuffer(const GraphBuffer& buffer,
                                                  std::size_t sizeBytes) {
    const std::string key = scopedBufferKey(buffer.scopeId(), buffer.name());
    std::lock_guard<std::mutex> lk(bufferMutex_);

    auto it = buffers_.find(key);
    if (it != buffers_.end() && it->second.capacity >= sizeBytes) {
        if (it->second.size < sizeBytes) it->second.size = sizeBytes;
        it->second.type = buffer.type();
        return it->second;
    }

    const std::uint32_t alignedOffset = alignUp(nextBufferOffset_, 64u);
    const std::uint64_t end =
        static_cast<std::uint64_t>(alignedOffset) + static_cast<std::uint64_t>(sizeBytes);
    if (end > fpga::Rp1BarWindow::kWindowSize) {
        throw std::out_of_range(
            "FpgaDevice: BAR-backed buffer arena exhausted while allocating '" +
            buffer.name() + "' (" + std::to_string(sizeBytes) + " bytes)");
    }

    BufferRecord rec;
    rec.offset = alignedOffset;
    rec.size = sizeBytes;
    rec.capacity = sizeBytes;
    rec.type = buffer.type();
    buffers_[key] = rec;
    nextBufferOffset_ = static_cast<std::uint32_t>(end);
    return rec;
}

std::uint64_t FpgaDevice::bufferDeviceAddress(const GraphBuffer& buffer,
                                              std::size_t sizeBytes) {
    const BufferRecord rec = ensureBuffer(buffer, sizeBytes);
    return RP1_CTRL_PHYS_ADDR + static_cast<std::uint64_t>(rec.offset);
}

FpgaKernelLocation FpgaDevice::resolveKernelLocation(const KernelDescriptor& kernel) const {
    if (vbinSpec_) {
        const std::string active = activeImageId();
        const std::string imageId = kernel.image ? *kernel.image : active;
        if (imageId.empty()) {
            throw std::runtime_error(
                "FpgaDevice: no active image available for kernel '" + kernel.name + "'");
        }
        if (kernel.image && !active.empty() && *kernel.image != active) {
            throw std::runtime_error(
                "FpgaDevice: kernel '" + kernel.name + "' requires image '" +
                *kernel.image + "' but active image is '" + active + "'");
        }
        const auto& spec = vbinSpec_->kernel(imageId, kernel.name);
        return FpgaKernelLocation{spec.r5_base_addr, 0};
    }
    if (!lookup_) {
        throw std::runtime_error("FpgaDevice: no kernel-location lookup configured");
    }
    const FpgaKernelLocation loc = lookup_(kernel.name);
    if (loc.r5_base_addr == 0) {
        throw std::runtime_error(
            "FpgaDevice: kernel-location lookup returned r5_base_addr=0 for kernel '" +
            kernel.name + "' (likely unmapped name)");
    }
    return loc;
}

std::uint64_t FpgaDevice::stagePdiBytes(const std::string& cacheKey,
                                        const std::vector<std::uint8_t>& bytes) {
    std::lock_guard<std::mutex> lk(pdiMutex_);
    if (pdiStagingDevice_) {
        auto cached = stagedPdis_.find(cacheKey);
        if (cached != stagedPdis_.end()) {
            std::cerr << "[FpgaDevice] stagePdiBytes: reusing DDR/QDMA staged PDI '"
                      << cacheKey << "' (" << cached->second.size
                      << " bytes) @ 0x" << std::hex << cached->second.physAddr
                      << std::dec << std::endl;
            return cached->second.physAddr;
        }

        std::cerr << "[FpgaDevice] stagePdiBytes: allocating DDR/QDMA buffer for '"
                  << cacheKey << "' (" << bytes.size() << " bytes)" << std::endl;
        auto buffer = std::make_unique<::vrt::Buffer<std::uint8_t>>(
            *pdiStagingDevice_, bytes.size(), ::vrt::MemoryRangeType::DDR);
        std::cerr << "[FpgaDevice] stagePdiBytes: DDR/QDMA buffer allocated"
                  << " @ 0x" << std::hex << buffer->getPhysAddr()
                  << std::dec << std::endl;
        if (!bytes.empty()) {
            std::cerr << "[FpgaDevice] stagePdiBytes: copying to host staging buffer" << std::endl;
            std::memcpy(buffer->get(), bytes.data(), bytes.size());
            std::cerr << "[FpgaDevice] stagePdiBytes: QDMA sync HOST_TO_DEVICE start" << std::endl;
            buffer->sync(::vrt::SyncType::HOST_TO_DEVICE);
            std::cerr << "[FpgaDevice] stagePdiBytes: QDMA sync HOST_TO_DEVICE complete" << std::endl;
        }
        const std::uint64_t phys = buffer->getPhysAddr();
        std::cerr << "[FpgaDevice] stagePdiBytes: staged " << bytes.size()
                  << " bytes via QDMA DDR @ 0x" << std::hex << phys
                  << std::dec << std::endl;
        stagedPdis_.emplace(cacheKey, StagedPdiRecord{std::move(buffer), phys, bytes.size()});
        return phys;
    }

    const std::string name = "__pdi_" + std::to_string(std::hash<std::string>{}(cacheKey));
    GraphBuffer token = GraphBuffer::make(BufferType::U8, name, 0);
    const BufferRecord rec = ensureBuffer(token, bytes.size());
    std::cerr << "[FpgaDevice] stagePdiBytes: WARNING no QDMA staging device configured; "
              << "falling back to BAR/RP1 window staging for " << bytes.size()
              << " bytes -> BAR window offset 0x" << std::hex << rec.offset
              << std::dec << std::endl;
    if (!bytes.empty()) {
        window_->writeAt(rec.offset, bytes.data(), bytes.size());
    }
    std::cerr << "[FpgaDevice] stagePdiBytes: BAR write complete" << std::endl;
    return RP1_CTRL_PHYS_ADDR + static_cast<std::uint64_t>(rec.offset);
}

std::uint64_t FpgaDevice::stagePdiFile(const std::string& pdiPath) {
    std::cerr << "[FpgaDevice] stagePdiFile: opening " << pdiPath << std::endl;
    std::ifstream in(pdiPath, std::ios::binary | std::ios::ate);
    if (!in) {
        throw std::runtime_error("FpgaDevice: cannot open PDI file '" + pdiPath + "'");
    }
    const std::streamsize size = in.tellg();
    if (size < 0) {
        throw std::runtime_error("FpgaDevice: failed to size PDI file '" + pdiPath + "'");
    }
    std::cerr << "[FpgaDevice] stagePdiFile: file size is " << size << " bytes" << std::endl;
    in.seekg(0, std::ios::beg);
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
    std::cerr << "[FpgaDevice] stagePdiFile: reading PDI into host memory" << std::endl;
    if (!bytes.empty() &&
        !in.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("FpgaDevice: short read while staging PDI file '" + pdiPath + "'");
    }
    std::cerr << "[FpgaDevice] stagePdiFile: host read complete" << std::endl;

    return stagePdiBytes(pdiPath, bytes);
}

void FpgaDevice::setActiveImage(std::string imageId) {
    if (vbinSpec_ && !vbinSpec_->hasImage(imageId)) {
        throw std::runtime_error("FpgaDevice: cannot activate unknown image '" + imageId + "'");
    }
    std::lock_guard<std::mutex> lk(imageMutex_);
    activeImageId_ = std::move(imageId);
}

std::string FpgaDevice::activeImageId() const {
    std::lock_guard<std::mutex> lk(imageMutex_);
    return activeImageId_;
}

void FpgaDevice::setInputBuffer(const std::string& bufferName,
                                const void*        data,
                                std::size_t        sizeBytes) {
    const std::string key = normalizeBufferKey(bufferName);
    BufferRecord rec;
    {
        std::lock_guard<std::mutex> lk(bufferMutex_);
        auto it = buffers_.find(key);
        if (it == buffers_.end() || it->second.capacity < sizeBytes) {
            const std::uint32_t alignedOffset = alignUp(nextBufferOffset_, 64u);
            const std::uint64_t end =
                static_cast<std::uint64_t>(alignedOffset) + static_cast<std::uint64_t>(sizeBytes);
            if (end > fpga::Rp1BarWindow::kWindowSize) {
                throw std::out_of_range(
                    "FpgaDevice: BAR-backed buffer arena exhausted while setting '" +
                    bufferName + "' (" + std::to_string(sizeBytes) + " bytes)");
            }
            BufferRecord fresh;
            fresh.offset = alignedOffset;
            fresh.size = sizeBytes;
            fresh.capacity = sizeBytes;
            fresh.type = (it == buffers_.end()) ? BufferType::U8 : it->second.type;
            it = buffers_.insert_or_assign(key, fresh).first;
            nextBufferOffset_ = static_cast<std::uint32_t>(end);
        } else {
            it->second.size = sizeBytes;
        }
        rec = it->second;

        if (sizeBytes == 0) return;
        if (data) {
            window_->writeAt(rec.offset, data, sizeBytes);
        } else {
            window_->zeroAt(rec.offset, sizeBytes);
        }
    }
}

void FpgaDevice::getOutputBuffer(const std::string& bufferName,
                                 void*              data,
                                 std::size_t        sizeBytes) const {
    if (sizeBytes == 0) return;
    if (!data) {
        throw std::invalid_argument("FpgaDevice::getOutputBuffer: data must not be null");
    }

    const std::string key = normalizeBufferKey(bufferName);
    std::lock_guard<std::mutex> lk(bufferMutex_);
    auto it = buffers_.find(key);
    if (it == buffers_.end()) {
        throw std::runtime_error("FpgaDevice::getOutputBuffer: unknown buffer '" + bufferName + "'");
    }
    if (sizeBytes > it->second.size) {
        throw std::out_of_range(
            "FpgaDevice::getOutputBuffer: requested " + std::to_string(sizeBytes) +
            " bytes but buffer '" + bufferName + "' holds " +
            std::to_string(it->second.size));
    }
    window_->readAt(it->second.offset, data, sizeBytes);
}

std::size_t FpgaDevice::bufferSize(const std::string& bufferName) const {
    const std::string key = normalizeBufferKey(bufferName);
    std::lock_guard<std::mutex> lk(bufferMutex_);
    auto it = buffers_.find(key);
    return (it == buffers_.end()) ? 0 : it->second.size;
}

std::unique_ptr<IDevicePlan> FpgaDevice::compilePlan(const DGraph& dg) {
    // -------------------------------------------------------------------
    // Pass 1: reject unsupported node variants and collect kernel nodes
    //         in topological (= DGraph) order.
    // -------------------------------------------------------------------
    std::vector<const CompiledKernelNode*> kernels;
    kernels.reserve(dg.nodes.size());
    bool hasBridgeOps = false;
    bool hasReprogramOps = false;

    for (const CompiledNode& node : dg.nodes) {
        std::visit(
            [&](const auto& concrete) {
                using T = std::decay_t<decltype(concrete)>;
                if constexpr (std::is_same_v<T, CompiledKernelNode>) {
                    if (concrete.kernel.type != DeviceType::FPGA) {
                        throw std::logic_error(
                            std::string("FpgaDevice phase 1: kernel '") +
                            concrete.kernel.name +
                            "' has DeviceType::" + deviceTypeName(concrete.kernel.type) +
                            "; expected FPGA");
                    }
                    kernels.push_back(&concrete);
                } else if constexpr (std::is_same_v<T, CompiledBridgeOpNode>) {
                    hasBridgeOps = true;
                } else if constexpr (std::is_same_v<T, CompiledReprogramNode>) {
                    hasReprogramOps = true;
                } else if constexpr (std::is_same_v<T, CompiledBoundaryNode>) {
                    throw std::logic_error(
                        std::string("FpgaDevice phase 1: graph-region boundaries are not "
                                    "yet supported, got '") + concrete.id + "'");
                } else if constexpr (std::is_same_v<T, CompiledLoopNode>) {
                    throw std::logic_error(
                        std::string("FpgaDevice phase 1: LOOP nodes are not yet supported, "
                                    "got '") + concrete.id + "' (will lower to RP1_OP_LOOP "
                                    "in a future phase)");
                } else if constexpr (std::is_same_v<T, CompiledConditionalNode>) {
                    throw std::logic_error(
                        std::string("FpgaDevice phase 1: COND nodes are not yet supported, "
                                    "got '") + concrete.id + "' (will lower to RP1_OP_COND "
                                    "in a future phase)");
                } else {
                    static_assert(sizeof(T) == 0, "Unhandled CompiledNode variant");
                }
            },
            node);
    }

    if (hasBridgeOps || hasReprogramOps) {
        return std::make_unique<FpgaDevicePlan>(*this,
                                                dg,
                                                dg.scalarValues,
                                                sentinelSlot_,
                                                sentinelValue_,
                                                waitTimeout_);
    }

    if (kernels.empty()) {
        throw std::logic_error("FpgaDevice: DGraph has no kernels to compile");
    }
    if (kernels.size() > kKernelBitsPerBucket) {
        throw std::logic_error(
            "FpgaDevice phase 1: maximum " + std::to_string(kKernelBitsPerBucket) +
            " kernels per graph (bucket 0 bit 31 is reserved for the sentinel SIGNAL); "
            "got " + std::to_string(kernels.size()) +
            ". Spreading kernels across buckets via NOP bridges is a future phase.");
    }

    // -------------------------------------------------------------------
    // Pass 2: allocate barrier bits and compute lookup tables.
    // -------------------------------------------------------------------
    std::unordered_map<std::string, std::size_t> idToIdx;
    idToIdx.reserve(kernels.size());
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        idToIdx[kernels[i]->id] = i;
    }

    std::vector<std::uint32_t> bitMask(kernels.size());
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        bitMask[i] = 1u << i;  // bits 0..kKernelBitsPerBucket-1 in bucket 0
    }

    // Identify "leaf" kernels (nothing within this DGraph depends on them).
    std::vector<bool> isLeaf(kernels.size(), true);
    for (std::size_t i = 0; i < kernels.size(); ++i) {
        for (const std::string& depId : kernels[i]->dependsOn) {
            auto it = idToIdx.find(depId);
            if (it != idToIdx.end()) {
                isLeaf[it->second] = false;
            }
            // Cross-DGraph deps are silently ignored in phase 1 (no bridges).
        }
    }

    // -------------------------------------------------------------------
    // Pass 3: build the RP1 node array + arg buffer + deferred-scalar list.
    // -------------------------------------------------------------------
    fpga::Rp1GraphImage         image;
    std::vector<DeferredScalar>  deferred;

    image.nodes.reserve(kernels.size() + 1);
    image.arg_buf.reserve(/*words*/ 0);

    std::uint32_t cursor_words = 0;
    std::uint32_t leafMask     = 0;

    for (std::size_t i = 0; i < kernels.size(); ++i) {
        const CompiledKernelNode& k = *kernels[i];

        const FpgaKernelLocation loc = resolveKernelLocation(k.kernel);

        // ---- Pack scalar args in IOTypeMap::inputScalars order. ----
        const std::uint32_t this_arg_offset = cursor_words;
        std::uint32_t this_arg_count = 0;

        const auto& boundScalars = k.ioMap.scalars();
        for (const ScalarPort& port : k.kernel.ioType.inputScalars) {
            auto bit = boundScalars.find(port.name);
            if (bit == boundScalars.end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + k.kernel.name +
                    "' input scalar port '" + port.name + "' has no IOMap binding");
            }
            const GraphScalar& gs    = bit->second;
            const std::uint32_t width = scalarWidthInWords(port.type);

            const std::uint64_t total = static_cast<std::uint64_t>(cursor_words) + width;
            if (total > (RP1_DEFAULT_SIG_ARRAY_OFFSET - RP1_DEFAULT_ARG_BUF_OFFSET) /
                            sizeof(std::uint32_t)) {
                throw std::logic_error(
                    "FpgaDevice: argument buffer overflow while packing kernel '" +
                    k.kernel.name + "' port '" + port.name + "'");
            }

            image.arg_buf.resize(cursor_words + width, 0u);

            if (gs.isConstant()) {
                writeScalarToArgWords(port.type, gs.constantBits(),
                                       image.arg_buf.data() + cursor_words);
            } else {
                DeferredScalar d;
                d.scopedKey       = scopedScalarKey(gs.scopeId(), gs.varName());
                d.fallbackKey     = gs.varName();
                d.hasFallback     = (gs.scopeId() == 0);
                d.type            = port.type;
                d.arg_word_offset = cursor_words;
                d.diagnostic      = k.id + "." + port.name;
                deferred.push_back(std::move(d));
            }
            cursor_words   += width;
            this_arg_count += width;
        }

        if (!k.kernel.ioType.outputScalars.empty()) {
            throw std::logic_error(
                "FpgaDevice phase 1: output scalar ports are not yet supported "
                "(kernel '" + k.kernel.name + "' declares " +
                std::to_string(k.kernel.ioType.outputScalars.size()) + "); "
                "RP1_OP_SCALAR_READ lowering lands in a future phase");
        }

        auto currentSize = [this](const GraphBuffer& buffer) -> std::size_t {
            return bufferSize(scopedBufferKey(buffer.scopeId(), buffer.name()));
        };
        std::size_t defaultBufferSize = 0;
        for (const BufferPort& port : k.kernel.ioType.inputBuffers) {
            auto bit = k.ioMap.inputBuffers().find(port.name);
            if (bit == k.ioMap.inputBuffers().end()) continue;
            const std::size_t size = currentSize(bit->second);
            if (size != 0) {
                defaultBufferSize = size;
                break;
            }
        }
        if (defaultBufferSize == 0) {
            for (const RWBufferPort& port : k.kernel.ioType.rwBuffers) {
                auto bit = std::find_if(k.ioMap.rwBuffers().begin(),
                                        k.ioMap.rwBuffers().end(),
                                        [&](const IOMap::RWBinding& binding) {
                                            return binding.inPort == port.in.name &&
                                                   binding.outPort == port.out.name;
                                        });
                if (bit == k.ioMap.rwBuffers().end()) continue;
                const std::size_t size = currentSize(bit->in);
                if (size != 0) {
                    defaultBufferSize = size;
                    break;
                }
            }
        }
        auto appendAddress = [&](const std::string& portName,
                                 const GraphBuffer& buffer,
                                 std::size_t sizeBytes) {
            ensureArgCapacity(cursor_words, 2u, k.kernel.name, portName);
            image.arg_buf.resize(cursor_words + 2u, 0u);
            const std::uint64_t addr = bufferDeviceAddress(buffer, sizeBytes);
            writeU64ToArgWords(addr, image.arg_buf.data() + cursor_words);
            cursor_words += 2u;
            this_arg_count += 2u;
        };

        for (const BufferPort& port : k.kernel.ioType.inputBuffers) {
            auto bit = k.ioMap.inputBuffers().find(port.name);
            if (bit == k.ioMap.inputBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + k.kernel.name +
                    "' input buffer port '" + port.name + "' has no IOMap binding");
            }
            appendAddress(port.name, bit->second, currentSize(bit->second));
        }
        for (const BufferPort& port : k.kernel.ioType.outputBuffers) {
            auto bit = k.ioMap.outputBuffers().find(port.name);
            if (bit == k.ioMap.outputBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + k.kernel.name +
                    "' output buffer port '" + port.name + "' has no IOMap binding");
            }
            appendAddress(port.name, bit->second,
                          std::max(defaultBufferSize, currentSize(bit->second)));
        }
        for (const RWBufferPort& port : k.kernel.ioType.rwBuffers) {
            auto bit = std::find_if(k.ioMap.rwBuffers().begin(),
                                    k.ioMap.rwBuffers().end(),
                                    [&](const IOMap::RWBinding& binding) {
                                        return binding.inPort == port.in.name &&
                                               binding.outPort == port.out.name;
                                    });
            if (bit == k.ioMap.rwBuffers().end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + k.kernel.name +
                    "' RW buffer ports '" + port.in.name + "'/'" +
                    port.out.name + "' have no IOMap binding");
            }
            const std::size_t inSize = currentSize(bit->in);
            appendAddress(port.in.name, bit->in, inSize);
            appendAddress(port.out.name, bit->out,
                          std::max(inSize, currentSize(bit->out)));
        }
        if (this_arg_count > UINT16_MAX) {
            throw std::logic_error(
                "FpgaDevice: kernel '" + k.kernel.name + "' has " +
                std::to_string(this_arg_count) + " arg words, exceeds uint16_t cap");
        }

        // ---- Compose the KERNEL_DISPATCH packet. ----
        rp1_node_t packet{};
        packet.opcode = RP1_OP_KERNEL_DISPATCH;
        packet.flags  = 0;
        packet.status = RP1_NODE_PENDING;

        std::uint32_t await_mask = 0;
        for (const std::string& depId : k.dependsOn) {
            auto it = idToIdx.find(depId);
            if (it == idToIdx.end()) continue;  // cross-DGraph; ignored in phase 1
            await_mask |= bitMask[it->second];
        }
        packet.barrier_await_bucket = 0;
        packet.barrier_await_mask   = await_mask;
        packet.barrier_set_bucket   = 0;
        packet.barrier_set_mask     = bitMask[i];

        auto& kd = packet.payload.kernel_dispatch;
        kd.kernel_base_addr  = loc.r5_base_addr;
        kd.arg_buffer_offset = this_arg_offset * sizeof(std::uint32_t);
        kd.arg_count         = static_cast<std::uint16_t>(this_arg_count);
        kd.ctrl_flags        = 0;
        kd.timeout_cycles    = loc.timeout_cycles;

        image.nodes.push_back(packet);

        if (isLeaf[i]) {
            leafMask |= bitMask[i];
        }
    }

    // -------------------------------------------------------------------
    // Pass 4: emit the trailing sentinel SIGNAL.  It awaits every leaf
    //         kernel's set-bit (OR) and writes (sentinelSlot,
    //         sentinelValue) so the host can confirm whole-graph
    //         completion without scanning the CQ.
    // -------------------------------------------------------------------
    rp1_node_t sentinel{};
    sentinel.opcode = RP1_OP_SIGNAL;
    sentinel.flags  = 0;
    sentinel.status = RP1_NODE_PENDING;
    sentinel.barrier_await_bucket = 0;
    sentinel.barrier_await_mask   = leafMask;
    sentinel.barrier_set_bucket   = kSentinelBucket;
    sentinel.barrier_set_mask     = kSentinelBit;
    sentinel.payload.signal.target_slot = sentinelSlot_;
    sentinel.payload.signal.value       = sentinelValue_;
    sentinel.payload.signal.operation   = RP1_SIGOP_SET;
    image.nodes.push_back(sentinel);

    image.clear_signal_slots.push_back(sentinelSlot_);

    return std::make_unique<FpgaDevicePlan>(submitter_,
                                            std::move(image),
                                            std::move(deferred),
                                            dg.scalarValues,
                                            sentinelSlot_,
                                            sentinelValue_,
                                            waitTimeout_);
}

}  // namespace vrt::graph
