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
#include <cctype>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <optional>
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

/// Default AXI-Lite register offset of the first argument when no system_map
/// is available (mock/lookup path).  Matches HLS' conventional AP-block base
/// and the historical contiguous layout.
constexpr std::uint32_t kApArgBlockBase = 0x10u;

/// Resolves the AXI-Lite register byte offset for each kernel port.
///
/// When constructed from a non-empty `system_map` offset table (real vbin),
/// every port must be present and its offset is honoured exactly.  When the
/// table is empty (mock/lookup path) offsets are handed out contiguously from
/// `kApArgBlockBase`, reproducing the historical dense layout.
class ArgLayout {
   public:
    ArgLayout(std::map<std::string, std::uint32_t> offsets,
              std::string kernelName)
        : offsets_(std::move(offsets)),
          kernelName_(std::move(kernelName)),
          haveSpec_(!offsets_.empty()) {}

    /// Returns the base register offset for a port occupying @p words 32-bit
    /// words.  On the contiguous path this also advances the running cursor.
    std::uint32_t take(const std::string& port, std::uint32_t words) {
        if (haveSpec_) {
            return baseFor(port);
        }
        const std::uint32_t base = fallback_;
        fallback_ += words * 4u;
        return base;
    }

   private:
    std::uint32_t baseFor(const std::string& port) const {
        auto it = offsets_.find(port);
        if (it != offsets_.end()) {
            return it->second;
        }
        // An RW buffer's synthetic "<name>_out" port shares the single HLS
        // pointer register named "<name>".  (Phase-1 limitation: in and out
        // addresses are both written to that one register; tracked for a
        // future protocol revision that distinguishes them.)
        if (port.size() > 4 &&
            port.compare(port.size() - 4, 4, "_out") == 0) {
            auto in = offsets_.find(port.substr(0, port.size() - 4));
            if (in != offsets_.end()) {
                return in->second;
            }
        }
        throw std::runtime_error(
            "FpgaDevice: kernel '" + kernelName_ + "' port '" + port +
            "' has no s_axilite register offset in the system_map");
    }

    std::map<std::string, std::uint32_t> offsets_;
    std::string                          kernelName_;
    bool                                 haveSpec_;
    std::uint32_t                        fallback_ = kApArgBlockBase;
};

/// Appends @p width (reg_offset, value) pairs to @p arg_buf, one per 32-bit
/// value word, with register offset `base_offset + 4*w`.  Returns the arg_buf
/// word index of the first *value* word (every other word from there, because
/// the pairs interleave offset/value) — used to patch deferred scalars later.
std::uint32_t appendArgWordsAsPairs(std::vector<std::uint32_t>& arg_buf,
                                    std::uint32_t& cursor_words,
                                    std::uint32_t base_offset,
                                    const std::uint32_t* words,
                                    std::uint32_t width,
                                    const std::string& kernelName,
                                    const std::string& portName) {
    ensureArgCapacity(cursor_words, width * 2u, kernelName, portName);
    arg_buf.resize(cursor_words + width * 2u, 0u);
    const std::uint32_t firstValueWord = cursor_words + 1u;
    for (std::uint32_t w = 0; w < width; ++w) {
        arg_buf[cursor_words + 2u * w]      = base_offset + 4u * w;
        arg_buf[cursor_words + 2u * w + 1u] = words[w];
    }
    cursor_words += width * 2u;
    return firstValueWord;
}

/// Bookkeeping for a single global-scalar binding that must be
/// re-read from the per-graph scalar map at launch time.
struct DeferredScalar {
    std::string  scopedKey;          // scope:N:varName, with N=scopeId
    std::string  fallbackKey;        // bare varName (used iff scopeId==0)
    bool         hasFallback;
    ScalarType   type;
    std::uint32_t arg_word_offset;   // arg_buf index of the first *value* word
                                     // of this scalar's (offset,value) pairs;
                                     // subsequent words are at +2 each.
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
            // Values live in the odd (value) words of the interleaved
            // (reg_offset, value) pairs, so scatter them at a stride of 2.
            std::uint32_t words[2] = {0u, 0u};
            writeScalarToArgWords(d.type, it->second, words);
            const std::uint32_t width = scalarWidthInWords(d.type);
            for (std::uint32_t w = 0; w < width; ++w) {
                image_.arg_buf[d.arg_word_offset + 2u * w] = words[w];
            }
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
                             ArgLayout& layout,
                             const std::string& portName,
                             const GraphBuffer& buffer,
                             std::size_t sizeBytes,
                             std::uint32_t& cursor_words,
                             std::uint32_t& arg_count) {
        const std::uint64_t addr = device_->bufferDeviceAddress(buffer, sizeBytes);
        std::uint32_t words[2] = {0u, 0u};
        writeU64ToArgWords(addr, words);
        const std::uint32_t base = layout.take(portName, 2u);
        appendArgWordsAsPairs(image.arg_buf, cursor_words, base, words, 2u,
                              node.kernel.name, portName);
        arg_count += 2u;
    }

    std::uint32_t packKernelArgs(fpga::Rp1GraphImage& image,
                                 const CompiledKernelNode& node) {
        std::uint32_t cursor_words = static_cast<std::uint32_t>(image.arg_buf.size());
        std::uint32_t arg_count = 0;

        ArgLayout layout(device_->kernelArgOffsets(node.kernel), node.kernel.name);

        const auto& boundScalars = node.ioMap.scalars();
        for (const ScalarPort& port : node.kernel.ioType.inputScalars) {
            auto it = boundScalars.find(port.name);
            if (it == boundScalars.end()) {
                throw std::runtime_error(
                    "FpgaDevice: kernel '" + node.kernel.name +
                    "' input scalar port '" + port.name + "' has no IOMap binding");
            }
            const std::uint32_t width = scalarWidthInWords(port.type);
            std::uint32_t words[2] = {0u, 0u};
            writeScalarToArgWords(port.type,
                                  scalarBits(it->second, node.id + "." + port.name),
                                  words);
            const std::uint32_t base = layout.take(port.name, width);
            appendArgWordsAsPairs(image.arg_buf, cursor_words, base, words, width,
                                  node.kernel.name, port.name);
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
            appendBufferAddress(image, node, layout, port.name, it->second,
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
            appendBufferAddress(image, node, layout, port.name, it->second,
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
            appendBufferAddress(image, node, layout, port.in.name, it->in,
                                inSize, cursor_words, arg_count);
            appendBufferAddress(image, node, layout, port.out.name, it->out,
                                std::max(inSize, currentBufferSize(it->out)),
                                cursor_words, arg_count);
        }

        if (arg_count > UINT16_MAX) {
            throw std::logic_error(
                "FpgaDevice: kernel '" + node.kernel.name + "' has " +
                std::to_string(arg_count) + " arg pairs, exceeds uint16_t cap");
        }
        return arg_count;
    }

    // Opt-in diagnostic: dump the kernel base and the (reg_offset, value)
    // argument pairs RP1 will write.  Enabled by setting VRT_FPGA_DEBUG_ARGS
    // in the environment; invaluable for confirming the v2 packing against a
    // real s_axilite register map during hardware bring-up.
    static void dumpKernelArgs(const std::string& kernelName,
                               const rp1_payload_kernel_dispatch_t& kd,
                               const std::vector<std::uint32_t>& arg_buf) {
        static const bool enabled = (std::getenv("VRT_FPGA_DEBUG_ARGS") != nullptr);
        if (!enabled) return;
        const std::uint32_t firstWord = kd.arg_buffer_offset / sizeof(std::uint32_t);
        std::cerr << "[FpgaDevice] dispatch '" << kernelName << "' base=0x" << std::hex
                  << kd.kernel_base_addr << std::dec << " arg_count=" << kd.arg_count
                  << " (reg_offset, value) pairs:";
        for (std::uint32_t i = 0; i < kd.arg_count; ++i) {
            const std::uint32_t w = firstWord + 2u * i;
            if (w + 1u >= arg_buf.size()) break;
            std::cerr << "\n    +0x" << std::hex << arg_buf[w]
                      << " = 0x" << arg_buf[w + 1u] << std::dec;
        }
        std::cerr << std::endl;
    }

    std::uint64_t stagePdi(const CompiledReprogramNode& node) {
        if (device_->vbinSpec_ && device_->vbinSpec_->hasImage(node.imageId)) {
            const auto& imageSpec = device_->vbinSpec_->image(node.imageId);
            return device_->stagePdiBytes(imageSpec.id, imageSpec.pdiBytes);
        }
        return device_->stagePdiFile(node.pdiPath);
    }

    const std::vector<std::string>& segNodeDeps(std::size_t rtIdx) const {
        const NodeRuntime& rt = runtime_[rtIdx];
        return (rt.kind == NodeKind::Reprogram) ? rt.reprogram.node.dependsOn
                                                : rt.kernel.node.dependsOn;
    }

    // Build one RP1 graph image for a contiguous FPGA segment (kernels +
    // reprograms, in topological order) plus a trailing sentinel SIGNAL.
    // Intra-segment dependsOn edges become in-image barriers (bucket 0, one
    // bit per node); deps outside the segment are already satisfied by a prior
    // segment or a host bridge, so they carry no barrier. PDI_LOAD and
    // KERNEL_DISPATCH nodes coexist, so a single submission can reconfigure and
    // then dispatch — the reprogram-drain ordering becomes an in-image barrier.
    fpga::Rp1GraphImage buildSegmentImage(const std::vector<std::size_t>& seg) {
        const std::size_t N = seg.size();

        // Node done-bits occupy bits 0..30 of buckets [0, nodeBuckets); bit 31
        // of bucket 0 is the sentinel. For N <= 31 this is a single bucket and
        // no aggregation is needed (bit-identical to the simple case).
        const std::size_t nodeBuckets =
            (N + (kKernelBitsPerBucket - 1)) / kKernelBitsPerBucket;
        if (nodeBuckets >= RP1_MAX_BUCKETS) {
            throw std::logic_error(
                "FpgaDevice: FPGA segment has " + std::to_string(N) + " nodes, needing " +
                std::to_string(nodeBuckets) +
                " barrier buckets and leaving no room for join/sentinel buckets");
        }

        auto nodeBucketOf = [](std::size_t p) -> std::uint8_t {
            return static_cast<std::uint8_t>(p / kKernelBitsPerBucket);
        };
        auto nodeBitOf = [](std::size_t p) -> std::uint32_t {
            return 1u << (p % kKernelBitsPerBucket);
        };

        std::unordered_map<std::size_t, std::size_t> posOf;
        for (std::size_t p = 0; p < N; ++p) posOf[seg[p]] = p;

        std::vector<bool> isLeaf(N, true);
        for (std::size_t p = 0; p < N; ++p) {
            for (const std::string& depId : segNodeDeps(seg[p])) {
                auto it = idToIdx_.find(depId);
                if (it == idToIdx_.end()) continue;
                auto pit = posOf.find(it->second);
                if (pit != posOf.end()) isLeaf[pit->second] = false;
            }
        }

        fpga::Rp1GraphImage image;
        std::vector<rp1_node_t> aggregators;

        // A node can only await ONE bucket, so when a node's predecessor
        // done-bits span multiple buckets we funnel each predecessor bucket
        // through a NOP aggregator into one shared join bucket. Join bits live
        // in buckets above the node-done buckets.
        std::uint8_t  joinBucket = static_cast<std::uint8_t>(nodeBuckets);
        std::uint32_t joinBit    = 0;
        auto resolveAwait =
            [&](const std::map<std::uint8_t, std::uint32_t>& groups)
            -> std::pair<std::uint8_t, std::uint32_t> {
            if (groups.empty()) return {std::uint8_t{0}, 0u};
            if (groups.size() == 1) return {groups.begin()->first, groups.begin()->second};

            const std::uint32_t m = static_cast<std::uint32_t>(groups.size());
            if (joinBit + m > kKernelBitsPerBucket) { ++joinBucket; joinBit = 0; }
            if (joinBucket >= RP1_MAX_BUCKETS) {
                throw std::logic_error(
                    "FpgaDevice: ran out of barrier buckets for cross-bucket join "
                    "aggregation in an FPGA segment");
            }
            const std::uint8_t  cb   = joinBucket;
            const std::uint32_t base = joinBit;
            joinBit += m;

            std::uint32_t consumerMask = 0;
            std::uint32_t j = 0;
            for (const auto& [bucket, mask] : groups) {
                rp1_node_t agg{};
                agg.opcode               = RP1_OP_NOP;
                agg.flags                = RP1_FLAG_SILENT;
                agg.status               = RP1_NODE_PENDING;
                agg.barrier_await_bucket = bucket;
                agg.barrier_await_mask   = mask;
                agg.barrier_set_bucket   = cb;
                agg.barrier_set_mask     = (1u << (base + j));
                consumerMask |= (1u << (base + j));
                ++j;
                aggregators.push_back(agg);
            }
            return {cb, consumerMask};
        };

        for (std::size_t p = 0; p < N; ++p) {
            const NodeRuntime& rt = runtime_[seg[p]];

            std::map<std::uint8_t, std::uint32_t> predGroups;
            for (const std::string& depId : segNodeDeps(seg[p])) {
                auto it = idToIdx_.find(depId);
                if (it == idToIdx_.end()) continue;
                auto pit = posOf.find(it->second);
                if (pit == posOf.end()) continue;
                predGroups[nodeBucketOf(pit->second)] |= nodeBitOf(pit->second);
            }
            const auto [awaitBucket, awaitMask] = resolveAwait(predGroups);

            rp1_node_t pkt{};
            pkt.status               = RP1_NODE_PENDING;
            pkt.barrier_await_bucket = awaitBucket;
            pkt.barrier_await_mask   = awaitMask;
            pkt.barrier_set_bucket   = nodeBucketOf(p);
            pkt.barrier_set_mask     = nodeBitOf(p);

            if (rt.kind == NodeKind::Kernel) {
                const CompiledKernelNode& k = rt.kernel.node;
                const FpgaKernelLocation loc = device_->resolveKernelLocation(k.kernel);
                const std::uint32_t argOffset =
                    static_cast<std::uint32_t>(image.arg_buf.size()) * sizeof(std::uint32_t);
                const std::uint32_t argCount = packKernelArgs(image, k);

                pkt.opcode = RP1_OP_KERNEL_DISPATCH;
                auto& kd = pkt.payload.kernel_dispatch;
                kd.kernel_base_addr  = loc.r5_base_addr;
                kd.arg_buffer_offset = argOffset;
                kd.arg_count         = static_cast<std::uint16_t>(argCount);
                kd.ctrl_flags        = 0;
                kd.timeout_cycles    = loc.timeout_cycles;
                kd.expected_image_id =
                    k.kernel.image ? device_->imageNumericId(*k.kernel.image) : 0u;
                dumpKernelArgs(k.kernel.name, kd, image.arg_buf);
            } else {  // Reprogram
                const CompiledReprogramNode& r = rt.reprogram.node;
                const std::uint64_t pdiAddr = stagePdi(r);
                pkt.opcode = RP1_OP_PDI_LOAD;
                pkt.flags  = RP1_FLAG_HALT_ON_ERROR;
                auto& pl = pkt.payload.pdi_load;
                pl.pdi_addr_lo    = static_cast<std::uint32_t>(pdiAddr & 0xFFFFFFFFull);
                pl.pdi_addr_hi    = static_cast<std::uint32_t>((pdiAddr >> 32) & 0xFFFFFFFFull);
                pl.timeout_cycles = r.timeoutCycles;
                pl.image_id       = device_->imageNumericId(r.imageId);
            }

            image.nodes.push_back(pkt);
        }

        // Sentinel awaits every leaf node (aggregated if leaves span buckets).
        std::map<std::uint8_t, std::uint32_t> leafGroups;
        for (std::size_t p = 0; p < N; ++p) {
            if (isLeaf[p]) leafGroups[nodeBucketOf(p)] |= nodeBitOf(p);
        }
        const auto [sentBucket, sentMask] = resolveAwait(leafGroups);

        // Aggregators (their bits set work-node done-bits) go after the work
        // nodes; the trailing sentinel is last. Array order is irrelevant to
        // the firmware scanner (barriers gate execution), but this keeps it tidy.
        for (const rp1_node_t& agg : aggregators) image.nodes.push_back(agg);

        rp1_node_t sentinel{};
        sentinel.opcode = RP1_OP_SIGNAL;
        sentinel.status = RP1_NODE_PENDING;
        sentinel.barrier_await_bucket = sentBucket;
        sentinel.barrier_await_mask   = sentMask;
        sentinel.barrier_set_bucket   = kSentinelBucket;
        sentinel.barrier_set_mask     = kSentinelBit;
        sentinel.payload.signal.target_slot = sentinelSlot_;
        sentinel.payload.signal.value       = sentinelValue_;
        sentinel.payload.signal.operation   = RP1_SIGOP_SET;
        image.nodes.push_back(sentinel);
        image.clear_signal_slots.push_back(sentinelSlot_);

        if (image.nodes.size() > RP1_MAX_NODES) {
            throw std::logic_error(
                "FpgaDevice: FPGA segment image has " +
                std::to_string(image.nodes.size()) + " nodes, exceeds RP1_MAX_NODES");
        }
        return image;
    }

    // Build, submit, and drain one FPGA segment, then advance the host's
    // active-image bookkeeping past any reprograms the segment contained.
    void submitSegment(const std::vector<std::size_t>& seg) {
        if (seg.empty()) return;
        fpga::Rp1GraphImage image = buildSegmentImage(seg);
        submitter_->submitAndWait(image, timeout_);
        auto cq = submitter_->drainCq();
        lastCq_.insert(lastCq_.end(), cq.begin(), cq.end());
        for (std::size_t idx : seg) {
            const NodeRuntime& rt = runtime_[idx];
            if (rt.kind == NodeKind::Reprogram) {
                device_->setActiveImage(rt.reprogram.node.imageId);
            }
        }
    }

    // Segment scheduler: accumulate contiguous FPGA work (kernels + reprograms)
    // into one RP1 submission, cutting only at host cross-device bridges.
    //
    // A work node is "scheduled" (its successors released) the instant it joins
    // the pending segment, because its in-image barrier orders it; a bridge op
    // is scheduled when its host action runs. Input bridges (consumers) write
    // FPGA buffers and run before the segment that uses them; output bridges
    // (producers) read FPGA buffers, so the segment is flushed before they run.
    void runScheduled() {
        std::vector<std::size_t> unmet;
        unmet.reserve(runtime_.size());
        for (const auto& rt : runtime_) unmet.push_back(rt.initialUnmet);

        std::deque<std::size_t>  readyWork;
        std::vector<std::size_t> readyProducers;
        std::vector<std::size_t> pendingConsumers;
        std::vector<std::size_t> segment;

        auto enqueue = [&](std::size_t idx) {
            switch (runtime_[idx].kind) {
                case NodeKind::Kernel:
                case NodeKind::Reprogram:  readyWork.push_back(idx);       break;
                case NodeKind::ProducerOp: readyProducers.push_back(idx);  break;
                case NodeKind::ConsumerOp: pendingConsumers.push_back(idx); break;
            }
        };
        auto schedule = [&](std::size_t idx) {
            for (std::size_t succ : runtime_[idx].successors) {
                if (--unmet[succ] == 0) enqueue(succ);
            }
        };

        for (std::size_t i = 0; i < runtime_.size(); ++i) {
            if (unmet[i] == 0) enqueue(i);
        }

        std::size_t rrCursor = 0;
        auto idleSince = std::chrono::steady_clock::now();
        for (;;) {
            // 1. Absorb all ready work into the current segment.
            if (!readyWork.empty()) {
                while (!readyWork.empty()) {
                    const std::size_t idx = readyWork.front();
                    readyWork.pop_front();
                    segment.push_back(idx);
                    schedule(idx);
                }
                idleSince = std::chrono::steady_clock::now();
                continue;
            }

            // 2. Run a ready host input bridge (consumer); it gates future work
            //    and never depends on the unsubmitted segment.
            bool fired = false;
            for (std::size_t step = 0; step < pendingConsumers.size(); ++step) {
                if (rrCursor >= pendingConsumers.size()) rrCursor = 0;
                const std::size_t idx = pendingConsumers[rrCursor];
                NodeRuntime& rt = runtime_[idx];
                if (!rt.tryReady || rt.tryReady()) {
                    pendingConsumers.erase(pendingConsumers.begin() +
                                           static_cast<std::ptrdiff_t>(rrCursor));
                    if (rt.action) rt.action();
                    schedule(idx);
                    fired = true;
                    idleSince = std::chrono::steady_clock::now();
                    break;
                }
                ++rrCursor;
            }
            if (fired) continue;

            // 3. Output bridges need the segment's results: flush, then run them.
            if (!readyProducers.empty()) {
                submitSegment(segment);
                segment.clear();
                for (std::size_t idx : readyProducers) {
                    NodeRuntime& rt = runtime_[idx];
                    if (rt.action) rt.action();
                    schedule(idx);
                }
                readyProducers.clear();
                idleSince = std::chrono::steady_clock::now();
                continue;
            }

            // 4. No work, no bridge progress: flush a terminal segment. Flushing
            //    (rather than waiting) is required to break cross-device cycles
            //    where a pending consumer transitively needs this segment's output.
            if (!segment.empty()) {
                submitSegment(segment);
                segment.clear();
                idleSince = std::chrono::steady_clock::now();
                continue;
            }

            // 5. Nothing left to run -> done. Otherwise we are blocked waiting
            //    on a cross-device consumer that is not ready yet.
            if (pendingConsumers.empty()) break;
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
    // An empty initial image is valid and is the default for the authoring API:
    // the user region starts with no active image, so every FPGA dispatch must
    // be gated behind an explicit reprogram (PDI_LOAD) of its image. A
    // non-empty initial image must still exist in the spec.
    if (!activeImageId_.empty() && !vbinSpec_->hasImage(activeImageId_)) {
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

FpgaDevice::BufferRecord FpgaDevice::ensureBufferByKey(const std::string& key,
                                                       BufferType type,
                                                       std::size_t sizeBytes) {
    // Caller must hold bufferMutex_.
    auto regionIt = bufferRegion_.find(key);
    const bool deviceMode = (pdiStagingDevice_ != nullptr) &&
                            (regionIt != bufferRegion_.end());

    auto it = buffers_.find(key);
    if (it != buffers_.end() && it->second.capacity >= sizeBytes &&
        ((it->second.mem != nullptr) == deviceMode)) {
        if (it->second.size < sizeBytes) it->second.size = sizeBytes;
        it->second.type = type;
        return it->second;
    }

    BufferRecord rec;
    rec.size = sizeBytes;
    rec.capacity = sizeBytes;
    rec.type = type;

    if (deviceMode) {
        // Allocate in the region the kernel's m_axi master can reach.  vrt
        // buffers are fixed-size, so allocate at least one byte and grow by
        // reallocation when a later size exceeds the current capacity.
        const std::size_t allocBytes = std::max<std::size_t>(sizeBytes, 1u);
        rec.mem = std::make_shared<::vrt::Buffer<std::uint8_t>>(
            *pdiStagingDevice_, allocBytes, regionIt->second);
        rec.capacity = allocBytes;
    } else {
        const std::uint32_t alignedOffset = alignUp(nextBufferOffset_, 64u);
        const std::uint64_t end = static_cast<std::uint64_t>(alignedOffset) +
                                  static_cast<std::uint64_t>(sizeBytes);
        if (end > fpga::Rp1BarWindow::kWindowSize) {
            throw std::out_of_range(
                "FpgaDevice: BAR-backed buffer arena exhausted while allocating '" +
                key + "' (" + std::to_string(sizeBytes) + " bytes)");
        }
        rec.offset = alignedOffset;
        nextBufferOffset_ = static_cast<std::uint32_t>(end);
    }

    buffers_[key] = rec;
    return rec;
}

FpgaDevice::BufferRecord FpgaDevice::ensureBuffer(const GraphBuffer& buffer,
                                                  std::size_t sizeBytes) {
    const std::string key = scopedBufferKey(buffer.scopeId(), buffer.name());
    std::lock_guard<std::mutex> lk(bufferMutex_);
    return ensureBufferByKey(key, buffer.type(), sizeBytes);
}

std::uint64_t FpgaDevice::bufferDeviceAddress(const GraphBuffer& buffer,
                                              std::size_t sizeBytes) {
    const BufferRecord rec = ensureBuffer(buffer, sizeBytes);
    if (rec.mem) {
        return rec.mem->getPhysAddr();
    }
    return RP1_CTRL_PHYS_ADDR + static_cast<std::uint64_t>(rec.offset);
}

void FpgaDevice::populateBufferRegions(const DGraph& dg) {
    auto record = [&](const KernelDescriptor& kernel, const std::string& portName,
                      const GraphBuffer& buffer) {
        auto region = resolveBufferRegion(kernel, portName);
        if (!region) return;
        const std::string key = scopedBufferKey(buffer.scopeId(), buffer.name());
        std::lock_guard<std::mutex> lk(bufferMutex_);
        bufferRegion_[key] = *region;
    };

    for (const CompiledNode& node : dg.nodes) {
        const auto* k = std::get_if<CompiledKernelNode>(&node);
        if (!k) continue;

        for (const BufferPort& port : k->kernel.ioType.inputBuffers) {
            auto it = k->ioMap.inputBuffers().find(port.name);
            if (it != k->ioMap.inputBuffers().end()) {
                record(k->kernel, port.name, it->second);
            }
        }
        for (const BufferPort& port : k->kernel.ioType.outputBuffers) {
            auto it = k->ioMap.outputBuffers().find(port.name);
            if (it != k->ioMap.outputBuffers().end()) {
                record(k->kernel, port.name, it->second);
            }
        }
        for (const RWBufferPort& port : k->kernel.ioType.rwBuffers) {
            for (const IOMap::RWBinding& binding : k->ioMap.rwBuffers()) {
                if (binding.inPort == port.in.name && binding.outPort == port.out.name) {
                    record(k->kernel, port.in.name, binding.in);
                    record(k->kernel, port.out.name, binding.out);
                }
            }
        }
    }
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

std::map<std::string, std::string>
FpgaDevice::descriptorPortToArgName(const KernelDescriptor& kernel) const {
    std::map<std::string, std::string> out;
    if (!vbinSpec_) {
        return out;
    }
    const std::string active = activeImageId();
    const std::string imageId = kernel.image ? *kernel.image : active;
    if (imageId.empty() || !vbinSpec_->hasImage(imageId)) {
        return out;
    }
    const auto& kernels = vbinSpec_->image(imageId).kernels;
    auto kit = kernels.find(kernel.name);
    if (kit == kernels.end()) {
        return out;
    }
    const IOTypeMap& d = kernel.ioType;  // descriptor (possibly renamed)

    // Map the descriptor's (possibly renamed) ports to the canonical system_map
    // arg names by positional correspondence, grouped only by scalar-vs-buffer.
    //
    // We deliberately do NOT trust the canonical IOTypeMap's input/output buffer
    // categories: the system_map marks every HLS m_axi pointer register as
    // write-only (r=0, w=1, because the *host* writes the pointer address), so
    // ioTypeMapFromFunctionalArgs lumps all buffer pointers into inputBuffers
    // regardless of data-flow direction.  A descriptor that splits ports into
    // input/output by intent would then fail to line up per-category.  Instead
    // we use the spec's idx-ordered `args` (the authoritative argument order)
    // and split scalar vs buffer by the arg type (matching the classification
    // in ioTypeMapFromFunctionalArgs), not by the m_axi port (a buffer may have
    // no connection and an empty port).
    auto isBufferArg = [](const fpga::FpgaKernelArgSpec& arg) {
        std::string t = arg.type;
        std::transform(t.begin(), t.end(), t.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        return t == "buffer" || t.find('*') != std::string::npos;
    };
    std::vector<std::string> specScalars;
    std::vector<std::string> specBuffers;
    for (const fpga::FpgaKernelArgSpec& arg : kit->second.args) {
        if (isBufferArg(arg)) {
            specBuffers.push_back(arg.name);
        } else {
            specScalars.push_back(arg.name);
        }
    }

    // Flatten the descriptor ports in the order the packer emits them
    // (scalars first, then input/output buffers, then RW in-pointers).
    std::vector<std::string> descScalars;
    std::vector<std::string> descBuffers;
    for (const ScalarPort& p : d.inputScalars)  descScalars.push_back(p.name);
    for (const ScalarPort& p : d.outputScalars) descScalars.push_back(p.name);
    for (const BufferPort& p : d.inputBuffers)  descBuffers.push_back(p.name);
    for (const BufferPort& p : d.outputBuffers) descBuffers.push_back(p.name);
    // An RW pair collapses onto a single underlying pointer arg: its in-port
    // consumes one buffer slot; its out-port aliases the same arg afterwards.
    for (const RWBufferPort& p : d.rwBuffers) descBuffers.push_back(p.in.name);

    for (std::size_t i = 0; i < descScalars.size() && i < specScalars.size(); ++i) {
        out[descScalars[i]] = specScalars[i];
    }
    for (std::size_t i = 0; i < descBuffers.size() && i < specBuffers.size(); ++i) {
        out[descBuffers[i]] = specBuffers[i];
    }
    for (const RWBufferPort& p : d.rwBuffers) {
        auto it = out.find(p.in.name);
        if (it != out.end()) {
            out[p.out.name] = it->second;
        }
    }
    return out;
}

std::map<std::string, std::uint32_t>
FpgaDevice::kernelArgOffsets(const KernelDescriptor& kernel) const {
    std::map<std::string, std::uint32_t> offsets;
    if (!vbinSpec_) {
        return offsets;  // mock/lookup path: caller falls back to 0x10 layout
    }
    const std::string active = activeImageId();
    const std::string imageId = kernel.image ? *kernel.image : active;
    if (imageId.empty() || !vbinSpec_->hasImage(imageId)) {
        return offsets;
    }
    const auto& kernels = vbinSpec_->image(imageId).kernels;
    auto kit = kernels.find(kernel.name);
    if (kit == kernels.end()) {
        return offsets;
    }
    std::map<std::string, std::uint32_t> byArgName;
    for (const fpga::FpgaKernelArgSpec& arg : kit->second.args) {
        byArgName[arg.name] = arg.offset;
    }
    // Key offsets by the descriptor's port names so the packer (which iterates
    // node.kernel.ioType) finds them even when ports were renamed.
    const auto trans = descriptorPortToArgName(kernel);
    for (const auto& [descPort, argName] : trans) {
        auto a = byArgName.find(argName);
        if (a != byArgName.end()) {
            offsets[descPort] = a->second;
        }
    }
    // Fallback for descriptors with no IOTypeMap to zip against: key by arg
    // name (preserves behaviour for specs whose port names already match).
    if (offsets.empty()) {
        offsets = std::move(byArgName);
    }
    return offsets;
}

std::optional<::vrt::MemoryConfig>
FpgaDevice::resolveBufferRegion(const KernelDescriptor& kernel,
                                const std::string& portName) const {
    if (!vbinSpec_) {
        return std::nullopt;  // mock/lookup path: caller uses the BAR arena
    }
    const std::string active = activeImageId();
    const std::string imageId = kernel.image ? *kernel.image : active;
    if (imageId.empty() || !vbinSpec_->hasImage(imageId)) {
        return std::nullopt;
    }
    const auto& kernels = vbinSpec_->image(imageId).kernels;
    auto kit = kernels.find(kernel.name);
    if (kit == kernels.end()) {
        return std::nullopt;
    }
    const auto& argMemory = kit->second.argMemory;

    // Translate the (possibly renamed) descriptor port to its system_map arg
    // name, using the same correspondence the offset packer uses so regions
    // and offsets stay consistent.
    const auto trans = descriptorPortToArgName(kernel);
    auto t = trans.find(portName);
    const std::string& argName = (t != trans.end()) ? t->second : portName;

    auto it = argMemory.find(argName);
    if (it != argMemory.end()) {
        return it->second;
    }
    return std::nullopt;
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

std::uint32_t FpgaDevice::imageNumericId(const std::string& imageId) const {
    if (!vbinSpec_ || imageId.empty()) {
        return 0;  // mock/lookup path or no image: guard disabled
    }
    // 1-based index in the spec's (name-sorted) image map. The id only needs
    // to be self-consistent within this device: firmware compares equality of
    // whatever the host wrote on PDI_LOAD vs KERNEL_DISPATCH.
    std::uint32_t id = 1;
    for (const auto& [name, spec] : vbinSpec_->images()) {
        (void)spec;
        if (name == imageId) return id;
        ++id;
    }
    return 0;  // unknown image: leave unguarded rather than mis-gate
}

void FpgaDevice::setInputBuffer(const std::string& bufferName,
                                const void*        data,
                                std::size_t        sizeBytes) {
    const std::string key = normalizeBufferKey(bufferName);
    std::lock_guard<std::mutex> lk(bufferMutex_);

    BufferType type = BufferType::U8;
    if (auto existing = buffers_.find(key); existing != buffers_.end()) {
        type = existing->second.type;
    }
    const BufferRecord rec = ensureBufferByKey(key, type, sizeBytes);

    if (sizeBytes == 0) return;
    if (rec.mem) {
        // Device-memory mode: stage into the host mapping and DMA to the
        // region the kernel reads from.
        if (data) {
            std::memcpy(rec.mem->get(), data, sizeBytes);
        } else {
            std::memset(rec.mem->get(), 0, sizeBytes);
        }
        rec.mem->sync(::vrt::SyncType::HOST_TO_DEVICE);
    } else if (data) {
        window_->writeAt(rec.offset, data, sizeBytes);
    } else {
        window_->zeroAt(rec.offset, sizeBytes);
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
    if (it->second.mem) {
        // Device-memory mode: DMA the kernel's results back before copying out.
        it->second.mem->sync(::vrt::SyncType::DEVICE_TO_HOST);
        std::memcpy(data, it->second.mem->get(), sizeBytes);
    } else {
        window_->readAt(it->second.offset, data, sizeBytes);
    }
}

std::size_t FpgaDevice::bufferSize(const std::string& bufferName) const {
    const std::string key = normalizeBufferKey(bufferName);
    std::lock_guard<std::mutex> lk(bufferMutex_);
    auto it = buffers_.find(key);
    return (it == buffers_.end()) ? 0 : it->second.size;
}

std::unique_ptr<IDevicePlan> FpgaDevice::compilePlan(const DGraph& dg) {
    // Resolve each kernel buffer's m_axi memory region up front so later
    // allocation (which may happen at bridge-consumer time, before the kernel
    // is packed) lands where the kernel master can reach it.
    populateBufferRegions(dg);

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

    // Reprogram / bridge graphs, and any kernels-only graph too large for a
    // single barrier bucket, go through the segment scheduler, which fuses
    // contiguous FPGA work into whole-DGraph submissions and allocates barrier
    // bits across multiple buckets (with NOP join aggregators).
    if (hasBridgeOps || hasReprogramOps || kernels.size() > kKernelBitsPerBucket) {
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
        // Protocol v2: the arg buffer is an array of (reg_offset, value)
        // pairs; arg_count counts pairs.  Register offsets come from the
        // system_map (or a contiguous 0x10 fallback on the mock path).
        const std::uint32_t this_arg_offset = cursor_words;
        std::uint32_t this_arg_count = 0;
        ArgLayout layout(kernelArgOffsets(k.kernel), k.kernel.name);

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
            const std::uint32_t base  = layout.take(port.name, width);

            std::uint32_t words[2] = {0u, 0u};
            if (gs.isConstant()) {
                writeScalarToArgWords(port.type, gs.constantBits(), words);
            }
            const std::uint32_t firstValueWord = appendArgWordsAsPairs(
                image.arg_buf, cursor_words, base, words, width,
                k.kernel.name, port.name);

            if (!gs.isConstant()) {
                DeferredScalar d;
                d.scopedKey       = scopedScalarKey(gs.scopeId(), gs.varName());
                d.fallbackKey     = gs.varName();
                d.hasFallback     = (gs.scopeId() == 0);
                d.type            = port.type;
                d.arg_word_offset = firstValueWord;
                d.diagnostic      = k.id + "." + port.name;
                deferred.push_back(std::move(d));
            }
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
            const std::uint64_t addr = bufferDeviceAddress(buffer, sizeBytes);
            std::uint32_t words[2] = {0u, 0u};
            writeU64ToArgWords(addr, words);
            const std::uint32_t base = layout.take(portName, 2u);
            appendArgWordsAsPairs(image.arg_buf, cursor_words, base, words, 2u,
                                  k.kernel.name, portName);
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
                std::to_string(this_arg_count) + " arg pairs, exceeds uint16_t cap");
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
        kd.expected_image_id = k.kernel.image ? imageNumericId(*k.kernel.image) : 0u;

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
