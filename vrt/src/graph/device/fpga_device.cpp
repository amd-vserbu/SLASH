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
#include <cstdint>
#include <cstring>
#include <exception>
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
#include <vrt/graph/node/compiled_node.hpp>
#include <vrt/graph/node/io_type_map.hpp>

namespace vrt::graph {

namespace {

// One bucket = 32 bits.  Bit 31 is reserved for the sentinel.
constexpr std::uint32_t kBarrierBitsPerBucket = 32u;
constexpr std::uint8_t  kSentinelBucket       = 0u;
constexpr std::uint32_t kSentinelBit          = 1u << 31;
constexpr std::uint32_t kKernelBitsPerBucket  = 31u;

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

    ~FpgaDevicePlan() override {
        try { wait(); } catch (...) { /* swallow */ }
    }

    void launch() override {
        wait();
        workerEx_ = nullptr;
        worker_ = std::thread([this] {
            try {
                resolveDeferredScalars();
                submitter_->submitAndWait(image_, timeout_);
                lastCq_ = submitter_->drainCq();
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
      lookup_(std::move(lookup)) {
    if (!window_) {
        throw std::invalid_argument("FpgaDevice: window must not be null");
    }
    if (!lookup_) {
        throw std::invalid_argument("FpgaDevice: kernel-location lookup must not be null");
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

std::unique_ptr<IDevicePlan> FpgaDevice::compilePlan(const DGraph& dg) {
    // -------------------------------------------------------------------
    // Pass 1: reject unsupported node variants and collect kernel nodes
    //         in topological (= DGraph) order.
    // -------------------------------------------------------------------
    std::vector<const CompiledKernelNode*> kernels;
    kernels.reserve(dg.nodes.size());

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
                    throw std::logic_error(
                        std::string("FpgaDevice phase 1: bridge ops are not yet supported, ")
                        + "got '" + concrete.id + "' (cross-device data movement is a "
                        + "future phase; pre-stage FPGA buffers outside the graph for now)");
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

        const FpgaKernelLocation loc = lookup_(k.kernel.name);
        if (loc.r5_base_addr == 0) {
            throw std::runtime_error(
                "FpgaDevice: kernel-location lookup returned r5_base_addr=0 for kernel '" +
                k.kernel.name + "' (likely unmapped name)");
        }

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
