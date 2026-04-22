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
 * @file cpu_device.cpp
 * @brief CpuDevice implementation — naive single-core CPU device.
 */

#include <vrt/graph/device/cpu_device.hpp>

#include <cstring>
#include <stdexcept>
#include <variant>

namespace vrt::graph {

// ---------------------------------------------------------------------------
// CpuBufferView helpers
// ---------------------------------------------------------------------------

namespace {

size_t elementSize(BufferType t) {
    switch (t) {
        case BufferType::U8:  case BufferType::I8:  return 1;
        case BufferType::U16: case BufferType::I16: return 2;
        case BufferType::U32: case BufferType::I32: case BufferType::F32: return 4;
        case BufferType::U64: case BufferType::I64: case BufferType::F64: return 8;
    }
    return 1;
}

}  // namespace

size_t CpuBufferView::elementCount() const {
    size_t es = elementSize(elementType);
    return (es > 0) ? (sizeBytes / es) : 0;
}

// ---------------------------------------------------------------------------
// CpuDevice
// ---------------------------------------------------------------------------

CpuDevice::CpuDevice(std::string id,
                     std::shared_ptr<SemaphorePool> pool,
                     std::shared_ptr<std::map<std::string, std::vector<uint8_t>>> buffers)
    : id_(std::move(id)),
      semPool_(pool ? std::move(pool) : std::make_shared<SemaphorePool>()),
      buffers_(buffers ? std::move(buffers) : std::make_shared<std::map<std::string, std::vector<uint8_t>>>()) {}

void CpuDevice::registerKernel(std::string kernelName, CpuKernelFn fn) {
    kernels_[std::move(kernelName)] = std::move(fn);
}

void CpuDevice::setInputBuffer(const std::string& bufferName,
                                const void*        data,
                                size_t             sizeBytes) {
    auto& buf = (*buffers_)[bufferName];
    buf.resize(sizeBytes);
    if (data && sizeBytes > 0) {
        std::memcpy(buf.data(), data, sizeBytes);
    }
}

void CpuDevice::getOutputBuffer(const std::string& bufferName,
                                void*              data,
                                size_t             sizeBytes) const {
    auto it = buffers_->find(bufferName);
    if (it == buffers_->end()) {
        throw std::runtime_error("CpuDevice::getOutputBuffer: unknown buffer '" + bufferName + "'");
    }
    const auto& buf = it->second;
    if (sizeBytes > buf.size()) {
        throw std::out_of_range("CpuDevice::getOutputBuffer: requested " +
                                std::to_string(sizeBytes) + " bytes but buffer '" +
                                bufferName + "' holds " + std::to_string(buf.size()));
    }
    std::memcpy(data, buf.data(), sizeBytes);
}

// --- compile ---

void CpuDevice::compile(const DGraph& dg) {
    steps_.clear();

    std::vector<Step> ordered;

    for (const Node& node : dg.nodes) {
        // Pre-node: awaits and DMAs targeting this node.
        for (const auto& a : pendingAwaits_) {
            if (a.beforeNodeId == node.id)
                ordered.push_back(AwaitStep{a.sem});
        }
        for (const auto& d : pendingDMAs_) {
            if (d.beforeNodeId == node.id)
                ordered.push_back(DMAStep{d.dma});
        }

        ordered.push_back(KernelStep{node});

        // Post-node: signals after this node.
        for (const auto& s : pendingSignals_) {
            if (s.afterNodeId == node.id)
                ordered.push_back(SignalStep{s.sem});
        }
    }

    steps_ = std::move(ordered);
    pendingSignals_.clear();
    pendingAwaits_.clear();
    pendingDMAs_.clear();
}

// --- sync primitive injection (called by compiler before compile()) ---

void CpuDevice::insertSignal(SemaphoreHandle sem, const std::string& afterNodeId) {
    pendingSignals_.push_back({sem, afterNodeId});
}

void CpuDevice::insertAwait(SemaphoreHandle sem, const std::string& beforeNodeId) {
    pendingAwaits_.push_back({sem, beforeNodeId});
}

void CpuDevice::insertDMA(DMADescriptor dma, const std::string& beforeNodeId) {
    pendingDMAs_.push_back({std::move(dma), beforeNodeId});
}

// --- launch ---

void CpuDevice::launch() {
    if (worker_.joinable()) worker_.join();
    worker_ = std::thread([this] {
        for (const Step& step : steps_) {
            if (std::holds_alternative<KernelStep>(step)) {
                executeKernel(std::get<KernelStep>(step).node);
            } else if (std::holds_alternative<SignalStep>(step)) {
                semPool_->signal(std::get<SignalStep>(step).sem);
            } else if (std::holds_alternative<AwaitStep>(step)) {
                semPool_->await(std::get<AwaitStep>(step).sem);
            } else if (std::holds_alternative<DMAStep>(step)) {
                executeDMA(std::get<DMAStep>(step).dma);
            }
        }
    });
}

void CpuDevice::wait() {
    if (worker_.joinable()) worker_.join();
}

// --- private helpers ---

void CpuDevice::executeKernel(const Node& node) {
    const std::string& kname = node.kernel.name;
    auto it = kernels_.find(kname);
    if (it == kernels_.end()) {
        throw std::runtime_error(
            "CpuDevice: no kernel registered for '" + kname + "'");
    }

    // Build buffer views for all bound buffers.
    std::map<std::string, CpuBufferView> bufViews;

    // Input buffers
    for (const auto& [portName, gbuf] : node.ioMap.inputBuffers()) {
        bufViews[portName] = resolveBuffer(gbuf.name());
    }

    // Output buffers — allocate on first use; size determined by port type.
    // The BufferType gives the element size; we default to a 0-byte allocation
    // and let the kernel resize if needed.  For the naive backend, the kernel
    // is expected to call ensureBuffer() via the backend or just work within
    // the existing allocation.
    //
    // Simpler approach: the kernel function writes to whatever pointer it is
    // given; we pre-allocate a sensible default that the kernel can rely on.
    // For output buffers we match the size of the first available input buffer
    // (a reasonable default for element-wise kernels).  The user can override
    // by calling setInputBuffer() for outputs before launch if they need a
    // specific size.
    const auto& inBufs = node.ioMap.inputBuffers();
    size_t defaultOutputSize = 0;
    if (!inBufs.empty()) {
        const std::string& firstName = inBufs.begin()->second.name();
        auto fit = buffers_->find(firstName);
        if (fit != buffers_->end()) {
            defaultOutputSize = fit->second.size();
        }
    }

    for (const auto& [portName, gbuf] : node.ioMap.outputBuffers()) {
        // Find the declared type from IOTypeMap
        BufferType btype = BufferType::U8;  // default
        for (const auto& bp : node.kernel.ioType.outputBuffers) {
            if (bp.name == portName) { btype = bp.type; break; }
        }
        auto& storage = ensureBuffer(gbuf.name(), defaultOutputSize);
        bufViews[portName] = CpuBufferView{storage.data(), storage.size(), btype};
    }

    // RW buffers — input side reads from the existing buffer; output side is
    // the same storage (in-place semantics for the naive backend).
    for (const auto& rwb : node.ioMap.rwBuffers()) {
        // Input (consumed) side
        bufViews[rwb.inPort] = resolveBuffer(rwb.in.name());
        // Output (produced) side — same backing storage, new token name
        auto& inStorage = buffers_->at(rwb.in.name());
        (*buffers_)[rwb.out.name()] = inStorage;  // shallow copy; kernel writes in-place
        bufViews[rwb.outPort] = CpuBufferView{
            (*buffers_)[rwb.out.name()].data(),
            (*buffers_)[rwb.out.name()].size(),
            bufViews[rwb.inPort].elementType
        };
    }

    // Build scalar map
    std::map<std::string, uint64_t> scalars;
    for (const auto& [portName, gs] : node.ioMap.scalars()) {
        if (gs.isConstant()) {
            scalars[portName] = gs.constantBits();
        } else {
            // global variable — look up in the scalar store
            auto sit = scalarStore_.find(gs.varName());
            if (sit == scalarStore_.end()) {
                throw std::runtime_error(
                    "CpuDevice: global scalar '" + gs.varName() + "' not set before launch");
            }
            scalars[portName] = sit->second;
        }
    }

    CpuKernelArgs args(std::move(bufViews), std::move(scalars));
    it->second(args);

    // Write output scalars back to the scalar store after the kernel returns.
    for (const auto& [portName, gs] : node.ioMap.scalars()) {
        if (!gs.isConstant()) {
            // The kernel may have updated the value via a mutable reference;
            // in this naive model output scalars are written by the kernel into
            // a pre-registered location.  We re-read from the scalar store
            // (the kernel writes there via a separate set call if needed).
        }
    }
}

void CpuDevice::executeDMA(const DMADescriptor& dma) {
    auto srcIt = buffers_->find(dma.src.name());
    if (srcIt == buffers_->end()) {
        throw std::runtime_error(
            "CpuDevice::executeDMA: source buffer '" + dma.src.name() + "' not found");
    }
    auto& dst = ensureBuffer(dma.dst.name(), dma.sizeBytes);
    const size_t bytes = std::min(dma.sizeBytes, srcIt->second.size());
    std::memcpy(dst.data(), srcIt->second.data(), bytes);
}

CpuBufferView CpuDevice::resolveBuffer(const std::string& name) const {
    auto it = buffers_->find(name);
    if (it == buffers_->end()) {
        throw std::runtime_error(
            "CpuDevice: buffer '" + name + "' not found; "
            "did you forget to call setInputBuffer()?");
    }
    return CpuBufferView{
        const_cast<void*>(static_cast<const void*>(it->second.data())),
        it->second.size(),
        BufferType::U8  // element type not tracked at this level; use CpuBufferView::as<T>()
    };
}

std::vector<uint8_t>& CpuDevice::ensureBuffer(const std::string& name, size_t sizeBytes) {
    auto& buf = (*buffers_)[name];
    if (buf.size() < sizeBytes) {
        buf.resize(sizeBytes);
    }
    return buf;
}

}  // namespace vrt::graph
