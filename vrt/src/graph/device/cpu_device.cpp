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
#include <vector>

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

CpuDevice::CpuDevice(std::string id) : id_(std::move(id)) {}

void CpuDevice::registerKernel(std::string kernelName, CpuKernelFn fn) {
    kernels_[std::move(kernelName)] = std::move(fn);
}

void CpuDevice::setInputBuffer(const std::string& bufferName,
                                const void*        data,
                                size_t             sizeBytes) {
    auto& buf = buffers_[bufferName];
    buf.resize(sizeBytes);
    if (data && sizeBytes > 0) {
        std::memcpy(buf.data(), data, sizeBytes);
    }
}

void CpuDevice::getOutputBuffer(const std::string& bufferName,
                                void*              data,
                                size_t             sizeBytes) const {
    auto it = buffers_.find(bufferName);
    if (it == buffers_.end()) {
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

size_t CpuDevice::bufferSize(const std::string& bufferName) const {
    auto it = buffers_.find(bufferName);
    return (it == buffers_.end()) ? 0 : it->second.size();
}

// --- compile ---

void CpuDevice::compile(const DGraph& dg) {
    runtime_.clear();
    runtime_.reserve(dg.nodes.size());
    idToIdx_.clear();
    idToIdx_.reserve(dg.nodes.size());

    // First pass: build per-node runtime records, keyed by id.
    for (const Node& node : dg.nodes) {
        NodeRuntime rt;
        std::visit(
            [&](const auto& n) {
                using T = std::decay_t<decltype(n)>;
                rt.id = n.id;
                if constexpr (std::is_same_v<T, KernelNode>) {
                    rt.kind   = NodeKind::Kernel;
                    rt.kernel = n;
                } else if constexpr (std::is_same_v<T, BridgeOpNode>) {
                    rt.kind = (n.side == BridgeOpNode::Side::Producer)
                                  ? NodeKind::ProducerOp
                                  : NodeKind::ConsumerOp;
                    rt.tryReady = n.tryReady;
                    rt.action   = n.action;
                }
            },
            node);
        idToIdx_[rt.id] = runtime_.size();
        runtime_.push_back(std::move(rt));
    }

    // Second pass: convert dependsOn ids → indices, build successors + unmet.
    // dependsOn may legitimately reference ids from other DGraphs (the
    // compiler annotates bounce-leg producers with the original cross-device
    // kernel id). Ids not local to this DGraph are ignored — cross-device
    // synchronisation is enforced by the bridge's tryReady probe instead.
    for (size_t i = 0; i < dg.nodes.size(); ++i) {
        const auto& deps = nodeDependsOn(dg.nodes[i]);
        for (const std::string& depId : deps) {
            auto it = idToIdx_.find(depId);
            if (it == idToIdx_.end()) continue;
            runtime_[it->second].successors.push_back(i);
            ++runtime_[i].unmet;
        }
    }
}

// --- launch ---

void CpuDevice::launch() {
    if (worker_.joinable()) worker_.join();
    worker_ = std::thread([this] {
        // Initial frontier: every node whose dependsOn set is empty.
        std::vector<size_t> readyKP;       // kernels + producer-side ops, FIFO
        std::vector<size_t> pendingCons;   // consumer-side ops awaiting tryReady
        readyKP.reserve(runtime_.size());

        auto promote = [&](size_t idx) {
            if (runtime_[idx].kind == NodeKind::ConsumerOp) {
                pendingCons.push_back(idx);
            } else {
                readyKP.push_back(idx);
            }
        };

        for (size_t i = 0; i < runtime_.size(); ++i) {
            if (runtime_[i].unmet == 0) promote(i);
        }

        auto runIndex = [&](size_t idx) {
            NodeRuntime& rt = runtime_[idx];
            if (rt.kind == NodeKind::Kernel) {
                executeKernel(rt.kernel);
            } else {
                rt.action();
            }
            for (size_t s : rt.successors) {
                if (--runtime_[s].unmet == 0) promote(s);
            }
        };

        size_t rrCursor = 0;
        for (;;) {
            // Drain all currently-ready kernels and producer-side ops.
            while (!readyKP.empty()) {
                size_t idx = readyKP.front();
                readyKP.erase(readyKP.begin());
                runIndex(idx);
            }
            if (pendingCons.empty()) break;

            // Round-robin poll the consumer-side ops.
            bool fired = false;
            for (size_t step = 0; step < pendingCons.size(); ++step) {
                if (rrCursor >= pendingCons.size()) rrCursor = 0;
                size_t idx = pendingCons[rrCursor];
                if (runtime_[idx].tryReady && runtime_[idx].tryReady()) {
                    pendingCons.erase(pendingCons.begin() +
                                      static_cast<std::ptrdiff_t>(rrCursor));
                    runIndex(idx);
                    fired = true;
                    break;
                }
                ++rrCursor;
            }
            if (!fired) {
                std::this_thread::yield();
            }
        }
    });
}

void CpuDevice::wait() {
    if (worker_.joinable()) worker_.join();
}

// --- private helpers ---

void CpuDevice::executeKernel(const KernelNode& node) {
    const std::string& kname = node.kernel.name;
    auto it = kernels_.find(kname);
    if (it == kernels_.end()) {
        throw std::runtime_error(
            "CpuDevice: no kernel registered for '" + kname + "'");
    }

    std::map<std::string, CpuBufferView> bufViews;

    for (const auto& [portName, gbuf] : node.ioMap.inputBuffers()) {
        CpuBufferView v = resolveBuffer(gbuf.name());
        v.elementType   = gbuf.type();
        bufViews[portName] = v;
    }

    const auto& inBufs = node.ioMap.inputBuffers();
    size_t defaultOutputSize = 0;
    if (!inBufs.empty()) {
        const std::string& firstName = inBufs.begin()->second.name();
        auto fit = buffers_.find(firstName);
        if (fit != buffers_.end()) {
            defaultOutputSize = fit->second.size();
        }
    }

    for (const auto& [portName, gbuf] : node.ioMap.outputBuffers()) {
        auto& storage = ensureBuffer(gbuf.name(), defaultOutputSize);
        bufViews[portName] = CpuBufferView{storage.data(), storage.size(), gbuf.type()};
    }

    for (const auto& rwb : node.ioMap.rwBuffers()) {
        bufViews[rwb.inPort] = resolveBuffer(rwb.in.name());
        auto& inStorage = buffers_.at(rwb.in.name());
        buffers_[rwb.out.name()] = inStorage;
        bufViews[rwb.outPort] = CpuBufferView{
            buffers_[rwb.out.name()].data(),
            buffers_[rwb.out.name()].size(),
            rwb.out.type()
        };
    }

    std::map<std::string, uint64_t> scalars;
    for (const auto& [portName, gs] : node.ioMap.scalars()) {
        if (gs.isConstant()) {
            scalars[portName] = gs.constantBits();
        } else {
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
}

CpuBufferView CpuDevice::resolveBuffer(const std::string& name) const {
    auto it = buffers_.find(name);
    if (it == buffers_.end()) {
        throw std::runtime_error(
            "CpuDevice: buffer '" + name + "' not found; "
            "did you forget to call setInputBuffer()?");
    }
    return CpuBufferView{
        const_cast<void*>(static_cast<const void*>(it->second.data())),
        it->second.size(),
        BufferType::U8
    };
}

std::vector<uint8_t>& CpuDevice::ensureBuffer(const std::string& name, size_t sizeBytes) {
    auto& buf = buffers_[name];
    if (buf.size() < sizeBytes) {
        buf.resize(sizeBytes);
    }
    return buf;
}

}  // namespace vrt::graph
