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
 * @file fpga_device.hpp
 * @brief FpgaDevice — IDevice backend that lowers DGraphs to RP1 graphs.
 *
 * Phase-1 scope (diamond parity):
 *   - Only `CompiledKernelNode` entries are honoured. Any other compiled
 *     node variant (`CompiledBridgeOpNode`, `CompiledBoundaryNode`,
 *     `CompiledLoopNode`, `CompiledConditionalNode`) causes
 *     `compilePlan()` to throw with a descriptive diagnostic.
 *   - Each kernel becomes one `RP1_OP_KERNEL_DISPATCH` packet; barriers
 *     are allocated in bucket 0 (up to 31 kernels). Bit 31 is reserved
 *     for the trailing sentinel `RP1_OP_SIGNAL` that writes
 *     `kDefaultSentinelValue` into `kDefaultSentinelSlot` once every
 *     leaf kernel completes.
 *   - Kernel arguments are taken from `IOMap` scalar bindings, packed in
 *     the order declared by the kernel's `IOTypeMap::inputScalars`.
 *     Constants are baked in at compile time; global-variable bindings
 *     are resolved at `launch()` time via the per-graph scalar map.
 *   - Input/output/RW buffer bindings are tolerated (their presence
 *     does not break compilation) but the actual data movement must be
 *     arranged by the user outside the graph for phase 1; the
 *     `CpuFpgaBridge` data path is still stubbed.  This matches the
 *     "buffers pre-staged" intent.
 *
 * Future phases will extend `compilePlan` to also emit DMA_COPY,
 * LOOP/COND/RERUN packets, and proper cross-device bridge handling.
 */

#ifndef VRT_GRAPH_DEVICE_FPGA_DEVICE_HPP
#define VRT_GRAPH_DEVICE_FPGA_DEVICE_HPP

#include <chrono>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/fpga/rp1_bar_window.hpp>
#include <vrt/graph/device/fpga/rp1_submitter.hpp>

namespace vrt::graph {

class Graph;

/**
 * @brief Resolves a kernel's logical name to its AXI-Lite base address
 *        in R5 address space.
 *
 * The R5 address is what the firmware writes to as `+0x10, +0x14, ...`
 * for arguments and `+0x00` for `ap_start`.  Convert from the host-view
 * address in `system_map.xml` via:
 *
 *     r5_addr = xml_addr - 0x0202'0000'0000 + 0x8800'0000
 */
struct FpgaKernelLocation {
    std::uint32_t r5_base_addr = 0;

    /// Optional default `timeout_cycles` for KERNEL_DISPATCH; 0 = firmware default.
    std::uint32_t timeout_cycles = 0;
};

using FpgaKernelLocationLookup =
    std::function<FpgaKernelLocation(const std::string& kernel_name)>;

/**
 * @brief Sentinel slot/value used by the trailing SIGNAL node in every
 *        compiled FPGA plan.  Matches the convention used by
 *        `examples/rp1_bringup/cmd_diamond`.
 */
constexpr std::uint32_t kDefaultSentinelSlot  = RP1_MAX_SIGNALS - 1u;
constexpr std::uint32_t kDefaultSentinelValue = 0xD1A1D0DDu;

/**
 * @brief Default timeout for `FpgaDevicePlan::wait()`.
 */
constexpr std::chrono::milliseconds kDefaultFpgaWaitTimeout{3000};

class FpgaDevicePlan;

/**
 * @brief IDevice backend that targets the RP1 command processor.
 *
 * Construction is light-weight; nothing is sent over the BAR until the
 * first compiled plan calls `launch()`.  Multiple plans built by the
 * same `FpgaDevice` share a single `Rp1Submitter`, so kernel
 * submissions are serialised across the device by construction.
 */
class FpgaDevice : public IDevice {
   public:
    FpgaDevice(std::string                       id,
               std::shared_ptr<fpga::Rp1BarWindow> window,
               FpgaKernelLocationLookup           lookup,
               std::uint32_t                      cq_size = fpga::kDefaultCqSize);

    ~FpgaDevice() override;

    FpgaDevice(const FpgaDevice&)            = delete;
    FpgaDevice& operator=(const FpgaDevice&) = delete;

    // ---- IDevice ----------------------------------------------------

    DeviceType  type() const override { return DeviceType::FPGA; }
    std::string id()   const override { return id_; }

    std::unique_ptr<IDevicePlan> compilePlan(const DGraph& dg) override;

    // ---- FpgaDevice-specific configuration --------------------------

    /// Index of the signal slot the auto-generated sentinel SIGNAL node
    /// writes to. Defaults to @c kDefaultSentinelSlot.
    void          setSentinelSlot(std::uint32_t slot);
    std::uint32_t sentinelSlot() const noexcept { return sentinelSlot_; }

    /// Value the sentinel writes; defaults to @c kDefaultSentinelValue.
    /// Tests use this to confirm graph completion.
    void          setSentinelValue(std::uint32_t value);
    std::uint32_t sentinelValue() const noexcept { return sentinelValue_; }

    /// Default per-plan wait timeout used by @c FpgaDevicePlan::wait().
    void                       setWaitTimeout(std::chrono::milliseconds t);
    std::chrono::milliseconds  waitTimeout() const noexcept { return waitTimeout_; }

    // ---- Shared backplane (used by tests and FpgaDevicePlan) --------

    /// Shared submitter; FpgaDevicePlan calls into this.
    std::shared_ptr<fpga::Rp1Submitter> submitter() const noexcept { return submitter_; }

    /// Backing BAR window; exposed for diagnostics and tests.
    std::shared_ptr<fpga::Rp1BarWindow> window() const noexcept { return window_; }

   private:
    friend class FpgaDevicePlan;

    std::string                            id_;
    std::shared_ptr<fpga::Rp1BarWindow>    window_;
    FpgaKernelLocationLookup               lookup_;
    std::shared_ptr<fpga::Rp1Submitter>    submitter_;
    std::uint32_t                          sentinelSlot_  = kDefaultSentinelSlot;
    std::uint32_t                          sentinelValue_ = kDefaultSentinelValue;
    std::chrono::milliseconds              waitTimeout_   = kDefaultFpgaWaitTimeout;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_FPGA_DEVICE_HPP
