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
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>

#include <vrt/buffer.hpp>
#include <vrt/device.hpp>
#include <vrt/graph/device/device.hpp>
#include <vrt/graph/device/dgraph.hpp>
#include <vrt/graph/device/fpga/rp1_bar_window.hpp>
#include <vrt/graph/device/fpga/rp1_submitter.hpp>

namespace vrt::graph {

class Graph;
namespace fpga {
class FpgaVbinSpec;
}

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
 *
 * Buffer arguments use the RP1-visible DDR window as a staging arena.
 * Kernel arguments are packed as all scalar inputs first, followed by
 * 64-bit DDR addresses for `IOTypeMap::inputBuffers`,
 * `IOTypeMap::outputBuffers`, and then each RW buffer pair's input and
 * output addresses in declaration order.
 */
class FpgaDevice : public IDevice {
   public:
    FpgaDevice(std::string                       id,
               std::shared_ptr<fpga::Rp1BarWindow> window,
               FpgaKernelLocationLookup           lookup,
               std::uint32_t                      cq_size = fpga::kDefaultCqSize);

    FpgaDevice(std::string                       id,
               std::shared_ptr<fpga::Rp1BarWindow> window,
               std::shared_ptr<fpga::FpgaVbinSpec> vbinSpec,
               std::string                       initialImageId = "",
               std::uint32_t                     cq_size = fpga::kDefaultCqSize);

    ~FpgaDevice() override;

    FpgaDevice(const FpgaDevice&)            = delete;
    FpgaDevice& operator=(const FpgaDevice&) = delete;

    // ---- IDevice ----------------------------------------------------

    DeviceType  type() const override { return DeviceType::FPGA; }
    std::string id()   const override { return id_; }

    std::unique_ptr<IDevicePlan> compilePlan(const DGraph& dg) override;

    // ---- BAR-backed buffer accessors (also used by CPU↔FPGA bridges) --

    /**
     * @brief Supply or preallocate data for a root-scope FPGA buffer.
     *
     * If @p data is null and @p sizeBytes is non-zero, the buffer is
     * zero-filled.  @p bufferName may be either a plain root-scope name
     * or an already-scoped key (`scope:N:name`).
     */
    void setInputBuffer(const std::string& bufferName,
                        const void*        data,
                        std::size_t        sizeBytes);

    /**
     * @brief Read back a BAR-backed FPGA buffer into host memory.
     */
    void getOutputBuffer(const std::string& bufferName,
                         void*              data,
                         std::size_t        sizeBytes) const;

    /**
     * @brief Returns the current logical size of @p bufferName.
     */
    std::size_t bufferSize(const std::string& bufferName) const;

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

    /**
     * @brief Provide the VRT hardware device used to stage partial PDIs.
     *
     * Reprogram PDIs must be copied into DDR via QDMA, and RP1 receives the
     * resulting DDR physical address in its PDI_LOAD packet.
     */
    void setPdiStagingDevice(::vrt::Device device);

    // ---- Shared backplane (used by tests and FpgaDevicePlan) --------

    /// Shared submitter; FpgaDevicePlan calls into this.
    std::shared_ptr<fpga::Rp1Submitter> submitter() const noexcept { return submitter_; }

    /// Backing BAR window; exposed for diagnostics and tests.
    std::shared_ptr<fpga::Rp1BarWindow> window() const noexcept { return window_; }

   private:
    friend class FpgaDevicePlan;

    struct BufferRecord {
        std::uint32_t offset = 0;     ///< Window-relative byte offset.
        std::size_t   size = 0;       ///< Logical bytes currently valid.
        std::size_t   capacity = 0;   ///< Allocated bytes in the BAR arena.
        BufferType    type = BufferType::U8;
    };

    struct StagedPdiRecord {
        std::unique_ptr<::vrt::Buffer<std::uint8_t>> buffer;
        std::uint64_t                                physAddr = 0;
        std::size_t                                  size = 0;
    };

    static std::string normalizeBufferKey(const std::string& bufferName);

    BufferRecord ensureBuffer(const GraphBuffer& buffer, std::size_t sizeBytes);
    std::uint64_t bufferDeviceAddress(const GraphBuffer& buffer, std::size_t sizeBytes);
    FpgaKernelLocation resolveKernelLocation(const KernelDescriptor& kernel) const;
    std::uint64_t stagePdiBytes(const std::string& cacheKey,
                                const std::vector<std::uint8_t>& bytes);
    std::uint64_t stagePdiFile(const std::string& pdiPath);
    void setActiveImage(std::string imageId);
    std::string activeImageId() const;

    std::string                            id_;
    std::shared_ptr<fpga::Rp1BarWindow>    window_;
    FpgaKernelLocationLookup               lookup_;
    std::shared_ptr<fpga::FpgaVbinSpec>     vbinSpec_;
    std::shared_ptr<fpga::Rp1Submitter>    submitter_;
    std::uint32_t                          sentinelSlot_  = kDefaultSentinelSlot;
    std::uint32_t                          sentinelValue_ = kDefaultSentinelValue;
    std::chrono::milliseconds              waitTimeout_   = kDefaultFpgaWaitTimeout;

    mutable std::mutex                     bufferMutex_;
    std::map<std::string, BufferRecord>    buffers_;
    std::uint32_t                          nextBufferOffset_ = 0;
    mutable std::mutex                     pdiMutex_;
    std::shared_ptr<::vrt::Device>          pdiStagingDevice_;
    std::map<std::string, StagedPdiRecord>  stagedPdis_;
    mutable std::mutex                     imageMutex_;
    std::string                            activeImageId_;
};

}  // namespace vrt::graph

#endif  // VRT_GRAPH_DEVICE_FPGA_DEVICE_HPP
