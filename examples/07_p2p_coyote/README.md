# 07_p2p_coyote

This example demonstrates BAR-level P2P control-plane interop:

- The SLASH side builds a simple AXI-Lite adder kernel (`slash_add`) with control
  registers exposed in BAR0.
- The Coyote side imports the exported SLASH BAR dma-buf FD and issues FPGA-side
  `LOCAL_WRITE/LOCAL_READ` requests to:
  - write operands,
  - start the SLASH kernel,
  - poll `ap_done`,
  - read the result.

The userspace app (`07_p2p_coyote`) orchestrates both runtimes.

## Register protocol

SLASH AXI-Lite offsets used by Coyote hardware:

- `0x00`: control (`ap_start` bit 0, `ap_done` bit 1)
- `0x10`: operand A (`uint32_t`)
- `0x18`: operand B (`uint32_t`)
- `0x20`: result (`uint32_t`)

Coyote AXI-Lite control registers used by host app:

- `0`: command (`bit0=start`, pulse)
- `1`: status (`bit0=done bit1=busy bit2=error bit3=timeout`)
- `2`: mapped external dmabuf token/base
- `3`: operand A
- `4`: operand B
- `5`: cThread ID
- `6`: result mirror
- `7`: poll count
- `8`: last observed SLASH control value
- `9`: error code

## Build SLASH example app + HLS/vbin

From `examples/07_p2p_coyote`:

```bash
cmake -S . -B build -G Ninja -DSLASH_USE_REPO=ON
cmake --build build --target hls p2p_coyote_hw 07_p2p_coyote
```

If you only want the host binary (no Vivado/Vitis in PATH):

```bash
cmake -S . -B build-host -G Ninja -DSLASH_USE_REPO=ON -DSLASH_ENABLE_FPGA_TARGETS=OFF
cmake --build build-host --target 07_p2p_coyote
```

## Build Coyote hardware bitstream

From `examples/07_p2p_coyote/coyote/hw`:

```bash
cmake -S . -B build -G Ninja -DFDEV_NAME=<u55c|u280|u250|v80>
cmake --build build
```

Program this bitstream through your normal Coyote deployment flow for the target
platform/region.

## Run

Prerequisites:

- SLASH driver and device present (`/dev/slash_ctlX`)
- Coyote runtime/driver active and vFPGA programmed with this example hardware
- The SLASH BAR used must report `p2p_capable=1`

Run:

```bash
./build-host/07_p2p_coyote /dev/slash_ctl0 0 0 7 9
```

Arguments:

```text
07_p2p_coyote <slash_ctl_path> <coyote_device> <coyote_vfid> <a> <b> [slash_bar]
```

Expected output contains:

- exported SLASH BAR metadata,
- hardware result and expected result,
- Coyote status/poll counters.

If `hw != expected`, the app exits with failure.

## Two-board deployment runbook

For a copy-paste command sequence on a Vivado machine with two FPGA boards,
see `RUN_ON_VIVADO_2FPGA.md` in this directory.

That runbook now also includes additional one-time Coyote host setup steps
(dependencies, kernel headers, hugepages guidance, and driver binding checks).
