# rp1_graph_vbin_full

End-to-end example for the VRT graph FPGA backend with:

- CPU kernels and FPGA kernels in the same `vrt::graph::Graph`;
- two exclusive user-region vbins loaded through `FpgaVbinSpec`;
- explicit `Graph::addReprogram(...)` nodes lowering to RP1 `PDI_LOAD`;
- a fixed-count graph loop that repeats CPU -> reprogram -> FPGA -> CPU -> reprogram -> FPGA -> CPU.

This example is intended as a runnable template for hardware bring-up. It builds
the host application without Vivado when `BUILD_KERNELS=OFF`, but executing the
graph requires a V80 with RP1 firmware and both hardware vbins.

## Graph

For each loop iteration:

```text
CPU stage
  -> PDI_LOAD imageA
  -> graph_kernel_0 from imageA: out[i] = in[i] + 1
  -> CPU mix
  -> PDI_LOAD imageB
  -> graph_kernel_0 from imageB: out[i] = in[i] * 2
  -> CPU finalize
```

The root graph also has `cpu_preprocess` before the loop and `cpu_report`
after the loop.

## Files

| Path | Purpose |
|------|---------|
| `hls_a/graph_kernel.cpp` | Image A FPGA kernel, increments every element. |
| `hls_b/graph_kernel.cpp` | Image B FPGA kernel, doubles every element. |
| `config_a.cfg`, `config_b.cfg` | One `graph_kernel_0` instance with two HBM-connected `m_axi` ports. |
| `rp1_graph_vbin_full.cpp` | Host graph authoring and execution. |
| `CMakeLists.txt` | Builds HLS/vbins when Vivado is available and always builds the host app. |

## Build

Against the repo tree:

```bash
cd examples/rp1_graph_vbin_full
cmake -B build -S . -DVRT_USE_REPO=ON
cmake --build build
```

Host-only build without Vivado:

```bash
cd examples/rp1_graph_vbin_full
cmake -B build -S . -DVRT_USE_REPO=ON -DBUILD_KERNELS=OFF
cmake --build build
```

With `BUILD_KERNELS=OFF`, pass vbins built elsewhere with `--vbin-a` and
`--vbin-b` when running.

## Run

Prerequisites:

- base platform loaded on the V80;
- RP1 firmware loaded and reporting `RP1_STATE_READY`;
- `vrtd` running and the current user authorized for the device;
- both vbins built for the same base platform and exclusive user-region layout.

Run from the build directory:

```bash
./rp1_graph_vbin_full --bdf 0000:65:00.0 --iterations 2 --elements 16
```

Or provide explicit vbin paths:

```bash
./rp1_graph_vbin_full \
  --bdf 0000:65:00.0 \
  --vbin-a /path/to/rp1_graph_vbin_full_a_hw.vbin \
  --vbin-b /path/to/rp1_graph_vbin_full_b_hw.vbin \
  --iterations 2 \
  --elements 16
```

Pass iff the final CPU buffer matches the host-computed reference. The program
prints the first few output values and returns non-zero on mismatch or runtime
errors.

## Notes

The FPGA graph ABI is scalar-first because `FpgaDevice` currently packs RP1
kernel arguments in `IOTypeMap` order: scalar inputs, then input buffer
addresses, then output buffer addresses. The HLS kernels therefore use:

```cpp
void graph_kernel(ap_uint<64> n, const int* in, int* out);
```

The example refines the vbin-derived descriptors to use `BufferType::I32` for
`in` and `out`, because current `system_map.xml` metadata identifies buffer
ports but does not carry element type.

Every loop iteration explicitly reprograms image A before the image A kernel
and image B before the image B kernel. That makes the active-image transitions
visible in graph authoring and avoids relying on out-of-band initial state.
