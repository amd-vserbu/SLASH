# rp1_bringup

Temporary scaffolding for the first on-hardware tests of the RP1 graph
processor. Will be deleted once libslash / VRT expose a real
`GraphBuilder` API.

The tool submits tiny graphs to the RP1 firmware over BAR4 (the same
BAR the existing `07_rp1_memcheck` uses) and polls `graph_done_seq` for
completion.

## Prerequisites

1. **Bitstream.** A PDI that maps the user-region AXI-Lite aperture
   (`0x0202_0000_0000 + 128 MB` from the host side) into R5 space at
   `0x8800_0000 + 128 MB` via the RPU → NoC path. The hardware/linker
   team's drop satisfies this.
2. **Firmware.** Build `rp1.elf` with `RP1_POLLING_BRINGUP=ON`. The GCQ
   doorbell (`irq_sq`) is not yet wired, so until it is, RP1 must
   spin-poll `graph_seq` instead of `wfi`-ing:
   ```bash
   cd linker/resources/aved/rp1
   cmake -S . -B build-bringup -DRP1_POLLING_BRINGUP=ON
   cmake --build build-bringup
   # produces build-bringup/rp1.elf
   ```
3. **Load `rp1.elf` onto R5-1 via xsdb** (same pattern as
   `rp1_memtest.elf`). After load, the firmware should sit in
   `RP1_STATE_READY` with `heartbeat` advancing.

## Build

```bash
cd examples/rp1_bringup
cmake -B build -S . -DSLASH_USE_REPO=ON
cmake --build build
```

## Run

Three subcommands, all targeting BAR4. Run them in order.

### 1. `dump` — sanity-check firmware liveness

```bash
./build/rp1_bringup dump /dev/slash_ctl0
```

Reads the RP1 control block, prints all fields, samples `heartbeat`
twice 500 ms apart. Expect:
- `magic == 0x53515231` ("SQR1")
- `rp1_state == 1 (READY)`
- heartbeat advancing

If `magic` is `0` or `0xFFFFFFFF`, the firmware never wrote the control
block. If `heartbeat` is stuck, RP1 is hung (often the `wfi` problem —
double-check `RP1_POLLING_BRINGUP` was set).

### 2. `signal` — Stage 0: no kernels

```bash
./build/rp1_bringup signal /dev/slash_ctl0
```

Submits a one-node `SIGNAL` graph that writes `0xDEADBEEF` into signal
slot 0. Pass iff slot 0 reads back `0xDEADBEEF`. This validates:

- the firmware boots end-to-end on real silicon
- BAR4 → RP1 shared DDR mapping is correct
- the flat scanner can process at least one immediate-completion node
- `graph_done_seq` advances and is visible to the host

Stage 0 deliberately does **not** exercise the new AXI-Lite path. If
this fails, the issue is in shared-DDR plumbing or the firmware itself,
not the new aperture.

### 3. `kernel` — Stage 1: one KERNEL_DISPATCH

```bash
./build/rp1_bringup kernel /dev/slash_ctl0 <r5_addr_hex> [arg0_hex ...]
```

Submits  `SIGNAL → KERNEL_DISPATCH → SIGNAL`. Pass iff both signal
slots read back their expected markers (`0xBEEFBEEF` pre, `0xCAFEBABE`
post). This validates:

- everything Stage 0 validated, plus
- RP1 can write to a kernel's AXI-Lite registers through the new
  LPD → NoC → user-region path
- the kernel asserts `ap_done`
- `check_inflight()` finalises the dispatched node and unblocks
  downstream

`<r5_addr_hex>` is the kernel's address **in R5 space**, computed from
the host-view `<BaseAddress>` in `system_map.xml`:

```
r5_addr = xml_addr - 0x0202_0000_0000 + 0x8800_0000
```

For example, if `system_map.xml` lists the kernel at `0x0202_0001_0000`,
the R5 address is `0x8801_0000`.

Optional `[arg0_hex arg1_hex ...]` are written to the kernel's AXI-Lite
arg registers at offsets `+0x10, +0x14, +0x18, ...`. For
`examples/00_axilite/increment(size, in)` you typically want `size=0`
(so the loop is a no-op and `ap_done` fires immediately) plus the two
words of an unused `in` pointer, e.g.:

```bash
./build/rp1_bringup kernel /dev/slash_ctl0 0x88010000 0 0 0
```

If you'd rather pre-stage the args yourself over the user-region BAR
before invoking this tool, leave the args off — `arg_count` will be 0
and the firmware will only emit `ap_start`.

## Diagnostics

On failure (or timeout) the tool prints the full control block. Useful
fields:

- `rp1_state` — `RUNNING` means stuck mid-graph; `ERROR` means the
  firmware bailed; `READY` means the graph is "done" but our marker
  signal didn't fire.
- `rp1_error_code` — `1` = inflight list full, `2` = kernel timeout.
- `rp1_current_node` — the last node the scanner activated.
- `cq_write_idx` — number of CQ entries written.

If `cq_write_idx` increased but the trailing SIGNAL didn't fire,
suspect the new AXI-Lite path: the kernel was dispatched but its
`ap_done` never came back through `check_inflight`.
