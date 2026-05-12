# 07_p2p_coyote: What To Run On The Vivado Machine (2 FPGAs)

This runbook assumes one machine with two FPGA boards:

- Board A: SLASH board (exports BAR as dma-buf)
- Board B: Coyote board (imports that dma-buf and drives AXI-Lite over P2P)

Board A is managed by the SLASH stack (typically visible in `v80-smi`).
Board B is managed by Coyote (`coyote_driver` and `/dev/coyote_fpga_*`) and
can be any Coyote-supported board (`u55c`, `u280`, `u250`, or `v80`). Board B
may NOT appear in `v80-smi`, even when healthy.

This guide is written for readers familiar with SLASH but new to Coyote.
Use this exact order.

## Coyote quick context (what Board B does)

In this example, Coyote is the runtime on Board B that imports an external
dma-buf exported by SLASH and then issues FPGA-side AXI-Lite reads/writes over
PCIe P2P to Board A.

`coyote_driver.ko` is the kernel module that makes this possible: it initializes
the Coyote PCI device, sets up DMA and interrupts, and exposes user-space
character devices such as `/dev/coyote_fpga_*`.

For this `07_p2p_coyote` flow, you only need the Coyote code already included
as a local submodule in this repository.

## 0) Additional Coyote install/bring-up prerequisites (one-time)

These are the extra steps typically required to load and use Coyote reliably on
an independent machine.

Install host dependencies (Ubuntu/Debian example):

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake ninja-build pkg-config git \
  libboost-dev libnuma-dev numactl pciutils \
  linux-headers-$(uname -r)
```

Notes:

- The Coyote driver must be compiled on the deployment machine against its
  running kernel.
- For this `07_p2p_coyote` example, no separate system-wide `make install`
  of Coyote software is required, because the host app builds Coyote from the
  local submodule.

Recommended by Coyote docs (not strictly required for this specific control
path demo): enable hugepages.

```bash
# Example: reserve 1024 x 2MB hugepages (adjust for your system)
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
grep -E 'HugePages_Total|HugePages_Free|Hugepagesize' /proc/meminfo
```

Before programming or reprogramming the Coyote board:

```bash
# If present from previous runs
sudo rmmod coyote_driver 2>/dev/null || true

# If the Coyote board is a V80 previously running AVED/SLASH
sudo rmmod ami 2>/dev/null || true
```

After programming the Coyote board (Step 5 below), verify PCI binding state:

```bash
lspci -s "$COYOTE_BDF" -nn
lspci -s "$COYOTE_BDF" -k
```

If another driver grabbed the Coyote board, unbind it first, then insert
`coyote_driver`:

```bash
if [[ -e /sys/bus/pci/devices/$COYOTE_BDF/driver/unbind ]]; then
  echo "$COYOTE_BDF" | sudo tee /sys/bus/pci/devices/$COYOTE_BDF/driver/unbind
fi
```

## 1) Prepare environment and choose Coyote board family

```bash
# Vivado + Vitis HLS environment
source <path-to-vivado>/settings64.sh
source <path-to-vitis-hls>/settings64.sh

# Optional sanity check
which vivado
v80-smi version
```

Identify both boards and choose roles.

Identify the SLASH board (Board A) via `v80-smi`:

```bash
v80-smi list -l
export SLASH_BDF=<bdf-for-slash-board>      # example: 0000:03:00.0
```

Identify the Coyote board (Board B) via PCI enumeration (not v80-smi):

```bash
lspci -D -nn | grep -Ei "xilinx|10ee|amd"
export COYOTE_BDF=<bdf-for-coyote-board>    # example: 0000:21:00.0
```

Choose Coyote board family variables using this mapping:

| Physical board | COYOTE_FDEV_NAME | COYOTE_PLATFORM | Programming image | Typical Vivado baseline |
| --- | --- | --- | --- | --- |
| Alveo U55C | `u55c` | `ultrascale_plus` | `cyt_top.bit` | 2022.1+ |
| Alveo U280 | `u280` | `ultrascale_plus` | `cyt_top.bit` | 2022.1+ |
| Alveo U250 | `u250` | `ultrascale_plus` | `cyt_top.bit` | 2022.1+ |
| Alveo V80 | `v80` | `versal` | `cyt_top.pdi` | 2024.2+ |

Then export variables:

```bash
# Pick the Coyote target family for this board
export COYOTE_FDEV_NAME=<u55c|u280|u250|v80>

# Coyote driver platform follows the board family
if [[ "$COYOTE_FDEV_NAME" == "v80" ]]; then
  export COYOTE_PLATFORM=versal
else
  export COYOTE_PLATFORM=ultrascale_plus
fi
```

Notes:

- UltraScale+ boards (`u55c`, `u280`, `u250`) generate `.bit` images.
- Versal boards (`v80`) generate `.pdi` images.
- If using Vivado versions older than the baseline above, expect possible
  compatibility fixes.

## 2) Build SLASH side artifacts + host app

```bash
cd /home/vserbu/SLASH/examples/07_p2p_coyote
cmake -S . -B build -G Ninja -DSLASH_USE_REPO=ON
cmake --build build --target hls p2p_coyote_hw 07_p2p_coyote
```

Resolve generated SLASH vrtbin path:

```bash
cd /home/vserbu/SLASH/examples/07_p2p_coyote
SLASH_VBIN=$(find build -type f \( -name 'p2p_coyote_hw*.vbin' -o -name 'p2p_coyote_hw*.vrtbin' \) | head -n1)
echo "SLASH_VBIN=$SLASH_VBIN"
```

## 3) Build Coyote hardware + Coyote driver

Build Coyote HW example:

```bash
cd /home/vserbu/SLASH/examples/07_p2p_coyote/coyote/hw
cmake -S . -B build -G Ninja -DFDEV_NAME="$COYOTE_FDEV_NAME"

# Coyote examples typically expose these targets
cmake --build build --target project bitgen
```

Resolve Coyote bitstream/PDI path:

```bash
cd /home/vserbu/SLASH/examples/07_p2p_coyote/coyote/hw
COYOTE_BIT=$(find build -type f \( -name 'cyt_top.pdi' -o -name 'cyt_top.bit' \) | head -n1)
echo "COYOTE_BIT=$COYOTE_BIT"

# Optional sanity check
test -n "$COYOTE_BIT"
```

Build Coyote driver:

```bash
cd /home/vserbu/SLASH/submodules/Coyote/driver
make TARGET_PLATFORM="$COYOTE_PLATFORM"
```

## 4) Program the SLASH board (Board A)

Ensure vrtd is up and program the SLASH board with the new vrtbin:

```bash
sudo systemctl enable --now vrtd.socket vrtd
sudo v80-smi program "$SLASH_VBIN" -d "$SLASH_BDF"
sudo v80-smi query -d "$SLASH_BDF"
```

## 5) Program the Coyote board (Board B)

You can program Board B using either Vivado Hardware Manager GUI or batch Tcl.
Both paths should use `$COYOTE_BIT` found in Step 3.

### Option A: Vivado Hardware Manager GUI (recommended for first bring-up)

1. Launch Vivado:

  ```bash
  vivado
  ```

2. In Vivado, open **Hardware Manager**.
3. Open target and connect to your hardware server (usually localhost).
4. Select the Coyote board device (Board B). If multiple boards are present,
  use your physical slot labels plus board family (`COYOTE_FDEV_NAME`) to
  avoid selecting the SLASH board.
5. Choose **Program Device** and set image file to `$COYOTE_BIT`.
6. Program and wait for completion (can take several minutes).
7. If Vivado prompts for optional probe files, leave them empty for this flow.

After GUI programming, verify PCI enumeration:

```bash
lspci -s "$COYOTE_BDF" -nn
lspci -s "$COYOTE_BDF" -k
```

### Option B: Batch Tcl flow (site-specific)

If your environment uses scripting, create a small local Tcl script like below
and run it in batch mode.

```tcl
# program_coyote.tcl
# Usage: vivado -mode batch -source program_coyote.tcl -tclargs <image> <device_index>

set image [lindex $argv 0]
set device_index [lindex $argv 1]

open_hw_manager
connect_hw_server
open_hw_target

set dev [lindex [get_hw_devices] $device_index]
current_hw_device $dev
refresh_hw_device $dev

set_property PROGRAM.FILE $image $dev
program_hw_devices $dev
refresh_hw_device $dev

close_hw_manager
exit
```

Run:

```bash
vivado -mode batch -source ./program_coyote.tcl -tclargs "$COYOTE_BIT" 0
```

Adjust `<device_index>` for your hardware ordering.

After batch programming, verify PCI enumeration:

```bash
lspci -s "$COYOTE_BDF" -nn
lspci -s "$COYOTE_BDF" -k
```

Notes:

- `v80` typically uses `.pdi`.
- `u55c/u280/u250` typically use `.bit`.
- Coyote board programming does not require `v80-smi`.

After programming, refresh PCIe if needed:

```bash
# Optional only if the new image does not enumerate cleanly
sudo sh -c "echo 1 > /sys/bus/pci/devices/$COYOTE_BDF/remove"
sudo sh -c "echo 1 > /sys/bus/pci/rescan"

# Re-check
lspci -s "$COYOTE_BDF" -nn
lspci -s "$COYOTE_BDF" -k
```

## 6) Load/reload Coyote driver, verify nodes, and pick device ID

`coyote_driver` will create one or more character devices that follow this
pattern:

- `/dev/coyote_fpga_<device>_v<vfid>`

For this example, use `vfid=0`.

```bash
sudo rmmod coyote_driver 2>/dev/null || true
sudo insmod /home/vserbu/SLASH/submodules/Coyote/driver/build/coyote_driver.ko

# Expect /dev/coyote_fpga_<device>_v0 nodes
ls -l /dev/coyote_fpga_*_v0

# Map Coyote numeric device ID to PCI bus/slot from kernel logs
sudo dmesg | grep -E "fpga device id|probe returning|coyote" | tail -n 40
```

Typical success indicators in logs:

- a line containing `fpga device id` (use this integer as `COYOTE_DEVICE`)
- a line containing `probe returning 0`

For this example, `ip_addr`/`mac_addr` module parameters are not required
(they are only needed for networking-focused Coyote setups).

Pick device ID from logs and export it:

```bash
export COYOTE_DEVICE=<device-id-from-dmesg>   # usually 0 with one Coyote board
export COYOTE_VFID=0
```

## 7) Verify SLASH control node and P2P readiness

```bash
ls -l /dev/slash_ctl*
```

Use the slash control node belonging to Board A. If unsure, inspect sysfs links:

```bash
for n in /dev/slash_ctl*; do
  echo "== $n =="
  readlink -f "/sys/class/misc/${n#/dev/}/device"
done
```

Set the correct node:

```bash
export SLASH_CTL=/dev/slash_ctl0
```

Confirm Board A exposes a P2P-capable BAR before running the app:

```bash
v80-smi query -d "$SLASH_BDF"
v80-smi query -d "$SLASH_BDF" | grep -Ei "bar|p2p"

# Choose a BAR index that reports p2p_capable=1 (often BAR 0)
export SLASH_BAR=0
```

## 8) Run the end-to-end P2P example

```bash
cd /home/vserbu/SLASH/examples/07_p2p_coyote
./build/07_p2p_coyote "$SLASH_CTL" "$COYOTE_DEVICE" "$COYOTE_VFID" 7 9 "$SLASH_BAR"
```

Expected success indicators:

- prints SLASH exporter metadata (BDF/BAR/len/fd)
- prints `Result: hw=16 expected=16`
- exits with code 0

## 9) Quick failure triage

```bash
# SLASH side
v80-smi list -l
v80-smi query -d "$SLASH_BDF"

# Coyote side
ls -l /dev/coyote_fpga_*_v0
lspci -s "$COYOTE_BDF" -nn
lspci -s "$COYOTE_BDF" -k
lspci -D -nn | grep -Ei "xilinx|10ee|amd"
sudo dmesg | tail -n 120
```

Focused checks by symptom:

- Symptom: Board B is missing or not stable in PCIe after programming.
  Action: rerun Step 5 verification (`lspci -s "$COYOTE_BDF" -nn` and `-k`),
  then run remove/rescan and recheck.

- Symptom: `insmod` succeeds but no `/dev/coyote_fpga_*_v0` appears.
  Action: rebuild the driver with the correct `TARGET_PLATFORM` for your board
  family, reload module, then inspect `dmesg` for probe errors.

- Symptom: app runs but times out or returns wrong result.
  Action: confirm `COYOTE_DEVICE` was taken from `fpga device id` in `dmesg`,
  confirm `COYOTE_VFID=0`, and confirm selected `SLASH_BAR` is P2P-capable.

- Symptom: behavior suggests board roles are swapped.
  Action: verify Board A is the one programmed with `v80-smi` and Board B is
  the one programmed from Vivado Hardware Manager.

Reference docs inside this repo:

- `submodules/Coyote/README.md`
- `submodules/Coyote/driver/README.md`
