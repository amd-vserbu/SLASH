# Design Document: PCI P2P Access to SLASH BAR via DMABUF from Coyote

**Date:** 2026-04-03
**Status:** DRAFT v3
**Authors:** (generated with Claude Code)

---

## 1. Executive Summary

This document describes the changes required to enable a Coyote FPGA design to
perform PCI peer-to-peer (P2P) read/write access to the SLASH user BAR
(PF2 / BAR0) via the Linux DMABUF framework. The goal is to allow a Coyote
vFPGA to directly read and write registers (or data) on a co-located SLASH
(AMD Alveo V80) device's BAR0, without host memory as an intermediary.

### Key Insight

**Coyote already has a fully working DMABUF P2P import path** (built for GPU
memory via `IOCTL_MAP_DMABUF` / `p2p_attach_dma_buf()`). It attaches to any
DMABUF exporter, maps the sg_table, extracts DMA bus addresses, and programs
them into the FPGA's hardware TLB. **No changes are needed on the Coyote side
for a working prototype.**

**The work is almost entirely on the SLASH side.** SLASH must register BAR0 as
P2PDMA memory using the kernel's `pci_p2pdma` subsystem, implement proper
`.attach` / `.map_dma_buf` / `.unmap_dma_buf` callbacks, and enforce DMABUF
invalidation rules during remove. For Phase 1, this design also enforces a
simple access policy: BAR0 is used either by userspace mmap or by a P2P
importer at one time (not both concurrently).

### Scope

**Phase 1 (this document):** Coyote user-space application obtains an fd for
the SLASH BAR0, passes it to the Coyote driver via the existing
`IOCTL_MAP_DMABUF` path, and the Coyote DMA engine issues PCIe reads/writes
directly to the SLASH BAR's physical address range.

---

## 2. Current State Analysis

### 2.1 SLASH Driver — BAR Export via DMABUF

**Files:** `SLASH/driver/slash_dmabuf.c`, `slash_ctldev.c`

The SLASH driver already exports PCI BARs as dma-buf objects:

- `slash_bar_dmabuf_create()` exports any active MMIO BAR as a `struct dma_buf`.
- Userspace obtains an fd via `SLASH_CTLDEV_IOCTL_GET_BAR_FD` on `/dev/slash_ctlN`.
- The dma-buf supports **userspace mmap** (fault-based `vmf_insert_pfn()`).
- **The BAR is NOT registered as P2PDMA memory.** No calls to
  `pci_p2pdma_add_resource()` exist anywhere in the SLASH codebase.

**Critical limitation:** The SLASH dma-buf **intentionally rejects** kernel-side
device attachments:

```c
// slash_dmabuf.c:69-73
static int slash_bar_dmabuf_attach(struct dma_buf *dmabuf,
                                   struct dma_buf_attachment *attach)
{
    dev_warn(attach->dev, "device attachments are not supported for BAR dmabuf");
    return -EOPNOTSUPP;
}
```

And `map_dma_buf` also returns `-EOPNOTSUPP`:
```c
// slash_dmabuf.c:80-85
static struct sg_table *slash_bar_dmabuf_map(struct dma_buf_attachment *attach,
                                             enum dma_data_direction dir)
{
    return ERR_PTR(-EOPNOTSUPP);
}
```

**Reason given:** "a PCI BAR is I/O memory, not DMA-able system RAM."

This is true for _normal_ DMA (you can't DMA _from_ a BAR into system memory
using the BAR as a source buffer). But for P2P, a peer device _can_ issue PCIe
memory read/write TLPs targeting a BAR — which is exactly what we want. The
kernel's P2PDMA subsystem exists precisely to handle this case: it gives PCI BAR
regions `struct page` backing (of type `MEMORY_DEVICE_PCI_P2PDMA`) so they can
participate in the standard `dma_map_sg()` / sg_table path.

### 2.2 Coyote Driver — P2P DMABUF Import (Already Works)

**Files:** `Coyote/driver/src/vfpga/vfpga_gup.c`, `vfpga_ops.c`

Coyote has a complete, working P2P DMABUF import path (designed for AMD GPU
memory but generic enough for any DMABUF exporter):

1. User calls `IOCTL_MAP_DMABUF` with `{buf_fd, vaddr, ctid, mem_block}`.
2. Driver calls `p2p_attach_dma_buf()` which:
   - `dma_buf_get(buf_fd)` — acquires the dma-buf from the fd.
   - `dma_buf_dynamic_attach(buf, dev, &gpu_importer_ops, priv)` — dynamic
     attach with `.allow_peer2peer = true` and `.move_notify` callback.
   - `dma_buf_map_attachment()` — calls the exporter's `.map_dma_buf` to get
     an `sg_table`.
   - Walks the sg_table extracting `sg_dma_address()` into `hpages[]`.
   - Programs `hpages[]` into the hardware TLB via `tlb_map_gup()`.
3. Card memory allocation (`cpages[]`) is **conditional on `en_mem`** and is
   only used for host↔card migration. The initial TLB mapping always points to
   the DMA addresses from the sg_table (`host = HOST_ACCESS`), not card memory.
4. The `move_notify` callback is also used for exporter-driven invalidation
   during remove; BAR mappings are generally stable in steady state, but
   remove-time revoke still depends on this callback path.

**This entire path works as-is for a SLASH BAR DMABUF**, provided that:
- SLASH's `.attach` accepts the Coyote device attachment.
- SLASH's `.map_dma_buf` returns a valid sg_table with DMA-mapped addresses
  that Coyote's DMA engine can target.

### 2.3 PCI BAR Topology

| Board  | PF  | Device ID | BAR(s) of Interest    | Purpose               |
|--------|-----|-----------|-----------------------|------------------------|
| SLASH  | PF2 | 0x50B6    | BAR0 (MMIO, user)     | User-facing registers  |
| Coyote | PF0 | varies    | BAR0/2/4 (3 x 64-bit) | Config, DMA, Shell     |

Both devices are PCIe endpoints on the same host. P2P transactions traverse
the PCIe switch/root complex without touching DRAM (when the topology allows).

---

## 3. The Problem

SLASH exports BAR0 as a dma-buf today, but only for userspace mmap. Two
callbacks block the kernel P2P import path:

| DMABUF Callback        | Current SLASH Behavior | Required for P2P      |
|------------------------|------------------------|-----------------------|
| `.attach`              | Returns `-EOPNOTSUPP`  | Must accept attachment |
| `.map_dma_buf`         | Returns `-EOPNOTSUPP`  | Must return `sg_table` |

The solution is to register the BAR as P2PDMA memory using the kernel's
`pci_p2pdma` subsystem, which gives the BAR `struct page` backing and enables
the standard `dma_map_sgtable()` path (including automatic IOMMU handling).

---

## 4. Proposed Design

### 4.1 Overview

```
 +-----------+     DMABUF fd      +------------+
 | SLASH     | ---- export -----> | Coyote     |
 | PF2/BAR0  |                    | vFPGA      |
 | (exporter)|  <-- PCIe P2P -->  | (importer) |
 +-----------+   read/write TLPs  +------------+
       |                                |
       |  ioctl GET_BAR_FD              |  ioctl MAP_DMABUF
       v                                v
   /dev/slash_ctl0              /dev/fpga_0
       |                                |
   [User-space application orchestrates both]
```

**Data Flow:**
1. User app opens `/dev/slash_ctl0`, calls `GET_BAR_FD` for BAR0 -> gets `bar_fd`.
2. User app opens `/dev/fpga_0` (Coyote vFPGA), calls `MAP_DMABUF` with `bar_fd`.
3. Coyote driver attaches to the SLASH dma-buf, gets the sg_table.
4. Coyote programs its TLB with the BAR's DMA address (IOVA or bus address).
5. Coyote FPGA logic issues PCIe reads/writes to those addresses — these become
   P2P memory transactions hitting the SLASH BAR directly.

### 4.2 Changes Required — SLASH Driver (This Is Where the Work Lives)

#### 4.2.1 Register BAR as P2PDMA Memory (`slash_pcie.c` or `slash_ctldev.c`)

The kernel's P2PDMA subsystem (`CONFIG_PCI_P2PDMA`) is designed exactly for
this use case. Calling `pci_p2pdma_add_resource()` on a BAR region:

- Creates `struct page` objects of type `MEMORY_DEVICE_PCI_P2PDMA` via
  `devm_memremap_pages()`.
- These pages can then be placed in an sg_table and passed to
  `dma_map_sgtable()`, which recognizes them as P2P pages and handles IOMMU
  mapping automatically.
- The kernel also exposes `pci_p2pdma_distance_many()` for topology checks
  (same switch? same root complex? IOMMU OK?).

During probe, after the BAR is validated as active MMIO:

```c
// In slash_ctldev_create() or slash_pcie_probe(), for each P2P-eligible BAR:

ret = pci_p2pdma_add_resource(pdev, bar_number, bar_len, 0);
if (ret) {
    dev_warn(&pdev->dev,
             "slash: failed to register BAR%d as P2PDMA: %d\n",
             bar_number, ret);
    // Non-fatal: BAR still works for userspace mmap, just not P2P
}
```

This call enables `struct page` backing for the BAR region and is a required
prerequisite, but it is not sufficient by itself: `.attach`/`.map_dma_buf`
and remove-time invalidation are still required for a correct DMABUF P2P path.

#### 4.2.2 Enable Device Attachment (`slash_dmabuf.c`)

Replace the rejecting `.attach` with P2PDMA topology validation:

```c
static int slash_bar_dmabuf_attach(struct dma_buf *dmabuf,
                                   struct dma_buf_attachment *attach)
{
    struct slash_bar_dmabuf_data *priv = dmabuf->priv;
      struct device *clients[1] = { attach->dev };

      /* Phase 1 policy: no P2P attach while userspace mmap is active. */
      if (atomic_read(&priv->cpu_mmap_count) > 0)
            return -EBUSY;

    /*
       * pci_p2pdma_distance_many() expects struct device * clients and
       * resolves PCI parents internally. It returns < 0 when topology,
       * ACS policy, or host bridge policy blocks P2P.
     */
      if (pci_p2pdma_distance_many(priv->pdev, clients,
                                                 ARRAY_SIZE(clients), true) < 0) {
        dev_warn(attach->dev,
                 "slash: P2P not supported for this device topology\n");
        return -EOPNOTSUPP;
    }

    dev_info(attach->dev,
             "slash: P2P attachment accepted for BAR%d\n",
             priv->bar_number);
    return 0;
}
```

#### 4.2.3 Implement `map_dma_buf` Using P2PDMA Pages (`slash_dmabuf.c`)

With the BAR registered as P2PDMA memory, build an sg_table from the BAR's
registered P2PDMA pages (not a transient `pci_alloc_p2pmem()` allocation),
then DMA-map it through the standard kernel path:

```c
static struct sg_table *slash_bar_dmabuf_map(struct dma_buf_attachment *attach,
                                             enum dma_data_direction dir)
{
    struct slash_bar_dmabuf_data *priv = attach->dmabuf->priv;
    struct sg_table *sgt;
   struct page **pages;
   unsigned long first_pfn;
   unsigned long npages;
   unsigned long i;
    int ret;

    sgt = kzalloc(sizeof(*sgt), GFP_KERNEL);
   if (!sgt)
        return ERR_PTR(-ENOMEM);

   npages = DIV_ROUND_UP(priv->len, PAGE_SIZE);
   pages = kvmalloc_array(npages, sizeof(*pages), GFP_KERNEL);
   if (!pages) {
      kfree(sgt);
      return ERR_PTR(-ENOMEM);
   }

   first_pfn = pci_resource_start(priv->pdev, priv->bar_number) >> PAGE_SHIFT;
   for (i = 0; i < npages; i++)
      pages[i] = pfn_to_page(first_pfn + i);

   ret = sg_alloc_table_from_pages(sgt, pages, npages, 0, priv->len,
                           GFP_KERNEL);
   kvfree(pages);
    if (ret) {
        kfree(sgt);
        return ERR_PTR(ret);
    }

    /*
     * dma_map_sgtable() on MEMORY_DEVICE_PCI_P2PDMA pages:
     * - With IOMMU active: creates an IOVA mapping in the attacher's domain
     * - With IOMMU passthrough: uses the PCI bus address directly
     * - Handles all the complexity we'd otherwise do manually
     */
    ret = dma_map_sgtable(attach->dev, sgt, dir, DMA_ATTR_SKIP_CPU_SYNC);
    if (ret) {
        sg_free_table(sgt);
        kfree(sgt);
        return ERR_PTR(ret);
    }

   atomic_inc(&priv->p2p_map_count);

    return sgt;
}
```

**Why this is the correct approach:**

- `pci_p2pdma_add_resource()` gives BAR pages type
   `MEMORY_DEVICE_PCI_P2PDMA`, so they can be represented in an sg_table.
- `map_dma_buf` maps the BAR's real page range, preserving fixed register
   offsets and avoiding per-map pool allocations.
- `dma_map_sgtable()` recognizes P2PDMA pages and handles them correctly:
  - With IOMMU: creates an IOVA in the _attaching_ device's domain.
  - Without IOMMU / passthrough: returns the PCI bus address directly.
- This is the same mechanism the kernel uses for NVMe CMB (Controller Memory
  Buffer) P2P — a well-tested production path.
- The previous draft's approach of manually setting `sg_dma_address` to
  `pci_resource_start()` would bypass IOMMU handling and break on systems with
  active IOMMUs (common in server environments).

#### 4.2.4 Implement `unmap_dma_buf` (`slash_dmabuf.c`)

```c
static void slash_bar_dmabuf_unmap(struct dma_buf_attachment *attach,
                                   struct sg_table *sgt,
                                   enum dma_data_direction dir)
{
   struct slash_bar_dmabuf_data *priv = attach->dmabuf->priv;

    dma_unmap_sgtable(attach->dev, sgt, dir, DMA_ATTR_SKIP_CPU_SYNC);
    sg_free_table(sgt);
    kfree(sgt);

   atomic_dec(&priv->p2p_map_count);
}
```

#### 4.2.5 Userspace and P2P Coexistence Policy

Yes: a BAR exported as dma-buf can be used by userspace (`mmap`) and by a
kernel importer (`attach`/`map_dma_buf`) because those are independent dma-buf
operations. However, allowing both at the same time requires strict ordering
and ownership rules.

For Phase 1, enforce a **serialized access policy**:

- Track active CPU mmaps (`cpu_mmap_count`) and active P2P maps
  (`p2p_map_count`) in `slash_bar_dmabuf_data`.
- In `.mmap`, return `-EBUSY` if `p2p_map_count > 0`.
- In `.attach` (or `.map_dma_buf`), return `-EBUSY` if `cpu_mmap_count > 0`.
- Keep the userspace-only and P2P-only modes both supported, just not
  concurrently.

Phase 2 can relax this if needed, with explicit fencing rules between CPU MMIO
and peer DMA traffic.

#### 4.2.6 Kernel Config Dependencies

Required kernel config:
- `CONFIG_PCI_P2PDMA=y` — P2PDMA infrastructure
- `CONFIG_DMABUF_MOVE_NOTIFY=y` — for Coyote's dynamic attach (already
  required by Coyote's GPU P2P)
- `CONFIG_ZONE_DEVICE=y` — required by P2PDMA for `struct page` backing
  (typically auto-selected)

Build flag for SLASH:
```makefile
ccflags-y += -DSLASH_ENABLE_P2P
```

Gate P2PDMA registration with `#ifdef SLASH_ENABLE_P2P` / `#ifdef CONFIG_PCI_P2PDMA`
to keep the non-P2P build working.

### 4.3 Changes Required — Coyote Driver (Minimal / None)

**Coyote's existing `p2p_attach_dma_buf()` path should work as-is.** Here's why:

| What Coyote does                      | Works with SLASH BAR? | Why                              |
|----------------------------------------|----------------------|----------------------------------|
| `dma_buf_get(fd)`                      | Yes                  | Standard fd -> dmabuf lookup     |
| `dma_buf_dynamic_attach()` with `allow_peer2peer=true` | Yes | SLASH attach now accepts peers |
| `dma_buf_map_attachment()`             | Yes                  | SLASH map now returns sg_table   |
| Walk sg_table for `sg_dma_address()`   | Yes                  | P2PDMA pages produce valid addrs |
| `tlb_map_gup()` with those addresses   | Yes                  | TLB doesn't care about addr type |
| Card memory allocation (`en_mem`)      | Harmless             | Allocates but never used if no offload |
| `move_notify` callback                 | Required             | Needed for exporter-driven remove invalidation |

**Optional improvements for production (not needed for Phase 1):**

1. **Skip card memory allocation for P2P BAR dmabufs.** Detection: check if
   `sg_page(sgt->sgl)` is a `MEMORY_DEVICE_PCI_P2PDMA` page via
   `is_pci_p2pdma_page()`. This saves HBM/DDR allocation on the Coyote side.

2. **Add a `CoyoteAllocType::P2P_BAR`** to the user-space API to semantically
   distinguish GPU vs BAR P2P allocations. For Phase 1, reuse `GPU` and pass
   the SLASH `bar_fd` as the dmabuf fd.

### 4.4 Changes Required — User Space

#### 4.4.1 Orchestration Application

A user-space application coordinates both devices:

```cpp
#include <fcntl.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <slash/uapi/slash_interface.h>
#include <coyote/cThread.hpp>

int main() {
   // 1. Open SLASH control device and request BAR0 dma-buf fd
   int ctl_fd = open("/dev/slash_ctl0", O_RDWR | O_CLOEXEC);
   if (ctl_fd < 0)
     return 1;

   struct slash_ioctl_bar_fd_request req = {
     .size = sizeof(req),
     .bar_number = 0,
     .flags = O_CLOEXEC,
   };

   int bar_fd = ioctl(ctl_fd, SLASH_CTLDEV_IOCTL_GET_BAR_FD, &req);
   if (bar_fd < 0) {
     close(ctl_fd);
     return 1;
   }

   uint64_t bar_size = req.length;

   // 2. Open Coyote vFPGA
    coyote::cThread coyote_thread(/*vfpga_id=*/0, getpid());

   // 3. Map the SLASH BAR into Coyote's address space via IOCTL_MAP_DMABUF
   //    Reuse CoyoteAllocType::GPU which passes the fd through the DMABUF path
    void *mapped = coyote_thread.getMem({
        coyote::CoyoteAllocType::GPU,
        bar_size,
        false,        // not remote
        0,            // device id
        bar_fd        // dmabuf fd from SLASH
    });

   // 4. Coyote FPGA can now read/write SLASH BAR0 via P2P PCIe transactions
    coyote::localSg sg = { .addr = mapped, .len = 4 };
    coyote_thread.invoke(coyote::CoyoteOper::LOCAL_READ, sg);

   // 5. Cleanup
    coyote_thread.freeMem(mapped);
    close(bar_fd);
   close(ctl_fd);
}
```

If a libslash helper is preferred, add a small `slash_bar_fd_open()` API that
returns the BAR dma-buf fd without implicitly creating a userspace mmap.

#### 4.4.2 Coyote FPGA User Logic

For Phase 1, a simple test design based on `examples/01_hello_world/` that:
1. Reads a 32-bit value from offset 0 of the mapped SLASH BAR.
2. Writes a value to a writable offset.
3. Reads it back and verifies correctness.

The FPGA user logic issues AXI read/write transactions through the Coyote
shell. The TLB translates the virtual address to the SLASH BAR's DMA address,
and the XDMA/QDMA engine emits PCIe memory TLPs targeting that address.

### 4.5 PCIe Topology Requirements

PCI P2P transactions require:

1. **Shared PCIe switch or root complex** — both devices must be routable
   peer-to-peer. `pci_p2pdma_distance_many()` checks this automatically.
2. **ACS (Access Control Services)** either disabled or P2P-permissive.
   Lab: `pci=noacs` kernel param. Production: selective ACS config.
3. **IOMMU compatibility** — handled automatically by P2PDMA + `dma_map_sgtable()`.
   No need for `iommu=pt` when using the P2PDMA subsystem properly.
4. **BAR is MMIO and reachable** — standard for any PCI BAR.

**Verification:**
```bash
# Verify topology
lspci -tv

# Check P2P feasibility (kernel >= 5.1)
# If both devices show under the same switch/RC, P2P should work
lspci -vvv -s <slash_bdf> | grep -i "p2p\|acs"
lspci -vvv -s <coyote_bdf> | grep -i "p2p\|acs"
```

### 4.6 Remove-Time Lifecycle Requirements (Must-Have in Phase 1)

For DMABUF-based MMIO P2P, invalidation is not optional:

1. On SLASH remove path, stop accepting new `.attach`, `.map_dma_buf`, and
   `.mmap` operations.
2. Notify importers using `dma_buf_move_notify(dmabuf)` so they drop DMA maps.
3. Wait until active P2P map count reaches zero before provider teardown.
4. Only then proceed with dmabuf teardown and final device removal.

This keeps remove semantics safe and avoids stale DMA mappings to disappearing
BAR MMIO.

---

## 5. Implementation Plan

### Phase 1: Lab Prototype

| Step | Component      | Change                                                   | Effort |
|------|----------------|----------------------------------------------------------|--------|
| 1    | SLASH driver   | `pci_p2pdma_add_resource()` for BAR0 during probe        | Small  |
| 2    | SLASH driver   | `.attach` — topology check + userspace/P2P mode gating   | Small  |
| 3    | SLASH driver   | `.map_dma_buf` — map BAR P2PDMA pages + `dma_map_sgtable()` | Small  |
| 4    | SLASH driver   | `.unmap_dma_buf` + remove-time invalidation (`move_notify`) | Medium |
| 5    | Coyote driver  | Keep dynamic attach path (for invalidation callbacks)    | None   |
| 6    | User app       | Orchestration app using `GET_BAR_FD` fd-only path        | Medium |
| 7    | Coyote HW      | Simple AXI read/write test design for P2P verification   | Medium |
| 8    | System         | Verify PCIe topology (`lspci -tv`) and ACS settings      | Small  |
| 9    | SLASH driver   | Validate mode serialization (`mmap` vs P2P attach)       | Small  |

### Phase 2: Production Hardening

| Step | Component      | Change                                                |
|------|----------------|-------------------------------------------------------|
| 10   | Coyote driver  | Skip card memory alloc for `is_pci_p2pdma_page()` buffers |
| 11   | Coyote driver  | Add `CoyoteAllocType::P2P_BAR` to sw API             |
| 12   | SLASH driver   | Optional concurrent CPU+P2P mode with explicit fencing |
| 13   | SLASH driver   | Support multiple concurrent P2P attachers             |
| 14   | Both           | Performance characterization and tuning               |

---

## 6. Risk Analysis

### 6.1 PCIe Topology Incompatibility

**Risk:** The two FPGAs are behind different root complexes.
**Mitigation:** `pci_p2pdma_distance_many()` in `.attach` detects this and
returns `-EOPNOTSUPP`. Verify with `lspci -tv` before deployment.

### 6.2 ACS Prevents P2P Routing

**Risk:** Access Control Services on the PCIe bridge block P2P TLPs.
**Mitigation:** `pci_p2pdma_distance_many()` checks ACS config. For lab:
`pci=noacs`. For production: selective ACS configuration.

### 6.3 Userspace/P2P Mode Contention

**Risk:** Userspace may hold an active BAR mmap while an app attempts P2P
attach (or vice versa), causing undefined ownership semantics.
**Mitigation:** Phase 1 enforces strict serialization (`-EBUSY` on mode
conflict). Add clear user-space error reporting and retry guidance.

### 6.4 BAR Ordering Semantics

**Risk:** Non-prefetchable BAR accesses enforce strict ordering, which may
limit throughput.
**Mitigation:** For register access, strict ordering is desired. For bulk data,
verify the BAR is prefetchable in the FPGA design.

### 6.5 Device Hot-Removal During P2P

**Risk:** SLASH removal while Coyote has an active TLB mapping. Reads return
`0xFFFFFFFF` (PCIe completion timeout), writes may cause machine checks.
**Mitigation:** Phase 1 must implement `dma_buf_move_notify()` invalidation and
wait-for-unmap synchronization before provider teardown.

### 6.6 Coyote TLB Address Width

**Risk:** Coyote's `TLB_PADDR_RANGE = 44 bits` may not cover the SLASH BAR's
physical address.
**Mitigation:** PCI BARs are almost always below 16 TB. Verify with `lspci -v`.

### 6.7 Card Memory Allocation for BAR dmabufs (Non-blocking)

**Risk:** Coyote's `p2p_attach_dma_buf` allocates card memory (`cpages[]`)
when `en_mem` is set — wasteful for a BAR that will be accessed directly.
**Impact:** Wasted HBM/DDR, but functionally correct. The TLB maps to
`hpages[]` (the BAR's DMA addresses), not `cpages[]`.
**Mitigation:** Phase 2: detect P2PDMA pages and skip card allocation.

---

## 7. File Change Summary

### SLASH Repository (All Changes Here)

| File                          | Change Type | Description                                         |
|-------------------------------|-------------|-----------------------------------------------------|
| `driver/slash_dmabuf.c`      | Modify      | Implement `.attach`, `.map_dma_buf`, `.unmap_dma_buf` for P2P |
| `driver/slash_ctldev.c`      | Modify      | Call `pci_p2pdma_add_resource()` during BAR probe   |
| `driver/slash_dmabuf.h`      | Modify      | (Optional) Add P2P-related fields to private data   |
| `driver/libslash/include/slash/ctldev.h` | Optional Modify | Add `slash_bar_fd_open()` helper (fd-only) |
| `driver/libslash/src/ctldev.c` | Optional Modify | Implement `slash_bar_fd_open()` helper |
| `driver/slash_config.h`      | Modify      | Add `SLASH_ENABLE_P2P` compile flag                 |
| `driver/Makefile`             | Modify      | Add P2P ccflag                                      |

### Coyote Repository (No Changes for Phase 1)

| File                                     | Change Type | Description                              |
|------------------------------------------|-------------|------------------------------------------|
| `driver/src/vfpga/vfpga_gup.c`          | None (Ph1)  | Existing path works unmodified           |
| `sw/include/coyote/cOps.hpp`            | Phase 2     | Add `P2P_BAR` alloc type                |
| `examples/NN_p2p_bar/`                  | New         | P2P BAR access example application       |

### New Artifacts

| Artifact                                   | Description                                  |
|--------------------------------------------|----------------------------------------------|
| User-space orchestration app               | Opens both devices, passes fd, runs test     |
| Access-mode policy note                    | Defines serialized userspace mmap vs P2P mode |
| Coyote test bitstream (user logic)         | Simple register read/write P2P test design   |

---

## 8. Testing Strategy

### 8.1 Unit Tests

1. **P2PDMA registration:** After SLASH probe, verify BAR0 appears in the
   P2PDMA pool (`/sys/bus/pci/devices/<bdf>/p2pmem/`).
2. **Attach acceptance:** Load both drivers, call `dma_buf_attach()` from
   Coyote's device on a SLASH BAR dmabuf. Verify attach succeeds.
3. **SG table validation:** Verify `dma_buf_map_attachment()` returns an
   sg_table with valid DMA address(es) within the BAR range.
4. **Mode conflict handling:** Verify active userspace mmap blocks P2P attach
   with `-EBUSY`, and active P2P map blocks new userspace mmap with `-EBUSY`.

### 8.2 Integration Tests

1. **Read-back test:** Coyote FPGA reads a known register from SLASH BAR0.
2. **Write-read test:** Write a value, read it back, verify.
3. **Bulk test:** Sequential read/write of the full BAR (if data region exists).

### 8.3 Stress / Error Tests

1. **Hot-removal:** Remove SLASH during active P2P. Verify importer receives
   invalidation, DMA mappings are torn down, and no stale DMA persists.
2. **Multiple mappings:** Map BAR from multiple Coyote vFPGAs (Phase 2).
3. **Topology negative test:** Attempt P2P between incompatible root complexes.
   Verify `.attach` rejects gracefully.

---

## 9. Why P2PDMA (Not Manual sg_dma_address)

An earlier draft proposed manually setting `sg_dma_address()` to
`pci_resource_start()`. This is **incorrect for production** because:

1. **IOMMU:** On systems with active IOMMU (Intel VT-d, AMD-Vi), the Coyote
   DMA engine operates in an IOMMU domain. It needs an IOVA, not a raw PCI bus
   address. Manually setting `sg_dma_address` bypasses IOMMU translation.

2. **No `struct page`:** Without P2PDMA pages, `sg_set_page(sgl, NULL, ...)`
   leaves the page pointer NULL. Kernel code that dereferences `sg_page()` will
   crash. Coyote's sg_table walk (`sg_dma_address()`) happens to not dereference
   the page, but other kernel paths might.

3. **No topology check:** Without `pci_p2pdma_distance_many()`, there's no
   validation that P2P is actually possible between the two devices.

The P2PDMA subsystem handles all three correctly. It is the kernel-blessed
mechanism for exactly this use case (NVMe CMB, GPU BAR, FPGA BAR P2P access).

---

## 10. Open Questions

1. **SLASH BAR0 register map:** Which registers are behind BAR0? Need the map
   to design meaningful test cases.

2. **Prefetchable vs non-prefetchable BAR0?** Affects ordering semantics.
   Verify from the Vivado FPGA design.

3. **Bidirectional P2P?** This document covers Coyote -> SLASH. The reverse
   (SLASH DMA engine targeting Coyote BAR) is a separate design.

4. **XDMA vs QDMA for P2P:** Coyote supports both. XDMA (UltraScale+) is
   known to support P2P. QDMA (Versal/V80) needs verification against PG347.

5. **Transfer granularity:** PCIe TLP MPS is typically 128-256B. Register-level
   4-byte reads work fine. Bulk performance needs characterization.

---

## 11. References

- SLASH source: `C:\Users\vserbu\claudius\SLASH\`
- Coyote source: `C:\Users\vserbu\claudius\Coyote\`
- Linux P2PDMA: `Documentation/PCI/p2pdma.rst`
- Linux DMABUF: `Documentation/driver-api/dma-buf.rst`
- Coyote GPU P2P example: `Coyote/examples/06_gpu_p2p/`
- SLASH BAR export: `SLASH/driver/slash_dmabuf.c`
- Coyote P2P import: `Coyote/driver/src/vfpga/vfpga_gup.c`
- NVMe CMB P2P (reference implementation): `drivers/nvme/host/pci.c`
- AMD XDMA: PG195
- AMD QDMA: PG347
