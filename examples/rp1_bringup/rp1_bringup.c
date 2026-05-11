/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Temporary RP1 bring-up tool.  Submits tiny graphs to the RP1 firmware
 * (running on R5 core 1, loaded via xsdb) over BAR4 and reports the
 * outcome.  Will be replaced by a libslash/VRT GraphBuilder once the
 * graph path is on the supported API surface.
 *
 * Subcommands:
 *
 *   dump <slash_ctl_path>
 *       Read and print the RP1 control block.  No graph submission.
 *       Use this to confirm the firmware is alive (magic == "SQR1",
 *       heartbeat advancing) before running the other modes.
 *
 *   signal <slash_ctl_path>
 *       Submit a one-node SIGNAL graph that writes 0xDEADBEEF into
 *       signal slot 0.  Proves the firmware boots on real silicon, the
 *       host's BAR window reaches RP1's DDR, and the flat scanner runs
 *       end-to-end.  No kernels involved.
 *
 *   kernel <slash_ctl_path> <kernel_r5_addr_hex> [arg0_hex ...]
 *       Submit  SIGNAL -> KERNEL_DISPATCH(<r5_addr>, args) -> SIGNAL.
 *       Proves the new LPD->NoC->AXI-Lite path lets RP1 reach a kernel
 *       in the user region (0x8800_0000 + offset in R5 space).  Verifies
 *       only that the trailing SIGNAL fires (i.e. ap_done came back);
 *       the kernel's functional output is not checked.
 *
 *       <kernel_r5_addr_hex> is computed from the kernel's host-view
 *       address (system_map.xml <BaseAddress>) as:
 *           r5_addr = xml_addr - 0x0202_0000_0000 + 0x8800_0000
 *
 *       Optional [arg0_hex ...] are written to the kernel's AXI-Lite
 *       arg registers at offsets +0x10, +0x14, +0x18, ...
 */

#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <slash/ctldev.h>
#include <slash/uapi/rp1_protocol.h>

/* -------------------------------------------------------------------------
 * Host-side constants.
 *
 * BAR4 maps RP1's BAR-visible DDR window.  The control block at R5
 * address 0x3000_0000 appears at BAR4 offset 64 MB on the host (same
 * relationship the existing examples/07_rp1_memcheck has validated).
 * If your bitstream uses a different BAR layout, override these.
 * ---------------------------------------------------------------------- */

#define BAR_NUMBER           4
#define BAR_CTRL_OFFSET      (64ULL * 1024ULL * 1024ULL)

#define POLL_TIMEOUT_NS      (3LL * 1000LL * 1000LL * 1000LL)  /* 3 s */
#define POLL_INTERVAL_NS     (1LL * 1000LL * 1000LL)           /* 1 ms */

#define BRINGUP_CQ_SIZE      64u
#define BRINGUP_MAX_ARGS     8

/* -------------------------------------------------------------------------
 * Small utilities
 * ---------------------------------------------------------------------- */

static long long monotonic_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long long)ts.tv_sec * 1000000000LL + (long long)ts.tv_nsec;
}

static void sleep_ns(long long ns)
{
    struct timespec ts = {
        .tv_sec  = (time_t)(ns / 1000000000LL),
        .tv_nsec = (long)(ns % 1000000000LL),
    };
    nanosleep(&ts, NULL);
}

/* Translate an R5 address inside the BAR-visible DDR window to a
 * BAR4-relative byte offset on the host. */
static uint64_t r5_to_bar(uint64_t r5_addr)
{
    return BAR_CTRL_OFFSET + (r5_addr - (uint64_t)RP1_CTRL_PHYS_ADDR);
}

static volatile void *bar_ptr(volatile void *bar_base, uint64_t r5_addr)
{
    return (volatile void *)((volatile uint8_t *)bar_base + r5_to_bar(r5_addr));
}

/* -------------------------------------------------------------------------
 * Pretty-printing
 * ---------------------------------------------------------------------- */

static const char *rp1_state_str(uint32_t s)
{
    switch (s) {
    case RP1_STATE_INIT:    return "INIT";
    case RP1_STATE_READY:   return "READY";
    case RP1_STATE_RUNNING: return "RUNNING";
    case RP1_STATE_ERROR:   return "ERROR";
    case RP1_STATE_HALTED:  return "HALTED";
    default:                return "?";
    }
}

static void dump_ctrl(volatile rp1_ctrl_t *c)
{
    printf("  magic            = 0x%08" PRIx32 " (%s)\n",
           c->magic, (c->magic == RP1_CTRL_MAGIC) ? "SQR1" : "BAD");
    printf("  version          = %" PRIu32 "\n",   c->version);
    printf("  node_count       = %" PRIu32 "\n",   c->node_count);
    printf("  cq_size          = %" PRIu32 "\n",   c->cq_size);
    printf("  node_base        = 0x%08" PRIx32 "_%08" PRIx32 "\n",
           c->node_base_hi, c->node_base_lo);
    printf("  cq_base          = 0x%08" PRIx32 "_%08" PRIx32 "\n",
           c->cq_base_hi, c->cq_base_lo);
    printf("  arg_buf_base     = 0x%08" PRIx32 "_%08" PRIx32 "\n",
           c->arg_buf_base_hi, c->arg_buf_base_lo);
    printf("  sig_array_base   = 0x%08" PRIx32 "_%08" PRIx32 "\n",
           c->sig_array_base_hi, c->sig_array_base_lo);
    printf("  graph_seq        = %" PRIu32 "\n",   c->graph_seq);
    printf("  graph_done_seq   = %" PRIu32 "\n",   c->graph_done_seq);
    printf("  cq_write_idx     = %" PRIu32 "\n",   c->cq_write_idx);
    printf("  rp1_state        = %" PRIu32 " (%s)\n",
           c->rp1_state, rp1_state_str(c->rp1_state));
    printf("  rp1_error_code   = %" PRIu32 "\n",   c->rp1_error_code);
    printf("  rp1_current_node = %" PRIu32 "\n",   c->rp1_current_node);
    printf("  heartbeat        = %" PRIu32 "\n",   c->heartbeat);
}

/* -------------------------------------------------------------------------
 * Graph helpers
 *
 * The control block / node array / signal array / arg buffer live at
 * the same DDR offsets the firmware expects (see RP1_DEFAULT_*_OFFSET
 * in rp1_protocol.h).  Each helper accesses them through the volatile
 * BAR mapping returned by libslash.
 * ---------------------------------------------------------------------- */

static volatile rp1_ctrl_t *get_ctrl(volatile void *bar)
{
    return (volatile rp1_ctrl_t *)bar_ptr(bar, RP1_CTRL_PHYS_ADDR);
}

static rp1_node_t *get_nodes(volatile void *bar)
{
    /* Nodes are written word-by-word but the underlying memory is BAR
     * MMIO; using a plain rp1_node_t * is fine because we never let the
     * compiler hold cached state across __sync_synchronize barriers. */
    return (rp1_node_t *)(uintptr_t)bar_ptr(bar,
        RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_NODE_ARRAY_OFFSET);
}

static uint32_t *get_args(volatile void *bar)
{
    return (uint32_t *)(uintptr_t)bar_ptr(bar,
        RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_ARG_BUF_OFFSET);
}

static volatile rp1_signal_slot_t *get_sigs(volatile void *bar)
{
    return (volatile rp1_signal_slot_t *)bar_ptr(bar,
        RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_SIG_ARRAY_OFFSET);
}

static volatile rp1_cq_entry_t *get_cq(volatile void *bar)
{
    return (volatile rp1_cq_entry_t *)bar_ptr(bar,
        RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_CQ_OFFSET);
}

/* Zero a region of BAR memory using volatile byte writes. */
static void bar_zero(volatile void *dst, size_t bytes)
{
    volatile uint8_t *p = (volatile uint8_t *)dst;
    for (size_t i = 0; i < bytes; i++) p[i] = 0;
}

/* Common control-block setup: program base addresses + cq_size. */
static void program_ctrl(volatile rp1_ctrl_t *c, uint32_t node_count)
{
    c->node_count        = node_count;
    c->cq_size           = BRINGUP_CQ_SIZE;
    c->node_base_lo      = (uint32_t)(RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_NODE_ARRAY_OFFSET);
    c->node_base_hi      = 0;
    c->cq_base_lo        = (uint32_t)(RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_CQ_OFFSET);
    c->cq_base_hi        = 0;
    c->arg_buf_base_lo   = (uint32_t)(RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_ARG_BUF_OFFSET);
    c->arg_buf_base_hi   = 0;
    c->sig_array_base_lo = (uint32_t)(RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_SIG_ARRAY_OFFSET);
    c->sig_array_base_hi = 0;
}

static void node_set_header(rp1_node_t *n, uint16_t opcode,
                            uint8_t aw_b, uint32_t aw_m,
                            uint8_t st_b, uint32_t st_m)
{
    n->opcode               = opcode;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;
}

static int wait_for_seq(volatile rp1_ctrl_t *c, uint32_t want_seq)
{
    long long deadline = monotonic_ns() + POLL_TIMEOUT_NS;
    while (c->graph_done_seq < want_seq) {
        if (monotonic_ns() > deadline) {
            fprintf(stderr,
                    "TIMEOUT: graph_done_seq=%" PRIu32 " (want %" PRIu32 ")\n",
                    c->graph_done_seq, want_seq);
            return -1;
        }
        sleep_ns(POLL_INTERVAL_NS);
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * Subcommand: dump
 * ---------------------------------------------------------------------- */

static int cmd_dump(volatile void *bar)
{
    volatile rp1_ctrl_t *c = get_ctrl(bar);
    printf("RP1 control block @ R5 0x%08lx (BAR4 + 0x%" PRIx64 "):\n",
           (unsigned long)RP1_CTRL_PHYS_ADDR, r5_to_bar(RP1_CTRL_PHYS_ADDR));
    dump_ctrl(c);

    uint32_t hb1 = c->heartbeat;
    sleep_ns(500LL * 1000LL * 1000LL);   /* 500 ms */
    uint32_t hb2 = c->heartbeat;
    if (hb2 != hb1)
        printf("Liveness: heartbeat advanced %" PRIu32 " -> %" PRIu32 " (running)\n",
               hb1, hb2);
    else
        printf("Liveness: heartbeat unchanged at %" PRIu32 " (stuck or not loaded)\n",
               hb1);
    return 0;
}

/* -------------------------------------------------------------------------
 * Subcommand: signal
 *
 * One-node SIGNAL graph.  Proves end-to-end firmware operation on
 * silicon without touching the new AXI-Lite path.
 * ---------------------------------------------------------------------- */

#define SIGNAL_MAGIC  0xDEADBEEFu

static int cmd_signal(volatile void *bar)
{
    volatile rp1_ctrl_t       *c     = get_ctrl(bar);
    rp1_node_t                *nodes = get_nodes(bar);
    volatile rp1_signal_slot_t *sigs = get_sigs(bar);

    bar_zero((volatile void *)&nodes[0], sizeof(rp1_node_t));
    node_set_header(&nodes[0], RP1_OP_SIGNAL,
                    /* await */ 0, 0x0,
                    /* set   */ 0, 0x1);
    nodes[0].payload.signal.target_slot = 0;
    nodes[0].payload.signal.value       = SIGNAL_MAGIC;
    nodes[0].payload.signal.operation   = RP1_SIGOP_SET;

    sigs[0].value            = 0;
    sigs[0].last_writer_node = 0;
    sigs[0].flags            = 0;

    program_ctrl(c, /* node_count */ 1);

    uint32_t want_seq = c->graph_done_seq + 1;
    __sync_synchronize();
    c->graph_seq = want_seq;
    __sync_synchronize();

    printf("signal: submitted seq=%" PRIu32 ", polling...\n", want_seq);
    if (wait_for_seq(c, want_seq) != 0) {
        dump_ctrl(c);
        return 1;
    }

    __sync_synchronize();
    uint32_t observed = sigs[0].value;
    if (observed != SIGNAL_MAGIC) {
        fprintf(stderr,
                "FAIL: signal slot 0 = 0x%08" PRIx32 ", expected 0x%08" PRIx32 "\n",
                observed, SIGNAL_MAGIC);
        dump_ctrl(c);
        return 1;
    }

    printf("PASS: slot[0] = 0x%08" PRIx32 ", cq_write_idx=%" PRIu32 ", state=%s\n",
           observed, c->cq_write_idx, rp1_state_str(c->rp1_state));
    return 0;
}

/* -------------------------------------------------------------------------
 * Subcommand: kernel
 *
 * SIGNAL -> KERNEL_DISPATCH(<r5_addr>, args) -> SIGNAL.  Verifies that
 * the new RPU -> NoC -> user-region AXI-Lite path reaches the kernel
 * and that ap_done propagates back through check_inflight().
 * ---------------------------------------------------------------------- */

#define PRE_MARKER   0xBEEFBEEFu
#define POST_MARKER  0xCAFEBABEu

static int cmd_kernel(volatile void *bar, uint32_t kernel_r5_addr,
                      const uint32_t *args, size_t arg_count)
{
    if (arg_count > BRINGUP_MAX_ARGS) {
        fprintf(stderr, "Too many args: %zu (max %d)\n",
                arg_count, BRINGUP_MAX_ARGS);
        return 1;
    }

    volatile rp1_ctrl_t       *c     = get_ctrl(bar);
    rp1_node_t                *nodes = get_nodes(bar);
    uint32_t                  *argbuf = get_args(bar);
    volatile rp1_signal_slot_t *sigs = get_sigs(bar);

    /* ---- Stage args (if any) at arg_buffer_offset=0 ---- */
    for (size_t i = 0; i < arg_count; i++)
        argbuf[i] = args[i];

    /* ---- Build 3-node graph ---- */
    bar_zero((volatile void *)&nodes[0], 3 * sizeof(rp1_node_t));

    /* Node 0: pre-dispatch marker. */
    node_set_header(&nodes[0], RP1_OP_SIGNAL,
                    /* await */ 0, 0x0,
                    /* set   */ 0, 0x1);
    nodes[0].payload.signal.target_slot = 0;
    nodes[0].payload.signal.value       = PRE_MARKER;
    nodes[0].payload.signal.operation   = RP1_SIGOP_SET;

    /* Node 1: dispatch the kernel. */
    node_set_header(&nodes[1], RP1_OP_KERNEL_DISPATCH,
                    /* await */ 0, 0x1,
                    /* set   */ 0, 0x2);
    nodes[1].payload.kernel_dispatch.kernel_base_addr  = kernel_r5_addr;
    nodes[1].payload.kernel_dispatch.arg_buffer_offset = 0;
    nodes[1].payload.kernel_dispatch.arg_count         = (uint16_t)arg_count;
    nodes[1].payload.kernel_dispatch.ctrl_flags        = 0;
    nodes[1].payload.kernel_dispatch.timeout_cycles    = 0;  /* default */

    /* Node 2: post-dispatch marker. */
    node_set_header(&nodes[2], RP1_OP_SIGNAL,
                    /* await */ 0, 0x2,
                    /* set   */ 0, 0x4);
    nodes[2].payload.signal.target_slot = 1;
    nodes[2].payload.signal.value       = POST_MARKER;
    nodes[2].payload.signal.operation   = RP1_SIGOP_SET;

    /* Clear the two slots we'll verify. */
    sigs[0].value = 0; sigs[0].last_writer_node = 0; sigs[0].flags = 0;
    sigs[1].value = 0; sigs[1].last_writer_node = 0; sigs[1].flags = 0;

    program_ctrl(c, /* node_count */ 3);

    uint32_t want_seq = c->graph_done_seq + 1;
    __sync_synchronize();
    c->graph_seq = want_seq;
    __sync_synchronize();

    printf("kernel: submitted seq=%" PRIu32 " (kernel @ R5 0x%08" PRIx32
           ", arg_count=%zu), polling...\n",
           want_seq, kernel_r5_addr, arg_count);
    if (wait_for_seq(c, want_seq) != 0) {
        dump_ctrl(c);
        return 1;
    }

    __sync_synchronize();
    uint32_t pre  = sigs[0].value;
    uint32_t post = sigs[1].value;

    if (pre != PRE_MARKER) {
        fprintf(stderr,
                "FAIL: pre-marker (slot 0) = 0x%08" PRIx32 ", expected 0x%08" PRIx32
                " -- graph never started\n",
                pre, PRE_MARKER);
        dump_ctrl(c);
        return 1;
    }
    if (post != POST_MARKER) {
        fprintf(stderr,
                "FAIL: post-marker (slot 1) = 0x%08" PRIx32 ", expected 0x%08" PRIx32
                " -- kernel never asserted ap_done\n",
                post, POST_MARKER);
        dump_ctrl(c);
        return 1;
    }

    printf("PASS: slot[0]=0x%08" PRIx32 " slot[1]=0x%08" PRIx32
           " cq_write_idx=%" PRIu32 " state=%s\n",
           pre, post, c->cq_write_idx, rp1_state_str(c->rp1_state));
    return 0;
}

/* -------------------------------------------------------------------------
 * Driver
 * ---------------------------------------------------------------------- */

static void usage(const char *argv0)
{
    fprintf(stderr,
            "Usage:\n"
            "  %s dump   <slash_ctl_path>\n"
            "  %s signal <slash_ctl_path>\n"
            "  %s kernel <slash_ctl_path> <kernel_r5_addr_hex> [arg0_hex ...]\n"
            "\n"
            "  <kernel_r5_addr_hex> = xml_addr - 0x0202_0000_0000 + 0x8800_0000\n",
            argv0, argv0, argv0);
}

int main(int argc, char **argv)
{
    if (argc < 3) {
        usage(argv[0]);
        return EXIT_FAILURE;
    }

    const char *mode = argv[1];
    const char *path = argv[2];

    struct slash_ctldev *dev = slash_ctldev_open(path);
    if (!dev) {
        perror("slash_ctldev_open");
        return EXIT_FAILURE;
    }

    struct slash_ioctl_bar_info *info = slash_bar_info_read(dev, BAR_NUMBER);
    if (!info) {
        perror("slash_bar_info_read");
        slash_ctldev_close(dev);
        return EXIT_FAILURE;
    }
    if (!info->usable) {
        fprintf(stderr, "BAR%d not usable\n", BAR_NUMBER);
        slash_bar_info_free(info);
        slash_ctldev_close(dev);
        return EXIT_FAILURE;
    }
    /* Need enough BAR to cover the signal array (the highest default offset). */
    const uint64_t bar_required = BAR_CTRL_OFFSET + RP1_DEFAULT_SIG_ARRAY_OFFSET
                                + 256u * sizeof(rp1_signal_slot_t);
    if ((uint64_t)info->length < bar_required) {
        fprintf(stderr,
                "BAR%d length 0x%" PRIx64 " < required 0x%" PRIx64 "\n",
                BAR_NUMBER, (uint64_t)info->length, bar_required);
        slash_bar_info_free(info);
        slash_ctldev_close(dev);
        return EXIT_FAILURE;
    }

    struct slash_bar_file *bar_file = slash_bar_file_open(dev, BAR_NUMBER, O_CLOEXEC);
    if (!bar_file) {
        perror("slash_bar_file_open");
        slash_bar_info_free(info);
        slash_ctldev_close(dev);
        return EXIT_FAILURE;
    }

    if (slash_bar_file_start_write(bar_file) != 0) {
        perror("slash_bar_file_start_write");
        slash_bar_file_close(bar_file);
        slash_bar_info_free(info);
        slash_ctldev_close(dev);
        return EXIT_FAILURE;
    }

    volatile void *bar = (volatile void *)bar_file->map;
    int rc;

    if (strcmp(mode, "dump") == 0) {
        rc = cmd_dump(bar);
    } else if (strcmp(mode, "signal") == 0) {
        rc = cmd_signal(bar);
    } else if (strcmp(mode, "kernel") == 0) {
        if (argc < 4) {
            fprintf(stderr, "kernel mode requires <kernel_r5_addr_hex>\n");
            usage(argv[0]);
            rc = 1;
        } else {
            errno = 0;
            char *endp = NULL;
            uint64_t r5 = strtoull(argv[3], &endp, 0);
            if (errno != 0 || endp == argv[3] || r5 == 0 || r5 > 0xFFFFFFFFu) {
                fprintf(stderr, "Invalid kernel_r5_addr_hex '%s'\n", argv[3]);
                rc = 1;
            } else {
                uint32_t args[BRINGUP_MAX_ARGS];
                size_t arg_count = 0;
                for (int i = 4; i < argc && arg_count < BRINGUP_MAX_ARGS; i++) {
                    errno = 0;
                    uint64_t a = strtoull(argv[i], &endp, 0);
                    if (errno != 0 || endp == argv[i] || a > 0xFFFFFFFFu) {
                        fprintf(stderr, "Invalid arg '%s'\n", argv[i]);
                        rc = 1;
                        goto done;
                    }
                    args[arg_count++] = (uint32_t)a;
                }
                rc = cmd_kernel(bar, (uint32_t)r5, args, arg_count);
            }
        }
    } else {
        fprintf(stderr, "Unknown subcommand: %s\n", mode);
        usage(argv[0]);
        rc = 1;
    }

done:
    slash_bar_file_end_write(bar_file);
    slash_bar_file_close(bar_file);
    slash_bar_info_free(info);
    slash_ctldev_close(dev);
    return rc == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
