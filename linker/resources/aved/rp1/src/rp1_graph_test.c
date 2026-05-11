/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * End-to-end graph tests for the RP1 flat scanner, running under Xilinx
 * QEMU with ARM semihosting.  Each test builds a tiny graph in shared
 * DDR, hands it to rp1_run() via the on_scan_pass / on_graph_done hooks
 * defined in rp1_loop.h, and asserts on the resulting node statuses,
 * barriers, signal slots, CQ entries, and dispatch order.
 *
 * The hook completes fake "kernels" by OR-ing 0x2 (ap_done) into the
 * ctrl-reg word of each in-flight kernel after activate_nodes() has
 * dispatched it.  The kernels themselves are just RAM pages at
 * FAKE_KERNEL_BASE; the firmware writes to them through axi_write32()
 * (a plain volatile store), which works fine over QEMU RAM.
 */

#ifdef QEMU_SEMIHOSTING

#include "rp1_test.h"
#include "rp1_store.h"
#include "rp1_run.h"

#include <slash/uapi/rp1_protocol.h>

#include <stddef.h>
#include <stdint.h>

/* -------------------------------------------------------------------------
 * DDR layout for the test graphs.
 *
 * Mirrors the protocol defaults so the test environment matches what the
 * host stack will eventually program into the control block.  The
 * control block sits at RP1_CTRL_PHYS_ADDR; nodes / CQ / args / signals
 * follow at the documented offsets.
 * ---------------------------------------------------------------------- */

#define G_CTRL  ((volatile rp1_ctrl_t *)(uintptr_t)(RP1_CTRL_PHYS_ADDR))
#define G_NODES ((rp1_node_t *)(uintptr_t) \
                 (RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_NODE_ARRAY_OFFSET))
#define G_CQ    ((volatile rp1_cq_entry_t *)(uintptr_t) \
                 (RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_CQ_OFFSET))
#define G_ARGS  ((uint32_t *)(uintptr_t) \
                 (RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_ARG_BUF_OFFSET))
#define G_SIGS  ((volatile rp1_signal_slot_t *)(uintptr_t) \
                 (RP1_CTRL_PHYS_ADDR + RP1_DEFAULT_SIG_ARRAY_OFFSET))

#define TEST_CQ_SIZE  64u

/* Fake AXI-Lite kernel: 256-byte page per kernel, word 0 is the
 * ap_start/ap_done control reg, word 4 (offset 0x10) is arg 0.  Lives in
 * QEMU RAM well clear of the BAR window. */
#define FAKE_KERNEL_BASE   0x40000000UL
#define FAKE_KERNEL_STRIDE 0x100UL
#define FAKE_KERNEL(i)     (FAKE_KERNEL_BASE + (uintptr_t)(i) * FAKE_KERNEL_STRIDE)

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static void tmemzero(volatile void *dst, uint32_t len)
{
    volatile uint8_t *p = (volatile uint8_t *)dst;
    while (len--) *p++ = 0;
}

/* -------------------------------------------------------------------------
 * Hook state — fake-kernel completion + dispatch tracing.
 *
 * on_scan_pass fires after activate_nodes() and check_inflight() in each
 * iteration of the dispatch loop, which gives us a single observation
 * point per scan.  We use it to (a) note which nodes have left PENDING,
 * (b) record the peak number of in-flight kernels (the proxy for "B and
 * C dispatched in parallel"), and (c) flip ap_done on any in-flight
 * fake kernel so the next iteration's check_inflight() finalises it.
 * ---------------------------------------------------------------------- */

#define TRACE_MAX 64u

static uint32_t s_trace[TRACE_MAX];
static uint32_t s_trace_count;
static uint32_t s_seen[TRACE_MAX / 32u];  /* bitmask of nodes already traced */
static uint32_t s_max_inflight;
static uint32_t s_node_count;
static int      s_graph_done_returns;

static void hook_reset(uint32_t node_count)
{
    s_trace_count = 0;
    s_max_inflight = 0;
    s_node_count = node_count;
    s_graph_done_returns = 1;   /* default: exit rp1_run after one graph */
    for (uint32_t i = 0; i < TRACE_MAX / 32u; i++) s_seen[i] = 0;
    for (uint32_t i = 0; i < TRACE_MAX; i++) s_trace[i] = 0;
}

static void hook_on_scan_pass(void)
{
    if (g_inflight_count > s_max_inflight)
        s_max_inflight = g_inflight_count;

    for (uint32_t i = 0; i < s_node_count && i < TRACE_MAX; i++) {
        if (s_seen[i >> 5] & (1u << (i & 31u))) continue;
        uint8_t st = g_node_status[i];
        if (st == RP1_NODE_DISPATCHED || st == RP1_NODE_DONE) {
            s_seen[i >> 5] |= (1u << (i & 31u));
            s_trace[s_trace_count++] = i;
        }
    }

    /* Complete each in-flight fake kernel (ap_start -> ap_start | ap_done). */
    for (uint32_t i = 0; i < g_inflight_count; i++) {
        volatile uint32_t *ctrl =
            (volatile uint32_t *)(uintptr_t)g_inflight[i].base_addr;
        if (*ctrl & 0x1u) *ctrl |= 0x2u;
    }
}

static int hook_on_graph_done(int result)
{
    (void)result;
    return s_graph_done_returns;
}

static const rp1_hooks_t s_hooks = {
    .on_scan_pass  = hook_on_scan_pass,
    .on_graph_done = hook_on_graph_done,
    .on_idle       = 0,
};

/* -------------------------------------------------------------------------
 * Graph setup
 * ---------------------------------------------------------------------- */

static void setup_graph(uint32_t node_count, uint32_t fake_kernel_count)
{
    /* Wipe only the regions we touch.  rp1_run() resets the BTCM state
     * (barriers, node_status, loop_iters, inflight) on each new graph
     * submission via rp1_store_init(). */
    tmemzero((volatile void *)G_CTRL,  sizeof(rp1_ctrl_t));
    tmemzero((volatile void *)G_NODES, node_count * sizeof(rp1_node_t));
    tmemzero((volatile void *)G_CQ,    TEST_CQ_SIZE * sizeof(rp1_cq_entry_t));
    tmemzero((volatile void *)G_ARGS,  64u * sizeof(uint32_t));
    tmemzero((volatile void *)G_SIGS,  64u * sizeof(rp1_signal_slot_t));
    if (fake_kernel_count > 0) {
        tmemzero((volatile void *)(uintptr_t)FAKE_KERNEL_BASE,
                 fake_kernel_count * FAKE_KERNEL_STRIDE);
    }

    G_CTRL->cq_size           = TEST_CQ_SIZE;
    G_CTRL->node_count        = node_count;
    G_CTRL->node_base_lo      = (uint32_t)(uintptr_t)G_NODES;
    G_CTRL->cq_base_lo        = (uint32_t)(uintptr_t)G_CQ;
    G_CTRL->arg_buf_base_lo   = (uint32_t)(uintptr_t)G_ARGS;
    G_CTRL->sig_array_base_lo = (uint32_t)(uintptr_t)G_SIGS;
    G_CTRL->graph_seq         = 1;

    hook_reset(node_count);
}

/* -------------------------------------------------------------------------
 * Node builders
 *
 * The flat scanner only reads what each opcode's payload defines, so we
 * touch exactly those fields and rely on setup_graph()'s tmemzero for
 * the rest.  Keeping the builders explicit avoids compound literals,
 * which can lower to a memset call under -ffreestanding -nostdlib.
 * ---------------------------------------------------------------------- */

static void make_kernel(rp1_node_t *n, uint32_t kernel_idx,
                        uint8_t aw_b, uint32_t aw_m,
                        uint8_t st_b, uint32_t st_m,
                        uint32_t arg_buf_offset, uint16_t arg_count)
{
    n->opcode               = RP1_OP_KERNEL_DISPATCH;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;

    n->payload.kernel_dispatch.kernel_base_addr  = (uint32_t)FAKE_KERNEL(kernel_idx);
    n->payload.kernel_dispatch.arg_buffer_offset = arg_buf_offset;
    n->payload.kernel_dispatch.arg_count         = arg_count;
    n->payload.kernel_dispatch.ctrl_flags        = 0;
    n->payload.kernel_dispatch.timeout_cycles    = 0; /* default */
}

static void make_signal(rp1_node_t *n,
                        uint32_t slot, uint32_t value, uint16_t op,
                        uint8_t aw_b, uint32_t aw_m,
                        uint8_t st_b, uint32_t st_m)
{
    n->opcode               = RP1_OP_SIGNAL;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;

    n->payload.signal.target_slot = slot;
    n->payload.signal.value       = value;
    n->payload.signal.operation   = op;
}

static void make_loop(rp1_node_t *n,
                      uint32_t body_start, uint32_t body_end,
                      uint32_t cond_signal, uint16_t cond_op, uint32_t cond_val,
                      uint8_t bucket_clear_start, uint8_t bucket_clear_end,
                      uint8_t loop_id, uint32_t max_iter,
                      uint8_t aw_b, uint32_t aw_m,
                      uint8_t st_b, uint32_t st_m)
{
    n->opcode               = RP1_OP_LOOP;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;

    n->payload.loop.body_start         = body_start;
    n->payload.loop.body_end           = body_end;
    n->payload.loop.max_iterations     = max_iter;
    n->payload.loop.condition_signal   = cond_signal;
    n->payload.loop.condition_value    = cond_val;
    n->payload.loop.condition_op       = cond_op;
    n->payload.loop.bucket_clear_start = bucket_clear_start;
    n->payload.loop.bucket_clear_end   = bucket_clear_end;
    n->payload.loop.loop_id            = loop_id;
}

static void make_rerun(rp1_node_t *n, uint32_t target_node,
                       uint8_t aw_b, uint32_t aw_m,
                       uint8_t st_b, uint32_t st_m)
{
    n->opcode               = RP1_OP_RERUN;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;

    n->payload.rerun.target_node = target_node;
    n->payload.rerun.rerun_flags = 0;
    n->payload.rerun.loop_id     = 0;
}

static void make_cond(rp1_node_t *n,
                      uint32_t cond_signal, uint16_t cond_op, uint32_t cond_val,
                      uint32_t body_start, uint32_t body_end,
                      uint8_t bucket_clear_start, uint8_t bucket_clear_end,
                      uint8_t done_bucket, uint32_t done_mask,
                      uint8_t aw_b, uint32_t aw_m,
                      uint8_t st_b, uint32_t st_m)
{
    n->opcode               = RP1_OP_COND;
    n->flags                = 0;
    n->barrier_await_mask   = aw_m;
    n->barrier_set_mask     = st_m;
    n->barrier_await_bucket = aw_b;
    n->barrier_set_bucket   = st_b;
    n->status               = RP1_NODE_PENDING;

    n->payload.cond.condition_signal   = cond_signal;
    n->payload.cond.condition_value    = cond_val;
    n->payload.cond.condition_op       = cond_op;
    n->payload.cond.bucket_clear_start = bucket_clear_start;
    n->payload.cond.bucket_clear_end   = bucket_clear_end;
    n->payload.cond.body_start         = body_start;
    n->payload.cond.body_end           = body_end;
    n->payload.cond.done_bucket        = done_bucket;
    n->payload.cond.done_mask          = done_mask;
}

/* -------------------------------------------------------------------------
 * test_diamond_dag
 *
 *        A (k0)
 *       / \
 *      B   C  (k1, k2)
 *       \ /
 *        D (k3)
 *
 * Verifies:
 *   - KERNEL_DISPATCH writes args to FAKE_K + 0x10 and ap_start to + 0x00.
 *   - barrier AND ({B,C} done) gates D.
 *   - parallel dispatch: B and C are both in flight at some point.
 *   - the CQ is populated in the natural dispatch order.
 * ---------------------------------------------------------------------- */

static int test_diamond_dag(void)
{
    setup_graph(/* node_count */ 4, /* fake_kernels */ 4);

    /* One arg per kernel: arg[i] = i, parked at G_ARGS[i] (byte offset i*4). */
    for (uint32_t i = 0; i < 4; i++) G_ARGS[i] = i;

    make_kernel(&G_NODES[0], 0, 0, 0x00, 0, 0x01, 0u * 4u, 1);
    make_kernel(&G_NODES[1], 1, 0, 0x01, 0, 0x02, 1u * 4u, 1);
    make_kernel(&G_NODES[2], 2, 0, 0x01, 0, 0x04, 2u * 4u, 1);
    make_kernel(&G_NODES[3], 3, 0, 0x06, 0, 0x08, 3u * 4u, 1);

    int rc = rp1_run(&s_hooks);
    CHECK_EQ32(rc, 0u, "diamond: rp1_run rc");

    CHECK_EQ32(s_trace_count, 4u, "diamond: nodes traced");
    CHECK_EQ32(s_trace[0],    0u, "diamond: A first");
    CHECK_EQ32(s_trace[1],    1u, "diamond: B second");
    CHECK_EQ32(s_trace[2],    2u, "diamond: C third");
    CHECK_EQ32(s_trace[3],    3u, "diamond: D last");
    CHECK(s_max_inflight >= 2u, "diamond: B and C in flight together");

    CHECK_EQ32(G_CTRL->cq_write_idx,   4u,                 "diamond: cq entries");
    CHECK_EQ32(G_CTRL->graph_done_seq, 1u,                 "diamond: graph_done_seq");
    CHECK_EQ32(G_CTRL->rp1_state,      RP1_STATE_READY,    "diamond: rp1_state");

    for (uint32_t i = 0; i < 4; i++) {
        volatile uint32_t *ctrl = (volatile uint32_t *)(uintptr_t)FAKE_KERNEL(i);
        CHECK_EQ32(ctrl[0],        0x3u, "diamond: ctrl reg ap_start|ap_done");
        CHECK_EQ32(ctrl[0x10 / 4], i,    "diamond: kernel arg[0]");
    }
    return 0;
}

/* -------------------------------------------------------------------------
 * test_signal_chain
 *
 *   n0 -> n1 -> n2 -> n3   (each writes a different signal slot)
 *
 * Pure-scanner sanity check: no kernels, only immediate-completion
 * SIGNAL ops chained via single-bit barrier dependencies in bucket 0.
 * Exercises the DDR-resolved pointers (g_nodes, g_signals, g_cq).
 * ---------------------------------------------------------------------- */

static int test_signal_chain(void)
{
    setup_graph(/* node_count */ 4, /* fake_kernels */ 0);

    make_signal(&G_NODES[0], 0, 0xA000u, RP1_SIGOP_SET, 0, 0x00, 0, 0x1);
    make_signal(&G_NODES[1], 1, 0xB001u, RP1_SIGOP_SET, 0, 0x01, 0, 0x2);
    make_signal(&G_NODES[2], 2, 0xC002u, RP1_SIGOP_SET, 0, 0x02, 0, 0x4);
    make_signal(&G_NODES[3], 3, 0xD003u, RP1_SIGOP_SET, 0, 0x04, 0, 0x8);

    int rc = rp1_run(&s_hooks);
    CHECK_EQ32(rc, 0u, "chain: rp1_run rc");

    CHECK_EQ32(G_SIGS[0].value, 0xA000u, "chain: slot 0");
    CHECK_EQ32(G_SIGS[1].value, 0xB001u, "chain: slot 1");
    CHECK_EQ32(G_SIGS[2].value, 0xC002u, "chain: slot 2");
    CHECK_EQ32(G_SIGS[3].value, 0xD003u, "chain: slot 3");

    CHECK_EQ32(s_trace_count,        4u, "chain: nodes traced");
    CHECK_EQ32(G_CTRL->cq_write_idx, 4u, "chain: cq entries");
    CHECK_EQ32(G_CTRL->graph_done_seq, 1u, "chain: graph_done_seq");
    return 0;
}

/* -------------------------------------------------------------------------
 * test_loop_decrement
 *
 *  init -> LOOP --(body)--> RERUN
 *           ^                 |
 *           +-----------------+
 *           |
 *           +--exit-> finalize
 *
 *  Node 0: SIGNAL  slot=0 SET 3
 *  Node 1: LOOP    body=[2,3], cond: slot[0] EQ 0, bucket_clear=[1,1]
 *  Node 2: SIGNAL  slot=0 ADD 0xFFFFFFFF       (decrement by 1)
 *  Node 3: RERUN   target=1
 *  Node 4: SIGNAL  slot=10 SET 0xCAFEBABE
 *
 * Expected:
 *   - body runs 3 times (slot 3 -> 2 -> 1 -> 0); the 4th LOOP pass hits
 *     the exit condition (loop_iters[0] is incremented before the check,
 *     so it lands at 4 on exit).
 *   - finalize fires after the LOOP node sets its own barrier on exit.
 *   - CQ entries: init + 3*(body SIGNAL + body RERUN) + LOOP_exit + final = 9.
 *     (LOOP does NOT write CQ on the continue path, only on exit.)
 * ---------------------------------------------------------------------- */

static int test_loop_decrement(void)
{
    setup_graph(/* node_count */ 5, /* fake_kernels */ 0);

    make_signal(&G_NODES[0], 0, 3u, RP1_SIGOP_SET, 0, 0x00, 0, 0x1);
    make_loop(  &G_NODES[1],
                /* body */ 2, 3,
                /* cond */ 0, RP1_COP_EQ, 0u,
                /* clear */ 1, 1,
                /* loop_id */ 0, /* max_iter */ 10u,
                /* await */ 0, 0x1, /* set on exit */ 0, 0x2);
    make_signal(&G_NODES[2], 0, 0xFFFFFFFFu, RP1_SIGOP_ADD, 1, 0x00, 1, 0x1);
    make_rerun( &G_NODES[3], /* target */ 1,
                /* await */ 1, 0x1, /* set */ 1, 0x2);
    make_signal(&G_NODES[4], 10, 0xCAFEBABEu, RP1_SIGOP_SET, 0, 0x02, 0, 0x4);

    int rc = rp1_run(&s_hooks);
    CHECK_EQ32(rc, 0u, "loop: rp1_run rc");

    CHECK_EQ32(G_SIGS[0].value,        0u,          "loop: slot[0] reached 0");
    CHECK_EQ32(G_SIGS[10].value,       0xCAFEBABEu, "loop: finalize ran");
    CHECK_EQ32(g_loop_iters[0],        4u,          "loop: iteration counter");
    CHECK_EQ32(G_CTRL->cq_write_idx,   9u,          "loop: cq entries");
    CHECK_EQ32(G_CTRL->graph_done_seq, 1u,          "loop: graph_done_seq");
    return 0;
}

/* -------------------------------------------------------------------------
 * test_cond_boolean
 *
 *  Node 0: SIGNAL  slot=5 SET <test_value>
 *  Node 1: COND    cond: slot[5] EQ 42
 *                  body=[empty], bucket_clear=[empty]
 *                  set=0/0x10  (always)   done=0/0x20  (only on met)
 *  Node 2: SIGNAL  slot=20 SET 0xAAAA  await=0/0x10  (always)
 *  Node 3: SIGNAL  slot=21 SET 0xBBBB  await=0/0x20  (only on met)
 *
 * Avoids the if/else-via-body pattern from ARCHITECTURE.md § E (which
 * relies on body_clear + node_status reset to gate body execution and
 * isn't airtight when body-await masks are zero) and instead exercises
 * COND as a pure boolean: condition evaluation, the always-set
 * barrier_set_mask, and the conditional done_mask in done_bucket.
 *
 * Run twice — once with the condition met, once without — to confirm
 * both branches of the conditional are reachable from the same graph
 * template.
 * ---------------------------------------------------------------------- */

static int test_cond_boolean(void)
{
    /* ---- Run 1: condition met (slot[5] == 42) ---- */
    setup_graph(/* node_count */ 4, /* fake_kernels */ 0);

    make_signal(&G_NODES[0], 5, 42u, RP1_SIGOP_SET, 0, 0x00, 0, 0x1);
    make_cond(  &G_NODES[1],
                /* cond */ 5, RP1_COP_EQ, 42u,
                /* body */ 255, 0,   /* empty range */
                /* clear */ 255, 0,  /* empty range */
                /* done */ 0, 0x20,
                /* await */ 0, 0x1, /* set */ 0, 0x10);
    make_signal(&G_NODES[2], 20, 0xAAAAu, RP1_SIGOP_SET, 0, 0x10, 0, 0x40);
    make_signal(&G_NODES[3], 21, 0xBBBBu, RP1_SIGOP_SET, 0, 0x20, 0, 0x80);

    int rc = rp1_run(&s_hooks);
    CHECK_EQ32(rc, 0u, "cond[met]: rp1_run rc");
    CHECK_EQ32(G_SIGS[20].value,     0xAAAAu, "cond[met]: 'always' branch ran");
    CHECK_EQ32(G_SIGS[21].value,     0xBBBBu, "cond[met]: 'met-only' branch ran");
    CHECK_EQ32(G_CTRL->cq_write_idx, 4u,      "cond[met]: cq entries");

    /* ---- Run 2: condition NOT met (slot[5] == 99) ---- */
    setup_graph(4, 0);

    make_signal(&G_NODES[0], 5, 99u, RP1_SIGOP_SET, 0, 0x00, 0, 0x1);
    make_cond(  &G_NODES[1],
                5, RP1_COP_EQ, 42u,
                255, 0,
                255, 0,
                0, 0x20,
                0, 0x1, 0, 0x10);
    make_signal(&G_NODES[2], 20, 0xAAAAu, RP1_SIGOP_SET, 0, 0x10, 0, 0x40);
    make_signal(&G_NODES[3], 21, 0xBBBBu, RP1_SIGOP_SET, 0, 0x20, 0, 0x80);

    rc = rp1_run(&s_hooks);
    CHECK_EQ32(rc, 0u, "cond[nomet]: rp1_run rc");
    CHECK_EQ32(G_SIGS[20].value,     0xAAAAu, "cond[nomet]: 'always' branch ran");
    CHECK_EQ32(G_SIGS[21].value,     0u,      "cond[nomet]: 'met-only' silent");
    CHECK_EQ32(G_CTRL->cq_write_idx, 3u,      "cond[nomet]: cq entries");
    CHECK_EQ32(g_node_status[3],     RP1_NODE_PENDING,
               "cond[nomet]: node 3 stayed PENDING");
    return 0;
}

/* -------------------------------------------------------------------------
 * Runner
 * ---------------------------------------------------------------------- */

static int run(const char *name, int (*fn)(void))
{
    semi_puts(name);
    semi_puts(": ");
    int r = fn();
    if (r == 0) semi_puts("PASS\n");
    return r;
}

void rp1_graph_test_run(void)
{
    run("diamond_dag",    test_diamond_dag);
    run("signal_chain",   test_signal_chain);
    run("loop_decrement", test_loop_decrement);
    run("cond_boolean",   test_cond_boolean);
}

#endif /* QEMU_SEMIHOSTING */
