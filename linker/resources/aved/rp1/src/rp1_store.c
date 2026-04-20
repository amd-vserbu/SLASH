/*
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * RP1 static storage definitions and initialisation.
 */

#include "rp1_store.h"
#include "rp1_types.h"
#include <stddef.h>

/* -------------------------------------------------------------------------
 * Compile-time size/offset assertions
 * ---------------------------------------------------------------------- */

/* C99 portable static assert via typedef trick (no <assert.h> on baremetal). */
#define STATIC_ASSERT(cond, name) \
    typedef char static_assert_##name[(cond) ? 1 : -1]

/* Node packet must be exactly 64 bytes. */
STATIC_ASSERT(sizeof(rp1_node_t) == 64,          node_size_64);

/* Header is 16 bytes; payload union starts at offset 16. */
STATIC_ASSERT(offsetof(rp1_node_t, payload) == 16, node_payload_offset_16);

/* Each payload variant must fit in the 48-byte payload union. */
STATIC_ASSERT(sizeof(rp1_payload_kernel_dispatch_t) == 48, kd_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_scalar_write_t)    == 48, sw_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_scalar_read_t)     == 48, sr_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_signal_t)          == 48, sig_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_dma_copy_t)        == 48, dma_copy_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_dma_fill_t)        == 48, dma_fill_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_loop_t)            == 48, loop_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_cond_t)            == 48, cond_payload_48);
STATIC_ASSERT(sizeof(rp1_payload_rerun_t)           == 48, rerun_payload_48);

/* Control block must be exactly 4 KB. */
STATIC_ASSERT(sizeof(rp1_ctrl_t) == 0x1000,      ctrl_size_4kb);

/* Signal slot must be 16 bytes. */
STATIC_ASSERT(sizeof(rp1_signal_slot_t) == 16,   signal_slot_16);

/* CQ entry must be 16 bytes. */
STATIC_ASSERT(sizeof(rp1_cq_entry_t) == 16,      cq_entry_16);

/* Inflight entry: 24 bytes (5 x uint32 + 1 byte + 3 pad = 24). */
STATIC_ASSERT(sizeof(rp1_inflight_t) == 24,       inflight_24);

/* -------------------------------------------------------------------------
 * BTCM-resident hot stores
 *
 * The .btcm attribute is used so the linker script can place these in the
 * BTCM region.  Under QEMU they land in BSS (zeroed by the boot stub).
 * ---------------------------------------------------------------------- */

#define BTCM_SECTION __attribute__((section(".btcm")))

uint32_t      g_barriers[RP1_MAX_BUCKETS]  BTCM_SECTION;
uint8_t       g_node_status[RP1_MAX_NODES] BTCM_SECTION;
uint32_t      g_loop_iters[RP1_MAX_LOOPS]  BTCM_SECTION;
rp1_inflight_t g_inflight[RP1_MAX_INFLIGHT] BTCM_SECTION;
uint32_t      g_inflight_count             BTCM_SECTION;

/* -------------------------------------------------------------------------
 * DDR-backed pointer table (set by rp1_store_init)
 * ---------------------------------------------------------------------- */

rp1_ctrl_t       *g_ctrl    = (rp1_ctrl_t *)0x10000000UL;
rp1_node_t       *g_nodes   = NULL;
rp1_cq_entry_t   *g_cq      = NULL;
rp1_signal_slot_t *g_signals = NULL;
uint32_t         *g_arg_buf  = NULL;

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

static void memzero(void *dst, uint32_t len)
{
    uint8_t *p = (uint8_t *)dst;
    while (len--)
        *p++ = 0;
}

static uint64_t make64(uint32_t lo, uint32_t hi)
{
    return ((uint64_t)hi << 32) | (uint64_t)lo;
}

/* -------------------------------------------------------------------------
 * Public API
 * ---------------------------------------------------------------------- */

void rp1_store_init(void)
{
    /* Resolve DDR pointers from control block fields. */
    g_nodes   = (rp1_node_t *)
                    (uintptr_t)make64(g_ctrl->node_base_lo, g_ctrl->node_base_hi);
    g_cq      = (rp1_cq_entry_t *)
                    (uintptr_t)make64(g_ctrl->cq_base_lo, g_ctrl->cq_base_hi);
    g_signals = (rp1_signal_slot_t *)
                    (uintptr_t)make64(g_ctrl->sig_array_base_lo, g_ctrl->sig_array_base_hi);
    g_arg_buf = (uint32_t *)
                    (uintptr_t)make64(g_ctrl->arg_buf_base_lo, g_ctrl->arg_buf_base_hi);

    rp1_store_reset_graph();
}

void rp1_store_reset_graph(void)
{
    memzero(g_barriers,    sizeof(g_barriers));
    memzero(g_node_status, sizeof(g_node_status));
    memzero(g_loop_iters,  sizeof(g_loop_iters));
    memzero(g_inflight,    sizeof(g_inflight));
    g_inflight_count = 0;
}
