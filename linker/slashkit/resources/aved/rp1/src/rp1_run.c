/*
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * RP1 outer loop — polls graph_seq for new submissions, resets state,
 * runs the flat scanner via rp1_loop(), and updates graph_done_seq.
 *
 * See ARCHITECTURE.md section D (rp1_main pseudocode).
 */

#include "rp1_run.h"
#include "rp1_cycles.h"
#include "rp1_hal.h"
#include "rp1_store.h"
#include <slash/uapi/rp1_protocol.h>
#include <stdint.h>

#ifdef QEMU_SEMIHOSTING
int rp1_run(const rp1_hooks_t *hooks)
#else
int rp1_run(void)
#endif
{
#ifdef QEMU_SEMIHOSTING
    int terminal_result = 0;
#endif

    rp1_pmu_init();

    /*
     * Startup publishes in dependency order: invalidate magic, initialize the
     * complete contract and READY state, barrier, then publish magic last.
     * Hosts may therefore treat visible magic as proof the fixed fields exist.
     */
    g_ctrl->magic                 = 0;
    g_ctrl->rp1_state             = RP1_STATE_INIT;
    g_ctrl->version               = RP1_PROTOCOL_VERSION;
    g_ctrl->capabilities          = RP1_REQUIRED_CAPABILITIES;
    g_ctrl->pdi_ipi_platform_id   = RP1_PLATFORM_ID;
    rp1_clear_error_latch();
    g_ctrl->graph_done_seq        = 0;
    g_ctrl->cq_write_idx          = 0;
    g_ctrl->heartbeat             = 0;
    g_ctrl->rp1_state             = RP1_STATE_READY;
    rp1_barrier();
    g_ctrl->magic = RP1_CTRL_MAGIC;
    rp1_barrier();

    /*
     * The outer loop has three cases: terminal firmware remains quiescent,
     * equal sequences mean idle, and a new sequence starts one accepted graph.
     * Equality, rather than ordering, keeps uint32_t sequence wrap harmless.
     */
    for (;;) {
        if (g_ctrl->rp1_state == RP1_STATE_ERROR ||
            g_ctrl->rp1_state == RP1_STATE_HALTED) {
            /* Terminal states are reset-only. A later graph_seq must never
             * reactivate work against an unproven hardware state. */
            g_ctrl->heartbeat++;
#ifdef QEMU_SEMIHOSTING
            if (hooks && hooks->on_idle && hooks->on_idle())
                return terminal_result;
#endif
            continue;
        }

        if (g_ctrl->graph_seq == g_ctrl->graph_done_seq) {
            g_ctrl->heartbeat++;

#ifdef QEMU_SEMIHOSTING
            if (hooks && hooks->on_idle) {
                if (hooks->on_idle())
                    return 0;
            }
#endif
            continue;
        }

        uint32_t accepted_seq = g_ctrl->graph_seq;
        uint32_t config_detail = 0u;
        uint32_t config_aux = 0u;
        int store_ready = 0;

        /*
         * Phase 1: latch the accepted sequence, enter RUNNING, then validate
         * every host-owned range before resolving or dereferencing DDR pointers.
         */
        rp1_clear_error_latch();
        g_ctrl->rp1_current_node = RP1_TERMINAL_ERROR_NODE_NONE;
        g_ctrl->rp1_state = RP1_STATE_RUNNING;
        rp1_barrier();
        if (rp1_store_init(&config_detail, &config_aux) != 0) {
            rp1_latch_error(RP1_ERR_INVALID_CONFIG,
                            RP1_TERMINAL_ERROR_NODE_NONE,
                            config_detail, config_aux);
        } else {
            store_ready = 1;
        }
        g_graph_start_cycles = rp1_cycles();
        if (store_ready)
            rp1_trace_emit(RP1_TRACE_GRAPH_START, 0xFFFFu,
                           accepted_seq, g_ctrl->node_count);

        /*
         * Phase 2: run only a fully initialized store. Configuration failure
         * skips the scanner so an invalid host pointer can have no side effect.
         */
        int result;
        if (!store_ready) {
            result = -1;
        } else {
#ifdef QEMU_SEMIHOSTING
            result = rp1_loop(hooks);
#else
            result = rp1_loop();
#endif
        }

        if (store_ready) {
            rp1_trace_emit(RP1_TRACE_GRAPH_DONE, 0xFFFFu,
                           (uint32_t)result, accepted_seq);
            /*
             * Publish the final partial BTCM page after graph work ends.
             * Periodic flushes are measured by FLUSH_START/END; this terminal
             * drain intentionally adds no recursive marker pair.
             */
            rp1_trace_flush_final();
        }

        /*
         * Phase 3: classify scanner completion without clearing its first-error
         * record; ERROR and HALTED remain reset-only in the next outer pass.
         */
        uint32_t terminal_state;
        if (result == -1)
            terminal_state = RP1_STATE_ERROR;
        else if (result == -2)
            terminal_state = RP1_STATE_HALTED;
        else
            terminal_state = RP1_STATE_READY;

        /*
         * Publish CQ, trace, and the error latch before terminal state; publish
         * state before graph_done_seq. Exact sequence completion is therefore
         * the host's release point for every result belonging to this graph.
         */
        rp1_barrier();
        g_ctrl->rp1_state = terminal_state;
        rp1_barrier();
        g_ctrl->graph_done_seq = accepted_seq;
        rp1_barrier();
#ifdef QEMU_SEMIHOSTING
        terminal_result = result;
#endif

#ifdef QEMU_SEMIHOSTING
        if (hooks && hooks->on_graph_done) {
            if (hooks->on_graph_done(result))
                return result;
        }
#endif
        /* TODO: ring GCQ doorbell (S01_AXI/0x000) when block design is wired. */
    }
}
