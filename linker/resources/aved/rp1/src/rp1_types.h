/*
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * RP1 HSA command processor — protocol types.
 *
 * All structures are fixed-size and naturally aligned so they can be placed
 * directly in DDR and accessed over AXI without padding surprises.  Sizes are
 * verified by static assertions in rp1_store.c.
 *
 * See ARCHITECTURE.md for the full specification.
 */

#ifndef RP1_TYPES_H
#define RP1_TYPES_H

#include <stdint.h>

/* -------------------------------------------------------------------------
 * Opcodes
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_OP_NOP             = 0x0000,
    RP1_OP_SIGNAL          = 0x0002,
    RP1_OP_KERNEL_DISPATCH = 0x0010,
    RP1_OP_SCALAR_WRITE    = 0x0011,
    RP1_OP_SCALAR_READ     = 0x0012,
    RP1_OP_DMA_COPY        = 0x0020,
    RP1_OP_DMA_FILL        = 0x0021,
    RP1_OP_LOOP            = 0x0040,
    RP1_OP_COND            = 0x0041,
    RP1_OP_RERUN           = 0x0042,
    RP1_OP_HALT            = 0x00FF,
} rp1_opcode_t;

/* -------------------------------------------------------------------------
 * Universal node flags (rp1_node_t.flags)
 * ---------------------------------------------------------------------- */

#define RP1_FLAG_HALT_ON_ERROR  (1u << 0)
#define RP1_FLAG_SILENT         (1u << 1)
#define RP1_FLAG_INFINITE       (1u << 2)   /* KERNEL_DISPATCH: node DONE immediately */

/* -------------------------------------------------------------------------
 * Node status (written by RP1 into rp1_node_t.status)
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_NODE_PENDING    = 0x0000,
    RP1_NODE_DISPATCHED = 0x0001,
    RP1_NODE_DONE       = 0x0002,
    RP1_NODE_ERROR      = 0x00FF,
} rp1_node_status_t;

/* -------------------------------------------------------------------------
 * Condition operators (used by LOOP and COND)
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_COP_EQ     = 0,  /* signal == value  */
    RP1_COP_NE     = 1,  /* signal != value  */
    RP1_COP_LT     = 2,  /* signal <  value  */
    RP1_COP_GE     = 3,  /* signal >= value  */
    RP1_COP_AND_NZ = 4,  /* (signal & value) != 0 */
    RP1_COP_AND_Z  = 5,  /* (signal & value) == 0 */
} rp1_condop_t;

/* -------------------------------------------------------------------------
 * SIGNAL operation (rp1_payload_signal_t.operation)
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_SIGOP_SET = 0,
    RP1_SIGOP_ADD = 1,
    RP1_SIGOP_OR  = 2,
    RP1_SIGOP_AND = 3,
} rp1_sigop_t;

/* -------------------------------------------------------------------------
 * RERUN flags
 * ---------------------------------------------------------------------- */

#define RP1_RERUN_CLEAR_STATE  (1u << 0)    /* reset loop_iterations[loop_id] */

/* -------------------------------------------------------------------------
 * RP1 state (control block rp1_state field)
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_STATE_INIT    = 0,
    RP1_STATE_READY   = 1,
    RP1_STATE_RUNNING = 2,
    RP1_STATE_ERROR   = 3,
    RP1_STATE_HALTED  = 4,
} rp1_state_t;

/* -------------------------------------------------------------------------
 * Payload structures (each 48 bytes, embedded in rp1_node_t)
 * ---------------------------------------------------------------------- */

/* KERNEL_DISPATCH (0x0010) */
typedef struct {
    uint32_t kernel_base_addr;   /* AXI-Lite base in R5 address space        */
    uint32_t arg_buffer_offset;  /* Byte offset into argument buffer          */
    uint16_t arg_count;          /* Number of 32-bit argument words           */
    uint16_t ctrl_flags;         /* Bit 0: auto-restart                       */
    uint32_t timeout_cycles;     /* Watchdog (0 = default 10M cycles)         */
    uint8_t  _reserved[32];
} rp1_payload_kernel_dispatch_t;

/* SCALAR_WRITE (0x0011) — up to 6 register writes, stop at first addr == 0. */
typedef struct {
    uint32_t addr;
    uint32_t value;
} rp1_write_pair_t;

#define RP1_SCALAR_WRITE_MAX  6

typedef struct {
    rp1_write_pair_t writes[RP1_SCALAR_WRITE_MAX];
} rp1_payload_scalar_write_t;

/* SCALAR_READ (0x0012) */
typedef struct {
    uint32_t source_addr;    /* AXI-Lite address to read            */
    uint32_t target_slot;    /* Signal array slot index (0-255)     */
    uint8_t  _reserved[40];
} rp1_payload_scalar_read_t;

/* SIGNAL (0x0002) */
typedef struct {
    uint32_t target_slot;    /* Signal array slot index (0-255)     */
    uint32_t value;
    uint16_t operation;      /* rp1_sigop_t                         */
    uint16_t _reserved0;
    uint8_t  _reserved1[36];
} rp1_payload_signal_t;

/* DMA_COPY (0x0020) */
typedef struct {
    uint32_t src_addr_lo;
    uint32_t src_addr_hi;
    uint32_t dst_addr_lo;
    uint32_t dst_addr_hi;
    uint32_t length;
    uint16_t src_type;   /* 0=DDR, 1=HBM, 2=HOST */
    uint16_t dst_type;
    uint8_t  _reserved[24];
} rp1_payload_dma_copy_t;

/* DMA_FILL (0x0021) */
typedef struct {
    uint32_t dst_addr_lo;
    uint32_t dst_addr_hi;
    uint32_t length;
    uint32_t pattern;
    uint16_t dst_type;
    uint16_t _reserved0;
    uint8_t  _reserved1[28];
} rp1_payload_dma_fill_t;

/* LOOP (0x0040) */
typedef struct {
    uint32_t body_start;          /* First node index of loop body     */
    uint32_t body_end;            /* Last node index (inclusive)        */
    uint32_t max_iterations;      /* Hard cap (0 = condition-only)     */
    uint32_t condition_signal;    /* Signal array slot to check        */
    uint32_t condition_value;     /* Exit when signal matches          */
    uint16_t condition_op;        /* rp1_condop_t                      */
    uint8_t  bucket_clear_start;  /* First bucket to clear per iter    */
    uint8_t  bucket_clear_end;    /* Last bucket to clear (inclusive)  */
    uint8_t  loop_id;             /* Index into loop_iterations[]      */
    uint8_t  _reserved[23];
} rp1_payload_loop_t;

/* COND (0x0041) */
typedef struct {
    uint32_t condition_signal;    /* Signal slot to evaluate           */
    uint32_t condition_value;
    uint16_t condition_op;        /* rp1_condop_t                      */
    uint8_t  bucket_clear_start;  /* First bucket to clear (inclusive) */
    uint8_t  bucket_clear_end;    /* Last bucket to clear (inclusive)  */
    uint32_t body_start;          /* First node index (inclusive)      */
    uint32_t body_end;            /* Last node index (inclusive)       */
    uint8_t  done_bucket;
    uint8_t  _reserved0[3];
    uint32_t done_mask;
    uint8_t  _reserved1[20];
} rp1_payload_cond_t;

/* RERUN (0x0042) */
typedef struct {
    uint32_t target_node;    /* Node index to reset DONE -> PENDING  */
    uint16_t rerun_flags;    /* RP1_RERUN_CLEAR_STATE                */
    uint8_t  loop_id;        /* Loop ID to clear (if CLEAR_STATE)    */
    uint8_t  _reserved0[1];
    uint8_t  _reserved1[40];
} rp1_payload_rerun_t;

/* -------------------------------------------------------------------------
 * Node packet — 64 bytes, 16-byte header + 48-byte payload
 * ---------------------------------------------------------------------- */

typedef struct {
    /* Header (16 bytes) */
    uint16_t opcode;               /* rp1_opcode_t                          */
    uint16_t flags;                /* RP1_FLAG_*                            */
    uint32_t barrier_await_mask;   /* Which bits in await_bucket must be set */
    uint32_t barrier_set_mask;     /* Which bits in set_bucket to raise      */
    uint8_t  barrier_await_bucket; /* Which of 32 buckets to check (0-31)   */
    uint8_t  barrier_set_bucket;   /* Which of 32 buckets to write (0-31)   */
    uint16_t status;               /* rp1_node_status_t, written by RP1     */

    /* Payload (48 bytes) */
    union {
        rp1_payload_kernel_dispatch_t kernel_dispatch;
        rp1_payload_scalar_write_t    scalar_write;
        rp1_payload_scalar_read_t     scalar_read;
        rp1_payload_signal_t          signal;
        rp1_payload_dma_copy_t        dma_copy;
        rp1_payload_dma_fill_t        dma_fill;
        rp1_payload_loop_t            loop;
        rp1_payload_cond_t            cond;
        rp1_payload_rerun_t           rerun;
        uint8_t raw[48];
    } payload;
} rp1_node_t;

/* -------------------------------------------------------------------------
 * Control block — 4KB DDR region, 0x1000_0000
 * ---------------------------------------------------------------------- */

#define RP1_CTRL_MAGIC  0x53515231UL  /* "SQR1" */

typedef struct {
    /* RP1 writes */
    volatile uint32_t magic;            /* 0x00: RP1_CTRL_MAGIC when ready       */
    volatile uint32_t version;          /* 0x04: protocol version                */
    /* Host writes, RP1 reads */
    volatile uint32_t node_count;       /* 0x08: nodes in this graph             */
    volatile uint32_t cq_size;          /* 0x0C: CQ entries (power of 2)         */
    volatile uint32_t node_base_lo;     /* 0x10: node array base (low 32 bits)   */
    volatile uint32_t node_base_hi;     /* 0x14: node array base (high 32 bits)  */
    volatile uint32_t cq_base_lo;       /* 0x18: CQ base (low 32 bits)           */
    volatile uint32_t cq_base_hi;       /* 0x1C: CQ base (high 32 bits)          */
    volatile uint32_t graph_seq;        /* 0x20: host increments per graph        */
    /* RP1 writes */
    volatile uint32_t graph_done_seq;   /* 0x24: last completed graph             */
    volatile uint32_t cq_write_idx;     /* 0x28: next CQ write position          */
    /* Host writes */
    volatile uint32_t cq_read_idx;      /* 0x2C: last consumed CQ position       */
    /* RP1 writes */
    volatile uint32_t rp1_state;        /* 0x30: rp1_state_t                     */
    volatile uint32_t rp1_error_code;   /* 0x34: last error code                 */
    volatile uint32_t rp1_current_node; /* 0x38: current node (debug)            */
    volatile uint32_t heartbeat;        /* 0x3C: liveness counter                */
    /* Host writes, RP1 reads */
    volatile uint32_t arg_buf_base_lo;  /* 0x40: argument buffer base            */
    volatile uint32_t arg_buf_base_hi;  /* 0x44                                  */
    volatile uint32_t sig_array_base_lo;/* 0x48: signal array base               */
    volatile uint32_t sig_array_base_hi;/* 0x4C                                  */
    uint8_t _reserved[0x1000 - 0x50];
} rp1_ctrl_t;

/* -------------------------------------------------------------------------
 * Signal array slot — 16 bytes, 256 slots in DDR
 * ---------------------------------------------------------------------- */

#define RP1_SIG_FLAG_HOST_VISIBLE  (1u << 0)

typedef struct {
    volatile uint32_t value;            /* Current 32-bit value               */
    volatile uint32_t _reserved;
    volatile uint32_t last_writer_node; /* Node that last wrote               */
    volatile uint32_t flags;            /* RP1_SIG_FLAG_*                     */
} rp1_signal_slot_t;

/* -------------------------------------------------------------------------
 * Completion queue entry — 16 bytes
 * ---------------------------------------------------------------------- */

typedef enum {
    RP1_CQ_OK      = 0,
    RP1_CQ_ERROR   = 1,
    RP1_CQ_TIMEOUT = 2,
} rp1_cq_status_t;

typedef struct {
    volatile uint32_t node_index;    /* Which node completed               */
    volatile uint32_t status;        /* rp1_cq_status_t                    */
    volatile uint32_t error_detail;  /* Opcode-specific error code         */
    volatile uint32_t timestamp;     /* R5 cycle counter at completion     */
} rp1_cq_entry_t;

/* -------------------------------------------------------------------------
 * In-flight kernel table entry (BTCM, max 32 entries)
 * ---------------------------------------------------------------------- */

typedef struct {
    uint32_t base_addr;        /* AXI-Lite base address in R5 space  */
    uint32_t node_index;
    uint32_t set_bucket;
    uint32_t set_mask;
    uint32_t timeout_remaining;
    uint8_t  infinite;         /* Non-zero: INFINITE flag set        */
    uint8_t  _reserved[3];
} rp1_inflight_t;

/* -------------------------------------------------------------------------
 * Compile-time size constants
 * ---------------------------------------------------------------------- */

#define RP1_MAX_NODES    4096
#define RP1_MAX_LOOPS      64
#define RP1_MAX_INFLIGHT   32
#define RP1_MAX_SIGNALS   256
#define RP1_MAX_BUCKETS    32

#define RP1_PROTOCOL_VERSION  1u

#endif /* RP1_TYPES_H */
