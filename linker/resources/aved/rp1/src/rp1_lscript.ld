/*
 * Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: MIT
 *
 * Linker script for RP1 baremetal firmware (ARM Cortex-R5 core 1).
 *
 * Memory regions use R5-1 local addresses:
 *   ATCM at 0x0      (64KB) - boot vectors + code
 *   BTCM at 0x20000  (64KB) - stack + data
 *   DDR  at 0x10000000      - shared data region (within AVED's free 864MB)
 */

_STACK_SIZE = 0x1000;   /* 4KB stack - sufficient for baremetal bringup */

MEMORY
{
    r5_1_atcm : ORIGIN = 0x00000000, LENGTH = 0x10000    /* 64KB */
    r5_1_btcm : ORIGIN = 0x00020000, LENGTH = 0x10000    /* 64KB */
    ddr_rp1   : ORIGIN = 0x10000000, LENGTH = 0x26000000 /* 608MB */
}

ENTRY(_boot)

SECTIONS
{
    .vectors : {
        KEEP(*(.vectors))
    } > r5_1_atcm

    .text : {
        *(.text)
        *(.text.*)
        *(.rodata)
        *(.rodata.*)
    } > r5_1_atcm

    .data : {
        __data_start = .;
        *(.data)
        *(.data.*)
        __data_end = .;
    } > r5_1_btcm

    /* Hot BTCM data: barriers, node status, loop counters, inflight table.
     * Explicitly placed before BSS so we know where they land. */
    .btcm (NOLOAD) : {
        . = ALIGN(4);
        *(.btcm)
        . = ALIGN(4);
    } > r5_1_btcm

    .bss (NOLOAD) : {
        . = ALIGN(4);
        __bss_start__ = .;
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(4);
        __bss_end__ = .;
    } > r5_1_btcm

    .stack (NOLOAD) : {
        . = ALIGN(16);
        _stack_end = .;
        . += _STACK_SIZE;
        . = ALIGN(16);
        __stack_top = .;
    } > r5_1_btcm

    /* Shared data region in DDR - used for heartbeat, future command queues */
    .rp1_shared (NOLOAD) : {
        *(.rp1_shared)
    } > ddr_rp1

    _end = .;
}
