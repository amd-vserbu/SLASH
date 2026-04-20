#!/usr/bin/env bash
# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Verify that RP1 firmware is running by reading the heartbeat from DDR.
# Usage: verify_rp1.sh [address] [bdf]
#   address: DDR address of RP1 heartbeat (default: 0x30000000)
#   bdf:     PCIe BDF of the device (default: auto-detect)
set -euo pipefail

ADDR=${1:-0x30000000}
EXPECTED_MAGIC="0x52503101"

BDF_ARGS=""
if [ -n "${2:-}" ]; then
    BDF_ARGS="--device $2"
fi

echo "Checking RP1 heartbeat at DDR address ${ADDR}..."

# Read 4 words: magic, counter, status, reserved
RESULT=$(v80-smi debug mem-poke ${BDF_ARGS} --region RAW --read --hex "${ADDR}" --word-size 4 --count 4 2>&1) || {
    echo "ERROR: Failed to read memory at ${ADDR}"
    echo "${RESULT}"
    exit 1
}

echo "Raw read: ${RESULT}"

# Read counter, wait, read again to check liveness
C1=$(v80-smi debug mem-poke ${BDF_ARGS} --region RAW --read --hex "$((ADDR + 4))" --word-size 4 --count 1 2>&1) || true
sleep 1
C2=$(v80-smi debug mem-poke ${BDF_ARGS} --region RAW --read --hex "$((ADDR + 4))" --word-size 4 --count 1 2>&1) || true

if [ "${C1}" != "${C2}" ]; then
    echo "PASS: RP1 counter is incrementing (${C1} -> ${C2})"
else
    echo "WARN: RP1 counter unchanged (${C1}). Firmware may be stuck or not running."
fi
