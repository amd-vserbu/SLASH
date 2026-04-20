# Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
# CMake build for RP1 baremetal firmware (ARM Cortex-R5 core 1).

cmake_minimum_required(VERSION 3.5.0)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR arm)
set(CMAKE_CROSSCOMPILING TRUE)
set(CMAKE_C_COMPILER arm-none-eabi-gcc)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_C_COMPILER_WORKS TRUE)
set(CMAKE_ASM_COMPILER_WORKS TRUE)

project(rp1 C ASM)

set(CPU_FLAGS "-mcpu=cortex-r5 -mfloat-abi=soft")
set(CMAKE_C_FLAGS "${CPU_FLAGS} -Wall -Werror -O2 -ffreestanding -nostdlib")
set(CMAKE_ASM_FLAGS "${CPU_FLAGS}")

option(QEMU_SEMIHOSTING "Enable ARM semihosting for QEMU testing" OFF)

set(LINKER_SCRIPT "${CMAKE_SOURCE_DIR}/src/rp1_lscript.ld")

# Common sources: boot stub + stores + dispatch loop + outer loop.
set(RP1_SOURCES
    src/rp1_boot.S
    src/rp1_store.c
    src/rp1_loop.c
    src/rp1_run.c
)

# In semihosting mode rp1_test.c provides rp1_main(); otherwise rp1_main.c does.
if(QEMU_SEMIHOSTING)
    list(APPEND RP1_SOURCES src/rp1_test.c)
else()
    list(APPEND RP1_SOURCES src/rp1_main.c)
endif()

add_executable(rp1.elf ${RP1_SOURCES})

if(QEMU_SEMIHOSTING)
    target_compile_definitions(rp1.elf PRIVATE QEMU_SEMIHOSTING)
endif()

target_link_libraries(rp1.elf
    "-T${LINKER_SCRIPT}"
    "-nostdlib"
    "-lgcc"
)

set_target_properties(rp1.elf PROPERTIES
    LINK_DEPENDS "${LINKER_SCRIPT}"
)
