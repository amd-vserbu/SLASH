# ##################################################################################################
#  The MIT License (MIT)
#  Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# ##################################################################################################

cmake_minimum_required(VERSION 3.20)
project(07_p2p_coyote LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(SLASH_USE_REPO "Build against the local repo tree instead of installed packages" ON)
option(SLASH_ENABLE_FPGA_TARGETS "Enable HLS/vbin targets (requires Vivado/Vitis)" ON)

if(SLASH_USE_REPO)
  get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/../.." REALPATH)
  if(SLASH_ENABLE_FPGA_TARGETS)
    list(APPEND CMAKE_MODULE_PATH "${REPO_ROOT}/cmake")
    include(SlashTools)
  endif()

  add_subdirectory(${REPO_ROOT}/driver/libslash ${CMAKE_CURRENT_BINARY_DIR}/libslash)
  add_subdirectory(${REPO_ROOT}/submodules/Coyote/sw ${CMAKE_CURRENT_BINARY_DIR}/coyote_sw)

  set(_SLASH_LIB slash)
  set(_COYOTE_LIB Coyote)
else()
  if(SLASH_ENABLE_FPGA_TARGETS)
    find_package(SlashTools REQUIRED)
  endif()
  find_package(slash REQUIRED CONFIG)
  find_package(Coyote REQUIRED CONFIG)

  set(_SLASH_LIB slash::slash)
  set(_COYOTE_LIB Coyote)
endif()

if(SLASH_ENABLE_FPGA_TARGETS)
  # --- HLS kernel ---
  set(DEVICE "xcv80-lsva4737-2MHP-e-S" CACHE STRING "Target device")

  build_hls_dir(
    TARGET      hls
    ROOT        "${CMAKE_CURRENT_SOURCE_DIR}/hls"
    DEVICE      "${DEVICE}"
    KERNELS     slash_add
    OUT_KERNELS _KERNELS
  )

  # --- SLASH VBIN targets ---
  set(CFG_FILE "${CMAKE_CURRENT_SOURCE_DIR}/config.cfg")

  add_vbin(TARGET "p2p_coyote_hw"  PLATFORM "hw"  CFG "${CFG_FILE}" KERNELS ${_KERNELS})
  add_vbin(TARGET "p2p_coyote_sim" PLATFORM "sim" CFG "${CFG_FILE}" KERNELS ${_KERNELS})
  add_vbin(TARGET "p2p_coyote_emu" PLATFORM "emu" CFG "${CFG_FILE}" KERNELS ${_KERNELS})
endif()

# --- Host orchestrator app ---
add_executable(${PROJECT_NAME} 07_p2p_coyote.cpp)
target_link_libraries(${PROJECT_NAME} PRIVATE ${_SLASH_LIB} ${_COYOTE_LIB})
