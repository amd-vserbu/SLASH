######################################################################################
# This file is part of the Coyote <https://github.com/fpgasystems/Coyote>
#
# MIT Licence
######################################################################################

cmake_minimum_required(VERSION 3.5)

get_filename_component(CYT_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../../submodules/Coyote" REALPATH)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CYT_DIR}/cmake)
find_package(CoyoteHW REQUIRED)

project(example_07_p2p_coyote_hw)
message("*** Coyote Example 07: SLASH BAR P2P Control [Hardware] ***")

set(EN_STRM 1)
set(N_REGIONS 1)

validation_checks_hw()

load_apps (
    VFPGA_C0_0 "src"
)

create_hw()
