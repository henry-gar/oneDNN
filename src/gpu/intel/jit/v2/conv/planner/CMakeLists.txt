#===============================================================================
# Copyright 2023-2025 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#===============================================================================

file(GLOB SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/*.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
    )

set(planner_main "${CMAKE_CURRENT_SOURCE_DIR}/planner_main.cpp")
list(REMOVE_ITEM SOURCES ${planner_main})

set(OBJ_LIB ${LIB_PACKAGE_NAME}_gpu_intel_jit_v2_conv_planner)
add_library(${OBJ_LIB} OBJECT ${SOURCES})

set_property(GLOBAL APPEND PROPERTY DNNL_LIB_DEPS
    $<TARGET_OBJECTS:${OBJ_LIB}>)

add_executable(gpu_conv_planner EXCLUDE_FROM_ALL ${planner_main})
target_link_libraries(gpu_conv_planner ${LIB_PACKAGE_NAME})
