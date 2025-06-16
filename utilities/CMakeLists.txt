# SPDX-FileCopyrightText: Copyright 2023-2024 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

###############################################################################
# Utilities library
###############################################################################

add_library(mlelutilities STATIC)

target_sources(mlelutilities PRIVATE
    device.cpp
    pipeline.cpp
    tensor.cpp)

target_include_directories(mlelutilities PUBLIC include)

target_link_libraries(mlelutilities PUBLIC
    VkLayer_Log
    Vulkan::Headers
    SPIRV-Tools
    glslang::glslang
    glslang::glslang-default-resource-limits
    glslang::SPIRV
    ${CMAKE_DL_LIBS})
