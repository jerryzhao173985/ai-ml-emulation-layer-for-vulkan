/*
 * SPDX-FileCopyrightText: Copyright 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
 * SPDX-License-Identifier: Apache-2.0
 *
 */

/*******************************************************************************
 * Includes
 *******************************************************************************/

#include "tensor_log.hpp"

namespace mlsdk::el::log {

/*****************************************************************************
 * Tensorlog
 *****************************************************************************/

log::Log tensorLog("VMEL_TENSOR_SEVERITY", "Tensor");
} // namespace mlsdk::el::log
