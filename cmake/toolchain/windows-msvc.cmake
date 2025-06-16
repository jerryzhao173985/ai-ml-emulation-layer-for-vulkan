#
# SPDX-FileCopyrightText: Copyright 2025 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0
#

# Set warnings as errors and enable all warnings
# Disable warning C4324 for padding
# Disable warning C4996 for potentially unsafe functions, for example 'getenv'
set(VMEL_COMPILE_OPTIONS /EHa /WX /W4 /wd4324 /wd4996)
