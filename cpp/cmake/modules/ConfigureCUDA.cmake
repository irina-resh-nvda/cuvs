# =============================================================================
# cmake-format: off
# SPDX-FileCopyrightText: Copyright (c) 2018-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# cmake-format: on
# =============================================================================

if(DISABLE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_CXX_FLAGS -Wno-deprecated-declarations -DRAFT_HIDE_DEPRECATION_WARNINGS)
  list(APPEND CUVS_CUDA_FLAGS -Xcompiler=-Wno-deprecated-declarations
       -DRAFT_HIDE_DEPRECATION_WARNINGS
  )
endif()

if(DISABLE_OPENMP)
  list(APPEND CUVS_CXX_FLAGS -Wno-unknown-pragmas)
  list(APPEND CUVS_CUDA_FLAGS -Xcompiler=-Wno-unknown-pragmas)
endif()

# Be very strict when compiling with GCC as host compiler (and thus more lenient when compiling with
# clang)
if(CMAKE_COMPILER_IS_GNUCXX)
  # unused-but-set-variable: raft 26.10 changed `get_cuda_stream` to return the header-only
  # `cuda::stream_ref`, so GCC can now prove that initializing a `stream` local has no side effects
  # and flags every one that went unused. Around 130 such locals exist across cpp/src, predating the
  # change. Downgraded to a warning for the same reason deprecated-declarations is.
  list(APPEND CUVS_CXX_FLAGS -Wall -Werror -Wno-unknown-pragmas -Wno-error=deprecated-declarations
       -Wno-error=unused-but-set-variable -Wno-reorder
  )
  list(APPEND CUVS_CUDA_FLAGS
       -Xcompiler=-Wall,-Werror,-Wno-error=deprecated-declarations,-Wno-error=unused-but-set-variable,-Wno-reorder
  )

  # set warnings as errors
  if(CMAKE_CUDA_COMPILER_VERSION VERSION_GREATER_EQUAL 11.2.0)
    list(APPEND CUVS_CUDA_FLAGS -Werror=all-warnings)
  endif()
endif()

if(CUDA_LOG_COMPILE_TIME)
  list(APPEND CUVS_CUDA_FLAGS "--time=nvcc_compile_log.csv")
endif()

list(APPEND CUVS_CUDA_FLAGS --expt-extended-lambda --expt-relaxed-constexpr)
list(APPEND CUVS_CXX_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
list(APPEND CUVS_CUDA_FLAGS "-DCUDA_API_PER_THREAD_DEFAULT_STREAM")
# make sure we produce smallest binary size
include(${rapids-cmake-dir}/cuda/enable_fatbin_compression.cmake)
rapids_cuda_enable_fatbin_compression(VARIABLE CUVS_CUDA_FLAGS TUNE_FOR rapids)

# Option to enable line info in CUDA device compilation to allow introspection when profiling /
# memchecking
if(CUDA_ENABLE_LINEINFO)
  list(APPEND CUVS_CUDA_FLAGS -lineinfo)
endif()

if(OpenMP_FOUND)
  list(APPEND CUVS_CUDA_FLAGS -Xcompiler=${OpenMP_CXX_FLAGS})
endif()

# Debug options
list(APPEND CUVS_DEBUG_CUDA_FLAGS -G -Xcompiler=-rdynamic --maxrregcount=64)
list(APPEND CUVS_DEBUG_CUDA_FLAGS -Xptxas --suppress-stack-size-warning)
