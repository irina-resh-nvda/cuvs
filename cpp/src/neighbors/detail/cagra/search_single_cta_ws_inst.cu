/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "search_single_cta_ws_kernel.cuh"

#include <raft/core/error.hpp>
#include <raft/core/logger.hpp>
#include <raft/util/cuda_rt_essentials.hpp>

#include <algorithm>
#include <atomic>
#include <map>
#include <mutex>
#include <utility>

/**
 * The only translation unit that compiles the warp-specialized CAGRA search kernel. It is built for
 * SM 100+ exclusively (see `CMakeLists.txt`), which is why the kernel cannot live in one of the
 * generated instantiation files.
 *
 * Instantiated for float data / uint32_t indices / float distances, over the metrics and
 * (dataset_block_dim, team_size) pairs that the standard dataset descriptor provides.
 */
namespace cuvs::neighbors::cagra::detail::single_cta_ws_search {

namespace {

/**
 * Opt the kernel into more than 48 KiB of dynamic shared memory. The attribute is per kernel and
 * sticky, so it is raised at most once per instantiation and high-water mark.
 */
template <auto Kernel>
void ensure_max_dynamic_smem(uint32_t smem_size)
{
  static std::atomic<uint32_t> requested{0};
  uint32_t previous = requested.load(std::memory_order_relaxed);
  while (smem_size > previous) {
    if (requested.compare_exchange_weak(
          previous, smem_size, std::memory_order_relaxed, std::memory_order_relaxed)) {
      RAFT_CUDA_TRY(cudaFuncSetAttribute(reinterpret_cast<const void*>(Kernel),
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         static_cast<int>(smem_size)));
      return;
    }
  }
}

/**
 * Blocks per SM the real search kernel reaches at this launch configuration.
 *
 * Every instantiation compiles to the same 64 registers per thread, because the launch bounds have to
 * admit 1024-thread blocks, and the JIT SINGLE_CTA kernel is built the same way. One instantiation
 * therefore stands in for all of them.
 */
template <typename DataT, typename IndexT, typename DistanceT>
auto search_blocks_per_sm(uint32_t block_size, uint32_t smem_size) -> int
{
  constexpr auto kKernel = &search_single_cta_ws_kernel<cuvs::distance::DistanceType::L2Expanded,
                                                       32,
                                                       512,
                                                       DataT,
                                                       IndexT,
                                                       DistanceT>;
  ensure_max_dynamic_smem<kKernel>(smem_size);
  int blocks = 0;
  RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
    &blocks, kKernel, static_cast<int>(block_size), smem_size));
  return blocks;
}

/**
 * Dynamic shared memory that makes the no-op kernel exactly as resident as the real one.
 *
 * An empty body needs almost no registers, so left alone the no-op fits twice as many blocks per SM
 * as the search kernel, whose 64 registers per thread cap it at half the thread ceiling. Then it would
 * drain a grid faster than any real kernel could and understate the floor it is meant to measure.
 * Registers cannot be padded from the host side, but shared memory can, and it is the other resource
 * the search kernel reserves anyway: ask for enough of it to be held to the same number of blocks.
 *
 * Occupancy falls monotonically as shared memory grows, so the smallest size that reaches the target
 * can be bisected. When the search kernel is already shared-memory bound, nothing is added.
 */
template <typename DataT, typename IndexT, typename DistanceT>
auto noop_smem_bytes(uint32_t block_size, uint32_t smem_size) -> uint32_t
{
  constexpr auto kNoopKernel = &search_noop_kernel<DataT, IndexT, DistanceT>;

  static std::mutex mutex;
  static std::map<std::pair<uint32_t, uint32_t>, uint32_t> cache;
  const std::lock_guard<std::mutex> lock{mutex};
  const auto key = std::make_pair(block_size, smem_size);
  if (const auto it = cache.find(key); it != cache.end()) { return it->second; }

  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int max_smem_per_block = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(
    &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
  // Raised once to the device maximum: the attribute only permits larger launches, while occupancy
  // still follows the size each launch actually requests.
  ensure_max_dynamic_smem<kNoopKernel>(static_cast<uint32_t>(max_smem_per_block));

  auto blocks_at = [&](uint32_t bytes) {
    int blocks = 0;
    RAFT_CUDA_TRY(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, kNoopKernel, static_cast<int>(block_size), bytes));
    return blocks;
  };

  const int target = search_blocks_per_sm<DataT, IndexT, DistanceT>(block_size, smem_size);
  uint32_t bytes   = smem_size;
  if (target > 0 && blocks_at(bytes) > target) {
    uint32_t low  = smem_size;
    uint32_t high = std::max<uint32_t>(smem_size, static_cast<uint32_t>(max_smem_per_block));
    while (low < high) {
      const uint32_t mid = low + (high - low) / 2;
      if (blocks_at(mid) > target) {
        low = mid + 1;
      } else {
        high = mid;
      }
    }
    bytes = low;
  }

  const int reached = blocks_at(bytes);
  if (reached == target) {
    RAFT_LOG_INFO(
      "CAGRA SINGLE_CTA_NOOP: %u threads/block, %u B of shared memory padded to %u B to match the "
      "search kernel's %d blocks/SM",
      block_size,
      smem_size,
      bytes,
      target);
  } else {
    RAFT_LOG_WARN(
      "CAGRA SINGLE_CTA_NOOP: %d blocks/SM at %u B of shared memory, against %d for the search "
      "kernel; the no-op launch is more resident than what it measures",
      reached,
      bytes,
      target);
  }
  cache.emplace(key, bytes);
  return bytes;
}

/** Calls `f.operator()<Metric>()` for the metrics the kernel implements. */
template <typename F>
void dispatch_metric(cuvs::distance::DistanceType metric, F f)
{
  using cuvs::distance::DistanceType;
  switch (metric) {
    case DistanceType::L2Expanded: return f.template operator()<DistanceType::L2Expanded>();
    case DistanceType::InnerProduct: return f.template operator()<DistanceType::InnerProduct>();
    case DistanceType::CosineExpanded: return f.template operator()<DistanceType::CosineExpanded>();
    case DistanceType::L1: return f.template operator()<DistanceType::L1>();
    default:
      RAFT_FAIL("CAGRA SINGLE_CTA_WS search does not support metric %d",
                static_cast<int>(metric));
  }
}

/** Calls `f.operator()<TeamSize, DatasetBlockDim>()` for the descriptor's configuration. */
template <typename F>
void dispatch_team_and_block_dim(uint32_t team_size, uint32_t dataset_block_dim, F f)
{
  if (dataset_block_dim == 128 && team_size == 8) { return f.template operator()<8, 128>(); }
  if (dataset_block_dim == 256 && team_size == 16) { return f.template operator()<16, 256>(); }
  if (dataset_block_dim == 512 && team_size == 32) { return f.template operator()<32, 512>(); }
  RAFT_FAIL(
    "CAGRA SINGLE_CTA_WS search does not support the combination team_size = %u, "
    "dataset_block_dim = %u",
    team_size,
    dataset_block_dim);
}

}  // namespace

template <typename DataT, typename IndexT, typename DistanceT>
auto launcher<DataT, IndexT, DistanceT>::total_smem_bytes(uint32_t cagra_smem_bytes) -> uint32_t
{
  return ws_total_smem_bytes(cagra_smem_bytes);
}

template <typename DataT, typename IndexT, typename DistanceT>
void launcher<DataT, IndexT, DistanceT>::check_device_support()
{
  int device = 0;
  RAFT_CUDA_TRY(cudaGetDevice(&device));
  int major = 0;
  int minor = 0;
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device));
  RAFT_CUDA_TRY(cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device));
  RAFT_EXPECTS(major >= 10,
               "CAGRA SINGLE_CTA_WS search requires a compute capability 10.0+ device (found %d.%d)",
               major,
               minor);
}

template <typename DataT, typename IndexT, typename DistanceT>
void launcher<DataT, IndexT, DistanceT>::run(const launch_args<DataT, IndexT, DistanceT>& args,
                                             cuvs::distance::DistanceType metric,
                                             uint32_t team_size,
                                             uint32_t dataset_block_dim,
                                             uint32_t block_size,
                                             uint32_t smem_size,
                                             cudaStream_t stream)
{
  dispatch_metric(metric, [&]<cuvs::distance::DistanceType Metric>() {
    dispatch_team_and_block_dim(
      team_size, dataset_block_dim, [&]<uint32_t TeamSize, uint32_t DatasetBlockDim>() {
        ensure_max_dynamic_smem<&search_single_cta_ws_kernel<Metric,
                                                            TeamSize,
                                                            DatasetBlockDim,
                                                            DataT,
                                                            IndexT,
                                                            DistanceT>>(smem_size);
        search_single_cta_ws_kernel<Metric, TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>
          <<<dim3(args.num_queries, 1, 1), dim3(block_size, 1, 1), smem_size, stream>>>(args);
        RAFT_CUDA_TRY(cudaPeekAtLastError());
      });
  });
}

template <typename DataT, typename IndexT, typename DistanceT>
void launcher<DataT, IndexT, DistanceT>::run_noop(
  const launch_args<DataT, IndexT, DistanceT>& args,
  uint32_t block_size,
  uint32_t smem_size,
  cudaStream_t stream)
{
  const auto smem_bytes = noop_smem_bytes<DataT, IndexT, DistanceT>(block_size, smem_size);
  search_noop_kernel<DataT, IndexT, DistanceT>
    <<<dim3(args.num_queries, 1, 1), dim3(block_size, 1, 1), smem_bytes, stream>>>(args);
  RAFT_CUDA_TRY(cudaPeekAtLastError());
}

template struct launcher<float, uint32_t, float>;

}  // namespace cuvs::neighbors::cagra::detail::single_cta_ws_search
