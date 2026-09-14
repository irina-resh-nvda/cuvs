/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compute_distance_standard-impl.cuh"
#include "hashmap.hpp"
#include "search_single_cta_ws.cuh"
#include "utils.hpp"

#include "jit_lto_kernels/apply_normalization_standard_impl.cuh"
#include "jit_lto_kernels/dist_op_impl.cuh"
#include "jit_lto_kernels/search_single_cta_device_helpers.cuh"

#include "../ann_utils.cuh"
#include "../neighbors_device_intrinsics.cuh"

#include <raft/core/detail/macros.hpp>
#include <raft/core/operators.hpp>
#include <raft/util/cudart_utils.hpp>
#include <raft/util/integer_utils.hpp>
#include <raft/util/pow2_utils.cuh>
#include <raft/util/warp_primitives.cuh>

#include <cub/detail/warpspeed/warpspeed.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cuda/ptx>

#include <cstdint>

/**
 * Device side of the warp-specialized single-CTA search. Included only by
 * `search_single_cta_ws_inst.cu`, which is compiled for SM 100+ exclusively.
 *
 * The per-query body deliberately mirrors the JIT `single_cta_search::search_core` so the two can
 * be compared directly. The device functions the JIT kernels resolve through LTO (`dist_op`,
 * `compute_distance`, `setup_workspace`) are not linkable from a normally compiled translation
 * unit, so the distance path is re-templated on the metric here; the parts that do not depend on
 * those symbols (bitonic top-k, parent selection, hash-table maintenance, workspace setup) are
 * reused as they are.
 *
 * The grid is still one block per query, but instead of retiring, a block claims a block index the
 * grid has not started yet (`clusterlaunchcontrol.try_cancel`) and continues with that query. The
 * claim is issued before the current query is searched, so its latency is hidden behind the search,
 * and it is what turns the tail of a long grid into work for the blocks that are already resident.
 */
namespace cuvs::neighbors::cagra::detail::single_cta_ws_search {

namespace ws = cub::detail::warpspeed;

using single_cta_search::hashmap_restore;
using single_cta_search::pickup_next_parents;
using single_cta_search::topk_by_bitonic_sort_and_merge;

/**
 * Stride used when initializing the warpspeed barriers. It must be at least the block size so that
 * no two threads initialize the same barrier; the largest block we launch is 1024 threads.
 */
constexpr int kMaxBlockThreads = 1024;

/** Alignment of the classic CAGRA shared-memory region (the descriptor needs 16 bytes). */
constexpr uint32_t kCagraSmemAlign = 16;

/**
 * Stages of the next-block-index resource. One stage means the block claims its next query while it
 * works on the current one, i.e. a single `try_cancel` is in flight at a time.
 */
constexpr int kNumStagesNextBlockIdx = 1;

static_assert(sizeof(cub::WarpScan<unsigned>::TempStorage) <= kSmemWorkBytes,
              "kSmemWorkBytes must cover the WarpScan temp storage used by the result compaction");

/** The descriptor layout this kernel understands: strided dataset, float query in shared memory. */
template <uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
using ws_descriptor_t =
  standard_dataset_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT, float>;

/** Metric-templated counterpart of the LTO-resolved `dist_op`. */
template <cuvs::distance::DistanceType Metric, typename QueryT, typename DistanceT>
RAFT_DEVICE_INLINE_FUNCTION auto ws_dist_op(QueryT a, QueryT b) -> DistanceT
{
  if constexpr (Metric == cuvs::distance::DistanceType::L2Expanded ||
                Metric == cuvs::distance::DistanceType::L2Unexpanded) {
    return dist_op_l2_impl<QueryT, DistanceT>(a, b);
  } else if constexpr (Metric == cuvs::distance::DistanceType::InnerProduct ||
                       Metric == cuvs::distance::DistanceType::CosineExpanded) {
    // As in the JIT planner, cosine reuses the inner-product op and divides by the dataset norm.
    return dist_op_inner_product_impl<QueryT, DistanceT>(a, b);
  } else if constexpr (Metric == cuvs::distance::DistanceType::L1) {
    return dist_op_l1_impl<QueryT, DistanceT>(a, b);
  } else {
    static_assert(sizeof(QueryT) == 0, "ws_dist_op: unsupported metric");
    return DistanceT{};
  }
}

/**
 * Per-thread partial distance between the query in shared memory and one dataset row.
 *
 * Copy of `compute_distance_standard_worker_impl` with the metric bound at compile time. NB: the
 * team lane comes from `threadIdx.x`, which holds as long as the distance squad starts at warp 0.
 */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto ws_compute_distance_worker(
  const typename DescriptorT::DATA_T* __restrict__ dataset_ptr,
  uint32_t dim,
  uint32_t query_smem_ptr) -> typename DescriptorT::DISTANCE_T
{
  using DATA_T                    = typename DescriptorT::DATA_T;
  using DISTANCE_T                = typename DescriptorT::DISTANCE_T;
  using LOAD_T                    = typename DescriptorT::LOAD_T;
  using QUERY_T                   = typename DescriptorT::QUERY_T;
  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto vlen             = device::get_vlen<LOAD_T, DATA_T>();
  constexpr auto reg_nelem =
    raft::div_rounding_up_unsafe<uint32_t>(kDatasetBlockDim, kTeamSize * vlen);

  DISTANCE_T r = 0;
  for (uint32_t elem_offset = (threadIdx.x % kTeamSize) * vlen; elem_offset < dim;
       elem_offset += kDatasetBlockDim) {
    DATA_T data[reg_nelem][vlen];
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
      device::ldg_cg(reinterpret_cast<LOAD_T&>(data[e]),
                     reinterpret_cast<const LOAD_T*>(dataset_ptr + k));
    }
#pragma unroll
    for (uint32_t e = 0; e < reg_nelem; e++) {
      const uint32_t k = e * (kTeamSize * vlen) + elem_offset;
      if (k >= dim) break;
#pragma unroll
      for (uint32_t v = 0; v < vlen; v++) {
        QUERY_T d;
        device::lds(
          d,
          query_smem_ptr +
            sizeof(QUERY_T) * device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(k + v));
        r += ws_dist_op<Metric, QUERY_T, DISTANCE_T>(
          d, cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(data[e][v]));
      }
    }
  }
  return r;
}

/** Per-thread distance including the metric's normalization (see `compute_distance_per_thread`). */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto ws_compute_distance_per_thread(
  const typename DescriptorT::args_t args, const typename DescriptorT::INDEX_T dataset_index) ->
  typename DescriptorT::DISTANCE_T
{
  auto distance = ws_compute_distance_worker<Metric, DescriptorT>(
    DescriptorT::ptr(args) + (static_cast<std::uint64_t>(DescriptorT::ld(args)) * dataset_index),
    args.dim,
    args.smem_ws_ptr);
  if constexpr (Metric == cuvs::distance::DistanceType::CosineExpanded) {
    distance = apply_normalization_standard_cosine_impl<DescriptorT::kTeamSize,
                                                        DescriptorT::kDatasetBlockDim,
                                                        typename DescriptorT::DATA_T,
                                                        typename DescriptorT::INDEX_T,
                                                        typename DescriptorT::DISTANCE_T,
                                                        typename DescriptorT::QUERY_T>(
      distance, args, dataset_index);
  }
  return distance;
}

/** Full distance for one dataset row, reduced across the team. */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto ws_compute_distance(const typename DescriptorT::args_t args,
                                                     typename DescriptorT::INDEX_T dataset_index,
                                                     bool valid) ->
  typename DescriptorT::DISTANCE_T
{
  using DISTANCE_T = typename DescriptorT::DISTANCE_T;
  auto per_thread =
    valid ? ws_compute_distance_per_thread<Metric, DescriptorT>(args, dataset_index) : DISTANCE_T{};
  return device::team_sum<DescriptorT::kTeamSize>(per_thread);
}

/**
 * Seed the result buffer with randomly picked (or user supplied) nodes.
 *
 * Follows `device::compute_distance_to_random_nodes_jit`, dropping the multi-CTA parameters
 * (block id / traversed hash table) that the single-CTA path never uses.
 */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION void ws_compute_distance_to_random_nodes(
  typename DescriptorT::INDEX_T* __restrict__ result_indices_ptr,
  typename DescriptorT::DISTANCE_T* __restrict__ result_distances_ptr,
  const DescriptorT* smem_desc,
  uint32_t num_pickup,
  uint32_t num_distilation,
  uint64_t rand_xor_mask,
  const typename DescriptorT::INDEX_T* __restrict__ seed_ptr,
  uint32_t num_seeds,
  typename DescriptorT::INDEX_T* __restrict__ visited_hash_ptr,
  uint32_t visited_hash_bitlen,
  typename DescriptorT::INDEX_T graph_size)
{
  using IndexT                       = typename DescriptorT::INDEX_T;
  using DistanceT                    = typename DescriptorT::DISTANCE_T;
  constexpr uint32_t kTeamSizeBits   = raft::Pow2<DescriptorT::kTeamSize>::Log2;

  const IndexT dataset_size = smem_desc->size;
  const auto args_load      = smem_desc->args.load();

  const auto max_i = raft::round_up_safe<uint32_t>(num_pickup, device::warp_size >> kTeamSizeBits);
  const IndexT seed_index_limit = graph_size > 0 ? graph_size : dataset_size;

  for (uint32_t i = threadIdx.x >> kTeamSizeBits; i < max_i;
       i += (blockDim.x >> kTeamSizeBits)) {
    const bool valid_i = (i < num_pickup);

    IndexT best_index_team_local    = raft::upper_bound<IndexT>();
    DistanceT best_norm2_team_local = raft::upper_bound<DistanceT>();
    for (uint32_t j = 0; j < num_distilation; j++) {
      IndexT seed_index = 0;
      if (valid_i) {
        const uint32_t gid = i + (num_pickup * j);
        if (seed_ptr && (gid < num_seeds)) {
          seed_index = seed_ptr[gid];
        } else {
          seed_index = device::xorshift64(gid ^ rand_xor_mask) % seed_index_limit;
        }
      }

      const auto norm2 = ws_compute_distance<Metric, DescriptorT>(args_load, seed_index, valid_i);

      if (valid_i && (norm2 < best_norm2_team_local)) {
        best_norm2_team_local = norm2;
        best_index_team_local = seed_index;
      }
    }

    const unsigned lane_id = threadIdx.x & ((1u << kTeamSizeBits) - 1u);
    if (valid_i && lane_id == 0) {
      if (best_index_team_local != raft::upper_bound<IndexT>()) {
        if (hashmap::insert(visited_hash_ptr, visited_hash_bitlen, best_index_team_local) == 0) {
          // Deactivate this entry as insertion into visited hash table has failed.
          best_norm2_team_local = raft::upper_bound<DistanceT>();
          best_index_team_local = raft::upper_bound<IndexT>();
        }
      }
      result_distances_ptr[i] = best_norm2_team_local;
      result_indices_ptr[i]   = best_index_team_local;
    }
  }
}

/**
 * Expand the current parents: gather their neighbors from the graph and compute their distances.
 *
 * Follows `device::compute_distance_to_child_nodes_jit` with a static result position and no
 * traversed hash table (the single-CTA parameters).
 */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION void ws_compute_distance_to_child_nodes(
  typename DescriptorT::INDEX_T* __restrict__ result_child_indices_ptr,
  typename DescriptorT::DISTANCE_T* __restrict__ result_child_distances_ptr,
  const DescriptorT* smem_desc,
  const typename DescriptorT::INDEX_T* __restrict__ knn_graph,
  uint32_t knn_k,
  typename DescriptorT::INDEX_T* __restrict__ visited_hashmap_ptr,
  uint32_t visited_hash_bitlen,
  const typename DescriptorT::INDEX_T* __restrict__ parent_indices,
  const typename DescriptorT::INDEX_T* __restrict__ internal_topk_list,
  uint32_t search_width)
{
  using IndexT                       = typename DescriptorT::INDEX_T;
  using DistanceT                    = typename DescriptorT::DISTANCE_T;
  constexpr IndexT index_msb_1_mask  = utils::gen_index_msb_1_mask<IndexT>::value;
  constexpr IndexT invalid_index     = ~static_cast<IndexT>(0);
  constexpr uint32_t kTeamSizeBits   = raft::Pow2<DescriptorT::kTeamSize>::Log2;

  // Read child indices of parents from the knn graph and check whether the distance computation is
  // necessary.
  for (uint32_t i = threadIdx.x; i < knn_k * search_width; i += blockDim.x) {
    const IndexT smem_parent_id = parent_indices[i / knn_k];
    IndexT child_id             = invalid_index;
    if (smem_parent_id != invalid_index) {
      const auto parent_id = internal_topk_list[smem_parent_id] & ~index_msb_1_mask;
      child_id             = knn_graph[(i % knn_k) + (static_cast<int64_t>(knn_k) * parent_id)];
    }
    if (child_id != invalid_index) {
      if (hashmap::insert(visited_hashmap_ptr, visited_hash_bitlen, child_id) == 0) {
        child_id = invalid_index;
      }
    }
    result_child_indices_ptr[i] = child_id;
  }
  __syncthreads();

  const auto num_k     = knn_k * search_width;
  const auto max_i     = raft::round_up_safe(num_k, device::warp_size >> kTeamSizeBits);
  const auto args      = smem_desc->args.load();
  const bool lead_lane = (threadIdx.x & ((1u << kTeamSizeBits) - 1u)) == 0;

  for (uint32_t i = threadIdx.x >> kTeamSizeBits; i < max_i; i += blockDim.x >> kTeamSizeBits) {
    const bool valid_i  = (i < num_k);
    const auto child_id = valid_i ? result_child_indices_ptr[i] : invalid_index;

    const auto per_thread =
      (child_id != invalid_index)
        ? ws_compute_distance_per_thread<Metric, DescriptorT>(args, child_id)
        : (lead_lane ? raft::upper_bound<DistanceT>() : DistanceT{});
    const DistanceT child_dist = device::team_sum<DescriptorT::kTeamSize>(per_thread);
    __syncwarp();

    if (valid_i && lead_lane) { result_child_distances_ptr[i] = child_dist; }
  }
}

/**
 * Copy the dataset descriptor into shared memory and point it at the shared-memory query buffer.
 *
 * This is the first half of `setup_workspace_standard_impl`. It is split off because it does not
 * depend on the query, so a block that runs several queries only has to do it once.
 */
template <typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION auto ws_stage_descriptor(const DescriptorT* desc,
                                                     std::uint8_t* smem) -> const DescriptorT*
{
  using QUERY_T   = typename DescriptorT::QUERY_T;
  using word_type = std::uint32_t;

  auto* r   = reinterpret_cast<DescriptorT*>(smem);
  auto* buf = reinterpret_cast<QUERY_T*>(r + 1);

  constexpr uint32_t kCount = sizeof(DescriptorT) / sizeof(word_type);
  using blob_type           = word_type[kCount];
  auto& src                 = reinterpret_cast<const blob_type&>(*desc);
  auto& dst                 = reinterpret_cast<blob_type&>(*r);
  for (uint32_t i = threadIdx.x; i < kCount; i += blockDim.x) {
    dst[i] = src[i];
  }
  const auto smem_ptr_offset =
    reinterpret_cast<std::uint8_t*>(&(r->args.smem_ws_ptr)) - reinterpret_cast<std::uint8_t*>(r);
  if (threadIdx.x == uint32_t(smem_ptr_offset / sizeof(word_type))) {
    r->args.smem_ws_ptr = uint32_t(__cvta_generic_to_shared(buf));
  }
  __syncthreads();
  return r;
}

/**
 * Stage one query vector into the workspace of an already-staged descriptor: the second half of
 * `setup_workspace_standard_impl`. The caller must sync before the query is read.
 */
template <typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION void ws_stage_query(std::uint8_t* smem,
                                                uint32_t dim,
                                                const typename DescriptorT::DATA_T* queries_ptr,
                                                uint32_t query_id)
{
  using DATA_T  = typename DescriptorT::DATA_T;
  using LOAD_T  = typename DescriptorT::LOAD_T;
  using QUERY_T = typename DescriptorT::QUERY_T;

  constexpr auto kTeamSize        = DescriptorT::kTeamSize;
  constexpr auto kDatasetBlockDim = DescriptorT::kDatasetBlockDim;
  constexpr auto vlen             = device::get_vlen<LOAD_T, DATA_T>();

  auto* buf          = reinterpret_cast<QUERY_T*>(reinterpret_cast<DescriptorT*>(smem) + 1);
  const auto buf_len = raft::round_up_safe<uint32_t>(dim, kDatasetBlockDim);
  queries_ptr += static_cast<size_t>(dim) * query_id;
  for (uint32_t i = threadIdx.x; i < buf_len; i += blockDim.x) {
    const uint32_t j = device::swizzling<kDatasetBlockDim, vlen * kTeamSize>(i);
    if (i < dim) {
      buf[j] = cuvs::spatial::knn::detail::utils::mapping<QUERY_T>{}(queries_ptr[i]);
    } else {
      buf[j] = 0;
    }
  }
}

/**
 * The per-query search body: the same algorithm as `single_cta_search::search_core` for the
 * bitonic-sort top-k without filtering.
 *
 * @param smem the classic CAGRA shared-memory region (query workspace, result buffers, hash table)
 * @param smem_desc the descriptor staged in `smem` by `ws_stage_descriptor`
 */
template <cuvs::distance::DistanceType Metric, typename DescriptorT>
RAFT_DEVICE_INLINE_FUNCTION void ws_search_core(
  // Not `__restrict__`: `smem_desc` points into this region.
  std::uint8_t* smem,
  const DescriptorT* smem_desc,
  const launch_args<typename DescriptorT::DATA_T,
                    typename DescriptorT::INDEX_T,
                    typename DescriptorT::DISTANCE_T>& args,
  uint32_t query_id)
{
  using IndexT    = typename DescriptorT::INDEX_T;
  using DistanceT = typename DescriptorT::DISTANCE_T;

  constexpr IndexT index_msb_1_mask = utils::gen_index_msb_1_mask<IndexT>::value;
  const IndexT invalid_index        = utils::get_max_value<IndexT>();

  const auto internal_topk = args.internal_topk;
  const auto search_width  = args.search_width;
  const auto graph_degree  = args.graph_degree;

  const auto result_buffer_size    = internal_topk + (search_width * graph_degree);
  const auto result_buffer_size_32 = raft::round_up_safe<uint32_t>(result_buffer_size, 32);
  const auto small_hash_size       = hashmap::get_size(args.small_hash_bitlen);

  const uint32_t smem_ws_size_in_bytes = smem_desc->smem_ws_size_in_bytes();

  ws_stage_query<DescriptorT>(smem, smem_desc->args.dim, args.queries, query_id);

  auto* __restrict__ result_indices_buffer =
    reinterpret_cast<IndexT*>(smem + smem_ws_size_in_bytes);
  auto* __restrict__ result_distances_buffer =
    reinterpret_cast<DistanceT*>(result_indices_buffer + result_buffer_size_32);
  auto* __restrict__ visited_hash_buffer =
    reinterpret_cast<IndexT*>(result_distances_buffer + result_buffer_size_32);
  auto* __restrict__ parent_list_buffer =
    reinterpret_cast<IndexT*>(visited_hash_buffer + small_hash_size);
  auto* __restrict__ topk_ws = reinterpret_cast<std::uint32_t*>(parent_list_buffer + search_width);
  auto* terminate_flag       = reinterpret_cast<std::uint32_t*>(topk_ws + 3);
  auto* __restrict__ smem_work_ptr = reinterpret_cast<std::uint32_t*>(terminate_flag + 1);

  auto to_source_index = [&args](IndexT x) -> IndexT {
    return args.source_indices == nullptr ? x : args.source_indices[x];
  };

  if (threadIdx.x == 0) {
    terminate_flag[0] = 0;
    topk_ws[0]        = ~0u;
  }

  IndexT* local_visited_hashmap_ptr;
  if (args.small_hash_bitlen) {
    local_visited_hashmap_ptr = visited_hash_buffer;
  } else {
    local_visited_hashmap_ptr =
      args.visited_hashmap + (static_cast<size_t>(hashmap::get_size(args.hash_bitlen)) * query_id);
  }
  hashmap::init(local_visited_hashmap_ptr, args.hash_bitlen, 0);
  __syncthreads();

  const IndexT* const local_seed_ptr =
    args.seeds ? args.seeds + (args.num_seeds * query_id) : nullptr;
  ws_compute_distance_to_random_nodes<Metric, DescriptorT>(result_indices_buffer,
                                                          result_distances_buffer,
                                                          smem_desc,
                                                          result_buffer_size,
                                                          args.num_random_samplings,
                                                          args.rand_xor_mask,
                                                          local_seed_ptr,
                                                          args.num_seeds,
                                                          local_visited_hashmap_ptr,
                                                          args.hash_bitlen,
                                                          args.graph_size);
  __syncthreads();

  std::uint32_t iter = 0;
  while (1) {
    // Reset the small-hash table. The threads that are not busy with the bitonic sort below do it;
    // the two touch disjoint shared-memory regions. Same split as the JIT kernel.
    if ((iter + 1) % args.small_hash_reset_interval == 0) {
      const bool sort_uses_two_warps = args.max_candidates > 128;
      unsigned hash_start_tid;
      if (blockDim.x == 32) {
        hash_start_tid = 0;
      } else if (blockDim.x == 64) {
        hash_start_tid = sort_uses_two_warps ? 0 : 32;
      } else {
        hash_start_tid = sort_uses_two_warps ? 64 : 32;
      }
      hashmap::init(local_visited_hashmap_ptr, args.hash_bitlen, hash_start_tid);
    }

    topk_by_bitonic_sort_and_merge<false>(result_distances_buffer,
                                         result_indices_buffer,
                                         args.max_itopk,
                                         internal_topk,
                                         result_distances_buffer + internal_topk,
                                         result_indices_buffer + internal_topk,
                                         args.max_candidates,
                                         search_width * graph_degree,
                                         topk_ws,
                                         (iter == 0));
    __syncthreads();

    if (iter + 1 == args.max_iterations) { break; }

    if (threadIdx.x < 32) {
      pickup_next_parents<true, IndexT>(
        terminate_flag, parent_list_buffer, result_indices_buffer, internal_topk, search_width);
    }

    // Restore the small-hash table by putting the internal-topk indices in it.
    if ((iter + 1) % args.small_hash_reset_interval == 0) {
      const unsigned first_tid = ((blockDim.x <= 32) ? 0 : 32);
      hashmap_restore(local_visited_hashmap_ptr,
                      args.hash_bitlen,
                      result_indices_buffer,
                      internal_topk,
                      first_tid);
    }
    __syncthreads();

    if (*terminate_flag && iter >= args.min_iterations) { break; }
    __syncthreads();

    ws_compute_distance_to_child_nodes<Metric, DescriptorT>(result_indices_buffer + internal_topk,
                                                           result_distances_buffer + internal_topk,
                                                           smem_desc,
                                                           args.graph,
                                                           graph_degree,
                                                           local_visited_hashmap_ptr,
                                                           args.hash_bitlen,
                                                           parent_list_buffer,
                                                           result_indices_buffer,
                                                           search_width);
    __syncthreads();

    // The JIT kernel clears this flag at the end of every iteration (there it doubles as the filter
    // flag); keep doing so, otherwise a `min_iterations` that outlives the first termination signal
    // would take a different path.
    if (threadIdx.x == 0) { *terminate_flag = 0; }
    __syncthreads();

    iter++;
  }
  __syncthreads();

  // Move invalid index items to the end of the buffer without sorting the entire buffer. Entries can
  // be invalid because a hash-table insertion failed.
  using scan_op_t    = cub::WarpScan<unsigned>;
  auto& temp_storage = *reinterpret_cast<typename scan_op_t::TempStorage*>(smem_work_ptr);

  constexpr std::uint32_t warp_size = 32;
  if (threadIdx.x < warp_size) {
    std::uint32_t num_found_valid = 0;
    for (std::uint32_t buffer_offset = 0; buffer_offset < internal_topk;
         buffer_offset += warp_size) {
      const auto src_position = buffer_offset + threadIdx.x;
      const std::uint32_t is_valid_index =
        (result_indices_buffer[src_position] & (~index_msb_1_mask)) == invalid_index ? 0 : 1;
      std::uint32_t new_position;
      scan_op_t(temp_storage).InclusiveSum(is_valid_index, new_position);
      if (is_valid_index) {
        const auto dst_position               = num_found_valid + (new_position - 1);
        result_indices_buffer[dst_position]   = result_indices_buffer[src_position];
        result_distances_buffer[dst_position] = result_distances_buffer[src_position];
      }

      num_found_valid += new_position;
      for (std::uint32_t offset = (warp_size >> 1); offset > 0; offset >>= 1) {
        const auto v = raft::shfl_xor(num_found_valid, offset);
        if ((threadIdx.x & offset) == 0) { num_found_valid = v; }
      }

      if (num_found_valid >= args.top_k) { break; }
    }

    if (num_found_valid < args.top_k) {
      for (std::uint32_t i = num_found_valid + threadIdx.x; i < internal_topk; i += warp_size) {
        result_indices_buffer[i]   = invalid_index;
        result_distances_buffer[i] = utils::get_max_value<DistanceT>();
      }
    }
  }

  // If the sufficient number of valid indexes are not in the internal topk, pick up from the
  // candidate list. The sync makes the condition block-uniform: only the first warp compacted the
  // buffer above, and the barrier inside the branch has to be reached by the whole block.
  __syncthreads();
  if (args.top_k > internal_topk || result_indices_buffer[args.top_k - 1] == invalid_index) {
    topk_by_bitonic_sort_and_merge<false>(result_distances_buffer,
                                          result_indices_buffer,
                                          args.max_itopk,
                                          internal_topk,
                                          result_distances_buffer + internal_topk,
                                          result_indices_buffer + internal_topk,
                                          args.max_candidates,
                                          search_width * graph_degree,
                                          topk_ws,
                                          (iter == 0));
  }
  __syncthreads();

  for (std::uint32_t i = threadIdx.x; i < args.top_k; i += blockDim.x) {
    const size_t j        = static_cast<size_t>(i) + (static_cast<size_t>(args.top_k) * query_id);
    const std::uint32_t ii = device::swizzling(i);
    if (args.result_distances != nullptr) {
      args.result_distances[j] = result_distances_buffer[ii];
    }
    // Clear the most significant bit, which marks an already used node.
    args.result_indices[j] = to_source_index(result_indices_buffer[ii] & ~index_msb_1_mask);
  }
  if (threadIdx.x == 0 && args.num_executed_iterations != nullptr) {
    args.num_executed_iterations[query_id] = iter + 1;
  }
}

/**
 * Response of `clusterlaunchcontrol.try_cancel`: the index of a block the grid has not started yet,
 * or an invalid marker when nothing was left to cancel.
 *
 * This is not part of `cub::detail::warpspeed`; it is the small helper from the warpspeed examples
 * (`kernels_common/warpspeed_next_block_idx.cuh`), reproduced here to keep this file self-contained.
 * The 16-byte alignment of `uint4` is required: `try_cancel` writes a `b128` to this address.
 */
struct ws_next_block_idx {
  uint4 data;

  _RAFT_DEVICE auto is_valid() const -> bool
  {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return cuda::ptx::clusterlaunchcontrol_query_cancel_is_canceled(data);
#else
    return false;
#endif
  }

  _RAFT_DEVICE auto x() const -> std::uint32_t
  {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
    return cuda::ptx::clusterlaunchcontrol_query_cancel_get_first_ctaid_x<int>(data);
#else
    return 0;
#endif
  }
};

/**
 * Shared-memory resources of the warp-specialized kernel, allocated in the same order on the host
 * (to size the dynamic shared memory) and on the device.
 *
 * The classic CAGRA region comes first so that its internal offsets are unchanged; the warpspeed
 * resources and their barriers follow.
 */
struct ws_resources {
  std::uint8_t* cagra_smem;
  ws::SmemResource<ws_next_block_idx> res_nbi;
};

/**
 * Lay out the shared memory and register the barriers.
 *
 * `squads` only determines the arrival counts of the barriers, not the amount of shared memory, so
 * the host sizing path below may pass a placeholder descriptor.
 */
template <int NumSquads>
RAFT_INLINE_FUNCTION auto ws_alloc_resources(ws::SyncHandler& sync_handler,
                                             ws::SmemAllocator& smem_allocator,
                                             uint32_t cagra_smem_bytes,
                                             const ws::SquadDesc (&squads)[NumSquads]) -> ws_resources
{
  ws_resources res{
    .cagra_smem =
      static_cast<std::uint8_t*>(smem_allocator.alloc(cagra_smem_bytes, kCagraSmemAlign)),
    .res_nbi = ws::SmemResource<ws_next_block_idx>(
      sync_handler, smem_allocator, ws::Stages(kNumStagesNextBlockIdx)),
  };

  // First phase: the block issues `try_cancel` into the buffer. Second phase: the block reads the
  // response once the asynchronous write has landed.
  res.res_nbi.addPhase(sync_handler, smem_allocator, squads);
  res.res_nbi.addPhase(sync_handler, smem_allocator, squads);

  return res;
}

/** Dynamic shared memory a launch needs for the given classic CAGRA region size. */
inline auto ws_total_smem_bytes(uint32_t cagra_smem_bytes) -> uint32_t
{
  ws::SyncHandler sync_handler{};
  ws::SmemAllocator smem_allocator{};
  const ws::SquadDesc squads[] = {ws::SquadDesc(0, 1)};
  (void)ws_alloc_resources(sync_handler, smem_allocator, cagra_smem_bytes, squads);
  // On the device the barriers are initialized by `clusterInitSync`, which is what the handler's
  // destructor checks for. Here it only accounts for their shared memory.
  sync_handler.mHasInitialized = true;
  return smem_allocator.sizeBytes();
}

template <cuvs::distance::DistanceType Metric, typename DescriptorT, int NumSquads>
RAFT_DEVICE_INLINE_FUNCTION void ws_search_body(
  ws::Squad squad,
  ws::SpecialRegisters sr,
  const ws::SquadDesc (&squads)[NumSquads],
  const launch_args<typename DescriptorT::DATA_T,
                    typename DescriptorT::INDEX_T,
                    typename DescriptorT::DISTANCE_T>& args)
{
  ws::SyncHandler sync_handler{};
  ws::SmemAllocator smem_allocator{};
  auto res = ws_alloc_resources(sync_handler, smem_allocator, args.cagra_smem_bytes, squads);
  sync_handler.clusterInitSync<kMaxBlockThreads>(sr, ws::SkipSync{});
  __syncthreads();

  // Query-independent, so it is staged once even if this block runs several queries.
  const auto* smem_desc = ws_stage_descriptor<DescriptorT>(
    static_cast<const DescriptorT*>(args.dataset_desc), res.cagra_smem);

  // The grid still has one block per query, so a block that is launched normally starts on its own
  // query and only then looks for work that has not been started yet.
  std::uint32_t query_id = sr.blockIdxX;
  ws_next_block_idx next_block_idx;

  if (!args.steal) {
    if (query_id < args.num_queries) {
      ws_search_core<Metric, DescriptorT>(res.cagra_smem, smem_desc, args, query_id);
    }
    return;
  }

  do {
    ws::SmemStage stage_nbi         = res.res_nbi.nextStage();
    auto [phase_nbi_w, phase_nbi_r] = ws::bindPhases<2>(stage_nbi);

    // Claim the next not-yet-started block. The response arrives asynchronously, so the search of
    // the current query overlaps with the claim.
    {
      ws::SmemRef ref_nbi = phase_nbi_w.acquireRef();
      if (squad.isLeaderThread()) {
        cuda::ptx::clusterlaunchcontrol_try_cancel(&ref_nbi.data(), ref_nbi.ptrCurBarrierRelease());
      }
      ref_nbi.squadIncreaseTxCount(squad, sizeof(ws_next_block_idx));
    }

    if (query_id < args.num_queries) {
      ws_search_core<Metric, DescriptorT>(res.cagra_smem, smem_desc, args, query_id);
    }

    // Reading the response arrives on the barrier that the next iteration's claim waits for, so it
    // also orders the CAGRA shared-memory region between two queries of this block.
    {
      ws::SmemRef ref_nbi = phase_nbi_r.acquireRef();
      next_block_idx      = ref_nbi.data();
      ref_nbi.setFenceLdsToAsyncProxy();
    }
    query_id = next_block_idx.x();
  } while (next_block_idx.is_valid());
}

template <cuvs::distance::DistanceType Metric,
          uint32_t TeamSize,
          uint32_t DatasetBlockDim,
          typename DataT,
          typename IndexT,
          typename DistanceT>
__global__ __launch_bounds__(kMaxBlockThreads, 1) void search_single_cta_ws_kernel(
  launch_args<DataT, IndexT, DistanceT> args)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000)
  using descriptor_t = ws_descriptor_t<TeamSize, DatasetBlockDim, DataT, IndexT, DistanceT>;

  const ws::SpecialRegisters sr = ws::getSpecialRegisters();
  const ws::SquadDesc squads[]  = {
    ws::SquadDesc(0, static_cast<int>(blockDim.x / device::warp_size))};
  ws::squadDispatch(sr, squads, [&](ws::Squad squad) {
    ws_search_body<Metric, descriptor_t>(squad, sr, squads, args);
  });
#else
  (void)args;
#endif
}

/**
 * Launched by `search_algo::SINGLE_CTA_NOOP` with the grid and block size of a real search, and does
 * nothing at all. What separates it from SINGLE_CTA is the search itself, so its runtime is the floor
 * that any search variant pays: launching one block per query, reserving the shared memory, and
 * retiring the grid.
 *
 * The body needs almost no registers, which on its own would make it more resident than the search
 * kernel; `noop_smem_bytes` pads the dynamic shared memory of the launch to hold it to the same
 * blocks per SM.
 *
 * Nothing is written, so the caller's result buffers keep whatever they held.
 */
template <typename DataT, typename IndexT, typename DistanceT>
__global__ __launch_bounds__(kMaxBlockThreads, 1) void search_noop_kernel(
  launch_args<DataT, IndexT, DistanceT> args)
{
  (void)args;
}

}  // namespace cuvs::neighbors::cagra::detail::single_cta_ws_search
