/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "hashmap.hpp"
#include "search_plan.cuh"
#include "search_single_cta_kernel_launcher_common.cuh"

#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/common.hpp>

#include <raft/core/device_mdspan.hpp>
#include <raft/core/logger.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <raft/core/resource/device_properties.hpp>
#include <raft/core/resources.hpp>
#include <raft/util/pow2_utils.cuh>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <type_traits>

/**
 * Warp-specialized (work-stealing) variant of the single-CTA CAGRA search.
 *
 * This header is host-only: it is included by `factory.cuh` (and therefore by every CAGRA
 * translation unit), so it must not pull in any device code. The kernel itself lives in
 * `search_single_cta_ws_kernel.cuh` and is compiled by `search_single_cta_ws_inst.cu` alone, for
 * SM 100+ only. `CUVS_CAGRA_ENABLE_SINGLE_CTA_WS` tells us whether that translation unit is part of
 * the build; without it, the algo is rejected at the factory.
 */
namespace cuvs::neighbors::cagra::detail::single_cta_ws_search {

#if defined(CUVS_CAGRA_ENABLE_SINGLE_CTA_WS)
inline constexpr bool kCompiledIn = true;
#else
inline constexpr bool kCompiledIn = false;
#endif

/**
 * Which instantiations `search_single_cta_ws_inst.cu` provides. The kernel is currently limited to
 * float data with 32-bit indices and no filtering; everything else falls back to an error in the
 * factory rather than silently picking another algorithm.
 */
template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SampleFilterT,
          typename SourceIndexT,
          typename OutputIndexT>
inline constexpr bool is_supported_instance_v =
  kCompiledIn && std::is_same_v<DataT, float> && std::is_same_v<IndexT, uint32_t> &&
  std::is_same_v<DistanceT, float> && std::is_same_v<SourceIndexT, IndexT> &&
  std::is_same_v<OutputIndexT, IndexT> &&
  std::is_same_v<SampleFilterT, cuvs::neighbors::filtering::none_sample_filter>;

/** Bytes reserved at the end of the classic CAGRA shared-memory region for `cub::WarpScan`. */
inline constexpr uint32_t kSmemWorkBytes = 32;

/** Everything the kernel needs; passed by value, so it must stay a POD of pointers and scalars. */
template <typename DataT, typename IndexT, typename DistanceT>
struct launch_args {
  const dataset_descriptor_base_t<DataT, IndexT, DistanceT>* dataset_desc;
  const IndexT* graph;             // [graph_size, graph_degree]
  const DataT* queries;            // [num_queries, dim]
  const IndexT* seeds;             // [num_queries, num_seeds], optional
  const IndexT* source_indices;    // [dataset_size], optional
  IndexT* visited_hashmap;         // [num_queries, 1 << hash_bitlen], unused with a small hash
  IndexT* result_indices;          // [num_queries, top_k]
  DistanceT* result_distances;     // [num_queries, top_k], optional
  uint32_t* num_executed_iterations;  // [num_queries], optional
  IndexT graph_size;
  uint32_t graph_degree;
  uint32_t num_queries;
  uint32_t top_k;
  uint32_t num_random_samplings;
  uint64_t rand_xor_mask;
  uint32_t num_seeds;
  uint32_t max_candidates;
  uint32_t max_itopk;
  uint32_t internal_topk;
  uint32_t search_width;
  uint32_t min_iterations;
  uint32_t max_iterations;
  uint32_t hash_bitlen;
  uint32_t small_hash_bitlen;
  uint32_t small_hash_reset_interval;
  /** Size of the classic CAGRA region, which sits at the front of the dynamic shared memory. */
  uint32_t cagra_smem_bytes;
  /** When false, every block searches only the query it was launched with. */
  bool steal;
};

/**
 * Diagnostic switch: `CUVS_CAGRA_WS_NO_STEAL=1` runs the same kernel with one query per block, which
 * separates what work stealing contributes from what the rest of the kernel does differently to the
 * JIT-compiled SINGLE_CTA path.
 */
inline auto steal_enabled() -> bool
{
  static const bool enabled = [] {
    const char* env     = std::getenv("CUVS_CAGRA_WS_NO_STEAL");
    const bool disabled = env != nullptr && env[0] != '\0' && env[0] != '0';
    if (disabled) {
      RAFT_LOG_INFO("CAGRA SINGLE_CTA_WS: work stealing disabled by CUVS_CAGRA_WS_NO_STEAL");
    }
    return !disabled;
  }();
  return enabled;
}

/**
 * Host entry points into the separately compiled kernel translation unit.
 *
 * Defined and explicitly instantiated in `search_single_cta_ws_inst.cu`; the runtime metric /
 * team-size / block-dim dispatch happens there.
 */
template <typename DataT, typename IndexT, typename DistanceT>
struct launcher {
  /** Total dynamic shared memory: the classic CAGRA region plus the warpspeed barriers. */
  static auto total_smem_bytes(uint32_t cagra_smem_bytes) -> uint32_t;

  /** Throws unless the current device can run the kernel. */
  static void check_device_support();

  static void run(const launch_args<DataT, IndexT, DistanceT>& args,
                  cuvs::distance::DistanceType metric,
                  uint32_t team_size,
                  uint32_t dataset_block_dim,
                  uint32_t block_size,
                  uint32_t smem_size,
                  cudaStream_t stream);

  /**
   * Launches a kernel that returns immediately, with the grid, block size and dynamic shared memory
   * of a real search. Used to measure what a search costs before doing any searching.
   */
  static void run_noop(const launch_args<DataT, IndexT, DistanceT>& args,
                       uint32_t block_size,
                       uint32_t smem_size,
                       cudaStream_t stream);
};

template <typename DataT,
          typename IndexT,
          typename DistanceT,
          typename SAMPLE_FILTER_T,
          typename SourceIndexT = IndexT,
          typename OutputIndexT = SourceIndexT>
struct search
  : search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T, SourceIndexT, OutputIndexT> {
  using base_type =
    search_plan_impl<DataT, IndexT, DistanceT, SAMPLE_FILTER_T, SourceIndexT, OutputIndexT>;
  using DATA_T     = typename base_type::DATA_T;
  using INDEX_T    = typename base_type::INDEX_T;
  using DISTANCE_T = typename base_type::DISTANCE_T;

  using base_type::algo;
  using base_type::hashmap_max_fill_rate;
  using base_type::hashmap_min_bitlen;
  using base_type::hashmap_mode;
  using base_type::itopk_size;
  using base_type::max_iterations;
  using base_type::max_queries;
  using base_type::min_iterations;
  using base_type::num_random_samplings;
  using base_type::rand_xor_mask;
  using base_type::search_width;
  using base_type::team_size;
  using base_type::thread_block_size;

  using base_type::dim;
  using base_type::graph_degree;
  using base_type::topk;

  using base_type::hash_bitlen;

  using base_type::dataset_size;
  using base_type::hashmap_size;
  using base_type::result_buffer_size;
  using base_type::small_hash_bitlen;
  using base_type::small_hash_reset_interval;

  using base_type::smem_size;

  using base_type::dataset_desc;
  using base_type::dev_seed;
  using base_type::hashmap;
  using base_type::num_executed_iterations;
  using base_type::num_seeds;

  using launcher_type = launcher<DataT, IndexT, DistanceT>;

  uint32_t num_itopk_candidates;
  /** Size of the classic CAGRA region; `smem_size` additionally covers the warpspeed barriers. */
  uint32_t cagra_smem_bytes;

  search(raft::resources const& res,
         search_params params,
         const dataset_descriptor_host<DataT, IndexT, DistanceT>& dataset_desc,
         int64_t dim,
         int64_t dataset_size,
         int64_t graph_degree,
         uint32_t topk)
    : base_type(res, params, dataset_desc, dim, dataset_size, graph_degree, topk)
  {
    set_params(res);
  }

  ~search() {}

  /** True when this plan only launches the no-op kernel (`search_algo::SINGLE_CTA_NOOP`). */
  [[nodiscard]] auto is_noop() const -> bool { return algo == search_algo::SINGLE_CTA_NOOP; }

  inline void set_params(raft::resources const& res)
  {
    // The no-op kernel computes no distances, so it needs neither the SM 100+ instructions nor a
    // dataset it knows how to read; it only has to reserve the same resources.
    if (!is_noop()) {
      launcher_type::check_device_support();
      RAFT_EXPECTS(!dataset_desc.is_vpq,
                   "SINGLE_CTA_WS does not support VPQ-compressed datasets yet");
    }

    num_itopk_candidates = search_width * graph_degree;
    result_buffer_size   = itopk_size + num_itopk_candidates;

    // The warp-specialized kernel only implements the bitonic-sort top-k with a single warp, which
    // is what the iterative build loop uses (search_width = 1, moderate graph degree).
    RAFT_EXPECTS(num_itopk_candidates <= 256,
                 "SINGLE_CTA_WS requires search_width * graph_degree <= 256 (got %u); the "
                 "radix-sort top-k is not implemented in the warp-specialized kernel",
                 num_itopk_candidates);
    RAFT_EXPECTS(itopk_size <= 256,
                 "SINGLE_CTA_WS requires internal_topk <= 256 (got %zu)",
                 static_cast<size_t>(itopk_size));

    typedef raft::Pow2<32> AlignBytes;
    const uint32_t result_buffer_size_32 = AlignBytes::roundUp(result_buffer_size);

    const uint32_t topk_ws_size = 3;
    cagra_smem_bytes            = dataset_desc.smem_ws_size_in_bytes +
                       (sizeof(INDEX_T) + sizeof(DISTANCE_T)) * result_buffer_size_32 +
                       sizeof(INDEX_T) * hashmap::get_size(small_hash_bitlen) +
                       sizeof(INDEX_T) * search_width + sizeof(uint32_t) * topk_ws_size +
                       sizeof(uint32_t) + kSmemWorkBytes;
    smem_size = launcher_type::total_smem_bytes(cagra_smem_bytes);

    // Block size heuristic of the SINGLE_CTA kernel, minus the radix-sort branch. Keeping it
    // identical makes the two algos directly comparable.
    constexpr uint32_t min_block_size = 64;
    constexpr uint32_t max_block_size = 1024;
    uint32_t block_size               = thread_block_size;
    if (block_size == 0) {
      block_size = min_block_size;

      // Increase block size according to shared memory requirements. If block size is 32, upper
      // limit of shared memory size per thread block is set to 4096.
      constexpr uint32_t ulimit_smem_size_cta32 = 4096;
      while (smem_size > ulimit_smem_size_cta32 / 32 * block_size) {
        block_size *= 2;
      }

      // Increase block size to improve GPU occupancy when the batch size is small.
      cudaDeviceProp deviceProp = raft::resource::get_device_properties(res);
      while ((block_size < max_block_size) &&
             (graph_degree * search_width * team_size >= block_size * 2) &&
             (max_queries <= (1024 / (block_size * 2)) * deviceProp.multiProcessorCount)) {
        block_size *= 2;
      }
    }
    RAFT_EXPECTS(block_size >= min_block_size,
                 "block_size cannot be smaller than min_block size, %u",
                 min_block_size);
    RAFT_EXPECTS(block_size <= max_block_size,
                 "block_size cannot be larger than max_block size %u",
                 max_block_size);
    thread_block_size = block_size;

    RAFT_LOG_DEBUG("# thread_block_size: %u", block_size);
    RAFT_LOG_DEBUG("# smem_size: %u (cagra region: %u)", smem_size, cagra_smem_bytes);

    hashmap_size = 0;
    if (small_hash_bitlen == 0) {
      hashmap_size = max_queries * hashmap::get_size(hash_bitlen);
      hashmap.resize(hashmap_size, raft::resource::get_cuda_stream(res));
    }
    RAFT_LOG_DEBUG("# hashmap_size: %zu", static_cast<size_t>(hashmap_size));
  }

  void operator()(raft::resources const& res,
                  raft::device_matrix_view<const INDEX_T, int64_t, raft::row_major> graph,
                  std::optional<raft::device_vector_view<const SourceIndexT, int64_t>> source_indices,
                  OutputIndexT* const result_indices_ptr,  // [num_queries, topk]
                  DISTANCE_T* const result_distances_ptr,  // [num_queries, topk]
                  const DATA_T* const queries_ptr,         // [num_queries, dataset_dim]
                  const std::uint32_t num_queries,
                  const INDEX_T* dev_seed_ptr,                   // [num_queries, num_seeds]
                  std::uint32_t* const num_executed_iterations,  // [num_queries]
                  uint32_t topk,
                  SAMPLE_FILTER_T sample_filter)
  {
    static_assert(is_supported_instance_v<DataT,
                                          IndexT,
                                          DistanceT,
                                          SAMPLE_FILTER_T,
                                          SourceIndexT,
                                          OutputIndexT>,
                  "single_cta_ws_search::search instantiated for an unsupported type combination");
    (void)sample_filter;  // none_sample_filter: nothing to apply.

    cudaStream_t stream = raft::resource::get_cuda_stream(res);

    const auto config = single_cta_search::compute_launch_config(
      num_itopk_candidates, itopk_size, static_cast<uint32_t>(thread_block_size));
    RAFT_EXPECTS(is_noop() ||
                   (config.topk_by_bitonic_sort && !config.bitonic_sort_and_merge_multi_warps),
                 "SINGLE_CTA_WS only implements the single-warp bitonic-sort top-k");

    launch_args<DataT, IndexT, DistanceT> args{};
    args.dataset_desc             = dataset_desc.dev_ptr(stream);
    args.graph                    = graph.data_handle();
    args.queries                  = queries_ptr;
    args.seeds                    = dev_seed_ptr;
    args.source_indices           = source_indices.has_value() ? source_indices->data_handle() : nullptr;
    args.visited_hashmap          = hashmap.data();
    args.result_indices           = result_indices_ptr;
    args.result_distances         = result_distances_ptr;
    args.num_executed_iterations  = num_executed_iterations;
    args.graph_size               = static_cast<INDEX_T>(graph.extent(0));
    args.graph_degree             = static_cast<uint32_t>(graph.extent(1));
    args.num_queries              = num_queries;
    args.top_k                    = topk;
    args.num_random_samplings     = num_random_samplings;
    args.rand_xor_mask            = rand_xor_mask;
    args.num_seeds                = num_seeds;
    args.max_candidates           = config.max_candidates;
    args.max_itopk                = config.max_itopk;
    args.internal_topk            = static_cast<uint32_t>(itopk_size);
    args.search_width             = static_cast<uint32_t>(search_width);
    args.min_iterations           = static_cast<uint32_t>(min_iterations);
    args.max_iterations           = static_cast<uint32_t>(max_iterations);
    args.hash_bitlen              = static_cast<uint32_t>(hash_bitlen);
    args.small_hash_bitlen        = static_cast<uint32_t>(small_hash_bitlen);
    args.small_hash_reset_interval = static_cast<uint32_t>(small_hash_reset_interval);
    args.cagra_smem_bytes         = cagra_smem_bytes;
    args.steal                    = steal_enabled();

    if (is_noop()) {
      launcher_type::run_noop(
        args, static_cast<uint32_t>(thread_block_size), smem_size, stream);
      return;
    }

    launcher_type::run(args,
                       dataset_desc.metric,
                       dataset_desc.team_size,
                       dataset_desc.dataset_block_dim,
                       static_cast<uint32_t>(thread_block_size),
                       smem_size,
                       stream);
  }
};

}  // namespace cuvs::neighbors::cagra::detail::single_cta_ws_search
