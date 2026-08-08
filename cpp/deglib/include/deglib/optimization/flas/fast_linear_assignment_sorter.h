#ifndef FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H
#define FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <random>
#include <span>

#include "deglib/distances.h"
#include "deglib/distance/fp32_l2.h"
#include "deglib/distance/evp_inner_product.h"

#include "deglib/optimization/flas/junker_volgenant_solver.h"

namespace flas {

// Stores element ID and a pointer to its feature vector for tracking during sorting.
struct MapField {
  int id;
  const float *feature;
};

// Initializes a single MapField with element ID and feature vector pointer.
inline void init_map_field(MapField &map_field, const int id, const float *const feature) {
  map_field.id = id;
  map_field.feature = feature;
}

// Fills a span of MapFields with sequential IDs (0..N-1) and pointers to contiguous features.
inline void init_map_fields(std::span<MapField> map_fields, const float *features, int dim) {
  const size_t dim_sz = static_cast<size_t>(dim);
  const size_t count = map_fields.size();
  for (size_t i = 0; i < count; ++i) {
    map_fields[i].id = static_cast<int>(i);
    map_fields[i].feature = features + i * dim_sz;
  }
}

// Helper: Creates a std::vector<MapField> pre-initialized with IDs (0..N-1) and feature pointers.
inline std::vector<MapField> make_map_fields(const float *features, int count, int dim) {
  std::vector<MapField> map_fields(static_cast<size_t>(count));
  init_map_fields(map_fields, features, dim);
  return map_fields;
}

using RandomEngine = std::mt19937;
constexpr int QUANT = 256;

enum class FlasMetric { L2, InnerProduct };

// Configuration parameters for the FLAS 1D sorting algorithm.
struct FlasSettings {
  float initial_radius_factor = 0.5f;
  float radius_decay = 0.9f;
  float radius_end = 1.0f;
  int num_filters = 1;
  int max_swap_positions = 9;
  float sample_factor = 1.0f;
  FlasMetric metric = FlasMetric::L2;
};

// Scratch buffers for candidate swap positions, quantized distance LUT, and JV solver state.
class SwapBuffers {
public:
  std::vector<int> swap_positions;
  std::vector<int> swap_indices;
  std::vector<MapField> swapped_elements;
  std::vector<const float *> fvs;
  std::vector<const float *> som_fvs;
  std::vector<int> dist_lut;
  std::vector<float> dist_lut_f;
  JVScratch jv_scratch;

  explicit SwapBuffers(int num_swap_positions = 0, int count = 0) {
    if (num_swap_positions > 0 || count > 0) {
      resize(static_cast<size_t>(num_swap_positions), static_cast<size_t>(count));
    }
  }

  // Resizes swap candidate and LUT buffers to `n` positions.
  void resize(size_t n, size_t count = 0) {
    swap_positions.resize(n);
    swapped_elements.resize(n);
    fvs.resize(n);
    som_fvs.resize(n);
    dist_lut.resize(n * n);
    dist_lut_f.resize(n * n);
    jv_scratch.init(static_cast<int>(n));
    if (count > 0) {
      swap_indices.resize(count);
    }
  }

  int num_swap_positions() const noexcept {
    return static_cast<int>(swap_positions.size());
  }
};

// Problem context binding map fields span, dimensions, RNG stream, and distance metric.
struct FlasContext {
  std::span<MapField> map_fields;
  int count;
  int dim;

  RandomEngine &rng;
  FlasMetric metric;
  deglib::DISTFUNC<float> dist_func;

  FlasContext(std::span<MapField> map_fields_, int count_, int dim_, RandomEngine &rng_, FlasMetric metric_)
    : map_fields(map_fields_), count(count_), dim(dim_), rng(rng_), metric(metric_) {

    if (metric == FlasMetric::InnerProduct) {
      dist_func = deglib::to_dist_func(deglib::to_flat_variant(deglib::distances::fp32_ip::select_dist(dim)));
    } else {
      dist_func = deglib::to_dist_func(deglib::to_flat_variant(deglib::distances::fp32_l2::select_dist(dim)));
    }
  }

  FlasContext(MapField *map_fields_, int count_, int dim_, RandomEngine &rng_, FlasMetric metric_)
    : FlasContext(std::span<MapField>(map_fields_, static_cast<size_t>(count_)), count_, dim_, rng_, metric_) {}

  FlasContext(const FlasContext&) = delete;
  FlasContext& operator=(const FlasContext&) = delete;
};

// Manages SOM buffer and moving-average filter scratch memory (Move-safe, Rule of Zero).
class SomGrid {
public:
  std::vector<float> som_buf;
  size_t som_offset = 0;

  std::vector<float> filtered_som_buf; // Buffer holding filtered SOM values (count * dim)
  std::vector<float> window_sum;       // Accumulator for sliding window filter (dim)

  explicit SomGrid(int count, int dim) {
    size_t max_ext = static_cast<size_t>(count / 2);
    som_offset = max_ext * static_cast<size_t>(dim);
    som_buf.resize(static_cast<size_t>(count + 2 * max_ext) * static_cast<size_t>(dim));
    filtered_som_buf.resize(static_cast<size_t>(count) * static_cast<size_t>(dim));
    window_sum.resize(static_cast<size_t>(dim));
  }

  inline float* som() noexcept { return som_buf.data() + som_offset; }
  inline const float* som() const noexcept { return som_buf.data() + som_offset; }

  inline float* row(int i, int dim) noexcept { return som() + i * dim; }
  inline const float* row(int i, int dim) const noexcept { return som() + i * dim; }

  // Mirrors boundary cells for circular/clamped window filtering.
  inline void apply_mirror_padding(int ext, int count, int dim) noexcept {
    float *s = som();
    for (int i = 0; i < ext; i++) {
      std::memcpy(&s[(-1 - i) * dim], &s[(i + 1) * dim], dim * sizeof(float));
      std::memcpy(&s[(count + i) * dim], &s[(count - 2 - i) * dim], dim * sizeof(float));
    }
  }
};

// Randomly shuffles the elements in a span in-place using the provided RNG.
inline void shuffle_array(std::span<int> data, RandomEngine &rng) {
  std::shuffle(data.begin(), data.end(), rng);
}

// Phase "copy": Copies current MapField feature vectors into the SOM grid buffer.
inline void copy_feature_vectors_to_som(const FlasContext &ctx, SomGrid &grid) {
  for (int i = 0; i < ctx.count; i++) {
    const MapField &map_field = ctx.map_fields[i];
    std::copy_n(map_field.feature, ctx.dim, grid.row(i, ctx.dim));
  }
}

// Phase "filter": Applies a 1D sliding-window moving-average filter over a 2*radius+1 window.
inline void filter_weighted_som_1d(int radius, const FlasContext &ctx, SomGrid &grid) {
  if (ctx.count <= 1 || radius <= 0)
    return;

  int filter_size = 2 * radius + 1;
  int ext = filter_size / 2;
  int size = ctx.count;
  int dims = ctx.dim;
  const float inv_filter_size = 1.0f / static_cast<float>(filter_size);

  float *window_sum = grid.window_sum.data();

  // Apply boundary mirror padding to handle edge elements
  grid.apply_mirror_padding(ext, size, dims);

  // Initialize sliding window sum for the first window [-ext, ext]
  const float *base_ptr = grid.row(-ext, dims);
  std::fill_n(window_sum, dims, 0.0f);
  for (int i = 0; i < filter_size; i++) {
    const float *cell = base_ptr + i * dims;
    for (int d = 0; d < dims; d++)
      window_sum[d] += cell[d];
  }

  // Store normalized average for index 0
  for (int d = 0; d < dims; d++)
    grid.filtered_som_buf[d] = window_sum[d] * inv_filter_size;

  // Slide window across remaining array elements (subtract outgoing cell, add incoming cell)
  for (int i = 1; i < size; i++) {
    const float *left_cell = base_ptr + (i - 1) * dims;
    const float *right_cell = base_ptr + (i - 1 + filter_size) * dims;

    for (int d = 0; d < dims; d++) {
      window_sum[d] += right_cell[d] - left_cell[d];
      grid.filtered_som_buf[i * dims + d] = window_sum[d] * inv_filter_size;
    }
  }

  // Copy filtered SOM back into main SOM buffer
  std::copy_n(grid.filtered_som_buf.data(), ctx.count * ctx.dim, grid.som());
}

// Calculates pairwise distance matrix between swap candidates and quantizes to [0, QUANT].
inline void calc_dist_lut_int(const FlasContext &ctx, SwapBuffers &swaps, int num_swaps) {
  float max_val = 0.0f;
  const size_t dim_sz = static_cast<size_t>(ctx.dim);
  auto dist_func = ctx.dist_func;

  // Compute exact floating-point distance matrix between candidates and SOM cells
  for (int i = 0; i < num_swaps; i++) {
    for (int j = 0; j < num_swaps; j++) {
      float val = dist_func(swaps.fvs[i], swaps.som_fvs[j], &dim_sz);
      swaps.dist_lut_f[i * num_swaps + j] = val;
      if (val > max_val)
        max_val = val;
    }
  }

  // Normalize and quantize distances to integer range [0, 256] for fast JV solver processing
  if (max_val < 1e-10f) max_val = 1.0f;
  const float inv_max_val = static_cast<float>(QUANT) / max_val;
  for (int i = 0; i < num_swaps; i++) {
    for (int j = 0; j < num_swaps; j++) {
      swaps.dist_lut[i * num_swaps + j] = static_cast<int>(std::lround(swaps.dist_lut_f[i * num_swaps + j] * inv_max_val));
    }
  }
}

// Solves linear assignment for chosen swap positions using JV solver and updates element order.
inline void do_swaps(const FlasContext &ctx, SomGrid &grid, SwapBuffers &swaps, int num_swaps) {
  if (num_swaps == 0) return;

  // Snapshot candidate MapFields and feature/SOM pointers
  for (int i = 0; i < num_swaps; i++) {
    int swap_position = swaps.swap_positions[i];
    MapField &swapped_element = ctx.map_fields[swap_position];
    swaps.swapped_elements[i] = swapped_element;
    swaps.fvs[i] = swapped_element.feature;
    swaps.som_fvs[i] = grid.row(swap_position, ctx.dim);
  }

  // Calculate distance matrix and solve optimal linear assignment problem via JV algorithm
  calc_dist_lut_int(ctx, swaps, num_swaps);
  compute_assignment(swaps.dist_lut.data(), num_swaps, swaps.jv_scratch);
  const int *permutation = swaps.jv_scratch.perm();

  // Write optimal permutation back into map_fields
  for (int i = 0; i < num_swaps; i++) {
    ctx.map_fields[swaps.swap_positions[permutation[i]]] = swaps.swapped_elements[i];
  }
}

// Selects candidate swap positions within a localized random window.
inline int find_swap_positions_1d(const FlasContext &ctx, SwapBuffers &swaps, std::span<const int> swap_indices, int num_swap_indices) {
  // Pick random center x0 and clamp window range [x_start, x_start + num_swap_indices]
  std::uniform_int_distribution<int> pos_dist(0, ctx.count - 1);
  int x0 = pos_dist(ctx.rng);

  int x_start = std::max(0, std::min(x0 - num_swap_indices / 2, ctx.count - num_swap_indices));

  // Determine starting index in shuffled offsets to select up to max_sp candidate positions
  int max_sp = swaps.num_swap_positions();
  int start_index = 0;
  if (num_swap_indices > max_sp) {
    std::uniform_int_distribution<int> index_dist(0, num_swap_indices - max_sp - 1);
    start_index = index_dist(ctx.rng);
  }

  // Collect candidate swap positions
  int num_swap_positions = 0;
  for (int j = start_index; j < num_swap_indices && num_swap_positions < max_sp; j++) {
    int dx = swap_indices[j];
    int pos = std::min(ctx.count - 1, std::max(0, x_start + dx));
    swaps.swap_positions[num_swap_positions++] = pos;
  }
  return num_swap_positions;
}

// Runs one iteration of localized random swap probes for the given radius.
inline void check_random_swaps_1d(const FlasContext &ctx, SomGrid &grid, SwapBuffers &swaps, int radius, float sample_factor) {
  int max_sp = swaps.num_swap_positions();
  if (max_sp == 0)
    return;

  // Determine active swap window size based on current neighborhood radius
  int num_swap_indices = std::min(2 * radius + 1, ctx.count);
  while (num_swap_indices < max_sp && num_swap_indices < ctx.count) {
    num_swap_indices++;
  }

  // Resize index buffer if needed and initialize indices [0..num_swap_indices-1]
  if (swaps.swap_indices.size() < static_cast<size_t>(num_swap_indices)) {
    swaps.swap_indices.resize(static_cast<size_t>(num_swap_indices));
  }

  for (int i = 0; i < num_swap_indices; i++)
    swaps.swap_indices[i] = i;

  // Shuffle active indices range to select randomized candidate subsets
  std::span<int> active_indices(swaps.swap_indices.data(), static_cast<size_t>(num_swap_indices));
  shuffle_array(active_indices, ctx.rng);

  // Perform multiple swap probes according to sample_factor
  int num_swap_tries = std::max(1, static_cast<int>(sample_factor * static_cast<float>(ctx.count) / static_cast<float>(max_sp)));
  for (int n = 0; n < num_swap_tries; n++) {
    int num_swaps = find_swap_positions_1d(ctx, swaps, active_indices, num_swap_indices);
    do_swaps(ctx, grid, swaps, num_swaps);
  }
}

// Main 1D FLAS sorter: repeatedly copies to SOM, applies moving-average filter, and performs random swaps.
inline void do_sorting_1d(
  std::span<MapField> map_fields, int dim, const FlasSettings &settings, RandomEngine &rng,
  const std::function<bool(float)>& progress_callback
) {
  int count = static_cast<int>(map_fields.size());
  if (count <= 0) return;
  float rad = static_cast<float>(count) * settings.initial_radius_factor;

  const int num_iterations = static_cast<int>(ceil(-log(rad / settings.radius_end) / log(settings.radius_decay)));
  int iteration_counter = 0;
  if (progress_callback && progress_callback(0.f))
    return;

  FlasContext ctx(map_fields, count, dim, rng, settings.metric);
  SomGrid grid(count, dim);
  SwapBuffers swaps(std::min(count, settings.max_swap_positions), count);

  do {
    // 1. Copy current feature vectors to SOM grid
    copy_feature_vectors_to_som(ctx, grid);

    int radius = std::max(1, static_cast<int>(std::round(rad)));
    int radius_1d = std::max(1, std::min(count / 2, radius));
    rad *= settings.radius_decay;

    // 2. Apply 1D sliding-window moving-average filter
    for (int i = 0; i < settings.num_filters; i++)
      filter_weighted_som_1d(radius_1d, ctx, grid);

    // 3. Perform localized linear assignment swaps
    check_random_swaps_1d(ctx, grid, swaps, radius, settings.sample_factor);

    // 4. Report progress and evaluate early termination callback
    iteration_counter++;
    float progress = static_cast<float>(iteration_counter) / static_cast<float>(num_iterations);
    if (progress_callback && progress_callback(progress))
      break;
  } while (rad > settings.radius_end);
}

} // namespace flas

#endif // FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H
