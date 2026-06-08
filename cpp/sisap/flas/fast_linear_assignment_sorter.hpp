//
// Created by Bruno Schilling on 10/28/24.
// Ported from https://github.com/Visual-Computing/DynamicExplorationGraph/tree/cb7243f7296ef4513b8a5177773a7f30826c5f7b/java/deg-visualization/src/main/java/com/vc/deg/viz/om
//
// Apache 2.0 License — Visual Computing Group, HTW Berlin
//
// Adapted for pure C++ usage (pybind11 dependency removed).
//

#ifndef EVP_FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H
#define EVP_FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H

#include <random>
#include <cstring>
#include <cmath>
#include <functional>

#include <distances.h>
#include <distance/fp32_l2.h>
#include <distance/fp32_inner_product.h>

#include "map_field.hpp"
#include "junker_volgenant_solver.hpp"

typedef std::mt19937 RandomEngine;
constexpr int QUANT = 256;

enum class FlasMetric { L2, InnerProduct };

// ------------------- UTILS -------------------
inline int min(const int a, const int b) {
  return a < b ? a : b;
}

inline int max(const int a, const int b) {
  return a > b ? a : b;
}

// ------------------- FLAS SETTINGS -------------------
class FlasSettings {
public:
  FlasSettings(bool do_wrap, float initial_radius_factor, float radius_decay, int num_filters, float radius_end,
    float weight_swappable, float weight_non_swappable, float weight_hole, float sample_factor, int max_swap_positions,
    int optimize_narrow_grids)
    : do_wrap(do_wrap),
      initial_radius_factor(initial_radius_factor),
      radius_decay(radius_decay),
      num_filters(num_filters),
      radius_end(radius_end),
      weight_swappable(weight_swappable),
      weight_non_swappable(weight_non_swappable),
      weight_hole(weight_hole),
      sample_factor(sample_factor),
      max_swap_positions(max_swap_positions),
      optimize_narrow_grids(optimize_narrow_grids)
  { }

  bool do_wrap;
  float initial_radius_factor;
  float radius_decay;
  int num_filters;
  float radius_end;
  float weight_swappable;
  float weight_non_swappable;
  float weight_hole;
  float sample_factor;
  int max_swap_positions;
  int optimize_narrow_grids;
  FlasMetric metric = FlasMetric::L2;
};

inline FlasSettings default_flas_settings() {
  return {false, 0.5f, 0.93f, 1, 1.0f, 1.0f, 100.f, 0.01f, 1.f, 9, 1};
}

// ------------------- INTERNAL DATA -------------------
struct InternalData {
  int columns;
  int rows;
  int grid_size;
  int dim;
  int num_swap_positions;
  MapField *map_fields;
  float *som;
  float *weights;
  int *swap_positions;
  const float **fvs;
  const float **som_fvs;
  MapField *swapped_elements;
  int *dist_lut;
  float *dist_lut_f;
  RandomEngine* rng;
  FlasMetric metric = FlasMetric::L2;
};

inline InternalData create_internal_data(MapField *map_fields, int columns, int rows, int dim, int max_swap_positions, RandomEngine* rng) {
  InternalData data{};

  data.columns = columns;
  data.rows = rows;
  data.grid_size = columns * rows;
  data.dim = dim;
  int num_valid_map_fields = get_num_swappable(map_fields, data.grid_size);

  data.map_fields = map_fields;

  data.som = static_cast<float *>(calloc(data.grid_size * dim, sizeof(float)));
  if (data.som == nullptr) {
    std::cerr << "Failed to allocate som.\n" << std::endl;
    exit(1);
  }

  data.weights = static_cast<float *>(malloc(data.grid_size * sizeof(float)));
  if (data.weights == nullptr) {
    std::cerr << "Failed to allocate weights.\n" << std::endl;
    exit(1);
  }

  data.num_swap_positions = min(max_swap_positions, num_valid_map_fields);

  data.swap_positions = static_cast<int *>(malloc(data.num_swap_positions * sizeof(int)));
  if (data.swap_positions == nullptr) {
    std::cerr << "Failed to allocate swap_positions.\n" << std::endl;
    exit(1);
  }

  data.fvs = static_cast<const float **>(malloc(data.num_swap_positions * sizeof(float *)));
  if (data.fvs == nullptr) {
    std::cerr << "Failed to allocate fvs.\n" << std::endl;
    exit(1);
  }

  data.som_fvs = static_cast<const float **>(malloc(data.num_swap_positions * sizeof(float *)));
  if (data.som_fvs == nullptr) {
    std::cerr << "Failed to allocate som_fvs.\n" << std::endl;
    exit(1);
  }

  data.swapped_elements = static_cast<MapField *>(malloc(data.num_swap_positions * sizeof(MapField)));
  if (data.swapped_elements == nullptr) {
    std::cerr << "Failed to allocate swapped_elements.\n" << std::endl;
    exit(1);
  }
  memcpy(data.swapped_elements, data.map_fields, data.num_swap_positions * sizeof(MapField));

  data.dist_lut = static_cast<int *>(malloc(data.num_swap_positions * data.num_swap_positions * sizeof(int)));
  if (data.dist_lut == nullptr) {
    std::cerr << "Failed to allocate dist_lut.\n" << std::endl;
    exit(1);
  }

  data.dist_lut_f = static_cast<float *>(malloc(data.num_swap_positions * data.num_swap_positions * sizeof(float)));
  if (data.dist_lut_f == nullptr) {
    std::cerr << "Failed to allocate dist_lut_f.\n" << std::endl;
    exit(1);
  }

  data.rng = rng;

  return data;
}

inline void free_internal_data(const InternalData *data) {
  free(data->som);
  free(data->weights);
  free(data->swap_positions);
  free(data->fvs);
  free(data->som_fvs);
  free(data->swapped_elements);
  free(data->dist_lut);
  free(data->dist_lut_f);
}


// ------------------- SORTING -------------------
inline void copy_feature_vectors_to_som(const InternalData *data, const FlasSettings *settings) {
  for (int pos = 0; pos < data->grid_size; pos++) {
    float *som_cell = data->som + (pos * data->dim);
    const MapField *cell = data->map_fields + pos;

    // handle holes
    if (cell->id > -1) {
      const float *const fv = cell->feature;

      // higher weight for fixed images
      float w = cell->is_swappable ? settings->weight_swappable : settings->weight_non_swappable;
      for (int i = 0; i < data->dim; i++)
        som_cell[i] = w * fv[i];
      data->weights[pos] = w;
    } else {
      for (int i = 0; i < data->dim; i++)
        som_cell[i] *= settings->weight_hole;
      data->weights[pos] = settings->weight_hole;
    }
  }
}

inline void filter_h_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (columns == 1) {
    memcpy(output, input, static_cast<size_t>(rows) * columns * dims * sizeof(float));
    return;
  }

  int ext = filter_size / 2;

  auto row_ext = static_cast<float **>(malloc((columns + 2 * ext) * sizeof(float *)));
  if (row_ext == nullptr) {
    std::cerr << "Failed to allocate row_ext.\n" << std::endl;
    exit(1);
  }

  auto tmp = static_cast<float *>(malloc(dims * sizeof(float)));
  if (tmp == nullptr) {
    std::cerr << "Failed to allocate tmp.\n" << std::endl;
    exit(1);
  }

  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = &input[(act_row + i) * dims];

    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[columns + ext - i - 1];
      row_ext[columns + ext + i] = row_ext[ext + i];
    }

    memset(tmp, 0, dims * sizeof(float));

    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += row_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[act_row * dims + d] = tmp[d] / static_cast<float>(filter_size);

    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += row_ext[right][d] - row_ext[left][d];
        output[(act_row + i) * dims + d] = tmp[d] / static_cast<float>(filter_size);
      }
    }
  }

  free(tmp);
  free(row_ext);
}

inline void filter_h_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (columns == 1) {
    memcpy(output, input, static_cast<size_t>(rows) * columns * sizeof(float));
    return;
  }

  int ext = filter_size / 2;

  auto row_ext = static_cast<float *>(malloc((columns + 2 * ext) * sizeof(float)));
  if (row_ext == nullptr) {
    std::cerr << "Failed to allocate row_ext.\n" << std::endl;
    exit(1);
  }

  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = input[act_row + i];

    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[columns + ext - i - 1];
      row_ext[columns + ext + i] = row_ext[ext + i];
    }

    float tmp = 0.0f;

    for (int i = 0; i < filter_size; i++)
      tmp += row_ext[i];

    output[act_row] = tmp / static_cast<float>(filter_size);

    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += row_ext[right] - row_ext[left];
      output[act_row + i] = tmp / static_cast<float>(filter_size);
    }
  }

  free(row_ext);
}

inline void filter_v_wrap(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2;

  auto col_ext = static_cast<float **>(malloc((rows + 2 * ext) * sizeof(float *)));
  if (col_ext == nullptr) {
    std::cerr << "Failed to allocate col_ext.\n" << std::endl;
    exit(1);
  }
  auto tmp = static_cast<float *>(malloc(dims * sizeof(float)));
  if (tmp == nullptr) {
    std::cerr << "Failed to allocate tmp.\n" << std::endl;
    exit(1);
  }

  for (int x = 0; x < columns; x++) {
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = &input[(x + i * columns) * dims];

    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[rows + ext - i - 1];
      col_ext[rows + ext + i] = col_ext[ext + i];
    }

    memset(tmp, 0, dims * sizeof(float));

    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += col_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[(x * dims) + d] = tmp[d] / static_cast<float>(filter_size);

    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += col_ext[right][d] - col_ext[left][d];
        output[(x + i * columns) * dims + d] = tmp[d] / static_cast<float>(filter_size);
      }
    }
  }

  free(tmp);
  free(col_ext);
}

inline void filter_v_wrap_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2;

  auto col_ext = static_cast<float *>(malloc((rows + 2 * ext) * sizeof(float)));
  if (col_ext == nullptr) {
    std::cerr << "Failed to allocate col_ext.\n" << std::endl;
    exit(1);
  }

  for (int x = 0; x < columns; x++) {
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = input[x + i * columns];

    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[rows + ext - i - 1];
      col_ext[rows + ext + i] = col_ext[ext + i];
    }

    float tmp = 0.0f;

    for (int i = 0; i < filter_size; i++)
      tmp += col_ext[i];

    output[x] = tmp / static_cast<float>(filter_size);

    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += col_ext[right] - col_ext[left];
      output[x + i * columns] = tmp / static_cast<float>(filter_size);
    }
  }

  free(col_ext);
}

inline void filter_h_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (columns == 1) {
    memcpy(output, input, static_cast<size_t>(rows) * columns * dims * sizeof(float));
    return;
  }

  int ext = filter_size / 2;

  auto row_ext = static_cast<float **>(malloc((columns + 2 * ext) * sizeof(float *)));
  if (row_ext == nullptr) {
    std::cerr << "Failed to allocate row_ext.\n" << std::endl;
    exit(1);
  }
  auto tmp = static_cast<float *>(malloc(dims * sizeof(float)));
  if (tmp == nullptr) {
    std::cerr << "Failed to allocate tmp.\n" << std::endl;
    exit(1);
  }

  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = &input[(act_row + i) * dims];

    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[ext + i + 1];
      row_ext[columns + ext + i] = row_ext[columns + ext - 2 - i];
    }

    memset(tmp, 0, dims * sizeof(float));

    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += row_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[act_row * dims + d] = tmp[d] / static_cast<float>(filter_size);

    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += row_ext[right][d] - row_ext[left][d];
        output[(act_row + i) * dims + d] = tmp[d] / static_cast<float>(filter_size);
      }
    }
  }

  free(tmp);
  free(row_ext);
}

inline void filter_h_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (columns == 1) {
    memcpy(output, input, static_cast<size_t>(rows) * columns * sizeof(float));
    return;
  }

  int ext = filter_size / 2;

  auto row_ext = static_cast<float *>(malloc((columns + 2 * ext) * sizeof(float)));
  if (row_ext == nullptr) {
    std::cerr << "Failed to allocate row_ext.\n" << std::endl;
    exit(1);
  }

  for (int y = 0; y < rows; y++) {
    int act_row = y * columns;

    for (int i = 0; i < columns; i++)
      row_ext[i + ext] = input[act_row + i];

    for (int i = 0; i < ext; i++) {
      row_ext[ext - 1 - i] = row_ext[ext + i + 1];
      row_ext[columns + ext + i] = row_ext[columns + ext - 2 - i];
    }

    float tmp = 0.0f;

    for (int i = 0; i < filter_size; i++)
      tmp += row_ext[i];

    output[act_row] = tmp / static_cast<float>(filter_size);

    for (int i = 1; i < columns; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += row_ext[right] - row_ext[left];
      output[act_row + i] = tmp / static_cast<float>(filter_size);
    }
  }

  free(row_ext);
}

inline void filter_v_mirror(float *input, float *output, int rows, int columns, int dims, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2;

  auto col_ext = static_cast<float **>(malloc((rows + 2 * ext) * sizeof(float *)));
  if (col_ext == nullptr) {
    std::cerr << "Failed to allocate col_ext.\n" << std::endl;
    exit(1);
  }
  auto tmp = static_cast<float *>(malloc(dims * sizeof(float)));
  if (tmp == nullptr) {
    std::cerr << "Failed to allocate tmp.\n" << std::endl;
    exit(1);
  }

  for (int x = 0; x < columns; x++) {
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = &input[(x + i * columns) * dims];

    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[ext + i + 1];
      col_ext[rows + ext + i] = col_ext[ext + rows - 2 - i];
    }

    memset(tmp, 0, dims * sizeof(float));

    for (int i = 0; i < filter_size; i++)
      for (int d = 0; d < dims; d++)
        tmp[d] += col_ext[i][d];

    for (int d = 0; d < dims; d++)
      output[x * dims + d] = tmp[d] / static_cast<float>(filter_size);

    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;

      for (int d = 0; d < dims; d++) {
        tmp[d] += col_ext[right][d] - col_ext[left][d];
        output[(x + i * columns) * dims + d] = tmp[d] / static_cast<float>(filter_size);
      }
    }
  }

  free(tmp);
  free(col_ext);
}

inline void filter_v_mirror_1d(const float *input, float *output, int rows, int columns, int filter_size) {
  if (rows == 1)
    return;

  int ext = filter_size / 2;

  auto col_ext = static_cast<float *>(malloc((rows + 2 * ext) * sizeof(float)));
  if (col_ext == nullptr) {
    std::cerr << "Failed to allocate col_ext.\n" << std::endl;
    exit(1);
  }

  for (int x = 0; x < columns; x++) {
    for (int i = 0; i < rows; i++)
      col_ext[i + ext] = input[x + i * columns];

    for (int i = 0; i < ext; i++) {
      col_ext[ext - 1 - i] = col_ext[ext + i + 1];
      col_ext[rows + ext + i] = col_ext[ext + rows - 2 - i];
    }

    float tmp = 0.0f;

    for (int i = 0; i < filter_size; i++)
      tmp += col_ext[i];

    output[x] = tmp / static_cast<float>(filter_size);

    for (int i = 1; i < rows; i++) {
      int left = i - 1;
      int right = left + filter_size;
      tmp += col_ext[right] - col_ext[left];
      output[x + i * columns] = tmp / static_cast<float>(filter_size);
    }
  }

  free(col_ext);
}


inline void filter_weighted_som(
  int act_radius_x, int act_radius_y, const InternalData *data, bool do_wrap
) {
  int filter_size_x = 2 * act_radius_x + 1;
  int filter_size_y = 2 * act_radius_y + 1;

  auto som_h = static_cast<float *>(malloc(data->grid_size * data->dim * sizeof(float)));
  if (som_h == nullptr) {
    std::cerr << "Failed to allocate som_h.\n" << std::endl;
    exit(1);
  }
  auto weights_h = static_cast<float *>(malloc(data->grid_size * sizeof(float)));
  if (weights_h == nullptr) {
    std::cerr << "Failed to allocate weights_h.\n" << std::endl;
    exit(1);
  }

  if (do_wrap) {
    filter_h_wrap(data->som, som_h, data->rows, data->columns, data->dim, filter_size_x);
    filter_h_wrap_1d(data->weights, weights_h, data->rows, data->columns, filter_size_x);

    filter_v_wrap(som_h, data->som, data->rows, data->columns, data->dim, filter_size_y);
    filter_v_wrap_1d(weights_h, data->weights, data->rows, data->columns, filter_size_y);
  } else {
    filter_h_mirror(data->som, som_h, data->rows, data->columns, data->dim, filter_size_x);
    filter_h_mirror_1d(data->weights, weights_h, data->rows, data->columns, filter_size_x);

    filter_v_mirror(som_h, data->som, data->rows, data->columns, data->dim, filter_size_y);
    filter_v_mirror_1d(weights_h, data->weights, data->rows, data->columns, filter_size_y);
  }
  free(som_h);
  free(weights_h);

  for (int i = 0; i < data->grid_size; i++) {
    float w = 1.f / data->weights[i];
    for (int d = 0; d < data->dim; d++) {
      *(data->som + i * data->dim + d) *= w;
    }
  }
}

inline void shuffle_array(int *array, int size, RandomEngine* rng) {
  for (int i = size - 1; i > 0; i--) {
    std::uniform_int_distribution<uint32_t> index_dist(0, i);
    const unsigned int index = index_dist(*rng);

    int temp = array[index];
    array[index] = array[i];
    array[i] = temp;
  }
}

inline int find_swap_positions_wrap(const InternalData *data, const int *swap_indices, const int num_swap_indices) {
  std::uniform_int_distribution<uint32_t> index_dist(0, num_swap_indices - data->num_swap_positions - 1);
  unsigned int start_index = (num_swap_indices - data->num_swap_positions > 0) ?
                      index_dist(*data->rng)
                      : 0;
  std::uniform_int_distribution<int32_t> pos_dist(0, data->grid_size - 1);
  const int pos0 = pos_dist(*data->rng);

  int swap_pos = 0;
  for (unsigned int j = start_index; j < num_swap_indices && swap_pos < data->num_swap_positions; j++) {
    int d = pos0 + swap_indices[j];
    int x = d % data->columns;
    int y = (d / data->columns) % data->rows;
    int pos = y * data->columns + x;

    if (data->map_fields[pos].is_swappable) {
      data->swap_positions[swap_pos++] = pos;
    }
  }
  return swap_pos;
}

template<class T>
T get_squared_l2_distance(const T *fv1, const T *fv2, int dim) {
  T dist = 0;
  for (int i = 0; i < dim; i++) {
    T d = fv1[i] - fv2[i];
    dist += d * d;
  }
  return dist;
}

template<class T>
T get_l2_distance(const T *fv1, const T *fv2, int dim) {
  return sqrt(get_squared_l2_distance<T>(fv1, fv2, dim));
}

inline void calc_dist_lut_int(const InternalData *data, int num_swaps) {
  float max = 0;
  const size_t dim_sz = static_cast<size_t>(data->dim);
  for (int i = 0; i < num_swaps; i++) {
    if (data->metric == FlasMetric::InnerProduct) {
      deglib::distances::compare_batch<deglib::distances::InnerProductFloat>(
        data->fvs[i], reinterpret_cast<const void* const*>(data->som_fvs), num_swaps, &dim_sz, &data->dist_lut_f[i * num_swaps]);
    } else {
      deglib::distances::compare_batch<deglib::distances::L2Float>(
        data->fvs[i], reinterpret_cast<const void* const*>(data->som_fvs), num_swaps, &dim_sz, &data->dist_lut_f[i * num_swaps]);
    }
    for (int j = 0; j < num_swaps; j++) {
      float val = data->dist_lut_f[i * num_swaps + j];
      if (val > max)
        max = val;
    }
  }

  if (max < 1e-10f) max = 1.0f; // avoid division by zero
  for (int i = 0; i < num_swaps; i++)
    for (int j = 0; j < num_swaps; j++) {
      data->dist_lut[i * num_swaps + j] = static_cast<int>(roundf(static_cast<float>(QUANT) * data->dist_lut_f[i * num_swaps + j] / max));
    }
}

inline void do_swaps(const InternalData *data, int num_swaps) {
  int num_valid = 0;
  for (int i = 0; i < num_swaps; i++) {
    int swap_position = data->swap_positions[i];
    MapField *swapped_element = &data->map_fields[swap_position];
    data->swapped_elements[i] = *swapped_element;

    if (swapped_element->id > -1) {
      data->fvs[i] = swapped_element->feature;
      num_valid++;
    } else {
      data->fvs[i] = &data->som[swap_position * data->dim];
    }

    data->som_fvs[i] = &data->som[swap_position * data->dim];
  }

  if (num_valid > 0) {
    calc_dist_lut_int(data, num_swaps);
    int *permutation = compute_assignment(data->dist_lut, num_swaps);

    for (int i = 0; i < num_swaps; i++) {
      data->map_fields[data->swap_positions[permutation[i]]] = data->swapped_elements[i];
    }

    free(permutation);
  }
}

inline int find_swap_positions(const InternalData *data, const int *swap_indices, int num_swap_indices, int swap_area_width,
                         int swap_area_height) {
  // calculate start position of swap area
  std::uniform_int_distribution<int> pos_dist(0, data->grid_size - 1);
  int pos0 = pos_dist(*data->rng);
  int x0 = pos0 % data->columns;
  int y0 = pos0 / data->columns;

  int x_start = max(0, x0 - swap_area_width / 2);
  int y_start = max(0, y0 - swap_area_height / 2);
  if (x_start + swap_area_width > data->columns)
    x_start = data->columns - swap_area_width;
  if (y_start + swap_area_height > data->rows)
    y_start = data->rows - swap_area_height;

  std::uniform_int_distribution<int> index_dist(0, num_swap_indices - data->num_swap_positions - 1);
  int start_index = num_swap_indices - data->num_swap_positions > 0 ?
                     index_dist(*data->rng)
                     : 0;
  int num_swap_positions = 0;
  for (int j = start_index; j < num_swap_indices && num_swap_positions < data->num_swap_positions; j++) {
    int dx = swap_indices[j] % data->columns;
    int dy = swap_indices[j] / data->columns;

    int x = (x_start + dx) % data->columns;
    int y = (y_start + dy) % data->rows;
    int pos = y * data->columns + x;

    if (data->map_fields[pos].is_swappable) {
      data->swap_positions[num_swap_positions++] = pos;
    }
  }
  return num_swap_positions;
}

inline void check_random_swaps(const InternalData *data, int radius, float sample_factor, bool do_wrap) {
  if (data->num_swap_positions == 0)
    return;
  int swap_area_width = min(2 * radius + 1, data->columns);
  int swap_area_height = min(2 * radius + 1, data->rows);
  for (int k = 0; swap_area_height * swap_area_width < data->num_swap_positions; k++) {
    if ((k & 0x1) == 0)
      swap_area_width = min(swap_area_width + 1, data->columns);
    else
      swap_area_height = min(swap_area_height + 1, data->rows);
  }

  const int num_swap_indices = swap_area_width * swap_area_height;
  int *swap_indices = static_cast<int *>(malloc(num_swap_indices * sizeof(int)));
  if (swap_indices == nullptr) {
    std::cerr << "Failed to allocate swap_indices.\n" << std::endl;
    exit(1);
  }

  int i = 0;
  for (int y = 0; y < swap_area_height; y++)
    for (int x = 0; x < swap_area_width; x++)
      swap_indices[i++] = y * data->columns + x;
  shuffle_array(swap_indices, num_swap_indices, data->rng);

  int num_swap_tries = max(1, static_cast<int>(sample_factor) * data->grid_size / data->num_swap_positions);
  if (do_wrap) {
    for (int n = 0; n < num_swap_tries; n++) {
      int num_swaps = find_swap_positions_wrap(data, swap_indices, num_swap_indices);
      do_swaps(data, num_swaps);
    }
  } else {
    for (int n = 0; n < num_swap_tries; n++) {
      int num_swaps = find_swap_positions(data, swap_indices, num_swap_indices, swap_area_width, swap_area_height);
      do_swaps(data, num_swaps);
    }
  }
  free(swap_indices);
}

/**
 * @param map_fields Array of MapFields with length columns * rows. If a map field has id == -1, it is not used for
 *                   swapping.
 * @param dim The dimensionality of the features
 * @param columns Number of columns in the grid to sort
 * @param rows Number of rows in the grid to sort
 * @param settings The settings of the sorting algorithm
 * @param rng The RandomEngine to use for pseudo number generation
 * @param callback A callback function that is called every iteration. The argument is the progress between 0 and 1. If
 *                 true is returned, the algorithm will stop.
 */
inline void do_sorting_full(
  MapField *map_fields, int dim, int columns, int rows, const FlasSettings *settings, RandomEngine* rng,
  const std::function<bool(float)>& callback
) {
  float rad = static_cast<float>(max(columns, rows)) * settings->initial_radius_factor;

  const int num_iterations = static_cast<int>(ceil(-log(rad / settings->radius_end) / log(settings->radius_decay)));
  int iteration_counter = 0;
  if (callback(0.f))
    return;

  int optimize_narrow = settings->optimize_narrow_grids;
  if (optimize_narrow == 1) {
    float aspect_ratio = static_cast<float>(columns) / static_cast<float>(rows);
    if (aspect_ratio > 0.1f) {
      optimize_narrow = 0;
    }
  }

  InternalData data = create_internal_data(map_fields, columns, rows, dim, settings->max_swap_positions, rng);
  data.metric = settings->metric;

  do {
    copy_feature_vectors_to_som(&data, settings);

    int radius = max(1, static_cast<int>(std::round(rad)));
    int radius_x;
    int radius_y;
    if (optimize_narrow) {
      radius_x = max(static_cast<int>(static_cast<float>(columns) * 0.8f), columns-2);
      radius_y = min(rows-1, radius);
    } else {
      radius_x = max(1, min(columns / 2, radius));
      radius_y = max(1, min(rows / 2, radius));
    }
    rad *= settings->radius_decay;

    for (int i = 0; i < settings->num_filters; i++)
      filter_weighted_som(radius_x, radius_y, &data, settings->do_wrap);

    check_random_swaps(&data, radius, settings->sample_factor, settings->do_wrap);

    iteration_counter++;
    float progress = static_cast<float>(iteration_counter) / static_cast<float>(num_iterations);
    if (callback(progress))
      break;
  } while (rad > settings->radius_end);

  free_internal_data(&data);
}

#endif // EVP_FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H
