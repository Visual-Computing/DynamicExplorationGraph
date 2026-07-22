// Apache 2.0 License — Visual Computing Group, HTW Berlin
#ifndef EVP_FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H
#define EVP_FLAS_FAST_LINEAR_ASSIGNMENT_SORTER_H

#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <functional>
#include <iostream>
#include <random>

#include "map_field.h"
#include "junker_volgenant_solver.h"
#include "distances.h"

using namespace std;

typedef std::mt19937 RandomEngine;

constexpr int QUANT = 1000;

enum class FlasMetric {
  L2 = 0,
  InnerProduct = 1
};

struct FlasSettings {
  float initial_radius_factor = 0.5f;
  float radius_decay = 0.9f;
  float radius_end = 1.0f;
  int num_filters = 1;
  int max_swap_positions = 50;
  float sample_factor = 1.0f;
  int do_wrap = 0;
  int optimize_narrow_grids = 1;
  FlasMetric metric = FlasMetric::L2;
};

inline void init_flas_settings(FlasSettings *settings) {
  settings->initial_radius_factor = 0.5f;
  settings->radius_decay = 0.9f;
  settings->radius_end = 1.0f;
  settings->num_filters = 1;
  settings->max_swap_positions = 50;
  settings->sample_factor = 1.0f;
  settings->do_wrap = 0;
  settings->optimize_narrow_grids = 1;
  settings->metric = FlasMetric::L2;
}

struct InternalData {
  MapField *map_fields;
  int columns;
  int rows;
  int dim;
  int grid_size;
  int max_swap_positions;

  float *som;
  float *weights_x;
  float *weights_y;
  float *temp_som;
  int *swap_positions;

  MapField *swapped_elements;
  const float **fvs;
  const float **som_fvs;
  int *dist_lut;
  float *dist_lut_f;

  int num_swap_positions;
  RandomEngine* rng;

  FlasMetric metric;
};

inline InternalData create_internal_data(MapField *map_fields, int columns, int rows, int dim, int max_swap_positions, RandomEngine *rng) {
  InternalData data;
  data.map_fields = map_fields;
  data.columns = columns;
  data.rows = rows;
  data.dim = dim;
  data.grid_size = columns * rows;
  data.max_swap_positions = max_swap_positions;
  data.rng = rng;
  data.metric = FlasMetric::L2;

  data.som = static_cast<float *>(malloc(data.grid_size * dim * sizeof(float)));
  data.weights_x = static_cast<float *>(malloc(columns * sizeof(float)));
  data.weights_y = static_cast<float *>(malloc(rows * sizeof(float)));
  data.temp_som = static_cast<float *>(malloc(data.grid_size * dim * sizeof(float)));
  data.swap_positions = static_cast<int *>(malloc(max_swap_positions * sizeof(int)));

  data.swapped_elements = static_cast<MapField *>(malloc(max_swap_positions * sizeof(MapField)));
  data.fvs = static_cast<const float **>(malloc(max_swap_positions * sizeof(float *)));
  data.som_fvs = static_cast<const float **>(malloc(max_swap_positions * sizeof(float *)));
  data.dist_lut = static_cast<int *>(malloc(max_swap_positions * max_swap_positions * sizeof(int)));
  data.dist_lut_f = static_cast<float *>(malloc(max_swap_positions * max_swap_positions * sizeof(float)));

  int num_swappable = get_num_swappable(map_fields, data.grid_size);
  data.num_swap_positions = min(num_swappable, max_swap_positions);

  return data;
}

inline void free_internal_data(InternalData *data) {
  free(data->som);
  free(data->weights_x);
  free(data->weights_y);
  free(data->temp_som);
  free(data->swap_positions);

  free(data->swapped_elements);
  free(data->fvs);
  free(data->som_fvs);
  free(data->dist_lut);
  free(data->dist_lut_f);
}

inline void shuffle_array(int *array, int size, RandomEngine *rng) {
  for (int i = size - 1; i > 0; --i) {
    std::uniform_int_distribution<int> dist(0, i);
    int j = dist(*rng);
    swap(array[i], array[j]);
  }
}

inline void copy_feature_vectors_to_som(InternalData *data, const FlasSettings *settings) {
  for (int i = 0; i < data->grid_size; i++) {
    MapField map_field = data->map_fields[i];
    if (map_field.id > -1) {
      copy_n(map_field.feature, data->dim, &data->som[i * data->dim]);
    } else {
      std::uniform_int_distribution<int> dist(0, data->grid_size - 1);
      int random_index = dist(*data->rng);

      while (data->map_fields[random_index].id == -1) {
        random_index = dist(*data->rng);
      }
      copy_n(data->map_fields[random_index].feature, data->dim, &data->som[i * data->dim]);
    }
  }
}

inline void calculate_weights(float *weights, int num_weights, int radius, bool do_wrap) {
  float sum = 0;
  for (int i = 0; i < num_weights; i++) {
    int d = abs(i - radius);
    if (do_wrap) {
      d = min(d, num_weights - d);
    }
    float w = exp(static_cast<float>(-d * d) / static_cast<float>(radius * radius));
    weights[i] = w;
    sum += w;
  }
  for (int i = 0; i < num_weights; i++) {
    weights[i] /= sum;
  }
}

inline void filter_weighted_som_x(int radius, InternalData *data, bool do_wrap) {
  int num_weights = 2 * radius + 1;
  calculate_weights(data->weights_x, num_weights, radius, do_wrap);

  for (int y = 0; y < data->rows; y++) {
    for (int x = 0; x < data->columns; x++) {
      int pos = y * data->columns + x;

      for (int i = 0; i < data->dim; i++) {
        data->temp_som[pos * data->dim + i] = 0;
      }

      for (int i = -radius; i <= radius; i++) {
        int neighbor_x = x + i;
        if (do_wrap) {
          neighbor_x = (neighbor_x + data->columns) % data->columns;
        } else {
          neighbor_x = max(0, min(data->columns - 1, neighbor_x));
        }

        int neighbor_pos = y * data->columns + neighbor_x;
        float weight = data->weights_x[i + radius];

        for (int d = 0; d < data->dim; d++) {
          data->temp_som[pos * data->dim + d] += data->som[neighbor_pos * data->dim + d] * weight;
        }
      }
    }
  }

  copy_n(data->temp_som, data->grid_size * data->dim, data->som);
}

inline void filter_weighted_som_y(int radius, InternalData *data, bool do_wrap) {
  int num_weights = 2 * radius + 1;
  calculate_weights(data->weights_y, num_weights, radius, do_wrap);

  for (int x = 0; x < data->columns; x++) {
    for (int y = 0; y < data->rows; y++) {
      int pos = y * data->columns + x;

      for (int i = 0; i < data->dim; i++) {
        data->temp_som[pos * data->dim + i] = 0;
      }

      for (int i = -radius; i <= radius; i++) {
        int neighbor_y = y + i;
        if (do_wrap) {
          neighbor_y = (neighbor_y + data->rows) % data->rows;
        } else {
          neighbor_y = max(0, min(data->rows - 1, neighbor_y));
        }

        int neighbor_pos = neighbor_y * data->columns + x;
        float weight = data->weights_y[i + radius];

        for (int d = 0; d < data->dim; d++) {
          data->temp_som[pos * data->dim + d] += data->som[neighbor_pos * data->dim + d] * weight;
        }
      }
    }
  }

  copy_n(data->temp_som, data->grid_size * data->dim, data->som);
}

inline void filter_weighted_som(int radius_x, int radius_y, InternalData *data, bool do_wrap) {
  filter_weighted_som_x(radius_x, data, do_wrap);
  filter_weighted_som_y(radius_y, data, do_wrap);
}

inline int find_swap_positions_wrap(const InternalData *data, const int *swap_indices, int num_swap_indices) {
  std::uniform_int_distribution<int32_t> index_dist(0, num_swap_indices - data->num_swap_positions > 0 ?
                                                       num_swap_indices - data->num_swap_positions - 1
                                                       : 0);
  const int start_index = index_dist(*data->rng);

  std::uniform_int_distribution<int32_t> pos_dist(0, data->grid_size - 1);
  const int pos0 = pos_dist(*data->rng);

  int swap_pos = 0;
  for (unsigned int j = start_index; j < static_cast<unsigned int>(num_swap_indices) && swap_pos < data->num_swap_positions; j++) {
    int d = pos0 + swap_indices[j];
    int x = d % data->columns;
    int y = (d / data->columns) % data->rows;
    int pos = y * data->columns + x;

    if (data->map_fields[pos].is_swappable) {
      bool duplicate = false;
      for (int k = 0; k < swap_pos; ++k) {
        if (data->swap_positions[k] == pos) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        data->swap_positions[swap_pos++] = pos;
      }
    }
  }
  return swap_pos;
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

  std::uniform_int_distribution<int> index_dist(0, num_swap_indices - data->num_swap_positions > 0 ?
                     num_swap_indices - data->num_swap_positions - 1
                     : 0);
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
      bool duplicate = false;
      for (int k = 0; k < num_swap_positions; ++k) {
        if (data->swap_positions[k] == pos) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) {
        data->swap_positions[num_swap_positions++] = pos;
      }
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

  int num_swap_tries = max(1, static_cast<int>(sample_factor * static_cast<float>(data->grid_size) / static_cast<float>(data->num_swap_positions)));
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
