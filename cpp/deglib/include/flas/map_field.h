// Apache 2.0 License — Visual Computing Group, HTW Berlin
#ifndef EVP_FLAS_MAP_FIELD_H
#define EVP_FLAS_MAP_FIELD_H

struct MapField {
  int id;
  const float *feature;
  bool is_swappable;
};

inline void init_map_field(MapField *map_field, const int id, const float *const feature, const bool is_swappable) {
  map_field->id = id;
  map_field->feature = feature;
  map_field->is_swappable = is_swappable;
}

inline void init_invalid_map_field(MapField *map_field, const bool is_swappable) {
  map_field->id = -1;
  map_field->feature = nullptr;
  map_field->is_swappable = is_swappable;
}

inline int get_num_swappable(const MapField *map_fields, const int num_map_fields) {
  int num_swappable = 0;
  for (int i = 0; i < num_map_fields; i++) {
    if (map_fields[i].is_swappable) {
      num_swappable++;
    }
  }
  return num_swappable;
}

#endif // EVP_FLAS_MAP_FIELD_H
