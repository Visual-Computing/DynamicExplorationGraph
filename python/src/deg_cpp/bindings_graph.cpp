#include "bindings_common.h"

// ============================================================================
// Graph Search, Explore & Path Wrappers
// ============================================================================

template <typename QuerySpanT>
deglib::search::SearchResults execute_graph_search_single(
    const deglib::DynamicExplorationGraph& graph,
    QuerySpanT q_span,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const bool return_distances,
    const bool unsorted
) {
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        return graph.search_ef(q_span, k, ef, /*include_entry=*/true, filter, max_distance_computation_count, return_distances, /*pad_missing=*/true);
    }
    return graph.search(q_span, k, eps_or_ef, filter, max_distance_computation_count, return_distances, unsorted, /*pad_missing=*/true);
}

deglib::search::SearchResults graph_search_single_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InArray& query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (query.size() != in_dim) {
            throw std::invalid_argument(std::format("Query vector size ({}) does not match transformations input dim ({})", query.size(), in_dim));
        }
        const auto dt = query.dtype();
        if (is_dtype_float16_only(dt)) {
            return execute_graph_search_single(
                graph, std::span<const uint16_t>(static_cast<const uint16_t*>(query.data()), in_dim), eps_or_ef, k, filter, max_distance_computation_count,
                return_distances, unsorted
            );
        } else if (is_dtype_float32(dt)) {
            return execute_graph_search_single(
                graph, std::span<const float>(static_cast<const float*>(query.data()), in_dim), eps_or_ef, k, filter, max_distance_computation_count,
                return_distances, unsorted
            );
        } else {
            throw std::invalid_argument(std::format("Unsupported query dtype (code: {}, bits: {})", dt.code, dt.bits));
        }
    }
    const size_t req_bytes = graph.getFeatureSpace().get_data_size();
    const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
    const size_t query_bytes = query.size() * query.itemsize();
    if (query_bytes != query_req_bytes && query_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Query vector size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_bytes, query_req_bytes, req_bytes
            )
        );
    }
    const std::byte* query_ptr = static_cast<const std::byte*>(query.data());
    std::span<const std::byte> query_span(query_ptr, query_bytes);
    return execute_graph_search_single(graph, query_span, eps_or_ef, k, filter, max_distance_computation_count, return_distances, unsorted);
}

// Batch search wrapper for DynamicExplorationGraph
deglib::search::SearchResults dynamic_exploration_graph_search_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InArray& query,
    const float eps_or_ef,
    const uint32_t k,
    const deglib::search::Filter* filter,
    const uint32_t max_distance_computation_count,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (query.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected query to have two dimensions, got {}", query.ndim()));
    }

    const uint32_t n_queries = static_cast<uint32_t>(query.shape(0));
    const size_t q_cols = query.shape(1);
    const size_t q_itemsize = query.itemsize();
    const void* q_ptr = query.data();
    const auto q_dt = query.dtype();

    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (q_cols != in_dim) {
            throw std::invalid_argument(std::format("Query row dimension ({}) does not match transformations input dim ({})", q_cols, in_dim));
        }
    } else {
        const size_t req_bytes = graph.getFeatureSpace().get_data_size();
        const size_t query_req_bytes = graph.getFeatureSpace().get_query_data_size();
        const size_t query_stride_bytes = q_cols * q_itemsize;
        if (query_stride_bytes != query_req_bytes && query_stride_bytes != req_bytes) {
            throw std::invalid_argument(
                std::format(
                    "Query row size in bytes ({}) does not match query size ({}) or feature space data size ({})", query_stride_bytes, query_req_bytes,
                    req_bytes
                )
            );
        }
    }

    nb::gil_scoped_release release;
    auto run_batch = [&](auto queries_span) -> deglib::search::SearchResults {
        if (eps_or_ef >= 1.0f) {
            const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
            return graph.search_batch_ef(queries_span, n_queries, k, ef, true, filter, max_distance_computation_count, threads, return_distances, unsorted);
        }
        return graph.search_batch(queries_span, n_queries, k, eps_or_ef, filter, max_distance_computation_count, threads, return_distances, unsorted);
    };

    if (graph.hasTransformations()) {
        const size_t in_dim = graph.getInputDim();
        if (is_dtype_float16_only(q_dt)) {
            return run_batch(std::span<const uint16_t>(static_cast<const uint16_t*>(q_ptr), static_cast<size_t>(n_queries) * in_dim));
        }
        return run_batch(std::span<const float>(static_cast<const float*>(q_ptr), static_cast<size_t>(n_queries) * in_dim));
    }
    const size_t query_stride_bytes = q_cols * q_itemsize;
    return run_batch(std::span<const std::byte>(static_cast<const std::byte*>(q_ptr), static_cast<size_t>(n_queries) * query_stride_bytes));
}

deglib::search::SearchResults graph_explore_single_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const uint32_t entry_external_label,
    const uint32_t k,
    const uint32_t max_distance_computation_count = 0,
    const float eps_or_ef = 0.0f,
    const bool include_entry = true,
    const deglib::search::Filter* filter = nullptr,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        return graph.explore_ef(entry_external_label, k, ef, max_distance_computation_count, include_entry, filter, return_distances);
    }
    return graph.explore(entry_external_label, k, max_distance_computation_count, eps_or_ef, include_entry, filter, return_distances, unsorted);
}

template <typename G>
nb::object graph_has_path_wrapper(
    const G& graph,
    const std::vector<uint32_t>& from_external_labels,
    const uint32_t to_external_label,
    const float eps,
    const uint32_t k,
    const bool return_distances = true
) {
    const std::vector<deglib::graph::ObjectDistance> path = graph.hasPath(from_external_labels, to_external_label, eps, k);

    const size_t count = path.size();
    uint32_t* path_indices_ptr = nullptr;
    auto path_indices = make_numpy_1d<uint32_t>(count, path_indices_ptr);

    nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig> path_distances;
    float* path_distances_ptr = nullptr;
    if (return_distances) {
        path_distances = make_numpy_1d<float>(count, path_distances_ptr);
    }

    for (size_t i = 0; i < count; ++i) {
        path_indices_ptr[i] = path[i].getIdentifier();
        if (path_distances_ptr) {
            path_distances_ptr[i] = path[i].getDistance();
        }
    }

    if (return_distances) {
        return nb::make_tuple(path_indices, path_distances);
    }
    return path_indices.cast();
}

deglib::search::SearchResults dynamic_exploration_graph_explore_batch_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InUint32Array& entry_external_labels,
    const uint32_t k,
    const uint32_t max_distance_computation_count,
    const float eps_or_ef,
    const bool include_entry,
    const deglib::search::Filter* filter,
    const uint32_t threads,
    const bool return_distances = true,
    const bool unsorted = false
) {
    if (entry_external_labels.ndim() != 1) {
        throw std::invalid_argument(std::format("Expected entry_external_labels to have 1 dimension, got {}", entry_external_labels.ndim()));
    }

    const uint32_t* entry_ptr = entry_external_labels.data();
    const uint32_t n_queries = static_cast<uint32_t>(entry_external_labels.shape(0));
    std::span<const uint32_t> entries_span(entry_ptr, n_queries);

    nb::gil_scoped_release release;
    if (eps_or_ef >= 1.0f) {
        const uint32_t ef = static_cast<uint32_t>(std::lround(eps_or_ef));
        return graph.explore_batch_ef(entries_span, k, ef, max_distance_computation_count, include_entry, filter, threads, return_distances, unsorted);
    }
    return graph.explore_batch(entries_span, k, max_distance_computation_count, eps_or_ef, include_entry, filter, threads, return_distances, unsorted);
}


// ============================================================================
// GraphBuilder Wrapper (owns RNG alongside EvenRegularGraphBuilder)
// ============================================================================

class GraphBuilderPy {
  private:
    std::mt19937 rng_;
    deglib::builder::EvenRegularGraphBuilder builder_;

  public:
    GraphBuilderPy(
        deglib::DynamicExplorationGraph& graph,
        std::optional<uint32_t> seed,
        const deglib::builder::OptimizationTarget optimization_target,
        const uint8_t extend_k,
        const float extend_eps,
        const uint8_t improve_k,
        const float improve_eps,
        const uint8_t max_path_length,
        const uint32_t improve_tries
    )
        : rng_(seed.value_or(7)),
          builder_(
              [&]() -> deglib::DynamicExplorationGraph& {
                  if (!graph.isMutable()) {
                      throw std::runtime_error("Graph must be mutable to use GraphBuilder");
                  }
                  return graph;
              }(),
              rng_,
              optimization_target,
              extend_k,
              extend_eps,
              improve_k,
              improve_eps,
              max_path_length,
              improve_tries
          ) {}

    void add_entry(const InUint32Array& label, const InArray& feature) {
        if (label.ndim() != 1) {
            throw std::invalid_argument(std::format("Expected label to have 1 dimension, got {}", label.ndim()));
        }
        if (feature.ndim() != 2) {
            throw std::invalid_argument(std::format("Expected feature to have two dimensions, got {}", feature.ndim()));
        }
        const size_t num_entries = feature.shape(0);
        if (label.shape(0) != num_entries) {
            throw std::invalid_argument("Number of labels must match number of feature rows");
        }
        const uint32_t* label_ptr = label.data();
        const size_t dim = feature.shape(1);
        const size_t itemsize = feature.itemsize();
        const bool is_f32 = is_dtype_float32(feature.dtype());
        const void* raw_ptr = feature.data();

        nb::gil_scoped_release release;
        if (builder_.hasTransformations() && is_f32) {
            const float* f_ptr = static_cast<const float*>(raw_ptr);
            for (size_t i = 0; i < num_entries; ++i) {
                builder_.addEntry(label_ptr[i], std::span<const float>(f_ptr + i * dim, dim));
            }
        } else {
            const std::byte* feature_ptr = static_cast<const std::byte*>(raw_ptr);
            const size_t feature_len = itemsize * dim;
            for (size_t i = 0; i < num_entries; ++i) {
                std::vector<std::byte> feature_vec(feature_ptr + (feature_len * i), feature_ptr + (feature_len * (i + 1)));
                builder_.addEntry(label_ptr[i], std::move(feature_vec));
            }
        }
    }

    void remove_entry(const uint32_t label) { builder_.removeEntry(label); }
    size_t get_num_new_entries() { return builder_.getNumNewEntries(); }
    size_t get_num_remove_entries() { return builder_.getNumRemoveEntries(); }
    void set_threads(const uint32_t threads) { builder_.setThreadCount(threads); }
    void set_batch_size(const uint32_t tasks_per_batch, const uint32_t task_size) { builder_.setBatchSize(tasks_per_batch, task_size); }
    uint32_t get_batch_size() const { return builder_.getBatchSize(); }

    deglib::builder::BuilderStatus build(nb::object callback = nb::none(), const bool infinite = false) {
        nb::handle cb_handle = callback.is_none() ? nb::handle() : nb::handle(callback.ptr());
        std::function<void(deglib::builder::BuilderStatus&)> cb = nullptr;
        if (cb_handle.is_valid()) {
            cb = [cb_handle](deglib::builder::BuilderStatus& status) {
                nb::gil_scoped_acquire acquire;
                cb_handle(deglib::builder::BuilderStatus(status));
            };
        }
        nb::gil_scoped_release release;
        return builder_.build(cb, infinite);
    }

    deglib::builder::BuilderStatus build_silent(const bool infinite = false) {
        nb::gil_scoped_release release;
        return builder_.build([](deglib::builder::BuilderStatus&) {}, infinite);
    }

    void stop() { builder_.stop(); }
};


// ============================================================================
// Graph Loading / Creation / Optimization / Analysis Wrappers
// ============================================================================

deglib::DynamicExplorationGraph load_readonly_graph_wrapper(const char* path) { return deglib::DynamicExplorationGraph::load_readonly(path); }

deglib::DynamicExplorationGraph load_sizebounded_graph_wrapper(const char* path, const uint32_t capacity = 0) {
    return deglib::load_mutable_graph(path, capacity);
}

nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig> presort_wrapper(
    const InArray& vectors,
    const deglib::distances::FloatSpace& space,
    float radius_decay,
    size_t threads,
    nb::object callback,
    uint32_t seed
) {
    if (vectors.ndim() != 2) {
        throw std::invalid_argument("vectors must be a 2D float32/float16 array");
    }
    const auto vec_dt = vectors.dtype();
    if (!is_dtype_float32(vec_dt) && !is_dtype_float16_only(vec_dt)) {
        throw std::invalid_argument(std::format("presort expects a float32 or float16 array, got dtype code {} bits {}", vec_dt.code, vec_dt.bits));
    }
    const size_t count = vectors.shape(0);
    const size_t dim = vectors.shape(1);
    if (count > 0 && dim != space.dim()) {
        throw std::invalid_argument(std::format("Vector dimension ({}) does not match FloatSpace dimension ({})", dim, space.dim()));
    }

    PyObject* cb_ptr = callback.is_none() ? nullptr : callback.ptr();
    std::function<bool(float)> cb = nullptr;
    if (cb_ptr != nullptr) {
        cb = [cb_ptr](float progress) -> bool {
            nb::gil_scoped_acquire acquire;
            try {
                nb::object res = nb::handle(cb_ptr)(progress);
                if (!res.is_none() && nb::isinstance<nb::bool_>(res)) {
                    return nb::cast<bool>(res);
                }
            } catch (const std::exception&) {
                // Ignore callback exceptions during sorting loop
            }
            return false;
        };
    }

    const void* vec_ptr = vectors.data();
    std::vector<uint32_t> sorted_indices;
    {
        nb::gil_scoped_release release;
        sorted_indices = deglib::optimization::presort(vec_ptr, count, space, radius_decay, threads, cb, seed);
    }

    uint32_t* out_ptr = nullptr;
    auto output = make_numpy_1d<uint32_t>(count, out_ptr);
    if (count > 0) {
        std::memcpy(out_ptr, sorted_indices.data(), count * sizeof(uint32_t));
    }
    return output;
}

deglib::DynamicExplorationGraph load_dynamic_graph_wrapper(const char* path, const uint32_t chunk_size = 1024) {
    return deglib::load_dynamic_graph(path, chunk_size);
}

deglib::DynamicExplorationGraph create_size_bounded_graph(
    const uint32_t max_vertex_count,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    return deglib::create_empty(max_vertex_count, edges_per_vertex, feature_space, std::move(transformations));
}

deglib::DynamicExplorationGraph create_dynamic_graph(
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t chunk_size = 1024,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    return deglib::create_dynamic_empty(edges_per_vertex, feature_space, chunk_size, std::move(transformations));
}

deglib::DynamicExplorationGraph create_random_graph(
    const InArray& features,
    const uint8_t edges_per_vertex,
    const deglib::distances::FloatSpace& feature_space,
    const uint32_t seed,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {}
) {
    if (features.ndim() != 2) {
        throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", features.ndim()));
    }
    const uint32_t vertex_count = static_cast<uint32_t>(features.shape(0));
    const size_t req_bytes = feature_space.get_data_size();
    const size_t row_bytes = features.shape(1) * features.itemsize();
    if (row_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format(
                "Feature row size in bytes ({}) does not match required feature space "
                "data size ({})",
                row_bytes, req_bytes
            )
        );
    }

    const std::byte* feature_data = static_cast<const std::byte*>(features.data());
    return deglib::create_random_graph(feature_data, vertex_count, edges_per_vertex, feature_space, seed, std::move(transformations));
}


// ============================================================================
// Analysis Wrapper Functions
// ============================================================================

float calc_avg_edge_weight_wrapper(const deglib::DynamicExplorationGraph& graph, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_avg_edge_weight");
    }
    return deglib::analysis::calc_avg_edge_weight(static_cast<const deglib::graph::MutableGraph&>(graph.internal()), scale);
}

std::vector<float> calc_edge_weight_histogram_wrapper(const deglib::DynamicExplorationGraph& graph, const bool sorted, const int scale) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_edge_weight_histogram");
    }
    return deglib::analysis::calc_edge_weight_histogram(static_cast<const deglib::graph::MutableGraph&>(graph.internal()), sorted, scale);
}

bool check_graph_weights_wrapper(const deglib::DynamicExplorationGraph& graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for check_graph_weights");
    }
    return deglib::analysis::check_graph_weights(static_cast<const deglib::graph::MutableGraph&>(graph.internal()));
}

uint32_t calc_non_mrng_edges_wrapper(const deglib::DynamicExplorationGraph& graph) {
    if (!graph.isMutable()) {
        throw std::runtime_error("Graph must be mutable for calc_non_mrng_edges");
    }
    return deglib::analysis::calc_non_mrng_edges(static_cast<const deglib::graph::MutableGraph&>(graph.internal()));
}

bool check_graph_regularity_wrapper(const deglib::DynamicExplorationGraph& graph, const uint32_t expected_vertices, const bool check_back_link) {
    return deglib::analysis::check_graph_regularity(graph.internal(), expected_vertices, check_back_link);
}

bool check_graph_connectivity_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::check_graph_connectivity(graph.internal()); }

uint32_t calc_search_reachability_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::calc_search_reachability(graph.internal()); }

float calc_exploration_reach_wrapper(const deglib::DynamicExplorationGraph& graph) { return deglib::analysis::calc_exploration_reach(graph.internal()); }

deglib::analysis::GraphStats analyze_graph_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    std::optional<InStridedUint32Array> base_top = std::nullopt,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    const uint32_t* ptr = nullptr;
    size_t rows = 0;
    size_t k = 0;
    size_t row_stride = 0;

    if (base_top.has_value() && base_top->is_valid() && base_top->size() > 0) {
        if (base_top->ndim() != 2) {
            throw std::invalid_argument("base_top must be a 2D numpy array");
        }
        rows = base_top->shape(0);
        k = base_top->shape(1);
        const int64_t col_stride = base_top->stride(1);
        if (col_stride != 1) {
            throw std::invalid_argument("base_top columns must be contiguous");
        }
        row_stride = static_cast<size_t>(base_top->stride(0));
        ptr = base_top->data();
    }

    nb::gil_scoped_release release;
    return deglib::analysis::analyze_graph(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
}

float calc_graph_quality_wrapper(
    const deglib::DynamicExplorationGraph& graph,
    const InStridedUint32Array& base_top,
    const size_t sample_size = 0,
    const size_t num_threads = 0
) {
    if (base_top.ndim() != 2) {
        throw std::invalid_argument("base_top must be a 2D numpy array");
    }
    const size_t rows = base_top.shape(0);
    const size_t k = base_top.shape(1);
    const int64_t col_stride = base_top.stride(1);
    if (col_stride != 1) {
        throw std::invalid_argument("base_top columns must be contiguous");
    }
    const size_t row_stride = static_cast<size_t>(base_top.stride(0));
    const uint32_t* ptr = base_top.data();

    nb::gil_scoped_release release;
    return deglib::analysis::calc_graph_quality(graph.internal(), ptr, rows, k, sample_size, num_threads, row_stride);
}


namespace deglib_py {
void bind_graph_and_builder(nb::module_& m) {
    // graphs
    nb::class_<deglib::DynamicExplorationGraph>(m, "DynamicExplorationGraph")
        .def_static(
            "create_empty",
            [](const uint32_t max_vertex_count, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space) {
                return deglib::create_empty(max_vertex_count, edges_per_vertex, feature_space);
            },
            nb::arg("max_vertex_count"), nb::arg("edges_per_vertex"), nb::arg("feature_space")
        )
        .def_static(
            "create_dynamic_empty",
            [](const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t chunk_size) {
                return deglib::create_dynamic_empty(edges_per_vertex, feature_space, chunk_size);
            },
            nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("chunk_size") = 1024
        )
        .def_static(
            "create_random_graph",
            [](const InArray& features, const uint8_t edges_per_vertex, const deglib::distances::FloatSpace& feature_space, const uint32_t seed) {
                if (features.ndim() != 2) {
                    throw std::invalid_argument(std::format("Expected features to have two dimensions, got {}", features.ndim()));
                }
                const size_t req_bytes = feature_space.get_data_size();
                const size_t row_bytes = features.shape(1) * features.itemsize();
                if (row_bytes != req_bytes) {
                    throw std::invalid_argument(
                        std::format(
                            "Feature row size in bytes ({}) does not match required "
                            "feature space data size ({})",
                            row_bytes, req_bytes
                        )
                    );
                }
                return deglib::create_random_graph(
                    static_cast<const std::byte*>(features.data()), static_cast<uint32_t>(features.shape(0)), edges_per_vertex, feature_space, seed
                );
            },
            nb::arg("features"), nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("seed") = 7
        )
        .def("size", &deglib::DynamicExplorationGraph::size)
        .def("get_edges_per_vertex", &deglib::DynamicExplorationGraph::getEdgesPerVertex)
        .def("get_po", &deglib::DynamicExplorationGraph::getPo)
        .def("get_pl", &deglib::DynamicExplorationGraph::getPl)
        .def("get_nl", &deglib::DynamicExplorationGraph::getNl)
        .def(
            "get_feature_space", [](const deglib::DynamicExplorationGraph& g) -> const deglib::distances::FloatSpace& { return g.getFeatureSpace(); },
            nb::rv_policy::reference_internal
        )
        .def("has_vertex", &deglib::DynamicExplorationGraph::hasVertex)
        .def(
            "get_labels",
            [](const deglib::DynamicExplorationGraph& g) {
                auto labels = g.getExternalLabels();
                uint32_t* ptr = nullptr;
                auto arr = make_numpy_1d<uint32_t>(labels.size(), ptr);
                if (!labels.empty()) {
                    std::memcpy(ptr, labels.data(), labels.size() * sizeof(uint32_t));
                }
                return arr;
            },
            "Returns a 1D numpy array of all external labels ordered by their internal graph index (0 .. size()-1)."
        )
        .def(
            "get_neighbors",
            [](const deglib::DynamicExplorationGraph& g, uint32_t external_label) { return g.getNeighbors(external_label); },
            nb::arg("external_label")
        )
        .def(
            "get_neighbors",
            [](const deglib::DynamicExplorationGraph& g, const InUint32Array& labels, size_t threads) {
                if (labels.ndim() != 1) {
                    throw std::invalid_argument(std::format("Expected labels to have 1 dimension, got {}", labels.ndim()));
                }
                const uint32_t n_labels = static_cast<uint32_t>(labels.shape(0));
                const uint8_t edges = g.getEdgesPerVertex();
                uint32_t* out_ptr = nullptr;
                auto result = make_numpy_2d<uint32_t>(n_labels, static_cast<size_t>(edges), out_ptr);
                std::span<const uint32_t> labels_span(labels.data(), n_labels);
                std::span<uint32_t> out_span(out_ptr, static_cast<size_t>(n_labels) * edges);
                {
                    nb::gil_scoped_release release;
                    g.getNeighbors(labels_span, out_span, threads);
                }
                return result;
            },
            nb::arg("external_labels"), nb::arg("threads") = 1
        )
        .def("is_mutable", &deglib::DynamicExplorationGraph::isMutable)
        .def(
            "to_readonly",
            [](const deglib::DynamicExplorationGraph& g) {
                nb::gil_scoped_release release;
                return g.to_readonly();
            }
        )
        .def(
            "to_mutable",
            [](const deglib::DynamicExplorationGraph& g, uint32_t new_max_size, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.to_mutable(new_max_size, num_threads);
            },
            nb::arg("max_size") = 0, nb::arg("threads") = 0
        )
        .def(
            "to_dynamic",
            [](const deglib::DynamicExplorationGraph& g, uint32_t chunk_size, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.to_dynamic(chunk_size, num_threads);
            },
            nb::arg("chunk_size") = 1024, nb::arg("threads") = 0
        )
        .def(
            "with_metric",
            [](const deglib::DynamicExplorationGraph& g, const deglib::distances::FloatSpace& target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.with_metric(target_space, num_threads);
            },
            nb::arg("target_space"), nb::arg("threads") = 0
        )
        .def(
            "with_transformations",
            [](const deglib::DynamicExplorationGraph& g, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
               std::optional<deglib::distances::FloatSpace> target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                return g.with_transformations(std::move(transformations), target_space, num_threads);
            },
            nb::arg("transformations"), nb::arg("target_space") = nb::none(), nb::arg("threads") = 0
        )
        .def(
            "with_features",
            [](const deglib::DynamicExplorationGraph& g, const deglib::distances::FloatSpace& target_space, InArray features,
               std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations, size_t num_threads) {
                const void* feat_ptr = extract_custom_features_ptr(g, target_space, features);
                nb::gil_scoped_release release;
                return g.with_features(target_space, feat_ptr, std::move(transformations), num_threads);
            },
            nb::arg("target_space"), nb::arg("features"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{},
            nb::arg("threads") = 0
        )
        .def(
            "_transform_readonly",
            [](const deglib::DynamicExplorationGraph& g, std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations,
               std::optional<deglib::distances::FloatSpace> target_space, size_t num_threads) {
                nb::gil_scoped_release release;
                auto combined = g.getTransformations();
                combined.insert(combined.end(), transformations.begin(), transformations.end());
                auto ro = std::make_unique<deglib::graph::ReadOnlyGraph>(
                    deglib::graph::ReadOnlyGraph::from_graph(g.internal(), target_space, transformations, num_threads)
                );
                return deglib::DynamicExplorationGraph(std::move(ro), std::move(combined));
            },
            nb::arg("transformations"), nb::arg("target_space") = nb::none(), nb::arg("threads") = 0
        )
        .def("_set_transformations", &deglib::DynamicExplorationGraph::setTransformations, nb::arg("transformations"))
        .def(
            "search", &graph_search_single_wrapper, nb::arg("query"), nb::arg("eps_or_ef"), nb::arg("k"), nb::arg("filter") = nullptr,
            nb::arg("max_distance_computation_count") = 0, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "search_batch", &dynamic_exploration_graph_search_batch_wrapper, nb::arg("query"), nb::arg("eps_or_ef"), nb::arg("k"), nb::arg("filter") = nullptr,
            nb::arg("max_distance_computation_count") = 0, nb::arg("threads") = 1, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "explore", &graph_explore_single_wrapper, nb::arg("entry_external_label"), nb::arg("k"), nb::arg("max_distance_computation_count") = 0,
            nb::arg("eps_or_ef") = 0.0f, nb::arg("include_entry") = true, nb::arg("filter") = nullptr, nb::arg("return_distances") = true,
            nb::arg("unsorted") = false
        )
        .def(
            "explore_batch", &dynamic_exploration_graph_explore_batch_wrapper, nb::arg("entry_external_labels"), nb::arg("k"),
            nb::arg("max_distance_computation_count") = 0, nb::arg("eps_or_ef") = 0.0f, nb::arg("include_entry") = true, nb::arg("filter") = nullptr,
            nb::arg("threads") = 1, nb::arg("return_distances") = true, nb::arg("unsorted") = false
        )
        .def(
            "has_path", &graph_has_path_wrapper<deglib::DynamicExplorationGraph>, nb::arg("from_external_labels"), nb::arg("to_external_label"),
            nb::arg("eps") = 0.0f, nb::arg("k") = 1, nb::arg("return_distances") = true
        )
        .def(
            "save_graph", [](deglib::DynamicExplorationGraph& g, const char* path) { g.saveGraph(path); }, nb::arg("path")
        )
        .def("has_transformations", &deglib::DynamicExplorationGraph::hasTransformations)
        .def("get_transformations", [](const deglib::DynamicExplorationGraph& g) { return g.getTransformations(); })
        .def("get_entry_vertex_indices", &deglib::DynamicExplorationGraph::getEntryVertexIndices)
        .def("set_entry_vertex_indices", &deglib::DynamicExplorationGraph::setEntryVertexIndices, nb::arg("indices"))
        .def(
            "optimize", &deglib::DynamicExplorationGraph::optimize, nb::arg("sample_count") = 300, nb::arg("median_subset") = 4096,
            "Auto-tune the graph's traversal prefetch parameters (po, pl, nl) by timing traversal over sampled vertices.\n\n"
            "Args:\n"
            "    sample_count: number of vertices sampled as queries (capped at the graph size).\n"
            "    median_subset: subset size for computing the graph-median entry point (<= 1 keeps native entries)."
        )
        .def(
            "to_recall_table",
            [](const deglib::DynamicExplorationGraph& g, uint32_t k, const std::string& mode, size_t sample_count, std::pair<float, float> recall_range,
               float recall_step, std::optional<std::vector<float>> target_recalls, size_t threads) {
                const auto parsed_mode = parse_recall_mode_py(mode);
                nb::gil_scoped_release release;
                if (target_recalls.has_value()) {
                    return g.to_recall_table(k, parsed_mode, std::span<const float>(*target_recalls), sample_count, threads);
                }
                return g.to_recall_table(k, parsed_mode, sample_count, recall_range, recall_step, threads);
            },
            nb::arg("k"), nb::arg("mode"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate a single-mode RecallTable ('ef' or 'eps') mapping target recall@k values to search parameters."
        )
        .def(
            "to_recall_tables",
            [](const deglib::DynamicExplorationGraph& g, uint32_t k, size_t sample_count, std::pair<float, float> recall_range, float recall_step,
               std::optional<std::vector<float>> target_recalls, size_t threads) {
                nb::gil_scoped_release release;
                if (target_recalls.has_value()) {
                    return g.to_recall_tables(k, std::span<const float>(*target_recalls), sample_count, threads);
                }
                return g.to_recall_tables(k, sample_count, recall_range, recall_step, threads);
            },
            nb::arg("k"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate both 'ef' and 'eps' RecallTables in one pass sharing exact ground truth."
        )
        .def(
            "to_pareto_line",
            [](const deglib::DynamicExplorationGraph& g, const std::vector<deglib::search::RecallTable>& recall_tables, size_t sample_count, size_t rounds) {
                nb::gil_scoped_release release;
                return g.to_pareto_line(std::span<const deglib::search::RecallTable>(recall_tables), sample_count, rounds);
            },
            nb::arg("recall_tables"), nb::arg("sample_count") = 0, nb::arg("rounds") = deglib::search::ParetoLine::kDefaultRounds,
            "Build a ParetoLine unifying measured operating points across one or more RecallTables by (measured_recall, latency_us) dominance."
        );

    m.def("load_readonly_graph", &load_readonly_graph_wrapper);
    m.def("load_dynamic_graph", &load_dynamic_graph_wrapper, nb::arg("path"), nb::arg("chunk_size") = 1024);
    m.def("load_sizebounded_graph", &load_sizebounded_graph_wrapper, nb::arg("path"), nb::arg("capacity") = 0);
    m.def(
        "presort", &presort_wrapper, nb::arg("vectors"), nb::arg("space"), nb::arg("radius_decay") = 0.9f, nb::arg("threads") = 0,
        nb::arg("callback") = nb::none(), nb::arg("seed") = 42
    );

    m.def(
        "create_size_bounded_graph", &create_size_bounded_graph, nb::arg("max_vertex_count"), nb::arg("edges_per_vertex"), nb::arg("feature_space"),
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_dynamic_graph", &create_dynamic_graph, nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("chunk_size") = 1024,
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );
    m.def(
        "create_random_graph", &create_random_graph, nb::arg("features"), nb::arg("edges_per_vertex"), nb::arg("feature_space"), nb::arg("seed") = 7,
        nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}
    );

    // GraphBuilder (renamed from EvenRegularGraphBuilder)
    nb::enum_<deglib::builder::OptimizationTarget>(m, "OptimizationTarget", nb::is_arithmetic())
        .value("StreamingData", deglib::builder::OptimizationTarget::StreamingData)
        .value("HighLID", deglib::builder::OptimizationTarget::HighLID)
        .value("LowLID", deglib::builder::OptimizationTarget::LowLID);

    nb::class_<GraphBuilderPy>(m, "GraphBuilder")
        .def(
            nb::init<
                deglib::DynamicExplorationGraph&, std::optional<uint32_t>, deglib::builder::OptimizationTarget, uint8_t, float, uint8_t, float, uint8_t,
                uint32_t>(),
            nb::arg("graph"), nb::arg("seed") = nb::none(), nb::arg("optimization_target") = deglib::builder::OptimizationTarget::LowLID,
            nb::arg("extend_k") = 0, nb::arg("extend_eps") = 0.1f, nb::arg("improve_k") = 0, nb::arg("improve_eps") = 0.001f, nb::arg("max_path_length") = 5,
            nb::arg("improve_tries") = 0, nb::keep_alive<1, 2>()
        )
        .def("add_entry", &GraphBuilderPy::add_entry, nb::arg("label"), nb::arg("feature"))
        .def("remove_entry", &GraphBuilderPy::remove_entry)
        .def("get_num_new_entries", &GraphBuilderPy::get_num_new_entries)
        .def("get_num_remove_entries", &GraphBuilderPy::get_num_remove_entries)
        .def("set_threads", &GraphBuilderPy::set_threads, nb::arg("threads"))
        .def("set_batch_size", &GraphBuilderPy::set_batch_size)
        .def("get_batch_size", &GraphBuilderPy::get_batch_size)
        .def("build", &GraphBuilderPy::build, nb::arg("callback") = nb::none(), nb::arg("infinite") = false)
        .def("build_silent", &GraphBuilderPy::build_silent, nb::arg("infinite") = false)
        .def("stop", &GraphBuilderPy::stop);

    m.def("calc_avg_edge_weight", &calc_avg_edge_weight_wrapper, nb::arg("graph"), nb::arg("scale") = 1);
    m.def("calc_edge_weight_histogram", &calc_edge_weight_histogram_wrapper, nb::arg("graph"), nb::arg("sort"), nb::arg("scale") = 1);
    m.def("check_graph_weights", &check_graph_weights_wrapper, nb::arg("graph"));
    m.def("check_graph_regularity", &check_graph_regularity_wrapper, nb::arg("graph"), nb::arg("expected_vertices"), nb::arg("check_back_link") = false);
    m.def("check_graph_connectivity", &check_graph_connectivity_wrapper, nb::arg("graph"));
    m.def("calc_non_mrng_edges", &calc_non_mrng_edges_wrapper, nb::arg("graph"));
    m.def("calc_search_reachability", &calc_search_reachability_wrapper, nb::arg("graph"));
    m.def("calc_exploration_reach", &calc_exploration_reach_wrapper, nb::arg("graph"));

    nb::class_<deglib::analysis::GraphStats>(m, "GraphStats", nb::dynamic_attr())
        .def_ro("vertex_count", &deglib::analysis::GraphStats::vertex_count)
        .def_ro("edge_count", &deglib::analysis::GraphStats::edge_count)
        .def_ro("feature_dims", &deglib::analysis::GraphStats::feature_dims)
        .def_ro("edges_per_vertex", &deglib::analysis::GraphStats::edges_per_vertex)
        .def_ro("avg_out_degree", &deglib::analysis::GraphStats::avg_out_degree)
        .def_ro("min_out_degree", &deglib::analysis::GraphStats::min_out_degree)
        .def_ro("max_out_degree", &deglib::analysis::GraphStats::max_out_degree)
        .def_ro("avg_in_degree", &deglib::analysis::GraphStats::avg_in_degree)
        .def_ro("min_in_degree", &deglib::analysis::GraphStats::min_in_degree)
        .def_ro("max_in_degree", &deglib::analysis::GraphStats::max_in_degree)
        .def_ro("source_vertices", &deglib::analysis::GraphStats::source_vertices)
        .def_ro("search_reachability", &deglib::analysis::GraphStats::search_reachability)
        .def_ro("exploration_reachability", &deglib::analysis::GraphStats::exploration_reachability)
        .def_ro("graph_quality", &deglib::analysis::GraphStats::graph_quality)
        .def_ro("memory_bytes", &deglib::analysis::GraphStats::memory_bytes);

    m.def("analyze_graph", &analyze_graph_wrapper, nb::arg("graph"), nb::arg("base_top") = nb::none(), nb::arg("sample_size") = 0, nb::arg("threads") = 0);
    m.def(
        "calc_graph_quality", &calc_graph_quality_wrapper, nb::arg("graph"), nb::arg("base_top"), nb::arg("sample_size") = 0, nb::arg("threads") = 0,
        "Computes graph quality comparing each vertex's outgoing edges against its true nearest neighbors."
    );
    m.def(
        "prune_non_mrng_edges",
        [](deglib::DynamicExplorationGraph& graph, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune non-mrng edges");
            }
            return deglib::optimization::prune_non_mrng_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), threads);
        },
        nb::arg("graph"), nb::arg("threads") = 0
    );
    m.def(
        "prune_worst_edges",
        [](deglib::DynamicExplorationGraph& graph, const uint8_t prune_worst, const size_t threads) {
            if (!graph.isMutable()) {
                throw std::runtime_error("Graph must be mutable to prune worst edges");
            }
            deglib::optimization::pruning::prune_worst_edges(static_cast<deglib::graph::MutableGraph&>(graph.internal()), prune_worst, threads);
        },
        nb::arg("graph"), nb::arg("prune_worst"), nb::arg("threads") = 0
    );

    nb::class_<deglib::builder::BuilderStatus>(m, "BuilderStatus")
        .def_rw("step", &deglib::builder::BuilderStatus::step)
        .def_rw("added", &deglib::builder::BuilderStatus::added)
        .def_rw("deleted", &deglib::builder::BuilderStatus::deleted)
        .def_rw("improved", &deglib::builder::BuilderStatus::improved)
        .def_rw("tries", &deglib::builder::BuilderStatus::tries)
        .def_ro("step_added_ids", &deglib::builder::BuilderStatus::step_added_ids)
        .def_ro("step_deleted_ids", &deglib::builder::BuilderStatus::step_deleted_ids)
        .def_ro("total_added_ids", &deglib::builder::BuilderStatus::total_added_ids)
        .def_ro("total_deleted_ids", &deglib::builder::BuilderStatus::total_deleted_ids);


}
}  // namespace deglib_py
