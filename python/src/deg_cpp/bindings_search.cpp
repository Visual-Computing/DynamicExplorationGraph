#include "bindings_common.h"

// ============================================================================
// Reranker Binding Helpers (thin marshaling around C++ owning Reranker)
// ============================================================================

static std::shared_ptr<deglib::search::Reranker> reranker_from_float32(
    const deglib::distances::FloatSpace& space,
    const InFloatArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const uint32_t expected_input_dim =
        (!transformations.empty() && transformations.front()) ? transformations.front()->input_dim() : static_cast<uint32_t>(space.dim());
    if (data.shape(1) != expected_input_dim) {
        throw std::invalid_argument(
            std::format("base_vectors column count ({}) does not match expected input dimension ({})", data.shape(1), expected_input_dim)
        );
    }
    const size_t n_rows = data.shape(0);
    const uint32_t dim = static_cast<uint32_t>(data.shape(1));
    std::span<const float> data_span(data.data(), n_rows * dim);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(deglib::search::Reranker::from_data(space, data_span, std::move(transformations), threads, idx_span));
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_fp16(
    const deglib::distances::FloatSpace& space,
    const InArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    if (!is_dtype_fp16_or_u16(data.dtype())) {
        throw std::invalid_argument("reranker_from_fp16: expected float16 or uint16 array");
    }
    const uint32_t expected_input_dim =
        (!transformations.empty() && transformations.front()) ? transformations.front()->input_dim() : static_cast<uint32_t>(space.dim());
    if (data.shape(1) != expected_input_dim) {
        throw std::invalid_argument(
            std::format("base_vectors column count ({}) does not match expected input dimension ({})", data.shape(1), expected_input_dim)
        );
    }
    const size_t n_rows = data.shape(0);
    const uint32_t dim = static_cast<uint32_t>(data.shape(1));
    std::span<const uint16_t> data_span(static_cast<const uint16_t*>(data.data()), n_rows * dim);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(deglib::search::Reranker::from_data(space, data_span, std::move(transformations), threads, idx_span));
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_raw(
    const deglib::distances::FloatSpace& space,
    const InArray& data,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0,
    std::optional<InUint32Array> row_indices = std::nullopt
) {
    if (data.ndim() != 2) {
        throw std::invalid_argument("base_vectors must be a 2D array");
    }
    const size_t req_bytes = space.get_data_size();
    const size_t stride_bytes = data.shape(1) * data.itemsize();
    if (stride_bytes != req_bytes) {
        throw std::invalid_argument(std::format("Base vector row size ({}) does not match feature space data size ({})", stride_bytes, req_bytes));
    }
    const size_t n_rows = data.shape(0);
    std::span<const std::byte> data_span(static_cast<const std::byte*>(data.data()), n_rows * stride_bytes);

    std::span<const uint32_t> idx_span;
    if (row_indices.has_value() && row_indices->is_valid()) {
        if (row_indices->ndim() != 1) {
            throw std::invalid_argument("row_indices must be a 1D array");
        }
        idx_span = std::span<const uint32_t>(row_indices->data(), row_indices->shape(0));
    }

    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(
        deglib::search::Reranker::from_raw(space, data_span, n_rows, std::move(transformations), threads, idx_span)
    );
}

static std::shared_ptr<deglib::search::Reranker> reranker_from_graph(
    const deglib::DynamicExplorationGraph& graph,
    const deglib::distances::FloatSpace& space,
    std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>> transformations = {},
    size_t threads = 0
) {
    nb::gil_scoped_release release;
    return std::make_shared<deglib::search::Reranker>(graph.to_reranker(space, std::move(transformations), threads));
}

static nb::object reranker_rerank_py(
    const deglib::search::Reranker& reranker,
    const InArray& queries,
    const InUint32Array& candidate_indices,
    size_t k_top = 0,
    size_t num_threads = 0,
    bool return_distances = false,
    bool unsorted = false
) {
    if (queries.ndim() != 2) {
        throw std::invalid_argument("queries must be a 2D array");
    }
    const size_t n_queries = queries.shape(0);

    if (candidate_indices.ndim() != 2) {
        throw std::invalid_argument("candidate_indices must be a 2D array");
    }
    if (candidate_indices.shape(0) != n_queries) {
        throw std::invalid_argument("Number of candidate rows must match number of queries");
    }

    const size_t evp_k = candidate_indices.shape(1);
    if (k_top == 0 || k_top > evp_k) {
        k_top = evp_k;
    }

    const bool is_f32_queries = is_dtype_float32(queries.dtype());
    const size_t q_dim = queries.shape(1);
    const size_t query_row_bytes = q_dim * queries.itemsize();
    const size_t req_bytes = reranker.getSpace().get_data_size();
    const bool use_f32_path =
        is_f32_queries && (!reranker.getTransformations().empty() || reranker.getSpace().metric().get_data_type() == deglib::distances::MetricDataType::FP32);

    if (!use_f32_path && query_row_bytes != req_bytes) {
        throw std::invalid_argument(
            std::format("Query row size in bytes ({}) does not match required feature space data size ({})", query_row_bytes, req_bytes)
        );
    }

    uint32_t* ind_ptr = nullptr;
    auto result_indices = make_numpy_2d<uint32_t>(n_queries, k_top, ind_ptr);

    nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig> result_distances;
    float* dist_ptr = nullptr;
    if (return_distances) {
        result_distances = make_numpy_2d<float>(n_queries, k_top, dist_ptr);
    }

    const uint32_t* cand_ptr = candidate_indices.data();
    const void* q_raw_ptr = queries.data();

    if (use_f32_path) {
        const float* q_ptr = static_cast<const float*>(q_raw_ptr);
        nb::gil_scoped_release release;
        deglib::concurrent::parallel_batch_for(
            size_t{0}, n_queries, num_threads,
            [&](size_t begin, size_t end, size_t) {
                for (size_t q = begin; q < end; ++q) {
                    reranker.rerank_f32(
                        std::span<const float>(q_ptr + q * q_dim, q_dim), std::span<const uint32_t>(cand_ptr + q * evp_k, evp_k), static_cast<uint32_t>(k_top),
                        std::span<uint32_t>(ind_ptr + q * k_top, k_top), return_distances ? std::span<float>(dist_ptr + q * k_top, k_top) : std::span<float>{},
                        return_distances, unsorted
                    );
                }
            },
            64
        );
    } else {
        nb::gil_scoped_release release;
        reranker.rerank(
            std::span<const std::byte>(static_cast<const std::byte*>(q_raw_ptr), n_queries * req_bytes), n_queries,
            std::span<const uint32_t>(cand_ptr, n_queries * evp_k), evp_k, static_cast<uint32_t>(k_top), std::span<uint32_t>(ind_ptr, n_queries * k_top),
            return_distances ? std::span<float>(dist_ptr, n_queries * k_top) : std::span<float>{}, return_distances, unsorted, num_threads
        );
    }

    if (return_distances) {
        return nb::make_tuple(result_indices, result_distances);
    }
    return result_indices.cast();
}


// ============================================================================
// Fast Searcher Wrapper
// ============================================================================

class SearcherPy {
  private:
    std::unique_ptr<deglib::search::Searcher> searcher_;

  public:
    SearcherPy(
        const deglib::DynamicExplorationGraph& graph,
        std::shared_ptr<deglib::search::Reranker> shared_reranker = nullptr,
        bool normalize_query = false
    )
        : searcher_(std::make_unique<deglib::search::Searcher>(graph.to_searcher(std::move(shared_reranker), normalize_query))) {}

    bool get_normalize_query() const noexcept { return searcher_->normalize_query(); }
    void set_normalize_query(bool val) noexcept { searcher_->set_normalize_query(val); }

    void optimize(uint32_t n_clusters = 256, uint32_t n_iter = 20, size_t sample_size = 0, size_t threads = 0, bool use_routing = true) {
        nb::gil_scoped_release release;
        searcher_->optimize(n_clusters, n_iter, sample_size, threads, use_routing);
    }

    deglib::search::SearchResults search(
        const InArray& query,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) {
        const size_t ndim = query.ndim();
        const void* query_ptr = query.data();
        const auto dt = query.dtype();
        const bool is_f16 = is_dtype_fp16_or_u16(dt);
        if (!is_f16 && !is_dtype_float32(dt)) {
            throw std::invalid_argument(std::format("Unsupported query dtype (code: {}, bits: {})", dt.code, dt.bits));
        }

        // 1D query or 2D single-row/single-column query -> execute fast single-query search
        if (ndim == 1 || (ndim == 2 && (query.shape(0) == 1 || query.shape(1) == 1))) {
            const size_t q_len = query.size();
            if (is_f16) {
                return searcher_->search<uint16_t>(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(query_ptr), q_len), k, eps_or_ef, rerank_factor, return_distances, unsorted,
                    max_distance_computation_count
                );
            }
            return searcher_->search<float>(
                std::span<const float>(static_cast<const float*>(query_ptr), q_len), k, eps_or_ef, rerank_factor, return_distances, unsorted,
                max_distance_computation_count
            );
        }

        // 2D batch query
        if (ndim == 2) {
            const size_t n_queries = query.shape(0);
            const size_t q_cols = query.shape(1);
            nb::gil_scoped_release release;
            if (is_f16) {
                return searcher_->search_batch<uint16_t>(
                    std::span<const uint16_t>(static_cast<const uint16_t*>(query_ptr), n_queries * q_cols), n_queries, k, eps_or_ef, rerank_factor, num_threads,
                    return_distances, unsorted, max_distance_computation_count
                );
            }
            return searcher_->search_batch<float>(
                std::span<const float>(static_cast<const float*>(query_ptr), n_queries * q_cols), n_queries, k, eps_or_ef, rerank_factor, num_threads,
                return_distances, unsorted, max_distance_computation_count
            );
        }

        throw std::invalid_argument("search query must be 1D vector or 2D batch array");
    }

    deglib::search::SearchResults search_batch(
        const InArray& queries,
        uint32_t k,
        float eps_or_ef = 0.1f,
        float rerank_factor = 1.0f,
        size_t num_threads = 1,
        bool return_distances = false,
        bool unsorted = false,
        uint32_t max_distance_computation_count = 0
    ) {
        return search(queries, k, eps_or_ef, rerank_factor, num_threads, return_distances, unsorted, max_distance_computation_count);
    }

    deglib::search::RecallTable to_recall_table(
        uint32_t k,
        const std::string& mode,
        float rerank_factor = 1.0f,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
        float recall_step = deglib::search::RecallTable::kDefaultRecallStep,
        std::optional<std::vector<float>> target_recalls = std::nullopt,
        size_t threads = 0
    ) const {
        const auto parsed_mode = parse_recall_mode_py(mode);
        nb::gil_scoped_release release;
        if (target_recalls.has_value()) {
            return searcher_->to_recall_table(k, parsed_mode, std::span<const float>(*target_recalls), rerank_factor, sample_count, threads);
        }
        return searcher_->to_recall_table(k, parsed_mode, rerank_factor, sample_count, recall_range, recall_step, threads);
    }

    std::vector<deglib::search::RecallTable> to_recall_tables(
        uint32_t k,
        size_t sample_count = deglib::search::RecallTable::kDefaultSampleCount,
        std::pair<float, float> recall_range = {deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
        float recall_step = deglib::search::RecallTable::kDefaultRecallStep,
        std::optional<std::vector<float>> target_recalls = std::nullopt,
        std::optional<std::vector<float>> rerank_factors = std::nullopt,
        size_t threads = 0
    ) const {
        if (rerank_factors.has_value() && rerank_factors->empty()) {
            throw std::invalid_argument("rerank_factors must not be empty");
        }
        const std::span<const float> rf_span = rerank_factors.has_value() ? std::span<const float>(*rerank_factors) : std::span<const float>{};
        nb::gil_scoped_release release;
        if (target_recalls.has_value()) {
            return searcher_->to_recall_tables(k, std::span<const float>(*target_recalls), sample_count, rf_span, threads);
        }
        return searcher_->to_recall_tables(k, sample_count, recall_range, recall_step, rf_span, threads);
    }

    deglib::search::ParetoLine to_pareto_line(
        const std::vector<deglib::search::RecallTable>& recall_tables,
        size_t sample_count = 0,
        size_t rounds = deglib::search::ParetoLine::kDefaultRounds
    ) const {
        nb::gil_scoped_release release;
        return searcher_->to_pareto_line(std::span<const deglib::search::RecallTable>(recall_tables), sample_count, rounds);
    }
};


namespace deglib_py {
void bind_search(nb::module_& m) {
    nb::module_ search_module = m.def_submodule("search", "Search utilities including Filter and SearchResults");
    nb::class_<deglib::search::Filter>(search_module, "Filter");

    static uint32_t s_empty_u32 = 0;
    static float s_empty_f32 = 0.0f;
    nb::class_<deglib::search::SearchResults>(
        search_module, "SearchResults", "Unified search results holding zero-copy NumPy arrays (`indices`, optional `distances`) and traversal telemetry."
    )
        .def_prop_ro(
            "indices",
            [](nb::handle self_h) -> nb::object {
                auto& self = nb::cast<deglib::search::SearchResults&>(self_h);
                uint32_t* ptr = self.indices.empty() ? &s_empty_u32 : self.indices.data();
                if (self.is_batch) {
                    size_t shape[2] = {self.n_queries, static_cast<size_t>(self.k)};
                    return nb::ndarray<nb::numpy, uint32_t, nb::ndim<2>, nb::c_contig>(ptr, 2, shape, self_h).cast();
                }
                size_t shape[1] = {self.indices.size()};
                return nb::ndarray<nb::numpy, uint32_t, nb::ndim<1>, nb::c_contig>(ptr, 1, shape, self_h).cast();
            },
            "Zero-copy NumPy uint32 ndarray of neighbor external labels (1D for single query, 2D for batch)."
        )
        .def_prop_ro(
            "distances",
            [](nb::handle self_h) -> nb::object {
                auto& self = nb::cast<deglib::search::SearchResults&>(self_h);
                if (!self.has_distances()) return nb::none();
                float* ptr = self.distances.empty() ? &s_empty_f32 : self.distances.data();
                if (self.is_batch) {
                    size_t shape[2] = {self.n_queries, static_cast<size_t>(self.k)};
                    return nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::c_contig>(ptr, 2, shape, self_h).cast();
                }
                size_t shape[1] = {self.distances.size()};
                return nb::ndarray<nb::numpy, float, nb::ndim<1>, nb::c_contig>(ptr, 1, shape, self_h).cast();
            },
            "Zero-copy NumPy float32 ndarray of neighbor distances, or None when return_distances=False."
        )
        .def_ro(
            "graph_distance_count", &deglib::search::SearchResults::graph_distance_count,
            "Total number of distance computations performed during graph traversal."
        )
        .def_ro(
            "rerank_distance_count", &deglib::search::SearchResults::rerank_distance_count,
            "Total number of exact distance computations performed during candidate reranking."
        )
        .def_ro(
            "graph_hop_count", &deglib::search::SearchResults::graph_hop_count, "Total number of graph vertex expansions (hops) performed during traversal."
        )
        .def("__len__", &deglib::search::SearchResults::size);

    m.attr("SearchResults") = search_module.attr("SearchResults");

    nb::class_<deglib::search::RecallTable>(search_module, "RecallTable", "Single-mode calibration table mapping target recall@k values to ef or eps.")
        .def_prop_ro("k", &deglib::search::RecallTable::k)
        .def_prop_ro("sample_count", &deglib::search::RecallTable::sample_count)
        .def_prop_ro("is_ef", [](const deglib::search::RecallTable& t) { return t.mode() == deglib::search::RecallTable::Mode::Ef; })
        .def_prop_ro("is_eps", [](const deglib::search::RecallTable& t) { return t.mode() == deglib::search::RecallTable::Mode::Eps; })
        .def_prop_ro("rerank_factor", &deglib::search::RecallTable::rerank_factor)
        .def_prop_ro("recall_bounds", &deglib::search::RecallTable::recall_bounds)
        .def_prop_ro(
            "entries",
            [](const deglib::search::RecallTable& t) {
                std::vector<std::tuple<float, float, float>> out;
                const auto& entries = t.entries();
                out.reserve(entries.size());
                for (const auto& e : entries) {
                    out.emplace_back(e.target_recall, e.eps_or_ef, e.measured_recall);
                }
                return out;
            }
        )
        .def("eps_or_ef_values", &deglib::search::RecallTable::eps_or_ef_values, "Returns deduplicated ascending calibrated eps_or_ef values.")
        .def(
            "param_for", &deglib::search::RecallTable::param_for, nb::arg("target_recall"), nb::arg("clamp") = false,
            "Looks up or linearly interpolates an eps_or_ef value for target_recall (returns None if out of bounds unless clamp=True)."
        )
        .def("__len__", &deglib::search::RecallTable::size)
        .def("__str__", &deglib::search::RecallTable::to_string)
        .def("__repr__", &deglib::search::RecallTable::to_string);

    nb::class_<deglib::search::ParetoLine>(search_module, "ParetoLine", "Unified recall-latency Pareto frontier combining one or more RecallTables.")
        .def_prop_ro("k", &deglib::search::ParetoLine::k)
        .def_prop_ro("sample_count", &deglib::search::ParetoLine::sample_count)
        .def_prop_ro("recall_bounds", &deglib::search::ParetoLine::recall_bounds)
        .def_prop_ro("recall_tables", [](const deglib::search::ParetoLine& line) { return line.recall_tables(); })
        .def_prop_ro(
            "entries",
            [](const deglib::search::ParetoLine& line) {
                std::vector<std::tuple<float, float, float, float, double, double>> out;
                const auto& entries = line.entries();
                out.reserve(entries.size());
                for (const auto& e : entries) {
                    out.emplace_back(e.target_recall, e.eps_or_ef, e.rerank_factor, e.measured_recall, e.latency_us, e.qps);
                }
                return out;
            }
        )
        .def("eps_or_ef_values", &deglib::search::ParetoLine::eps_or_ef_values, "Returns deduplicated eps_or_ef parameters along the Pareto frontier.")
        .def("params", &deglib::search::ParetoLine::params, "Returns deduplicated (eps_or_ef, rerank_factor) parameter pairs along the Pareto frontier.")
        .def(
            "param_for", &deglib::search::ParetoLine::param_for, nb::arg("target_recall"), nb::arg("clamp") = false,
            "Returns the (eps_or_ef, rerank_factor) pair of the frontier entry whose measured_recall is nearest to target_recall (returns None if out of "
            "bounds "
            "unless clamp=True)."
        )
        .def("__len__", &deglib::search::ParetoLine::size)
        .def("__str__", &deglib::search::ParetoLine::to_string)
        .def("__repr__", &deglib::search::ParetoLine::to_string);

    search_module.def("create_filter", [](const InIntArray& valid_labels, size_t max_value, size_t max_label_count) {
        if (valid_labels.ndim() != 1) {
            throw std::invalid_argument(std::format("Expected feature to have only one dimension, got {}", valid_labels.ndim()));
        }
        const int* ptr = valid_labels.data();
        const size_t size = valid_labels.shape(0);
        return std::make_unique<deglib::search::Filter>(ptr, size, max_value, max_label_count);
    });

    nb::class_<deglib::search::Reranker>(search_module, "Reranker", "Owning C++ candidate reranker.")
        .def_static(
            "from_float32", &reranker_from_float32, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_fp16", &reranker_from_fp16, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_raw", &reranker_from_raw, nb::arg("space"), nb::arg("data"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0,
            nb::arg("row_indices") = nb::none()
        )
        .def_static(
            "from_graph", &reranker_from_graph, nb::arg("graph"), nb::arg("space"),
            nb::arg("transformations") = std::vector<std::shared_ptr<deglib::transformation::FeatureTransformation>>{}, nb::arg("threads") = 0
        )
        .def("get_num_base_vectors", &deglib::search::Reranker::getNumBaseVectors)
        .def("get_dim", [](const deglib::search::Reranker& r) { return r.getSpace().dim(); })
        .def("get_instruction", [](const deglib::search::Reranker& r) { return r.getSpace().get_instruction(); })
        .def("get_po", &deglib::search::Reranker::getPo, "Prefetch lookahead offset (0 disables rerank prefetching).")
        .def("get_pl", &deglib::search::Reranker::getPl, "Cache lines prefetched per feature vector (0 prefetches the whole vector).")
        .def(
            "optimize",
            [](deglib::search::Reranker& r, size_t sample_count, size_t num_candidates) {
                nb::gil_scoped_release release;
                r.optimize(sample_count, num_candidates);
            },
            nb::arg("sample_count") = 300, nb::arg("num_candidates") = 150,
            "Auto-tune the reranker's prefetch parameters (po, pl) by timing reranking over sampled base vectors."
        )
        .def(
            "rerank", &reranker_rerank_py, nb::arg("queries"), nb::arg("candidate_indices"), nb::arg("k_top") = 0, nb::arg("threads") = 0,
            nb::arg("return_distances") = false, nb::arg("unsorted") = false
        );

    nb::class_<SearcherPy>(search_module, "Searcher", "Search pipeline. Results are the graph's labels.")
        .def(
            nb::init<const deglib::DynamicExplorationGraph&, std::shared_ptr<deglib::search::Reranker>, bool>(), nb::arg("graph"),
            nb::arg("reranker") = nullptr, nb::arg("normalize_query") = false, nb::keep_alive<1, 2>(),
            "Construct a Searcher with optional shared owning Reranker."
        )
        .def_prop_rw("normalize_query", &SearcherPy::get_normalize_query, &SearcherPy::set_normalize_query)
        .def(
            "search", &SearcherPy::search, nb::arg("query"), nb::arg("k") = 10, nb::arg("eps_or_ef") = 0.1f, nb::arg("rerank_factor") = 1.0f,
            nb::arg("threads") = 1, nb::arg("return_distances") = false, nb::arg("unsorted") = false, nb::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors of a single query vector (1D) or batch of query vectors (2D).\n\n"
            "Args:\n"
            "    query: 1D query vector or 2D batch array.\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for batch search (default: 1).\n"
            "    return_distances: If True, populates `.distances` on the returned SearchResults.\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited).\n\n"
            "Returns:\n"
            "    SearchResults: Object with zero-copy `.indices`, optional `.distances`, and telemetry counts."
        )
        .def(
            "search_batch", &SearcherPy::search_batch, nb::arg("queries"), nb::arg("k") = 10, nb::arg("eps_or_ef") = 0.1f, nb::arg("rerank_factor") = 1.0f,
            nb::arg("threads") = 1, nb::arg("return_distances") = false, nb::arg("unsorted") = false, nb::arg("max_distance_computation_count") = 0,
            "Search for nearest neighbors for a batch of query vectors.\n\n"
            "Args:\n"
            "    queries: 2D query array (shape N x Dim).\n"
            "    k: Number of nearest neighbors to return per query.\n"
            "    eps_or_ef: Exploration parameter. Values >= 1.0 are treated as fixed pool size (ef),\n"
            "               values < 1.0 are treated as relative distance margin (eps).\n"
            "    rerank_factor: Expansion factor for exact candidate reranking (default: 1.0).\n"
            "    threads: Number of worker threads for parallel search.\n"
            "    return_distances: If True, populates `.distances` on the returned SearchResults.\n"
            "    unsorted: If True, returns candidates in heap order instead of ascending distance.\n"
            "    max_distance_computation_count: Optional hard budget of distance computations per query (0 = unlimited).\n\n"
            "Returns:\n"
            "    SearchResults: Object with zero-copy `.indices`, optional `.distances`, and telemetry counts."
        )
        .def(
            "optimize", &SearcherPy::optimize, nb::arg("n_clusters") = 256, nb::arg("n_iter") = 20, nb::arg("sample_size") = 0, nb::arg("threads") = 0,
            nb::arg("use_routing") = true,
            "Select entry vertices via k-means medoids and tune prefetch parameters.\n\nArgs:\n    n_clusters: number of entry vertices to select (maximum "
            "512).\n    n_iter: number of "
            "k-means "
            "iterations.\n    sample_size: vertices sampled for clustering, 0 selects 3% of the graph size.\n    threads: worker threads, 0 selects "
            "a library default.\n    use_routing: build a k-means routing graph for entry selection (used by the QG/query-graph searcher)."
        )
        .def(
            "to_recall_table", &SearcherPy::to_recall_table, nb::arg("k"), nb::arg("mode"), nb::arg("rerank_factor") = 1.0f,
            nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(), nb::arg("threads") = 0,
            "Calibrate a single-mode RecallTable ('ef' or 'eps') for this Searcher pipeline."
        )
        .def(
            "to_recall_tables", &SearcherPy::to_recall_tables, nb::arg("k"), nb::arg("sample_count") = deglib::search::RecallTable::kDefaultSampleCount,
            nb::arg("recall_range") = std::pair<float, float>{deglib::search::RecallTable::kDefaultMinRecall, deglib::search::RecallTable::kDefaultMaxRecall},
            nb::arg("recall_step") = deglib::search::RecallTable::kDefaultRecallStep, nb::arg("target_recalls") = nb::none(),
            nb::arg("rerank_factors") = nb::none(), nb::arg("threads") = 0,
            "Calibrate ('ef', 'eps') RecallTables across rerank_factors in a shared single-pass evaluation."
        )
        .def(
            "to_pareto_line", &SearcherPy::to_pareto_line, nb::arg("recall_tables"), nb::arg("sample_count") = 0,
            nb::arg("rounds") = deglib::search::ParetoLine::kDefaultRounds,
            "Build an end-to-end ParetoLine unifying measured operating points across RecallTables by (measured_recall, latency_us) dominance."
        );


}
}  // namespace deglib_py
