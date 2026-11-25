import numpy as np
import pytest

import deglib
from deglib.distances import Metric


class CallbackTester:
    def __init__(self):
        self.num_callbacks = 0
        self.last_status = None

    def __call__(self, status: deglib.builder.BuilderStatus):
        assert isinstance(status, deglib.builder.BuilderStatus), (
            'Got instance of type "{}" for builder_status in callback'.format(type(status))
        )
        self.last_status = status
        self.num_callbacks += 1


class TestGraphs:
    def setup_method(self):
        self.samples = 100
        self.dims = 128
        self.edges_per_vertex = self.samples // 10

        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)

    @pytest.mark.parametrize("batch", [True, False])
    def test_add_entry(self, batch):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        if batch:
            builder.add_entry(range(self.data.shape[0]), self.data)
        else:
            for i, vec in enumerate(self.data):
                vec: np.ndarray
                builder.add_entry(i, vec)

    @pytest.mark.parametrize("optimization_target", list(deglib.builder.OptimizationTarget))
    def test_build_simple(self, optimization_target):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(
            graph, extend_k=30, extend_eps=0.2, improve_k=30, optimization_target=optimization_target
        )
        for i, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(i, vec)

        builder.build()

    @pytest.mark.parametrize("optimization_target", list(deglib.builder.OptimizationTarget))
    def test_build_batch(self, optimization_target):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(
            graph, extend_k=30, extend_eps=0.2, improve_k=30, optimization_target=optimization_target
        )
        builder.add_entry(range(self.data.shape[0]), self.data)

        builder.build()

    def test_build_with_remove(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        for label, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(label, vec)

        # remove half of the vertices
        for label in range(0, self.data.shape[0], 2):
            builder.remove_entry(label)

        builder.build()

    def test_remove_entry_batch(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)
        builder.build()

        labels_to_remove = np.arange(0, self.data.shape[0], 5, dtype=np.uint32)
        builder.remove_entry(labels_to_remove)
        assert builder.get_num_remove_entries() == len(labels_to_remove)
        status = builder.build()
        assert status.deleted == len(labels_to_remove)
        assert graph.size() == self.data.shape[0] - len(labels_to_remove)

    def test_get_num_entries(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        def _check_entries(expected: int, action: str):
            assert action in ("new", "remove")

            actual = builder.get_num_new_entries() if action == "new" else builder.get_num_remove_entries()
            assert actual == expected, "Added {} {} entries, but get_num_{}_entries() returned {}".format(
                expected, action, action, builder.get_num_new_entries()
            )

        _check_entries(0, "new")
        _check_entries(0, "remove")

        for label, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(label, vec)

        _check_entries(self.data.shape[0], "new")
        _check_entries(0, "remove")

        # remove half of the vertices
        for label in range(0, self.data.shape[0], 2):
            builder.remove_entry(label)

        _check_entries(self.data.shape[0] // 2, "remove")

    def test_callback(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        tester = CallbackTester()
        builder.build(callback=tester)
        assert tester.num_callbacks > 0, "Expected at least 1 callback execution"
        assert tester.last_status is not None
        assert tester.last_status.step > 0

    def test_concurrency_settings(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        # Test thread count
        builder.set_threads(2)

        # Test batch size setting and getting
        builder.set_batch_size(tasks_per_batch=16, task_size=5)
        expected_batch_size = 2 * 16 * 5
        assert builder.get_batch_size() == expected_batch_size

        builder.add_entry(range(self.data.shape[0]), self.data)
        builder.build()

    def test_stop(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        stopped_in_callback = False

        def _stopping_callback(status: deglib.builder.BuilderStatus):
            nonlocal stopped_in_callback
            stopped_in_callback = True
            builder.stop()

        builder.build(callback=_stopping_callback)
        assert stopped_in_callback

    def test_build_returns_status(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        status = builder.build()
        assert isinstance(status, deglib.builder.BuilderStatus), "Expected BuilderStatus, got {}".format(type(status))
        assert status.added == self.data.shape[0]
        assert status.deleted == 0
        assert len(status.total_added_ids) == self.data.shape[0]
        assert len(status.total_deleted_ids) == 0

    def test_show_progress(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        status = builder.build(show_progress=True)
        assert isinstance(status, deglib.builder.BuilderStatus)
        assert status.added == self.data.shape[0]

    def test_builder_status_facade_exposed(self):
        assert not hasattr(deglib, "BuilderStatus")
        assert hasattr(deglib.builder, "BuilderStatus")

    def test_build_status_callback_step_ids(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        callback_statuses = []

        def _collect_callback(status: deglib.builder.BuilderStatus):
            callback_statuses.append(status)

        result = builder.build(callback=_collect_callback)

        assert len(callback_statuses) > 0
        # Verify that total_added_ids grows monotonically across callbacks
        prev_total = 0
        for cb_status in callback_statuses:
            assert len(cb_status.total_added_ids) >= prev_total
            assert len(cb_status.step_added_ids) > 0 or cb_status.added == 0
            prev_total = len(cb_status.total_added_ids)

        # The last callback's total should match the return value
        last_cb = callback_statuses[-1]
        assert last_cb.total_added_ids == result.total_added_ids
        assert last_cb.total_deleted_ids == result.total_deleted_ids

    def test_build_mixed_add_delete(self):
        graph = deglib.graph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, metric=Metric.FP32_L2
        )
        builder = deglib.builder.GraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        for label, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(label, vec)

        # remove some vertices
        for label in range(0, self.data.shape[0], 3):
            builder.remove_entry(label)

        status = builder.build()
        assert isinstance(status, deglib.builder.BuilderStatus)
        assert status.added == self.data.shape[0]
        assert status.deleted > 0
        assert len(status.total_added_ids) == self.data.shape[0]
        assert len(status.total_deleted_ids) > 0

    @pytest.mark.parametrize("dim", [48, 80, 100, 784])
    @pytest.mark.parametrize("metric", [Metric.FP32_L2, Metric.FP32_InnerProduct])
    def test_build_unaligned_dimensions(self, dim, metric):
        samples = 60
        edges_per_vertex = 16
        np.random.seed(42)
        data = np.random.random((samples, dim)).astype(np.float32)
        if metric == Metric.FP32_InnerProduct:
            data = data / np.linalg.norm(data, axis=1, keepdims=True)

        graph = deglib.graph.create_empty(samples, dim, edges_per_vertex, metric=metric)
        builder = deglib.builder.GraphBuilder(
            graph,
            extend_k=32,
            extend_eps=0.1,
            optimization_target=deglib.builder.OptimizationTarget.LowLID
            if metric == Metric.FP32_L2
            else deglib.builder.OptimizationTarget.HighLID,
        )
        builder.add_entry(range(samples), data)
        status = builder.build()
        assert status.added == samples
        assert graph.size() == samples

        # Verify search works and returns valid neighbors
        res = graph.search(data[0], k=10, eps_or_ef=0.1)
        indices, dists = res.indices, res.distances
        assert len(indices) == 10
        assert not np.isnan(dists).any()

    def test_sliding_window_continuous_updates(self):
        window_size = 50
        batch_size = 10
        rounds = 5
        dims = 16
        edges_per_vertex = 8

        np.random.seed(42)
        total_vectors = window_size + batch_size * rounds
        all_data = np.random.random((total_vectors, dims)).astype(np.float32)

        graph = deglib.graph.create_empty(window_size + batch_size * 2, dims, edges_per_vertex, metric=Metric.FP32_L2)
        builder = deglib.builder.GraphBuilder(
            graph,
            extend_k=16,
            extend_eps=0.1,
            improve_k=8,
            optimization_target=deglib.builder.OptimizationTarget.LowLID,
        )

        # Initial build
        builder.add_entry(range(window_size), all_data[:window_size])
        builder.build()
        assert graph.size() == window_size

        active_start = 0
        next_insert = window_size

        for r in range(rounds):
            # Remove oldest
            for lbl in range(active_start, active_start + batch_size):
                builder.remove_entry(lbl)

            # Add newest
            new_labels = list(range(next_insert, next_insert + batch_size))
            new_data = all_data[next_insert : next_insert + batch_size]
            builder.add_entry(new_labels, new_data)

            builder.build()

            active_start += batch_size
            next_insert += batch_size

            assert graph.size() == window_size

            # Test search on latest added vector
            query = all_data[next_insert - 1]
            res = graph.search(query, k=1, eps_or_ef=0.1)
            indices, dists = res.indices, res.distances
            assert len(indices) == 1
            assert indices[0] == next_insert - 1
            assert np.isclose(dists[0], 0.0, atol=1e-5)


class TestBuildFunction:
    def test_build_mutable_and_to_readonly(self):
        rng = np.random.default_rng(7)
        data = rng.random((80, 16), dtype=np.float32)

        mutable = deglib.build(data, edges_per_vertex=16, extend_k=16, threads=2)
        assert isinstance(mutable, deglib.DynamicExplorationGraph)
        assert mutable.is_mutable()
        assert mutable.size() == 80
        assert mutable.get_dim() == 16
        assert mutable.get_metric() == Metric.FP32_L2

        ro = mutable.to_readonly()
        assert isinstance(ro, deglib.DynamicExplorationGraph)
        assert not ro.is_mutable()
        assert ro.size() == 80
        assert ro.get_dim() == 16
        assert ro.get_metric() == Metric.FP32_L2

        empty = deglib.build(np.empty((0, 16), dtype=np.float32), edges_per_vertex=16)
        assert empty.size() == 0
        assert empty.is_mutable()

    def test_build_presort_bool_and_indices(self):
        rng = np.random.default_rng(11)
        data = rng.random((60, 16), dtype=np.float32)

        g1 = deglib.build(data, edges_per_vertex=12, extend_k=16, presort=True, optimize=False)
        assert g1.size() == 60
        for i in range(60):
            assert g1.has_vertex(i)

        sorted_indices = deglib.optimization.presort(data, metric=Metric.FP32_L2, threads=1)
        g2 = deglib.build(data, edges_per_vertex=12, extend_k=16, presort=sorted_indices, optimize=False)
        assert g2.size() == 60
        indices = g2.search(data[5], k=1, eps_or_ef=0.1).indices
        assert int(indices[0]) == 5

    def test_build_quantized_metrics(self):
        rng = np.random.default_rng(23)
        raw = rng.normal(size=(80, 32)).astype(np.float32)
        unit = raw / np.linalg.norm(raw, axis=1, keepdims=True)

        g_evp = deglib.build(
            unit,
            metric=Metric.EVP_InnerProduct,
            edges_per_vertex=16,
            extend_k=16,
            optimize=True,
            threads=1,
        )
        assert g_evp.size() == 80
        assert g_evp.get_metric() == Metric.EVP_InnerProduct
        idx = g_evp.search(unit[0], k=5, eps_or_ef=0.2).indices
        assert len(idx) == 5

        g_int8 = deglib.build(
            raw,
            metric="Int8_L2",
            edges_per_vertex=16,
            extend_k=16,
            optimize=True,
            threads=1,
        ).to_readonly()
        assert isinstance(g_int8, deglib.DynamicExplorationGraph)
        assert not g_int8.is_mutable()
        assert g_int8.size() == 80
        assert g_int8.get_metric() == Metric.Int8_L2
        idx2 = g_int8.search(raw[3], k=5, eps_or_ef=0.2).indices
        assert len(idx2) == 5
