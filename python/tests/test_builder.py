import numpy as np
import pytest

import deglib
import deglib_cpp


class CallbackTester:
    def __init__(self):
        self.num_callbacks = 0
        self.last_status = None

    def __call__(self, status: deglib_cpp.BuilderStatus):
        assert isinstance(status, deglib_cpp.BuilderStatus), \
            'Got instance of type \"{}\" for builder_status in callback'.format(type(status))
        self.last_status = status
        self.num_callbacks += 1


class TestGraphs:
    def setup_method(self):
        self.samples = 100
        self.dims = 128
        self.edges_per_vertex = self.samples // 10

        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)

    @pytest.mark.parametrize('batch', [True, False])
    def test_add_entry(self, batch):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        if batch:
            builder.add_entry(range(self.data.shape[0]), self.data)
        else:
            for i, vec in enumerate(self.data):
                vec: np.ndarray
                builder.add_entry(i, vec)

    @pytest.mark.parametrize('optimization_target', list(deglib.builder.OptimizationTarget))
    def test_build_simple(self, optimization_target):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30, optimization_target=optimization_target)
        for i, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(i, vec)

        builder.build()

    @pytest.mark.parametrize('optimization_target', list(deglib.builder.OptimizationTarget))
    def test_build_batch(self, optimization_target):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30, optimization_target=optimization_target)
        builder.add_entry(range(self.data.shape[0]), self.data)

        builder.build()

    def test_build_with_remove(self):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        for label, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(label, vec)

        # remove half of the vertices
        for label in range(0, self.data.shape[0], 2):
            builder.remove_entry(label)

        builder.build()

    def test_get_num_entries(self):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        def _check_entries(expected: int, action: str):
            assert action in ('new', 'remove')

            actual = builder.get_num_new_entries() if action == 'new' else builder.get_num_remove_entries()
            assert actual == expected, \
                'Added {} {} entries, but get_num_{}_entries() returned {}'.format(
                    expected, action, action, builder.get_num_new_entries()
                )

        _check_entries(0, 'new')
        _check_entries(0, 'remove')

        for label, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(label, vec)

        _check_entries(self.data.shape[0], 'new')
        _check_entries(0, 'remove')

        # remove half of the vertices
        for label in range(0, self.data.shape[0], 2):
            builder.remove_entry(label)

        _check_entries(self.data.shape[0] // 2, 'remove')

    def test_callback(self):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        tester = CallbackTester()
        builder.build(callback=tester)
        assert tester.num_callbacks > 0, 'Expected at least 1 callback execution'
        assert tester.last_status is not None
        assert tester.last_status.step > 0

    def test_concurrency_settings(self):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

        # Test thread count
        builder.set_thread_count(2)

        # Test batch size setting and getting
        builder.set_batch_size(tasks_per_batch=16, task_size=5)
        expected_batch_size = 2 * 16 * 5
        assert builder.get_batch_size() == expected_batch_size

        builder.add_entry(range(self.data.shape[0]), self.data)
        builder.build()

    def test_stop(self):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
        builder.add_entry(range(self.data.shape[0]), self.data)

        stopped_in_callback = False

        def _stopping_callback(status: deglib_cpp.BuilderStatus):
            nonlocal stopped_in_callback
            stopped_in_callback = True
            builder.stop()

        builder.build(callback=_stopping_callback)
        assert stopped_in_callback

