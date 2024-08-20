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

    @pytest.mark.parametrize('lid', list(deglib.builder.LID))
    def test_build_simple(self, lid):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30, lid=lid)
        for i, vec in enumerate(self.data):
            vec: np.ndarray
            builder.add_entry(i, vec)

        builder.build()

    @pytest.mark.parametrize('lid', list(deglib.builder.LID))
    def test_build_batch(self, lid):
        graph = deglib.graph.SizeBoundedGraph.create_empty(
            self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
        )
        builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30, lid=lid)
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

#     def test_callback(self):
#         graph = deglib.graph.SizeBoundedGraph.create_empty(
#             self.data.shape[0], self.data.shape[1], self.edges_per_vertex, deglib.Metric.L2
#         )
#         builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)
#         for i, vec in enumerate(self.data):
#             vec: np.ndarray
#             builder.add_entry(i, vec)
#
#         tester = CallbackTester()
#         builder.build(callback=tester)
#         assert tester.num_callbacks == self.data.shape[0], 'Got {} callbacks, but expected {}'.format(
#             tester.num_callbacks, self.data.shape[0]
#         )
#         assert tester.last_status.step == self.data.shape[0], 'Got {} steps, but expected {}'.format(
#             tester.last_status.step, self.data.shape[0]
#         )
