import os
import pytest
import pathlib
import tempfile
import numpy as np

import deglib


DEFAULT_DIMS = 128
DEFAULT_SAMPLES = 100


def get_tmp_graph_file(samples: int, dims: int) -> pathlib.Path:
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_test')
    os.makedirs(tmpdir, exist_ok=True)

    return pathlib.Path(os.path.join(tmpdir, 'test_graph_S{}_D{}.deg'.format(samples, dims)))


def create_size_bounded_graph(samples: int, dims: int):
    data = np.random.random((samples, dims)).astype(np.float32)

    rnd = deglib.Mt19937()
    feature_space = deglib.FloatSpace(dims, deglib.Metric.L2)
    graph = deglib.graph.SizeBoundedGraph(samples, dims, feature_space)

    builder = deglib.builder.EvenRegularGraphBuilder(graph, rnd, extend_k=30, extend_eps=0.2, improve_k=30)

    for i, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(i, vec)

    def improvement_callback(_status):
        pass

    builder.build(improvement_callback, False)

    return graph


def get_graph_file(samples: int, dims: int):
    g_file = get_tmp_graph_file(samples, dims)

    if g_file.is_file():
        return g_file

    graph = create_size_bounded_graph(samples, dims)
    graph.save_graph(g_file)

    return g_file


def get_read_only_graph(samples: int, dims: int) -> deglib.graph.ReadOnlyGraph:
    """
    Build a graph file, that can be used for tests.
    """
    g_file = get_graph_file(samples, dims)
    return deglib.graph.load_readonly_graph(g_file)


@pytest.fixture()
def size_bounded_graph():
    return create_size_bounded_graph(DEFAULT_DIMS, DEFAULT_DIMS)


@pytest.fixture()
def graph_file():
    """
    Build a graph file, that can be used for tests.
    """
    return get_graph_file(DEFAULT_SAMPLES, DEFAULT_DIMS)


@pytest.fixture()
def read_only_graph():
    return get_read_only_graph(DEFAULT_SAMPLES, DEFAULT_DIMS)


def test_get_feature_vector(read_only_graph):
    for i in range(read_only_graph.size()):
        fv = read_only_graph.get_feature_vector(i)
        assert fv.shape == (DEFAULT_DIMS,)
        assert fv.dtype == np.float32
