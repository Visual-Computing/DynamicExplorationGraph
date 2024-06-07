import numpy as np
import tqdm

import deglib
import deglib_cpp


# TODO: replace ReadOnlyGraph with python wrapper
def test_graph_anns(
        graph: deglib_cpp.ReadOnlyGraph, query_repository: deglib_cpp.StaticFeatureRepository, ground_truth: np.ndarray,
        ground_truth_dims: int, repeat_test: int, k: int):
    entry_vertex_id = 0

    get_near_avg_entry_index(graph)


def get_near_avg_entry_index(graph: deglib.graph.ReadOnlyGraph):
    feature_dims = graph.get_feature_space().dim()
    graph_size = graph.size()
    avg_fv = np.zeros(feature_dims, dtype=np.float32)
    for i in tqdm.trange(graph_size, desc='calculating average vector'):
        fv = graph.get_feature_vector(i)
        avg_fv += fv

    avg_fv /= graph_size
    seed = [graph.get_internal_index(0)]
    result_queue = graph.search(seed, avg_fv, 0.1, 30)
    entry_vertex_id = result_queue.top().get_internal_index()

    print('entry vertex id:', entry_vertex_id)
