from concurrent.futures import ThreadPoolExecutor, wait

import deglib_cpp
import numpy as np
import deglib
from deglib.builder import EvenRegularGraphBuilder
from deglib.search import Filter


def main():
    samples = 1000
    dims = 8

    data = np.random.random((samples, dims)).astype(np.float32)
    # data = data / np.linalg.norm(data, axis=1)[:, None]  # L2 normalize

    graph = deglib.graph.SizeBoundedGraph.create_empty(data.shape[0], data.shape[1], 16, deglib.Metric.L2)
    builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=32, extend_eps=0.01, improve_k=0)

    for i, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(i, vec)

    builder.build(callback='progress')

    valid_labels = np.random.choice(graph.size(), size=5, replace=False)

    query = np.random.random(dims).astype(np.float32)

    results, dists = graph.search(query, filter_labels=valid_labels, eps=0.0, k=8)

    print(results)

    print('indices:', results.shape, results.dtype)
    print('valid:', valid_labels.shape)
    print('all results in labels:', np.all(np.isin(results, valid_labels)))


def dump_data(seed):
    np.random.seed(seed)

    samples = 100
    dims = 128
    data = np.random.random((samples, dims)).astype(np.float32)

    dim_row = np.zeros((samples, 1), dtype=np.int32) + dims

    print(data.shape, dim_row.shape)
    data_to_dump = np.concatenate((dim_row, data.view(np.int32)), axis=1)

    print(data_to_dump.shape, data_to_dump.dtype)

    data_to_dump.tofile('crash_data.fvecs')
    print('data dumped to crash_data.fvecs')


def do_build_with_remove(seed, edges_per_vertex):
    np.random.seed(seed)

    samples = 100
    dims = 128
    # edges_per_vertex = 2  # samples // 10
    data = np.random.random((samples, dims)).astype(np.float32)

    # data = deglib.repository.fvecs_read('crash_data.fvecs')
    graph = deglib.graph.SizeBoundedGraph.create_empty(
        data.shape[0], data.shape[1], edges_per_vertex, deglib.Metric.L2
    )
    builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

    for label, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(label, vec)

    # remove half of the vertices
    for label in range(0, data.shape[0], 2):
        builder.remove_entry(label)

    builder.build(callback='progress')


KNOWN_CRASHES = {
    (1, 10)
}


def do_all():
    for i in range(100):
        for epv in range(2, 34, 2):
            if (i, epv) in KNOWN_CRASHES:
                print('skipping seed:', i, ' epv:', epv)
            else:
                print('seed:', i, ' epv:', epv)
                do_build_with_remove(i, epv)


def build_graph(jobname, data, dim):
    print('starting', jobname)
    graph = deglib.graph.SizeBoundedGraph.create_empty(1_000_000, dim, edges_per_vertex=8)
    print(graph)

    builder = EvenRegularGraphBuilder(graph, improve_k=0, extend_eps=0, extend_k=8)
    print(builder)

    builder.add_entry(range(data.shape[0]), data)

    builder.build()


class FinishPrinter:
    def __init__(self, jobname: str):
        self.jobname = jobname

    def __call__(self, fut):
        print('finish', self.jobname)


def test_free_memory():
    dim = 512
    data = np.random.random((100_000, dim)).astype(np.float32)

    jobs = 2

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = []
        for i in range(10):
            jobname = 'job {}'.format(i)
            print('start: {}'.format(jobname))
            future = executor.submit(build_graph, jobname, data, dim)

            future.add_done_callback(FinishPrinter(jobname))
            futures.append(future)
        wait(futures)


if __name__ == '__main__':
    main()
    # do_build_with_remove(1, 10)
    # do_all()
    # dump_data(1)
    # test_free_memory()
