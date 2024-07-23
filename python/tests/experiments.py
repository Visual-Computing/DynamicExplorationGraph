import numpy as np
import deglib


def main():
    samples = 1000
    dims = 128

    data = np.random.random((samples, dims)).astype(np.float32)

    graph = deglib.graph.SizeBoundedGraph.create_empty(data.shape[0], data.shape[1], 16, deglib.Metric.InnerProduct)
    builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=32, extend_eps=0.01, improve_k=0)

    for i, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(i, vec)

    builder.build(callback='progress')

    query = np.random.random((5, dims)).astype(np.float32)
    # query = np.random.random((dims,)).astype(np.float32)
    results, dists = graph.search(query, eps=0.0, k=3)
    print(results, results.dtype, results.shape)
    print(dists, dists.dtype, dists.shape)


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


if __name__ == '__main__':
    main()
    # do_build_with_remove(1, 10)
    # do_all()
    # dump_data(1)
