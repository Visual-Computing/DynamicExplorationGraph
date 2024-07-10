import numpy as np
import deglib


def main():
    samples = 1000
    dims = 128

    data = np.random.random((samples, dims)).astype(np.float32)

    graph = deglib.graph.SizeBoundedGraph.create_empty(data.shape[0], data.shape[1], 32, deglib.Metric.L2)
    builder = deglib.builder.EvenRegularGraphBuilder(graph, extend_k=30, extend_eps=0.2, improve_k=30)

    for i, vec in enumerate(data):
        vec: np.ndarray
        builder.add_entry(i, vec)

    builder.build(callback='progress')


if __name__ == '__main__':
    main()
