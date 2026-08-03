import pytest
import numpy as np

import deglib
from deglib.search import Filter


class TestFilterUnit:
    def test_empty_filter(self):
        empty_labels = np.array([], dtype=np.int32)
        f = Filter(empty_labels, max_value=10, max_label_count=10)
        assert f.max_value == 10
        assert f.max_label_count == 10
        assert len(f.valid_labels) == 0

    def test_single_label(self):
        labels = np.array([5], dtype=np.int32)
        f = Filter(labels)
        assert f.max_value == 5
        assert np.array_equal(f.valid_labels, labels)

    def test_multiple_labels(self):
        labels = np.array([0, 3, 7, 10], dtype=np.int32)
        f = Filter(labels)
        assert f.max_value == 10
        assert len(f.valid_labels) == 4

    def test_duplicates_and_out_of_range(self):
        labels = np.array([5, 5, 5, 5], dtype=np.int32)
        f = Filter(labels, max_value=10)
        assert f.max_value == 10
        assert len(f.valid_labels) == 4

    def test_large_max_value(self):
        labels = np.array([0, 63, 64, 127, 128], dtype=np.int32)
        f = Filter(labels)
        assert f.max_value == 128


class TestFilterIntegration:
    def setup_method(self):
        self.samples = 100
        self.dims = 16
        self.edges_per_vertex = 10
        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)
        self.graph = deglib.builder.build_from_data(
            self.data, edges_per_vertex=self.edges_per_vertex, metric=deglib.Metric.L2
        )
        self.query = np.random.random((self.dims,)).astype(np.float32)

    def test_filter_matching_results(self):
        valid_labels = np.array([0, 10, 20, 30, 40], dtype=np.int32)
        k = 3
        results, _ = self.graph.search(self.query, filter_labels=Filter(valid_labels), eps=0.1, k=k)
        assert results.shape[-1] == k
        assert np.all(np.isin(results, valid_labels))

    def test_filter_all_labels(self):
        all_labels = np.arange(self.samples, dtype=np.int32)
        k = 10
        results, _ = self.graph.search(self.query, filter_labels=Filter(all_labels), eps=0.1, k=k)
        assert results.shape[-1] == k

    def test_filter_no_valid_labels(self):
        no_labels = np.array([], dtype=np.int32)
        k = 5
        with pytest.warns(UserWarning):
            results, _ = self.graph.search(self.query, filter_labels=Filter(no_labels, max_value=0), eps=0.1, k=k)
        assert results.shape[-1] == 0
