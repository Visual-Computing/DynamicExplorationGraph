from typing import Optional, Union
import numpy as np

import deglib_cpp.search as cpp_search

from deglib.utils import assure_contiguous


class Filter:
    def __init__(self, valid_labels: np.ndarray, max_value: int = -1, max_label_count: int = -1):
        """
        Creates an object that can be used to limit the set of possible results.

        :param valid_labels: A numpy array with dtype int32, that contains all labels that can be returned.
                              All other labels will not be included in the result set.
        :param max_value: The maximum value in valid_labels. Will be computed automatically, if set to -1.
        :param max_label_count: The size of the whole dataset. If not set, the size of the search graph is assumed.
        """
        self.valid_labels = valid_labels
        if max_value < 0:
            max_value = np.max(valid_labels)
        self.max_value = max_value
        self.max_label_count = max_label_count

    def create_filter_obj(self, graph_size: int) -> cpp_search.Filter:
        """
        Only for internal use.
        Creates a filter object that can be used to limit the set of possible results.
        """
        valid_labels = assure_contiguous(self.valid_labels.astype(np.int32, copy=False), 'filter_labels')
        filter_obj = None
        if valid_labels is not None:
            max_label_count = self.max_label_count
            if max_label_count <= 0:
                max_label_count = graph_size
            filter_obj = cpp_search.create_filter(valid_labels, self.max_value, max_label_count)

        return filter_obj

    @staticmethod
    def create_filter(filter_labels: Union[None, np.ndarray, 'Filter'], graph_size: int) -> Optional[cpp_search.Filter]:
        if filter_labels is None:
            return None
        if isinstance(filter_labels, np.ndarray):
            filter_labels = Filter(filter_labels)
        if not isinstance(filter_labels, Filter):
            raise TypeError('filter_labels must be a None, numpy array or Filter, got {}'.format(type(filter_labels)))
        return filter_labels.create_filter_obj(graph_size)


__all__ = ['Filter']
