from typing import Self, Iterator, Union, Optional
import numpy as np

import deglib_cpp

from deglib.utils import assure_contiguous


class ObjectDistance:
    def __init__(self, object_distance_cpp: deglib_cpp.ObjectDistance):
        """
        Most users should not create an ObjectDistance themselves.
        """
        if not isinstance(object_distance_cpp, deglib_cpp.ObjectDistance):
            raise TypeError('Expected type deglib_cpp.ObjectDistance, got {}'.format(type(object_distance_cpp)))
        self.object_distance_cpp = object_distance_cpp
        self.distance: float = self.object_distance_cpp.get_distance()
        self.internal_index: int = self.object_distance_cpp.get_internal_index()

    def get_internal_index(self) -> int:
        """
        The internal index of the found vertex.
        """
        return self.internal_index

    def get_distance(self) -> float:
        """
        The distance of the found vertex to the query.
        """
        return self.distance

    def __eq__(self, other: Self):
        """
        Checks equality with other. Two ObjectDistances are equal, if the found vertex and the distance is equal.
        """
        return self.object_distance_cpp == other.object_distance_cpp

    def __lt__(self, other: Self):
        """
        Checks whether this vertex is closer to the query than other.
        """
        return self.object_distance_cpp < other.object_distance_cpp

    def __gt__(self, other: Self):
        """
        Checks whether this vertex is further to the query than other.
        """
        return self.object_distance_cpp > other.object_distance_cpp

    def __repr__(self):
        return f'ObjectDistance(internal_index={self.internal_index} distance={self.distance})'


class ResultSet:
    def __init__(self, result_set_cpp: deglib_cpp.ResultSet):
        """
        Most users should not create a ResultSet themselves.
        """
        if not isinstance(result_set_cpp, deglib_cpp.ResultSet):
            raise TypeError('result_set_cpp must be of type deglib_cpp.ResultSet')
        self.result_set_cpp = result_set_cpp
        self.result_list = sorted(
            [ObjectDistance(self.result_set_cpp[i]) for i in range(self.result_set_cpp.size())],
            key=lambda obj_dist: obj_dist.distance
        )

    def top(self) -> ObjectDistance:
        """
        Get the "best" result of the search.
        """
        return self.result_list[0]

    def pop(self):
        """
        Get the "best" result of the search and remove it from the set.
        """
        result = self.result_list[0]
        del self.result_list[0]
        return result

    def size(self) -> int:
        """
        :returns: the number of results
        """
        return len(self.result_list)

    def empty(self) -> bool:
        """
        Whether the set is empty
        """
        return not bool(self.result_list)

    def __getitem__(self, index: int) -> ObjectDistance:
        """
        Get item by index.
        """
        return self.result_list[index]

    def __len__(self) -> int:
        """
        Same as size().
        """
        return self.size()

    def __iter__(self) -> Iterator[ObjectDistance]:
        """
        Iterator over all elements of the set sorted by distance (first entry has lowest distance).
        """
        return iter(self.result_list)

    def __repr__(self):
        return f'ResultSet(size={self.size()} best_dist={self.top().distance:.3f})'


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

    def create_filter_obj(self, graph_size: int) -> deglib_cpp.Filter:
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
            filter_obj = deglib_cpp.create_filter(valid_labels, self.max_value, max_label_count)

        return filter_obj

    @staticmethod
    def create_filter(filter_labels: Union[None, np.ndarray, 'Filter'], graph_size: int) -> Optional[deglib_cpp.Filter]:
        if filter_labels is None:
            return None
        if isinstance(filter_labels, np.ndarray):
            filter_labels = Filter(filter_labels)
        if not isinstance(filter_labels, Filter):
            raise TypeError('filter_labels must be a None, numpy array or Filter, got {}'.format(type(filter_labels)))
        return filter_labels.create_filter_obj(graph_size)


__all__ = ['ObjectDistance', 'ResultSet', 'Filter']