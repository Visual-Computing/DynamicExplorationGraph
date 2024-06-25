from typing import Self, Iterator

import deglib_cpp


class ObjectDistance:
    def __init__(self, object_distance_cpp: deglib_cpp.ObjectDistance):
        """
        Most users should not create an ObjectDistance themselves.
        """
        if not isinstance(object_distance_cpp, deglib_cpp.ObjectDistance):
            raise TypeError('Expected type deglib_cpp.ObjectDistance, got {}'.format(type(object_distance_cpp)))
        self.object_distance_cpp = object_distance_cpp

    def get_internal_index(self) -> int:
        """
        The internal index of the found vertex.
        """
        return self.object_distance_cpp.get_internal_index()

    def get_distance(self) -> float:
        """
        The distance of the found vertex to the query.
        """
        return self.object_distance_cpp.get_distance()

    def __eq__(self, other: Self):
        """
        Checks equality with other. Two ObjectDistances are equal, if the the found vertex and the distance is equal.
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


class ResultSet:
    def __init__(self, result_set_cpp: deglib_cpp.ResultSet):
        """
        Most users should not create a ResultSet themselves.
        """
        if not isinstance(result_set_cpp, deglib_cpp.ResultSet):
            raise TypeError('result_set_cpp must be of type deglib_cpp.ResultSet')
        self.result_set_cpp = result_set_cpp

    def top(self) -> ObjectDistance:
        """
        Get the "best" result of the search.
        """
        return ObjectDistance(self.result_set_cpp.top())

    def pop(self):
        """
        Get the "best" result of the search and remove it from the set.
        """
        self.result_set_cpp.pop()

    def size(self) -> int:
        """
        :returns: the number of results
        """
        return self.result_set_cpp.size()

    def empty(self) -> bool:
        """
        Whether the set is empty
        """
        return self.result_set_cpp.empty()

    def __getitem__(self, index: int) -> ObjectDistance:
        """
        Get item by index.
        """
        return ObjectDistance(self.result_set_cpp[index])

    def __len__(self) -> int:
        """
        Same as size().
        """
        return self.result_set_cpp.size()

    def __iter__(self) -> Iterator[ObjectDistance]:
        """
        Iterator over all elements of the set. The results are NOT ordered by quality!
        TODO: order by quality
        """
        for i in range(self.size()):
            yield self[i]

    def __repr__(self):
        return f'ResultSet(size={self.size()} best_dist={self.top().get_distance():.3f})'
