from typing import Self, Iterator

import deglib_cpp


class ObjectDistance:
    def __init__(self, object_distance_cpp: deglib_cpp.ObjectDistance):
        if not isinstance(object_distance_cpp, deglib_cpp.ObjectDistance):
            raise TypeError('Expected type deglib_cpp.ObjectDistance, got {}'.format(type(object_distance_cpp)))
        self.object_distance_cpp = object_distance_cpp

    def get_internal_index(self) -> int:
        return self.object_distance_cpp.get_internal_index()

    def get_distance(self) -> float:
        return self.object_distance_cpp.get_distance()

    def __eq__(self, other: Self):
        return self.object_distance_cpp == other.object_distance_cpp

    def __lt__(self, other: Self):
        return self.object_distance_cpp < other.object_distance_cpp

    def __gt__(self, other: Self):
        return self.object_distance_cpp > other.object_distance_cpp


class ResultSet:
    def __init__(self, result_set_cpp: deglib_cpp.ResultSet):
        if not isinstance(result_set_cpp, deglib_cpp.ResultSet):
            raise TypeError('result_set_cpp must be of type deglib_cpp.ResultSet')
        self.result_set_cpp = result_set_cpp

    def top(self) -> deglib_cpp.ObjectDistance:
        return self.result_set_cpp.top()

    def size(self) -> int:
        return self.result_set_cpp.size()

    def empty(self) -> bool:
        return self.result_set_cpp.empty()

    def __getitem__(self, index: int):
        return self.result_set_cpp[index]

    def __len__(self) -> int:
        return self.result_set_cpp.size()

    def __iter__(self) -> Iterator[ObjectDistance]:
        for i in range(self.size()):
            yield self[i]
