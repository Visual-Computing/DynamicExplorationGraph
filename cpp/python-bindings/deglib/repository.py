import pathlib
import numpy as np

import deglib_cpp


class StaticFeatureRepository:
    def __init__(self, repository_cpp: deglib_cpp.StaticFeatureRepository):
        self.repository_cpp = repository_cpp

    def get_feature(self, vertex_id: int) -> np.ndarray:
        memory_view = self.repository_cpp.get_feature(vertex_id)
        feature_vector = np.asarray(memory_view)
        return feature_vector

    def size(self) -> int:
        return self.repository_cpp.size()

    def dims(self) -> int:
        return self.repository_cpp.dims()

    def clear(self):
        self.repository_cpp.clear()


def load_static_repository(path: pathlib.Path | str) -> StaticFeatureRepository:
    repo_cpp = deglib_cpp.load_static_repository(str(path))
    return StaticFeatureRepository(repo_cpp)
