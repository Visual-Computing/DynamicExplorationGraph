import pathlib

import deglib_cpp


# TODO: wrap StaticFeatureRepository in python class
def load_static_repository(path: pathlib.Path | str) -> deglib_cpp.StaticFeatureRepository:
    return deglib_cpp.load_static_repository(str(path))
