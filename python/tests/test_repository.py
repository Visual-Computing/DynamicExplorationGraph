import os
import tempfile
import pathlib
import numpy as np
import pytest

import deglib
import deglib_cpp


def create_temp_fvecs_file(data: np.ndarray) -> pathlib.Path:
    """
    Creates a temporary fvecs file with the given float32 data matrix [N, D].
    fvecs format: for each vector, 4 bytes int (dimension D), followed by D * 4 bytes floats.
    """
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_repo_test')
    os.makedirs(tmpdir, exist_ok=True)
    filepath = pathlib.Path(os.path.join(tmpdir, 'test_data.fvecs'))

    n, d = data.shape
    with open(filepath, 'wb') as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(data[i].astype(np.float32).tobytes())

    return filepath


def create_temp_ivecs_file(data: np.ndarray) -> pathlib.Path:
    """
    Creates a temporary ivecs file with the given int32 data matrix [N, D].
    ivecs format: for each vector, 4 bytes int (dimension D), followed by D * 4 bytes int32.
    """
    tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_repo_test')
    os.makedirs(tmpdir, exist_ok=True)
    filepath = pathlib.Path(os.path.join(tmpdir, 'test_data.ivecs'))

    n, d = data.shape
    with open(filepath, 'wb') as f:
        for i in range(n):
            f.write(np.int32(d).tobytes())
            f.write(data[i].astype(np.int32).tobytes())

    return filepath


class TestRepository:
    def setup_method(self):
        self.samples = 50
        self.dims = 32
        self.data = np.random.random((self.samples, self.dims)).astype(np.float32)
        self.fvecs_path = create_temp_fvecs_file(self.data)

    def teardown_method(self):
        if self.fvecs_path.exists():
            os.remove(self.fvecs_path)

    def test_fvecs_read(self):
        loaded_data = deglib.repository.fvecs_read(self.fvecs_path)
        assert loaded_data.shape == (self.samples, self.dims)
        assert loaded_data.dtype == np.float32
        assert np.allclose(loaded_data, self.data)

    def test_ivecs_read(self):
        int_data = np.random.randint(0, 1000, size=(20, 10), dtype=np.int32)
        ivecs_path = create_temp_ivecs_file(int_data)
        try:
            loaded = deglib.repository.ivecs_read(ivecs_path)
            assert loaded.shape == (20, 10)
            assert loaded.dtype == np.int32
            assert np.array_equal(loaded, int_data)
        finally:
            if ivecs_path.exists():
                os.remove(ivecs_path)

    def test_load_static_repository(self):
        repo = deglib_cpp.load_static_repository(str(self.fvecs_path))
        assert isinstance(repo, deglib_cpp.StaticFeatureRepository)
        assert repo.size() == self.samples
        assert repo.dims() == self.dims

        # Check feature contents for each vertex
        for i in range(self.samples):
            feat = repo.get_feature(i)
            assert isinstance(feat, memoryview)
            feat_arr = np.frombuffer(feat, dtype=np.float32)
            assert feat_arr.shape == (self.dims,)
            assert np.allclose(feat_arr, self.data[i])

        # Test clear
        repo.clear()

    def test_load_static_u8vecs_repository(self):
        u8_data = np.random.randint(0, 256, size=(40, 16), dtype=np.uint8)
        tmpdir = os.path.join(tempfile.gettempdir(), 'deglib_repo_test')
        os.makedirs(tmpdir, exist_ok=True)
        u8_path = pathlib.Path(os.path.join(tmpdir, 'test_data.u8vecs'))

        with open(u8_path, 'wb') as f:
            for i in range(40):
                f.write(np.int32(16).tobytes())
                f.write(u8_data[i].tobytes())

        try:
            repo = deglib_cpp.load_static_repository(str(u8_path))
            assert repo.size() == 40
            assert repo.dims() == 16
        finally:
            if u8_path.exists():
                os.remove(u8_path)


