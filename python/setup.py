"""
We copy the pybind11 include dir and the deglib include dir to ./include to make it available for this package.
"""
import codecs
import os
import shutil
import sys
import pybind

from setuptools import Extension, setup


INCLUDE_DIR = 'include'


def read(rel_path: str):
    here = os.path.abspath(os.path.dirname(os.path.abspath(__file__)))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


def is_tmp_build():
    top_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return 'DynamicExplorationGraph' != os.path.basename(top_dir)


def prepare_include_dir():
    if is_tmp_build():
        return
    os.makedirs(INCLUDE_DIR, exist_ok=True)
    to_copy = [
        (os.path.join('..', 'deglib', 'include'), 'deglib')
    ]

    for source_path, target_path in to_copy:
        target_path = os.path.join(INCLUDE_DIR, target_path)
        if os.path.exists(target_path):
            if os.path.isdir(target_path):
                shutil.rmtree(target_path)
            if os.path.isfile(target_path):
                os.remove(target_path)

        if os.path.isdir(source_path):
            print('-- copy dir {} -> {}'.format(source_path, target_path))
            shutil.copytree(source_path, target_path)
        elif os.path.isfile(source_path):
            print('-- copy file {} -> {}'.format(source_path, target_path))
            shutil.copy(source_path, target_path)


prepare_include_dir()


def get_compile_args():
    if sys.platform.startswith('linux'):
        return ['-std=c++20', '-fopenmp', '-mavx', '-march=native']
    elif sys.platform.startswith('darwin'):
        return ['-std=c++20', '-Xpreprocessor', '-fopenmp', '-lomp', '-mavx']
    elif sys.platform.startswith('win32'):
        return ['/std:c++20', '/openmp', '/arch:AVX']
    else:
        raise OSError('deglib does not support platform "{}".'.format(sys.platform))


setup(
    version=get_version('deglib/__init__.py'),
    ext_modules=[
        Extension(
            name="deglib_cpp",
            sources=["deg_cpp/deglib_cpp.cpp"],
            # TODO: maybe replace with pybind11.get_include() and remove pybind11 from local include directory
            include_dirs=[pybind11.get_include(), os.path.join('.', INCLUDE_DIR, 'deglib'), os.path.join('.', INCLUDE_DIR)],
	    language='c++',
            extra_compile_args=get_compile_args()
        ),
    ],
    packages=['deglib'],
    # package_data={
    #     '': [os.path.join(INCLUDE_DIR, '*.h')]
    # },
    # include_package_data=True
)
