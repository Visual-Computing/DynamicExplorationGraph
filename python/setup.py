"""
We copy the pybind11 include dir and the deglib include dir to ./include to make it available for this package.
"""
import codecs
import os
import shutil
import sys
import pybind11

from setuptools import Extension, setup, Command, find_packages
from setuptools.command.sdist import sdist as sdist_class

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


class CopyBuildCommand(Command):
    description = 'Copy necessary build files'
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        copy_dirs = [
            (os.path.join('..', 'cpp'), 'lib')
        ]

        for src, dst in copy_dirs:
            if os.path.isdir(dst):
                print(f"Removing {dst}")
                shutil.rmtree(dst)
            print(f"Copying {src} to {dst}")
            shutil.copytree(src, dst)
        print("Files copied successfully.")


class CopySDist(sdist_class):
    def run(self):
        self.run_command('copy_build_files')
        super().run()


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
    version=get_version(os.path.join('src', 'deglib', '__init__.py')),
    ext_modules=[
        Extension(
            name="deglib_cpp",
            sources=[os.path.join('src', 'deg_cpp', 'deglib_cpp.cpp')],
            include_dirs=[
                os.path.join('lib', 'deglib', 'include'),
                pybind11.get_include(),
            ],
            language='c++',
            extra_compile_args=get_compile_args()
        ),
    ],
    cmdclass={
        'copy_build_files': CopyBuildCommand,
        'sdist': CopySDist,
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    # package_data={
    #     '': [os.path.join(INCLUDE_DIR, '*.h')]
    # },
    # include_package_data=True
)
