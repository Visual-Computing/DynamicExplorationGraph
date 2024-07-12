"""
We copy the pybind11 include dir and the deglib include dir to ./include to make it available for this package.
"""
import codecs
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pybind11

from setuptools import Extension, setup, Command, find_packages
from setuptools.command.sdist import sdist as sdist_class
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


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
            print(f"Copying {src} to {dst}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        print("Files copied successfully.")

        rm_dirs = [
            os.path.join('lib', 'external'),
            os.path.join('lib', 'cmake-build-default'),
            os.path.join('lib', 'build'),
            os.path.join('lib', 'benchmark'),
        ]
        for rm_dir in rm_dirs:
            if os.path.isdir(rm_dir):
                print(f'Removing "{rm_dir}"')
                shutil.rmtree(rm_dir)


class CopySDist(sdist_class):
    def run(self):
        self.run_command('copy_build_files')
        super().run()


# A CMakeExtension needs a sourcedir instead of a file list.
# The name must be the _single_ output extension from the CMake build.
# If you need multiple extensions, see scikit-build.
class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = os.fspath(Path(sourcedir).resolve())


def call_cmake_checked(command, cwd):
    result = subprocess.run(command, cwd=cwd, capture_output=True)

    print('STDOUT: ' + '-'*33 + '\n', result.stdout.decode('utf-8'), '-'*42)
    print('STDERR: ' + '-'*33 + '\n', result.stderr.decode('utf-8'), '-'*42)
    if result.returncode != 0:
        result.check_returncode()


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        # Must be in this form due to bug in .resolve() only fixed in Python 3.10+
        ext_fullpath = Path.cwd() / self.get_ext_fullpath(ext.name)
        extdir = ext_fullpath.parent.resolve()

        # Using this requires trailing slash for auto-detection & inclusion of
        # auxiliary "native" libs

        debug = int(os.environ.get("DEBUG", 0)) if self.debug is None else self.debug
        cfg = "Debug" if debug else "Release"

        # CMake lets you override the generator - we need to check this.
        # Can be set with Conda-Build, for example.
        cmake_generator = os.environ.get("CMAKE_GENERATOR", "")

        # Set Python_EXECUTABLE instead if you use PYBIND11_FINDPYTHON
        # EXAMPLE_VERSION_INFO shows you how to pass a value into the C++ code
        # from Python.
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}{os.sep}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",  # not used on MSVC, but no harm
        ]
        build_args = []
        # Adding CMake arguments set as environment variable
        # (needed e.g. to build for ARM OSx on conda-forge)
        if "CMAKE_ARGS" in os.environ:
            cmake_args += [item for item in os.environ["CMAKE_ARGS"].split(" ") if item]

        if os.environ.get("FORCE_AVX2") is not None and os.environ["FORCE_AVX2"] == "1":
            cmake_args.append("-DFORCE_AVX2=ON")
            print('Enabling FORCE_AVX2')

        # In this example, we pass in the version to C++. You might not need to.
        # cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        if self.compiler.compiler_type != "msvc":
            # Using Ninja-build since it a) is available as a wheel and b)
            # multi-threads automatically. MSVC would require all variables be
            # exported for Ninja to pick it up, which is a little tricky to do.
            # Users can override the generator with CMAKE_GENERATOR in CMake
            # 3.15+.
            if not cmake_generator or cmake_generator == "Ninja":
                try:
                    # noinspection PyUnresolvedReferences
                    import ninja

                    ninja_executable_path = Path(ninja.BIN_DIR) / "ninja"
                    cmake_args += [
                        "-GNinja",
                        f"-DCMAKE_MAKE_PROGRAM:FILEPATH={ninja_executable_path}",
                    ]
                except ImportError:
                    pass

        else:
            # Single config generators are handled "normally"
            single_config = any(x in cmake_generator for x in {"NMake", "Ninja"})

            # CMake allows an arch-in-generator style for backward compatibility
            contains_arch = any(x in cmake_generator for x in {"ARM", "Win64"})

            # Specify the arch if using MSVC generator, but only if it doesn't
            # contain a backward-compatibility arch spec already in the
            # generator name.
            if not single_config and not contains_arch:
                cmake_args += ["-A", PLAT_TO_CMAKE[self.plat_name]]

            # Multi-config generators have a different way to specify configs
            if not single_config:
                cmake_args += [
                    f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{cfg.upper()}={extdir}"
                ]
                build_args += ["--config", cfg]

        if sys.platform.startswith("darwin"):
            # Cross-compile support for macOS - respect ARCHFLAGS if set
            archs = re.findall(r"-arch (\S+)", os.environ.get("ARCHFLAGS", ""))
            if archs:
                cmake_args += ["-DCMAKE_OSX_ARCHITECTURES={}".format(";".join(archs))]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        # across all generators.
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # self.parallel is a Python 3 only way to set parallel jobs by hand
            # using -j in the build_ext call, not supported by pip or PyPA-build.
            if hasattr(self, "parallel") and self.parallel:
                # CMake 3.12+ only.
                build_args += [f"-j{self.parallel}"]

        build_temp = Path(self.build_temp) / ext.name
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        print('python interpreter in setup.py:', sys.executable, sys.version)
        print('calling cmake')
        call_cmake_checked(
            ["cmake", ext.sourcedir, '-Dpybind11_DIR={}'.format(pybind11.get_cmake_dir()), *cmake_args],
            cwd=build_temp
        )
        print('calling cmake --build')
        call_cmake_checked(["cmake", "--build", ".", *build_args], cwd=build_temp)


setup(
    version=get_version(os.path.join('src', 'deglib', '__init__.py')),
    ext_modules=[CMakeExtension("deglib_cpp")],
    cmdclass={
        'copy_build_files': CopyBuildCommand,
        'sdist': CopySDist,
        'build_ext': CMakeBuild,
    },
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
)

