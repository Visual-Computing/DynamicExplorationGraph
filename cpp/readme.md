# deglib: C++ library of the Dynamic Exploration Graph

Header only C++ library of the Dynamic Exploration Graph (DEG) and its predecessor continuous refining Exploration Graph (crEG).

## How to use

### Prepare the data

Download and extract the data set files from the main [readme](../readme.md) file.

### Prerequisites

* **C++ Compiler**: A modern C++20 compiler (GCC 10.0+, Clang 11.0+, MSVC 2022+, or AppleClang).
* **CMake**: Version 3.19+

IMPORTANT NOTE: this code is highly optimized using AVX2 instructions for fast distance computation.

### Compile

#### 1. Install Dependencies

Select your operating system and preferred setup method:

##### Windows (Command-line setup via winget)
```powershell
# Install MSVC C++ Compiler (Visual Studio Build Tools)
$ winget install --id Microsoft.VisualStudio.2022.BuildTools --override "--add Microsoft.VisualStudio.Workload.VCTools --passive"

# Install CMake
$ winget install --id Kitware.CMake
```

##### Windows (Manual setup)
1. Go to the [Visual Studio Download Page](https://visualstudio.microsoft.com/downloads/).
2. Scroll down to the bottom of the page and expand the section **"Tools for Visual Studio"**.
3. Download the installer for **"Build Tools for Visual Studio 2022"**.
4. Run the installer, select the **"Desktop development with C++"** workload (which installs the required C++ Build Tools), and complete the installation.
5. Download and install [CMake for Windows](https://cmake.org/download/).

##### Linux (Ubuntu/Debian)
```bash
$ sudo apt-get update && sudo apt-get install build-essential cmake
```

##### macOS (Command-line setup)
```bash
# Install AppleClang C++ Compiler (Xcode Command Line Tools)
$ xcode-select --install

# Install CMake via Homebrew
$ brew install cmake
```

##### macOS (Manual setup)
1. Open the **Mac App Store**, search for **"Xcode"**, and install it (this will install the AppleClang compiler). Launch Xcode once after installation to accept the license agreement.
2. Download and install [CMake for macOS](https://cmake.org/download/) directly.

#### 2. Configure and Build (via CMake Presets)

Rename `CMakePresets.json.sample` to `CMakePresets.json` and change the `DATA_PATH` cache variable inside of the file to point to the directory where your datasets are located.

Then, compile the project using standard CMake Presets from the root directory:

```bash
# Configure using your environment's preset (e.g. "windows-msvc", "linux-gcc", or "macos-clang")
cmake --preset <Preset-Name>

# Compile the Release target (e.g. "windows-msvc-release", "linux-gcc-release", or "macos-clang-release")
cmake --build --preset <Build-Preset-Name>
```

#### 3. Running Unit Tests

Each unit test module compiles to its own executable in the build output directory (e.g. `test_fp32_l2`, `test_config_cascade`, `test_distances`, etc.). 

To run the unit tests for the L2 Float distance calculations we just implemented:

##### On Windows:
```powershell
# From the repository root directory:
.\build\windows-msvc\bin\Release\test_fp32_l2.exe
```

##### On Linux / macOS:
```bash
# From the repository root directory:
./build/<Preset-Name>/bin/Release/test_fp32_l2
```

You can run any of the other test executables (e.g. `test_distances`, `test_fp32_inner_product`, etc.) in the same directory.

### Reproduce our results

To create and evaluate a new graph, run the `deglib_build_and_test` executable. Existing graphs can be tested and evaluated using the `deglib_test` executable. There are some parameters which are used in older papers. Parameters with a value of 0 are ignored. 

Parameters:

|  Dataset  |  d  | k_ext | eps_ext | k_opt | eps_opt | i_opt |
|:---------:|:---:|:-----:|:-------:|:-----:|:-------:|:-----:|
|  SIFT1M   | 30  |  60   |   0.1   |   0   |    0    |   0   |
|  DEEP1M   | 30  |  60   |   0.1   |   0   |    0    |   0   |
| GloVe-100 | 30  |  60   |   0.1   |   0   |    0    |   0   |

## Pre-build Graphs

The provided Dynamic Exploration Graphs are used in the experiments section of our paper.

|   Dataset    |                                               DEG                                                |
|:------------:|:------------------------------------------------------------------------------------------------:|
|    SIFT1M    |  [sift_128D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/sift_128D_L2_DEG30.zip)  |
| Deep1M | [deep1m_96D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/deep1m_96D_L2_DEG30.zip) |
|  GloVe-100   | [glove_100D_L2_DEG30.deg](https://static.visual-computing.com/paper/DEG/glove_100D_L2_DEG30.zip) |
