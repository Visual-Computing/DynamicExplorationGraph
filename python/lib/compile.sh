#!/bin/bash

if [ ! -d build ]; then
	mkdir build
fi

# cmake -B build -DCMAKE_BUILD_TYPE=Release "-GUnix Makefiles" -D DATA_PATH="/run/media/alok/SanDiskBruno/DEG"
cmake -B build -DCMAKE_BUILD_TYPE=Release --preset default -G "Unix Makefiles"  # -DENABLE_BENCHMARKS=OFF
# cmake -B build -DCMAKE_BUILD_TYPE=Release -G "Unix Makefiles"

cd build
make -j 6
