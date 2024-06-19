#!/bin/bash

# removes and recreates the symlink of the deglib_cpp. This is mainly used to make IDEs recognice
# an updated version of the library.

if [ "$(basename $PWD)" != "python-bindings" ]; then
	echo "ERROR: This script should be executed from within the python-bindings directory!"
	echo "  $ cd path/to/python-bindings; scripts/link-lib.sh"
	exit 1
fi

if [[ -z "$CONDA_DEFAULT_ENV" ]] && [[ -z "$VIRTUAL_ENV" ]]; then
	echo "ERROR: You are not in a virtual environment!"
	exit 1
fi

# this works for conda and virtualenv
lib_path="$(python -c "import site; print(site.getsitepackages()[0])")"

source_path="$(realpath $PWD/../build/python-bindings/deglib_cpp.*.so)"
target_path="$lib_path/$(basename $source_path)"

if [ ! -f "$source_path" ]; then
	echo "ERROR: Source \"$source_path\" is not present. You have to compile first, before linking for python!"
	exit 1
fi

if [ -f "$target_path" ]; then
	echo "Removing target \"$target_path\""
	rm "$target_path"
	sleep 2
else
	echo "WARN: Target \"$target_path\" is not present. Creating link, without removing"
fi

echo "link \"$source_path\""
echo "  -> \"$target_path\""
ln -s "$source_path" "$target_path"