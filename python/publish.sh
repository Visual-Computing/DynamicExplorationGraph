#!/bin/bash

set -e

read -p "have you tested with BuildAndTest-Workflow? y/[n]: " answer
if [ "$answer" != "y" ]; then
	exit 0
fi

# get new version from __init__.py
new_version=$(grep --color=never -oP '__version__\s*=\s*"\K[0-9]+\.[0-9]+\.[0-9]+' "src/deglib/__init__.py")
echo "detected new version: $new_version"

# get old version from pypi
old_version=$(curl -s "https://pypi.org/pypi/deglib/json" | jq -r '.info.version')

# compare versions
if [ "$old_version" = "$new_version" ]; then
	echo "ERROR: current version equals version on pypi"
	read -p "have you changed the version in \"src/deglib/__init__.py\"? y/[n]: " answer
	if [ "$answer" != "y" ]; then
		exit 0
	fi
fi

# publish package
git add -A && git commit -m "v$new_version"
git push

# publish tag
git tag -a "v$new_version" -m "v$new_version"
git push origin --tags

