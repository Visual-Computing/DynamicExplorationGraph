#!/bin/bash

sphinx-build -M markdown ./ _build
sphinx-build -M html ./ _build
