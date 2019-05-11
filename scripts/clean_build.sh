#!/bin/bash

rm -r .tox .pytest_cache
find . -type d -name "__pycache__" -o -name "*.egg-info" -o -name "build"| xargs rm -r