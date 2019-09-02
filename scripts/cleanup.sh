#!/usr/bin/env bash

rm -r .tox .pytest_cache
rm .coverage
find . -type d -name "__pycache__" -o -name "*.egg-info" -o -name "build" | xargs rm -r
