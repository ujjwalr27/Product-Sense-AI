#!/usr/bin/env bash

# Exit on error
set -e

# Upgrade pip and core tools
python -m pip install --upgrade pip setuptools wheel

# Install build dependencies first
python -m pip install --no-cache-dir build

# Install binary packages first with specific versions
python -m pip install --no-cache-dir --only-binary :all: numpy==1.24.3
python -m pip install --no-cache-dir --only-binary :all: pandas matplotlib

# Install wordcloud and transformers with their dependencies
python -m pip install --no-cache-dir wordcloud
python -m pip install --no-cache-dir transformers

# Finally install remaining requirements
python -m pip install --no-cache-dir -r requirements.txt
