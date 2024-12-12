#!/usr/bin/env bash
set -e

# Upgrade pip and core tools
python3 -m pip install --upgrade pip setuptools wheel

# Continue with Python package installation
python3 -m pip install --no-cache-dir --only-binary :all: numpy==1.24.3
python3 -m pip install --no-cache-dir --only-binary :all: pandas matplotlib
python3 -m pip install --no-cache-dir tokenizers
python3 -m pip install --no-cache-dir wordcloud
python3 -m pip install --no-cache-dir transformers
python3 -m pip install --no-cache-dir -r requirements.txt