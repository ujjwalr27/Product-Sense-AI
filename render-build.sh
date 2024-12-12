#!/usr/bin/env bash
set -e

# Add required build packages for Render environment
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get update && apt-get install -y \
    git-lfs \
    python3-dev \
    gcc \
    g++ \
    libfreetype6-dev

# Continue with Python package installation
python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install --no-cache-dir --only-binary :all: numpy==1.24.3
python3 -m pip install --no-cache-dir --only-binary :all: pandas matplotlib
python3 -m pip install --no-cache-dir tokenizers
python3 -m pip install --no-cache-dir wordcloud
python3 -m pip install --no-cache-dir transformers
python3 -m pip install --no-cache-dir -r requirements.txt