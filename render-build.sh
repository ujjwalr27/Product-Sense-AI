#!/usr/bin/env bash
set -e

# Upgrade pip and core tools
python3 -m pip install --upgrade pip setuptools wheel

# Configure Rust environment to use a writable directory
export CARGO_HOME=/tmp/.cargo
mkdir -p $CARGO_HOME
export PATH=$CARGO_HOME/bin:$PATH

# Set additional environment variables for Rust (optional but recommended)
export RUSTUP_HOME=/tmp/.rustup

# Install Python packages with pre-built binaries where possible
python3 -m pip install --no-cache-dir --only-binary :all: numpy==1.24.3
python3 -m pip install --no-cache-dir --only-binary :all: pandas matplotlib

# Install tokenizers (Rust-based package)
python3 -m pip install --no-cache-dir tokenizers

# Install wordcloud
python3 -m pip install --no-cache-dir wordcloud

# Install transformers
python3 -m pip install --no-cache-dir transformers

# Finally, install remaining requirements
python3 -m pip install --no-cache-dir -r requirements.txt