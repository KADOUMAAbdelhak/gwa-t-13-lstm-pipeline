#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"
CONFIG=configs/config.yaml
python3 src/training/train.py --config "$CONFIG"