#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../models/aqfcdetr/ops"
python setup.py build install
python test.py
