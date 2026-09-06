#!/usr/bin/env bash
set -euo pipefail
data_root="${1:?AITODv2 data root required}"
for config in configs/ablations/*.py; do
  name="$(basename "$config" .py)"
  python main.py --config "$config" --data-root "$data_root" \
    --output-dir "outputs/ablations/$name" --seed 42
done
