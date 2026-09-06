#!/usr/bin/env bash
set -euo pipefail
python main.py --config configs/aitodv2/aqfc_r50_5scale_24e.py \
  --data-root "${1:?AITODv2 data root required}" \
  --output-dir "${2:-outputs/aitodv2_main}"
