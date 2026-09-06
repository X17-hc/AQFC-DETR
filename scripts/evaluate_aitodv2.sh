#!/usr/bin/env bash
set -euo pipefail
python main.py --config configs/aitodv2/aqfc_r50_5scale_24e.py \
  --data-root "${1:?AITODv2 data root required}" \
  --resume "${2:?checkpoint required}" --output-dir "${3:-outputs/aitodv2_eval}" --eval
