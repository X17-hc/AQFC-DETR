#!/usr/bin/env bash
# Compile with this project's own CUDA 12.4 / GCC toolchain, without using a GPU.
set -euo pipefail
AQFC_ENV="${AQFC_ENV:-/opt/conda/envs/AQFC-DETR}"
AQFC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export CUDA_HOME="$AQFC_ENV"
export PATH="$AQFC_ENV/bin:$PATH"
export CC="$AQFC_ENV/bin/x86_64-conda-linux-gnu-cc"
export CXX="$AQFC_ENV/bin/x86_64-conda-linux-gnu-c++"
export CUDA_VISIBLE_DEVICES=""
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export MAX_JOBS="${MAX_JOBS:-2}"
# PyTorch's pip CUDA packages contain the matching cuSPARSE/cuBLAS headers.
AQFC_SITE="$("$AQFC_ENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
export CPATH="$(find "$AQFC_SITE/nvidia" -maxdepth 2 -type d -name include | paste -sd:)${CPATH:+:$CPATH}"
cd "$AQFC_ROOT/models/aqfcdetr/ops"
"$AQFC_ENV/bin/python" -m pip install --no-build-isolation .
