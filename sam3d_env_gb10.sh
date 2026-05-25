#!/usr/bin/env bash
# Activate the sam-3d-objects inference environment ported to NVIDIA GB10 (aarch64 + Blackwell sm_121, CUDA 13).
# Usage:  source sam3d_env_gb10.sh
#
# Built by porting effort on 2026-05-25: torch 2.9.1+cu130, pytorch3d/kaolin/spconv(+cumm)/gsplat
# all source-built for aarch64+cu130. spconv/cumm/gsplat JIT-compile CUDA kernels at runtime,
# so ninja + nvcc must be on PATH and CUDA_HOME set (this script handles it).

# conda env
source /home/jysim/miniconda3/etc/profile.d/conda.sh
conda activate sam3d

# CUDA toolkit (system) — needed for runtime JIT (spconv/cumm/gsplat). Put env bin first so `ninja` resolves.
export PATH="/home/jysim/miniconda3/envs/sam3d/bin:/usr/local/cuda/bin:$PATH"
export CUDA_HOME="/usr/local/cuda"

# GB10 is sm_121; torch/exts are built for sm_120 + PTX which JIT-forwards to sm_121.
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
export CUMM_CUDA_ARCH_LIST="12.0+PTX"

# Use torch built-in scaled-dot-product attention (avoids xformers/flash_attn, which we did NOT build).
export ATTN_BACKEND="sdpa"

echo "[sam3d-gb10] env ready. python=$(which python)  nvcc=$(nvcc --version | tail -1)"
echo "[sam3d-gb10] run demo:  cd third_party/sam-3d-objects && python demo.py   (needs checkpoints/hf)"
