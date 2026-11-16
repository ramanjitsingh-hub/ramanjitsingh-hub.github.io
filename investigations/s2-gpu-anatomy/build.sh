#!/usr/bin/env bash
# Builds bin/anat. Requires CUDA 13.x and MSVC 2022 Build Tools (Windows) or nvcc + g++ (Linux).
set -e
cd "$(dirname "$0")"
mkdir -p bin
NVCC="${NVCC:-/c/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.5/bin/nvcc.exe}"
CCBIN="${CCBIN:-$(ls -d "/c/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Tools/MSVC/"*/bin/Hostx64/x64 2>/dev/null | tail -1)}"
"$NVCC" ${CCBIN:+-ccbin "$CCBIN"} -O3 -arch=sm_86 -lineinfo -std=c++17 -o bin/anat src/anat.cu
echo built bin/anat
