#!/usr/bin/env bash
# Build audiocpp_server and audiocpp_cli from a pinned audio.cpp commit, with CUDA, for the models the nodes use.
#
#   scripts/build_audiocpp.sh <directory>
#
# Clones audio.cpp into <directory> (or reuses a clone there), checks out AUDIOCPP_COMMIT and builds into <directory>/build/release.
# Environment:
#   AUDIOCPP_COMMIT  the audio.cpp commit to build (default: the one these nodes were tested with)
#   AUDIOCPP_MODELS  comma-separated model families to compile in, or "full" for all of them (default: nemotron_transcribe)
#   CUDA_ARCH        CMAKE_CUDA_ARCHITECTURES (default: native, the GPUs in this computer; 120 for RTX 50xx)
#   BACKEND          cuda or cpu (default: cuda)
#   JOBS             parallel compile jobs (default: number of CPUs)
set -euo pipefail

AUDIOCPP_COMMIT=${AUDIOCPP_COMMIT:-d2b2076e47affef4973a9f69433750b339be7d8d}
AUDIOCPP_MODELS=${AUDIOCPP_MODELS:-nemotron_transcribe}
CUDA_ARCH=${CUDA_ARCH:-native}
BACKEND=${BACKEND:-cuda}
JOBS=${JOBS:-$(nproc)}
DEST=${1:?usage: build_audiocpp.sh <directory>}

if [[ ! -d "$DEST/.git" ]]; then
    git clone https://github.com/0xShug0/audio.cpp.git "$DEST"
fi
git -C "$DEST" fetch --quiet origin
git -C "$DEST" checkout --quiet "$AUDIOCPP_COMMIT"

args=(-S "$DEST" -B "$DEST/build/release" -DCMAKE_BUILD_TYPE=Release -DAUDIOCPP_DEPLOYMENT_BUILD=ON)
if [[ "$AUDIOCPP_MODELS" == "full" ]]; then
    args+=(-DAUDIOCPP_MODEL_SET=full)
else
    args+=(-DAUDIOCPP_MODEL_SET=custom "-DAUDIOCPP_MODELS=$AUDIOCPP_MODELS")
fi
if [[ "$BACKEND" == "cuda" ]]; then
    # Name the toolkit explicitly: a distro nvcc earlier on PATH would otherwise link an older libcudart (audio.cpp docs/build/linux.md).
    CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
    args+=(-DENGINE_ENABLE_CUDA=ON "-DCUDAToolkit_ROOT=$CUDA_HOME" "-DCMAKE_CUDA_COMPILER=$CUDA_HOME/bin/nvcc" "-DCMAKE_CUDA_ARCHITECTURES=$CUDA_ARCH")
fi

cmake "${args[@]}"
cmake --build "$DEST/build/release" --target audiocpp_server audiocpp_cli --parallel "$JOBS"

echo
echo "built $(git -C "$DEST" rev-parse HEAD):"
ls -l "$DEST/build/release/bin/audiocpp_server" "$DEST/build/release/bin/audiocpp_cli"
