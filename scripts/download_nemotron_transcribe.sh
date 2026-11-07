#!/usr/bin/env bash
# Download the Nemotron 3.5 Transcribe GGUF package for audio.cpp at a pinned revision (GGUF only, no pickles).
#
#   scripts/download_nemotron_transcribe.sh <directory> [decoder]
#
# decoder: Q4_K_S (default, ~21.7 GB), IQ4_NL (~20.4 GB), Q3_K_S (~17.3 GB) or Q2_K (~13.9 GB). The shared F16 encoder
# (~1.7 GB, with the CTC timestamp head) is always downloaded beside it: audio.cpp loads it as the decoder's sibling.
# Needs the `hf` command from huggingface_hub (pip install -U huggingface_hub, or uvx hf).
set -euo pipefail

REPO=Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF
REVISION=${NEMOTRON_REVISION:-d2affa8419718ffe0917cb69dc6bc693084497f5}
DEST=${1:?usage: download_nemotron_transcribe.sh <directory> [Q4_K_S|IQ4_NL|Q3_K_S|Q2_K]}
DECODER=${2:-Q4_K_S}

case "$DECODER" in
    Q4_K_S|IQ4_NL|Q3_K_S|Q2_K) ;;
    *) echo "unknown decoder $DECODER: use Q4_K_S, IQ4_NL, Q3_K_S or Q2_K" >&2; exit 2 ;;
esac

hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" \
    "nemotron-transcribe-30b-a4b-experts-$DECODER.gguf" \
    mmproj-NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-Encoder-F16.gguf \
    LICENSE THIRD_PARTY_NOTICES.md README.md licenses/Qwen3-ASR-LICENSE.txt licenses/vLLM-LICENSE.txt

echo
echo "decoder: $DEST/nemotron-transcribe-30b-a4b-experts-$DECODER.gguf (pass this file as the model path)"
