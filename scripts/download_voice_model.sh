#!/usr/bin/env bash
# Download the WavLM-large + ECAPA-TDNN speaker-verification weights for H3 Voice QA's voice_model, at a pinned revision.
#
#   scripts/download_voice_model.sh <directory>      (for example ComfyUI/models/audio_encoders/wavlm_large_ecapa_sv)
#
# Only the safetensors weights and the licence (CC BY-SA 3.0): the pack has its own implementation of the model, so the
# repository's Python file is not needed. Needs the `hf` command from huggingface_hub.
set -euo pipefail

REPO=prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native
REVISION=${VOICE_MODEL_REVISION:-13ad4fcc740cbd47992bcc2aaa59186d4072b7c9}
DEST=${1:?usage: download_voice_model.sh <directory>}

hf download "$REPO" --revision "$REVISION" --local-dir "$DEST" model.safetensors LICENSE README.md
echo
echo "voice model: $DEST/model.safetensors (pick it in audio.cpp Voice Model)"
