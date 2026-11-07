"""Against a real audiocpp_server (skipped unless AUDIOCPP_TEST_PORT is set).

The server needs a PocketTTS entry with the id in AUDIOCPP_TEST_POCKET_ID (default "pocket-tts", built-in voice), for example the
config in the README.
"""

import os

import pytest
import torch
from conftest import result

from comfyui_audiocpp import client

PORT = os.environ.get("AUDIOCPP_TEST_PORT")
POCKET = os.environ.get("AUDIOCPP_TEST_POCKET_ID", "pocket-tts")
needs_server = pytest.mark.skipif(not PORT, reason="set AUDIOCPP_TEST_PORT to the port of a running audiocpp_server")


@pytest.fixture
def server():
    return client.Server(int(PORT), 300)


def speak(nodes, server, text, seed):
    return result(nodes.AudioCPPTextToSpeech.execute(server, POCKET, text, seed, "", "", 1.0, 0, ""))[0]


@needs_server
def test_doctor_lists_the_model(nodes, server):
    report, = result(nodes.AudioCPPDoctor.execute(server))
    assert f"- {POCKET}: family pocket_tts, task tts" in report


@needs_server
def test_speech_is_audio_and_follows_the_seed(nodes, server):
    a = speak(nodes, server, "The quick brown fox jumps over the lazy dog.", 1234)
    b = speak(nodes, server, "The quick brown fox jumps over the lazy dog.", 1234)
    c = speak(nodes, server, "The quick brown fox jumps over the lazy dog.", 99)
    assert a["sample_rate"] == 24000 and a["waveform"].shape[:2] == (1, 1)
    assert 1.0 < a["waveform"].shape[-1] / a["sample_rate"] < 10.0
    assert a["waveform"].abs().max() > 0.05
    assert torch.equal(a["waveform"], b["waveform"])
    assert not torch.equal(a["waveform"], c["waveform"])


VOICE_MODEL = os.environ.get("AUDIOCPP_TEST_VOICE_MODEL")


@pytest.mark.skipif(not VOICE_MODEL, reason="set AUDIOCPP_TEST_VOICE_MODEL to the WavLM-large ECAPA safetensors file")
def test_voice_model_embeds():
    from safetensors.torch import load_file

    from comfyui_audiocpp import speaker_embedding

    net = speaker_embedding.WavLMLargeECAPA()
    net.load_state_dict(load_file(VOICE_MODEL))
    a = net.eval()(torch.sin(torch.arange(16000) / 7.0))
    assert a.shape == (256,) and torch.isfinite(a).all()
