import base64
import json

import pytest
import torch
from conftest import free_port, make_wav, result

from comfyui_audiocpp import audio, client


def tts(nodes, server, model="tts-a", text="Hello.", seed=0, voice="", language="", speed=1.0, max_tokens=0, options="", **optional):
    return result(nodes.AudioCPPTextToSpeech.execute(server, model, text, seed, voice, language, speed, max_tokens, options, **optional))[0]


def test_schema_ids(nodes):
    for cls, node_id in [(nodes.AudioCPPServer, "AudioCPPServer"), (nodes.AudioCPPDoctor, "AudioCPPDoctor"),
                         (nodes.AudioCPPTextToSpeech, "AudioCPPTextToSpeech")]:
        assert cls.define_schema().node_id == node_id


def test_payload_sends_only_what_is_set(nodes):
    assert nodes.speech_payload("m", "Hi", 5) == {"model": "m", "input": "Hi", "seed": "5"}
    payload = nodes.speech_payload("m", "Hi", 2**64 - 1, voice=" alba ", language="en", speed=1.25, max_tokens=300,
                                   options='{"temperature": 0.7, "x_vector_only_mode": true}', reference_text=" ref ")
    assert payload == {"model": "m", "input": "Hi", "seed": "18446744073709551615", "voice": "alba", "language": "en", "speed": 1.25,
                       "max_tokens": 300, "options": {"temperature": 0.7, "x_vector_only_mode": True}, "reference_text": "ref"}


@pytest.mark.parametrize("options", ["[1]", "{bad", '{"a": {"b": 1}}', '{"a": null}'])
def test_options_must_be_an_object_of_scalars(nodes, options):
    with pytest.raises(ValueError):
        nodes.speech_payload("m", "Hi", 0, options=options)


def test_empty_text_is_rejected_before_any_request(nodes, fake):
    with pytest.raises(ValueError, match="text is empty"):
        tts(nodes, client.Server(fake.port, 5), text=" \n ")
    assert fake.requests == []


def test_reference_audio_is_sent_as_base64_wav(nodes):
    clip = audio.wav_to_audio(make_wav(seconds=1.0, rate=16000))
    payload = nodes.speech_payload("m", "Hi", 0, reference_audio=clip, reference_text="hello")
    assert payload["voice_ref"]["type"] == "base64"
    back = audio.wav_to_audio(base64.b64decode(payload["voice_ref"]["data"]))
    assert back["sample_rate"] == 16000 and torch.equal(back["waveform"], clip["waveform"])


def test_reference_audio_over_the_server_limit(nodes):
    clip = {"waveform": torch.zeros(1, 2, 48000 * 30), "sample_rate": 48000}
    with pytest.raises(ValueError, match="at most 5 MiB"):
        nodes.speech_payload("m", "Hi", 0, reference_audio=clip)


def test_text_to_speech_returns_audio(nodes, fake):
    out = tts(nodes, client.Server(fake.port, 10), seed=42)
    assert out["sample_rate"] == 24000 and out["waveform"].shape == (1, 1, 12000)
    assert fake.requests[-1][2] == {"model": "tts-a", "input": "Hello.", "seed": "42"}


def test_empty_model_picks_the_only_speech_model(nodes, fake):
    tts(nodes, client.Server(fake.port, 10), model="")
    assert fake.requests[-1][2]["model"] == "tts-a"
    fake.models.append(dict(fake.models[0], id="tts-c"))
    with pytest.raises(ValueError, match="tts-a.*tts-c"):
        tts(nodes, client.Server(fake.port, 10), model="")


def test_unknown_model_lists_the_server_models(nodes, fake):
    with pytest.raises(client.AudioCPPError, match="unknown model id: nope.*tts-a \\(pocket_tts, task tts\\), asr-b"):
        tts(nodes, client.Server(fake.port, 10), model="nope")


def test_server_error_reaches_the_user(nodes, fake):
    fake.speech_status, fake.speech_body = 500, json.dumps({"error": {"message": "PocketTTS text prompt is required", "type": "server_error"}}).encode()
    with pytest.raises(client.ServerError, match="PocketTTS text prompt is required"):
        tts(nodes, client.Server(fake.port, 10))


def test_doctor_reports_models_and_tasks(nodes, fake):
    report, = result(nodes.AudioCPPDoctor.execute(client.Server(fake.port, 10)))
    assert "ok, backend cpu, 2 model(s)" in report
    assert "- tts-a: family pocket_tts, task tts, mode offline, not loaded, device 0 (speech)" in report
    assert "- asr-b: family qwen3_asr, task asr, mode offline, loaded, device 0 (not for Text to Speech)" in report
    assert nodes.AudioCPPDoctor.fingerprint_inputs(server=None) != nodes.AudioCPPDoctor.fingerprint_inputs(server=None)


def test_doctor_reports_a_stopped_server(nodes):
    port = free_port()
    report, = result(nodes.AudioCPPDoctor.execute(client.Server(port, 5)))
    assert report.startswith(f"audio.cpp server on http://127.0.0.1:{port}: not reachable.")


def test_doctor_without_speech_models(nodes, fake):
    fake.models = [fake.models[1]]
    report, = result(nodes.AudioCPPDoctor.execute(client.Server(fake.port, 10)))
    assert "No model with a speech task" in report


def test_server_node_validates(nodes):
    server, = result(nodes.AudioCPPServer.execute(8080, 600.0))
    assert server == client.Server(8080, 600.0)
    with pytest.raises(ValueError):
        nodes.AudioCPPServer.execute(0, 600.0)


NEMO = {"id": "nemo-c", "object": "model", "family": "nemotron_transcribe", "task": "asr", "mode": "offline", "device": 0, "loaded": True,
        "configured": True}


def test_new_schema_ids(nodes):
    for cls in (nodes.AudioCPPLocalServer, nodes.AudioCPPReferenceSpeakers, nodes.AudioCPPNemotronTranscribe, nodes.AudioCPPH3VoiceQA,
                nodes.AudioCPPVoiceModelLoader, nodes.AudioCPPUnload):
        assert cls.define_schema().node_id == cls.__name__


def test_reference_speakers_need_one_name_per_clip(nodes):
    clip = audio.wav_to_audio(make_wav(seconds=1.0, rate=24000))
    refs, = result(nodes.AudioCPPReferenceSpeakers.execute("Maren\nBrann\n", {"audio0": clip, "audio1": clip, "audio2": None}))
    assert [n for n, _ in refs] == ["Maren", "Brann"] and refs[0][1].shape == (16000,)
    with pytest.raises(ValueError, match="1 name"):
        nodes.AudioCPPReferenceSpeakers.execute("Maren", {"audio0": clip, "audio1": clip})


def test_transcribe_names_speakers_and_qa_reads_the_details(nodes, fake):
    fake.models.append(NEMO)
    rate = 16000
    # references: Maren 0-1 s, Brann 2-3 s; the take starts at 4 s
    fake.task_result = {
        "text": "<spk:0> ref <spk:1> ref <spk:0> its you",
        "speaker_turns": [{"start_sample": 0, "end_sample": rate, "speaker_id": "speaker_0", "confidence": 0},
                          {"start_sample": 2 * rate, "end_sample": 3 * rate, "speaker_id": "speaker_1", "confidence": 0},
                          {"start_sample": int(4.5 * rate), "end_sample": 5 * rate, "speaker_id": "speaker_0", "confidence": 0}],
        "custom_schema_output": {"schema": "nemotron_transcribe.result.v1", "data": {"words": [
            {"word": "its", "display": "it's", "start": 4.5, "end": 4.7, "speaker_column": 0, "speaker_confidence": 0.99, "transcript_speaker_tag": "spk:0"},
            {"word": "you", "display": "you", "start": 4.7, "end": 5.0, "speaker_column": 0, "speaker_confidence": 0.99, "transcript_speaker_tag": "spk:0"}]}}}
    clip = audio.wav_to_audio(make_wav(seconds=1.0, rate=rate))
    refs, = result(nodes.AudioCPPReferenceSpeakers.execute("Maren\nBrann", {"audio0": clip, "audio1": clip}))
    text, details = result(nodes.AudioCPPNemotronTranscribe.execute(client.Server(fake.port, 10), "", clip, "en", "", speakers=refs))
    assert text == "[Maren] it's you"
    sent = fake.requests[-1][2]
    assert sent["model"] == "nemo-c" and sent["request"]["options"] == {"diarization": True, "return_timestamps": True, "language": "en"}
    assert len(base64.b64decode(sent["request"]["audio_base64"])) == 44 + 2 * 5 * rate
    d = json.loads(details)
    assert d["words"][0] == {"word": "it's", "start": 0.5, "end": 0.7, "speaker": "Maren", "speaker_column": 0, "speaker_confidence": 0.99,
                             "transcript_speaker_tag": "spk:0"}
    assert d["server_result"] == fake.task_result
    prompt = "[Shot 1] Brann (S1) waits. Maren (S2) says: <d>[English] It's you.</d>"
    summary, report = result(nodes.AudioCPPH3VoiceQA.execute(details, prompt, ""))
    assert json.loads(report)["lines"][0]["verdict"] == "ok"
    assert summary.splitlines()[1] == 'OK: shot 1 "It\'s you." expected Maren, heard Maren at 0.50-1.00 s'
    summary, report = result(nodes.AudioCPPH3VoiceQA.execute(details, prompt, "S2: Brann"))
    assert json.loads(report)["lines"][0]["verdict"] == "wrong_voice"


def test_unload(nodes, fake):
    report, = result(nodes.AudioCPPUnload.execute(client.Server(fake.port, 10), "", False))
    assert report == 'unloaded: {"unloaded": ["asr-b"]}'
    report, = result(nodes.AudioCPPUnload.execute(client.Server(fake.port, 10), "nope", True))
    assert "not_found" in report and f"no server started by this ComfyUI on port {fake.port}" in report


def test_voice_model_needs_the_take_and_two_references(nodes):
    details = json.dumps({"speakers": [{"name": "Maren"}], "words": [], "audio_seconds": 1.0})
    with pytest.raises(ValueError, match="at least two voices"):
        nodes.AudioCPPH3VoiceQA.execute(details, "", "", voice_model=object())
