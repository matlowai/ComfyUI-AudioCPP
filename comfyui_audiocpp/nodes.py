"""ComfyUI nodes for audio.cpp: speech from an audiocpp_server on this computer, and Nemotron Transcribe (words, word times and
speakers named from reference clips) with a check of MiniMax H3 takes against their prompts."""

from __future__ import annotations

import base64
import json
import logging
import os

import numpy as np
import torch
from comfy_api.latest import ComfyExtension, io, ui

import comfy.audio
import comfy.model_management
import comfy.model_patcher
import comfy.utils
import folder_paths

from . import audio, client, h3qa, local_server, speaker_embedding, speakers

# Tasks whose models take text on POST /v1/audio/speech (task names as GET /v1/models reports them).
SPEECH_TASKS = ("tts", "clon", "vdes")

AudioCPPServerType = io.Custom("AUDIOCPP_SERVER")
SpeakersType = io.Custom("AUDIOCPP_SPEAKERS")
VoiceModelType = io.Custom("AUDIOCPP_VOICE_MODEL")
# The diarizer has 8 speaker columns (pee.diar.num_speakers); one stays free for a voice that matches no reference.
MAX_REFERENCES = 7


def _check_interrupt():
    comfy.model_management.throw_exception_if_processing_interrupted()


def _speech_models(models: list[dict]) -> list[dict]:
    return [m for m in models if m.get("task") in SPEECH_TASKS]


def _model_list(models: list[dict]) -> str:
    return ", ".join(f"{m.get('id')} ({m.get('family')}, task {m.get('task')})" for m in models) or "none"


def _options(text: str) -> dict:
    """The options input: a JSON object of strings, numbers or booleans (the server's request options are scalars)."""
    if not text.strip():
        return {}
    try:
        value = json.loads(text)
    except ValueError as error:
        raise ValueError(f"options must be a JSON object, e.g. {{\"temperature\": 0.7}}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError("options must be a JSON object, e.g. {\"temperature\": 0.7}")
    for key, item in value.items():
        if isinstance(item, (dict, list)) or item is None:
            raise ValueError(f"option {key!r} must be a string, number or boolean")
    return value


def speech_payload(model, text, seed, voice="", language="", speed=1.0, max_tokens=0, options="", reference_audio=None, reference_text=""):
    """The JSON body of POST /v1/audio/speech for the Text to Speech node's inputs."""
    if not text.strip():
        raise ValueError("text is empty")
    payload = {"model": model, "input": text}
    extra = _options(options)
    if extra:
        payload["options"] = extra
    # A string keeps all 64 bits of the seed (the server parses request options from strings either way).
    payload["seed"] = str(seed)
    if voice.strip():
        payload["voice"] = voice.strip()
    if language.strip():
        payload["language"] = language.strip()
    if speed != 1.0:
        payload["speed"] = speed
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    if reference_audio is not None:
        wav = audio.audio_to_wav(reference_audio)
        if len(wav) > client.MAX_VOICE_REF_BYTES:
            seconds = reference_audio["waveform"].shape[-1] / reference_audio["sample_rate"]
            raise ValueError(f"reference_audio is {seconds:.1f} s ({len(wav) / 2**20:.1f} MiB as 16-bit WAV); the server takes at most "
                             f"{client.MAX_VOICE_REF_BYTES // 2**20} MiB, so trim it")
        payload["voice_ref"] = {"type": "base64", "data": base64.b64encode(wav).decode("ascii")}
    if reference_text.strip():
        payload["reference_text"] = reference_text.strip()
    return payload


class AudioCPPServer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPServer",
            display_name="audio.cpp Server",
            category="audio/audio.cpp",
            description=("Connects the audio.cpp nodes to an audiocpp_server that you started on this computer (127.0.0.1). The nodes never "
                         "start, stop or download anything; the server's config decides which models are available."),
            inputs=[
                io.Int.Input("port", default=8080, min=1, max=65535, tooltip="the server's --port (or \"port\" in its config)"),
                io.Float.Input("timeout", default=600.0, min=1.0, max=86400.0, step=1.0,
                               tooltip="seconds to wait for one answer; a first request also loads the model"),
            ],
            outputs=[AudioCPPServerType.Output(display_name="server")],
        )

    @classmethod
    def execute(cls, port, timeout) -> io.NodeOutput:
        return io.NodeOutput(client.Server(port, float(timeout)))


class AudioCPPDoctor(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPDoctor",
            display_name="audio.cpp Doctor",
            category="audio/audio.cpp",
            description="Checks the server (GET /health) and lists its models (GET /v1/models): the ids to use and which of them speak.",
            inputs=[AudioCPPServerType.Input("server")],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, server):
        return float("nan")  # the server can change between runs

    @classmethod
    def execute(cls, server) -> io.NodeOutput:
        report = doctor_report(server)
        return io.NodeOutput(report, ui=ui.PreviewText(report))


def doctor_report(server: client.Server) -> str:
    try:
        status = client.health(server, _check_interrupt)
        models = client.models(server, _check_interrupt)
    except client.AudioCPPError as error:
        return f"audio.cpp server on {server.address}: not reachable.\n{error}"
    lines = [f"audio.cpp server on {server.address}: {status.get('status')}, backend {status.get('backend')}, {len(models)} model(s)"]
    for m in models:
        speaks = "speech" if m.get("task") in SPEECH_TASKS else "not for Text to Speech"
        lines.append(f"- {m.get('id')}: family {m.get('family')}, task {m.get('task')}, mode {m.get('mode')}, "
                     f"{'loaded' if m.get('loaded') else 'not loaded'}, device {m.get('device')} ({speaks})")
    if not _speech_models(models):
        lines.append(f"No model with a speech task ({', '.join(SPEECH_TASKS)}); add one to the server's config.")
    return "\n".join(lines)


class AudioCPPTextToSpeech(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPTextToSpeech",
            display_name="audio.cpp Text to Speech",
            category="audio/audio.cpp",
            description=("Speech from text through POST /v1/audio/speech. Which voices, languages and options work depends on the model; "
                         "see its page in the audio.cpp docs."),
            inputs=[
                AudioCPPServerType.Input("server"),
                io.String.Input("model", default="", tooltip="a model id from the server's config (audio.cpp Doctor lists them); empty: the only speech model"),
                io.String.Input("text", multiline=True, default="Hello from audio.cpp."),
                io.Int.Input("seed", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True),
                io.String.Input("voice", default="",
                                tooltip="a configured voice preset, a voice_dir reference name or the model's own voice id; empty: the default"),
                io.String.Input("language", default="", tooltip="the model's language code, e.g. en; empty: the model's default"),
                io.Float.Input("speed", default=1.0, min=0.25, max=4.0, step=0.01,
                               tooltip="1: unchanged; other values need a model that supports speed"),
                io.Int.Input("max_tokens", default=0, min=0, max=1000000, tooltip="0: the model's default; caps the length of autoregressive models"),
                io.String.Input("options", multiline=True, default="", tooltip='model request options as a JSON object, e.g. {"temperature": 0.7}'),
                io.Audio.Input("reference_audio", optional=True,
                               tooltip="a voice to clone, for models that take reference audio (sent as a WAV of up to 5 MiB)"),
                io.String.Input("reference_text", default="", optional=True, tooltip="the transcript of reference_audio, for models that need it"),
            ],
            outputs=[io.Audio.Output(display_name="audio")],
        )

    @classmethod
    def execute(cls, server, model, text, seed, voice, language, speed, max_tokens, options, reference_audio=None, reference_text="") -> io.NodeOutput:
        model = model.strip() or _only_speech_model(server)
        payload = speech_payload(model, text, seed, voice, language, speed, max_tokens, options, reference_audio, reference_text)
        try:
            wav, headers = client.speech(server, payload, _check_interrupt)
        except client.ServerError as error:
            if "unknown model id" in error.message:
                raise client.AudioCPPError(f"{error}. The server has: {_model_list(client.models(server))}") from error
            raise
        result = audio.wav_to_audio(wav)
        logging.info("audio.cpp %s: %.2f s of audio in %s ms (RTF %s)", model, result["waveform"].shape[-1] / result["sample_rate"],
                     headers.get("x-audiocpp-wall-ms"), headers.get("x-audiocpp-rtf"))
        return io.NodeOutput(result)


def _only_speech_model(server: client.Server) -> str:
    models = client.models(server, _check_interrupt)
    speaking = _speech_models(models)
    if len(speaking) != 1:
        raise ValueError(f"set the model input to one of the server's speech models: {_model_list(speaking or models)}")
    return speaking[0]["id"]


# The audio_encoders model folders, listing GGUF files (ComfyUI lists only its own weight formats there).
GGUF_FOLDER = "audiocpp_gguf"
folder_paths.folder_names_and_paths[GGUF_FOLDER] = (folder_paths.get_folder_paths("audio_encoders"), {".gguf"})


def _gguf_models() -> list[str]:
    """Decoder GGUFs in the audio_encoders model folders (the mmproj- encoder sibling is loaded by audio.cpp itself)."""
    return [f for f in folder_paths.get_filename_list(GGUF_FOLDER) if not os.path.basename(f).startswith("mmproj-")]


def _comfy_gpu() -> str | None:
    """The GPU-<uuid> of the CUDA device ComfyUI computes on, or None when it does not use one."""
    device = comfy.model_management.get_torch_device()
    if device.type != "cuda":
        return None
    return f"GPU-{torch.cuda.get_device_properties(device).uuid}"


class AudioCPPLocalServer(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPLocalServer",
            display_name="audio.cpp Local Server (Nemotron Transcribe)",
            category="audio/audio.cpp",
            description=("Starts an audiocpp_server on 127.0.0.1 for Nemotron 3.5 Transcribe, or keeps the one already running with these "
                         "settings. The program is the AUDIOCPP_SERVER environment variable or audiocpp_server on PATH; see the README. "
                         "On the GPU ComfyUI uses, ComfyUI's models are unloaded before each transcription."),
            inputs=[
                io.Combo.Input("model", options=_gguf_models(),
                               tooltip="the decoder GGUF in models/audio_encoders; its mmproj-...-Encoder-F16.gguf must sit beside it"),
                io.Int.Input("port", default=8090, min=1, max=65535),
                io.String.Input("gpu", default="", tooltip="empty: the GPU ComfyUI uses; or a CUDA index or GPU-<uuid> from nvidia-smi -L"),
                io.Combo.Input("memory", options=list(local_server.MEMORY), default="all on GPU",
                               tooltip="all on GPU: ~22 GB VRAM (Q4_K_S). Experts in RAM: ~6.7 GB VRAM and ~17 GB of host RAM, about 3x slower"),
                io.Int.Input("idle_unload_seconds", default=300, min=0, max=86400, tooltip="unload the model after this long unused; 0 keeps it loaded"),
                io.Float.Input("timeout", default=600.0, min=1.0, max=86400.0, step=1.0,
                               tooltip="seconds to wait for one answer; a first request also loads the model (about 15-20 s)"),
            ],
            outputs=[AudioCPPServerType.Output(display_name="server")],
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")  # the process can stop between runs

    @classmethod
    def execute(cls, model, port, gpu, memory, idle_unload_seconds, timeout) -> io.NodeOutput:
        comfy_gpu = _comfy_gpu()
        gpu = gpu.strip() or comfy_gpu
        if gpu is None:
            raise ValueError("ComfyUI does not run on a CUDA GPU here: set gpu to a CUDA index or GPU-<uuid> from nvidia-smi -L")
        model_path = folder_paths.get_full_path_or_raise(GGUF_FOLDER, model)
        local_server.start(model_path, port, gpu, memory, idle_unload_seconds * 1000, os.path.join(folder_paths.get_temp_directory(), "audiocpp"))
        return io.NodeOutput(client.Server(port, float(timeout), shares_comfy_gpu=gpu == comfy_gpu))


class AudioCPPReferenceSpeakers(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPReferenceSpeakers",
            display_name="audio.cpp Reference Speakers",
            category="audio/audio.cpp",
            description=("Named voice clips for Nemotron Transcribe: the clips are played to the model before the audio it checks, so its "
                         "speakers get these names. A few seconds of clean speech per voice is enough."),
            inputs=[
                io.String.Input("names", multiline=True, default="", tooltip="one name per line, in the order of the audio inputs"),
                io.Autogrow.Input("clips", template=io.Autogrow.TemplatePrefix(io.Audio.Input("audio"), prefix="audio", min=1, max=MAX_REFERENCES)),
            ],
            outputs=[SpeakersType.Output(display_name="speakers")],
        )

    @classmethod
    def execute(cls, names, clips: io.Autogrow.Type) -> io.NodeOutput:
        names = [n.strip() for n in names.splitlines() if n.strip()]
        clips = [clips[k] for k in sorted(clips, key=lambda k: int(k.removeprefix("audio"))) if clips[k] is not None]
        if len(names) != len(clips):
            raise ValueError(f"names has {len(names)} name(s) for {len(clips)} audio clip(s): give one name per line, in the order of the clips")
        if len(set(names)) != len(names):
            raise ValueError(f"the names must differ: {names}")
        return io.NodeOutput([(n, mono_16k(c)) for n, c in zip(names, clips)])


def mono_16k(clip: dict) -> np.ndarray:
    """The first clip of an AUDIO as 16 kHz mono float32, the rate Nemotron Transcribe runs at."""
    wave = clip["waveform"][0].to(torch.float32).mean(0)
    return comfy.audio.resample(wave, int(clip["sample_rate"]), speakers.RATE).cpu().numpy()


def _wav_16k(samples: np.ndarray) -> bytes:
    return audio.audio_to_wav({"waveform": torch.from_numpy(samples)[None, None], "sample_rate": speakers.RATE})


def transcribe(server: client.Server, model: str, clip: dict, references, language: str, options: str) -> dict:
    """Nemotron Transcribe with word times and speakers; with references, the speakers carry their names (see speakers.py)."""
    take = mono_16k(clip)
    combined, spans, start = speakers.combine(references or [], take)
    request_options = {"diarization": True, "return_timestamps": True, **_options(options)}
    if language.strip():
        request_options["language"] = language.strip()
    if server.shares_comfy_gpu:
        comfy.model_management.unload_all_models()
        comfy.model_management.soft_empty_cache()
    result = client.run_task(server, {"model": model, "request": {"audio_base64": base64.b64encode(_wav_16k(combined)).decode("ascii"),
                                                                  "options": request_options}}, _check_interrupt)
    named = speakers.attribute(result, spans, start)
    text, current = [], None
    for w in named["words"]:
        if w["speaker"] != current:
            current = w["speaker"]
            text.append(f"[{current}]")
        text.append(w["word"])
    return {"text": " ".join(text), "audio_seconds": round(len(take) / speakers.RATE, 3), **named,
            "reference_layout": {"references": spans, "audio_start": start, "gap_seconds": speakers.GAP_SECONDS}, "server_result": result}


class AudioCPPNemotronTranscribe(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPNemotronTranscribe",
            display_name="audio.cpp Nemotron Transcribe",
            category="audio/audio.cpp",
            description=("Words, word times and speakers of a recording with NVIDIA Nemotron 3.5 Transcribe (POST /v1/tasks/run). With "
                         "Reference Speakers, speakers get the references' names; a voice no reference matches stays unassigned and a "
                         "speaker the model heard in two references is reported as ambiguous, never guessed."),
            inputs=[
                AudioCPPServerType.Input("server"),
                io.String.Input("model", default="", tooltip="the server's model id; empty: its only nemotron_transcribe model"),
                io.Audio.Input("audio"),
                SpeakersType.Input("speakers", optional=True),
                io.String.Input("language", default="", tooltip="source-language hint such as en; empty: automatic"),
                io.String.Input("options", multiline=True, default="", tooltip='more request options as a JSON object, e.g. {"repetition_guard": true}'),
            ],
            outputs=[io.String.Output(display_name="text"), io.String.Output(display_name="details",
                                                                               tooltip="JSON: words, speaker_turns, speakers and the server's whole result")],
        )

    @classmethod
    def execute(cls, server, model, audio, language, options, speakers=None) -> io.NodeOutput:
        model = model.strip() or _only_model(server, "nemotron_transcribe")
        details = transcribe(server, model, audio, speakers, language, options)
        return io.NodeOutput(details["text"], json.dumps(details, ensure_ascii=False), ui=ui.PreviewText(details["text"]))


def _only_model(server: client.Server, family: str) -> str:
    models = client.models(server, _check_interrupt)
    found = [m for m in models if m.get("family") == family]
    if len(found) != 1:
        raise ValueError(f"set the model input to one of the server's {family} models: {_model_list(found or models)}")
    return found[0]["id"]


class AudioCPPH3VoiceQA(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPH3VoiceQA",
            display_name="audio.cpp H3 Voice QA",
            category="audio/audio.cpp",
            description=("Checks a MiniMax H3 take against its prompt: for each <d>...</d> line, who said it and whether it started in its "
                         "shot ([Shot N] At MM:SS.mmm). Also lists speech the prompt does not script, such as a voice reference copied into "
                         "a silent shot. Voices need Nemotron Transcribe run with Reference Speakers; without them only the words and times are checked."),
            inputs=[
                io.String.Input("details", force_input=True),
                io.String.Input("prompt", multiline=True, default="", tooltip="the H3 prompt the take was rendered from"),
                io.String.Input("cast", multiline=True, default="",
                                tooltip="speaker ids or audio labels to reference names, one per line, e.g. S1: Maren or Audio 2: Liesl; "
                                        "not needed when the prompt names the speaker next to each line"),
                VoiceModelType.Input("voice_model", optional=True,
                                     tooltip="scores each line's own span against every reference (needs audio and speakers): tells "
                                             "apart similar voices that the diarizer hears as one"),
                io.Audio.Input("audio", optional=True, tooltip="the take that was transcribed, for voice_model"),
                SpeakersType.Input("speakers", optional=True, tooltip="the same Reference Speakers, for voice_model (two or more)"),
            ],
            outputs=[io.String.Output(display_name="summary", tooltip="one line per scripted line and per unscripted stretch, for a person"),
                     io.String.Output(display_name="report", tooltip="JSON: counts, every line with its verdict and words, unscripted speech")],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, details, prompt, cast, voice_model=None, audio=None, speakers=None) -> io.NodeOutput:
        d = json.loads(details)
        names = [s["name"] for s in d["speakers"]]
        voice = None
        if voice_model is not None:
            if audio is None or speakers is None or len(speakers) < 2:
                raise ValueError("voice_model compares each line with every reference: connect the take's audio and the Reference "
                                 "Speakers (at least two voices)")
            voice = voice_scorer(voice_model, mono_16k(audio), speakers)
        report = h3qa.check(prompt, d["words"], d["audio_seconds"], names, h3qa.parse_cast(cast), voice)
        summary = qa_text(report)
        return io.NodeOutput(summary, json.dumps(report, ensure_ascii=False), ui=ui.PreviewText(summary))


def voice_scorer(voice_model, take: np.ndarray, references):
    """(start, end) seconds -> {reference name: cosine similarity} of that span of the take, with 0.1 s on each side."""
    comfy.model_management.load_model_gpu(voice_model)
    net, device = voice_model.model, voice_model.load_device

    def embed(samples: np.ndarray) -> torch.Tensor:
        return torch.nn.functional.normalize(net(torch.from_numpy(samples).to(device)), dim=-1)

    refs = {name: embed(clip) for name, clip in references}

    def score(start: float, end: float) -> dict:
        lo, hi = max(0, int((start - 0.1) * speakers.RATE)), int((end + 0.1) * speakers.RATE)
        e = embed(take[lo:hi])
        return {name: float(e @ r) for name, r in refs.items()}

    return score


class AudioCPPVoiceModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPVoiceModelLoader",
            display_name="audio.cpp Voice Model (WavLM-large ECAPA)",
            category="audio/audio.cpp",
            description=("Loads UniSpeech's WavLM-large + ECAPA-TDNN speaker-verification weights (one safetensors file in "
                         "models/audio_encoders) for H3 Voice QA's voice_model."),
            inputs=[io.Combo.Input("model_name", options=folder_paths.get_filename_list("audio_encoders"))],
            outputs=[VoiceModelType.Output()],
        )

    @classmethod
    def execute(cls, model_name) -> io.NodeOutput:
        sd = comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("audio_encoders", model_name), safe_load=True)
        if not speaker_embedding.is_wavlm_large_ecapa(sd):
            raise ValueError(f"{model_name} is not the WavLM-large + ECAPA-TDNN speaker model (no ecapa.* weights); see the README")
        net = speaker_embedding.WavLMLargeECAPA()
        net.load_state_dict(sd)
        return io.NodeOutput(comfy.model_patcher.ModelPatcher(net.eval(), load_device=comfy.model_management.get_torch_device(),
                                                              offload_device=comfy.model_management.unet_offload_device()))


def qa_text(report: dict) -> str:
    """The report as lines for a person: the counts, then each scripted line and each unscripted stretch."""
    s = report["summary"]
    lines = [", ".join(f"{k} {v}" for k, v in s.items())]
    for r in report["lines"]:
        when = f"{r['start']:.2f}-{r['end']:.2f} s" if "start" in r else "not heard"
        window = r["shot_window"]
        timing = f", wrong time ({r['drift_seconds']:+.2f} s outside shot {r['shot']}: {window[0]:.2f}-{window[1]:.2f} s)" if r.get("wrong_time") else ""
        heard = f"{r['voice_match']} by voice (margin {r['voice_margin']:.2f})" if "voice_match" in r else r.get("detected_speaker")
        lines.append(f"{r['verdict'].upper()}: shot {r['shot']} \"{r['text']}\" expected {r['expected_speaker']}, "
                     f"heard {heard} at {when}{timing}")
    for u in report["unscripted"]:
        who = f"{u['voice_match']} by voice (margin {u['voice_margin']:.2f})" if "voice_match" in u else u["speaker"]
        lines.append(f"UNSCRIPTED: \"{u['text']}\" by {who} at {u['start']:.2f}-{u['end']:.2f} s")
    return "\n".join(lines)


class AudioCPPUnload(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="AudioCPPUnload",
            display_name="audio.cpp Unload",
            category="audio/audio.cpp",
            description=("Frees the server's model memory (POST /v1/tasks/unload_models or unload_all_models); the next request loads it "
                         "again. Optionally stops a server that the Local Server node started."),
            inputs=[
                AudioCPPServerType.Input("server"),
                io.String.Input("model", default="", tooltip="a model id; empty: every loaded model"),
                io.Boolean.Input("stop_server", default=False, tooltip="also stop the audiocpp_server this ComfyUI started on that port"),
            ],
            outputs=[io.String.Output(display_name="report")],
            is_output_node=True,
        )

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float("nan")

    @classmethod
    def execute(cls, server, model, stop_server) -> io.NodeOutput:
        unloaded = client.unload(server, [model.strip()] if model.strip() else None, _check_interrupt)
        report = f"unloaded: {json.dumps(unloaded)}"
        if stop_server:
            report += "; server stopped" if local_server.stop(server.port) else f"; no server started by this ComfyUI on port {server.port}"
        return io.NodeOutput(report, ui=ui.PreviewText(report))


class AudioCPPExtension(ComfyExtension):
    async def get_node_list(self):
        return [AudioCPPServer, AudioCPPDoctor, AudioCPPTextToSpeech, AudioCPPLocalServer, AudioCPPReferenceSpeakers,
                AudioCPPNemotronTranscribe, AudioCPPH3VoiceQA, AudioCPPVoiceModelLoader, AudioCPPUnload]


async def comfy_entrypoint() -> AudioCPPExtension:
    return AudioCPPExtension()
