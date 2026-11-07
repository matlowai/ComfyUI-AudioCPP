# ComfyUI-AudioCPP

ComfyUI nodes for [audio.cpp](https://github.com/0xShug0/audio.cpp): they send text to an `audiocpp_server` running on the same
computer and return the speech as ComfyUI `AUDIO`, ready for Save Audio, Preview Audio or any other audio node.

This fork adds [NVIDIA Nemotron 3.5 Transcribe 30B-A4B](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B):
words, word times and speakers, with **named reference clips** deciding who is speaking, and a check of
[MiniMax H3](https://github.com/MiniMax-AI/MiniMax-H3) takes against their prompts (who says each line, and when). See
[Nemotron Transcribe and H3 Voice QA](#nemotron-transcribe-and-h3-voice-qa) for the step-by-step setup.

The nodes only talk to the server over HTTP on `127.0.0.1` and need no Python packages beyond ComfyUI's. Only **audio.cpp Local
Server** starts anything: the `audiocpp_server` program named by the `AUDIOCPP_SERVER` environment variable (never a path from a
workflow). Nothing downloads models; the scripts in `scripts/` do that when you run them.

## Nodes

| Node | What it does |
| --- | --- |
| **audio.cpp Server** | The connection: the server's port on this computer and how long to wait for one answer. |
| **audio.cpp Doctor** | Checks `GET /health` and lists `GET /v1/models`: each model's id, family, task, mode and whether it is loaded. Use it to find the model ids. |
| **audio.cpp Text to Speech** | `POST /v1/audio/speech`: model id, text, seed, voice, language, speed, `max_tokens` and model options, plus optional reference audio and its transcript for voice cloning. Returns `AUDIO`. |
| **audio.cpp Local Server (Nemotron Transcribe)** | Starts `audiocpp_server` for Nemotron Transcribe on 127.0.0.1, on a GPU chosen by index or UUID (default: ComfyUI's), all on GPU or with the experts in RAM, with idle unload. Keeps the running one when the settings are unchanged. |
| **audio.cpp Reference Speakers** | Named voice clips (one name per line, one `AUDIO` per name, up to 7). |
| **audio.cpp Nemotron Transcribe** | `POST /v1/tasks/run` with diarization and word timestamps. Returns `text` (`[Name] words ...`) and `details` (JSON: every word with start, end and speaker, the speaker turns, which reference claimed which speaker, and the server's whole result). |
| **audio.cpp H3 Voice QA** | Reads an H3 prompt (`[Shot N] At MM:SS.mmm`, `(S1)`, `<d>[English] ...</d>`, `<Audio N>`), finds each line in the words and reports its speaker and time against its shot. Returns a `summary` for people and a `report` (JSON). |
| **audio.cpp Unload** | `POST /v1/tasks/unload_models` or `unload_all_models`; optionally stops the server the Local Server node started. |

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/0xShug0/ComfyUI-AudioCPP
```

For Nemotron Transcribe and H3 Voice QA, clone this fork's branch instead:

```bash
git clone -b nemotron-diarization https://github.com/matlowai/ComfyUI-AudioCPP
```

There is nothing to install with pip. Restart ComfyUI; the nodes are under `audio/audio.cpp`, and an example workflow is in
`example_workflows/`.

## Start a server

Use a [release](https://github.com/0xShug0/audio.cpp/releases) build or [build](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md#build)
`audiocpp_server`, install a model package (see [docs/tts.md](https://github.com/0xShug0/audio.cpp/blob/main/docs/tts.md) and
[docs/models](https://github.com/0xShug0/audio.cpp/tree/main/docs/models)) and register it in a config. For example, with
[PocketTTS](https://github.com/0xShug0/audio.cpp/blob/main/docs/tts.md#pockettts) and its built-in voice `alba` (as in
[examples/server/pocket-tts.json](https://github.com/0xShug0/audio.cpp/blob/main/examples/server/pocket-tts.json)):

```json
{
  "host": "127.0.0.1",
  "port": 8080,
  "backend": "cuda",
  "lazy_load": true,
  "models": [
    {
      "id": "pocket-tts",
      "family": "pocket_tts",
      "path": "C:/models/PocketTTS-GGUF/english/pocket-tts-english-q8_0.gguf",
      "task": "tts",
      "mode": "offline",
      "load_options": { "language": "english" },
      "default_voice_preset": { "voice_id": "alba" }
    }
  ]
}
```

```bash
audiocpp_server --config server.json --no-ui --log
```

Set **audio.cpp Server**'s port to the server's, and **Text to Speech**'s `model` to the config's `id` (or leave it empty when the
server has only one speech model). The [server guide](https://github.com/0xShug0/audio.cpp/blob/main/app/server/README.md) explains
configs, voice presets, `voice_dir` reference libraries and memory settings.

## Voices, languages and options

What works depends on the model; check its section in the audio.cpp docs.

- `voice` is a configured preset name, a `voice_dir` reference name or the model's own voice id.
- `language` is the model's language code or name (for example `en`).
- `options` takes the model's request options as a JSON object, for example `{"temperature": 0.7}` or
  `{"x_vector_only_mode": true}` for Qwen3-TTS Base.
- `seed` is sent with every request, so the same inputs give the same audio with the models that sample from a seed.
- **Voice cloning**: connect `reference_audio` (for example from Load Audio) and fill `reference_text` with its transcript for
  models that need one, such as Qwen3-TTS Base. The audio goes to the server as an inline 16-bit WAV, which the server limits to 5 MiB.

## Notes

- Only `127.0.0.1` is used: a workflow chooses a port, never a host or URL. Redirects are not followed and proxy settings are not
  used.
- **Cancel** stops ComfyUI from waiting, but the server finishes the generation it has started (its default runtime has no cancel
  request), and the next request to that model waits until it is done.
- `timeout` covers one answer, including model loading on the first request; raise it for long texts and slow models.
- Errors from the server are shown as it sends them, for example `unknown model id: ...` (together with the ids it has) or
  `PocketTTS text prompt is required`.

## Nemotron Transcribe and H3 Voice QA

Tested on Linux with an RTX 5090, CUDA 13.2, GCC 13 and CMake 3.28, audio.cpp
[`d2b2076e`](https://github.com/0xShug0/audio.cpp/commit/d2b2076e47affef4973a9f69433750b339be7d8d) and the GGUF repository at
revision [`d2affa84`](https://huggingface.co/Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF/tree/d2affa8419718ffe0917cb69dc6bc693084497f5).

### 1. Build audio.cpp

Needs git, CMake (3.24 or newer for `CUDA_ARCH=native`), GCC 13 or newer and the CUDA Toolkit 12 or 13. From this folder:

```bash
scripts/build_audiocpp.sh ~/audio.cpp
```

It clones audio.cpp at the pinned commit and builds `audiocpp_server` and `audiocpp_cli` into `~/audio.cpp/build/release/bin/`
with only the `nemotron_transcribe` family (about 5 minutes on 12 cores). Set `CUDA_ARCH=120` to build for RTX 50xx without
the GPU present (audio.cpp turns it into `120a`), `AUDIOCPP_MODELS=full` for every model family (needed for Text to Speech
models), `CUDA_HOME` for a toolkit outside `/usr/local/cuda`. The script prints the commit it built.

### 2. Download the model

Needs the `hf` command (`pip install -U huggingface_hub`). Into ComfyUI's `models/audio_encoders` folder:

```bash
scripts/download_nemotron_transcribe.sh ComfyUI/models/audio_encoders/nemotron            # Q4_K_S, ~23.3 GB in all
scripts/download_nemotron_transcribe.sh ComfyUI/models/audio_encoders/nemotron Q2_K       # or IQ4_NL, Q3_K_S, Q2_K
```

This downloads one decoder GGUF and the shared `mmproj-...-Encoder-F16.gguf` (1.7 GB, includes the CTC timestamp head) beside it,
plus the licence files. GGUF files only. They are the community conversion by
[Jmica](https://huggingface.co/Jmica/NVIDIA-Nemotron-3.5-Transcribe-30B-A4B-GGUF) for audio.cpp; see audio.cpp's
[model page](https://github.com/0xShug0/audio.cpp/blob/main/docs/models/nemotron_transcribe.md).

### 3. Tell ComfyUI where the server is, and start it

```bash
export AUDIOCPP_SERVER=~/audio.cpp/build/release/bin/audiocpp_server
python main.py
```

The **Local Server** node lists the decoder GGUFs under `models/audio_encoders` (not the `mmproj-` file). Without
`AUDIOCPP_SERVER`, it looks for `audiocpp_server` on `PATH`. The server it starts stops when ComfyUI exits (on Linux also when
ComfyUI is killed), or with **Unload** and `stop_server`. It refuses a port where another server already answers; connect to
that one with **audio.cpp Server** instead. Its config and log are in ComfyUI's `temp/audiocpp/`.

### 4. Run the example

Open `example_workflows/nemotron_h3_voice_qa.json`: three **Load Audio** nodes with each character's voice clip, one with the
take, **Reference Speakers** (`Maren`, `Brann`, `Liesl`), **Local Server**, **Nemotron Transcribe**, **H3 Voice QA** with the
take's prompt and `S1: Brann` / `S2: Maren` / `S3: Liesl`, and two **Preview as Text** nodes. Put your own clips in
`ComfyUI/input` and pick them. The first run loads the model (15-20 s); later runs take well under a second for a short take.

To use a server you run yourself instead, connect **audio.cpp Server** (the port) to Nemotron Transcribe. A config for it:

```json
{
  "host": "127.0.0.1", "port": 8090, "backend": "cuda", "lazy_load": true, "max_loaded_models": 1, "idle_unload_ms": 300000,
  "models": [{"id": "nemotron-transcribe", "family": "nemotron_transcribe", "task": "asr", "mode": "offline",
              "path": "/absolute/path/to/nemotron-transcribe-30b-a4b-experts-Q4_K_S.gguf", "session_options": {}}]
}
```

Start it with `CUDA_VISIBLE_DEVICES=<GPU-uuid from nvidia-smi -L> audiocpp_server --config server.json --no-ui`. A server you start
yourself does not unload ComfyUI's models first; the Local Server node does that when it runs on ComfyUI's GPU.

### Memory and speed

Q4_K_S, RTX 5090, model already loaded, `audiocpp_server` peak per nvidia-smi:

| memory | peak VRAM | host RAM (server) | 3 s take | 22 s scene | 162 s |
| --- | ---: | ---: | ---: | ---: | ---: |
| all on GPU | 22.3 GB | 1.2 GB | 0.17 s | 0.46 s | 2.0 s (80x real time) |
| experts in pinned RAM | 6.7 GB | 18.2 GB | 0.83 s | 1.8 s | 5.9 s (27x) |
| experts in RAM | 6.7 GB | 17.6 GB | 1.3 s | 2.6 s | 8.3 s (19x) |

Loading takes 15-20 s from a warm disk cache. **Unload** (or idle unload) frees the weights: VRAM drops to about 0.6 GB, the
CUDA context the server keeps until it stops (Unload with `stop_server`). Transcripts were the same in all three modes on our
clips except one trailing filler word.

### How names are given to speakers

Nemotron has no speaker enrollment. Its diarizer numbers up to 8 speakers in order of first appearance and keeps them across one
recording, so the node plays the reference clips first (1 s of silence after each), then the audio to check, and names each
speaker by the reference it was heard in. A speaker heard in two references is **ambiguous** (`ambiguous: Maren | Liesl`), a
voice no reference matches is **unassigned** (`unassigned (speaker_2)`); names are never guessed. Times in `details` are on the
checked audio's own clock.

**Know the limit.** On our H3 test set (five generated anime voices), the diarizer separated men from women every time but put
voices of the same sex and age in one speaker: Maren, Liesl and Hiyori (three young women) always came out as one speaker, as did
two old men. Longer or repeated reference clips, wider gaps and larger speaker-cache settings did not change it. So a wrong voice
of the other sex is caught, while a swap between two similar voices comes back `ambiguous` (H3 Voice QA: `unsure_voice`), not
`ok`. Pair it with a speaker-embedding check on the line's time span when similar voices matter.

### H3 Voice QA verdicts

Each `<d>` line gets its speaker from the nearest `(Sx)`, `<Audio N>` or reference name before it in its shot; `cast` maps
`S1` or `Audio 1` to reference names when the prompt does not name them (`<Audio N> is the voice-timbre reference for ... (Sx)`
lines are followed). Shot times are cut times, so a line passes timing when it starts inside its shot (0.25 s slack).

| verdict | meaning |
| --- | --- |
| `ok` | the line's words were heard, in the expected reference's voice |
| `wrong_voice` | heard in another reference's voice, or a voice no reference matches |
| `unsure_voice` | the diarizer heard the expected voice and another reference as one speaker |
| `no_reference` | no reference clip for the line's speaker |
| `missing` | the line was not heard |
| `wrong_time` (flag) | the line starts outside its shot; `drift_seconds` says by how much |
| unscripted | speech the prompt does not script, e.g. a voice reference copied into a silent shot |

## Tests

```bash
AUDIOCPP_COMFYUI_DIR=/path/to/ComfyUI python -m pytest
```

The tests use a stand-in HTTP server with the server's response shapes. With `AUDIOCPP_TEST_PORT` set to the port of a running
`audiocpp_server` that has a PocketTTS entry `pocket-tts`, they also run against it. `tests/test_speakers.py` and
`tests/test_h3qa.py` cover the naming and the prompt check without a server.

## License

Apache-2.0, like audio.cpp. The models keep their own licenses; see
[docs/model_licenses.md](https://github.com/0xShug0/audio.cpp/blob/main/docs/model_licenses.md). The Nemotron 3.5 Transcribe
weights are under [OpenMDW-1.1](https://openmdw.ai/license/1-1/) with the notices in the model repository
(`THIRD_PARTY_NOTICES.md`, `licenses/`), which the download script fetches beside the GGUFs.
