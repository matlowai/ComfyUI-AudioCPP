# Nemotron diarization and H3 voice QA: the study behind the nodes

Date: 2026-11-07. This is the write-up of why the `nemotron-diarization` branch exists, what we measured, what failed and how
to repeat it. The how-to-run guide is in the [README](../README.md#nemotron-transcribe-and-h3-voice-qa); this page is the evidence.

## Why the nodes exist

The operator wanted a ComfyUI path for NVIDIA Nemotron 3.5 Transcribe 30B-A4B (transcript, diarization and word times) through
audio.cpp, "that can also take the reference audio clips for determining what's up", and asked that "whatever we do make it's
easy for other folks to follow after us". The first use is quality control of generated video with speech (MiniMax H3 / Pika H3
style prompts, where each character's voice comes from a reference clip): is a line spoken at all, at the right time, by the right
voice, and is there speech nobody scripted? Whisper had shown text but its word timestamps pin the first word to 0.00 s, and it
has no idea who is speaking.

Everything below was run on one RTX 5090 with audio.cpp `d2b2076e` (Release, CUDA 13.2) and the Q4_K_S decoder with the F16
encoder (GGUF repository revision `d2affa84`). The nodes: Local Server, Reference Speakers, Nemotron Transcribe, Voice Model,
H3 Voice QA, Unload.

## Memory and speed

Per-process VRAM from nvidia-smi, host RAM is the server's peak RSS. Requests after a warm-up in the same session.

| mode | peak VRAM | peak RSS | load + warm-up | 3.07 s take | 22.1 s scene | 161.9 s file | after unload |
|---|---:|---:|---:|---:|---:|---:|---|
| all on GPU | 22,268 MiB | 1.2 GB | 19.0 s | 0.17 s | 0.46 s | 2.04 s (79x real time) | 608 MiB / 0.67 GB |
| experts pinned in RAM | 6,658 MiB | 18.2 GB | 12.2 s | 0.83 s | 1.80 s | 5.87 s (27.6x) | 624 MiB / 0.90 GB |
| experts in RAM, unpinned | 6,658 MiB | 17.6 GB | 6.2 s (warm page cache) | 1.29 s | 2.64 s | 8.27 s (19.6x) | 624 MiB / 0.85 GB |

Idle unload (`idle_unload_ms` 10000) dropped 22,190 to 572 MiB about 10 s after the last request. Transcripts were identical
across modes except one trailing filler word. The roughly 0.6 GB left after unloading is the CUDA context, which goes when the
process stops.

## Question 1: can Nemotron's diarizer say who is speaking, given reference clips?

Nemotron has no speaker enrollment. Its diarizer (Sortformer-style, up to 8 speakers) numbers speakers in order of first
appearance, so the node plays the reference clips first (1 s of silence after each), then the audio to check, and names each
speaker by the reference it appeared in.

Test set: 12 real generated takes of five original H3 voices (Maren, Liesl, Hiyori are young women; Brann and Kuroe are men), each
with its designed voice reference. A wrong-voice test is the same take claimed for another cast member: 48 such swaps (20
same-sex, 28 cross-sex).

| method | false alarms | swaps caught | same-sex swaps | cross-sex swaps | who is it |
|---|---|---|---|---|---|
| Nemotron, no prefix | n/a | 0 | 0 | 0 | none (one anonymous speaker per take) |
| Nemotron + claimed ref only | 0/12 | 31/48 | 3/20 | 28/28 | n/a |
| Nemotron + 3-voice cast, 6 orders | 0/48 | 60/96 | 0/36 | 60/60 | 12/48 unique, 36/48 ambiguous |
| Nemotron + all 5 refs, 2 orders | 0/24 | 56/96 | 0/40 | 56/56 | 0/24 unique, all ambiguous within the right sex |
| `nemotron_3_diar` + claimed ref | 0/12 | 30/48 | 2/20 | 28/28 | n/a |
| `sortformer_diar` + claimed ref | 0/12 | 28/48 | 0/20 | 28/28 | n/a |
| `nemotron_3_diar` / `sortformer_diar` + 3-voice cast, 6 orders | 0/48 | 60/96 | 0/36 | 60/60 | 12/48 unique (Brann only) |
| `nemotron_3_diar` + all 5 | 0/24 | 56/96 | 0/40 | 56/56 | 0/24 unique |
| `sortformer_diar` + all 5 | 0/24 | 64/96 | 8/40 | 56/56 | 6/24, order-dependent |

Result: cross-sex wrong voices are caught every time, but voices of the same sex and age merge into one speaker (Maren, Liesl and
Hiyori always one column; Brann and Kuroe one column). Without a reference prefix both dedicated diarizers found exactly one
speaker per take.

What was tried to split them, and failed (16 hard same-sex cases each): reference repeated x1/x2/x3, gaps of 1, 2 and 4 s,
speaker cache 1000, FIFO 0, speaker update 1000, encoder window 1000: 0 or 1 out of 16 every time. After two or three variants
giving the same merge the approach was dropped; the merge is the model family's, not the layout. audio.cpp's two dedicated
diarizers (`nemotron_3_diar`, `sortformer_diar`) show the same merge, so it is a Sortformer-family trait on these voices, not a
Transcribe quirk.

## Question 2: what does Nemotron do well for H3 QA?

Scene-level checks on two edited short films (6 and 4 shots), with an H3-format prompt (subject definitions, `(S1..S3)`
labels, `<Audio N>` references) and the cast references as prefix:

| case | truth | QA output | caught |
|---|---|---|---|
| as rendered, shot 1 silent | the silent shot carries a character's reference line | unscripted speech found at 0.18-4.47 s | yes |
| cross-sex relabel | wrong voice | `wrong_voice` | yes |
| same-sex relabel (diarizer only) | wrong voice | `unsure_voice` | no |
| line moved +3.5 s | wrong time | `wrong_time`, 2.46 s past its shot | yes |
| line moved -2.5 s | wrong time | `wrong_time`, 0.87 s before its shot | yes |
| shot audio moved onto the next shot | late line, one line lost | `wrong_time` +0.27 s and `missing` | yes |
| one cast reference removed | speaker unknown | `no_reference` | yes |
| second film as rendered | silent shot repeats the reference line | unscripted copy found | yes |
| second film, cross-sex relabel | wrong voice | `wrong_voice` | yes |
| second film, same-sex relabel, single reference | wrong voice | `ok` | no (false pass) |

It found two real problems in already-shipped edits: voice-reference leaks, where H3 copied a reference line into a shot that
was meant to be silent. Whisper had shown both as text but at broken times.

Word timestamps against Whisper large-v3 (no ground truth exists): Whisper put the first word at 0.00 s in 14 of 14 files.
Excluding first words (60 matched words), the median start difference is 0.21 s (p90 0.54 s) and Nemotron starts about 0.20 s
later. Against a crude energy onset, Nemotron's first word is off by a median 0.06 s and Whisper's by 0.44 s. The shot-window
checks use 0.25 s of slack.

## Question 3: what settles same-sex voices? A speaker-embedding model

Plain speaker-embedding scoring (cosine against each reference) was tried with two models on the same 12 takes:

| method | false alarms | swaps caught | same-sex swaps | cross-sex swaps | who is it |
|---|---|---|---|---|---|
| WavLM-base-plus-sv, single ref, cos < 0.80 | 0/12 | 34/48 | 6/20 | 28/28 | n/a |
| WavLM-base-plus-sv, argmax over all 5 | 2/12 | 46/48 | 18/20 | 28/28 | 10/12 |
| WavLM-base-plus-sv, margin rule 0.03 | 1 wrong + 3 unsure / 12 | 35/48 (+12 unsure, 1 missed) | 14/20 (+5 unsure) | 21/28 (+7 unsure) | 10/12 |
| **WavLM-large + ECAPA-TDNN, argmax over all 5, margin rule, whole take** | **0/12** | **48/48** | **20/20** | **28/28** | **12/12** (margins 0.15-0.65) |
| **same, on the Nemotron line span** | **0/12** | **48/48** | **20/20** | **28/28** | **12/12** (margins 0.12-0.54) |

Cosine scales differ by model: do not reuse a margin across models. WavLM-large + ECAPA gave 0.46-0.96 (median 0.75) for the
own reference and 0.45 for the best same-sex other reference. Between references: Liesl-Hiyori 0.47 (base model 0.95), Maren-Liesl
0.50 (0.87), Brann-Kuroe 0.31 (0.77). The base model's same-sex cosines are so close to the same-speaker ones that no cutoff
works. The model used is the UniSpeech WavLM-large + ECAPA-TDNN speaker-verification checkpoint (safetensors,
[prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native](https://huggingface.co/prj-beatrice/unispeech-wavlm-large-ecapa-tdnn-torch-native),
CC-BY-SA-3.0); the pack has its own implementation of the published architecture and loads only the weights (strict load: 0
missing, 0 unexpected keys). The repository's own modeling file is never downloaded or run.

**Shipped design:** Nemotron Transcribe supplies words, word times, each line's span, unscripted speech and timing; the voice
model embeds each line's span (plus or minus 0.1 s) and compares it with every cast reference. A line is `unsure_voice` when the
best reference beats the second-best speaker by less than 0.08. The diarizer's own names stay in the report as a cross-check.

## Question 4: is the 0.08 "unsure" margin safe on a wider set?

The first set was small (5 voices, 12 takes), so a wider labelled set was built from all voiced takes of five short films
(250 takes transcribed): 40 real labelled lines across 8 voices, of which 5 voices (Maren, Liesl, Brann, Hiyori, Kuroe) have
designed reference clips ("strong" labels) and 3 (Nagi, Ayame, Rook) do not ("weak" labels: their voices came from pictures and a
prompt line, so the label itself is uncertain). Derived clips from the real lines: 74 sub-second crops, 80 clips mixed over music
(speech 6 dB and 0 dB under/over it), and 160 real ffmpeg splices (a line's span replaced by another speaker's line). In all, 354
clips, 194 correct-line events and 1712 swap events (every clip relabelled as each other speaker, plus the splices). The cast had
9 reference speakers, so the unreferenced-test voices act as distractors.

Strong set (252 clips, 156 correct, 1344 swaps), WavLM-large + ECAPA is right on every clip: real lines 32/32, crops 60/60, music
64/64, splices 96/96.

| cutoff | correct lines flagged unsure | wrong_voice on a correct line | swaps caught | swaps unsure | swaps missed |
|---|---|---|---|---|---|
| 0.04 | 0/156 | 0 | 1344 | 0 | 0 |
| 0.06 | 3/156 | 0 | 1320 | 24 | 0 |
| **0.08** | **5/156 (3.2%)** | 0 | 1304 | 40 (3.0%) | **0** |
| 0.10 | 9/156 | 0 | 1272 | 72 | 0 |
| 0.12 | 11/156 | 0 | 1256 | 88 | 0 |

Every unsure line at 0.08 is a sub-second crop. All 32 real lines, all 64 music clips and all 96 splices pass with 0 unsure at
0.08 and at 0.12. Smallest correct margins: real lines 0.121 (Maren, a 0.6 s line, against Liesl), crops 0.044, music 0.091,
splices 0.139. Lines under 1 s: 5/66 correct flagged and 40/535 swaps unsure; lines over 1 s: 0/90 and 0/809. There is no knee in
the curve: a cutoff only moves lines from "caught" to "unsure"; hard misses are zero from 0.04 up. The base model on the same
clips: top-1 on real lines 29/32, margins tiny (real minimum 0.007, median 0.073), so no cutoff is useful.

With the weak labels in, top-1 is 324/354 and every failure is Ayame, Nagi or Rook:

| cutoff | correct ok / unsure / wrong_voice | swaps caught / unsure / missed |
|---|---|---|
| 0.04 | 177 / 8 / 9 | 1625 / 75 / 12 |
| 0.06 | 171 / 16 / 7 | 1556 / 146 / 10 |
| 0.08 | 168 / 21 / 5 | 1505 / 201 / 6 |
| 0.10 | 158 / 35 / 1 | 1382 / 328 / 2 |
| 0.12 | 154 / 39 / 1 | 1343 / 368 / 1 |

Where it fails: Ayame's two soft takes are heard as Nagi (margins 0.122 and 0.086), but her only reference is a firm command
from a sibling take; Nagi's own takes vary in pitch (F0 132-261 Hz across five takes of one voice) and one is heard as Liesl by
0.028; Rook's 0.76 s shout (F0 415 Hz) is scored against a 114 Hz reference. With no designed reference we cannot tell scorer
error from voice drift, so this is a label problem, not evidence against 0.08.

**Recommendation: keep 0.08, and never raise it above 0.12** (the smallest correct real margin). Expect "unsure" about once in
twelve for lines under a second. The evidence is limited to five strongly labelled voices; more voices need more films with a
designed voice reference per speaking character.

## What breaks

- Same-sex voice separation by any Sortformer-family diarizer on these voices (above).
- Transcript style with diarization is lower case without punctuation, and names vary ("tamoshi"/"tomoshi").
- `/v1/audio/transcriptions/details` returns `speaker_turns` confidence of 0 always for `nemotron_transcribe`; the real per-word
  speaker confidence comes only from `/v1/tasks/run` (it accepts `audio_base64`) in `custom_schema_output`.
- `sortformer_diar`'s default 20 s graph refuses longer audio ("request exceeds fixed graph capacity"); raise
  `sortformer_diar.session_len_sec` above the clip length.
- ComfyUI's `audio_encoders` listing ignores `.gguf`; the pack registers a second folder key over the same paths.
- The ComfyUI frontend of the time rendered `ui` text only for Preview Any; the example uses Preview as Text nodes.
- A subprocess started from a node outlives ComfyUI when ComfyUI is killed with SIGTERM (atexit is skipped) and keeps its port
  and VRAM; a later start then talks to the old server. Fix in the pack: refuse a port that already answers, and set
  `PR_SET_PDEATHSIG` on Linux.
- Line matching must prefer the occurrence inside the line's shot (H3 sometimes copies a reference into a silent shot), and very
  short lines such as "It's you." need a match ratio of at least 0.75 with window lengths n-1..n+2.

## Reproduce

1. Build audio.cpp at the pinned commit with `scripts/build_audiocpp.sh` (about 4-5 minutes with only the Nemotron family; add
   `nemotron_3_diar` and `sortformer_diar` to reproduce the diarizer rows).
2. Download the decoder with `scripts/download_nemotron_transcribe.sh` and the voice model with
   `scripts/download_voice_model.sh`. GGUF and safetensors only.
3. Start ComfyUI with `AUDIOCPP_SERVER` set and open `example_workflows/nemotron_h3_voice_qa.json`.
4. To redo the tables: take a few generated voiced clips with a designed reference each; build the claimed-speaker swaps by
   relabelling each clip as every other cast member; compare verdicts. For the splice and music rows use ffmpeg to replace a
   line's span with another speaker's line and to mix a music bed at 6 dB and 0 dB.
5. Check VRAM with `nvidia-smi --query-compute-apps` while the server is up, before and after Unload.

The labelled clips are generated audio and are not included here.
