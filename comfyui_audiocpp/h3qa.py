"""Check a MiniMax H3 take's speech against its prompt: who says each scripted line, and when.

The prompt format is MiniMax-H3's (.claude/skills/h3-prompt-writing/references in github.com/MiniMax-AI/MiniMax-H3): `[Shot 1]` opens the video, later shots
start with `[Shot N] At MM:SS.mmm` (a cut time, not a speech start), speakers are `(S1)`, `(S2)`, ..., dialogue is
`<d>[Language] ...</d>`, and `<Audio N> is the voice-timbre reference for <Subject M> (Sx)` ties a voice reference to a
speaker. Prompts that name the character next to the line ("Brann, with the voice of <Audio 1>, says: <d>...") work too.

Each line's speaker comes from the last speaker mark of each kind before its `<d>` in the same shot, the first that resolves
to a reference name: `(Sx)`, then `<Audio N>`, then a reference name itself. `cast` maps `S1`/`Audio 1` to reference
names when the prompt does not use the names itself.
"""

from __future__ import annotations

import difflib
import re

SHOT = re.compile(r"\[Shot (\d+)\](?:\s*At (\d+):(\d+(?:\.\d+)?))?")
LINE = re.compile(r"<d>\s*(?:\[([^\]]*)\])?(.*?)</d>", re.S)
SPEAKER_ID = re.compile(r"\((S\d+)\)")
AUDIO = re.compile(r"<Audio (\d+)>")
AUDIO_DEF = re.compile(r"<Audio (\d+)> is the voice[- ]timbre reference for [^.\n]*?\((S\d+)\)")
# A line counts as found when its words match the transcript at least this well (difflib ratio on normalised text).
MATCH_RATIO = 0.75
# Seconds a line may start before its shot's cut before it is reported as mistimed; it must start this long before the next cut.
TIMING_SLACK = 0.25


def normalise(text: str) -> list[str]:
    return [w for w in re.sub(r"[^a-z0-9' ]+", " ", text.lower().replace("’", "'")).replace("'", "").split() if w]


def parse_cast(text: str) -> dict[str, str]:
    """`S1: Maren`, `Audio 2 = Liesl` lines -> {"S1": "Maren", "Audio 2": "Liesl"}."""
    cast = {}
    for row in text.splitlines():
        key, sep, name = row.replace("=", ":").partition(":")
        key = " ".join(key.strip().strip("<>()").split())
        if sep and key and name.strip():
            cast[key[0].upper() + key[1:]] = name.strip()
    return cast


def parse_prompt(prompt: str, names: list[str], cast: dict[str, str]) -> tuple[list[dict], list[dict]]:
    """(shots [{"shot", "start"}], lines [{"shot", "text", "language", "speaker", "speaker_mark"}]) from an H3 prompt."""
    audio_speaker = {f"Audio {n}": s for n, s in AUDIO_DEF.findall(prompt)}
    marks = [(m.start(), int(m.group(1)), 0.0 if m.group(2) is None else int(m.group(2)) * 60 + float(m.group(3))) for m in SHOT.finditer(prompt)]
    shots = [{"shot": n, "start": start} for _, n, start in marks] or [{"shot": 1, "start": 0.0}]
    name_pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True)) + r")\b", re.I) if names else None
    lines, previous_end = [], 0
    for m in LINE.finditer(prompt):
        shot_at = max((p for p in marks if p[0] < m.start()), default=None)
        context_start = max(previous_end, shot_at[0] if shot_at else 0)
        context = prompt[context_start:m.start()]
        candidates = [found[-1] for found in (SPEAKER_ID.findall(context), [f"Audio {n}" for n in AUDIO.findall(context)],
                                              name_pattern.findall(context) if name_pattern else []) if found]
        resolved = [(mark, _resolve(mark, names, cast, audio_speaker)) for mark in candidates]
        mark, speaker = next(((k, n) for k, n in resolved if n), resolved[0] if resolved else (None, None))
        lines.append({"shot": shot_at[1] if shot_at else 1, "text": m.group(2).strip(), "language": m.group(1) or "",
                      "speaker_mark": mark, "speaker": speaker})
        previous_end = m.end()
    return shots, lines


def _resolve(mark, names, cast, audio_speaker):
    if mark is None:
        return None
    if mark in cast:
        return cast[mark]
    if mark in audio_speaker and audio_speaker[mark] in cast:
        return cast[audio_speaker[mark]]
    for n in names:
        if n.lower() == mark.lower():
            return n
    return None


def _match(line_words, norm, words, used, window):
    """The unused run of transcript words that matches the line: (ratio, i, j).

    Of the runs within 0.1 of the best ratio, the one starting closest to the line's shot wins, so a line H3 said twice
    (for example a voice reference leaking into another shot) is matched where it was scripted and the other copy is
    reported as unscripted.
    """
    target = " ".join(line_words)
    n = len(line_words)
    found = []
    for i in range(len(norm)):
        for j in range(i + max(1, n - 1), min(len(norm), i + n + 2) + 1):
            if not any(used[i:j]):
                start = words[i]["start"]
                found.append((difflib.SequenceMatcher(None, target, " ".join(norm[i:j])).ratio(), i, j, max(0.0, window[0] - start, start - window[1])))
    if not found:
        return 0.0, 0, 0
    top = max(f[0] for f in found)
    near = [f for f in found if f[0] >= max(MATCH_RATIO, top - 0.1)] or [max(found)]
    r, i, j, _ = min(near, key=lambda f: (f[3], -f[0]))
    return r, i, j


def _speaker(words):
    weight = {}
    for w in words:
        weight[w["speaker"]] = weight.get(w["speaker"], 0.0) + max(0.01, w["end"] - w["start"])
    return max(weight, key=weight.get) if weight else None


def check(prompt: str, words: list[dict], duration: float, names: list[str], cast: dict[str, str] | None = None) -> dict:
    """Compare a prompt's lines with transcribed words ([{"word", "start", "end", "speaker"}] on the take's clock).

    Returns {"lines": [...], "unscripted": [...], "shots": [...], "summary": {...}}. Verdicts per line: ok, wrong_voice,
    unsure_voice (the diarizer put the expected voice and another reference in one speaker), no_reference (no reference
    for the expected speaker), missing (the line was not heard) and, separately, wrong_time with the drift in seconds.
    """
    shots, lines = parse_prompt(prompt, names, cast or {})
    norm = [(normalise(w["word"]) or [""])[0] for w in words]
    used = [False] * len(words)
    rows = []
    for k, line in enumerate(lines):
        ends = [s["start"] for s in shots if s["shot"] > line["shot"]]
        shot = next((s for s in shots if s["shot"] == line["shot"]), {"start": 0.0})
        window = [shot["start"], min(ends) if ends else duration]
        row = {"line": k + 1, "shot": line["shot"], "text": line["text"], "expected_speaker": line["speaker"],
               "speaker_mark": line["speaker_mark"], "shot_window": [round(window[0], 3), round(window[1], 3)]}
        ratio, i, j = _match(normalise(line["text"]), norm, words, used, window)
        if ratio < MATCH_RATIO:
            row.update({"verdict": "missing", "match_ratio": round(ratio, 3)})
            rows.append(row)
            continue
        for x in range(i, j):
            used[x] = True
        heard = words[i:j]
        detected = _speaker(heard)
        start, end = heard[0]["start"], heard[-1]["end"]
        row.update({"heard": " ".join(w["word"] for w in heard), "match_ratio": round(ratio, 3), "start": start, "end": end,
                    "detected_speaker": detected, "words": heard})
        expected = line["speaker"]
        if expected is None or expected not in names:
            row["verdict"] = "no_reference"
        elif detected == expected:
            row["verdict"] = "ok"
        elif detected and detected.startswith("ambiguous: ") and expected in detected.removeprefix("ambiguous: ").split(" | "):
            row["verdict"] = "unsure_voice"
        else:
            row["verdict"] = "wrong_voice"
        # A line belongs to the shot it starts in; dialogue may run over the next cut (<scenetrans>).
        drift = 0.0
        if start < window[0] - TIMING_SLACK:
            drift = start - window[0]
        elif start > window[1] - TIMING_SLACK:
            drift = start - window[1]
        row["wrong_time"] = bool(drift)
        row["drift_seconds"] = round(drift, 3)
        rows.append(row)
    unscripted, run = [], []
    for w, u in zip(words + [None], used + [True]):
        if w is not None and not u:
            run.append(w)
        elif run:
            unscripted.append({"text": " ".join(x["word"] for x in run), "start": run[0]["start"], "end": run[-1]["end"],
                               "speaker": _speaker(run), "words": run})
            run = []
    summary = {v: sum(r["verdict"] == v for r in rows) for v in ("ok", "wrong_voice", "unsure_voice", "no_reference", "missing")}
    summary["wrong_time"] = sum(bool(r.get("wrong_time")) for r in rows)
    summary["unscripted"] = len(unscripted)
    summary["lines"] = len(rows)
    return {"summary": summary, "lines": rows, "unscripted": unscripted, "shots": shots}
