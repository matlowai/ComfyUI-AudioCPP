"""Named speakers for Nemotron Transcribe's diarization, from reference clips.

The model has no speaker enrollment. Its diarizer (Sortformer) numbers speakers in order of first appearance and keeps
those numbers across one recording, so the reference clips go first, one after another with silence between them, and
the audio to check goes last. The speaker column that is active during a reference clip gets that clip's name. A column
claimed by two references (the diarizer heard one voice in both) stays ambiguous, and a column no reference claims stays
unassigned: names are never guessed.
"""

from __future__ import annotations

import numpy as np

RATE = 16000
GAP_SECONDS = 1.0
# A reference names a column when that column covers at least this share of the reference's speech.
CLAIM_SHARE = 0.6


def combine(references: list[tuple[str, np.ndarray]], audio: np.ndarray) -> tuple[np.ndarray, list[dict], float]:
    """The references, then the audio, each followed by GAP_SECONDS of silence, all 16 kHz mono float32.

    Returns (combined samples, [{"name", "start", "end"} in seconds for each reference], the audio's start in seconds).
    """
    gap = np.zeros(int(GAP_SECONDS * RATE), dtype=np.float32)
    parts, spans, at = [], [], 0
    for name, clip in references:
        spans.append({"name": name, "start": at / RATE, "end": (at + len(clip)) / RATE})
        parts += [clip.astype(np.float32), gap]
        at += len(clip) + len(gap)
    parts.append(audio.astype(np.float32))
    return np.concatenate(parts), spans, at / RATE


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def claims(turns: list[dict], spans: list[dict]) -> list[dict]:
    """For each reference: the speaker column the diarizer heard during it, and that column's share of the speech there.

    turns: [{"start", "end", "column"}] in seconds on the combined clock. A reference whose speech is split between
    columns below CLAIM_SHARE claims nothing.
    """
    out = []
    for span in spans:
        per_column: dict[int, float] = {}
        for t in turns:
            o = _overlap(t["start"], t["end"], span["start"], span["end"])
            if o > 0:
                per_column[t["column"]] = per_column.get(t["column"], 0.0) + o
        total = sum(per_column.values())
        column, share = None, 0.0
        if total > 0:
            best = max(per_column, key=per_column.get)
            share = per_column[best] / total
            if share >= CLAIM_SHARE:
                column = best
        out.append({"name": span["name"], "column": column, "share": round(share, 3), "speech_seconds": round(total, 3),
                    "columns": {f"speaker_{c}": round(s, 3) for c, s in sorted(per_column.items())}})
    return out


def column_names(found: list[dict]) -> dict[int, list[str]]:
    """Speaker column -> the reference names that claimed it (more than one name: ambiguous)."""
    names: dict[int, list[str]] = {}
    for c in found:
        if c["column"] is not None:
            names.setdefault(c["column"], []).append(c["name"])
    return names


def label(column: int, names: dict[int, list[str]] | None) -> str:
    """A column's name: the one reference that claimed it, "ambiguous: A | B" when several did, "unassigned (speaker_N)" when none.

    Without references (names None) a column is just "speaker_N".
    """
    if column < 0:
        return "unknown"
    if names is None:
        return f"speaker_{column}"
    claimed = names.get(column, [])
    if len(claimed) == 1:
        return claimed[0]
    if claimed:
        return "ambiguous: " + " | ".join(claimed)
    return f"unassigned (speaker_{column})"


def turns_from_result(result: dict) -> list[dict]:
    """The server's speaker_turns (sample offsets at RATE, "speaker_N") as [{"start", "end", "column"}] in seconds."""
    return [{"start": t["start_sample"] / RATE, "end": t["end_sample"] / RATE, "column": int(t["speaker_id"].rsplit("_", 1)[1])}
            for t in result.get("speaker_turns", [])]


def words_from_result(result: dict) -> list[dict]:
    """The model's CTC words (custom_schema_output nemotron_transcribe.result.v1), times in seconds on the request clock."""
    data = (result.get("custom_schema_output") or {}).get("data") or {}
    return list(data.get("words", []))


def attribute(result: dict, spans: list[dict], audio_start: float) -> dict:
    """Names for a /v1/tasks/run result on combined audio, moved to the checked audio's own clock.

    Returns {"speakers": per-reference claims, "words": [...], "speaker_turns": [...]} with only the checked audio's words and
    turns, each carrying "speaker" (a name, "ambiguous: ...", "unassigned (speaker_N)" or "unknown") and "speaker_column".
    """
    turns = turns_from_result(result)
    found = claims(turns, spans)
    names = column_names(found) if spans else None
    words = []
    for w in words_from_result(result):
        if (w["start"] + w["end"]) / 2 < audio_start:
            continue
        words.append({"word": w.get("display") or w["word"], "start": round(w["start"] - audio_start, 3), "end": round(w["end"] - audio_start, 3),
                      "speaker": label(int(w["speaker_column"]), names), "speaker_column": int(w["speaker_column"]),
                      "speaker_confidence": w.get("speaker_confidence"), "transcript_speaker_tag": w.get("transcript_speaker_tag")})
    own_turns = []
    for t in turns:
        if t["end"] <= audio_start:
            continue
        own_turns.append({"start": round(max(t["start"], audio_start) - audio_start, 3), "end": round(t["end"] - audio_start, 3),
                          "speaker": label(t["column"], names), "speaker_column": t["column"]})
    return {"speakers": found, "words": words, "speaker_turns": own_turns}
