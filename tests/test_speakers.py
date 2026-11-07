import numpy as np
import pytest

from comfyui_audiocpp import speakers


def turn(start, end, column):
    return {"start_sample": int(start * speakers.RATE), "end_sample": int(end * speakers.RATE), "speaker_id": f"speaker_{column}", "confidence": 0}


def word(w, start, end, column):
    return {"word": w, "display": w, "start": start, "end": end, "speaker_column": column, "speaker_confidence": 0.9, "transcript_speaker_tag": "spk:0"}


def test_combine_puts_the_references_first_with_gaps():
    a, b, take = np.ones(16000, np.float32), np.ones(32000, np.float32), np.ones(8000, np.float32)
    combined, spans, start = speakers.combine([("Maren", a), ("Brann", b)], take)
    assert spans == [{"name": "Maren", "start": 0.0, "end": 1.0}, {"name": "Brann", "start": 2.0, "end": 4.0}]
    assert start == 5.0 and len(combined) == 5 * 16000 + 8000
    assert not combined[16000:32000].any()


def test_names_follow_the_reference_columns_and_move_to_the_audio_clock():
    spans = [{"name": "Maren", "start": 0.0, "end": 4.0}, {"name": "Brann", "start": 5.0, "end": 9.0}]
    result = {"speaker_turns": [turn(0.2, 3.8, 0), turn(5.1, 8.9, 1), turn(10.5, 11.0, 0), turn(11.2, 11.9, 2)],
              "custom_schema_output": {"data": {"words": [word("ref", 1.0, 1.5, 0), word("its", 10.5, 10.7, 0), word("you", 10.7, 11.0, 0),
                                                          word("who", 11.2, 11.9, 2)]}}}
    out = speakers.attribute(result, spans, 10.0)
    assert [(w["word"], w["start"], w["speaker"]) for w in out["words"]] == [("its", 0.5, "Maren"), ("you", 0.7, "Maren"),
                                                                             ("who", 1.2, "unassigned (speaker_2)")]
    assert [t["speaker"] for t in out["speaker_turns"]] == ["Maren", "unassigned (speaker_2)"]
    assert [(c["name"], c["column"]) for c in out["speakers"]] == [("Maren", 0), ("Brann", 1)]


def test_one_column_in_two_references_is_ambiguous():
    spans = [{"name": "Maren", "start": 0.0, "end": 4.0}, {"name": "Liesl", "start": 5.0, "end": 9.0}]
    result = {"speaker_turns": [turn(0, 4, 0), turn(5, 9, 0), turn(10, 11, 0)],
              "custom_schema_output": {"data": {"words": [word("hush", 10, 11, 0)]}}}
    assert speakers.attribute(result, spans, 10.0)["words"][0]["speaker"] == "ambiguous: Maren | Liesl"


@pytest.mark.parametrize("shares, column", [((3.0, 1.0), 0), ((2.0, 2.0), None)])
def test_a_split_reference_claims_nothing(shares, column):
    turns = [{"start": 0.0, "end": shares[0], "column": 0}, {"start": 10.0, "end": 10.0 + shares[1], "column": 1}]
    found = speakers.claims(turns, [{"name": "A", "start": 0.0, "end": 20.0}])
    assert found[0]["column"] == column


def test_without_references_speakers_are_numbered():
    result = {"speaker_turns": [turn(0, 1, 1)], "custom_schema_output": {"data": {"words": [word("hi", 0.1, 0.4, 1)]}}}
    assert speakers.attribute(result, [], 0.0)["words"][0]["speaker"] == "speaker_1"
