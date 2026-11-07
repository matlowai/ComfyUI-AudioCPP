from comfyui_audiocpp import h3qa

PROMPT = """subject_definitions:
<Subject 1> is the swordswoman.
<Subject 2> is the old headman.
<Audio 1> is the voice-timbre reference for <Subject 2> (S1).
<Audio 2> is the voice-timbre reference for <Subject 1> (S2).

detailed_description:
[Shot 1] The gate. Nobody speaks.
[Shot 2] At 00:04.000, <Subject 2> (S1) bows and says: <d>[English] The price, as agreed... please.</d> She says nothing.
[Shot 3] At 00:07.500, a close-up. <Subject 1> (S2) says quietly: <d>[English] It's you.</d>
"""
CAST = {"S1": "Brann", "S2": "Maren"}
NAMES = ["Maren", "Brann"]


def words(text, start, speaker, step=0.3):
    return [{"word": w, "start": round(start + i * step, 3), "end": round(start + (i + 1) * step, 3), "speaker": speaker} for i, w in enumerate(text.split())]


def test_parse_shots_lines_and_speakers():
    shots, lines = h3qa.parse_prompt(PROMPT, NAMES, CAST)
    assert shots == [{"shot": 1, "start": 0.0}, {"shot": 2, "start": 4.0}, {"shot": 3, "start": 7.5}]
    assert [(x["shot"], x["text"], x["speaker_mark"], x["speaker"]) for x in lines] == [
        (2, "The price, as agreed... please.", "S1", "Brann"), (3, "It's you.", "S2", "Maren")]


def test_audio_labels_resolve_through_their_speaker_id():
    prompt = PROMPT.replace("<Subject 1> (S2) says quietly", "<Subject 1>, with the voice of <Audio 2>, says quietly")
    _, lines = h3qa.parse_prompt(prompt, NAMES, {"S2": "Maren"})
    assert (lines[1]["speaker_mark"], lines[1]["speaker"]) == ("Audio 2", "Maren")


def test_names_in_the_prompt_need_no_cast():
    prompt = "[Shot 1] Brann, with the voice of <Audio 1>, says: <d>[English] Please.</d> [Shot 2] At 00:02.000, Maren says: <d>[English] No.</d>"
    _, lines = h3qa.parse_prompt(prompt, NAMES, {})
    assert [x["speaker"] for x in lines] == ["Brann", "Maren"]
    assert h3qa.parse_cast("S1: Maren\n<Audio 2> = Liesl\n\n") == {"S1": "Maren", "Audio 2": "Liesl"}


def test_a_clean_take_passes():
    heard = words("the price as agreed please", 4.5, "Brann") + words("its you", 9.0, "Maren")
    report = h3qa.check(PROMPT, heard, 11.0, NAMES, CAST)
    assert report["summary"] == {"ok": 2, "wrong_voice": 0, "unsure_voice": 0, "no_reference": 0, "missing": 0, "wrong_time": 0, "unscripted": 0, "lines": 2}


def test_wrong_voice_unsure_voice_and_timing():
    heard = words("its you", 2.0, "ambiguous: Maren | Liesl") + words("the price as agreed please", 4.5, "ambiguous: Maren | Liesl")
    report = h3qa.check(PROMPT, heard, 11.0, NAMES, CAST)
    first, second = report["lines"]
    assert first["verdict"] == "wrong_voice" and not first["wrong_time"]
    assert second["verdict"] == "unsure_voice" and second["wrong_time"] and second["drift_seconds"] == -5.5


def test_missing_and_unscripted_lines():
    heard = words("i dont need your thanks", 0.5, "Maren") + words("the price as agreed please", 4.5, "Brann")
    report = h3qa.check(PROMPT, heard, 11.0, NAMES, CAST)
    assert report["lines"][1]["verdict"] == "missing"
    assert [(u["text"], u["speaker"]) for u in report["unscripted"]] == [("i dont need your thanks", "Maren")]


def test_a_line_said_twice_is_matched_in_its_shot():
    heard = words("its you", 1.0, "Maren") + words("the price as agreed please", 4.5, "Brann") + words("its you", 8.0, "Maren")
    report = h3qa.check(PROMPT, heard, 11.0, NAMES, CAST)
    assert report["lines"][1]["start"] == 8.0 and not report["lines"][1]["wrong_time"]
    assert [u["start"] for u in report["unscripted"]] == [1.0]
