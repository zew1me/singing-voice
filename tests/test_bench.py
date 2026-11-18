import json
from datetime import datetime

import numpy as np
import pytest
import soundfile as sf

from singing_voice.bench import TestSet, TestSetError, compare, format_table, load_summary, new_run_dir, rescore_run, run_bench


def write_testset(tmp_path, clips, reference="audio/ref.wav"):
    (tmp_path / "sets").mkdir(exist_ok=True)
    path = tmp_path / "sets" / "m1.json"
    path.write_text(json.dumps({"name": "m1", "reference": f"../{reference}", "clips": clips}))
    return path


@pytest.fixture
def audio_dir(tmp_path, tone):
    (tmp_path / "audio").mkdir()
    audio, sr = tone(seconds=3.0, sample_rate=44_100)
    for name in ("clean.wav", "stem.wav", "ref.wav"):
        sf.write(tmp_path / "audio" / name, audio, sr)
    return tmp_path / "audio"


def test_load_resolves_paths_relative_to_the_file(tmp_path, audio_dir):
    path = write_testset(tmp_path, [
        {"id": "clean", "kind": "clean", "source": "../audio/clean.wav", "start_s": 1, "duration_s": 1.5},
        {"id": "bleed", "kind": "bleed", "source": str(audio_dir / "stem.wav"), "duration_s": 2, "reference": "../audio/stem.wav"},
    ])
    testset = TestSet.load(path)
    assert testset.name == "m1" and [c.id for c in testset.clips] == ["clean", "bleed"]
    assert testset.clips[0].source == audio_dir / "clean.wav"
    assert testset.clips[0].reference == audio_dir / "ref.wav"
    assert testset.clips[1].reference == audio_dir / "stem.wav"
    assert testset.problems() == []


def test_problems_lists_missing_files_and_overlong_excerpts(tmp_path, audio_dir):
    path = write_testset(tmp_path, [
        {"id": "gone", "source": "../audio/missing.wav", "duration_s": 1},
        {"id": "long", "source": "../audio/clean.wav", "start_s": 2, "duration_s": 5},
    ])
    problems = TestSet.load(path).problems()
    assert any("gone: source" in p for p in problems)
    assert any("long: excerpt 2-7 s runs past the end" in p for p in problems)


@pytest.mark.parametrize(
    "clip, message",
    [
        ({"id": "a b", "source": "x.wav", "duration_s": 1}, "id must be"),
        ({"id": "a", "source": "x.wav"}, "duration_s"),
        ({"id": "a", "source": "x.wav", "duration_s": 120}, "inline transport"),
        ({"id": "a", "source": "x.wav", "duration_s": 1, "kind": "duet"}, "kind must be"),
        ({"id": "a", "duration_s": 1}, "source must be a path"),
        ({"id": "run", "source": "x.wav", "duration_s": 1}, "'run' is reserved"),
        ({"id": "summary", "source": "x.wav", "duration_s": 1}, "'summary' is reserved"),
    ],
)
def test_bad_clips_are_rejected(tmp_path, clip, message):
    with pytest.raises(TestSetError, match=message):
        TestSet.load(write_testset(tmp_path, [clip]))


def test_duplicate_ids_and_empty_sets_are_rejected(tmp_path):
    clip = {"id": "a", "source": "x.wav", "duration_s": 1}
    with pytest.raises(TestSetError, match="duplicate"):
        TestSet.load(write_testset(tmp_path, [clip, clip]))
    with pytest.raises(TestSetError, match="no clips"):
        TestSet.load(write_testset(tmp_path, []))


def test_new_run_dir_is_timestamped_and_safe(tmp_path):
    run_dir = new_run_dir(tmp_path, "steps 50 / cfg", now=datetime(2026, 11, 17, 9, 30))
    assert run_dir.name == "20261117T093000_steps-50-cfg"


def test_run_rescore_and_compare(tmp_path, audio_dir):
    path = write_testset(tmp_path, [
        {"id": "clean", "kind": "clean", "source": "../audio/clean.wav", "duration_s": 1},
        {"id": "bleed", "kind": "bleed", "source": "../audio/stem.wav", "duration_s": 1},
    ])
    testset = TestSet.load(path)

    def fake_convert(clip, out):
        if clip.id == "bleed":
            raise RuntimeError("job FAILED")
        sf.write(out, np.zeros(44_100, dtype=np.float32), 44_100)
        meta = {"options": {"semitone_shift": 0}, "reference": str(clip.reference),
                "source_excerpt": {"path": str(clip.source), "start_s": clip.start_s, "duration_s": clip.duration_s}}
        out.with_suffix(".json").write_text(json.dumps(meta))
        return meta

    cents = iter([12.0, 8.0])

    def fake_score(inputs):
        assert inputs.source == audio_dir / "clean.wav" and inputs.duration_s == 1
        return {"summary": {"pitch_median_abs_cents": next(cents), "timbre_to_reference": 0.5, "alignment_lag_ms": -10.0}}

    run_dir = tmp_path / "runs" / "a"
    summary = run_bench(testset, run_dir, fake_convert, fake_score, label="baseline", options={"diffusion_steps": 40})
    assert summary["label"] == "baseline" and summary["options"] == {"diffusion_steps": 40}
    assert summary["clips"]["clean"]["pitch_median_abs_cents"] == 12.0
    assert summary["clips"]["clean"]["kind"] == "clean"
    assert "job FAILED" in summary["clips"]["bleed"]["error"]
    assert (run_dir / "clean.score.json").is_file() and (run_dir / "clean.json").is_file()
    assert load_summary(run_dir) == summary
    with pytest.raises(FileExistsError):
        run_bench(testset, run_dir, fake_convert, fake_score, label="again", options={})

    before = json.loads((run_dir / "summary.json").read_text())
    after = rescore_run(run_dir, fake_score)
    assert after["clips"]["clean"]["pitch_median_abs_cents"] == 8.0
    assert "error" in after["clips"]["bleed"]

    rows = compare(before, after)
    cents_row = next(r for r in rows if r["metric"] == "pitch_median_abs_cents")
    assert cents_row == {"clip": "clean", "metric": "pitch_median_abs_cents", "before": 12.0, "after": 8.0, "delta": -4.0, "verdict": "better"}
    assert next(r for r in rows if r["metric"] == "timbre_to_reference")["verdict"] == ""
    table = format_table(rows, "a", "b")
    assert "pitch_median_abs_cents" in table and "better" in table


@pytest.mark.parametrize(
    "metric, before, after, verdict",
    [
        ("timbre_to_reference", 0.5, 0.6, "better"),
        ("timbre_to_source", 0.5, 0.6, "worse"),
        ("alignment_lag_ms", -20.0, 10.0, "better"),
        ("alignment_lag_ms", 0.0, 10.0, "worse"),
        ("alignment_lag_ms", -20.0, 20.0, ""),
        ("timbre_source_to_reference", 0.3, 0.4, ""),
    ],
)
def test_compare_knows_which_way_is_better(metric, before, after, verdict):
    rows = compare({"clips": {"c": {metric: before}}}, {"clips": {"c": {metric: after}}})
    assert rows == [{"clip": "c", "metric": metric, "before": before, "after": after, "delta": round(after - before, 4), "verdict": verdict}]


def test_a_report_that_cannot_be_written_is_recorded_and_the_run_goes_on(tmp_path, audio_dir):
    path = write_testset(tmp_path, [
        {"id": "clean", "source": "../audio/clean.wav", "duration_s": 1},
        {"id": "bleed", "source": "../audio/stem.wav", "duration_s": 1},
    ])

    def fake_convert(clip, out):
        sf.write(out, np.zeros(4410, dtype=np.float32), 44_100)
        meta = {"reference": str(clip.reference), "source_excerpt": {"path": str(clip.source), "duration_s": 1}}
        out.with_suffix(".json").write_text(json.dumps(meta))
        if clip.id == "clean":
            out.with_suffix(".score.json").mkdir()  # a directory where the report should go
        return meta

    run_dir = tmp_path / "run"
    summary = run_bench(TestSet.load(path), run_dir, fake_convert, lambda inputs: {"summary": {"wer": 0.1}}, label="x", options={})
    assert "score: IsADirectoryError" in summary["clips"]["clean"]["error"]
    assert summary["clips"]["bleed"]["wer"] == 0.1
    assert (run_dir / "summary.json").is_file()
