import pytest
from pydantic import ValidationError

from ttrpg_recorder.core import config
from ttrpg_recorder.core.config import (
    Settings,
    deep_merge,
    load_settings,
    migrate,
    mode_issues,
    mode_table,
    parse_override,
    settings_to_toml,
)


def test_defaults_are_the_conversation_preset():
    s = load_settings()
    assert s.preset == "conversation"
    assert s.asr.mode == "quality"
    assert s.model.id == "large-v3"
    assert s.decoding.condition_on_previous_text is False
    assert s.decoding.word_timestamps is True
    v = s.vad
    assert (v.threshold, v.min_speech_duration_ms, v.min_silence_duration_ms) == (0.5, 0, 800)
    assert (v.speech_pad_ms, v.max_speech_duration_s) == (400, 30.0)


def test_layer_order(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text('[decoding]\nbeam_size = 3\nlanguage = "en"\n', encoding="utf-8")
    s = load_settings(
        preset="fast",
        config_path=cfg,
        overrides=["decoding.beam_size=2"],
        cli={"decoding": {"language": "hu"}},
    )
    assert s.model.id == "large-v3" and s.asr.mode == "fast" and s.asr.batch_size == 4  # preset
    assert s.decoding.beam_size == 2  # --set beats file
    assert s.decoding.language == "hu"  # CLI beats everything


def test_config_file_can_name_its_preset(tmp_path):
    cfg = tmp_path / "s.toml"
    cfg.write_text('preset = "fast"\n', encoding="utf-8")
    assert load_settings(preset=None, config_path=cfg).asr.mode == "fast"


def test_user_preset_shadows_and_unknown_preset(tmp_path):
    (tmp_path / "mine.toml").write_text('[decoding]\nlanguage = "hu"\n', encoding="utf-8")
    s = load_settings(preset="mine", user_presets=tmp_path)
    assert s.decoding.language == "hu" and s.preset == "mine"
    with pytest.raises(ValueError, match="unknown preset"):
        load_settings(preset="nope", user_presets=tmp_path)


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        ("decoding.beam_size=3", {"decoding": {"beam_size": 3}}),
        ("decoding.temperature=[0.0, 0.4]", {"decoding": {"temperature": [0.0, 0.4]}}),
        ("vocabulary.hotwords=Gandalf Mordor", {"vocabulary": {"hotwords": "Gandalf Mordor"}}),
        ('decoding.language="hu"', {"decoding": {"language": "hu"}}),
        ("vad.enabled=false", {"vad": {"enabled": False}}),
    ],
)
def test_parse_override(item, expected):
    assert parse_override(item) == expected


def test_parse_override_rejects_garbage():
    with pytest.raises(ValueError):
        parse_override("no-equals-sign")


def test_unknown_field_rejected():
    with pytest.raises(ValidationError):
        load_settings(overrides=["decoding.beam=3"])


def test_english_only_models():
    assert load_settings(cli={"model": {"id": "medium.en"}}).decoding.language == "en"
    with pytest.raises(ValidationError, match="English-only"):
        load_settings(cli={"model": {"id": "distil-large-v3"}, "decoding": {"language": "hu"}})


def test_hallucination_threshold_needs_word_timestamps():
    with pytest.raises(ValidationError, match="word_timestamps"):
        load_settings(
            overrides=["decoding.hallucination_silence_threshold=2.0"],
            cli={"decoding": {"word_timestamps": False}},
        )


def test_fast_mode_needs_vad():
    with pytest.raises(ValidationError, match="needs VAD"):
        load_settings(preset="fast", overrides=["vad.enabled=false"])


def test_mode_table_is_pinned():
    # Behaviour of faster-whisper 1.2.1 (see test_asr_regression.py); update both together.
    assert mode_table() == {
        "asr.batch_size": {"quality": "ignored", "fast": "honoured"},
        "asr.oom_fallback": {"quality": "ignored", "fast": "honoured"},
        "decoding.best_of": {"quality": "honoured", "fast": "ignored"},
        "decoding.temperature": {"quality": "honoured", "fast": "first_only"},
        "decoding.condition_on_previous_text": {"quality": "honoured", "fast": "ignored"},
        "decoding.no_speech_threshold": {"quality": "honoured", "fast": "report_only"},
        "decoding.log_prob_threshold": {"quality": "honoured", "fast": "report_only"},
        "decoding.compression_ratio_threshold": {"quality": "honoured", "fast": "report_only"},
        "decoding.hallucination_silence_threshold": {"quality": "honoured", "fast": "ignored"},
    }


def test_pinned_library_version():
    import faster_whisper

    # Bumping faster-whisper requires re-running the slow regression test first.
    assert faster_whisper.__version__ == "1.2.1"


def test_mode_issues_only_for_explicit_values():
    assert mode_issues(load_settings(preset="fast")) == []
    assert mode_issues(load_settings()) == []
    s = load_settings(
        preset="fast",
        overrides=[
            "decoding.condition_on_previous_text=true",
            "decoding.temperature=[0.0, 0.5]",
            "decoding.log_prob_threshold=-0.5",
            "decoding.hallucination_silence_threshold=2.0",
            "decoding.best_of=3",
        ],
    )
    issues = {i.path: i.support for i in mode_issues(s)}
    assert issues == {
        "decoding.condition_on_previous_text": "ignored",
        "decoding.temperature": "first_only",
        "decoding.log_prob_threshold": "report_only",
        "decoding.hallucination_silence_threshold": "ignored",
        "decoding.best_of": "ignored",
    }
    quality = load_settings(overrides=["asr.batch_size=4"])
    assert [i.path for i in mode_issues(quality)] == ["asr.batch_size"]


def test_toml_round_trip(tmp_path):
    s = load_settings(overrides=["vocabulary.initial_prompt=Árvíztűrő tükörfúrógép, ő ű"])
    path = tmp_path / "out.toml"
    config.write_settings(s, path)
    assert path.read_bytes()[:3] != b"\xef\xbb\xbf"
    again = load_settings(preset=None, config_path=path)
    assert again == s
    assert "None" not in settings_to_toml(s)


def test_migrations():
    steps = {1: lambda d: deep_merge(d, {"renamed": True})}
    assert migrate({"schema_version": 1, "x": 1}, steps) == {
        "schema_version": 2,
        "x": 1,
        "renamed": True,
    }
    with pytest.raises(ValueError, match="newer"):
        migrate({"schema_version": 99})
    assert migrate({"a": 1}) == {"a": 1}  # missing version = current


def test_v1_settings_drop_absolute_floor(tmp_path):
    cfg = tmp_path / "old.toml"
    toml = 'schema_version = 1\n[bleed]\nfloor_db = -60.0\nmode = "pre"\n'
    cfg.write_text(toml, encoding="utf-8")
    s = load_settings(preset=None, config_path=cfg)
    assert s.bleed.mode == "pre" and s.bleed.floor_above_noise_db == 10.0


def test_settings_schema_version_must_match():
    with pytest.raises(ValidationError):
        Settings(schema_version=99)


def test_resolve_runtime(monkeypatch):
    monkeypatch.setattr(config, "cuda_device_count", lambda: 0)
    monkeypatch.setattr(config, "available_compute_types", lambda device: ("int8", "float32"))
    rt = config.resolve_runtime(load_settings())
    assert rt.device == "cpu" and rt.compute_type == "int8" and rt.cpu_threads >= 1


def test_bleed_profiles():
    assert load_settings(preset="conservative").bleed.mode == "post"
    strong = load_settings(preset="strong").bleed
    assert (strong.mode, strong.pre_attenuation_db, strong.mute) == ("pre", 15.0, False)
