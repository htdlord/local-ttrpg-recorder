"""Settings model, presets, schema migrations and mode-aware parameter metadata (SPEC §3).

Settings are layered: model defaults -> preset -> config file -> ``--set`` overrides ->
explicit CLI options. Each ASR/decoding field may declare which ASR modes honour it
(``json_schema_extra={"modes": {...}}``); :func:`mode_issues` reports values that the
chosen mode ignores. Device/compute-type validity needs ctranslate2 and is checked at
runtime by :func:`resolve_runtime`, so a config file stays loadable on any machine.
"""

from __future__ import annotations

import os
import sys
import tomllib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import tomli_w
from pydantic import BaseModel, ConfigDict, Field, model_validator

from sessionscribe.core.cuda_libs import import_ctranslate2

SCHEMA_VERSION = 2

# Known-good compute types per device, in order of preference (SPEC §3.1).
KNOWN_GOOD_COMPUTE_TYPES: dict[str, tuple[str, ...]] = {
    "cuda": ("float16", "int8_float16", "bfloat16", "int8", "float32"),
    "cpu": ("int8", "int8_float32", "float32"),
}
DEFAULT_COMPUTE_TYPE: dict[str, str] = {"cuda": "float16", "cpu": "int8"}
DEVICES = ("cuda", "cpu")

AsrMode = Literal["quality", "fast"]
ASR_MODES: tuple[str, ...] = ("quality", "fast")

# How a mode treats a setting.
#   honoured     - passed to the library and effective
#   ignored      - the library overrides or never uses it
#   first_only   - only the first list element is used (temperature in fast mode)
#   report_only  - not used for decisions by the library; our own post-checks still use it
ModeSupport = Literal["honoured", "ignored", "first_only", "report_only"]

# Fast mode starts at batch_size (default 16) and halves on GPU OOM down to 1 (SPEC §3.2).
DEFAULT_FAST_BATCH_SIZE = 16


def _modes(quality: ModeSupport, fast: ModeSupport) -> dict[str, Any]:
    return {"modes": {"quality": quality, "fast": fast}}


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class ModelSettings(_Section):
    id: str = Field("large-v3", description="Model size name or local model folder.")
    revision: str | None = Field(None, description="Hugging Face revision (default: latest).")
    device: Literal["auto", "cuda", "cpu"] = "auto"
    compute_type: str | None = Field(None, description="Default: float16 on cuda, int8 on cpu.")
    cpu_threads: int | None = Field(None, ge=0, description="Default: physical cores.")
    num_workers: int = Field(1, ge=1)
    model_dir: Path | None = Field(None, description="Download folder (default: HF cache).")
    offline: bool = Field(False, description="Use locally cached models only, no network.")


class AsrSettings(_Section):
    mode: AsrMode = "quality"
    batch_size: int | None = Field(
        None,
        ge=1,
        description="Fast mode starting batch size (default 16); halves on OOM down to 1.",
        json_schema_extra=_modes("ignored", "honoured"),
    )
    oom_fallback: bool = Field(
        True,
        description="Fast mode: on GPU OOM retry with half the batch size (for smaller GPUs).",
        json_schema_extra=_modes("ignored", "honoured"),
    )
    # Chunking (SPEC §5.1), in track-local time; changing these re-runs ASR.
    chunk_owned_s: float = Field(600.0, gt=0, description="Owned range per chunk.")
    chunk_context_s: float = Field(5.0, ge=0, description="Extra audio ASR sees on each side.")
    chunk_boundary_search_s: float = Field(
        15.0, ge=0, description="How far a boundary may move to the quietest point."
    )
    # Whisper sometimes skips speech after a window start (DECISIONS 2026-10-30): VAD
    # speech that got no words is transcribed again as its own short window.
    refill_holes: bool = Field(True, description="Re-transcribe VAD speech without words.")
    refill_min_s: float = Field(1.5, gt=0, description="Shortest word-less stretch to refill.")
    refill_pad_s: float = Field(0.5, ge=0, description="Audio added around a refill window.")


DEFAULT_TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


class DecodingSettings(_Section):
    language: str = Field("auto", description="ISO code (hu, en, ...) or auto.")
    task: Literal["transcribe"] = "transcribe"
    beam_size: int = Field(5, ge=1)
    best_of: int = Field(5, ge=1, json_schema_extra=_modes("honoured", "ignored"))
    temperature: list[float] = Field(
        default_factory=lambda: list(DEFAULT_TEMPERATURES),
        min_length=1,
        json_schema_extra=_modes("honoured", "first_only"),
    )
    condition_on_previous_text: bool = Field(False, json_schema_extra=_modes("honoured", "ignored"))
    no_speech_threshold: float | None = Field(
        0.6, json_schema_extra=_modes("honoured", "report_only")
    )
    log_prob_threshold: float | None = Field(
        -1.0, json_schema_extra=_modes("honoured", "report_only")
    )
    compression_ratio_threshold: float | None = Field(
        2.4, json_schema_extra=_modes("honoured", "report_only")
    )
    hallucination_silence_threshold: float | None = Field(
        None, gt=0, json_schema_extra=_modes("honoured", "ignored")
    )
    word_timestamps: bool = True


class VocabularySettings(_Section):
    initial_prompt: str | None = None
    hotwords: str | None = None


class VadSettings(_Section):
    """Silero VAD; defaults are the "Conversation" starting point (SPEC §3.4)."""

    enabled: bool = True
    threshold: float = Field(0.5, gt=0, lt=1)
    min_speech_duration_ms: int = Field(0, ge=0)
    min_silence_duration_ms: int = Field(800, ge=0)
    speech_pad_ms: int = Field(400, ge=0)
    max_speech_duration_s: float = Field(30.0, gt=0)


class BleedSettings(_Section):
    """Bleed handling (SPEC §3.5). Defaults are provisional until measured on clip02."""

    mode: Literal["post", "pre", "off"] = "post"
    coupling: Literal["auto", "guided", "manual"] = "auto"
    ownership_margin_db: float = 6.0
    pre_attenuation_db: float = Field(15.0, ge=0)
    mute: bool = Field(False, description="Expert option: mute instead of attenuate (pre mode).")
    frame_ms: int = Field(20, gt=0)
    hold_ms: int = Field(200, ge=0, description="Ownership lasts this long after a frame.")
    release_ms: int = Field(300, ge=0, description="Pre mode: gain ramp back to attenuation.")
    # Per track, relative: mic gains differ by 15-20 dB (owner decision 2026-10-30).
    floor_above_noise_db: float = Field(
        10.0, description="A track owns nothing quieter than its noise level + this."
    )
    suppress_below: float = Field(
        0.5, ge=0, le=1, description="Ownership under this may be suppressed as bleed."
    )
    low_ownership_below: float = Field(
        0.6, ge=0, le=1, description="Ownership under this is flagged low_ownership."
    )
    text_similarity: float = Field(
        70.0, ge=0, le=100, description="Bleed text must match the source's words this well."
    )
    coupling_min_frames: int = Field(250, ge=1, description="Frames needed per mic pair.")
    coupling_default_db: float = Field(-30.0, description="Coupling for pairs with too few frames.")


class DedupSettings(_Section):
    """Used from M3 on (SPEC §3.6)."""

    enabled: bool = True
    similarity: float = Field(85.0, ge=0, le=100)
    max_time_gap_s: float = Field(1.0, ge=0)
    min_words: int = Field(3, ge=1)
    mark_overlaps: bool = True


ExportFormat = Literal["txt", "srt", "vtt", "json", "md"]


def _default_formats() -> list[ExportFormat]:
    return ["json", "txt"]


class QualitySettings(_Section):
    """Thresholds for quality flags (SPEC §5.6); a flag marks, it never removes."""

    enabled: bool = True
    low_logprob: float = Field(-1.0, description="Flag avg_logprob below this.")
    # no_speech_prob and avg_logprob belong to the whole 30 s decoding window, so a high
    # no_speech_prob alone says little; like Whisper's silence rule, it needs a low logprob.
    high_no_speech: float = Field(0.6, ge=0, le=1, description="Flag no_speech_prob above this.")
    high_no_speech_logprob: float = Field(
        -0.5, description="... when avg_logprob is also below this."
    )
    high_compression: float = Field(2.4, gt=0, description="Flag compression ratio above this.")
    repeated_ngram: int = Field(3, ge=1, description="n of the repeated n-gram check.")
    repeated_ngram_times: int = Field(3, ge=2, description="Occurrences that count as a loop.")
    max_chars_per_s: float = Field(
        35.0, gt=0, description="Text too long for its voiced time (at least 1 s counted)."
    )
    min_vad_fraction: float = Field(
        0.5, ge=0, le=1, description="Flag segments with less of their time in VAD speech."
    )


class OutputSettings(_Section):
    formats: list[ExportFormat] = Field(default_factory=_default_formats, min_length=1)
    timestamp_style: Literal["hh:mm:ss", "hh:mm:ss.mmm"] = "hh:mm:ss"
    merge_same_speaker_gap_s: float = Field(0.0, ge=0)
    filename_template: str = "{stem}"
    output_dir: Path | None = None
    work_dir: Path | None = None


def english_only_model(model_id: str) -> bool:
    """``.en`` and distil English models cannot transcribe other languages (SPEC §3.1)."""
    name = Path(model_id).name.lower()
    return name.endswith(".en") or name.startswith("distil-")


class Settings(_Section):
    schema_version: int = SCHEMA_VERSION
    preset: str | None = Field(None, description="Name of the preset these settings start from.")
    model: ModelSettings = Field(default_factory=ModelSettings)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    decoding: DecodingSettings = Field(default_factory=DecodingSettings)
    vocabulary: VocabularySettings = Field(default_factory=VocabularySettings)
    vad: VadSettings = Field(default_factory=VadSettings)
    bleed: BleedSettings = Field(default_factory=BleedSettings)
    dedup: DedupSettings = Field(default_factory=DedupSettings)
    quality: QualitySettings = Field(default_factory=QualitySettings)
    output: OutputSettings = Field(default_factory=OutputSettings)

    @model_validator(mode="after")
    def _check_combinations(self) -> Settings:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version {self.schema_version} after migration; expected {SCHEMA_VERSION}"
            )
        if english_only_model(self.model.id):
            if self.decoding.language == "auto":
                self.decoding.language = "en"
            elif self.decoding.language != "en":
                raise ValueError(
                    f"model {self.model.id!r} is English-only; "
                    f"language {self.decoding.language!r} is not possible"
                )
        if (
            self.asr.mode == "quality"
            and self.decoding.hallucination_silence_threshold is not None
            and not self.decoding.word_timestamps
        ):
            raise ValueError("decoding.hallucination_silence_threshold needs word_timestamps")
        if self.asr.mode == "fast" and not self.vad.enabled:
            raise ValueError(
                "fast mode needs VAD (the batched pipeline cuts the audio at speech segments)"
            )
        return self


# --- mode metadata ------------------------------------------------------------------------

SECTIONS: tuple[str, ...] = (
    "model",
    "asr",
    "decoding",
    "vocabulary",
    "vad",
    "bleed",
    "dedup",
    "output",
)


def mode_support(path: str, mode: str) -> ModeSupport:
    """How ``mode`` treats the setting at ``section.field``; unannotated fields are honoured."""
    section, _, name = path.partition(".")
    model_cls = Settings.model_fields[section].annotation
    assert isinstance(model_cls, type) and issubclass(model_cls, BaseModel), path
    extra = model_cls.model_fields[name].json_schema_extra
    if isinstance(extra, dict):
        modes = extra.get("modes")
        if isinstance(modes, dict) and mode in modes:
            return cast(ModeSupport, modes[mode])
    return "honoured"


def mode_table() -> dict[str, dict[str, ModeSupport]]:
    """All annotated settings: ``{path: {mode: support}}``. Pinned by a regression test."""
    table: dict[str, dict[str, ModeSupport]] = {}
    for section in SECTIONS:
        model_cls = Settings.model_fields[section].annotation
        assert isinstance(model_cls, type) and issubclass(model_cls, BaseModel)
        for name, info in model_cls.model_fields.items():
            extra = info.json_schema_extra
            if isinstance(extra, dict) and isinstance(extra.get("modes"), dict):
                table[f"{section}.{name}"] = {
                    m: mode_support(f"{section}.{name}", m) for m in ASR_MODES
                }
    return table


@dataclass(frozen=True)
class ModeIssue:
    path: str
    support: ModeSupport
    message: str


def mode_issues(settings: Settings) -> list[ModeIssue]:
    """Settings whose value has no (full) effect in the chosen ASR mode."""
    mode = settings.asr.mode
    defaults = Settings()
    issues: list[ModeIssue] = []
    for path, modes in mode_table().items():
        support = modes[mode]
        if support == "honoured":
            continue
        value = get_path(settings, path)
        if value == get_path(defaults, path):
            continue  # defaults are documented per mode; warn about explicit choices only
        if support == "first_only":
            if isinstance(value, list) and len(value) > 1:
                issues.append(
                    ModeIssue(path, support, f"{path}: only the first value is used in {mode} mode")
                )
            continue
        what = "is ignored" if support == "ignored" else "is only reported, not used for retries"
        issues.append(ModeIssue(path, support, f"{path} {what} in {mode} mode"))
    return issues


def get_path(settings: Settings, path: str) -> Any:
    section, _, name = path.partition(".")
    return getattr(getattr(settings, section), name)


# --- layering, overrides, migrations, TOML ------------------------------------------------

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
    """bleed.floor_db (absolute, never used) gave way to the per-track floor_above_noise_db."""
    bleed = dict(data.get("bleed", {}))
    bleed.pop("floor_db", None)
    return {**data, "bleed": bleed} if "bleed" in data else data


# MIGRATIONS[n] turns a schema_version n mapping into schema_version n + 1.
MIGRATIONS: dict[int, Migration] = {1: _v1_to_v2}


def migrate(
    data: Mapping[str, Any], migrations: Mapping[int, Migration] | None = None
) -> dict[str, Any]:
    """Bring a settings mapping up to :data:`SCHEMA_VERSION`. Missing version = current."""
    steps = MIGRATIONS if migrations is None else migrations
    out = dict(data)
    version = int(out.get("schema_version", SCHEMA_VERSION))
    target = max([SCHEMA_VERSION, *(v + 1 for v in steps)])
    if version > target:
        raise ValueError(
            f"settings schema_version {version} is newer than this app supports ({target})"
        )
    while version < target:
        if version not in steps:
            raise ValueError(f"no migration from settings schema_version {version}")
        out = steps[version](out)
        version += 1
        out["schema_version"] = version
    return out


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def parse_override(item: str) -> dict[str, Any]:
    """``section.field=value`` -> nested dict. The value is parsed as TOML, else a string."""
    key, sep, raw = item.partition("=")
    key = key.strip()
    if not sep or not key:
        raise ValueError(f"override {item!r} must look like section.field=value")
    raw = raw.strip()
    try:
        value: Any = tomllib.loads(f"v = {raw}")["v"]
    except tomllib.TOMLDecodeError:
        value = raw
    parts = key.split(".")
    nested: dict[str, Any] = {parts[-1]: value}
    for part in reversed(parts[:-1]):
        nested = {part: nested}
    return nested


def read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as fh:
        return tomllib.load(fh)


def settings_to_toml(settings: Settings) -> str:
    """TOML text; ``None`` values are left out (they mean "default")."""
    return tomli_w.dumps(settings.model_dump(mode="json", exclude_none=True))


def write_settings(settings: Settings, path: Path) -> None:
    from sessionscribe.core.fileio import atomic_write_text

    atomic_write_text(path, settings_to_toml(settings))


def user_config_dir() -> Path:
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
        return Path(base) / "sessionscribe"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "sessionscribe"


# --- presets ------------------------------------------------------------------------------

BUILTIN_PRESETS: dict[str, dict[str, Any]] = {
    # Default: quality mode, large-v3, conversation VAD (the Settings defaults).
    "conversation": {},
    # Optional, faster: batched large-v3 at batch size 4 (M1 benchmark: about 2.4x faster,
    # +3 pp WER); temperature fallback and anti-hallucination retries are not available.
    "fast": {"asr": {"mode": "fast", "batch_size": 4}},
    # Bleed profiles (SPEC §3.5). Conservative = the default post mode: attribution after
    # ASR, nothing re-transcribed when its parameters change.
    "conservative": {"bleed": {"mode": "post"}},
    # Strong: attenuate frames a track does not own before ASR (needs re-ASR to retune).
    "strong": {"bleed": {"mode": "pre", "pre_attenuation_db": 15.0}},
}
DEFAULT_PRESET = "conversation"


def user_presets_dir() -> Path:
    return user_config_dir() / "presets"


def list_presets(user_dir: Path | None = None) -> dict[str, Path | None]:
    """Preset name -> file (``None`` for built-ins). User presets may shadow built-ins."""
    found: dict[str, Path | None] = dict.fromkeys(BUILTIN_PRESETS)
    folder = user_presets_dir() if user_dir is None else user_dir
    if folder.is_dir():
        for path in sorted(folder.glob("*.toml")):
            found[path.stem] = path
    return found


def load_preset(name: str, user_dir: Path | None = None) -> dict[str, Any]:
    presets = list_presets(user_dir)
    if name not in presets:
        raise ValueError(f"unknown preset {name!r}; available: {', '.join(presets)}")
    path = presets[name]
    data = BUILTIN_PRESETS[name] if path is None else migrate(read_toml(path))
    return deep_merge(data, {"preset": name})


def load_settings(
    *,
    preset: str | None = DEFAULT_PRESET,
    config_path: Path | None = None,
    overrides: Iterable[str] = (),
    cli: Mapping[str, Any] | None = None,
    user_presets: Path | None = None,
    session: Mapping[str, Any] | None = None,
    session_preset: str | None = None,
) -> Settings:
    """Resolve settings from all layers; raises ``ValueError`` / ``ValidationError``.

    Layers: preset -> config file -> session ``[settings]`` -> ``--set`` -> CLI flags.
    """
    data: dict[str, Any] = {}
    file_data: dict[str, Any] = {}
    if config_path is not None:
        file_data = migrate(read_toml(config_path))
    # An explicit preset argument wins over the session's, which wins over the config file's.
    preset_name = preset if preset is not None else session_preset or file_data.get("preset")
    if preset_name:
        data = deep_merge(data, load_preset(preset_name, user_presets))
    data = deep_merge(data, file_data)
    if session:
        data = deep_merge(data, session)
    for item in overrides:
        data = deep_merge(data, parse_override(item))
    if cli:
        data = deep_merge(data, cli)
    data["schema_version"] = SCHEMA_VERSION
    return Settings.model_validate(data)


# --- runtime device / compute type --------------------------------------------------------


def available_compute_types(device: str, supported: Iterable[str] | None = None) -> tuple[str, ...]:
    """Known-good compute types that ctranslate2 actually supports on ``device``.

    ``supported`` overrides the runtime query (used in tests).
    """
    if device not in KNOWN_GOOD_COMPUTE_TYPES:
        raise ValueError(f"unknown device {device!r}; expected one of {DEVICES}")
    if supported is None:
        supported = import_ctranslate2().get_supported_compute_types(device)
    supported_set = set(supported)
    return tuple(ct for ct in KNOWN_GOOD_COMPUTE_TYPES[device] if ct in supported_set)


def cuda_device_count() -> int:
    return int(import_ctranslate2().get_cuda_device_count())


def resolve_device(device: str) -> str:
    """Map ``auto`` to ``cuda`` when a CUDA device is visible, otherwise ``cpu``."""
    if device == "auto":
        return "cuda" if cuda_device_count() > 0 else "cpu"
    if device not in DEVICES:
        raise ValueError(f"unknown device {device!r}; expected auto, cuda or cpu")
    return device


def validate_compute_type(device: str, compute_type: str | None) -> str:
    """Return the compute type to use on ``device``, or raise if it is not available."""
    available = available_compute_types(device)
    if compute_type is None:
        default = DEFAULT_COMPUTE_TYPE[device]
        if default in available:
            return default
        if not available:
            raise ValueError(f"no known-good compute type is supported on {device}")
        return available[0]
    if compute_type not in available:
        raise ValueError(
            f"compute type {compute_type!r} is not supported on {device}; "
            f"available: {', '.join(available) or 'none'}"
        )
    return compute_type


def physical_cores() -> int:
    logical = os.cpu_count() or 2
    return max(1, logical // 2)  # SMT assumed; good enough for a thread-count default


@dataclass(frozen=True)
class RuntimeModel:
    """Device, compute type and thread count actually used for a run."""

    device: str
    compute_type: str
    cpu_threads: int


def resolve_runtime(settings: Settings) -> RuntimeModel:
    device = resolve_device(settings.model.device)
    compute_type = validate_compute_type(device, settings.model.compute_type)
    threads = settings.model.cpu_threads or physical_cores()
    return RuntimeModel(device, compute_type, threads)
