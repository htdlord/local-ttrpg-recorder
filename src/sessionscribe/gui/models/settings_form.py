"""The settings form as data (no Qt): one field spec per setting, generated from the
pydantic model, plus mode awareness and the diff that goes into a session's
``[settings]`` table."""

from __future__ import annotations

import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from sessionscribe.core.config import Settings, mode_support

FieldKind = Literal["bool", "int", "float", "choice", "multichoice", "text", "path", "floats"]

# Shown on the Basic page, in this order; everything is also on Advanced.
BASIC_FIELDS = (
    "model.id",
    "decoding.language",
    "asr.mode",
    "bleed.mode",
    "output.formats",
    "output.timestamp_style",
)
MODEL_CHOICES = ("large-v3", "large-v3-turbo", "medium", "small", "base", "tiny")
LANGUAGE_CHOICES = ("hu", "en", "auto")


@dataclass(frozen=True)
class FieldSpec:
    section: str
    name: str
    kind: FieldKind
    default: Any
    description: str = ""
    choices: tuple[str, ...] = ()
    optional: bool = False  # None allowed ("automatic" / unset)
    minimum: float | None = None
    maximum: float | None = None
    editable_choices: bool = False  # a choice list that also accepts free text

    @property
    def path(self) -> str:
        return f"{self.section}.{self.name}"


def sections() -> list[str]:
    return [name for name in Settings.model_fields if name not in ("schema_version", "preset")]


def _unwrap_optional(annotation: Any) -> tuple[Any, bool]:
    origin = typing.get_origin(annotation)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return annotation, False


def _kind(annotation: Any) -> tuple[FieldKind, tuple[str, ...]]:
    origin = typing.get_origin(annotation)
    if origin is Literal:
        return "choice", tuple(str(a) for a in typing.get_args(annotation))
    if origin is list:
        (item,) = typing.get_args(annotation)
        if typing.get_origin(item) is Literal:
            return "multichoice", tuple(str(a) for a in typing.get_args(item))
        return "floats", ()
    if annotation is bool:
        return "bool", ()
    if annotation is int:
        return "int", ()
    if annotation is float:
        return "float", ()
    if annotation is Path:
        return "path", ()
    return "text", ()


def _bounds(metadata: list[Any]) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in metadata:
        for attr in ("ge", "gt"):
            if getattr(m, attr, None) is not None:
                lo = float(getattr(m, attr))
        for attr in ("le", "lt"):
            if getattr(m, attr, None) is not None:
                hi = float(getattr(m, attr))
    return lo, hi


def field_specs() -> list[FieldSpec]:
    defaults = Settings().model_dump(mode="json")
    specs: list[FieldSpec] = []
    for section in sections():
        model_cls = Settings.model_fields[section].annotation
        assert isinstance(model_cls, type) and issubclass(model_cls, BaseModel)
        for name, info in model_cls.model_fields.items():
            annotation, optional = _unwrap_optional(info.annotation)
            kind, choices = _kind(annotation)
            lo, hi = _bounds(info.metadata)
            editable = False
            if f"{section}.{name}" == "model.id":
                kind, choices, editable = "choice", MODEL_CHOICES, True
            elif f"{section}.{name}" == "decoding.language":
                kind, choices, editable = "choice", LANGUAGE_CHOICES, True
            specs.append(
                FieldSpec(
                    section=section,
                    name=name,
                    kind=kind,
                    default=defaults[section][name],
                    description=info.description or "",
                    choices=choices,
                    optional=optional,
                    minimum=lo,
                    maximum=hi,
                    editable_choices=editable,
                )
            )
    return specs


def basic_specs() -> list[FieldSpec]:
    by_path = {s.path: s for s in field_specs()}
    return [by_path[p] for p in BASIC_FIELDS]


def mode_note(spec: FieldSpec, mode: str) -> str | None:
    """Why a field has no (full) effect in ``mode``; None when it works normally."""
    support = mode_support(spec.path, mode)
    return {
        "ignored": "no effect in this mode",
        "first_only": "only the first value is used in this mode",
        "report_only": "reported in the output, does not filter in this mode",
    }.get(support)


def is_disabled(spec: FieldSpec, mode: str) -> bool:
    return mode_support(spec.path, mode) == "ignored"


def get_value(settings: Settings, path: str) -> Any:
    section, _, name = path.partition(".")
    return settings.model_dump(mode="json")[section][name]


def with_value(settings: Settings, path: str, value: Any) -> Settings:
    """A validated copy with one field changed (raises pydantic ``ValidationError``)."""
    data = settings.model_dump(mode="json")
    section, _, name = path.partition(".")
    data[section][name] = value
    return Settings.model_validate(data)


def settings_diff(base: Settings, current: Settings) -> dict[str, Any]:
    """Nested ``{section: {field: value}}`` of what ``current`` changes against ``base``."""
    a, b = base.model_dump(mode="json"), current.model_dump(mode="json")
    out: dict[str, Any] = {}
    for section in sections():
        changed = {k: v for k, v in b[section].items() if a[section].get(k) != v}
        if changed:
            out[section] = changed
    return out


def parse_text(spec: FieldSpec, text: str) -> Any:
    """Value of a text input for ``spec`` (raises ValueError on bad input)."""
    text = text.strip()
    if spec.optional and text == "":
        return None
    if spec.kind == "int":
        return int(text)
    if spec.kind == "float":
        return float(text.replace(",", "."))
    if spec.kind == "floats":
        return [float(x.replace(",", ".")) for x in text.replace(";", " ").split() if x]
    return text


def format_value(spec: FieldSpec, value: Any) -> str:
    if value is None:
        return ""
    if spec.kind == "floats":
        return " ".join(f"{v:g}" for v in value)
    return str(value)


def changed_paths(diff: Mapping[str, Any]) -> list[str]:
    return [f"{sec}.{name}" for sec, fields in diff.items() for name in fields]
