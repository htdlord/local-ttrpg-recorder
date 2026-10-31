"""The participant table of the Session tab as data (no Qt): rows in, validated
``SessionConfig`` (or per-row errors) out."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from ttrpg_recorder.core.session import Participant, SessionConfig

COLUMNS = ("id", "name", "source", "channel", "offset_s", "color")


@dataclass
class ParticipantRow:
    id: str = ""
    name: str = ""
    source: str = ""
    channel: str = ""  # text in the table; "" = mono file / folder
    offset_s: str = "0"
    color: str = ""

    @classmethod
    def from_participant(cls, p: Participant) -> ParticipantRow:
        return cls(
            id=p.id,
            name=p.name,
            source=p.source,
            channel="" if p.channel is None else str(p.channel),
            offset_s=f"{p.offset_s:g}",
            color=p.color or "",
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.id.strip(),
            "name": self.name.strip(),
            "source": self.source.strip(),
        }
        if self.channel.strip():
            out["channel"] = self.channel.strip()
        out["offset_s"] = self.offset_s.strip().replace(",", ".") or "0"
        if self.color.strip():
            out["color"] = self.color.strip()
        return out


@dataclass
class Validation:
    config: SessionConfig | None
    row_errors: dict[int, list[str]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.config is not None


def validate(base: SessionConfig, rows: list[ParticipantRow]) -> Validation:
    """``base`` with its participants replaced by ``rows``; errors mapped to rows."""
    data = base.model_dump(mode="json")
    data["participants"] = [r.to_dict() for r in rows]
    try:
        return Validation(SessionConfig.model_validate(data))
    except ValidationError as exc:
        result = Validation(None)
        for err in exc.errors():
            loc = err["loc"]
            if len(loc) >= 2 and loc[0] == "participants" and isinstance(loc[1], int):
                column = str(loc[2]) if len(loc) > 2 else ""
                result.row_errors.setdefault(loc[1], []).append(
                    f"{column}: {err['msg']}" if column else err["msg"]
                )
            else:
                result.errors.append(str(err["msg"]).removeprefix("Value error, "))
        return result
