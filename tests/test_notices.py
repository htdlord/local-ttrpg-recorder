"""THIRD_PARTY_NOTICES.md lists every runtime dependency and extra of pyproject.toml."""

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _key(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def test_notices_cover_every_runtime_dependency_and_extra():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    wanted = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        wanted += extra
    names = {_key(re.split(r"[\s<>=!~;\[]", spec, maxsplit=1)[0]) for spec in wanted}
    notices = (ROOT / "THIRD_PARTY_NOTICES.md").read_text(encoding="utf-8")
    listed = {_key(m) for m in re.findall(r"^\| ([^|\s]+) \| [^|]+ \|", notices, re.MULTILINE)}
    missing = sorted(names - listed)
    assert not missing, f"regenerate with scripts/third_party_notices.py; missing: {missing}"
