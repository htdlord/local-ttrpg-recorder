"""Settings folder and window preferences carried over from SessionScribe."""

import os
import shutil
import sys

import pytest

from ttrpg_recorder.core import config


@pytest.fixture
def base(tmp_path, monkeypatch):
    monkeypatch.setenv("APPDATA" if sys.platform == "win32" else "XDG_CONFIG_HOME", str(tmp_path))
    return tmp_path


def preset(folder, name="mine"):
    (folder / "presets").mkdir(parents=True)
    (folder / "presets" / f"{name}.toml").write_text('[decoding]\nlanguage = "hu"\n')


def test_new_folder_without_an_old_one(base):
    assert config.user_config_dir() == base / "local-ttrpg-recorder"
    assert not (base / "local-ttrpg-recorder").exists()  # created only when written


def test_old_folder_is_copied_once_and_kept(base):
    preset(base / "sessionscribe")
    assert config.user_config_dir() == base / "local-ttrpg-recorder"
    assert (base / "local-ttrpg-recorder" / "presets" / "mine.toml").is_file()
    assert (base / "sessionscribe" / "presets" / "mine.toml").is_file()
    assert "mine" in config.list_presets()


def test_an_existing_new_folder_is_not_overwritten(base):
    preset(base / "sessionscribe", "old")
    preset(base / "local-ttrpg-recorder", "new")
    config.user_config_dir()
    assert not (base / "local-ttrpg-recorder" / "presets" / "old.toml").exists()


def test_failed_copy_reads_the_old_folder(base, monkeypatch):
    preset(base / "sessionscribe")

    def broken(src, dst, **kw):
        os.makedirs(dst)
        raise OSError("denied")

    monkeypatch.setattr(shutil, "copytree", broken)
    assert config.user_config_dir() == base / "sessionscribe"
    assert not (base / "local-ttrpg-recorder").exists()  # no half copy left behind
    assert "mine" in config.list_presets()


def test_window_preferences_are_copied_into_empty_ones(tmp_path):
    pytest.importorskip("PySide6")
    from PySide6.QtCore import QSettings

    from ttrpg_recorder.gui.prefs import migrate_prefs

    def ini(name):
        return QSettings(str(tmp_path / f"{name}.ini"), QSettings.Format.IniFormat)

    old = ini("old")
    old.setValue("theme", "dark")
    old.setValue("recent", ["a/session.toml", "b/session.toml"])
    old.setValue("recording/rig", "rig.toml")
    new = ini("new")
    assert migrate_prefs(new, old)
    assert new.value("theme") == "dark" and new.value("recording/rig") == "rig.toml"
    assert list(new.value("recent")) == ["a/session.toml", "b/session.toml"]
    new.setValue("theme", "light")
    assert not migrate_prefs(new, old)  # only into empty preferences
    assert new.value("theme") == "light"
