from sessionscribe.core.cuda_libs import (
    REQUIRED_FILES,
    check_package,
    library_dir,
    locate,
    shared_objects,
)


def touch(folder, *names):
    folder.mkdir(parents=True, exist_ok=True)
    for n in names:
        (folder / n).write_bytes(b"")


def test_locate_respects_priority(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    touch(a, "x.dll")
    touch(b, "x.dll")
    assert locate("x.dll", [(a, "pip"), (b, "PATH")]) == (a / "x.dll", "pip")
    assert locate("y.dll", [(a, "pip")]) is None


def test_missing_cudnn_sublibrary_is_reported(tmp_path):
    files = REQUIRED_FILES["win32"]["cudnn"]
    touch(tmp_path / "ct2", files[0])  # the wheel ships only the main DLL
    touch(tmp_path / "sys", *files[1:3])  # cudnn_cnn missing
    dirs = [(tmp_path / "ct2", "ctranslate2 wheel"), (tmp_path / "sys", "PATH")]
    status = check_package("cudnn", "win32", dirs, load=False)
    assert not status.ok
    assert status.missing == ("cudnn_cnn64_9.dll",)
    assert any("ctranslate2 wheel" in loc for loc in status.locations)


def test_all_found(tmp_path):
    touch(tmp_path, *REQUIRED_FILES["linux"]["cublas"])
    status = check_package("cublas", "linux", [(tmp_path, "pip")], load=False)
    assert status.ok and len(status.locations) == 2


def test_shared_objects_filter(tmp_path):
    touch(tmp_path, "libcudnn.so.9", "libcudnn_ops.so.9", "README", "__init__.py")
    assert [p.name for p in shared_objects(tmp_path)] == ["libcudnn.so.9", "libcudnn_ops.so.9"]


def test_library_dir_per_platform(tmp_path):
    assert library_dir(tmp_path, "win32") == tmp_path / "bin"
    assert library_dir(tmp_path, "linux") == tmp_path / "lib"
