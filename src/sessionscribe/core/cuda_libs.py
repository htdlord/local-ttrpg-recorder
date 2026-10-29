"""Make CUDA runtime libraries (cuBLAS, cuDNN 9) visible to ctranslate2 and report them.

The optional ``cuda`` extra installs them as pip packages under ``site-packages/nvidia``.
They must be registered before ``ctranslate2`` is imported: on Windows the folders are
added with ``os.add_dll_directory`` and to ``PATH`` (cuDNN loads its sub-libraries by
name); on Linux the shared objects are preloaded with ``RTLD_GLOBAL``.

Libraries may also come from the ctranslate2 wheel itself or from a system CUDA/cuDNN
install; the report says where each file was found.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)

# Files each package must provide. cuDNN 9 loads its sub-libraries lazily, so a missing
# sub-library only shows up as a crash at inference time; check them explicitly.
REQUIRED_FILES: dict[str, dict[str, tuple[str, ...]]] = {
    "win32": {
        "cublas": ("cublas64_12.dll", "cublasLt64_12.dll"),
        "cudnn": ("cudnn64_9.dll", "cudnn_graph64_9.dll", "cudnn_ops64_9.dll", "cudnn_cnn64_9.dll"),
    },
    "linux": {
        "cublas": ("libcublas.so.12", "libcublasLt.so.12"),
        "cudnn": ("libcudnn.so.9", "libcudnn_graph.so.9", "libcudnn_ops.so.9", "libcudnn_cnn.so.9"),
    },
}
PACKAGES = ("cublas", "cudnn")
INSTALL_HINT = "install with: uv sync --extra cuda"


@dataclass(frozen=True)
class CudaLibraryStatus:
    package: str  # "cublas" | "cudnn"
    locations: tuple[str, ...]  # "<file>: <where>" for each file found
    missing: tuple[str, ...]  # files not found anywhere
    error: str | None = None  # load error of the primary library

    @property
    def ok(self) -> bool:
        return not self.missing and self.error is None


_prepared: tuple[CudaLibraryStatus, ...] | None = None
_dll_dir_handles: list[Any] = []  # keep add_dll_directory handles alive


def _platform_key() -> str:
    return "win32" if sys.platform == "win32" else "linux"


def library_dir(package_root: Path, platform_key: str) -> Path:
    """Folder holding the shared libraries inside an ``nvidia.<pkg>`` package."""
    return package_root / ("bin" if platform_key == "win32" else "lib")


def _find_package_root(name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return Path(next(iter(spec.submodule_search_locations)))
    return Path(spec.origin).parent if spec.origin else None


def pip_library_dirs(platform_key: str) -> list[Path]:
    """Library folders of the pip-installed ``nvidia.*`` packages that exist."""
    dirs = []
    for package in PACKAGES:
        root = _find_package_root(f"nvidia.{package}")
        if root is not None and library_dir(root, platform_key).is_dir():
            dirs.append(library_dir(root, platform_key))
    return dirs


def search_dirs(platform_key: str) -> list[tuple[Path, str]]:
    """Folders searched for CUDA libraries, in priority order, with a source label."""
    dirs = [(d, "pip") for d in pip_library_dirs(platform_key)]
    ct2 = _find_package_root("ctranslate2")
    if ct2 is not None:
        dirs.append((ct2, "ctranslate2 wheel"))
        dirs.append((ct2.parent / "ctranslate2.libs", "ctranslate2 wheel"))
    if platform_key == "win32":
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path:
            dirs.append((Path(cuda_path) / "bin", "CUDA_PATH"))
        dirs += [(Path(p), "PATH") for p in os.environ.get("PATH", "").split(os.pathsep) if p]
    else:
        dirs += [
            (Path(p), "LD_LIBRARY_PATH")
            for p in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)
            if p
        ]
    return dirs


def locate(filename: str, dirs: list[tuple[Path, str]]) -> tuple[Path, str] | None:
    for folder, label in dirs:
        candidate = folder / filename
        if candidate.is_file():
            return candidate, label
    return None


def shared_objects(folder: Path) -> list[Path]:
    """Linux shared objects in ``folder`` (``lib*.so*``), sorted for determinism."""
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.name.startswith("lib") and ".so" in p.name)


def _preload_linux(folder: Path) -> None:
    # Load order matters (dependencies between the libraries); retry until no progress.
    pending = shared_objects(folder)
    while pending:
        failed = []
        for path in pending:
            try:
                ctypes.CDLL(str(path), mode=ctypes.RTLD_GLOBAL)
            except OSError:
                failed.append(path)
        if len(failed) == len(pending):
            logger.debug("could not preload: %s", [p.name for p in failed])
            break
        pending = failed


def _register_windows(folder: Path) -> None:
    if sys.platform == "win32":
        _dll_dir_handles.append(os.add_dll_directory(str(folder)))
    os.environ["PATH"] = str(folder) + os.pathsep + os.environ.get("PATH", "")


def _load(path_or_name: str) -> str | None:
    """Load a library; return an error string on failure."""
    try:
        if sys.platform == "win32":
            ctypes.WinDLL(path_or_name)
        else:
            ctypes.CDLL(path_or_name, mode=ctypes.RTLD_GLOBAL)
    except OSError as exc:
        return str(exc)
    return None


def check_package(
    package: str, platform_key: str, dirs: list[tuple[Path, str]], *, load: bool = True
) -> CudaLibraryStatus:
    locations = []
    missing = []
    primary: str | None = None
    for filename in REQUIRED_FILES[platform_key][package]:
        found = locate(filename, dirs)
        if found is not None:
            locations.append(f"{filename}: {found[1]} ({found[0].parent})")
            primary = primary or str(found[0])
        elif platform_key == "linux" and load and _load(filename) is None:
            locations.append(f"{filename}: system linker path")
            primary = primary or filename
        else:
            missing.append(filename)
    error = _load(primary) if load and primary is not None and not missing else None
    return CudaLibraryStatus(package, tuple(locations), tuple(missing), error)


def prepare_cuda_libraries() -> tuple[CudaLibraryStatus, ...]:
    """Register pip-installed CUDA libraries and report what is available. Idempotent."""
    global _prepared
    if _prepared is not None:
        return _prepared
    key = _platform_key()
    for folder in pip_library_dirs(key):
        if key == "win32":
            _register_windows(folder)
        else:
            _preload_linux(folder)
    dirs = search_dirs(key)
    _prepared = tuple(check_package(p, key, dirs) for p in PACKAGES)
    return _prepared


def import_ctranslate2() -> ModuleType:
    """Import ctranslate2 after the CUDA libraries have been registered."""
    prepare_cuda_libraries()
    import ctranslate2

    return ctranslate2
