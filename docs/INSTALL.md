# Installing Local TTRPG Recorder

Local TTRPG Recorder runs from a clone of the repository with [uv](https://docs.astral.sh/uv/),
which installs the exact package versions in `uv.lock`. The same steps run on clean
Windows and Ubuntu machines in CI (`.github/workflows/install.yml`). See
[SUPPORTED.md](SUPPORTED.md) for what has been tested.

Run the commands in a terminal: on Windows in **PowerShell** or **Windows Terminal**, on
Linux in any shell. Every `uv run …` command runs from the cloned `local-ttrpg-recorder` folder
(the one with `pyproject.toml`), so `cd` into it first.

## 1. Prerequisites

| | Windows 10/11 | Linux (Ubuntu 22.04/24.04 or similar) |
|---|---|---|
| Git | [git-scm.com](https://git-scm.com/download/win) | `sudo apt install git` |
| uv | `powershell -c "irm https://astral.sh/uv/install.ps1 \| iex"` | `curl -LsSf https://astral.sh/uv/install.sh \| sh` |
| Python | installed by uv (3.11 or 3.12) | installed by uv (3.11 or 3.12) |
| System libraries | – | `sudo apt install libportaudio2 libegl1 libxkbcommon0 libgl1` |
| NVIDIA GPU (optional) | NVIDIA driver R525 or newer (a current driver is recommended) | NVIDIA driver R525 or newer (a current driver is recommended) |

You do **not** need to install the CUDA Toolkit or cuDNN: the `cuda` extra brings the
libraries ctranslate2 needs (cuBLAS 12, cuDNN 9) as Python packages. A system CUDA
install does no harm. Without an NVIDIA GPU everything runs on the CPU, much slower.

**Driver version.** R525 is the minimum NVIDIA lists for CUDA 12.x minor version
compatibility, the same on Windows and Linux; newer drivers (R580 and later included) run
CUDA 12 applications through backward compatibility. Under minor version compatibility
some features are limited and code that needs PTX compilation does not run, so if
`doctor` reports a CUDA error on an older driver, install the current driver (the
`cuda` extra currently brings CUDA 12.9 libraries, whose own driver generation is
R575/R576). Sources: NVIDIA
[CUDA minor version compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
and the [CUDA Toolkit release notes](https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html)
(checked 2026-10-31).

## 2. Install

```bash
git clone https://github.com/htdlord/local-ttrpg-recorder.git
cd local-ttrpg-recorder
# only after v0.1.0 is released (the tag does not exist yet); without it you get
# the latest development state on main:
git checkout v0.1.0
uv sync --locked --extra gui --extra cuda
```

- Leave out `--extra cuda` on a machine without an NVIDIA GPU.
- Leave out `--extra gui` if you only want the command line.

The environment takes about 3 GB of disk with both extras (the NVIDIA libraries are 2 GB of that, the GUI about 0.65 GB).

## 3. Check the installation

```bash
uv run ttrpg doctor
```

`doctor` prints one line per check:

| Check | What it means |
|---|---|
| `platform`, `packages` | Python and the pinned library versions |
| `cuda-libraries` | where cuBLAS and cuDNN were found (`pip` = the `cuda` extra) |
| `compute` | the compute types ctranslate2 supports on CPU and GPU |
| `inference` | a real short transcription with the `tiny` model (downloads it once, ~75 MB) |
| `audio` | the audio host APIs and input devices; with `--host-api asio --channels 1-8` it also opens your interface |

Any `FAIL` line says what is missing. Then try it without recording anything:

```bash
uv run ttrpg demo                 # a 2-minute, 4-player demo session (see DEMO.md)
uv run ttrpg transcribe --session demo
uv run ttrpg-gui                  # desktop app
```

The first real transcription downloads the Whisper model (`large-v3`, about 3 GB) into the
Hugging Face cache. After that, set `model.offline = true` (or pass `--offline`) and
nothing goes over the network any more.

## 4. Audio interfaces

**Windows.** Prefer the interface's **ASIO** driver (`--host-api asio`): all channels in
one stream with low latency. Local TTRPG Recorder enables ASIO in the `sounddevice` package from
PyPI; do not put a custom `portaudio.dll` on `PATH`. Only one program can use an ASIO
driver at a time, so close the DAW or the vendor's control software. **WASAPI**
(`--host-api wasapi`) is the fallback; many interfaces offer a multi-channel WASAPI
endpoint (e.g. `IN 01-08` on the Behringer UMC1820).

**Linux.** ALSA, PulseAudio and PipeWire devices are listed by `ttrpg record
devices`. Your user needs access to the audio devices (on most distributions: the `audio`
group or a desktop session). Multi-channel interfaces work best through ALSA directly or
PipeWire's pro-audio profile.

Save the device once in a rig file and use it everywhere (see the README).

## 5. Update

```bash
git pull                     # or: git fetch --tags && git checkout v0.x.y
uv sync --locked --extra gui --extra cuda
```

**Do not update while a recording is running**: stop it first (the recorder process uses
the installed package).

Settings files and sessions keep working: they carry a schema version and are migrated
when read.

**Coming from SessionScribe** (the project's name until 2026-10-31): GitHub redirects the
old repository URL, but point your clone at the new one, then sync:

```bash
git remote set-url origin https://github.com/htdlord/local-ttrpg-recorder.git
git pull
uv sync --locked --extra gui --extra cuda
```

The commands are now `ttrpg` and `ttrpg-gui`; `sessionscribe` and `sessionscribe-gui`
still work with a warning until after v0.1.0. On first start your presets
(`%APPDATA%\sessionscribe` or `~/.config/sessionscribe`) are copied to the new
`local-ttrpg-recorder` folder and the desktop app's preferences (theme, language, recent
sessions) are carried over; existing sessions and their caches stay valid.

## 6. Uninstall

Delete the cloned folder. Downloaded models stay in the Hugging Face cache
(`%USERPROFILE%\.cache\huggingface` on Windows, `~/.cache/huggingface` on Linux, or the
folder set in `model.model_dir`); delete them there if you no longer need them. Your
recordings and sessions are wherever you put them.

## Troubleshooting

| Symptom | What to do |
|---|---|
| `cuda-libraries` FAIL, `cublas64_12.dll` / `libcublas.so.12` not found | `uv sync --locked --extra cuda` |
| `inference` FAIL on CUDA, driver errors | update the NVIDIA driver; `doctor --device cpu` confirms the rest works |
| `compute` shows no CUDA device | no NVIDIA GPU or driver; transcription falls back to the CPU |
| ASIO device missing or "Failed to load ASIO driver" | another program uses the ASIO driver; close it, or use WASAPI |
| fewer channels than expected | the device or host API offers fewer channels; `record devices` lists them per host API |
| GUI does not start on Linux (`xcb` / `EGL` errors) | install the system libraries from step 1 |
| `uv sync --locked` says the lock file is out of date | you edited `pyproject.toml`; run `git checkout pyproject.toml uv.lock` or `uv lock` |
