# Supported configurations

| Level | Meaning |
|---|---|
| **Tested** | run end to end on the reference machine (below) |
| **CI** | installed and tested automatically on GitHub's clean runners, CPU only, no audio hardware |
| **Expected** | should work, not tested by us; reports welcome |
| **Not supported** | known not to work, or out of scope |

Reference machine: Windows 10 Pro, Ryzen 9 5900X (12 cores), RTX 4070 Ti Super (16 GB),
Behringer UMC1820 over ASIO.

## Operating system and Python

| | Level | Notes |
|---|---|---|
| Windows 10 / 11, x64 | Tested (10), CI (Server 2022/2025) | |
| Ubuntu 22.04 / 24.04, x64 | CI | GPU and audio interfaces: Expected |
| Other Linux distributions, x64 | Expected | install the system libraries named in [INSTALL.md](INSTALL.md) |
| macOS | Not supported | no CUDA; not tested |
| Python 3.11, 3.12 | Tested (3.12), CI (both) | uv installs it |
| Python 3.13+ | Expected | not in the lock-tested matrix yet |

## Transcription hardware

Measured on the reference machine (M1 benchmark, Hungarian clip; DECISIONS 2026-10-29).
VRAM is the extra device memory used while transcribing.

| Model, compute type, mode | VRAM | Speed (10 h of speech) | Level |
|---|---|---|---|
| large-v3, float16, quality (default) | ~4.3 GB | ~1 h 45 min | Tested |
| large-v3, float16, fast (batched) | ~5.1–5.5 GB | ~40–45 min | Tested |
| large-v3, int8_float16, quality | ~2.6 GB | ~2 h 20 min | Tested |
| large-v3-turbo, float16, quality | ~2.3 GB | ~30 min | Tested (clearly worse on Hungarian) |

| GPU | Level |
|---|---|
| NVIDIA RTX 40 series, 16 GB | Tested |
| Other NVIDIA GPUs with CUDA 12 support and ≥ 6 GB VRAM | Expected (`doctor` lists the supported compute types) |
| Smaller NVIDIA GPUs | Expected with `int8_float16` or a smaller model; fast mode halves the batch on out-of-memory |
| CPU only | CI (tiny model); real sessions work but take many times longer than on a GPU |
| AMD / Intel GPUs | Not supported (ctranslate2 uses CUDA) |

CUDA libraries: the `cuda` extra alone is enough. On the reference machine, with the
system CUDA 12.8 and cuDNN 9 hidden from the environment, `doctor` ran a real CUDA
inference with cuBLAS and cuDNN loaded from the extra
(`scripts/clean_env_check.py`, DECISIONS 2026-10-31).

## Recording

| | Level | Notes |
|---|---|---|
| ASIO (Windows), Behringer UMC1820, 8 channels, 48 kHz / 24 bit | Tested | 8 h recording without loss; 30 min probe with no overflow |
| WASAPI shared (Windows), UMC1820 `IN 01-08` endpoint | Tested (30 min probe) | recording tests with WASAPI still open |
| Other ASIO / WASAPI interfaces | Expected | `record devices` shows what each host API offers |
| ALSA / PulseAudio / PipeWire (Linux) | Expected | |
| Up to 8 participants (tracks) | Tested (designed for 8) | more are possible but not designed for |
| Sessions up to 8 h | Tested | segmented FLAC (default) or RF64; classic WAV is not used (4 GB limit) |

## Languages and models

| | Level |
|---|---|
| Hungarian, English with `large-v3` | Tested |
| Other Whisper languages | Expected (set `decoding.language`) |
| `large-v3-turbo`, `medium`, `small` | Expected; noticeably less accurate on Hungarian |
| English-only models (`*.en`, `distil-*`) with a non-English language | Not supported (rejected by the settings check) |

## Known limits

- **Fast (batched) mode** ignores some anti-hallucination options and the temperature
  fallback; the settings check warns about options without effect in the chosen mode.
- **Bleed handling** is conservative: when unsure it keeps the segment and flags it, so
  some duplicates remain for review rather than real speech being lost.
- The desktop app is English and Hungarian.
