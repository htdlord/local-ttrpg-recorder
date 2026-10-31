"""A demo session to try Local TTRPG Recorder without recording anything.

Four fictional players speak in turns; each voice is a short stretch of a public-domain
LibriVox reading (Short Story Collection Vol. 014, read by four different volunteers).
Every track also carries the other voices as bleed with a known coupling, there is one
true overlap, one track starts late (offset) and the session opens with a guided mic
check (each player alone), so bleed handling, overlap marks and guided coupling all show.

Only the first ``RANGE_BYTES`` of each MP3 are downloaded (HTTP Range) and checked
against a pinned SHA-256; the downloads are kept in ``<folder>/sources/`` so a second
run works offline.
"""

from __future__ import annotations

import hashlib
import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

RATE = 48_000
RANGE_BYTES = 1_500_000  # about 3 minutes of each 64 kbit/s MP3
SKIP_S = 45.0  # past the LibriVox introduction
LANGUAGE = "en"
COLLECTION = "https://librivox.org/short-story-collection-vol-14/"
ARCHIVE = "https://archive.org/download/short_stories14_librivox/"


@dataclass(frozen=True)
class DemoVoice:
    id: str
    name: str  # fictional player
    title: str
    author: str
    reader: str
    file: str
    sha256: str  # of the first RANGE_BYTES

    @property
    def url(self) -> str:
        return ARCHIVE + self.file


VOICES: tuple[DemoVoice, ...] = (
    DemoVoice(
        "gm",
        "Game Master",
        "Trusty John (The Blue Fairy Book)",
        "Andrew Lang, ed. (1844-1912)",
        "Clarica",
        "trusty_john_lang_cjrg_64kb.mp3",
        "8fa556d14afc87871a6bd7b90dee5b1ded6febf10256d600a26881b442a9256b",
    ),
    DemoVoice(
        "bodor",
        "Bodor",
        "The Four-Fifteen Express",
        "Amelia B. Edwards (1831-1892)",
        "Paul Allerton",
        "four_fifteen_express_edwards_pra_64kb.mp3",
        "8afb227b8adba760333e266c2067392b1a5ebe4b1247d9eb063c18cc190518f2",
    ),
    DemoVoice(
        "kira",
        "Kira",
        "The History of England",
        "Jane Austen (1775-1817)",
        "Kiki Baessell",
        "history_of_england_austen_kp_64kb.mp3",
        "f0f812f005524f90026d67ee3615aed118c1dbff389a4f0b1aadc3c55cda4903",
    ),
    DemoVoice(
        "lin",
        "Lin",
        "Ningyo-no-Haka",
        "Lafcadio Hearn (1850-1904)",
        "David Barnes",
        "ningyo-no-haka_hearn_drb_64kb.mp3",
        "0e8be600524a3e1241bea806ef7f067c3b6caf8771a918f34528564ef9cae1ca",
    ),
)

# Level of voice j on mic i relative to mic j (dB); diagonal unused.
# Bleed copies are also duller and reverberant (see ``room``); with clean, louder copies
# Whisper transcribes nearly every bleed word, which is not what lavaliers give.
COUPLING_DB = (
    (0.0, -30.0, -34.0, -38.0),
    (-28.0, 0.0, -32.0, -36.0),
    (-34.0, -29.0, 0.0, -31.0),
    (-38.0, -35.0, -27.0, 0.0),
)
NOISE_DBFS = -52.0  # room tone
VOICE_RMS_DBFS = -24.0
LATE_START_S = {"lin": 0.75}  # this mic was switched on late: track offset
MIC_CHECK_S = 4.0


@dataclass(frozen=True)
class Line:
    speaker: int
    target_s: float  # wanted length; the cut moves to the nearest pause
    gap_s: float  # silence before it; negative = starts before the previous line ends


@dataclass(frozen=True)
class Placed:
    speaker: int
    start: int  # session samples
    end: int


def script(n: int = 4, seed: int = 14) -> list[Line]:
    """Mic check (each speaker alone), then about 90 s of turns with one overlap."""
    rng = np.random.default_rng(seed)
    lines = [Line(i, MIC_CHECK_S, 1.0) for i in range(n)]
    previous = n - 1
    for k in range(18):
        speaker = int(rng.choice([s for s in range(n) if s != previous]))
        if k % 5 == 0:
            speaker = 0  # the game master speaks more often
            if speaker == previous:
                speaker = 1
        length = float(rng.uniform(2.5, 7.5))
        gap = float(rng.uniform(0.3, 0.9))
        if k == 7:
            gap = -1.2  # true overlap: two players at once
        if k == 12:
            length = 1.0  # a short interjection
        lines.append(Line(speaker, length, gap))
        previous = speaker
    return lines


def _frame_rms(x: np.ndarray, frame: int) -> np.ndarray:
    n = len(x) // frame
    return np.sqrt(np.mean(x[: n * frame].reshape(n, frame).astype(np.float64) ** 2, axis=1))


def quiet_point(voice: np.ndarray, around: int, reach: int, rate: int = RATE) -> int:
    """Sample index of the quietest 20 ms frame within ``around ± reach``."""
    frame = rate // 50
    lo = max(0, around - reach)
    hi = min(len(voice), around + reach)
    if hi - lo < frame:
        return min(max(around, 0), len(voice))
    rms = _frame_rms(voice[lo:hi], frame)
    return lo + int(np.argmin(rms)) * frame + frame // 2


def assemble(
    voices: list[np.ndarray],
    lines: list[Line],
    coupling_db: tuple[tuple[float, ...], ...] | np.ndarray = COUPLING_DB,
    *,
    rate: int = RATE,
    skip_s: float = SKIP_S,
    noise_dbfs: float = NOISE_DBFS,
    seed: int = 14,
) -> tuple[list[np.ndarray], list[Placed]]:
    """Mix ``lines`` into one track per speaker: own voice + bleed of the others + noise.

    Each speaker's audio is read in order from ``voices[i]`` starting at ``skip_s``; cuts
    move to the quietest point near the wanted length so words are not split. Returns the
    tracks (float32, session time) and where each line was placed.
    """
    n = len(voices)
    coupling = np.asarray(coupling_db, dtype=np.float64)
    cursors = [quiet_point(v, round(skip_s * rate), rate // 2, rate) for v in voices]
    fade = rate // 100
    pieces: list[tuple[Placed, np.ndarray]] = []
    t = 0
    for line in lines:
        voice = voices[line.speaker]
        start = cursors[line.speaker]
        end = quiet_point(voice, start + round(line.target_s * rate), rate * 3 // 4, rate)
        end = max(end, start + rate // 2)
        piece = voice[start:end].astype(np.float32).copy()
        if len(piece) == 0:
            raise ValueError(f"voice {line.speaker} is too short for the script")
        ramp = np.linspace(0.0, 1.0, min(fade, len(piece) // 2), dtype=np.float32)
        piece[: len(ramp)] *= ramp
        piece[len(piece) - len(ramp) :] *= ramp[::-1]
        cursors[line.speaker] = end
        at = max(0, t + round(line.gap_s * rate))
        placed = Placed(line.speaker, at, at + len(piece))
        pieces.append((placed, piece))
        t = max(t, placed.end)
    total = t + rate
    dry = [np.zeros(total, np.float32) for _ in range(n)]
    for placed, piece in pieces:
        dry[placed.speaker][placed.start : placed.end] += piece
    rng = np.random.default_rng(seed)
    # what another mic hears: duller and with a bit of room (computed once per voice)
    far = [room(d, rate, rng) for d in dry]
    tracks = []
    for i in range(n):
        track = dry[i].copy()
        for j in range(n):
            if j != i:
                delay = 24 * (abs(i - j) + 1)  # 0.5 ms per mic distance step
                gain = np.float32(10 ** (coupling[i, j] / 20))
                track[delay:] += gain * far[j][: total - delay]
        track += room_tone(total, noise_dbfs, rng)
        tracks.append(track)
    return tracks, [p for p, _ in pieces]


def room(x: np.ndarray, rate: int, rng: np.random.Generator) -> np.ndarray:
    """``x`` as heard across a table: low-passed (about 3 kHz) with a short decaying tail."""
    tail_n = round(0.25 * rate)
    t = np.arange(tail_n) / rate
    ir = rng.normal(0.0, 1.0, tail_n) * np.exp(-t / 0.06) * 0.08
    ir[0] = 1.0
    size = 1 << int(np.ceil(np.log2(len(x) + tail_n)))
    freqs = np.fft.rfftfreq(size, 1 / rate)
    lowpass = 1 / np.sqrt(1 + (freqs / 3000.0) ** 4)
    spectrum = np.fft.rfft(x, size) * np.fft.rfft(ir, size) * lowpass
    out = np.fft.irfft(spectrum, size)[: len(x)]
    level = np.sqrt(np.mean(x.astype(np.float64) ** 2)) / max(np.sqrt(np.mean(out**2)), 1e-12)
    return (out * level).astype(np.float32)


def room_tone(n: int, dbfs: float, rng: np.random.Generator) -> np.ndarray:
    """Pinkish background noise at ``dbfs`` RMS."""
    size = 1 << int(np.ceil(np.log2(max(n, 2))))
    spectrum = np.fft.rfft(rng.normal(0.0, 1.0, size))
    spectrum[1:] /= np.sqrt(np.arange(1, len(spectrum)))
    spectrum[0] = 0
    noise = np.fft.irfft(spectrum, size)[:n]
    noise *= 10 ** (dbfs / 20) / max(float(np.sqrt(np.mean(noise**2))), 1e-12)
    return noise.astype(np.float32)


# --- sources ----------------------------------------------------------------------------


def fetch(
    voice: DemoVoice, folder: Path, *, offline: bool = False, timeout_s: float = 60.0
) -> Path:
    """The first ``RANGE_BYTES`` of the voice's MP3, downloaded once and checked."""
    path = folder / voice.file
    if path.is_file() and _sha256(path) == voice.sha256:
        return path
    if offline:
        raise FileNotFoundError(f"{path} is missing (offline)")
    folder.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        voice.url,
        headers={"Range": f"bytes=0-{RANGE_BYTES - 1}", "User-Agent": "local-ttrpg-recorder"},
    )
    data = _download(request, timeout_s)
    if hashlib.sha256(data).hexdigest() != voice.sha256:
        raise ValueError(f"{voice.file}: checksum mismatch (the archive copy changed?)")
    tmp = path.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return path


def _download(request: urllib.request.Request, timeout_s: float, attempts: int = 4) -> bytes:
    """archive.org answers 5xx now and then: retry those and network errors."""
    import time
    import urllib.error

    for attempt in range(1, attempts + 1):
        logger.info("downloading %s (attempt %d)", request.full_url, attempt)
        try:
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                return bytes(response.read(RANGE_BYTES))
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == attempts:
                raise
        except urllib.error.URLError:
            if attempt == attempts:
                raise
        time.sleep(2.0 * attempt)
    raise AssertionError("unreachable")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(path: Path, rate: int = RATE) -> np.ndarray:
    """Mono float32 at ``rate``; a truncated last MP3 frame is ignored."""
    import av

    out: list[np.ndarray] = []
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="flt", layout="mono", rate=rate)
        try:
            for frame in container.decode(audio=0):
                out.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(frame))
        except av.error.InvalidDataError:
            pass  # the cut end of a ranged download
        out.extend(f.to_ndarray().reshape(-1) for f in resampler.resample(None))
    return np.concatenate(out).astype(np.float32)


def _normalise(voice: np.ndarray, rate: int = RATE) -> np.ndarray:
    rms = _frame_rms(voice, rate // 50)
    active = rms[rms > np.max(rms) * 0.05]
    level = float(np.sqrt(np.mean(active**2))) if len(active) else 1.0
    return (voice * np.float32(10 ** (VOICE_RMS_DBFS / 20) / max(level, 1e-9))).astype(np.float32)


# --- the session --------------------------------------------------------------------------


def create_demo(
    folder: Path,
    *,
    offline: bool = False,
    voices: list[np.ndarray] | None = None,
) -> Path:
    """Write the demo session to ``folder``; returns its ``session.toml``.

    ``voices`` (one array per entry of ``VOICES``, at ``RATE``) skips downloading; tests
    pass synthetic audio.
    """
    import soundfile as sf

    from ttrpg_recorder.core.session import (
        SESSION_FILE,
        Participant,
        SessionConfig,
        write_session,
    )

    folder = Path(folder)
    if (folder / SESSION_FILE).exists():
        raise FileExistsError(f"{folder / SESSION_FILE} exists")
    if voices is None:
        sources = folder / "sources"
        voices = [_normalise(decode(fetch(v, sources, offline=offline))) for v in VOICES]
    tracks, placed = assemble(voices, script(len(VOICES)))
    tracks_dir = folder / "tracks"
    tracks_dir.mkdir(parents=True, exist_ok=True)
    participants = []
    for i, (voice, track) in enumerate(zip(VOICES, tracks, strict=True), start=1):
        late = LATE_START_S.get(voice.id, 0.0)
        name = f"ch{i:02d}_{voice.id}.flac"
        sf.write(
            str(tracks_dir / name),
            np.clip(track[round(late * RATE) :], -1.0, 1.0),
            RATE,
            format="FLAC",
            subtype="PCM_24",
        )
        participants.append(
            Participant(id=voice.id, name=voice.name, source=f"tracks/{name}", offset_s=late)
        )
    mic_check = {
        VOICES[p.speaker].id: (round(p.start / RATE, 3), round(p.end / RATE, 3))
        for p in placed[: len(VOICES)]
    }
    config = SessionConfig(
        name="Local TTRPG Recorder demo",
        settings={"decoding": {"language": LANGUAGE}},
        participants=participants,
        mic_check=mic_check,
    )
    write_session(folder / SESSION_FILE, config)
    (folder / "SOURCES.md").write_text(sources_text(), encoding="utf-8")
    return folder / SESSION_FILE


def sources_text() -> str:
    lines = [
        "# Demo audio sources",
        "",
        "The voices are short excerpts of LibriVox recordings from "
        f"[Short Story Collection Vol. 014]({COLLECTION}). LibriVox recordings are in the "
        "public domain (https://librivox.org/pages/public-domain/); the texts are by authors "
        "who died more than 70 years ago, so they are in the public domain in the EU as well. "
        "The player names are fictional; the bleed, overlap and late start are simulated.",
        "",
        "| Player | Reading | Author | Read by | File |",
        "|---|---|---|---|---|",
    ]
    for v in VOICES:
        lines.append(f"| {v.name} | {v.title} | {v.author} | {v.reader} | [{v.file}]({v.url}) |")
    return "\n".join(lines) + "\n"
