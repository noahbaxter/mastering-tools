"""Shared utilities for mastering analysis tools."""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


BANDS = {
    "Sub": (20, 60),
    "Low": (60, 250),
    "Low-Mid": (250, 500),
    "Mid": (500, 2000),
    "Upper-Mid": (2000, 4000),
    "Presence": (4000, 8000),
    "Air": (8000, 20000),
}

BAND_NAMES = list(BANDS.keys())
BAND_ABBREVS = ["SUB", "LOW", "L-MID", "MID", "U-MID", "PRES", "AIR"]


def load_audio(filepath: str) -> tuple[np.ndarray, int]:
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)
    try:
        data, sr = sf.read(filepath)
        return data, sr
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        sys.exit(1)


def load_audio_mono(filepath: str) -> tuple[np.ndarray, int]:
    data, sr = load_audio(filepath)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return data, sr


def compute_ltas(data: np.ndarray, sr: int, fft_size: int = 4096,
                 hop_ratio: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """Compute Long-Term Average Spectrum using windowed FFT with overlap.

    Returns (freqs, magnitudes_db) where magnitudes_db is the average
    power spectrum in dB.
    """
    hop = int(fft_size * hop_ratio)
    window = np.hanning(fft_size)
    n_frames = max(1, (len(data) - fft_size) // hop + 1)

    # Truncate to exact frame boundary, then reshape into overlapping frames
    # using stride_tricks (no copy — creates a view with overlapping strides)
    from numpy.lib.stride_tricks import as_strided
    needed = (n_frames - 1) * hop + fft_size
    padded = data[:needed] if len(data) >= needed else np.pad(data, (0, needed - len(data)))
    frames = as_strided(padded,
                        shape=(n_frames, fft_size),
                        strides=(hop * padded.strides[0], padded.strides[0]))

    # Batch windowing + FFT — no Python loop
    windowed = frames * window
    spectra = np.fft.rfft(windowed, axis=1)
    power_avg = np.mean(np.abs(spectra) ** 2, axis=0)

    freqs = np.fft.rfftfreq(fft_size, 1.0 / sr)

    # Convert to dB, floor at -120
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitudes_db = 10.0 * np.log10(power_avg + 1e-30)

    return freqs, magnitudes_db


def band_energies(freqs: np.ndarray, magnitudes_db: np.ndarray) -> dict[str, float]:
    """Sum energy per band in dB from LTAS data."""
    result = {}
    for name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs < hi)
        if not np.any(mask):
            result[name] = -120.0
            continue
        # Convert back to linear, sum, convert to dB
        linear = 10.0 ** (magnitudes_db[mask] / 10.0)
        result[name] = 10.0 * np.log10(np.sum(linear) + 1e-30)
    return result


def bandpass_fft(data: np.ndarray, sr: int, low_hz: float,
                 high_hz: float) -> np.ndarray:
    """FFT-based bandpass filter (zero out bins outside range)."""
    n = len(data)
    spectrum = np.fft.rfft(data)
    freqs = np.fft.rfftfreq(n, 1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    spectrum[~mask] = 0
    return np.fft.irfft(spectrum, n=n)


def compute_rms(data: np.ndarray) -> float:
    return float(np.sqrt(np.mean(data ** 2)))


def db(value: float, ref: float = 1.0) -> float:
    if value <= 0 or ref <= 0:
        return -120.0
    return 20.0 * np.log10(value / ref)


def try_import_matplotlib():
    """Returns (plt, None) on success or (None, error_msg) on failure."""
    try:
        import matplotlib.pyplot as plt
        return plt, None
    except ImportError:
        return None, "matplotlib not installed. Install with: pip install audio-tools[plot]"


def format_timestamp(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = seconds % 60
    return f"{mins}:{secs:04.1f}"


def format_time_short(seconds: float) -> str:
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins}:{secs:02d}"


def add_files_arg(parser: argparse.ArgumentParser):
    parser.add_argument("files", nargs="+", help="Audio file(s) to analyze")


def add_refs_arg(parser: argparse.ArgumentParser):
    parser.add_argument("-r", "--refs", nargs="+", help="Reference tracks to compare against")


def add_plot_arg(parser: argparse.ArgumentParser):
    parser.add_argument("--plot", action="store_true", help="Show plot (requires matplotlib)")


def truncate_name(name: str, max_len: int = 40) -> str:
    if len(name) <= max_len:
        return name
    return name[:max_len - 3] + "..."
