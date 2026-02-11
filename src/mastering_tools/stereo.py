"""Stereo field analysis — correlation, mono compatibility, S/M ratio."""

import statistics
import sys
from pathlib import Path

import numpy as np

from mastering_tools.utils import (
    BANDS, BAND_NAMES, load_audio, compute_rms, db,
    add_files_arg, add_refs_arg, truncate_name,
)


def analyze_stereo(filepath: str, stereo_data: tuple = None) -> dict | None:
    if stereo_data is None:
        stereo_data = load_audio(filepath)
    data, sr = stereo_data
    name = Path(filepath).name

    # Handle mono
    if data.ndim == 1:
        return {
            "name": name,
            "path": filepath,
            "correlation": 1.0,
            "per_band_corr": {b: 1.0 for b in BAND_NAMES},
            "mono_loss_db": 0.0,
            "side_mid_ratio_db": -120.0,
            "is_mono": True,
        }

    left = data[:, 0]
    right = data[:, 1]
    mid = (left + right) / 2.0
    side = (left - right) / 2.0

    # Overall correlation
    correlation = float(np.corrcoef(left, right)[0, 1])

    # Per-band correlation (frequency domain — 2 FFTs instead of 14)
    left_fft = np.fft.rfft(left)
    right_fft = np.fft.rfft(right)
    freqs = np.fft.rfftfreq(len(left), 1.0 / sr)

    per_band = {}
    for band_name, (lo, hi) in BANDS.items():
        mask = (freqs >= lo) & (freqs <= hi)
        L = left_fft[mask]
        R = right_fft[mask]
        power_l = np.sum(np.abs(L) ** 2)
        power_r = np.sum(np.abs(R) ** 2)
        if power_l < 1e-20 or power_r < 1e-20:
            per_band[band_name] = 1.0
        else:
            cross = np.sum(L * np.conj(R)).real
            per_band[band_name] = float(cross / np.sqrt(power_l * power_r))

    # Mono fold-down loss
    rms_stereo = compute_rms(data.flatten())
    rms_mid = compute_rms(mid)
    mono_loss = db(rms_mid, rms_stereo) if rms_stereo > 0 else 0.0

    # Side/Mid ratio
    rms_side = compute_rms(side)
    sm_ratio = db(rms_side, rms_mid) if rms_mid > 0 else -120.0

    return {
        "name": name,
        "path": filepath,
        "correlation": round(correlation, 3),
        "per_band_corr": {b: round(v, 2) for b, v in per_band.items()},
        "mono_loss_db": round(mono_loss, 1),
        "side_mid_ratio_db": round(sm_ratio, 1),
        "is_mono": False,
    }


def _dispatch(args):
    if not args.files:
        print("Error: No files specified.", file=sys.stderr)
        sys.exit(1)

    ref_results = []
    track_results = []

    if args.refs:
        for fp in args.refs:
            r = analyze_stereo(fp)
            if r:
                ref_results.append(r)

    for fp in args.files:
        r = analyze_stereo(fp)
        if r:
            track_results.append(r)

    if not track_results:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    all_names = [r["name"] for r in ref_results + track_results]
    max_name = min(40, max(len(n) for n in all_names)) if all_names else 40

    header = (f"{'':>{max_name}}  {'CORR':>6}  {'MONO-Δ':>7}  {'S/M':>6}"
              f"  {'SUB-r':>6}  {'LOW-r':>6}")

    ref_avg = {}
    if ref_results:
        print("\nREFERENCES")
        print(header)
        print("-" * len(header))

        for r in ref_results:
            name = truncate_name(r["name"], max_name)
            print(f"{name:>{max_name}}  {r['correlation']:>6.2f}"
                  f"  {r['mono_loss_db']:>+7.1f}  {r['side_mid_ratio_db']:>6.1f}"
                  f"  {r['per_band_corr']['Sub']:>6.2f}"
                  f"  {r['per_band_corr']['Low']:>6.2f}")

        ref_avg["corr"] = statistics.mean(r["correlation"] for r in ref_results)
        ref_avg["mono_loss"] = statistics.mean(r["mono_loss_db"] for r in ref_results)
        ref_avg["sm"] = statistics.mean(r["side_mid_ratio_db"] for r in ref_results)
        ref_avg["sub_r"] = statistics.mean(r["per_band_corr"]["Sub"] for r in ref_results)
        ref_avg["low_r"] = statistics.mean(r["per_band_corr"]["Low"] for r in ref_results)

        print(f"{'(average)':>{max_name}}  {ref_avg['corr']:>6.2f}"
              f"  {ref_avg['mono_loss']:>+7.1f}  {ref_avg['sm']:>6.1f}"
              f"  {ref_avg['sub_r']:>6.2f}  {ref_avg['low_r']:>6.2f}")

    # Tracks
    delta_col = "  ΔCORR" if ref_avg else ""
    track_header = header + delta_col

    print(f"\nYOUR TRACKS" if ref_results else "\nTRACKS")
    print(track_header)
    print("-" * len(track_header))

    for r in track_results:
        name = truncate_name(r["name"], max_name)
        mono_str = "mono" if r["is_mono"] else f"{r['mono_loss_db']:>+7.1f}"
        line = (f"{name:>{max_name}}  {r['correlation']:>6.2f}"
                f"  {mono_str:>7}  {r['side_mid_ratio_db']:>6.1f}"
                f"  {r['per_band_corr']['Sub']:>6.2f}"
                f"  {r['per_band_corr']['Low']:>6.2f}")
        if ref_avg:
            delta = r["correlation"] - ref_avg["corr"]
            line += f"  {delta:>+6.2f}"
        print(line)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("stereo", help="Stereo field / mono compatibility analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stereo field analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
