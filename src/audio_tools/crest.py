"""Crest factor and Peak-to-Loudness Ratio (PLR) analysis."""

import statistics
import sys
from pathlib import Path

import numpy as np

from audio_tools.utils import (
    load_audio_mono, compute_ltas, band_energies, compute_rms, db,
    add_files_arg, add_refs_arg, truncate_name,
)
from audio_tools.loudness import get_audio_stats


def analyze_crest(filepath: str, loudness_stats: dict | None = None,
                  mono_data: tuple = None, ltas: tuple = None) -> dict | None:
    if mono_data is None:
        mono_data = load_audio_mono(filepath)
    data, sr = mono_data

    peak = float(np.max(np.abs(data)))
    rms = compute_rms(data)
    crest_db = db(peak, rms) if rms > 0 else 0.0
    rms_dbfs = db(rms)

    # LTAS for peak band identification (reuse if provided)
    if ltas is None:
        ltas = compute_ltas(data, sr)
    freqs, magnitudes = ltas
    bands = band_energies(freqs, magnitudes)
    peak_band = max(bands, key=bands.get)

    # PLR from ffmpeg loudness data (reuse if provided)
    if loudness_stats is None:
        loudness_stats = get_audio_stats(filepath)
    true_peak = loudness_stats["true_peak"] if loudness_stats else db(peak)
    integrated = loudness_stats["integrated_lufs"] if loudness_stats else None
    plr = true_peak - integrated if integrated is not None else None

    return {
        "name": Path(filepath).name,
        "path": filepath,
        "crest_factor_db": round(crest_db, 1),
        "rms_db": round(rms_dbfs, 1),
        "true_peak_dbfs": round(true_peak, 1),
        "plr_db": round(plr, 1) if plr is not None else None,
        "peak_band": peak_band,
        "integrated_lufs": integrated,
    }


def _dispatch(args):
    if not args.files:
        print("Error: No files specified.", file=sys.stderr)
        sys.exit(1)

    ref_results = []
    track_results = []

    if args.refs:
        for fp in args.refs:
            r = analyze_crest(fp)
            if r:
                ref_results.append(r)

    for fp in args.files:
        r = analyze_crest(fp)
        if r:
            track_results.append(r)

    if not track_results:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    all_names = [r["name"] for r in ref_results + track_results]
    max_name = min(40, max(len(n) for n in all_names)) if all_names else 40

    header = (f"{'':>{max_name}}  {'CREST':>6}  {'RMS':>6}  {'PEAK':>6}"
              f"  {'PLR':>6}  {'PEAK-BAND':>10}")

    ref_avg = {}
    if ref_results:
        print("\nREFERENCES")
        print(header)
        print("-" * len(header))

        for r in ref_results:
            name = truncate_name(r["name"], max_name)
            plr_str = f"{r['plr_db']:>6.1f}" if r["plr_db"] is not None else "     -"
            print(f"{name:>{max_name}}  {r['crest_factor_db']:>6.1f}"
                  f"  {r['rms_db']:>6.1f}  {r['true_peak_dbfs']:>+6.1f}"
                  f"  {plr_str}  {r['peak_band']:>10}")

        ref_avg["crest"] = statistics.mean(r["crest_factor_db"] for r in ref_results)
        ref_avg["rms"] = statistics.mean(r["rms_db"] for r in ref_results)
        ref_avg["peak"] = statistics.mean(r["true_peak_dbfs"] for r in ref_results)
        plr_vals = [r["plr_db"] for r in ref_results if r["plr_db"] is not None]
        ref_avg["plr"] = statistics.mean(plr_vals) if plr_vals else None

        plr_avg_str = f"{ref_avg['plr']:>6.1f}" if ref_avg["plr"] is not None else "     -"
        print(f"{'(average)':>{max_name}}  {ref_avg['crest']:>6.1f}"
              f"  {ref_avg['rms']:>6.1f}  {ref_avg['peak']:>+6.1f}"
              f"  {plr_avg_str}  {'':>10}")

    # Tracks
    delta_col = "  ΔCREST" if ref_avg else ""
    track_header = header + delta_col

    print(f"\nYOUR TRACKS" if ref_results else "\nTRACKS")
    print(track_header)
    print("-" * len(track_header))

    for r in track_results:
        name = truncate_name(r["name"], max_name)
        plr_str = f"{r['plr_db']:>6.1f}" if r["plr_db"] is not None else "     -"
        line = (f"{name:>{max_name}}  {r['crest_factor_db']:>6.1f}"
                f"  {r['rms_db']:>6.1f}  {r['true_peak_dbfs']:>+6.1f}"
                f"  {plr_str}  {r['peak_band']:>10}")
        if ref_avg:
            delta = r["crest_factor_db"] - ref_avg["crest"]
            line += f"  {delta:>+6.1f}"
        print(line)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("crest", help="Crest factor and PLR analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Crest factor analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
