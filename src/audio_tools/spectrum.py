"""Spectral balance analysis — LTAS comparison against references."""

import statistics
import sys
from pathlib import Path

from audio_tools.utils import (
    BANDS, BAND_NAMES, BAND_ABBREVS,
    load_audio_mono, compute_ltas, band_energies, add_files_arg, add_refs_arg,
    add_plot_arg, truncate_name, try_import_matplotlib,
)


def analyze_spectrum(filepath: str, mono_data: tuple = None,
                     ltas: tuple = None) -> dict | None:
    if mono_data is None:
        mono_data = load_audio_mono(filepath)
    data, sr = mono_data

    if ltas is None:
        ltas = compute_ltas(data, sr)
    freqs, magnitudes = ltas

    bands = band_energies(freqs, magnitudes)
    return {
        "name": Path(filepath).name,
        "path": filepath,
        "bands": bands,
        "ltas_freqs": freqs,
        "ltas_magnitudes": magnitudes,
    }


def _print_table(label: str, results: list[dict], ref_avg: dict | None = None,
                 max_name: int = 40):
    """Print a formatted table of band energies."""
    header = f"{'':>{max_name}}  " + "  ".join(f"{a:>6}" for a in BAND_ABBREVS)
    print(f"\n{label}")
    print(header)
    print("-" * len(header))

    for r in results:
        name = truncate_name(r["name"], max_name)
        vals = "  ".join(f"{r['bands'][b]:>6.1f}" for b in BAND_NAMES)
        print(f"{name:>{max_name}}  {vals}")

    if ref_avg:
        vals = "  ".join(f"{ref_avg[b]:>6.1f}" for b in BAND_NAMES)
        print(f"{'(average)':>{max_name}}  {vals}")


def _print_delta_table(results: list[dict], ref_avg: dict, max_name: int = 40):
    """Print tracks with delta rows beneath each."""
    delta_header = f"{'':>{max_name}}  " + "  ".join(f"{'Δ'+a:>6}" for a in BAND_ABBREVS)
    header = f"{'':>{max_name}}  " + "  ".join(f"{a:>6}" for a in BAND_ABBREVS)

    print(f"\nYOUR TRACKS")
    print(header)
    print(delta_header)
    print("-" * len(header))

    for r in results:
        name = truncate_name(r["name"], max_name)
        vals = "  ".join(f"{r['bands'][b]:>6.1f}" for b in BAND_NAMES)
        print(f"{name:>{max_name}}  {vals}")
        deltas = "  ".join(
            f"{r['bands'][b] - ref_avg[b]:>+6.1f}" for b in BAND_NAMES
        )
        print(f"{'':>{max_name}}  {deltas}")


def _plot_ltas(ref_results: list[dict], track_results: list[dict]):
    plt, err = try_import_matplotlib()
    if not plt:
        print(err, file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    for r in ref_results:
        ax.semilogx(r["ltas_freqs"], r["ltas_magnitudes"],
                     color="gray", alpha=0.5, linewidth=0.8,
                     label=f"ref: {r['name'][:30]}")
    for r in track_results:
        ax.semilogx(r["ltas_freqs"], r["ltas_magnitudes"],
                     linewidth=1.2, label=r["name"][:30])

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("Long-Term Average Spectrum")
    ax.set_xlim(20, 20000)
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)

    # Draw band boundaries
    for _, (lo, hi) in BANDS.items():
        ax.axvline(lo, color="gray", alpha=0.15, linewidth=0.5)

    plt.tight_layout()
    plt.show()


def _dispatch(args):
    if not args.files:
        print("Error: No files specified.", file=sys.stderr)
        sys.exit(1)

    ref_results = []
    track_results = []

    if args.refs:
        for fp in args.refs:
            r = analyze_spectrum(fp)
            if r:
                ref_results.append(r)

    for fp in args.files:
        r = analyze_spectrum(fp)
        if r:
            track_results.append(r)

    if not track_results:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    all_names = [r["name"] for r in ref_results + track_results]
    max_name = min(40, max(len(n) for n in all_names)) if all_names else 40

    ref_avg = None
    if ref_results:
        _print_table("REFERENCES", ref_results, max_name=max_name)
        ref_avg = {}
        for band in BAND_NAMES:
            ref_avg[band] = statistics.mean(r["bands"][band] for r in ref_results)
        vals = "  ".join(f"{ref_avg[b]:>6.1f}" for b in BAND_NAMES)
        print(f"{'(average)':>{max_name}}  {vals}")

    if ref_avg:
        _print_delta_table(track_results, ref_avg, max_name)
    else:
        _print_table("TRACKS", track_results, max_name=max_name)

    if args.plot:
        _plot_ltas(ref_results, track_results)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("spectrum", help="Spectral balance (LTAS) analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    add_plot_arg(parser)
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Spectral balance analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    add_plot_arg(parser)
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
