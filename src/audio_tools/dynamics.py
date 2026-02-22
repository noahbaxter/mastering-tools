"""Short-term loudness timeline analysis."""

import statistics
import sys
from pathlib import Path

from audio_tools.utils import (
    add_files_arg, add_refs_arg, add_plot_arg, truncate_name, try_import_matplotlib,
)
from audio_tools.loudness import get_audio_stats


def analyze_dynamics(filepath: str, loudness_stats: dict | None = None,
                     gate_offset: float = 20.0) -> dict | None:
    """Analyze short-term loudness dynamics.

    gate_offset: readings more than this many LU below integrated are
                 considered silence/intros/outros and excluded from the range.
    """
    if loudness_stats is None:
        loudness_stats = get_audio_stats(filepath)
    if not loudness_stats:
        return None

    st_values = loudness_stats.get("short_term_values", [])
    timestamps = loudness_stats.get("timestamps", [])
    integrated = loudness_stats.get("integrated_lufs")

    # Relative gate: ignore anything far below integrated loudness
    # This filters out intros, outros, breakdowns, silence
    if integrated is not None:
        gate = integrated - gate_offset
    else:
        gate = -60.0

    audible_st = [v for v in st_values if v > gate]
    if not audible_st:
        return None

    return {
        "name": Path(filepath).name,
        "path": filepath,
        "st_values": st_values,
        "timestamps": timestamps,
        "st_min": min(audible_st),
        "st_max": max(audible_st),
        "st_range": max(audible_st) - min(audible_st),
        "integrated_lufs": integrated,
        "gate": round(gate, 1),
    }


def _plot_dynamics(ref_results: list[dict], track_results: list[dict]):
    plt, err = try_import_matplotlib()
    if not plt:
        print(err, file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(14, 5))

    for r in ref_results:
        ax.plot(r["timestamps"], r["st_values"],
                color="gray", alpha=0.4, linewidth=0.8,
                label=f"ref: {r['name'][:30]}")

    for r in track_results:
        ax.plot(r["timestamps"], r["st_values"],
                linewidth=1.0, label=r["name"][:30])

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Short-Term LUFS")
    ax.set_title("Short-Term Loudness Timeline")
    ax.legend(fontsize=7, loc="lower left")
    ax.grid(True, alpha=0.3)
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
            r = analyze_dynamics(fp)
            if r:
                ref_results.append(r)

    for fp in args.files:
        r = analyze_dynamics(fp)
        if r:
            track_results.append(r)

    if not track_results:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    all_names = [r["name"] for r in ref_results + track_results]
    max_name = min(40, max(len(n) for n in all_names)) if all_names else 40

    header = (f"{'':>{max_name}}  {'ST-MIN':>7}  {'ST-MAX':>7}  {'ST-RNG':>7}"
              f"  {'I-LUFS':>7}")

    ref_avg = {}
    if ref_results:
        print("\nREFERENCES")
        print(header)
        print("-" * len(header))

        for r in ref_results:
            name = truncate_name(r["name"], max_name)
            lufs_str = f"{r['integrated_lufs']:>+7.1f}" if r["integrated_lufs"] else "      -"
            print(f"{name:>{max_name}}  {r['st_min']:>+7.1f}  {r['st_max']:>+7.1f}"
                  f"  {r['st_range']:>7.1f}  {lufs_str}")

        ref_avg["st_range"] = statistics.mean(r["st_range"] for r in ref_results)
        lufs_vals = [r["integrated_lufs"] for r in ref_results if r["integrated_lufs"]]
        ref_avg["lufs"] = statistics.mean(lufs_vals) if lufs_vals else None
        ref_avg["st_min"] = statistics.mean(r["st_min"] for r in ref_results)
        ref_avg["st_max"] = statistics.mean(r["st_max"] for r in ref_results)

        lufs_avg_str = f"{ref_avg['lufs']:>+7.1f}" if ref_avg["lufs"] else "      -"
        print(f"{'(average)':>{max_name}}  {ref_avg['st_min']:>+7.1f}"
              f"  {ref_avg['st_max']:>+7.1f}  {ref_avg['st_range']:>7.1f}"
              f"  {lufs_avg_str}")

    # Tracks
    delta_col = "  ΔST-RNG" if ref_avg else ""
    track_header = header + delta_col

    print(f"\nYOUR TRACKS" if ref_results else "\nTRACKS")
    print(track_header)
    print("-" * len(track_header))

    for r in track_results:
        name = truncate_name(r["name"], max_name)
        lufs_str = f"{r['integrated_lufs']:>+7.1f}" if r["integrated_lufs"] else "      -"
        line = (f"{name:>{max_name}}  {r['st_min']:>+7.1f}  {r['st_max']:>+7.1f}"
                f"  {r['st_range']:>7.1f}  {lufs_str}")
        if ref_avg:
            delta = r["st_range"] - ref_avg["st_range"]
            flag = " \u26a0" if abs(delta) > 4.0 else ""
            line += f"  {delta:>+7.1f}{flag}"
        print(line)

    if args.plot:
        _plot_dynamics(ref_results, track_results)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("dynamics", help="Short-term loudness timeline analysis")
    add_files_arg(parser)
    add_refs_arg(parser)
    add_plot_arg(parser)
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Short-term loudness timeline")
    add_files_arg(parser)
    add_refs_arg(parser)
    add_plot_arg(parser)
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
