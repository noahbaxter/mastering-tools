"""Compare previous vs current masters against reference targets."""

import statistics
import subprocess
import sys
import threading
from pathlib import Path

from audio_tools.utils import (
    BAND_NAMES, load_audio, compute_ltas,
)
from audio_tools.loudness import stats_from_output
from audio_tools.spectrum import analyze_spectrum

BAR_WIDTH = 20
NAME_WIDTH = 38


class _Progress:
    def __init__(self, filepaths):
        self.files = filepaths
        self.state = {fp: (0.0, "") for fp in filepaths}
        self.lock = threading.Lock()
        self.drawn = False

    def update(self, filepath, fraction, label=""):
        with self.lock:
            self.state[filepath] = (fraction, label)
            self._render()

    def _render(self):
        n = len(self.files)
        if self.drawn:
            sys.stderr.write(f"\033[{n}A")
        for fp in self.files:
            frac, label = self.state[fp]
            name = Path(fp).name
            if len(name) > NAME_WIDTH:
                name = name[:NAME_WIDTH - 3] + "..."
            filled = int(frac * BAR_WIDTH)
            bar = "\u2588" * filled + "\u00b7" * (BAR_WIDTH - filled)
            if frac >= 1.0:
                status = "\u2713"
            elif label:
                status = label
            else:
                status = ""
            sys.stderr.write(f"\033[2K  {name:<{NAME_WIDTH}}  [{bar}] {status}\n")
        sys.stderr.flush()
        self.drawn = True

    def finish(self):
        pass


def _trend(prev_delta, cur_delta, threshold=0.1):
    prev_abs = abs(prev_delta)
    cur_abs = abs(cur_delta)
    diff = prev_abs - cur_abs
    if abs(diff) < threshold:
        return ("  ~", "same")
    elif cur_abs < prev_abs:
        return (" >>", "closer")
    else:
        return (" <<", "regressed")


def _print_section(title, rows):
    print(f"\n{title}")
    print(f"  {'':14} {'Ref Avg':>10} {'Prev Δ':>10} {'Cur Δ':>10}  {'Trend':>14}")
    print(f"  {'-' * 62}")
    for name, ref_avg, prev_delta, cur_delta, symbol, label in rows:
        print(f"  {name:14} {ref_avg:>10.1f} {prev_delta:>+10.1f} {cur_delta:>+10.1f}  {symbol} {label}")


def _run_compare(prev_paths, cur_paths, ref_paths):
    n_prev = len(prev_paths)
    n_cur = len(cur_paths)
    n_ref = len(ref_paths)

    all_paths = ref_paths + prev_paths + cur_paths
    prog = _Progress(all_paths)

    # Phase 1: ffmpeg loudness in parallel
    procs = {}
    for fp in all_paths:
        cmd = ["ffmpeg", "-i", fp, "-filter_complex",
               "ebur128=peak=true:framelog=info", "-f", "null", "-"]
        procs[fp] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        prog.update(fp, 0.05, "loudness")

    loudness_map = {}
    lock = threading.Lock()

    def _collect_ffmpeg(fp):
        output = procs[fp].stderr.read().decode(errors="replace")
        stats = stats_from_output(fp, output)
        with lock:
            loudness_map[fp] = stats
        prog.update(fp, 0.5)

    threads = [threading.Thread(target=_collect_ffmpeg, args=(fp,))
               for fp in all_paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Phase 2: spectrum analysis (sequential per file)
    spectrum_map = {}
    for fp in all_paths:
        prog.update(fp, 0.55, "loading")
        data, sr = load_audio(fp)
        mono = data.mean(axis=1) if data.ndim > 1 else data

        prog.update(fp, 0.7, "spectrum")
        ltas = compute_ltas(mono, sr)
        result = analyze_spectrum(fp, mono_data=(mono, sr), ltas=ltas)
        spectrum_map[fp] = result
        prog.update(fp, 1.0)

    prog.finish()
    print(file=sys.stderr)

    # Collect stats per group
    def _collect_group(paths):
        loudness = [loudness_map[fp] for fp in paths if loudness_map.get(fp)]
        spectrum = [spectrum_map[fp] for fp in paths if spectrum_map.get(fp)]
        return loudness, spectrum

    ref_loud, ref_spec = _collect_group(ref_paths)
    prev_loud, prev_spec = _collect_group(prev_paths)
    cur_loud, cur_spec = _collect_group(cur_paths)

    for label, group in [("reference", ref_loud), ("previous", prev_loud), ("current", cur_loud)]:
        if not group:
            print(f"Error: No {label} files successfully analyzed.", file=sys.stderr)
            sys.exit(1)

    # Header
    print(f"\nCOMPARE: {n_prev} prev vs {n_cur} current vs {n_ref} references")
    print("=" * 56)

    rows_loudness = []
    rows_spectrum = []

    # Loudness metrics
    metrics = [
        ("I-LUFS", "integrated_lufs"),
        ("LRA", "loudness_range"),
    ]
    for name, key in metrics:
        ref_vals = [s[key] for s in ref_loud if s[key] is not None]
        prev_vals = [s[key] for s in prev_loud if s[key] is not None]
        cur_vals = [s[key] for s in cur_loud if s[key] is not None]
        if not ref_vals or not prev_vals or not cur_vals:
            continue

        ref_avg = statistics.mean(ref_vals)
        prev_delta = statistics.mean(prev_vals) - ref_avg
        cur_delta = statistics.mean(cur_vals) - ref_avg
        symbol, label = _trend(prev_delta, cur_delta)
        rows_loudness.append((name, ref_avg, prev_delta, cur_delta, symbol, label))

    _print_section("LOUDNESS", rows_loudness)

    # Spectrum metrics
    for band in BAND_NAMES:
        ref_vals = [s["bands"][band] for s in ref_spec]
        prev_vals = [s["bands"][band] for s in prev_spec]
        cur_vals = [s["bands"][band] for s in cur_spec]

        ref_avg = statistics.mean(ref_vals)
        prev_delta = statistics.mean(prev_vals) - ref_avg
        cur_delta = statistics.mean(cur_vals) - ref_avg
        symbol, label = _trend(prev_delta, cur_delta)
        rows_spectrum.append((band, ref_avg, prev_delta, cur_delta, symbol, label))

    _print_section("SPECTRUM", rows_spectrum)

    # Normalized summary
    lufs_row = next((r for r in rows_loudness if r[0] == "I-LUFS"), None)
    if not lufs_row:
        print("\nSUMMARY: insufficient loudness data")
        print()
        return

    norm_gain = -lufs_row[3]  # gain to bring cur to ref LUFS
    prev_norm_gain = -lufs_row[2]

    # Free loudness: peak headroom that can be claimed via normalization
    CEILING = -0.1
    max_cur_peak = max(s["true_peak"] for s in cur_loud)
    free_gain = max(0.0, CEILING - max_cur_peak)

    if norm_gain > 0.05:
        print(f"\nSUMMARY — {norm_gain:.1f} dB below ref loudness")
    elif norm_gain < -0.05:
        print(f"\nSUMMARY — {-norm_gain:.1f} dB above ref loudness")
    else:
        print(f"\nSUMMARY — matches ref loudness")
    print("-" * 56)

    if free_gain > 0.05 and norm_gain > 0.05:
        usable = min(free_gain, norm_gain)
        remaining = norm_gain - usable
        print(f"  Free gain: {usable:+.1f} dB peak norm to {max_cur_peak + usable:+.1f} dBFS")
        if remaining > 0.05:
            print(f"  Remaining {remaining:.1f} dB needs limiting")
    elif norm_gain > 0.05:
        print(f"  No free gain (peak at {max_cur_peak:+.1f} dBFS)"
              f" — {norm_gain:.1f} dB gap needs limiting")

    # LRA (unaffected by gain)
    lra_row = next((r for r in rows_loudness if r[0] == "LRA"), None)
    if lra_row and abs(lra_row[3]) > 0.5:
        direction = "wider" if lra_row[3] > 0 else "tighter"
        change = abs(lra_row[3] - lra_row[2])
        print(f"  LRA: {lra_row[3]:+.1f} LU {direction} than refs", end="")
        if change > 0.2:
            print(f" (was {lra_row[2]:+.1f})")
        else:
            print(" (unchanged)")

    # Normalized spectrum deltas
    norm_bands = []
    for name, ref_avg, prev_d, cur_d, _, _ in rows_spectrum:
        norm_prev = prev_d + prev_norm_gain
        norm_cur = cur_d + norm_gain
        norm_bands.append((name, norm_prev, norm_cur))

    # Biggest remaining gaps (> 1.0 dB after norm)
    gaps = [(n, nc) for n, _, nc in
            sorted(norm_bands, key=lambda x: -abs(x[2])) if abs(nc) > 1.0]
    if gaps:
        gap_strs = [f"{n} {nc:+.1f}" for n, nc in gaps]
        print(f"  Gaps:     {', '.join(gap_strs)}")

    # Most improved (> 0.5 dB closer to ref after norm)
    improved = [(n, np_, nc) for n, np_, nc in norm_bands
                if abs(np_) - abs(nc) > 0.5]
    improved.sort(key=lambda x: -(abs(x[1]) - abs(x[2])))
    if improved:
        imp_strs = [f"{n} ({np_:+.1f} \u2192 {nc:+.1f})" for n, np_, nc in improved]
        print(f"  Improved: {', '.join(imp_strs)}")

    # Regressed (> 0.5 dB further from ref after norm)
    regressed = [(n, np_, nc) for n, np_, nc in norm_bands
                 if abs(nc) - abs(np_) > 0.5]
    regressed.sort(key=lambda x: abs(x[2]) - abs(x[1]))
    if regressed:
        reg_strs = [f"{n} ({np_:+.1f} \u2192 {nc:+.1f})" for n, np_, nc in regressed]
        print(f"  Regressed: {', '.join(reg_strs)}")

    print()


def _dispatch(args):
    if not args.prev:
        print("Error: --prev is required.", file=sys.stderr)
        sys.exit(1)
    if not args.cur:
        print("Error: --cur is required.", file=sys.stderr)
        sys.exit(1)
    if not args.refs:
        print("Error: -r/--refs is required.", file=sys.stderr)
        sys.exit(1)

    _run_compare(args.prev, args.cur, args.refs)


def register_subcommand(subparsers):
    parser = subparsers.add_parser(
        "compare",
        help="Compare prev vs current masters against reference targets",
    )
    parser.add_argument("--prev", nargs="+", required=True,
                        help="Previous version audio file(s)")
    parser.add_argument("--cur", nargs="+", required=True,
                        help="Current version audio file(s)")
    parser.add_argument("-r", "--refs", nargs="+", required=True,
                        help="Reference track(s)")
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="Compare prev vs current masters against reference targets",
    )
    parser.add_argument("--prev", nargs="+", required=True,
                        help="Previous version audio file(s)")
    parser.add_argument("--cur", nargs="+", required=True,
                        help="Current version audio file(s)")
    parser.add_argument("-r", "--refs", nargs="+", required=True,
                        help="Reference track(s)")
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
