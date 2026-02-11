"""Unified mastering check — runs all analyses and flags deviations."""

import statistics
import subprocess
import sys
import threading
from pathlib import Path

from mastering_tools.utils import (
    BAND_NAMES, add_files_arg, add_refs_arg, truncate_name,
    load_audio, compute_ltas,
)
from mastering_tools.loudness import stats_from_output
from mastering_tools.spectrum import analyze_spectrum
from mastering_tools.crest import analyze_crest
from mastering_tools.stereo import analyze_stereo
from mastering_tools.dynamics import analyze_dynamics

# Deviation thresholds
LUFS_THRESHOLD = 2.0
PEAK_OVER_ZERO = 0.0
PEAK_DELTA_THRESHOLD = 0.5
BAND_THRESHOLD = 3.0
CORR_LOW_THRESHOLD = 0.3
SUB_CORR_THRESHOLD = 0.8
MONO_LOSS_THRESHOLD = 3.0
ST_RANGE_THRESHOLD = 4.0

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
        # Move past the display — it stays visible
        pass


def _flag(condition: bool) -> str:
    return " \u26a0" if condition else ""


def _run_check(track_paths: list[str], ref_paths: list[str]):
    n_tracks = len(track_paths)
    n_refs = len(ref_paths)

    all_paths = ref_paths + track_paths
    total = len(all_paths)
    prog = _Progress(all_paths)

    # --- Phase 1: ffmpeg loudness in parallel ---
    # Launch ALL ffmpeg processes at the OS level simultaneously.
    # Popen starts real processes — no GIL, true parallelism.
    procs = {}
    for fp in all_paths:
        cmd = ["ffmpeg", "-i", fp, "-filter_complex",
               "ebur128=peak=true:framelog=info", "-f", "null", "-"]
        procs[fp] = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        prog.update(fp, 0.05, "loudness")

    # Collect output via threads (just pipe reads, releases GIL)
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

    # --- Phase 2: audio-based analyses (load once, share data) ---
    ref_loudness, ref_spectrum, ref_crest, ref_stereo, ref_dynamics = [], [], [], [], []
    track_loudness, track_spectrum, track_crest, track_stereo, track_dynamics = [], [], [], [], []

    def collect(target_lists, r):
        loud, spec, cre, ste, dyn = target_lists
        if r["loudness"]:
            loud.append(r["loudness"])
        if r["dynamics"]:
            dyn.append(r["dynamics"])
        if r["spectrum"]:
            spec.append(r["spectrum"])
        if r["crest"]:
            cre.append(r["crest"])
        if r["stereo"]:
            ste.append(r["stereo"])

    for fp in all_paths:
        ls = loudness_map.get(fp)
        results = {"loudness": ls}
        results["dynamics"] = analyze_dynamics(fp, loudness_stats=ls) if ls else None

        prog.update(fp, 0.55, "loading")
        stereo_data = load_audio(fp)
        data, sr = stereo_data
        mono = data.mean(axis=1) if data.ndim > 1 else data
        mono_data = (mono, sr)

        prog.update(fp, 0.65, "spectrum")
        ltas = compute_ltas(mono, sr)
        results["spectrum"] = analyze_spectrum(fp, mono_data=mono_data, ltas=ltas)

        prog.update(fp, 0.75, "crest")
        results["crest"] = analyze_crest(fp, loudness_stats=ls,
                                         mono_data=mono_data, ltas=ltas)

        prog.update(fp, 0.88, "stereo")
        results["stereo"] = analyze_stereo(fp, stereo_data=stereo_data)

        prog.update(fp, 1.0)

        if fp in ref_paths:
            collect((ref_loudness, ref_spectrum, ref_crest, ref_stereo, ref_dynamics), results)
        else:
            collect((track_loudness, track_spectrum, track_crest, track_stereo, track_dynamics), results)

    prog.finish()
    print(file=sys.stderr)

    if not track_loudness:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 56}")
    print(f"  MASTERING CHECK: {n_tracks} track{'s' if n_tracks != 1 else ''}"
          f" vs {n_refs} reference{'s' if n_refs != 1 else ''}")
    print(f"{'=' * 56}")

    flags = 0
    oks = 0

    # --- LOUDNESS ---
    print("\nLOUDNESS")
    if ref_loudness:
        ref_lufs_avg = statistics.mean(
            s["integrated_lufs"] for s in ref_loudness if s["integrated_lufs"]
        )
        track_lufs_vals = [s["integrated_lufs"] for s in track_loudness if s["integrated_lufs"]]
        if track_lufs_vals:
            track_lufs_avg = statistics.mean(track_lufs_vals)
            delta = track_lufs_avg - ref_lufs_avg
            direction = "quieter" if delta < 0 else "louder"
            flagged = abs(delta) > LUFS_THRESHOLD
            print(f"  Avg Δ from refs: {delta:+.1f} LUFS ({direction}){_flag(flagged)}")
            flags += flagged
            oks += not flagged

        # True peak
        track_peaks = [s["true_peak"] for s in track_loudness]
        ref_peaks = [s["true_peak"] for s in ref_loudness]
        peak_min, peak_max = min(track_peaks), max(track_peaks)
        ref_peak_min, ref_peak_max = min(ref_peaks), max(ref_peaks)
        peak_over = peak_max > PEAK_OVER_ZERO
        peak_far = abs(statistics.mean(track_peaks) - statistics.mean(ref_peaks)) > PEAK_DELTA_THRESHOLD
        flagged = peak_over or peak_far
        print(f"  True peak: {peak_min:+.1f} to {peak_max:+.1f}"
              f" (refs: {ref_peak_min:+.1f} to {ref_peak_max:+.1f}){_flag(flagged)}")
        flags += flagged
        oks += not flagged

        # LRA spread
        track_lra = [s["loudness_range"] for s in track_loudness if s["loudness_range"]]
        ref_lra = [s["loudness_range"] for s in ref_loudness if s["loudness_range"]]
        if track_lra and ref_lra:
            lra_spread = f"{min(track_lra):.1f} - {max(track_lra):.1f}"
            ref_lra_spread = f"{min(ref_lra):.1f} - {max(ref_lra):.1f}"
            lra_flagged = max(track_lra) > max(ref_lra) * 2
            print(f"  LRA spread: {lra_spread} (refs: {ref_lra_spread}){_flag(lra_flagged)}")
            flags += lra_flagged
            oks += not lra_flagged

    # --- SPECTRUM ---
    print("\nSPECTRUM")
    if ref_spectrum and track_spectrum:
        ref_band_avg = {}
        for band in BAND_NAMES:
            ref_band_avg[band] = statistics.mean(r["bands"][band] for r in ref_spectrum)

        band_flags = []
        for band in BAND_NAMES:
            track_band_avg = statistics.mean(t["bands"][band] for t in track_spectrum)
            delta = track_band_avg - ref_band_avg[band]
            flagged = abs(delta) > BAND_THRESHOLD
            label = _flag(flagged) if flagged else ""
            print(f"  {band}: {delta:+.1f}dB vs refs{label}")
            if flagged:
                band_flags.append(band)

        flags += len(band_flags)
        oks += len(BAND_NAMES) - len(band_flags)
    else:
        print("  (insufficient data)")

    # --- CREST ---
    print("\nCREST")
    if ref_crest and track_crest:
        ref_crest_avg = statistics.mean(r["crest_factor_db"] for r in ref_crest)
        track_crest_avg = statistics.mean(t["crest_factor_db"] for t in track_crest)
        delta = track_crest_avg - ref_crest_avg
        flagged = delta > 1.5
        note = " peaks not translating to loudness" if flagged else ""
        print(f"  Avg crest: {track_crest_avg:.1f} dB (refs: {ref_crest_avg:.1f})"
              f"{_flag(flagged)}{note}")
        flags += flagged
        oks += not flagged
    else:
        print("  (insufficient data)")

    # --- STEREO ---
    print("\nSTEREO")
    if ref_stereo and track_stereo:
        # Mono loss
        ref_mono_avg = statistics.mean(r["mono_loss_db"] for r in ref_stereo)
        track_mono_avg = statistics.mean(t["mono_loss_db"] for t in track_stereo if not t["is_mono"])
        mono_flagged = abs(track_mono_avg) > MONO_LOSS_THRESHOLD
        print(f"  Mono fold-down: {track_mono_avg:+.1f}dB avg"
              f" (refs: {ref_mono_avg:+.1f}){_flag(mono_flagged)}"
              f"{'  OK' if not mono_flagged else ''}")
        flags += mono_flagged
        oks += not mono_flagged

        # Sub correlation
        track_sub_corrs = [t["per_band_corr"]["Sub"] for t in track_stereo if not t["is_mono"]]
        if track_sub_corrs:
            sub_avg = statistics.mean(track_sub_corrs)
            sub_flagged = sub_avg < SUB_CORR_THRESHOLD
            print(f"  Sub correlation: {sub_avg:.2f} avg{_flag(sub_flagged)}"
                  f"{'  OK' if not sub_flagged else ''}")
            flags += sub_flagged
            oks += not sub_flagged

        # Overall correlation
        track_corrs = [t["correlation"] for t in track_stereo if not t["is_mono"]]
        if track_corrs:
            corr_min = min(track_corrs)
            corr_flagged = corr_min < CORR_LOW_THRESHOLD
            if corr_flagged:
                worst = min(track_stereo, key=lambda t: t["correlation"])
                print(f"  Low correlation: {corr_min:.2f} ({worst['name'][:30]}){_flag(True)}")
                flags += 1
    else:
        print("  (insufficient data)")

    # --- DYNAMICS ---
    print("\nDYNAMICS")
    if ref_dynamics and track_dynamics:
        ref_ranges = [r["st_range"] for r in ref_dynamics]
        ref_min_range = min(ref_ranges)
        ref_max_range = max(ref_ranges)
        margin = 3.0  # LU margin beyond ref spread

        print(f"  Ref ST-range: {ref_min_range:.1f} - {ref_max_range:.1f} LU")

        dyn_flags = 0
        for t in track_dynamics:
            st = t["st_range"]
            below = st < ref_min_range - margin
            above = st > ref_max_range + margin
            flagged = below or above
            if flagged:
                name = truncate_name(t["name"], 30)
                direction = "tighter" if below else "wider"
                print(f"  \"{name}\" ST-range: {st:.1f} ({direction}){_flag(True)}")
                dyn_flags += 1

        if dyn_flags == 0:
            print(f"  All tracks within ref range  OK")
            oks += 1
        flags += dyn_flags
    else:
        print("  (insufficient data)")

    # --- SUMMARY ---
    print(f"\nSUMMARY: {flags} item{'s' if flags != 1 else ''} flagged,"
          f" {oks} item{'s' if oks != 1 else ''} OK")
    print()


def _dispatch(args):
    if not args.files:
        print("Error: No files specified.", file=sys.stderr)
        sys.exit(1)
    if not args.refs:
        print("Error: check requires reference tracks (-r).", file=sys.stderr)
        sys.exit(1)

    _run_check(args.files, args.refs)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("check", help="Unified mastering report (runs all analyses)")
    add_files_arg(parser)
    add_refs_arg(parser)
    parser.set_defaults(func=_dispatch)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Unified mastering check")
    add_files_arg(parser)
    add_refs_arg(parser)
    args = parser.parse_args()
    _dispatch(args)


if __name__ == "__main__":
    main()
