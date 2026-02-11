#!/usr/bin/env python3
"""Detect and remove single-sample digital clicks from audio files."""

import argparse
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf


def detect_dropouts(samples: np.ndarray, dropout_threshold: float = 0.05,
                    exact_zero_threshold: float = 0.008,
                    dip_ratio: float = 0.05, near_zero_threshold: float = 0.001) -> np.ndarray:
    """Detect zero-sample dropouts (e.g., ADAT sync errors).

    Detects three patterns:
    1. Exact zeros (|x| < 1e-9) with any non-trivial neighbors (low threshold)
    2. Near-zeros (|x| < near_zero_threshold) with significant neighbors (high threshold)
    3. Dips: samples much smaller than both neighbors (|x| < min_neighbor * dip_ratio)

    Args:
        samples: Audio sample array
        dropout_threshold: Minimum neighbor amplitude for near-zero/dip detection
        exact_zero_threshold: Lower threshold for exact zeros (more sensitive)
        dip_ratio: Sample must be this fraction of min neighbor to trigger dip detection
        near_zero_threshold: Absolute threshold for "near zero" detection
    """
    if len(samples) < 3:
        return np.array([], dtype=np.int64)

    dropout_candidates = set()

    for i in range(1, len(samples) - 1):
        val = abs(samples[i])
        left = abs(samples[i - 1])
        right = abs(samples[i + 1])
        min_neighbor = min(left, right)
        max_neighbor = max(left, right)

        # Pattern 1: Exact zero with any non-trivial neighbors
        # Exact zeros are very likely dropouts, so use lower threshold
        # But at zero crossings in quiet audio, exact zeros can occur naturally
        if val < 1e-9 and max_neighbor > exact_zero_threshold:
            is_zero_crossing = (samples[i - 1] > 0) != (samples[i + 1] > 0)
            # At zero crossings, require moderate neighbor amplitude to be suspicious
            # (exact zeros in quiet zero-crossing regions may be natural)
            if not is_zero_crossing or max_neighbor > 0.02:
                dropout_candidates.add(i)
            continue

        # Pattern 2: Near-zero with significant neighbors
        # Near-zeros need higher threshold to avoid false positives at zero crossings
        # Only flag if neighbors have same sign (at zero crossings, near-zero is expected)
        if val < near_zero_threshold and max_neighbor > dropout_threshold:
            if (samples[i - 1] > 0) == (samples[i + 1] > 0):  # Same sign
                dropout_candidates.add(i)
            continue

        # Pattern 3: Dip detection - sample is much smaller than BOTH neighbors
        # This catches partial dropouts and near-zeros regardless of absolute value
        # Only trigger if neighbors are significant (avoids false positives in quiet sections)
        # and have the same sign (not a legitimate zero crossing)
        if min_neighbor > dropout_threshold and val < min_neighbor * dip_ratio:
            if (samples[i - 1] > 0) == (samples[i + 1] > 0):  # Same sign
                dropout_candidates.add(i)
            continue

    # Iteratively expand to include near-zeros adjacent to detected dropouts
    # (multi-sample dropouts where edge samples might not be exact zero)
    changed = True
    while changed:
        changed = False
        for i in range(1, len(samples) - 1):
            if i not in dropout_candidates and abs(samples[i]) < near_zero_threshold:
                if (i - 1) in dropout_candidates or (i + 1) in dropout_candidates:
                    dropout_candidates.add(i)
                    changed = True

    return np.array(sorted(dropout_candidates), dtype=np.int64)


def detect_sync_artifacts(samples: np.ndarray, min_jump: float = 0.04,
                          tolerance: float = 0.001, min_consecutive: int = 3) -> np.ndarray:
    """Detect sync/phase artifacts: consecutive samples with identical large jumps.

    These artifacts occur when audio sync is lost and samples are displaced uniformly,
    creating runs of identical sample-to-sample differences that don't occur in natural audio.

    Args:
        samples: Audio sample array
        min_jump: Minimum absolute jump size to consider
        tolerance: Maximum difference between consecutive jumps to consider them "identical"
        min_consecutive: Minimum number of consecutive identical jumps to flag
    """
    if len(samples) < min_consecutive + 2:
        return np.array([], dtype=np.int64)

    diffs = np.diff(samples)
    artifact_samples = set()

    i = 0
    while i < len(diffs) - min_consecutive + 1:
        # Check if we have min_consecutive large, similar jumps starting at i
        if abs(diffs[i]) > min_jump:
            # Count consecutive similar jumps
            count = 1
            j = i + 1
            while j < len(diffs) and abs(diffs[j]) > min_jump and abs(diffs[j] - diffs[i]) < tolerance:
                count += 1
                j += 1

            if count >= min_consecutive:
                # Mark all samples in this artifact region for repair
                # The artifact spans from sample i+1 to sample i+count (inclusive)
                for k in range(i + 1, i + count + 1):
                    artifact_samples.add(k)
                i = j
                continue
        i += 1

    return np.array(sorted(artifact_samples), dtype=np.int64)


def detect_ratio_clicks(samples: np.ndarray, ratio_threshold: float = 10.0,
                        min_diff_threshold: float = 0.01) -> np.ndarray:
    """Detect single-sample clicks using ratio-based detection."""
    if len(samples) < 3:
        return np.array([], dtype=np.int64)

    epsilon = 1e-10

    diff_before = np.abs(samples[1:-1] - samples[:-2])
    diff_after = np.abs(samples[1:-1] - samples[2:])
    continuity = np.abs(samples[:-2] - samples[2:])

    avg_diff = (diff_before + diff_after) / 2
    ratio = avg_diff / (continuity + epsilon)

    click_mask = (ratio > ratio_threshold) & (avg_diff > min_diff_threshold)
    return np.where(click_mask)[0] + 1


def detect_clicks(samples: np.ndarray, ratio_threshold: float = 10.0,
                  min_diff_threshold: float = 0.01,
                  dropout_threshold: float = 0.05,
                  do_clicks: bool = True,
                  do_dropouts: bool = True,
                  do_sync: bool = False) -> np.ndarray:
    """Detect clicks, dropouts, and/or sync artifacts in audio data."""
    if len(samples) < 3:
        return np.array([], dtype=np.int64)

    click_set = set()

    if do_clicks:
        click_set.update(detect_ratio_clicks(samples, ratio_threshold, min_diff_threshold))

    if do_dropouts:
        click_set.update(detect_dropouts(samples, dropout_threshold))

    if do_sync:
        click_set.update(detect_sync_artifacts(samples))

    return np.array(sorted(click_set), dtype=np.int64)


def group_consecutive(indices: np.ndarray) -> list[list[int]]:
    """Group consecutive indices into runs."""
    if len(indices) == 0:
        return []

    runs = []
    current_run = []
    for idx in sorted(indices):
        if not current_run or idx == current_run[-1] + 1:
            current_run.append(idx)
        else:
            runs.append(current_run)
            current_run = [idx]
    if current_run:
        runs.append(current_run)
    return runs


def repair_clicks(samples: np.ndarray, click_indices: np.ndarray,
                  context_samples: int = 5, poly_degree: int = 3) -> np.ndarray:
    """Repair clicks using polynomial interpolation from surrounding good samples.

    For runs of 1-2 bad samples, uses linear interpolation (fast, sufficient).
    For longer runs (3+), uses polynomial curve fitting to better match waveform shape.

    Args:
        samples: Audio sample array
        click_indices: Indices of samples to repair
        context_samples: Number of good samples on each side for polynomial fitting
        poly_degree: Degree of polynomial for curve fitting (3 = cubic)
    """
    repaired = samples.copy()

    if len(click_indices) == 0:
        return repaired

    click_set = set(click_indices)
    runs = group_consecutive(click_indices)

    for run in runs:
        start_idx = run[0]
        end_idx = run[-1]
        run_length = len(run)

        # Find nearest good sample before the run
        left_idx = start_idx - 1
        while left_idx >= 0 and left_idx in click_set:
            left_idx -= 1

        # Find nearest good sample after the run
        right_idx = end_idx + 1
        while right_idx < len(samples) and right_idx in click_set:
            right_idx += 1

        # Edge cases
        if left_idx < 0 and right_idx >= len(samples):
            continue  # Can't repair, no good neighbors
        elif left_idx < 0:
            # No left neighbor, use right value
            for idx in run:
                repaired[idx] = samples[right_idx]
        elif right_idx >= len(samples):
            # No right neighbor, use left value
            for idx in run:
                repaired[idx] = samples[left_idx]
        elif run_length <= 2:
            # Short runs: linear interpolation is sufficient
            left_val = samples[left_idx]
            right_val = samples[right_idx]
            span = right_idx - left_idx
            for idx in run:
                t = (idx - left_idx) / span
                repaired[idx] = left_val + t * (right_val - left_val)
        else:
            # Longer runs: polynomial interpolation for smoother curve
            # Gather context samples on each side (avoiding other bad samples)
            x_good = []
            y_good = []

            # Left context
            i = left_idx
            count = 0
            while i >= 0 and count < context_samples:
                if i not in click_set:
                    x_good.append(i)
                    y_good.append(samples[i])
                    count += 1
                i -= 1

            # Right context
            i = right_idx
            count = 0
            while i < len(samples) and count < context_samples:
                if i not in click_set:
                    x_good.append(i)
                    y_good.append(samples[i])
                    count += 1
                i += 1

            if len(x_good) < poly_degree + 1:
                # Not enough context, fall back to linear
                left_val = samples[left_idx]
                right_val = samples[right_idx]
                span = right_idx - left_idx
                for idx in run:
                    t = (idx - left_idx) / span
                    repaired[idx] = left_val + t * (right_val - left_val)
            else:
                # Fit polynomial and interpolate
                x_good = np.array(x_good)
                y_good = np.array(y_good)
                # Center x values around start_idx for numerical stability
                coeffs = np.polyfit(x_good - start_idx, y_good, poly_degree)
                poly = np.poly1d(coeffs)
                for idx in run:
                    repaired[idx] = poly(idx - start_idx)

    return repaired


def process_audio(data: np.ndarray, ratio_threshold: float = 10.0,
                  min_diff_threshold: float = 0.01,
                  do_clicks: bool = True, do_dropouts: bool = True,
                  verbose: bool = False) -> tuple[np.ndarray, int]:
    """Process audio data (mono or stereo), return repaired data and click count."""
    total_clicks = 0

    if data.ndim == 1:
        clicks = detect_clicks(data, ratio_threshold, min_diff_threshold,
                               do_clicks=do_clicks, do_dropouts=do_dropouts)
        total_clicks = len(clicks)
        if verbose and total_clicks > 0:
            print(f"  Channel 1: {total_clicks} samples detected")
        repaired = repair_clicks(data, clicks)
    else:
        repaired = np.zeros_like(data)
        for ch in range(data.shape[1]):
            clicks = detect_clicks(data[:, ch], ratio_threshold, min_diff_threshold,
                                   do_clicks=do_clicks, do_dropouts=do_dropouts)
            total_clicks += len(clicks)
            if verbose and len(clicks) > 0:
                print(f"  Channel {ch + 1}: {len(clicks)} samples detected")
            repaired[:, ch] = repair_clicks(data[:, ch], clicks)

    return repaired, total_clicks


def analyze_file(filepath: Path, do_clicks: bool = False, do_dropouts: bool = True) -> dict:
    """Analyze a file for dropouts/clicks, return stats."""
    try:
        data, sr = sf.read(filepath)
    except Exception as e:
        return {"error": str(e)}

    channels = 1 if data.ndim == 1 else data.shape[1]
    duration = len(data) / sr

    dropout_events = 0
    click_events = 0

    def count_channel(channel_data):
        d_events = c_events = 0
        if do_dropouts:
            dropouts = detect_dropouts(channel_data)
            d_events = len(group_consecutive(dropouts))
        if do_clicks:
            clicks = detect_ratio_clicks(channel_data)
            c_events = len(group_consecutive(clicks))
        return d_events, c_events

    if data.ndim == 1:
        dropout_events, click_events = count_channel(data)
    else:
        for ch in range(data.shape[1]):
            d, c = count_channel(data[:, ch])
            dropout_events += d
            click_events += c

    return {
        "filepath": filepath,
        "duration": duration,
        "channels": channels,
        "samplerate": sr,
        "dropout_events": dropout_events,
        "click_events": click_events,
    }


def expand_paths(patterns: list[str]) -> list[Path]:
    """Expand glob patterns and directories into file list."""
    files = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            # Recursively find audio files in directory
            for ext in ["*.wav", "*.aif", "*.aiff", "*.flac"]:
                files.extend(path.rglob(ext))
        elif "*" in pattern or "?" in pattern:
            files.extend(Path(p) for p in glob.glob(pattern, recursive=True))
        elif path.exists():
            files.append(path)
    return sorted(set(files))


def cmd_analyze(args):
    """Analyze files for dropouts/clicks."""
    files = expand_paths(args.inputs)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    results = []
    for filepath in files:
        stats = analyze_file(filepath, do_clicks=args.clicks, do_dropouts=args.dropouts)
        if "error" in stats:
            print(f"Error reading {filepath}: {stats['error']}", file=sys.stderr)
            continue
        results.append(stats)

    def total_events(r):
        return r["dropout_events"] + r["click_events"]

    files_with_issues = [r for r in results if total_events(r) > 0]

    if args.verbose:
        for r in results:
            parts = []
            if args.dropouts and r["dropout_events"]:
                parts.append(f"{r['dropout_events']} dropouts")
            if args.clicks and r["click_events"]:
                parts.append(f"{r['click_events']} clicks")
            status = ", ".join(parts) if parts else "clean"
            print(f"{r['filepath']}: {status}")
        print()

    print(f"Scanned {len(results)} files")
    if files_with_issues:
        total_d = sum(r["dropout_events"] for r in results)
        total_c = sum(r["click_events"] for r in results)

        summary = []
        if args.dropouts and total_d:
            summary.append(f"{total_d} dropout events")
        if args.clicks and total_c:
            summary.append(f"{total_c} click events")

        print(f"Found {', '.join(summary)} in {len(files_with_issues)} files:")
        for r in sorted(files_with_issues, key=lambda x: -total_events(x)):
            parts = []
            if args.dropouts:
                parts.append(f"{r['dropout_events']:2d}d")
            if args.clicks:
                parts.append(f"{r['click_events']:2d}c")
            print(f"  {' '.join(parts)}  {r['filepath']}")
    else:
        print("None found.")


def cmd_repair(args):
    """Repair a single file."""
    input_path = Path(args.inputs[0])
    if not input_path.exists():
        print(f"Error: File not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    # Default: in-place replacement
    output_path = Path(args.output) if args.output else input_path
    in_place = (output_path == input_path)

    if args.verbose:
        print(f"Reading: {input_path}")

    # Preserve original file metadata for in-place edits
    if in_place:
        orig_stat = os.stat(input_path)

    try:
        info = sf.info(input_path)
        data, samplerate = sf.read(input_path)
    except Exception as e:
        print(f"Error reading audio file: {e}", file=sys.stderr)
        sys.exit(1)

    mode = []
    if args.dropouts:
        mode.append("dropouts")
    if args.clicks:
        mode.append("clicks")
    mode_str = " + ".join(mode)

    if args.verbose:
        channels = 1 if data.ndim == 1 else data.shape[1]
        duration = len(data) / samplerate
        print(f"  {samplerate} Hz, {channels} channel(s), {duration:.2f}s")
        print(f"Detecting: {mode_str}")

    repaired, click_count = process_audio(
        data,
        ratio_threshold=args.ratio,
        min_diff_threshold=args.min_diff,
        do_clicks=args.clicks,
        do_dropouts=args.dropouts,
        verbose=args.verbose
    )

    if click_count == 0:
        print("Nothing detected.")
        return

    if args.backup and in_place:
        backup_path = input_path.with_suffix(".bak" + input_path.suffix)
        shutil.copy2(input_path, backup_path)
        if args.verbose:
            print(f"Backup: {backup_path}")

    try:
        sf.write(output_path, repaired, samplerate, subtype=info.subtype)
    except Exception as e:
        print(f"Error writing audio file: {e}", file=sys.stderr)
        sys.exit(1)

    # Restore original timestamps for in-place edits
    if in_place:
        os.utime(output_path, (orig_stat.st_atime, orig_stat.st_mtime))

    if in_place:
        print(f"Repaired {click_count} sample(s) in {input_path}")
    else:
        print(f"Repaired {click_count} sample(s) → {output_path}")


def repair_file(filepath: Path, do_clicks: bool, do_dropouts: bool,
                ratio_threshold: float, min_diff_threshold: float,
                verbose: bool = False, backup: bool = False) -> tuple[int, int]:
    """Repair a single file in-place. Returns (samples_fixed, events_fixed)."""
    orig_stat = os.stat(filepath)

    try:
        info = sf.info(filepath)
        data, samplerate = sf.read(filepath)
    except Exception as e:
        print(f"Error reading {filepath}: {e}", file=sys.stderr)
        return 0, 0

    repaired, sample_count = process_audio(
        data,
        ratio_threshold=ratio_threshold,
        min_diff_threshold=min_diff_threshold,
        do_clicks=do_clicks,
        do_dropouts=do_dropouts,
        verbose=False
    )

    if sample_count == 0:
        return 0, 0

    # Count events for reporting
    click_set = set()
    if data.ndim == 1:
        if do_dropouts:
            click_set.update(detect_dropouts(data))
        if do_clicks:
            click_set.update(detect_ratio_clicks(data))
    else:
        for ch in range(data.shape[1]):
            if do_dropouts:
                click_set.update(detect_dropouts(data[:, ch]))
            if do_clicks:
                click_set.update(detect_ratio_clicks(data[:, ch]))
    event_count = len(group_consecutive(np.array(sorted(click_set))))

    if backup:
        backup_path = filepath.with_suffix(".bak" + filepath.suffix)
        shutil.copy2(filepath, backup_path)

    try:
        sf.write(filepath, repaired, samplerate, subtype=info.subtype)
    except Exception as e:
        print(f"Error writing {filepath}: {e}", file=sys.stderr)
        return 0, 0

    os.utime(filepath, (orig_stat.st_atime, orig_stat.st_mtime))

    if verbose:
        print(f"  {event_count} events  {filepath}")

    return sample_count, event_count


def cmd_batch_repair(args):
    """Repair multiple files in-place."""
    files = expand_paths(args.inputs)
    if not files:
        print("No files found.", file=sys.stderr)
        sys.exit(1)

    print(f"Repairing {len(files)} files in-place...")

    total_samples = 0
    total_events = 0
    files_fixed = 0

    for filepath in files:
        samples, events = repair_file(
            filepath,
            do_clicks=args.clicks,
            do_dropouts=args.dropouts,
            ratio_threshold=args.ratio,
            min_diff_threshold=args.min_diff,
            verbose=args.verbose,
            backup=args.backup
        )
        if events > 0:
            total_samples += samples
            total_events += events
            files_fixed += 1

    if files_fixed:
        print(f"Fixed {total_events} events ({total_samples} samples) in {files_fixed} files")
    else:
        print("No issues found.")


def _build_parser(parser=None):
    if parser is None:
        parser = argparse.ArgumentParser(
            description="Detect and remove digital clicks/dropouts from audio files.",
            epilog="Examples:\n"
                   "  declick -d input.wav              Fix dropouts in single file\n"
                   "  declick -d ~/recordings/          Fix all files in folder\n"
                   "  declick -d '*.wav'                Fix all matching files\n"
                   "  declick -da ~/recordings/         Analyze only (no changes)",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
    parser.add_argument("inputs", nargs="+", help="Input file(s), glob pattern, or directory")
    parser.add_argument("-o", "--output", help="Output to different file (default: in-place)")
    parser.add_argument(
        "-d", "--dropouts", action="store_true",
        help="Detect zero-sample dropouts (e.g., ADAT sync errors)"
    )
    parser.add_argument(
        "-c", "--clicks", action="store_true",
        help="Detect single-sample clicks (ratio-based detection)"
    )
    parser.add_argument(
        "-r", "--ratio", type=float, default=10.0,
        help="Click detection ratio threshold (default: 10.0, lower=more aggressive)"
    )
    parser.add_argument(
        "-m", "--min-diff", type=float, default=0.01,
        help="Click detection minimum difference (default: 0.01)"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument("-b", "--backup", action="store_true", help="Create backup before in-place repair")
    parser.add_argument(
        "-a", "--analyze", action="store_true",
        help="Analyze files without repairing (counts events)"
    )
    return parser


def _dispatch(args):
    if not args.dropouts and not args.clicks:
        print("Error: Specify at least one detection mode: -d (dropouts) and/or -c (clicks)", file=sys.stderr)
        sys.exit(1)

    if args.analyze:
        cmd_analyze(args)
    elif len(args.inputs) == 1 and not Path(args.inputs[0]).is_dir() and "*" not in args.inputs[0]:
        cmd_repair(args)
    else:
        cmd_batch_repair(args)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("declick", help="Detect and remove digital clicks/dropouts")
    _build_parser(parser)
    parser.set_defaults(func=_dispatch)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if not args.dropouts and not args.clicks:
        parser.error("Specify at least one detection mode: -d (dropouts) and/or -c (clicks)")
    _dispatch(args)


if __name__ == "__main__":
    main()
