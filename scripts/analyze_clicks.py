#!/usr/bin/env python3
"""Analyze click detection gaps by comparing original vs fixed files.

Usage:
    python scripts/analyze_clicks.py                    # Analyze sara_* files
    python scripts/analyze_clicks.py --dump-region 12345 20  # Dump 20 samples around index 12345
    python scripts/analyze_clicks.py --file some.wav    # Analyze specific file
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from audio_tools.declick import detect_dropouts, detect_ratio_clicks, detect_clicks, group_consecutive


def phase_cancel(original: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    """Subtract fixed from original to isolate what changed."""
    return original - fixed


def find_all_discontinuities(samples: np.ndarray, threshold: float = 0.05) -> list[dict]:
    """Find ALL sudden jumps in the signal, not just zeros.

    A discontinuity is where the sample-to-sample difference is much larger
    than what the local signal trajectory would predict.
    """
    if len(samples) < 5:
        return []

    results = []

    # Method 1: Simple difference threshold - find large jumps
    diff = np.abs(np.diff(samples))
    large_jumps = np.where(diff > threshold)[0]

    for idx in large_jumps:
        # Check if this is a spike (jumps both in and out)
        if idx + 1 < len(samples) - 1:
            jump_in = abs(samples[idx + 1] - samples[idx])
            jump_out = abs(samples[idx + 2] - samples[idx + 1])
            if jump_in > threshold and jump_out > threshold:
                results.append({
                    'index': idx + 1,
                    'type': 'spike',
                    'value': samples[idx + 1],
                    'neighbors': (samples[idx], samples[idx + 2]),
                    'jump_magnitude': (jump_in + jump_out) / 2
                })

    # Method 2: Look for samples that break the local trend
    # Compare each sample to linear prediction from neighbors
    for i in range(2, len(samples) - 2):
        # Predict sample[i] from neighbors
        predicted = (samples[i - 1] + samples[i + 1]) / 2
        actual = samples[i]
        error = abs(actual - predicted)

        # Also check the continuity of neighbors (they should be similar)
        neighbor_diff = abs(samples[i - 1] - samples[i + 1])

        # If error is large relative to neighbor continuity, it's suspicious
        if neighbor_diff < 0.1 and error > 0.02:
            # This sample doesn't fit the local trajectory
            already_found = any(r['index'] == i for r in results)
            if not already_found:
                results.append({
                    'index': i,
                    'type': 'discontinuity',
                    'value': actual,
                    'predicted': predicted,
                    'neighbors': (samples[i - 1], samples[i + 1]),
                    'error': error
                })

    return sorted(results, key=lambda x: x['index'])


def find_near_zero_anomalies(samples: np.ndarray, zero_threshold: float = 0.001,
                              context_threshold: float = 0.01) -> list[dict]:
    """Find samples that are near-zero but surrounded by non-trivial signal.

    Current detector uses 1e-9 (exact zero). Real dropouts might not be
    exactly zero due to dithering or DC offset.
    """
    results = []

    for i in range(2, len(samples) - 2):
        if abs(samples[i]) < zero_threshold:
            # Check context - are neighbors significantly larger?
            context = [samples[i - 2], samples[i - 1], samples[i + 1], samples[i + 2]]
            max_context = max(abs(c) for c in context)

            if max_context > context_threshold:
                results.append({
                    'index': i,
                    'type': 'near_zero',
                    'value': samples[i],
                    'max_context': max_context,
                    'neighbors': (samples[i - 1], samples[i + 1])
                })

    return results


def analyze_file_pair(orig_path: Path, fixed_path: Path):
    """Analyze original vs fixed to understand what we caught and missed."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {orig_path.name}")
    print(f"{'=' * 60}")

    orig_data, sr = sf.read(orig_path)
    fixed_data, _ = sf.read(fixed_path)

    if orig_data.ndim == 1:
        orig_data = orig_data.reshape(-1, 1)
        fixed_data = fixed_data.reshape(-1, 1)

    channels = orig_data.shape[1]

    for ch in range(channels):
        print(f"\n--- Channel {ch + 1} ---")
        orig = orig_data[:, ch]
        fixed = fixed_data[:, ch]

        # What did we change?
        diff = phase_cancel(orig, fixed)
        changed_samples = np.where(np.abs(diff) > 1e-10)[0]

        print(f"\nSamples modified by current algorithm: {len(changed_samples)}")
        if len(changed_samples) > 0:
            runs = group_consecutive(changed_samples)
            print(f"  Number of repair events: {len(runs)}")
            for run in runs:
                start, end = run[0], run[-1]
                time_sec = start / sr
                print(f"  - Samples {start}-{end} ({end - start + 1} samples) at {time_sec:.4f}s")
                # Show what the values were
                for idx in run[:5]:  # First 5 samples of run
                    print(f"      [{idx}] orig={orig[idx]:.6f} -> fixed={fixed[idx]:.6f}")
                if len(run) > 5:
                    print(f"      ... and {len(run) - 5} more")

        # What does current detector find?
        print(f"\nCurrent detector finds:")
        dropouts = detect_dropouts(orig)
        ratio_clicks = detect_ratio_clicks(orig)
        print(f"  Dropouts (exact zeros): {len(dropouts)} samples")
        print(f"  Ratio clicks: {len(ratio_clicks)} samples")

        # What SHOULD we be finding?
        print(f"\nBroader discontinuity search:")

        # Near-zero anomalies (relaxed threshold)
        near_zeros = find_near_zero_anomalies(orig, zero_threshold=0.005, context_threshold=0.02)
        print(f"  Near-zero anomalies (|x| < 0.005 with context > 0.02): {len(near_zeros)}")
        for nz in near_zeros[:10]:
            time_sec = nz['index'] / sr
            print(f"    [{nz['index']}] at {time_sec:.4f}s: value={nz['value']:.6f}, "
                  f"neighbors=({nz['neighbors'][0]:.4f}, {nz['neighbors'][1]:.4f})")

        # General discontinuities
        disconts = find_all_discontinuities(orig, threshold=0.03)
        print(f"  General discontinuities (jump > 0.03): {len(disconts)}")

        # Filter to ones we DIDN'T catch
        dropout_set = set(dropouts)
        ratio_set = set(ratio_clicks)
        missed = [d for d in disconts if d['index'] not in dropout_set and d['index'] not in ratio_set]

        print(f"  Discontinuities NOT caught by current detector: {len(missed)}")
        for m in missed[:20]:
            time_sec = m['index'] / sr
            if m['type'] == 'spike':
                print(f"    [{m['index']}] at {time_sec:.4f}s SPIKE: value={m['value']:.6f}, "
                      f"neighbors=({m['neighbors'][0]:.4f}, {m['neighbors'][1]:.4f}), "
                      f"jump={m['jump_magnitude']:.4f}")
            else:
                print(f"    [{m['index']}] at {time_sec:.4f}s DISCONT: value={m['value']:.6f}, "
                      f"predicted={m['predicted']:.4f}, error={m['error']:.4f}")

        # Look specifically for the pattern: sample much smaller than neighbors
        print(f"\n  Checking for 'dip' pattern (sample << neighbors):")
        for i in range(2, len(orig) - 2):
            val = abs(orig[i])
            left = abs(orig[i - 1])
            right = abs(orig[i + 1])
            min_neighbor = min(left, right)

            # Is this sample much smaller than both neighbors?
            if min_neighbor > 0.01 and val < min_neighbor * 0.1:
                time_sec = i / sr
                print(f"    [{i}] at {time_sec:.4f}s: value={orig[i]:.6f}, "
                      f"neighbors=({orig[i - 1]:.4f}, {orig[i + 1]:.4f})")


def dump_region(samples: np.ndarray, center: int, radius: int, sr: int):
    """Dump samples around a specific index for detailed inspection."""
    start = max(0, center - radius)
    end = min(len(samples), center + radius + 1)

    print(f"\nSample dump around index {center} (time: {center/sr:.6f}s)")
    print(f"{'Index':>8} {'Time(s)':>12} {'Value':>12} {'|Value|':>12} {'Diff':>12} Note")
    print("-" * 70)

    prev_val = None
    for i in range(start, end):
        val = samples[i]
        diff = val - prev_val if prev_val is not None else 0
        note = ""

        if i == center:
            note = "<-- TARGET"
        elif abs(val) < 1e-9:
            note = "EXACT ZERO"
        elif abs(val) < 0.005:
            note = "near zero"

        print(f"{i:>8} {i/sr:>12.6f} {val:>12.6f} {abs(val):>12.6f} {diff:>12.6f} {note}")
        prev_val = val


def analyze_single_file(filepath: Path):
    """Analyze a single file for all potential clicks/dropouts."""
    print(f"\n{'=' * 60}")
    print(f"Analyzing: {filepath}")
    print(f"{'=' * 60}")

    data, sr = sf.read(filepath)

    if data.ndim == 1:
        data = data.reshape(-1, 1)

    channels = data.shape[1]
    print(f"Sample rate: {sr}, Channels: {channels}, Duration: {len(data)/sr:.2f}s")

    for ch in range(channels):
        print(f"\n--- Channel {ch + 1} ---")
        samples = data[:, ch]

        # Current detector results
        detected = detect_clicks(samples)
        print(f"\nDetected by current algorithm: {len(detected)} samples")

        if len(detected) > 0:
            runs = group_consecutive(detected)
            print(f"  As {len(runs)} events:")
            for run in runs[:20]:
                start, end = run[0], run[-1]
                print(f"  - [{start}:{end}] at {start/sr:.4f}s ({end-start+1} samples)")
                for idx in run[:3]:
                    print(f"      [{idx}] = {samples[idx]:.6f}")

        # Look for potential misses
        print(f"\nPotential undetected issues:")
        detected_set = set(detected)

        # Check for dips
        dip_count = 0
        for i in range(2, len(samples) - 2):
            if i in detected_set:
                continue
            val = abs(samples[i])
            left = abs(samples[i - 1])
            right = abs(samples[i + 1])
            min_neighbor = min(left, right)

            if min_neighbor > 0.02 and val < min_neighbor * 0.1:
                if dip_count < 10:
                    print(f"  DIP at [{i}] ({i/sr:.4f}s): {samples[i]:.6f}, "
                          f"neighbors=({samples[i-1]:.4f}, {samples[i+1]:.4f})")
                dip_count += 1

        if dip_count > 10:
            print(f"  ... and {dip_count - 10} more dips")

        # Check for discontinuities
        disconts = find_all_discontinuities(samples, threshold=0.05)
        missed_disconts = [d for d in disconts if d['index'] not in detected_set]
        print(f"  Discontinuities not caught: {len(missed_disconts)}")
        for d in missed_disconts[:10]:
            print(f"    [{d['index']}] at {d['index']/sr:.4f}s: {d['type']}, value={d['value']:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Analyze audio files for clicks/dropouts")
    parser.add_argument("--file", "-f", type=Path, help="Analyze a specific file")
    parser.add_argument("--dump-region", "-d", nargs=2, type=int, metavar=("INDEX", "RADIUS"),
                        help="Dump samples around INDEX with RADIUS samples on each side")
    parser.add_argument("--channel", "-c", type=int, default=0, help="Channel to dump (0-indexed)")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent

    if args.file:
        filepath = args.file if args.file.is_absolute() else repo_root / args.file
        if not filepath.exists():
            print(f"File not found: {filepath}")
            sys.exit(1)

        data, sr = sf.read(filepath)
        if data.ndim == 1:
            data = data.reshape(-1, 1)

        if args.dump_region:
            center, radius = args.dump_region
            dump_region(data[:, args.channel], center, radius, sr)
        else:
            analyze_single_file(filepath)
    else:
        # Default: analyze sara files
        file_pairs = [
            ("sara_main_ORIGINAL.wav", "sara_main_FIXED.wav"),
            ("sara_right_ORIGINAL.wav", "sara_right_FIXED.wav"),
        ]

        for orig_name, fixed_name in file_pairs:
            orig_path = repo_root / orig_name
            fixed_path = repo_root / fixed_name

            if not orig_path.exists():
                print(f"Missing: {orig_path}")
                continue
            if not fixed_path.exists():
                print(f"Missing: {fixed_path}")
                continue

            analyze_file_pair(orig_path, fixed_path)


if __name__ == "__main__":
    main()
