#!/usr/bin/env python3
"""Analyze declick repairs by comparing original and repaired audio files.

Uses phase cancellation to show exactly what was changed and provides
context around each repair to help identify false positives.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load audio file, return samples and sample rate."""
    samples, sr = sf.read(path)
    if samples.ndim > 1:
        samples = samples[:, 0]  # Take first channel for analysis
    return samples, sr


def find_changed_samples(original: np.ndarray, repaired: np.ndarray) -> np.ndarray:
    """Find indices where samples differ."""
    diff = repaired - original
    return np.where(np.abs(diff) > 1e-9)[0]


def group_consecutive(indices: np.ndarray) -> list[list[int]]:
    """Group consecutive indices into runs."""
    if len(indices) == 0:
        return []
    runs = []
    current = [indices[0]]
    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            current.append(indices[i])
        else:
            runs.append(current)
            current = [indices[i]]
    runs.append(current)
    return runs


def analyze_repair_context(original: np.ndarray, repaired: np.ndarray,
                           idx: int, context: int = 5) -> dict:
    """Analyze a single repair with surrounding context."""
    start = max(0, idx - context)
    end = min(len(original), idx + context + 1)

    orig_ctx = original[start:end]
    rep_ctx = repaired[start:end]

    # Check if this looks like a zero crossing
    left_idx = idx - 1 if idx > 0 else idx
    right_idx = idx + 1 if idx < len(original) - 1 else idx
    left_sign = original[left_idx] > 0
    right_sign = original[right_idx] > 0
    is_zero_crossing = left_sign != right_sign

    # Check if neighbors have similar magnitude (smooth signal)
    left_mag = abs(original[left_idx])
    right_mag = abs(original[right_idx])
    orig_mag = abs(original[idx])
    neighbor_ratio = min(left_mag, right_mag) / max(left_mag, right_mag) if max(left_mag, right_mag) > 0 else 1

    # Check if repair matches expected interpolation
    expected_interp = (original[left_idx] + original[right_idx]) / 2
    repair_matches_interp = abs(repaired[idx] - expected_interp) < 0.002

    # Suspicious zero at zero crossing = likely dropout
    is_suspicious_zero = is_zero_crossing and abs(original[idx]) < 0.001 and repair_matches_interp

    return {
        'index': idx,
        'original': original[idx],
        'repaired': repaired[idx],
        'delta': repaired[idx] - original[idx],
        'context_orig': orig_ctx,
        'context_rep': rep_ctx,
        'context_start': start,
        'is_zero_crossing': is_zero_crossing,
        'is_suspicious_zero': is_suspicious_zero,
        'expected_interp': expected_interp,
        'repair_matches_interp': repair_matches_interp,
        'left_neighbor': original[left_idx],
        'right_neighbor': original[right_idx],
        'neighbor_ratio': neighbor_ratio,
    }


def print_analysis(original: np.ndarray, repaired: np.ndarray,
                   sample_rate: int, max_repairs: int = 50):
    """Print detailed analysis of repairs."""
    changed = find_changed_samples(original, repaired)
    groups = group_consecutive(changed)

    print(f"Total samples changed: {len(changed)}")
    print(f"Repair events (consecutive groups): {len(groups)}")
    print(f"Duration: {len(original) / sample_rate:.2f}s @ {sample_rate}Hz")
    print()

    # Try to detect which detector flagged each sample
    try:
        from audio_tools.declick import detect_dropouts, detect_ratio_clicks, detect_sync_artifacts
        dropout_set = set(detect_dropouts(original))
        ratio_set = set(detect_ratio_clicks(original))
        sync_set = set(detect_sync_artifacts(original))
        has_detectors = True

        print("Detection breakdown:")
        print(f"  dropout detector:  {len(dropout_set)} samples")
        print(f"  ratio detector:    {len(ratio_set)} samples")
        print(f"  sync detector:     {len(sync_set)} samples")
        print()
    except ImportError:
        has_detectors = False
        dropout_set = ratio_set = sync_set = set()

    # Categorize repairs
    suspicious_zeros = []  # Exact zeros at zero crossings - likely real dropouts
    false_positive_candidates = []  # Non-zero values at zero crossings
    likely_dropouts = []
    sync_artifacts = []
    uncertain = []

    for group in groups:
        first_idx = group[0]
        info = analyze_repair_context(original, repaired, first_idx)
        info['group_size'] = len(group)
        info['group'] = group

        # Determine which detector flagged this
        if has_detectors:
            detectors = []
            if first_idx in dropout_set:
                detectors.append('dropout')
            if first_idx in ratio_set:
                detectors.append('ratio')
            if first_idx in sync_set:
                detectors.append('sync')
            info['detectors'] = detectors
        else:
            info['detectors'] = []

        # Categorize based on characteristics
        if 'sync' in info['detectors'] and 'dropout' not in info['detectors']:
            sync_artifacts.append(info)
        elif info['is_suspicious_zero']:
            suspicious_zeros.append(info)  # Exact zero at crossing = likely dropout
        elif info['is_zero_crossing'] and abs(info['original']) > 0.01:
            false_positive_candidates.append(info)  # Non-trivial value at crossing
        elif abs(info['original']) < 0.01:
            likely_dropouts.append(info)
        else:
            uncertain.append(info)

    # Print summary
    print("=" * 60)
    print("CATEGORIZED REPAIRS")
    print("=" * 60)

    if sync_artifacts:
        print(f"\n⚠️  SYNC ARTIFACTS ({len(sync_artifacts)}) - smooth audio flagged as artifacts:")
        for info in sync_artifacts[:8]:
            det = ', '.join(info['detectors'])
            print(f"  [{info['index']}] {info['original']:.6f} -> {info['repaired']:.6f}  [{det}]")
            print(f"       context: {np.round(info['context_orig'], 4)}")
        if len(sync_artifacts) > 8:
            print(f"  ... and {len(sync_artifacts) - 8} more")

    if false_positive_candidates:
        print(f"\n⚠️  ZERO CROSSING FALSE POSITIVES ({len(false_positive_candidates)}):")
        for info in false_positive_candidates[:8]:
            det = ', '.join(info['detectors'])
            print(f"  [{info['index']}] {info['original']:.6f} -> {info['repaired']:.6f}  [{det}]")
            print(f"       neighbors: {info['left_neighbor']:.4f} ... {info['right_neighbor']:.4f}")
        if len(false_positive_candidates) > 8:
            print(f"  ... and {len(false_positive_candidates) - 8} more")

    if suspicious_zeros:
        print(f"\n✓ SUSPICIOUS ZEROS ({len(suspicious_zeros)}) - exact zeros at crossings, likely dropouts:")
        for info in suspicious_zeros[:8]:
            det = ', '.join(info['detectors'])
            print(f"  [{info['index']}] {info['original']:.6f} -> {info['repaired']:.6f}  [{det}]")
            print(f"       expected: {info['expected_interp']:.6f}  (repair matches: {info['repair_matches_interp']})")
        if len(suspicious_zeros) > 8:
            print(f"  ... and {len(suspicious_zeros) - 8} more")

    if likely_dropouts:
        print(f"\n✓ LIKELY DROPOUTS ({len(likely_dropouts)}) - near-zero values repaired:")
        for info in likely_dropouts[:8]:
            det = ', '.join(info['detectors'])
            size_str = f"({info['group_size']} samples)" if info['group_size'] > 1 else ""
            print(f"  [{info['index']}] {info['original']:.6f} -> {info['repaired']:.6f} {size_str}  [{det}]")
        if len(likely_dropouts) > 8:
            print(f"  ... and {len(likely_dropouts) - 8} more")

    if uncertain:
        print(f"\n? UNCERTAIN ({len(uncertain)}) - review manually:")
        for info in uncertain[:8]:
            det = ', '.join(info['detectors'])
            print(f"  [{info['index']}] {info['original']:.6f} -> {info['repaired']:.6f}  [{det}]")
            print(f"       context: {np.round(info['context_orig'], 4)}")
        if len(uncertain) > 8:
            print(f"  ... and {len(uncertain) - 8} more")

    print()

    # Summary
    total_issues = len(sync_artifacts) + len(false_positive_candidates)
    total_good = len(suspicious_zeros) + len(likely_dropouts)
    print(f"Summary: {total_good} likely good repairs, {total_issues} potential false positives, {len(uncertain)} uncertain")

    return {
        'sync_artifacts': sync_artifacts,
        'false_positive_candidates': false_positive_candidates,
        'suspicious_zeros': suspicious_zeros,
        'likely_dropouts': likely_dropouts,
        'uncertain': uncertain,
    }


def plot_repairs(original: np.ndarray, repaired: np.ndarray,
                 sample_rate: int, output_path: str = None):
    """Generate plots showing repairs in context."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping plots")
        return

    changed = find_changed_samples(original, repaired)
    groups = group_consecutive(changed)

    if not groups:
        print("No repairs to plot")
        return

    # Plot up to 6 repair events
    n_plots = min(6, len(groups))
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 2.5 * n_plots))
    if n_plots == 1:
        axes = [axes]

    for i, group in enumerate(groups[:n_plots]):
        ax = axes[i]
        center = group[len(group) // 2]
        context = 50  # samples on each side
        start = max(0, center - context)
        end = min(len(original), center + context)

        x = np.arange(start, end)
        ax.plot(x, original[start:end], 'b-', alpha=0.7, label='Original')
        ax.plot(x, repaired[start:end], 'r--', alpha=0.7, label='Repaired')

        # Mark the changed samples
        for idx in group:
            if start <= idx < end:
                ax.axvline(idx, color='orange', alpha=0.3, linewidth=2)
                ax.plot(idx, original[idx], 'bo', markersize=8)
                ax.plot(idx, repaired[idx], 'r^', markersize=8)

        ax.set_title(f"Repair at sample {center} ({center/sample_rate:.4f}s)")
        ax.set_xlabel("Sample")
        ax.set_ylabel("Amplitude")
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150)
        print(f"Plot saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('original', help='Original audio file')
    parser.add_argument('repaired', help='Repaired audio file')
    parser.add_argument('--plot', '-p', action='store_true', help='Generate plots')
    parser.add_argument('--plot-output', '-o', help='Save plot to file instead of showing')
    parser.add_argument('--max-repairs', '-m', type=int, default=50,
                        help='Max repairs to show in detail')
    args = parser.parse_args()

    if not Path(args.original).exists():
        print(f"Error: {args.original} not found")
        sys.exit(1)
    if not Path(args.repaired).exists():
        print(f"Error: {args.repaired} not found")
        sys.exit(1)

    print(f"Loading {args.original}...")
    original, sr = load_audio(args.original)

    print(f"Loading {args.repaired}...")
    repaired, _ = load_audio(args.repaired)

    if len(original) != len(repaired):
        print(f"Error: file lengths differ ({len(original)} vs {len(repaired)})")
        sys.exit(1)

    print()
    results = print_analysis(original, repaired, sr, args.max_repairs)

    if args.plot:
        plot_repairs(original, repaired, sr, args.plot_output)

    # Exit with error if there are definite false positives
    fp_count = len(results['false_positive_candidates'])
    sync_count = len(results['sync_artifacts'])
    if fp_count > 0:
        print(f"⚠️  Found {fp_count} likely false positives at zero crossings!")
        sys.exit(1)
    if sync_count > 0:
        print(f"⚠️  Found {sync_count} sync artifact false positives (smooth audio flagged)")
        sys.exit(2)


if __name__ == '__main__':
    main()
