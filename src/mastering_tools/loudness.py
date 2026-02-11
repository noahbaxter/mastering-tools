#!/usr/bin/env python3
"""
Audio loudness analysis for mastering workflow.
Wraps ffmpeg's ebur128 filter to extract peak, LUFS, and loudness change timestamps.
"""

import shutil
import subprocess
import sys
import re
import statistics
from pathlib import Path


def run_ffmpeg_analysis(filepath: str) -> tuple[str, str | None]:
    """Run ffmpeg with ebur128 filter and return (stderr output, error message)."""
    cmd = [
        "ffmpeg", "-i", filepath,
        "-filter_complex", "ebur128=peak=true:framelog=info",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Check for common errors
    if "Invalid data found when processing input" in result.stderr:
        return "", "file may be online-only (Dropbox/iCloud) - sync it first"
    if "No such file or directory" in result.stderr:
        return "", "file not found"
    if result.returncode != 0 and not result.stderr:
        return "", "ffmpeg failed to process file"

    return result.stderr, None


def parse_ebur128_output(output: str) -> dict:
    """Parse ffmpeg ebur128 output."""
    data = {
        "momentary_values": [],
        "short_term_values": [],
        "timestamps": [],
        "true_peak": None,
        "integrated_lufs": None,
        "loudness_range": None,
    }

    # Parse frame-by-frame momentary and short-term loudness
    # Format: [Parsed_ebur128_0 @ ...] t: 0.0999773  TARGET:-23 LUFS    M:-120.7 S:-120.7 ...
    frame_pattern = re.compile(r"t:\s*([\d.]+)\s+TARGET.*?M:\s*([-\d.]+)\s+S:\s*([-\d.]+)")

    for line in output.split("\n"):
        if "Parsed_ebur128" in line and "M:" in line and "TARGET" in line:
            match = frame_pattern.search(line)
            if match:
                timestamp = float(match.group(1))
                momentary = float(match.group(2))
                short_term = float(match.group(3))
                if momentary > -120:
                    data["timestamps"].append(timestamp)
                    data["momentary_values"].append(momentary)
                    data["short_term_values"].append(short_term)

            # Also grab true peak from frame data (TPK field at end)
            tpk_match = re.search(r"TPK:\s*([-\d.]+)\s+([-\d.]+)\s*dBFS", line)
            if tpk_match:
                peak_l = float(tpk_match.group(1))
                peak_r = float(tpk_match.group(2))
                frame_peak = max(peak_l, peak_r)
                if data["true_peak"] is None or frame_peak > data["true_peak"]:
                    data["true_peak"] = frame_peak

    # Parse summary section for integrated loudness and LRA
    in_summary = False
    for line in output.split("\n"):
        if "Summary:" in line:
            in_summary = True
        if in_summary:
            # Integrated loudness: "    I:         -27.8 LUFS"
            if "I:" in line and "LUFS" in line:
                match = re.search(r"I:\s*([-\d.]+)\s*LUFS", line)
                if match:
                    data["integrated_lufs"] = float(match.group(1))
            # LRA: "    LRA:        20.0 LU"
            if line.strip().startswith("LRA:"):
                match = re.search(r"LRA:\s*([\d.]+)\s*LU", line)
                if match:
                    data["loudness_range"] = float(match.group(1))
            # Peak: "    Peak:      -24.1 dBFS" (summary peak, single value for mono)
            if line.strip().startswith("Peak:"):
                match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", line)
                if match:
                    summary_peak = float(match.group(1))
                    if data["true_peak"] is None or summary_peak > data["true_peak"]:
                        data["true_peak"] = summary_peak

    return data


def find_loudness_changes(timestamps: list, values: list, threshold_db: float = 3.0, window_sec: float = 2.0) -> list:
    """
    Find timestamps where average loudness changes significantly.
    Uses a sliding window comparison to detect jumps.
    """
    if len(values) < 10:
        return []

    changes = []
    # Estimate samples per window based on typical 100ms ebur128 frame rate
    avg_interval = (timestamps[-1] - timestamps[0]) / len(timestamps) if len(timestamps) > 1 else 0.1
    window_samples = max(3, int(window_sec / avg_interval))

    i = window_samples
    while i < len(values) - window_samples:
        prev_window = values[i - window_samples:i]
        next_window = values[i:i + window_samples]

        prev_avg = statistics.mean(prev_window)
        next_avg = statistics.mean(next_window)
        delta = next_avg - prev_avg

        if abs(delta) >= threshold_db:
            changes.append({
                "timestamp": timestamps[i],
                "delta_db": round(delta, 1),
                "from_lufs": round(prev_avg, 1),
                "to_lufs": round(next_avg, 1),
            })
            # Skip ahead to avoid duplicate detections
            i += window_samples
        else:
            i += 1

    return changes


from mastering_tools.utils import format_timestamp, format_time_short


def _find_transition_indices(values: list, window_samples: int, min_segment_samples: int, threshold_db: float) -> list[int]:
    """Two-pass transition detection: coarse scan then narrow refinement."""
    # Pass 1: coarse 2-second window, find peak delta in each crossing region
    candidates = []
    i = window_samples
    last_transition = 0
    while i < len(values) - window_samples:
        if i - last_transition < min_segment_samples:
            i += 1
            continue

        prev_avg = statistics.mean(values[i - window_samples:i])
        next_avg = statistics.mean(values[i:i + window_samples])
        delta = abs(next_avg - prev_avg)

        if delta >= threshold_db:
            # Scan forward to find where delta peaks within this crossing
            best_idx, best_delta = i, delta
            for j in range(i + 1, min(len(values) - window_samples, i + window_samples)):
                dj = abs(statistics.mean(values[j:j + window_samples]) -
                         statistics.mean(values[j - window_samples:j]))
                if dj > best_delta:
                    best_delta, best_idx = dj, j
            candidates.append(best_idx)
            last_transition = best_idx
            i = best_idx + window_samples
        else:
            i += 1

    # Pass 2: refine with narrow window (0.5s ≈ window_samples/4)
    narrow = max(2, window_samples // 4)
    neighborhood = max(3, window_samples // 2)
    refined = []
    for idx in candidates:
        best_idx, best_delta = idx, 0
        scan_start = max(narrow, idx - neighborhood)
        scan_end = min(len(values) - narrow, idx + neighborhood)
        for j in range(scan_start, scan_end):
            dj = abs(statistics.mean(values[j:j + narrow]) -
                     statistics.mean(values[j - narrow:j]))
            if dj > best_delta:
                best_delta, best_idx = dj, j
        refined.append(best_idx)

    return refined


def find_segments(timestamps: list, values: list, threshold_db: float = 4.0, min_segment_sec: float = 5.0) -> list:
    """
    Find distinct loudness segments in a track.
    Returns list of segments with start/end times and average loudness.
    """
    if len(values) < 10:
        return []

    # Estimate time per sample
    avg_interval = (timestamps[-1] - timestamps[0]) / len(timestamps) if len(timestamps) > 1 else 0.1
    window_samples = max(3, int(2.0 / avg_interval))  # 2 second window for smoothing
    min_segment_samples = max(5, int(min_segment_sec / avg_interval))

    transitions = [0]
    transitions.extend(_find_transition_indices(values, window_samples, min_segment_samples, threshold_db))
    transitions.append(len(values) - 1)

    # Build segments from transitions
    segments = []
    for j in range(len(transitions) - 1):
        start_idx = transitions[j]
        end_idx = transitions[j + 1]

        segment_values = values[start_idx:end_idx]
        if len(segment_values) < 3:
            continue

        # Filter out silence for average calculation
        audible_values = [v for v in segment_values if v > -60]
        if not audible_values:
            avg_lufs = -70  # Silence
        else:
            avg_lufs = statistics.mean(audible_values)

        segments.append({
            "start_time": timestamps[start_idx],
            "end_time": timestamps[end_idx] if end_idx < len(timestamps) else timestamps[-1],
            "avg_lufs": avg_lufs,
            "min_lufs": min(segment_values),
            "max_lufs": max(segment_values),
        })

    return segments


def analyze_segments(filepath: str, threshold: float = 4.0, min_len: float = 5.0, target_lufs: float = None):
    """Analyze a track and show segment-by-segment gain adjustments."""
    stats = get_audio_stats(filepath)
    if not stats:
        sys.exit(1)

    print(f"Analyzing segments: {stats['name']}")
    print("-" * 60)

    segments = find_segments(
        stats["timestamps"],
        stats["momentary_values"],
        threshold_db=threshold,
        min_segment_sec=min_len
    )

    if not segments:
        print("Could not detect distinct segments.")
        return {
            "name": stats["name"], "body_avg": None, "body_gain": None,
            "true_peak": stats["true_peak"], "peak_after_gain": None,
            "integrated_lufs": stats["integrated_lufs"],
            "loudness_range": stats["loudness_range"], "is_wall": False,
        }

    # Find loudest segment (by average) for relative mode and labeling
    audible = [s["avg_lufs"] for s in segments if s["avg_lufs"] > -60]
    if not audible:
        print("All segments are silence.")
        return {
            "name": stats["name"], "body_avg": None, "body_gain": None,
            "true_peak": stats["true_peak"], "peak_after_gain": None,
            "integrated_lufs": stats["integrated_lufs"],
            "loudness_range": stats["loudness_range"], "is_wall": False,
        }
    loudest_avg = max(audible)

    # Label segments by type relative to loudest
    body_threshold = 4.0    # within this of loudest = body of the song
    brief_threshold = 3.0   # seconds — too short to automate

    for i, seg in enumerate(segments):
        duration = seg["end_time"] - seg["start_time"]
        delta = loudest_avg - seg["avg_lufs"]

        if seg["avg_lufs"] <= -60:
            seg["label"] = "silence"
        elif delta <= body_threshold:
            # Close to loudest — this is the body of the song
            if duration < brief_threshold:
                seg["label"] = "brief"
            else:
                seg["label"] = "body"
        else:
            # Quieter than body
            if i == 0:
                seg["label"] = "intro"
            elif i == len(segments) - 1:
                seg["label"] = "tail"
            elif duration < brief_threshold:
                seg["label"] = "brief"
            else:
                seg["label"] = "dip"

    body_segs = [s for s in segments if s["label"] == "body"]
    dip_segs = [s for s in segments if s["label"] == "dip"]

    # Wall-to-wall: only 1 body section, no dips
    is_wall = len(body_segs) <= 1 and not dip_segs and body_segs

    label_display = {
        "body": "LOUD", "dip": "soft", "intro": "·",
        "tail": "·", "brief": "·", "silence": "·",
    }

    print(f"\n{'#':<4} {'TIME RANGE':<14} {'LUFS':>7} {'GAIN':>9}  {'':>4}")
    print("-" * 46)

    for i, seg in enumerate(segments, 1):
        time_range = f"{format_time_short(seg['start_time'])} - {format_time_short(seg['end_time'])}"
        label = seg["label"]
        tag = label_display.get(label, "")

        if label == "silence":
            avg_str = "silence"
            gain_str = "-"
        elif target_lufs is not None:
            avg_str = f"{seg['avg_lufs']:+.1f}"
            gain = target_lufs - seg["avg_lufs"]
            gain_str = f"{gain:+.1f} dB"
        else:
            avg_str = f"{seg['avg_lufs']:+.1f}"
            gain = loudest_avg - seg["avg_lufs"]
            gain_str = f"{gain:+.1f} dB" if abs(gain) > 0.5 else "ok"

        print(f"{i:<4} {time_range:<14} {avg_str:>7} {gain_str:>9}  {tag:>4}")

    # Compute body stats for return value and output
    body_avg = statistics.mean(s["avg_lufs"] for s in body_segs) if body_segs else None
    body_gain = (target_lufs - body_avg) if target_lufs is not None and body_avg is not None else None
    peak_after_gain = (stats["true_peak"] + body_gain) if body_gain is not None else None

    int_str = f"{stats['integrated_lufs']:+.1f} LUFS" if stats['integrated_lufs'] else "N/A"
    lra_str = f"{stats['loudness_range']:.1f} LU" if stats['loudness_range'] else "N/A"
    print(f"\n  Integrated: {int_str}    LRA: {lra_str}")

    peak_line = f"  True peak: {stats['true_peak']:+.1f} dBFS"
    if peak_after_gain is not None:
        peak_line += f" → {peak_after_gain:+.1f} after gain"
        if peak_after_gain > -0.3:
            peak_line += "  !! WILL CLIP"
    print(peak_line)

    if is_wall:
        seg = body_segs[0]
        if target_lufs is not None:
            print(f"\n  Wall-to-wall → {body_gain:+.1f} dB to target")
        else:
            print(f"\n  Wall-to-wall at {seg['avg_lufs']:+.1f} LUFS")
    else:
        if body_segs:
            body_min = min(s["avg_lufs"] for s in body_segs)
            body_max = max(s["avg_lufs"] for s in body_segs)
            body_spread = body_max - body_min
            body_time = sum(s["end_time"] - s["start_time"] for s in body_segs)
            total_time = segments[-1]["end_time"] - segments[0]["start_time"]

            if target_lufs is not None:
                print(f"\n  Body: {len(body_segs)} section{'s' if len(body_segs) != 1 else ''}"
                      f" avg {body_avg:+.1f} LUFS → {body_gain:+.1f} dB to target"
                      f"  ({format_time_short(body_time)}/{format_time_short(total_time)})")
            else:
                print(f"\n  Body: {len(body_segs)} section{'s' if len(body_segs) != 1 else ''}"
                      f" avg {body_avg:+.1f} LUFS"
                      f"  ({format_time_short(body_time)}/{format_time_short(total_time)})")

            if body_spread < 1.5:
                print(f"  Consistent ({body_spread:.1f} dB spread) — no per-section automation needed")
            elif body_spread > 2.0:
                print(f"  Spread: {body_spread:.1f} dB across body ({body_min:+.1f} to {body_max:+.1f})")

        if dip_segs:
            dip_count = len(dip_segs)
            dip_avg = statistics.mean(s["avg_lufs"] for s in dip_segs)
            print(f"  {dip_count} dip{'s' if dip_count != 1 else ''}"
                  f" avg {dip_avg:+.1f} LUFS — intentional dynamics, don't flatten")

    return {
        "name": stats["name"], "body_avg": body_avg, "body_gain": body_gain,
        "true_peak": stats["true_peak"], "peak_after_gain": peak_after_gain,
        "integrated_lufs": stats["integrated_lufs"],
        "loudness_range": stats["loudness_range"], "is_wall": is_wall,
    }


def stats_from_output(filepath: str, output: str) -> dict | None:
    """Parse ffmpeg ebur128 output into stats dict."""
    path = Path(filepath)
    data = parse_ebur128_output(output)

    if not data["momentary_values"]:
        return None

    values = data["momentary_values"]
    st_values = data["short_term_values"]
    return {
        "name": path.name,
        "path": filepath,
        "true_peak": data["true_peak"] if data["true_peak"] is not None else -100,
        "integrated_lufs": data["integrated_lufs"],
        "min_lufs": min(values),
        "max_lufs": max(values),
        "median_lufs": statistics.median(values),
        "loudness_range": data["loudness_range"],
        "timestamps": data["timestamps"],
        "momentary_values": values,
        "short_term_values": st_values,
    }


def get_audio_stats(filepath: str) -> dict | None:
    """Analyze a single file and return stats dict."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None

    output, error = run_ffmpeg_analysis(filepath)
    if error:
        print(f"Error: {path.name} - {error}", file=sys.stderr)
        return None

    stats = stats_from_output(filepath, output)
    if stats is None:
        print(f"Error: Could not parse loudness data for {path.name}", file=sys.stderr)
    return stats


def analyze_single(filepath: str, change_threshold: float = 3.0):
    """Detailed analysis for a single file."""
    stats = get_audio_stats(filepath)
    if not stats:
        sys.exit(1)

    print(f"Analyzing: {stats['name']}")
    print("-" * 50)

    true_peak = stats["true_peak"]

    # Section 1: Peak/Clipping info
    print("\n[PEAK / HEADROOM]")
    print(f"  True Peak:     {true_peak:+.1f} dBFS")
    if true_peak > 0:
        print(f"  CLIPPING by:   {true_peak:.1f} dB")
        print(f"  → Turn down at least {true_peak + 0.5:.1f} dB in DAW to avoid clipping")
    elif true_peak > -1:
        print(f"  Headroom:      {-true_peak:.1f} dB (tight!)")
        print(f"  → Consider turning down {true_peak + 1:.1f} dB for -1 dBFS ceiling")
    else:
        print(f"  Headroom:      {-true_peak:.1f} dB")
        print(f"  → Safe. Could gain up to {-true_peak - 1:.1f} dB and stay under -1 dBFS")

    # Section 2: LUFS loudness
    print("\n[LOUDNESS (LUFS)]")
    print(f"  Integrated:    {stats['integrated_lufs']:+.1f} LUFS" if stats['integrated_lufs'] else "  Integrated:    N/A")
    print(f"  Momentary Min: {stats['min_lufs']:+.1f} LUFS")
    print(f"  Momentary Med: {stats['median_lufs']:+.1f} LUFS")
    print(f"  Momentary Max: {stats['max_lufs']:+.1f} LUFS")
    if stats['loudness_range']:
        print(f"  Loudness Range:{stats['loudness_range']:+.1f} LU")

    # Target loudness suggestions
    print("\n[TARGET SUGGESTIONS]")
    targets = [
        ("Spotify/YouTube", -14),
        ("Apple Music", -16),
        ("Club/DJ", -6),
    ]
    for name, target in targets:
        diff = target - stats['max_lufs']
        print(f"  {name:14} ({target:+d} LUFS): {diff:+.1f} dB adjustment on loudest sections")

    # Section 3: Loudness changes
    changes = find_loudness_changes(
        stats["timestamps"],
        stats["momentary_values"],
        threshold_db=change_threshold
    )

    if changes:
        print(f"\n[LOUDNESS CHANGES (>{change_threshold} dB)]")
        for c in changes:
            direction = "↑" if c["delta_db"] > 0 else "↓"
            print(f"  {format_timestamp(c['timestamp']):>7}  {direction} {c['delta_db']:+.1f} dB  ({c['from_lufs']:+.1f} → {c['to_lufs']:+.1f} LUFS)")
    else:
        print(f"\n[LOUDNESS CHANGES]")
        print(f"  No major changes detected (threshold: {change_threshold} dB)")


def calculate_target_lufs(stats_list: list[dict], ceiling: float = -1.0) -> float:
    """Calculate target LUFS that matches all tracks while staying under ceiling."""
    quietest_max_lufs = min(s["max_lufs"] for s in stats_list)

    # Calculate what each track's peak would be after LUFS matching
    peaks_after_match = []
    for s in stats_list:
        lufs_gain = quietest_max_lufs - s["max_lufs"]
        peak_after = s["true_peak"] + lufs_gain
        peaks_after_match.append(peak_after)

    hottest_peak = max(peaks_after_match)

    # If any track would still clip, apply extra reduction
    extra_headroom = 0.0
    if hottest_peak > ceiling:
        extra_headroom = ceiling - hottest_peak

    return quietest_max_lufs + extra_headroom


def analyze_batch(filepaths: list[str], ceiling: float = -1.0):
    """Batch analysis for multiple files - calculates gain adjustments."""
    print(f"Analyzing {len(filepaths)} files...\n")

    all_stats = []
    for fp in filepaths:
        stats = get_audio_stats(fp)
        if stats:
            all_stats.append(stats)
        else:
            print(f"  FAILED: {Path(fp).name}")

    if not all_stats:
        print("No files successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    target_lufs = calculate_target_lufs(all_stats, ceiling)

    # Sort by track name for consistent ordering
    all_stats.sort(key=lambda s: s["name"])

    # Calculate max name length for alignment
    max_name = min(50, max(len(s["name"]) for s in all_stats))

    # Print header
    print(f"{'TRACK':{max_name}}  {'PEAK':>8}  {'GAIN':>8}")
    print("-" * (max_name + 22))

    for s in all_stats:
        name = s["name"][:max_name]

        # Final gain to apply (matches LUFS AND prevents clipping)
        final_gain = target_lufs - s["max_lufs"]

        peak_str = f"{s['true_peak']:+.1f}"
        gain_str = f"{final_gain:+.1f} dB" if final_gain != 0 else "ok"

        print(f"{name:{max_name}}  {peak_str:>8}  {gain_str:>8}")

    print()
    print(f"PEAK = current true peak (dBFS)")
    print(f"GAIN = apply this to match loudness AND stay under {ceiling:+.1f} dBFS")
    print(f"       (targeting {target_lufs:+.1f} LUFS)")


def analyze_against_refs(track_paths: list[str], ref_paths: list[str], ceiling: float = -1.0):
    """Compare unmastered tracks against references to get ballpark gain."""
    print(f"Analyzing {len(ref_paths)} references...")

    # Analyze references first
    ref_stats = []
    for fp in ref_paths:
        stats = get_audio_stats(fp)
        if stats:
            ref_stats.append(stats)

    if not ref_stats:
        print("No reference files successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    ref_target_lufs = calculate_target_lufs(ref_stats, ceiling)

    # Show reference gains
    print(f"\n{'REFERENCE':{50}}  {'GAIN':>10}")
    print("-" * 64)

    ref_stats.sort(key=lambda s: s["name"])
    for s in ref_stats:
        name = s["name"][:50]
        ref_gain = ref_target_lufs - s["max_lufs"]
        gain_str = f"{ref_gain:+.1f} dB" if ref_gain != 0 else "ok"
        print(f"{name:{50}}  {gain_str:>10}")

    print(f"\n  → Target: {ref_target_lufs:+.1f} LUFS (under {ceiling:+.1f} dBFS)\n")

    # Now analyze unmastered tracks
    print(f"Analyzing {len(track_paths)} tracks...\n")

    track_stats = []
    for fp in track_paths:
        stats = get_audio_stats(fp)
        if stats:
            track_stats.append(stats)
        else:
            print(f"  FAILED: {Path(fp).name}")

    if not track_stats:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    # Sort by name
    track_stats.sort(key=lambda s: s["name"])

    max_name = min(45, max(len(s["name"]) for s in track_stats))

    print(f"{'TRACK':{max_name}}  {'MAX':>8}  {'TARGET':>8}  {'BALLPARK':>10}")
    print("-" * (max_name + 32))

    for s in track_stats:
        name = s["name"][:max_name]
        current_lufs = s["max_lufs"]
        ballpark_gain = ref_target_lufs - current_lufs

        now_str = f"{current_lufs:+.1f}"
        target_str = f"{ref_target_lufs:+.1f}"
        gain_str = f"{ballpark_gain:+.1f} dB"

        print(f"{name:{max_name}}  {now_str:>8}  {target_str:>8}  {gain_str:>10}")

    print()
    print(f"MAX      = loudest section (momentary LUFS)")
    print(f"TARGET   = reference loudest sections ({ref_target_lufs:+.1f} LUFS)")
    print(f"BALLPARK = gain to reach target (you'll need limiting to not clip)")


def compare_to_refs(track_paths: list[str], ref_paths: list[str]):
    """Compare mastered tracks against references - show how close they are."""
    print(f"Analyzing {len(ref_paths) + len(track_paths)} files...\n")

    ref_stats = []
    track_stats = []

    for fp in ref_paths:
        stats = get_audio_stats(fp)
        if stats:
            ref_stats.append(stats)

    for fp in track_paths:
        stats = get_audio_stats(fp)
        if stats:
            track_stats.append(stats)

    if not ref_stats:
        print("No reference files successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    if not track_stats:
        print("No tracks successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    # Calculate reference averages
    ref_int_avg = statistics.mean(s["integrated_lufs"] for s in ref_stats if s["integrated_lufs"])
    ref_peak_avg = statistics.mean(s["true_peak"] for s in ref_stats)
    ref_lra_values = [s["loudness_range"] for s in ref_stats if s["loudness_range"]]
    ref_lra_avg = statistics.mean(ref_lra_values) if ref_lra_values else None

    # Print reference stats
    max_name = min(40, max(len(s["name"]) for s in ref_stats + track_stats))

    print(f"{'REFERENCES':{max_name}}  {'I-LUFS':>8}  {'PEAK':>8}  {'LRA':>6}")
    print("-" * (max_name + 28))

    ref_stats.sort(key=lambda s: s["name"])
    for s in ref_stats:
        name = s["name"][:max_name]
        lufs_str = f"{s['integrated_lufs']:+.1f}" if s['integrated_lufs'] else "N/A"
        peak_str = f"{s['true_peak']:+.1f}"
        lra_str = f"{s['loudness_range']:.1f}" if s['loudness_range'] else "-"
        print(f"{name:{max_name}}  {lufs_str:>8}  {peak_str:>8}  {lra_str:>6}")

    # Reference average line
    lra_avg_str = f"{ref_lra_avg:.1f}" if ref_lra_avg else "-"
    print(f"{'  (average)':{max_name}}  {ref_int_avg:+.1f}  {ref_peak_avg:+.1f}  {lra_avg_str:>6}")

    # Print tracks with delta from reference
    print(f"\n{'YOUR TRACKS':{max_name}}  {'I-LUFS':>8}  {'PEAK':>8}  {'LRA':>6}  {'DELTA':>8}")
    print("-" * (max_name + 38))

    track_stats.sort(key=lambda s: s["name"])
    for s in track_stats:
        name = s["name"][:max_name]
        lufs = s['integrated_lufs']
        lufs_str = f"{lufs:+.1f}" if lufs else "N/A"
        peak_str = f"{s['true_peak']:+.1f}"
        lra_str = f"{s['loudness_range']:.1f}" if s['loudness_range'] else "-"

        # Delta from reference average
        if lufs:
            delta = lufs - ref_int_avg
            delta_str = f"{delta:+.1f}"
        else:
            delta_str = "-"

        print(f"{name:{max_name}}  {lufs_str:>8}  {peak_str:>8}  {lra_str:>6}  {delta_str:>8}")

    print()
    print(f"I-LUFS = integrated loudness (whole track average)")
    print(f"PEAK  = true peak (dBFS)")
    print(f"LRA   = loudness range (dynamic range)")
    print(f"DELTA = difference from reference average ({ref_int_avg:+.1f} LUFS)")
    print(f"        negative = quieter, positive = louder than refs")


def _build_parser(parser=None):
    import argparse
    if parser is None:
        parser = argparse.ArgumentParser(description="Audio loudness analysis for mastering")
    parser.add_argument("files", nargs="*", help="Audio file(s) to analyze")
    parser.add_argument("-r", "--refs", nargs="+", help="Reference tracks to compare against")
    parser.add_argument("-cmp", "--compare", action="store_true",
                        help="Compare mode - see how close your masters are to references")
    parser.add_argument("-s", "--segments", action="store_true",
                        help="Segment analysis mode - detect sections and show per-segment gain")
    parser.add_argument("-ch", "--changes", action="store_true",
                        help="Show timestamps where loudness changes significantly")
    parser.add_argument("-t", "--threshold", type=float, default=3.0,
                        help="Loudness change detection threshold in dB (default: 3.0)")
    parser.add_argument("-m", "--min-segment", type=float, default=5.0,
                        help="Minimum segment length in seconds (default: 5.0)")
    parser.add_argument("-c", "--ceiling", type=float, default=-1.0,
                        help="Target ceiling in dBFS (default: -1.0)")
    return parser


def _print_segment_summary(results: list[dict]):
    max_name = min(30, max(len(r["name"]) for r in results))
    print(f"\nSUMMARY")
    print(f"{'TRACK':{max_name}}  {'BODY':>7}  {'GAIN':>7}  {'PEAK':>7}  {'AFTER':>7}")
    print("-" * (max_name + 36))
    for r in results:
        name = r["name"][:max_name]
        body_str = f"{r['body_avg']:+.1f}" if r["body_avg"] is not None else "-"
        gain_str = f"{r['body_gain']:+.1f}" if r["body_gain"] is not None else "-"
        peak_str = f"{r['true_peak']:+.1f}"
        if r["peak_after_gain"] is not None:
            after_str = f"{r['peak_after_gain']:+.1f}"
            flag = "  !! CLIP" if r["peak_after_gain"] > -0.3 else "  ok"
        else:
            after_str = "-"
            flag = ""
        print(f"{name:{max_name}}  {body_str:>7}  {gain_str:>7}  {peak_str:>7}  {after_str:>7}{flag}")


def _dispatch(args):
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    if args.segments:
        if not args.files:
            print("Error: segment mode requires at least one file", file=sys.stderr)
            sys.exit(1)

        target_lufs = None
        if args.refs:
            ref_results = []
            for fp in args.refs:
                result = analyze_segments(fp, args.threshold, args.min_segment)
                if result:
                    ref_results.append(result)
                print()

            ref_body_avgs = [r["body_avg"] for r in ref_results if r.get("body_avg") is not None]
            if not ref_body_avgs:
                print("No reference body levels detected.", file=sys.stderr)
                sys.exit(1)
            target_lufs = statistics.mean(ref_body_avgs)
            count = len(ref_body_avgs)
            print(f"Reference target: {target_lufs:+.1f} LUFS (body avg of {count} reference{'s' if count != 1 else ''})")
            print("=" * 60)
            print()

        track_results = []
        for f in args.files:
            result = analyze_segments(f, args.threshold, args.min_segment, target_lufs=target_lufs)
            if result:
                track_results.append(result)
            if len(args.files) > 1:
                print()

        if len(track_results) > 1:
            _print_segment_summary(track_results)
    elif args.changes:
        if not args.files:
            print("Error: changes mode requires at least one file", file=sys.stderr)
            sys.exit(1)
        for f in args.files:
            stats = get_audio_stats(f)
            if not stats:
                continue
            changes = find_loudness_changes(stats["timestamps"], stats["momentary_values"], args.threshold)
            print(f"{stats['name']}")
            if changes:
                for c in changes:
                    direction = "+" if c["delta_db"] > 0 else "-"
                    print(f"  {format_timestamp(c['timestamp']):>7}  {direction}{abs(c['delta_db']):.1f} dB  ({c['from_lufs']:+.1f} → {c['to_lufs']:+.1f})")
            else:
                print(f"  No changes > {args.threshold} dB")
            if len(args.files) > 1:
                print()
    elif args.compare and args.refs and args.files:
        compare_to_refs(args.files, args.refs)
    elif args.refs and args.files:
        analyze_against_refs(args.files, args.refs, args.ceiling)
    elif args.refs:
        analyze_batch(args.refs, args.ceiling)
    elif len(args.files) == 1:
        analyze_single(args.files[0], args.threshold)
    elif args.files:
        analyze_batch(args.files, args.ceiling)
    else:
        print("No files specified. Use -h for help.", file=sys.stderr)
        sys.exit(1)


def register_subcommand(subparsers):
    parser = subparsers.add_parser("loudness", help="Loudness analysis (LUFS, peak, LRA)")
    _build_parser(parser)
    parser.set_defaults(func=_dispatch)


def main():
    parser = _build_parser()
    args = parser.parse_args()
    if not args.files and not args.refs:
        parser.print_help()
        return
    _dispatch(args)


if __name__ == "__main__":
    main()
