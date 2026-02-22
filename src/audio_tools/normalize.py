#!/usr/bin/env python3
"""
Audio normalization by peak or LUFS.
Supports individual and group modes for batch processing.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LOSSY_FORMATS = {".mp3", ".aac", ".ogg", ".opus", ".m4a", ".wma"}


def run_ffmpeg_analysis(filepath: str) -> tuple[str, str | None]:
    """Run ffmpeg with ebur128 filter and return (stderr output, error message)."""
    cmd = [
        "ffmpeg", "-i", filepath,
        "-filter_complex", "ebur128=peak=true",
        "-f", "null", "-"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if "Invalid data found when processing input" in result.stderr:
        return "", "file may be online-only (Dropbox/iCloud) - sync it first"
    if "No such file or directory" in result.stderr:
        return "", "file not found"
    if result.returncode != 0 and not result.stderr:
        return "", "ffmpeg failed to process file"

    return result.stderr, None


def parse_ebur128_output(output: str) -> dict:
    """Parse ffmpeg ebur128 output for true peak and integrated LUFS."""
    data = {"true_peak": None, "integrated_lufs": None}

    # Parse frame-by-frame for true peak (TPK field)
    for line in output.split("\n"):
        if "Parsed_ebur128" in line and "TPK:" in line:
            tpk_match = re.search(r"TPK:\s*([-\d.]+)\s+([-\d.]+)\s*dBFS", line)
            if tpk_match:
                peak_l = float(tpk_match.group(1))
                peak_r = float(tpk_match.group(2))
                frame_peak = max(peak_l, peak_r)
                if data["true_peak"] is None or frame_peak > data["true_peak"]:
                    data["true_peak"] = frame_peak

    # Parse summary section
    in_summary = False
    for line in output.split("\n"):
        if "Summary:" in line:
            in_summary = True
        if in_summary:
            if "I:" in line and "LUFS" in line:
                match = re.search(r"I:\s*([-\d.]+)\s*LUFS", line)
                if match:
                    data["integrated_lufs"] = float(match.group(1))
            if line.strip().startswith("Peak:"):
                match = re.search(r"Peak:\s*([-\d.]+)\s*dBFS", line)
                if match:
                    summary_peak = float(match.group(1))
                    if data["true_peak"] is None or summary_peak > data["true_peak"]:
                        data["true_peak"] = summary_peak

    return data


def get_audio_info(filepath: str) -> dict | None:
    """Get audio stats and format info for a file."""
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        return None

    # Run ebur128 analysis
    output, error = run_ffmpeg_analysis(filepath)
    if error:
        print(f"Error: {path.name} - {error}", file=sys.stderr)
        return None

    data = parse_ebur128_output(output)

    if data["true_peak"] is None:
        print(f"Error: Could not parse peak data for {path.name}", file=sys.stderr)
        return None

    # Get format info via ffprobe
    probe_cmd = [
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", filepath
    ]
    probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)

    sample_rate = 44100
    bitrate = None
    if probe_result.returncode == 0:
        try:
            probe_data = json.loads(probe_result.stdout)
            for stream in probe_data.get("streams", []):
                if stream.get("codec_type") == "audio":
                    sample_rate = int(stream.get("sample_rate", 44100))
                    if "bit_rate" in stream:
                        bitrate = int(stream["bit_rate"]) // 1000
                    break
            if bitrate is None and "format" in probe_data:
                fmt_bitrate = probe_data["format"].get("bit_rate")
                if fmt_bitrate:
                    bitrate = int(fmt_bitrate) // 1000
        except (json.JSONDecodeError, KeyError, ValueError):
            pass

    return {
        "path": filepath,
        "name": path.name,
        "suffix": path.suffix.lower(),
        "true_peak": data["true_peak"],
        "integrated_lufs": data["integrated_lufs"],
        "sample_rate": sample_rate,
        "bitrate": bitrate,
    }


def calculate_gains(
    files: list[dict],
    mode: str,
    target: float,
    group: bool,
    peak_ceiling: float = -0.1
) -> list[dict]:
    """
    Calculate gain adjustments for each file.

    mode: 'peak' or 'lufs'
    target: target peak dB or target LUFS
    group: if True, apply same gain to all files
    peak_ceiling: max allowed peak after normalization
    """
    results = []

    if mode == "peak":
        if group:
            max_peak = max(f["true_peak"] for f in files)
            base_gain = target - max_peak
            for f in files:
                results.append({**f, "gain": base_gain})
        else:
            for f in files:
                gain = target - f["true_peak"]
                results.append({**f, "gain": gain})
    else:  # lufs
        if group:
            max_lufs = max(f["integrated_lufs"] for f in files if f["integrated_lufs"])
            base_gain = target - max_lufs
            # Clamp so no file exceeds peak ceiling
            for f in files:
                clamped_gain = base_gain
                peak_after = f["true_peak"] + base_gain
                if peak_after > peak_ceiling:
                    clamped_gain = peak_ceiling - f["true_peak"]
                results.append({**f, "gain": clamped_gain})
        else:
            for f in files:
                if f["integrated_lufs"] is None:
                    print(f"Warning: {f['name']} has no LUFS data, skipping", file=sys.stderr)
                    continue
                gain = target - f["integrated_lufs"]
                # Clamp so peak doesn't exceed ceiling
                peak_after = f["true_peak"] + gain
                if peak_after > peak_ceiling:
                    gain = peak_ceiling - f["true_peak"]
                results.append({**f, "gain": gain})

    return results


def get_output_path(input_path: str, output_dir: str | None) -> Path:
    """Determine output path for a file."""
    inp = Path(input_path)
    if output_dir:
        return Path(output_dir) / inp.name
    else:
        return inp  # Replace in-place


def process_file(info: dict, output_path: Path, dry_run: bool) -> bool:
    """Apply gain to a file using ffmpeg."""
    gain = info["gain"]
    is_lossy = info["suffix"] in LOSSY_FORMATS
    in_place = Path(info["path"]).resolve() == output_path.resolve()

    if is_lossy:
        print(f"  ⚠ {info['name']} is lossy - will be re-encoded (quality loss)")

    if dry_run:
        return True

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Use temp file for in-place replacement
    if in_place:
        temp_fd, temp_path = tempfile.mkstemp(suffix=info["suffix"])
        import os
        os.close(temp_fd)
        actual_output = temp_path
    else:
        actual_output = str(output_path)

    # Build ffmpeg command
    cmd = ["ffmpeg", "-y", "-i", info["path"]]

    # Audio filter for gain
    cmd.extend(["-af", f"volume={gain}dB"])

    # Preserve sample rate
    cmd.extend(["-ar", str(info["sample_rate"])])

    # Handle format-specific encoding
    if is_lossy and info["bitrate"]:
        suffix = info["suffix"]
        if suffix == ".mp3":
            cmd.extend(["-b:a", f"{info['bitrate']}k"])
        elif suffix in {".aac", ".m4a"}:
            cmd.extend(["-b:a", f"{info['bitrate']}k"])
        elif suffix == ".ogg":
            cmd.extend(["-b:a", f"{info['bitrate']}k"])
        elif suffix == ".opus":
            cmd.extend(["-b:a", f"{info['bitrate']}k"])

    cmd.append(actual_output)

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error processing {info['name']}: {result.stderr[:200]}", file=sys.stderr)
        if in_place:
            Path(temp_path).unlink(missing_ok=True)
        return False

    # Move temp file over original for in-place
    if in_place:
        shutil.move(temp_path, str(output_path))

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Normalize audio files by peak or LUFS"
    )
    parser.add_argument("files", nargs="+", help="Audio files to normalize")
    parser.add_argument(
        "-p", "--peak", nargs="?", type=float, const=-0.1, default=None,
        metavar="dB", help="Peak normalization (default: -0.1 dB)"
    )
    parser.add_argument(
        "-l", "--lufs", type=float, default=None,
        metavar="dB", help="LUFS normalization (value required)"
    )
    parser.add_argument(
        "-i", "--individual", action="store_true",
        help="Individual mode: normalize each file independently (default for single file)"
    )
    parser.add_argument(
        "-o", "--output", metavar="DIR",
        help="Output directory (default: replace in-place)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would happen without processing"
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install with: brew install ffmpeg", file=sys.stderr)
        sys.exit(1)

    # Determine normalization mode
    if args.lufs is not None:
        mode = "lufs"
        target = args.lufs
    elif args.peak is not None:
        mode = "peak"
        target = args.peak
    else:
        # Default: peak normalize to -0.1 dB
        mode = "peak"
        target = -0.1

    # Analyze all files
    print(f"Analyzing {len(args.files)} file(s)...")
    file_infos = []
    for f in args.files:
        info = get_audio_info(f)
        if info:
            file_infos.append(info)

    if not file_infos:
        print("No files successfully analyzed.", file=sys.stderr)
        sys.exit(1)

    # Group mode is default for multiple files
    group = len(file_infos) > 1 and not args.individual

    # Calculate gains
    results = calculate_gains(file_infos, mode, target, group)

    if not results:
        print("No files to process.", file=sys.stderr)
        sys.exit(1)

    # Create output directory if specified
    if args.output:
        Path(args.output).mkdir(parents=True, exist_ok=True)

    # Display plan and process
    mode_str = f"LUFS → {target}" if mode == "lufs" else f"Peak → {target} dB"
    group_str = " (group)" if group else ""
    print(f"\nNormalization: {mode_str}{group_str}")
    if args.dry_run:
        print("DRY RUN - no files will be modified\n")
    print()

    max_name = min(50, max(len(r["name"]) for r in results))
    print(f"{'FILE':{max_name}}  {'PEAK':>8}  {'LUFS':>8}  {'GAIN':>10}")
    print("-" * (max_name + 32))

    success_count = 0
    for r in results:
        name = r["name"][:max_name]
        peak_str = f"{r['true_peak']:+.1f}"
        lufs_str = f"{r['integrated_lufs']:+.1f}" if r["integrated_lufs"] else "N/A"
        gain_str = f"{r['gain']:+.2f} dB"

        print(f"{name:{max_name}}  {peak_str:>8}  {lufs_str:>8}  {gain_str:>10}")

        output_path = get_output_path(r["path"], args.output)

        if not args.dry_run:
            if process_file(r, output_path, args.dry_run):
                success_count += 1

    print()
    if args.dry_run:
        print(f"Would process {len(results)} file(s)")
    else:
        print(f"Processed {success_count}/{len(results)} file(s)")
        if args.output:
            print(f"Output: {args.output}/")


if __name__ == "__main__":
    main()
