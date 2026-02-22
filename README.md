# audio-tools

CLI tools for audio production and mastering.

## Mastering

Unified `mastering` CLI with subcommands for analysis. All subcommands accept one or more audio files and optional `-r` reference tracks.

| Command | What it does | Key flags |
|---------|-------------|-----------|
| `mastering check` | Run all analysis passes at once | `-r` refs (required) |
| `mastering compare` | A/B previous vs current masters | `--prev`, `--cur`, `-r` refs (all required) |
| `mastering loudness` | LUFS, peak, and loudness range | `-s` segments, `-ch` changes, `-t` threshold (3 dB), `-c` ceiling (-1 dBFS) |
| `mastering spectrum` | LTAS frequency balance across bands | `--plot` |
| `mastering dynamics` | Dynamic range analysis | `--plot` |
| `mastering crest` | Peak-to-RMS crest factor | |
| `mastering stereo` | Stereo correlation, width, balance | |

```bash
# Full analysis against a reference
mastering check master.wav -r reference.wav

# A/B compare old vs new master
mastering compare --prev old.wav --cur new.wav -r reference.wav

# Loudness with segment breakdown
mastering loudness -s master.wav

# Spectrum plot against reference
mastering spectrum --plot master.wav -r reference.wav
```

`loudness` and `mastering-compare` are also available as standalone commands.

## General Audio

| Command | What it does | Key flags |
|---------|-------------|-----------|
| `stems` | Stem separation (6-stem RoFormer) | `--vocals`, `--drums`, `--bass`, `--guitar`, `--piano`, `--other`, `--all`; `--flac` (default), `--wav`, `--mp3` |
| `normalize` | Peak or LUFS normalization | `-p` peak (-0.1 dB default), `-l` LUFS target, `-i` individual mode, `-o` output dir, `--dry-run` |
| `declick` | Remove clicks and zero-sample dropouts | `-c` clicks, `-d` dropouts (at least one required), `-r` ratio (10.0), `-a` analyze only, `-b` backup |

```bash
# Extract drums and bass stems
stems --drums --bass track.wav

# All stems as MP3
stems --all --mp3 track.wav

# Batch stem separation
stems --drums *.flac

# Peak normalize (default -0.1 dB)
normalize track.wav

# LUFS normalize to -14
normalize -l -14 track.wav

# Group normalize (same gain for all files)
normalize *.wav

# Analyze clicks without repairing
declick -dc -a input.wav

# Fix dropouts in-place with backup
declick -d -b input.wav
```

## Install

Requires Python 3.10+ and ffmpeg (`brew install ffmpeg`).

```bash
cd /path/to/audio-tools
venv
./setup.sh
```

## Development

```bash
venv
pip install -e .
pip install pytest
pytest tests/
```
