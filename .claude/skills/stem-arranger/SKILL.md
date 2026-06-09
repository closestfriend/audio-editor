---
name: stem-arranger
description: Use when generating arrangement variations (extended mix, radio edit, ambient, dub, etc.) from a folder of time-aligned audio stems in this repo, or editing/rendering its arrangements.py variation table with stem_arranger.py — including choosing which stems play in which sections, reordering song sections, applying per-section gain/fades, or rendering WAV and MP3 previews.
---

# stem-arranger

## Overview

`stem_arranger.py` assembles arrangement variations from a folder of **time-aligned stems** (full-length WAVs that all start at bar 1 and share length + sample rate, e.g. an Ableton "one file per track" export). Cuts are **sample-accurate** (no millisecond drift at any BPM) and the mix bus is **clip-safe** (sections sum at unity; the final mix is scaled once to a peak ceiling).

You almost never edit `stem_arranger.py`. You edit **`arrangements.py`** — a plain-data table — and run the CLI.

## Core mental model

A variation is an ordered list of **sections**. Each section pulls a **bar range from the original song** and lists **which stems play**. Sections are concatenated in order. Because sections reference the original by bar, you can reorder/reuse real musical content (repeat the chorus, lift the breakdown from where it lives, top-and-tail with a DJ intro/outro).

```python
EXTENDED = [
    {"source_bars": (1, 16),  "stems": ["drums", "bass"]},                  # intro
    {"source_bars": (25, 40), "stems": ["drums", "bass", "synth", "vocals"]},  # full
    {"source_bars": (17, 24), "stems": ["synth", "vocals"], "fade_in_ms": 200},  # breakdown
]
```

### Section keys

| Key | Required | Meaning |
|-----|----------|---------|
| `source_bars` | yes | `(start, end)` — 1-indexed, **inclusive** bar range from the original |
| `stems` | yes | list of stem keys that play; omitted stems are silent here |
| `gain_db` | no | section level offset in dB (default 0) |
| `fade_in_ms` | no | linear fade-in at the section start (default 0) |
| `fade_out_ms` | no | linear fade-out at the section end (default 0) |

**Stem keys are a normalized slug of the filename:** lowercased, every run of non-alphanumeric characters becomes one underscore, edges trimmed. `Drums.wav` → `drums`, `Lead Synth.wav` → `lead_synth`, `Vocals (Wet).wav` → `vocals_wet`, `FX-01.wav` → `fx_01`. Two files that normalize to the same key are a hard error (rename one). Never guess keys — discover them with `--list`.

`gain_db` is a *per-section* offset. Independently, the finished mix is scaled once so its peak hits the ceiling, and stdout reports that global cut (e.g. `−4.16 dB applied for headroom`) — so the printed number is the bus-level adjustment, not your section gain.

## Workflow

1. **Discover the real stem keys** (do this first, always):
   ```bash
   .venv/bin/python stem_arranger.py --folder stems/ --list
   ```
2. **Edit `arrangements.py`**: make `stems` names match the keys, and set each `source_bars` to where that part actually sits in the song. The shipped numbers are placeholders for a ~48-bar track.
3. **Render**:
   ```bash
   .venv/bin/python stem_arranger.py --folder stems/ --bpm 124            # all variations
   .venv/bin/python stem_arranger.py --folder stems/ --bpm 124 \
       --variation radio --out renders/ --crossfade-ms 0 --preview        # one + MP3 preview
   ```
   Output keys become filenames: `out/extended.wav`, etc. `--out` may be any path (relative or absolute) and missing directories are created. With `--preview` the tool prints both the WAV and `.mp3` paths.

## Good to know

- **Predict output length** to sanity-check the BPM: 4/4 is assumed, so a bar = `240/BPM` seconds. A variation's duration = `(total bars across its sections) × 240/BPM`. Wrong BPM → wrong length, which this lets you catch.
- **Channel count mirrors the input** — stereo stems give a stereo render, mono gives mono. All stems must match (all-mono or all-stereo).

## CLI flags

`--folder` (req) · `--bpm` (req unless `--list`) · `--out` (default `out`) · `--variation` (repeatable; default all) · `--crossfade-ms` (default 5; equal-power; 0 = hard cut) · `--ceiling-db` (default −1) · `--subtype` (default `PCM_24`) · `--preview` (also write MP3) · `--list`. Run with `--help` for the authoritative list.

## Finding the BPM (`detect_tempo.py`)

The renderer needs `--bpm`. If unknown, estimate it from the **drums stem** with the optional companion (requires `pip install -r requirements-analysis.txt`; librosa is heavy and NOT needed by the renderer):

```bash
.venv/bin/python detect_tempo.py --drums stems/drums.wav --bpm-hint 180
```

Pass a rough tapped tempo as `--bpm-hint` — it resolves librosa's metric ambiguity (half-time, or 3:2/triplet cross-rhythms). The tool snaps near-integer-bar results to the exact authored tempo and prints the **grid phase** (how far the first downbeat is from sample 0; large values mean the track has a lead-in and whole-bar cuts won't land exactly on musical downbeats). It only *reports* — you confirm the number and feed it to `--bpm`.

## Common mistakes

- **Guessing stem names** instead of running `--list` → `KeyError`. The error names the missing stem and the available keys.
- **Forgetting `--bpm`** → cut timing is wrong; it's required (the tool errors if missing).
- **Leaving placeholder `source_bars`** → variation pulls the wrong parts. Always set them to the actual song structure; bars past the end clamp with a warning.
- **Mismatched stems** → sample-rate, channel-count, and name-collision mismatches all raise clear errors naming the offenders; a length mismatch pads to the longest with a warning. All stems must be all-mono or all-stereo.
- **Editing `stem_arranger.py` for an arrangement change** — arrangements live in `arrangements.py`.

## Verifying changes

`.venv/bin/python -m pytest -q` (32 tests cover bar math, clip protection, assembly, fades, loading, CLI). See `README.md` for full docs.
