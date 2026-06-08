# stem-arranger

Assemble arrangement variations (Extended Mix, Radio Edit, Subtractive Ambient, …)
from a folder of time-aligned stems — **sample-accurate** and **clip-safe**.

Built on `soundfile` + `numpy` (not pydub): cuts land on exact sample boundaries
at any BPM, and the mix bus is protected against the clipping you get from
summing stems at unity.

## Install

```bash
python3.12 -m venv .venv
.venv/bin/pip install soundfile numpy pytest
```

## Use

1. Export your stems from the DAW as **full-length, bar-1-aligned** WAVs
   (Ableton: Export Audio → *Render as one file per track*). Every stem must be
   the same length and start at the song's bar 1.

2. See what stem keys your folder exposes (keys are the lowercased filenames):

   ```bash
   .venv/bin/python stem_arranger.py --folder stems/ --list
   ```

3. Edit `arrangements.py` so the stem names and `source_bars` match your track,
   then render:

   ```bash
   .venv/bin/python stem_arranger.py --folder stems/ --bpm 124
   # -> out/extended.wav, out/radio.wav, out/ambient.wav
   ```

   Render one variation, into a chosen folder, with hard cuts:

   ```bash
   .venv/bin/python stem_arranger.py --folder stems/ --bpm 124 \
       --variation radio --out renders/ --crossfade-ms 0
   ```

## How variations are defined

`arrangements.py` is plain data. A variation is an ordered list of sections;
each section pulls a **bar range from the original song** and lists **which
stems play**:

```python
EXTENDED_MIX = [
    {"source_bars": (1, 16),  "stems": ["drums", "bass"]},               # DJ intro
    {"source_bars": (25, 40), "stems": ["drums", "bass", "synth", "vocals"]},  # full
    {"source_bars": (17, 24), "stems": ["synth", "vocals"]},             # breakdown
    ...
]
```

Because sections reference the original by bar, you can **reorder and reuse**
real musical content — lift the breakdown from where it actually lives, repeat
the chorus, top-and-tail with a DJ intro/outro.

## What it guarantees

- **Sample-accurate cuts** — `samples_per_bar = round(sr · 60/bpm · 4)`; no
  millisecond rounding drift (the bug that clicks at 124 BPM).
- **Clip protection** — sections sum at unity; the finished arrangement is
  scaled once so its peak sits at the ceiling (default −1 dBFS), and the applied
  gain is printed. Relative section levels are preserved.
- **Equal-power crossfades** at section joins (default 5 ms) to kill clicks.
- **Sanity checks on load** — errors on sample-rate mismatch; pads length
  mismatches to the longest stem with a warning.

## Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--folder` | (required) | Folder of `.wav` stems |
| `--bpm` | (required) | Tempo of the stems |
| `--out` | `out` | Output directory |
| `--variation` | all | Variation name (repeatable) |
| `--crossfade-ms` | `5.0` | Section-join crossfade; `0` for hard cuts |
| `--ceiling-db` | `-1.0` | Peak ceiling for clip protection |
| `--subtype` | `PCM_24` | Output WAV bit depth (soundfile subtype) |
| `--list` | — | Print detected stem keys and exit |

## Known limits

- Assumes all stems share channel count (all mono or all stereo) and start at
  bar 1. Trimmed/offset clips aren't auto-aligned.

## Tests

```bash
.venv/bin/python -m pytest -q
```
