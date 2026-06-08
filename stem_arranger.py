"""Sample-accurate stem arrangement engine.

Reads time-aligned stems (full-length, bar-1 aligned) and assembles
arrangement variations by pulling bar ranges from the original and choosing
which stems play in each section. All cutting is done in samples (never
milliseconds) so boundaries land exactly on the grid at any BPM.
"""

import argparse
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

DEFAULT_BEATS_PER_BAR = 4
DEFAULT_CEILING_DB = -1.0


def load_stems(folder):
    """Load every .wav in a folder as a float64 stem.

    Returns (stems, sr) where stems maps a lowercased filename (no extension)
    to a 2D array of shape (frames, channels). Stems must share a sample rate
    (mismatch is an error); differing lengths are padded up to the longest
    with silence so bar slicing always stays in range.
    """
    folder = Path(folder)
    wavs = sorted(p for p in folder.iterdir() if p.suffix.lower() == ".wav")
    if not wavs:
        raise FileNotFoundError(f"No .wav stems found in {folder}")

    stems, rates = {}, {}
    for path in wavs:
        data, sr = sf.read(path, dtype="float64", always_2d=True)
        stems[path.stem.lower()] = data
        rates[path.stem.lower()] = sr

    distinct = set(rates.values())
    if len(distinct) > 1:
        detail = ", ".join(f"{k}={v}Hz" for k, v in rates.items())
        raise ValueError(f"Stems disagree on sample rate: {detail}")
    sr = distinct.pop()

    lengths = {k: v.shape[0] for k, v in stems.items()}
    longest = max(lengths.values())
    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in lengths.items())
        warnings.warn(
            f"Stems differ in length ({detail}); padding to {longest} samples.",
            UserWarning,
        )
        for k, v in stems.items():
            if v.shape[0] < longest:
                pad = np.zeros((longest - v.shape[0], v.shape[1]))
                stems[k] = np.concatenate([v, pad], axis=0)

    return stems, sr


def samples_per_bar(sr, bpm, beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Length of one bar in whole samples, rounded once.

    seconds_per_bar = (60 / bpm) * beats_per_bar
    """
    seconds_per_bar = (60.0 / bpm) * beats_per_bar
    return round(seconds_per_bar * sr)


def bar_range_to_samples(start_bar, end_bar, sr, bpm,
                         beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Convert a 1-indexed, inclusive bar range to a [start, end) sample slice.

    Bars (1, 16) -> the first 16 bars, i.e. samples [0, 16 * spb).
    """
    spb = samples_per_bar(sr, bpm, beats_per_bar)
    start_sample = (start_bar - 1) * spb
    end_sample = end_bar * spb
    return start_sample, end_sample


def mix_down(arrays, ceiling_db=DEFAULT_CEILING_DB):
    """Sum arrays in float64 and protect against clipping.

    Returns (mixed, applied_gain_db). If the summed peak exceeds the ceiling,
    the whole mix is scaled so its peak sits exactly at the ceiling and the
    applied gain (a negative dB value) is reported. If it's already under the
    ceiling, the mix is returned untouched with 0.0 dB applied.
    """
    mixed = np.sum([a.astype(np.float64) for a in arrays], axis=0)
    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    ceiling_lin = 10 ** (ceiling_db / 20.0)
    if peak > ceiling_lin and peak > 0:
        gain = ceiling_lin / peak
        return mixed * gain, 20 * np.log10(gain)
    return mixed, 0.0


def render_section(stems, section, sr, bpm,
                   beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Render one section: sum the listed stems over the section's bar range.

    No peak protection here — sections are summed at unity and the finished
    arrangement is protected once at the end so relative section levels are
    preserved.
    """
    start_bar, end_bar = section["source_bars"]
    s, e = bar_range_to_samples(start_bar, end_bar, sr, bpm, beats_per_bar)
    available = min(stems[name].shape[0] for name in section["stems"])
    if e > available:
        warnings.warn(
            f"Section bars {start_bar}..{end_bar} reach sample {e} but the "
            f"stems are only {available} samples long; clamping to the end.",
            UserWarning,
        )
        e = available
    slices = [stems[name][s:e].astype(np.float64) for name in section["stems"]]
    return np.sum(slices, axis=0)


def equal_power_fades(n):
    """Equal-power fade curves of length n: (fade_out, fade_in).

    Constant power across the overlap (fade_out**2 + fade_in**2 == 1), so
    section joins don't dip or click.
    """
    t = np.linspace(0.0, 1.0, n)
    return np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)


def assemble(variation, stems, sr, bpm, crossfade_ms=5.0,
             ceiling_db=DEFAULT_CEILING_DB,
             beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Render every section and join them, then protect the final peak.

    Returns (audio, applied_gain_db).
    """
    sections = [render_section(stems, sec, sr, bpm, beats_per_bar)
                for sec in variation]
    x = round(crossfade_ms / 1000.0 * sr)

    result = sections[0]
    for nxt in sections[1:]:
        if x > 0:
            fo, fi = equal_power_fades(x)
            fo = fo[:, None]
            fi = fi[:, None]
            blended = result[-x:] * fo + nxt[:x] * fi
            result = np.concatenate([result[:-x], blended, nxt[x:]], axis=0)
        else:
            result = np.concatenate([result, nxt], axis=0)

    return mix_down([result], ceiling_db=ceiling_db)


def write_wav(path, audio, sr, subtype="PCM_24"):
    """Write audio to a WAV, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr, subtype=subtype)


def render_variation(folder, bpm, variation, out_path, crossfade_ms=5.0,
                     ceiling_db=DEFAULT_CEILING_DB, subtype="PCM_24",
                     beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Load stems, assemble one variation, and write it to out_path.

    Returns the applied peak-protection gain in dB (0.0 if none needed).
    """
    stems, sr = load_stems(folder)
    audio, applied_db = assemble(variation, stems, sr, bpm,
                                 crossfade_ms=crossfade_ms,
                                 ceiling_db=ceiling_db,
                                 beats_per_bar=beats_per_bar)
    write_wav(out_path, audio, sr, subtype=subtype)
    return applied_db


def main(argv=None, arrangements=None):
    """CLI entry point. Returns the list of written output paths.

    `arrangements` lets callers (and tests) inject a variation table; when
    omitted it is imported from the sibling ``arrangements`` module.
    """
    parser = argparse.ArgumentParser(
        description="Assemble arrangement variations from time-aligned stems.")
    parser.add_argument("--folder", required=True,
                        help="Folder of .wav stems (full-length, bar-1 aligned)")
    parser.add_argument("--bpm", type=float,
                        help="Tempo of the stems (required unless --list)")
    parser.add_argument("--out", default="out",
                        help="Output directory (default: ./out)")
    parser.add_argument("--variation", action="append", dest="variations",
                        help="Variation name to render (repeatable; "
                             "default: all defined)")
    parser.add_argument("--crossfade-ms", type=float, default=5.0,
                        help="Equal-power crossfade at section joins "
                             "(default: 5 ms; 0 for hard cuts)")
    parser.add_argument("--ceiling-db", type=float, default=DEFAULT_CEILING_DB,
                        help="Peak ceiling for clip protection "
                             "(default: -1 dBFS)")
    parser.add_argument("--subtype", default="PCM_24",
                        help="Output WAV subtype (default: PCM_24)")
    parser.add_argument("--list", action="store_true",
                        help="List the stem keys detected in --folder and exit")
    args = parser.parse_args(argv)

    if args.list:
        stems, sr = load_stems(args.folder)
        print(f"{len(stems)} stems detected at {sr} Hz:")
        for key in sorted(stems):
            print(f"  {key}")
        return []

    if args.bpm is None:
        parser.error("--bpm is required unless --list is given")

    if arrangements is None:
        import arrangements as arrangements_mod
        arrangements = arrangements_mod.ARRANGEMENTS

    names = args.variations or list(arrangements)
    out_dir = Path(args.out)
    written = []
    for name in names:
        if name not in arrangements:
            parser.error(f"Unknown variation '{name}'. "
                         f"Available: {', '.join(arrangements)}")
        out_path = out_dir / f"{name}.wav"
        applied_db = render_variation(
            args.folder, args.bpm, arrangements[name], out_path,
            crossfade_ms=args.crossfade_ms, ceiling_db=args.ceiling_db,
            subtype=args.subtype)
        note = (f" (−{abs(applied_db):.2f} dB applied for headroom)"
                if applied_db < 0 else "")
        print(f"Wrote {out_path}{note}")
        written.append(out_path)
    return written


if __name__ == "__main__":
    main()
