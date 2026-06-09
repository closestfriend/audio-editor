"""Tempo / downbeat analysis companion for stem_arranger.

Deliberately separate from the deterministic engine: this module *estimates*
(BPM, grid offset) from audio, while stem_arranger stays exact-by-construction.
Run it, eyeball the result, then feed the confirmed --bpm into the renderer.

Best practice: point it at the isolated drums stem — beat tracking on clean
percussion is far more reliable than on a full mix. For fixed-grid electronic
music the integer-bar snap usually recovers the exact authored tempo.
"""

import argparse
import math

import numpy as np

DEFAULT_BEATS_PER_BAR = 4
# Common metric ambiguities in tempo estimation: factors of two (half/double
# time) AND three (triplet / 3:2 cross-rhythms), plus their combinations.
TEMPO_FACTORS = (1/4, 1/3, 1/2, 2/3, 3/4, 1.0, 4/3, 3/2, 2.0, 3.0, 4.0)


def resolve_to_hint(raw_bpm, hint_bpm):
    """Resolve metric ambiguity in a tempo estimate using a tap-tempo prior.

    A tapped BPM nails the metric level but not the precision; the raw estimate
    is precise but ambiguous by factors of 2 and 3 (half/double time, triplet
    and 3:2 cross-rhythms). Pick the metric multiple of the estimate closest
    (in log-ratio) to the hint. With no hint, return the estimate as-is.
    """
    if hint_bpm is None:
        return raw_bpm
    candidates = [raw_bpm * f for f in TEMPO_FACTORS]
    return min(candidates, key=lambda c: abs(math.log2(c / hint_bpm)))


def snap_to_integer_bars(bpm, duration_s, beats_per_bar=DEFAULT_BEATS_PER_BAR,
                         bar_tol=0.12):
    """Snap a near-integer-bar tempo to the exact tempo that yields whole bars.

    Fixed-grid tracks are a whole number of bars long, so if the measured bar
    count is within bar_tol of an integer we treat that as evidence of a grid
    and solve for the exact BPM. Returns (bpm, bars, snapped):
      - snapped True:  bpm is the clean tempo, bars is the integer bar count
      - snapped False: bpm unchanged, bars is the measured (float) bar count
    """
    bars = duration_s * bpm / 60.0 / beats_per_bar
    nearest = round(bars)
    if nearest > 0 and abs(bars - nearest) <= bar_tol:
        snapped_bpm = nearest * beats_per_bar * 60.0 / duration_s
        return snapped_bpm, nearest, True
    return bpm, bars, False


def estimate_tempo(drums_path):
    """Run librosa beat tracking. Returns (raw_bpm, beat_times_s, duration_s)."""
    import librosa  # lazy: keeps librosa optional and out of the engine

    y, sr = librosa.load(str(drums_path), sr=None, mono=True)
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    raw_bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr)
    duration_s = librosa.get_duration(y=y, sr=sr)
    return raw_bpm, beat_times, duration_s


def detect_tempo(drums_path, bpm_hint=None, beats_per_bar=DEFAULT_BEATS_PER_BAR):
    """Estimate tempo + grid offset and report a confirmed BPM to use.

    Pipeline: raw estimate -> octave-resolve against the hint -> integer-bar
    snap. Returns a dict with every intermediate value so the result is
    auditable before you trust it.
    """
    raw_bpm, beat_times, duration_s = estimate_tempo(drums_path)
    resolved = resolve_to_hint(raw_bpm, bpm_hint)
    bpm, bars, snapped = snap_to_integer_bars(resolved, duration_s, beats_per_bar)

    # librosa's beat list often starts a few beats in (DP warm-up), so the raw
    # first-beat timestamp isn't the grid origin. The stable quantity is the
    # phase of the grid relative to sample 0: first beat modulo one beat.
    first_beat = float(beat_times[0]) if len(beat_times) else 0.0
    beat_period = 60.0 / bpm
    grid_offset = first_beat % beat_period
    if grid_offset > beat_period / 2:        # report nearest beat (can be negative)
        grid_offset -= beat_period
    return {
        "raw_bpm": raw_bpm,
        "resolved_bpm": resolved,
        "bpm": bpm,
        "snapped": snapped,
        "n_bars": bars,
        "first_beat_s": first_beat,
        "grid_offset_s": grid_offset,
        "duration_s": duration_s,
    }


def _format_report(rep, bpm_hint):
    lines = [
        f"Duration:        {rep['duration_s']:.2f} s",
        f"Raw estimate:    {rep['raw_bpm']:.2f} BPM",
    ]
    if bpm_hint is not None:
        lines.append(f"Octave-resolved: {rep['resolved_bpm']:.2f} BPM "
                     f"(against {bpm_hint:g} hint)")
    if rep["snapped"]:
        lines.append(f"Integer-bar snap: {rep['bpm']:.4f} BPM "
                     f"({rep['n_bars']} bars) ✓ grid recovered")
        use = rep["bpm"]
    else:
        rounded = round(rep["bpm"])
        lines.append(f"No clean snap:   {rep['bpm']:.2f} BPM "
                     f"({rep['n_bars']:.2f} bars) — trailing silence or off-grid")
        lines.append(f"Nearest integer: {rounded} BPM "
                     f"(electronic tracks are almost always whole-number)")
        use = rounded
    lines.append(f"Grid phase:      {rep['grid_offset_s'] * 1000:+.1f} ms "
                 f"(beat offset from sample 0; ~0 means cuts land clean)")
    lines.append("")
    lines.append(f"==> Use:  --bpm {use:g}")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Estimate BPM + grid offset from a drums stem.")
    parser.add_argument("--drums", required=True,
                        help="Path to the (ideally isolated) drums stem")
    parser.add_argument("--bpm-hint", type=float, default=None,
                        help="Rough tapped tempo; resolves half/double errors")
    parser.add_argument("--beats-per-bar", type=int, default=DEFAULT_BEATS_PER_BAR)
    args = parser.parse_args(argv)

    rep = detect_tempo(args.drums, bpm_hint=args.bpm_hint,
                       beats_per_bar=args.beats_per_bar)
    print(_format_report(rep, args.bpm_hint))
    return rep


if __name__ == "__main__":
    main()
