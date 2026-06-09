import numpy as np
import soundfile as sf

import detect_tempo as dt


# --- tempo resolution against a tap-tempo prior --------------------------

def test_resolve_no_hint_returns_raw():
    assert dt.resolve_to_hint(90.0, None) == 90.0


def test_resolve_doubles_half_time_estimate_toward_hint():
    # librosa often reports half-time; the tap hint pulls it back up.
    assert np.isclose(dt.resolve_to_hint(90.0, 180.0), 180.0)


def test_resolve_halves_double_time_estimate_toward_hint():
    assert np.isclose(dt.resolve_to_hint(360.0, 180.0), 180.0)


def test_resolve_handles_three_two_cross_rhythm():
    # The real failure mode: librosa locks onto 2/3 of the true tempo.
    # 120.19 x 3/2 = 180.3, which the 180 hint should recover (not 240).
    assert np.isclose(dt.resolve_to_hint(120.19, 180.0), 180.285, atol=0.5)


def test_resolve_keeps_near_hint_value_not_forced_to_hint():
    # 200 is closer (in ratio) to a 180 hint than any metric multiple -> keep.
    assert np.isclose(dt.resolve_to_hint(200.0, 180.0), 200.0)


# --- integer-bar snap (recover exact authored tempo) ---------------------

def test_snap_leaves_already_integer_bars_and_flags_snapped():
    # 96 s at 120 BPM, 4/4 = exactly 48 bars.
    bpm, bars, snapped = dt.snap_to_integer_bars(120.0, 96.0)
    assert snapped is True
    assert bars == 48
    assert np.isclose(bpm, 120.0)


def test_snap_pulls_near_integer_bars_to_clean_tempo():
    # 120.18 BPM over 96 s -> 48.07 bars -> snap to 48 -> 120.0 exactly.
    bpm, bars, snapped = dt.snap_to_integer_bars(120.18, 96.0)
    assert snapped is True
    assert bars == 48
    assert np.isclose(bpm, 120.0)


def test_snap_leaves_clearly_offgrid_tempo_alone():
    # 118 BPM over 96 s -> 47.2 bars: too far from integer, don't snap.
    bpm, bars, snapped = dt.snap_to_integer_bars(118.0, 96.0)
    assert snapped is False
    assert np.isclose(bpm, 118.0)
    assert np.isclose(bars, 47.2, atol=0.05)


# --- end-to-end detection on a synthetic 180 BPM click -------------------

def _click_track(path, bpm=180, n_beats=64, sr=22050):
    spb = round(sr * 60 / bpm)            # samples per beat (7350 at 180/22050)
    y = np.zeros(n_beats * spb, dtype=np.float32)
    click = np.hanning(64).astype(np.float32) * np.random.default_rng(0).normal(0, 1, 64).astype(np.float32)
    for k in range(n_beats):
        y[k * spb:k * spb + 64] += click
    sf.write(path, y, sr, subtype="FLOAT")
    return sr


def test_detect_tempo_recovers_180_from_drums(tmp_path):
    path = tmp_path / "drums.wav"
    _click_track(path, bpm=180)
    rep = dt.detect_tempo(path, bpm_hint=180.0)
    assert abs(rep["bpm"] - 180.0) < 180.0 * 0.02      # within 2%
    assert abs(rep["grid_offset_s"]) < 0.02            # grid phase ~0 (clicks on the one)
    assert 14 <= rep["n_bars"] <= 18                   # 64 beats = 16 bars
