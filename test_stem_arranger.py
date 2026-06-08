import warnings

import numpy as np
import pytest
import soundfile as sf

import stem_arranger as sa


# --- bar / sample math ---------------------------------------------------

def test_samples_per_bar_is_sample_accurate_at_awkward_bpm():
    # 124 BPM, 44100 Hz, 4 beats/bar:
    # seconds_per_bar = 60/124 * 4 = 1.935483...s ; * 44100 = 85354.83 samples
    # The old ms-based approach rounded a bar to 1935 ms and drifted. We round
    # once, in samples, to the nearest whole sample.
    assert sa.samples_per_bar(sr=44100, bpm=124) == 85355


def test_samples_per_bar_exact_at_round_bpm():
    # 120 BPM @ 48k: 60/120*4 = 2.0s ; * 48000 = 96000 samples exactly
    assert sa.samples_per_bar(sr=48000, bpm=120) == 96000


def test_bar_range_is_one_indexed_inclusive():
    # Bars 1..16 (1-indexed, inclusive) == 16 bars starting at sample 0.
    spb = sa.samples_per_bar(sr=48000, bpm=120)
    start, end = sa.bar_range_to_samples(1, 16, sr=48000, bpm=120)
    assert start == 0
    assert end == 16 * spb


def test_bar_range_later_section_offsets_correctly():
    spb = sa.samples_per_bar(sr=48000, bpm=120)
    start, end = sa.bar_range_to_samples(17, 32, sr=48000, bpm=120)
    assert start == 16 * spb
    assert end == 32 * spb


# --- clip-safe mixdown ---------------------------------------------------

def test_mixdown_below_ceiling_is_untouched():
    a = np.full((100, 2), 0.25)
    b = np.full((100, 2), 0.25)  # sums to 0.5, well under 0 dBFS
    mixed, applied_db = sa.mix_down([a, b], ceiling_db=0.0)
    assert applied_db == 0.0
    assert np.isclose(np.max(np.abs(mixed)), 0.5)


def test_mixdown_attenuates_when_sum_would_clip():
    a = np.full((100, 2), 0.9)
    b = np.full((100, 2), 0.6)  # sums to 1.5 -> over 0 dBFS, would clip
    mixed, applied_db = sa.mix_down([a, b], ceiling_db=0.0)
    # Peak pulled down to exactly the ceiling (1.0 linear at 0 dBFS).
    assert np.isclose(np.max(np.abs(mixed)), 1.0)
    # Gain reported, and it's a cut (negative dB).
    assert applied_db < 0
    assert np.isclose(applied_db, 20 * np.log10(1.0 / 1.5))


def test_mixdown_respects_negative_ceiling_headroom():
    a = np.full((50, 2), 1.0)  # already at full scale
    mixed, applied_db = sa.mix_down([a], ceiling_db=-1.0)
    ceiling_lin = 10 ** (-1.0 / 20.0)  # ~0.8913
    assert np.isclose(np.max(np.abs(mixed)), ceiling_lin)
    assert np.isclose(applied_db, -1.0)


# --- section rendering ---------------------------------------------------

def _stem(value, n_frames, channels=2):
    return np.full((n_frames, channels), value, dtype=np.float64)


def test_render_section_sums_only_listed_stems_over_bar_range():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {
        "drums": _stem(0.1, spb * 4),
        "bass": _stem(0.2, spb * 4),
        "synth": _stem(0.4, spb * 4),  # present but NOT listed -> excluded
    }
    section = {"source_bars": (1, 2), "stems": ["drums", "bass"]}
    out = sa.render_section(stems, section, sr, bpm)
    assert out.shape == (2 * spb, 2)          # bars 1..2 == 2 bars
    assert np.allclose(out, 0.3)              # 0.1 + 0.2, synth excluded


def test_render_section_pulls_correct_source_offset():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    # A stem that is 0.0 in bars 1..2 and 0.5 from bar 3 on.
    arr = np.zeros((spb * 4, 1))
    arr[2 * spb:] = 0.5
    stems = {"synth": arr}
    section = {"source_bars": (3, 4), "stems": ["synth"]}
    out = sa.render_section(stems, section, sr, bpm)
    assert np.allclose(out, 0.5)


def test_render_section_unknown_stem_names_the_culprit():
    stems = {"drums": _stem(0.1, 100)}
    section = {"source_bars": (1, 1), "stems": ["vocal"]}
    with pytest.raises(KeyError, match="vocal"):
        sa.render_section(stems, section, 48000, 120)


# --- per-section gain & fades -------------------------------------------

def test_render_section_applies_gain_db():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.4, spb)}
    section = {"source_bars": (1, 1), "stems": ["a"], "gain_db": -6.0}
    out = sa.render_section(stems, section, sr, bpm)
    assert np.allclose(out, 0.4 * 10 ** (-6.0 / 20.0))


def test_render_section_fade_in_ramps_from_silence():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.5, spb * 2)}
    n = round(10 / 1000 * sr)  # 480-sample fade
    section = {"source_bars": (1, 2), "stems": ["a"], "fade_in_ms": 10.0}
    out = sa.render_section(stems, section, sr, bpm)
    assert np.isclose(out[0, 0], 0.0)            # starts silent
    assert np.isclose(out[n - 1, 0], 0.5)        # full by end of the fade
    assert np.isclose(out[n // 2, 0], 0.25, atol=1e-3)  # ~halfway up midway
    assert np.allclose(out[n:], 0.5)             # unaffected after the fade


def test_render_section_fade_out_ramps_to_silence():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.5, spb * 2)}
    section = {"source_bars": (1, 2), "stems": ["a"], "fade_out_ms": 10.0}
    out = sa.render_section(stems, section, sr, bpm)
    assert np.isclose(out[-1, 0], 0.0)           # ends silent
    assert np.allclose(out[0], 0.5)              # untouched at the start


# --- equal-power crossfade ----------------------------------------------

def test_equal_power_fades_preserve_power():
    fo, fi = sa.equal_power_fades(64)
    assert np.isclose(fo[0], 1.0) and np.isclose(fi[0], 0.0)
    assert np.isclose(fo[-1], 0.0, atol=1e-6) and np.isclose(fi[-1], 1.0)
    assert np.allclose(fo ** 2 + fi ** 2, 1.0)  # constant power across the fade


# --- assembly ------------------------------------------------------------

def test_assemble_no_crossfade_is_plain_concatenation():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.1, spb * 4), "b": _stem(0.2, spb * 4)}
    variation = [
        {"source_bars": (1, 1), "stems": ["a"]},
        {"source_bars": (1, 1), "stems": ["b"]},
    ]
    audio, applied_db = sa.assemble(variation, stems, sr, bpm,
                                    crossfade_ms=0.0)
    assert applied_db == 0.0                       # low level, no attenuation
    assert audio.shape == (2 * spb, 2)
    assert np.allclose(audio[:spb], 0.1)
    assert np.allclose(audio[spb:], 0.2)


def test_assemble_crossfade_overlaps_sections():
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.1, spb * 4)}
    variation = [
        {"source_bars": (1, 1), "stems": ["a"]},
        {"source_bars": (1, 1), "stems": ["a"]},
    ]
    xfade_samples = round(1.0 / 1000 * sr)  # 48
    audio, _ = sa.assemble(variation, stems, sr, bpm, crossfade_ms=1.0)
    # Two 1-bar sections overlapped by the crossfade length.
    assert audio.shape == (2 * spb - xfade_samples, 2)


def test_assemble_drops_fully_out_of_range_sections(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.2, spb * 2)}  # only 2 bars exist
    variation = [
        {"source_bars": (1, 2), "stems": ["a"]},
        {"source_bars": (5, 6), "stems": ["a"]},  # entirely past the end -> empty
    ]
    # The clamp warns; assemble must not crash on the zero-length section,
    # and the surviving audio is just the valid section.
    with pytest.warns(UserWarning):
        audio, _ = sa.assemble(variation, stems, sr, bpm, crossfade_ms=5.0)
    assert audio.shape == (2 * spb, 2)


def test_assemble_all_sections_empty_raises_clearly(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.2, spb * 2)}
    variation = [{"source_bars": (9, 16), "stems": ["a"]}]  # all past the end
    with pytest.warns(UserWarning):
        with pytest.raises(ValueError, match="no audio"):
            sa.assemble(variation, stems, sr, bpm, crossfade_ms=0.0)


# --- loading stems from disk --------------------------------------------

def _write_wav(path, value, frames, sr, channels=2):
    sf.write(path, np.full((frames, channels), value, dtype=np.float64),
             sr, subtype="FLOAT")


def test_load_stems_keys_are_lowercased_filenames(tmp_path):
    _write_wav(tmp_path / "Drums.wav", 0.1, 1000, 48000)
    _write_wav(tmp_path / "Bass.wav", 0.2, 1000, 48000)
    (tmp_path / "notes.txt").write_text("ignore me")  # non-wav ignored
    stems, sr = sa.load_stems(tmp_path)
    assert sr == 48000
    assert set(stems) == {"drums", "bass"}
    assert stems["drums"].shape == (1000, 2)
    assert stems["drums"].dtype == np.float64


def test_load_stems_rejects_sample_rate_mismatch(tmp_path):
    _write_wav(tmp_path / "Drums.wav", 0.1, 1000, 48000)
    _write_wav(tmp_path / "Bass.wav", 0.2, 1000, 44100)
    with pytest.raises(ValueError, match="sample rate"):
        sa.load_stems(tmp_path)


def test_load_stems_pads_length_mismatch_and_warns(tmp_path):
    _write_wav(tmp_path / "Drums.wav", 0.1, 1000, 48000)
    _write_wav(tmp_path / "Bass.wav", 0.2, 800, 48000)  # short by 200
    with pytest.warns(UserWarning, match="length"):
        stems, _ = sa.load_stems(tmp_path)
    assert stems["drums"].shape[0] == 1000
    assert stems["bass"].shape[0] == 1000        # padded up to the longest
    assert np.allclose(stems["bass"][800:], 0.0)  # padding is silence


def test_render_section_clamps_and_warns_past_end(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    stems = {"a": _stem(0.1, spb * 4)}  # only 4 bars exist
    section = {"source_bars": (3, 8), "stems": ["a"]}  # asks through bar 8
    with pytest.warns(UserWarning, match="clamping"):
        out = sa.render_section(stems, section, sr, bpm)
    assert out.shape == (2 * spb, 2)  # clamped to the available bars 3..4


# --- writing output ------------------------------------------------------

def test_write_wav_roundtrips_and_creates_parent(tmp_path):
    audio = np.full((500, 2), 0.42, dtype=np.float64)
    out = tmp_path / "nested" / "Mix.wav"
    sa.write_wav(out, audio, 48000, subtype="FLOAT")
    assert out.exists()
    back, sr = sf.read(out, dtype="float64", always_2d=True)
    assert sr == 48000
    assert np.allclose(back, audio)


# --- compressed preview --------------------------------------------------

def test_write_preview_is_compressed_and_readable(tmp_path):
    sr = 48000
    tone = 0.3 * np.sin(2 * np.pi * 440 * np.arange(sr) / sr)  # 1s
    audio = np.column_stack([tone, tone])
    wav = tmp_path / "a.wav"
    prev = tmp_path / "a.mp3"
    sa.write_wav(wav, audio, sr, subtype="PCM_24")
    sa.write_preview(prev, audio, sr)
    assert prev.exists()
    back, got_sr = sf.read(prev, dtype="float64", always_2d=True)
    assert got_sr == sr
    assert back.shape[1] == 2
    assert abs(back.shape[0] - audio.shape[0]) < sr * 0.1     # length within 100ms
    assert prev.stat().st_size < wav.stat().st_size           # actually compressed


# --- full pipeline integration ------------------------------------------

def test_render_variation_end_to_end(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    _write_wav(tmp_path / "Drums.wav", 0.1, spb * 4, sr)
    _write_wav(tmp_path / "Synth.wav", 0.2, spb * 4, sr)
    variation = [
        {"source_bars": (1, 2), "stems": ["synth"]},
        {"source_bars": (3, 4), "stems": ["drums", "synth"]},
    ]
    out = tmp_path / "out" / "Variation.wav"
    sa.render_variation(tmp_path, bpm, variation, out, crossfade_ms=0.0)
    back, got_sr = sf.read(out, dtype="float64", always_2d=True)
    assert got_sr == sr
    assert back.shape == (4 * spb, 2)
    assert np.allclose(back[:2 * spb], 0.2)        # synth-only intro
    assert np.allclose(back[2 * spb:], 0.3)        # drums + synth


# --- CLI ----------------------------------------------------------------

def test_main_renders_selected_variations(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    _write_wav(tmp_path / "Drums.wav", 0.1, spb * 4, sr)
    _write_wav(tmp_path / "Synth.wav", 0.2, spb * 4, sr)
    arrangements = {
        "mini": [
            {"source_bars": (1, 2), "stems": ["synth"]},
            {"source_bars": (3, 4), "stems": ["drums", "synth"]},
        ]
    }
    out_dir = tmp_path / "out"
    written = sa.main(
        ["--folder", str(tmp_path), "--bpm", "120",
         "--out", str(out_dir), "--crossfade-ms", "0"],
        arrangements=arrangements,
    )
    assert written == [out_dir / "mini.wav"]
    back, _ = sf.read(out_dir / "mini.wav", dtype="float64", always_2d=True)
    assert back.shape == (4 * spb, 2)


def test_main_preview_writes_compressed_alongside_wav(tmp_path):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    _write_wav(tmp_path / "Synth.wav", 0.2, spb * 4, sr)
    arrangements = {"mini": [{"source_bars": (1, 2), "stems": ["synth"]}]}
    out_dir = tmp_path / "out"
    sa.main(
        ["--folder", str(tmp_path), "--bpm", "120", "--out", str(out_dir),
         "--crossfade-ms", "0", "--preview"],
        arrangements=arrangements,
    )
    assert (out_dir / "mini.wav").exists()
    assert (out_dir / "mini.mp3").exists()


def test_main_preview_reports_mp3_in_output(tmp_path, capsys):
    sr, bpm = 48000, 120
    spb = sa.samples_per_bar(sr, bpm)
    _write_wav(tmp_path / "Synth.wav", 0.2, spb * 4, sr)
    arrangements = {"mini": [{"source_bars": (1, 2), "stems": ["synth"]}]}
    sa.main(
        ["--folder", str(tmp_path), "--bpm", "120", "--out", str(tmp_path / "o"),
         "--crossfade-ms", "0", "--preview"],
        arrangements=arrangements,
    )
    out = capsys.readouterr().out
    assert "mini.mp3" in out  # preview path is echoed, not silent


def test_main_list_reports_detected_stems(tmp_path, capsys):
    _write_wav(tmp_path / "Drums.wav", 0.1, 1000, 48000)
    _write_wav(tmp_path / "Synth.wav", 0.2, 1000, 48000)
    written = sa.main(["--folder", str(tmp_path), "--list"], arrangements={})
    assert written == []
    out = capsys.readouterr().out
    assert "drums" in out and "synth" in out
