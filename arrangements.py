"""Editable arrangement variations consumed by stem_arranger.py.

This is the file you tweak. Nothing here is magic — it's plain data.

A variation is an ordered list of SECTIONS. Each section says:
    - source_bars: (start, end)  1-indexed, inclusive bar range to pull from
                                 the ORIGINAL song (so you can reorder/reuse
                                 the real musical content: repeat the chorus,
                                 lift the breakdown from where it lives, etc.)
    - stems:       [names]       which stems play in this section. A stem you
                                 leave out is silent there.

Sections are concatenated in list order to build the final arrangement.

STEM NAMES are the lowercased filenames (no extension) of the WAVs in your
stem folder. "Drums.wav" -> "drums", "Vocals.wav" -> "vocals". Run:
    python stem_arranger.py --folder stems/ --list
to print exactly what keys your folder exposes, then make the names below
match.

>>> IMPORTANT: the source_bars below are PLACEHOLDERS for a ~48-bar song.
    Adjust them to your track's actual structure (where its intro/build/
    drop/breakdown actually sit). <<<
"""

# Rename these to match your stems (see --list). Defined once, reused below.
DRUMS = "drums"
BASS = "bass"
SYNTH = "synth"
VOCAL = "vocals"


EXTENDED_MIX = [
    {"source_bars": (1, 16),  "stems": [DRUMS, BASS]},                  # DJ intro: rhythm bed
    {"source_bars": (17, 24), "stems": [DRUMS, BASS, SYNTH]},           # build
    {"source_bars": (25, 40), "stems": [DRUMS, BASS, SYNTH, VOCAL]},    # full body
    {"source_bars": (17, 24), "stems": [SYNTH, VOCAL]},                 # breakdown (drop drums)
    {"source_bars": (25, 40), "stems": [DRUMS, BASS, SYNTH, VOCAL]},    # full body again
    {"source_bars": (1, 16),  "stems": [DRUMS, BASS]},                  # DJ outro
]

RADIO_EDIT = [
    {"source_bars": (1, 4),   "stems": [DRUMS, SYNTH]},                 # short intro
    {"source_bars": (25, 32), "stems": [DRUMS, BASS, SYNTH, VOCAL]},    # straight to the hook
    {"source_bars": (17, 20), "stems": [SYNTH, VOCAL]},                 # quick breakdown
    {"source_bars": (33, 40), "stems": [DRUMS, BASS, SYNTH, VOCAL]},    # final chorus
    {"source_bars": (41, 44), "stems": [SYNTH]},                        # short tail
]

SUBTRACTIVE_AMBIENT = [
    {"source_bars": (17, 24), "stems": [SYNTH, VOCAL]},                 # pads + voice only
    {"source_bars": (25, 32), "stems": [SYNTH]},                        # pads alone
    {"source_bars": (33, 40), "stems": [SYNTH, VOCAL]},
    {"source_bars": (1, 8),   "stems": [SYNTH]},                        # dissolve
]


# The table stem_arranger reads. Keys become output filenames (e.g. out/extended.wav).
ARRANGEMENTS = {
    "extended": EXTENDED_MIX,
    "radio": RADIO_EDIT,
    "ambient": SUBTRACTIVE_AMBIENT,
}
