"""Deterministic HDRI choice.

The manifest lives in the container, but scenario sampling must be reproducible
on the host too (and must not need network access just to build a SceneSpec).
So the id list is baked here and the worker resolves it against the live asset
source. Regenerate with scripts/refresh_hdri_ids.py if the manifest changes.
"""
from __future__ import annotations

from typing import List

# A curated indoor/outdoor spread from HDRI Haven's 458-asset train split.
HDRI_IDS: List[str] = [
    "abandoned_games_room_01", "abandoned_church", "adams_place_bridge",
    "aerodynamics_workshop", "art_studio", "autoshop_01", "bathroom",
    "bell_park_pier", "blaubeuren_night", "cabin", "circus_arena",
    "colorful_studio", "cyclorama_hard_light", "dancing_hall", "de_balie",
    "derelict_underpass", "empty_warehouse_01", "furstenstein",
    "garden_nook", "graffiti_shelter", "hansaplatz", "hotel_room",
    "industrial_pipe_and_valve_01", "killesberg_park", "lebombo",
    "lythwood_room", "modern_buildings_2", "museum_of_ethnography",
    "old_bus_depot", "paul_lobe_haus", "photo_studio_01", "reading_room",
    "royal_esplanade", "school_hall", "shanghai_bund", "small_empty_room_1",
    "st_fagans_interior", "studio_small_03", "surgery", "teufelsberg_lookout",
    "urban_alley_01", "vulture_hide", "wooden_lounge", "workshop",
]


def pick(rng) -> str:
    return str(HDRI_IDS[int(rng.randint(0, len(HDRI_IDS)))])
