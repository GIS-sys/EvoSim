"""Regenerates presets/*.json from code. Run: python scripts/make_presets.py"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from evosim.config import Config

OUT = os.path.join(os.path.dirname(__file__), "..", "presets")
os.makedirs(OUT, exist_ok=True)

presets = {
    # bright, nutrient-rich pond: photosynthesis pays, crowding is the limit
    "sunny_pond": dict(sun_I=1.0, ambient_I=0.05, nutrient_pattern="uniform", nutrient_cap=0.05,
                       initial_count=5000, initial_seed_files="seeds/seed_replicator.asm"),
    # almost no light; life lives on nutrient vents (grazers swim between them)
    "dark_vents": dict(sun_I=0.02, ambient_I=0.01, nutrient_pattern="vents", vent_count=12, nutrient_cap=0.4,
                       nutrient_regen=0.003, D_nutrient=1.5, initial_count=5000,
                       initial_seed_files="seeds/seed_grazer.asm"),
    # photosynthesisers + grazers + predators, predation on, mutation up a little
    "predator_world": dict(sun_I=1.0, nutrient_pattern="patches", nutrient_cap=0.08, predation_enabled=True,
                           mu_point=0.003, initial_count=8000, initial_source="seed",
                           initial_seed_files="seeds/seed_replicator.asm,seeds/seed_grazer.asm,seeds/seed_predator.asm"),
    # small world for debugging; 5k cells, cheap
    "fast_debug": dict(world_w=512.0, world_h=512.0, max_cells=16384, initial_count=5000, ledger_interval=10,
                       sun_I=1.0, initial_seed_files="seeds/seed_replicator.asm"),
}
for name, kw in presets.items():
    Config(**kw).save(os.path.join(OUT, f"{name}.json"))
    print("wrote", name)
