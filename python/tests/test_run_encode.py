"""Checks for the run arrays and action slots: .venv/bin/python python/tests/test_run_encode.py [steps]

Plays runs with a random legal slot each step: every game must always have a legal slot, every
chosen slot must be accepted, IDs must stay inside their vocabularies, and every kind of screen
and option should come up.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import collections
import sys
import time

import numpy as np

from core.sts_env import ACTIONS, LAYOUT, RunEnv

N = 256
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
SCREENS = ["-"] + LAYOUT["screens"]
KINDS = ["-"] + LAYOUT["option_kinds"]
V = LAYOUT["vocab"]
VOCAB_OF = {
    "i_run_relic": "relic", "i_run_potion": "potion", "i_deck": "card", "i_opt_card": "card", "i_opt_relic": "relic",
    "i_opt_potion": "potion", "i_opt_kind": "option_kind", "i_screen": "screen", "i_boss": "encounter", "i_room": "room",
    "i_opt_room": "room", "i_event": "event", "i_event_screen": "event_screen", "i_opt_index": "event_index",
    "i_grid_purpose": "grid_purpose", "i_opt_neow": "neow", "i_opt_drawback": "neow_drawback",
    "i_hand_card": "card", "i_monster": "monster", "i_player_power": "power",
}

env = RunEnv(N, seed=11)
rng = np.random.default_rng(0)
screens, kinds = collections.Counter(), collections.Counter()
most_options = most_deck = 0
ended, floors, acts = 0, [], collections.Counter()
t = time.time()
for step in range(STEPS):
    obs = env.observe()
    mask = obs["mask"]
    assert mask.shape == (N, ACTIONS)
    assert mask.any(axis=1).all(), f"step {step}: a game has no legal slot"
    for field, vocab in VOCAB_OF.items():
        assert obs[field].min() >= 0 and obs[field].max() < V[vocab], f"{field} outside {vocab} vocabulary"
    for s in obs["i_screen"][:, 0]:
        screens[SCREENS[s]] += 1
    for k in obs["i_opt_kind"].ravel():
        if k:
            kinds[KINDS[k]] += 1
    most_options = max(most_options, int((obs["i_opt_kind"] > 0).sum(1).max()))
    most_deck = max(most_deck, int(obs["f_run_scalars"][:, 6].max()))
    # Options outside combat must be legal only where an option exists.
    outside = obs["i_screen"][:, 0] != SCREENS.index("combat")
    pick0, _ = LAYOUT["actions"]["pick"]
    has_opt = obs["i_opt_kind"] > 0
    assert not (mask[outside, pick0:pick0 + 64] & ~has_opt[outside]).any(), "a legal pick slot without an option"
    status = env.step(np.array([rng.choice(np.flatnonzero(m)) for m in mask]))
    for g in np.flatnonzero(status):
        import json
        ep = json.loads(env.last_episode(int(g)))
        ended += 1
        floors.append(ep["floor"])
        acts[ep["act"]] += 1
dt = time.time() - t
print(f"{N * STEPS / dt:,.0f} encoded run steps/s; {ended} runs ended, mean floor {np.mean(floors):.1f}, max {max(floors)}, by act {dict(acts)}")
print("screens:", dict(screens.most_common()))
print("option kinds:", dict(kinds.most_common()))
print(f"most options on a screen {most_options}, biggest deck {most_deck}")
missing = [k for k in LAYOUT["option_kinds"] if k not in kinds and k not in LAYOUT.get("retired_option_kinds", [])]
print("option kinds never seen:", missing)
print("ok")
