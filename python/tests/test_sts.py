"""Checks for the `sts` module (plain asserts, no pytest needed): python3 python/tests/test_sts.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import json

import sts


def play(mode, seed, steps):
    envs = sts.Envs(4, mode=mode, seed=seed)
    trace = []
    for k in range(steps):
        trace.append(tuple(envs.step_random(k)))
    return trace, [envs.observe(i) for i in range(4)]


# Same seeds, same games.
for mode in ("fight", "run"):
    assert play(mode, 3, 300) == play(mode, 3, 300), mode

# A bad action index is an error, not a crash.
envs = sts.Envs(2, mode="fight", seed=1)
try:
    envs.step([999, 0])
    raise AssertionError("expected an error")
except ValueError as e:
    assert "out of range" in str(e)

# Finished games start a new episode and report the old one.
envs = sts.Envs(8, mode="fight", seed=2, encounters=["Cultist"])
finished = 0
for k in range(500):
    finished += sum(1 for s in envs.step_random(k) if s)
assert finished > 0
assert any(envs.last_episode(i) for i in range(8))
assert all(n > 0 for n in envs.num_legal())

# The observation is the engine's JSON view, and legal actions parse as JSON.
obs = json.loads(envs.observe(0))
assert "legal_actions" in obs
assert all(json.loads(a)["type"] for a in envs.legal_actions(0))

# Runs start at Neow.
runs = sts.Envs(2, mode="run", seed=9)
assert json.loads(runs.legal_actions(0)[0])["type"] == "neow"
assert "The Heart" in sts.all_encounters()

# Observations written into reused buffers (RunEnv(reuse_obs=True), Envs.observe_into) are the
# fresh arrays', step after step, and the last one holds through the next observe.
import numpy as np

from core.sts_env import RunEnv

fresh, reused = RunEnv(16, seed=5, character="watcher"), RunEnv(16, seed=5, character="watcher", reuse_obs=True)
rng = np.random.default_rng(5)
last = None
for _ in range(150):
    a, b = fresh.observe(), reused.observe()
    if last is not None:
        assert all(np.array_equal(last[0][k], last[1][k]) for k in ("ints", "floats", "mask")), "the last observation, one observe later"
    assert all(np.array_equal(a[k], b[k]) for k in ("ints", "floats", "mask"))
    last = ({k: a[k].copy() for k in ("ints", "floats", "mask")}, b)
    acts = [int(rng.choice(np.flatnonzero(m))) for m in a["mask"]]
    fresh.step(acts), reused.step(acts)

print("ok")
