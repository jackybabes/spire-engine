"""Checks for the fight arrays and action slots: .venv/bin/python python/tests/test_encode.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import time

import numpy as np

import sts
from core.sts_env import ACTIONS, FLOAT_LEN, INT_LEN, LAYOUT, FightEnv

env = FightEnv(64, seed=4)
rng = np.random.default_rng(0)
wins = losses = picks = 0
t = time.time()
steps = 3000
for k in range(steps):
    obs = env.observe()
    mask = obs["mask"]
    assert mask.shape == (64, ACTIONS)
    assert mask.any(axis=1).all(), "every game must have a legal slot"
    vocab = LAYOUT["vocab"]
    assert obs["i_hand_card"].max() < vocab["card"]
    assert obs["i_monster"].max() < vocab["monster"]
    assert obs["i_player_power"].max() < vocab["power"]
    # A random legal slot per game.
    choice = np.array([rng.choice(np.flatnonzero(m)) for m in mask])
    picks += int((choice >= LAYOUT["actions"]["pick"][0]).sum())
    status = env.step(choice)
    wins += int((status == 1).sum())
    losses += int((status == 2).sum())
dt = time.time() - t
print(f"{64 * steps / dt:,.0f} encoded steps/s; {wins} won, {losses} lost, {picks} selection picks")
assert wins + losses > 0 and picks > 0
print("ints", INT_LEN, "floats", FLOAT_LEN, "actions", ACTIONS, "vocab", LAYOUT["vocab"])
print("ok")
