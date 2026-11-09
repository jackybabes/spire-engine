"""Checks for late starts (plain asserts, no pytest needed; on the CPU, no network to speak of):

    python python/tests/test_late_starts.py

Envs.set_late_starts: from each game's next episode a share of the games neither practice fights nor seeded start from a random line, their
first observation showing its deck, floor and HP, no keys and no practice mask; every episode of those games and no others' carries "late";
the other games are unchanged seed by seed, and all of them with no share. train_runs: the pool by group, weights held, the same on a second reading;
a late run's rewards are a run's from where it starts (nothing for the floors and acts behind it at its first step or its last, nothing from
the habit rewards for its setup), and it still earns the Act 3 boss, the Heart and the key rules, with keys it took on the way to Act 4 too; the
habit counts are the other games'; the trainer runs with them, seeded runs, a resume and either side of the split, and its log counts them apart from the runs."""
import contextlib
import csv
import gzip
import io
import json
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fixed_shapes import FixedShapes
from core.run_model import COMBAT, PICK, RUN_FLOAT_AT, RUN_INT_AT, RunNet
from core.sts_env import ACTIONS, LAYOUT, RunEnv
from training.train_runs import RS, W_ACT, W_FLOOR, Collector, late_start_pool, main, monster_hp, parse_args, rollout_buffers

DECK = [["Strike_G", 0], ["Strike_G", 0], ["Defend_G", 1], ["Neutralize", 1], ["Survivor", 0], ["Noxious Fumes", 1], ["Catalyst", 0], ["Acrobatics", 1]]
NAMES = ["Strike", "Strike", "Defend", "Neutralize", "Survivor", "Noxious Fumes", "Catalyst", "Acrobatics"]
OUTCOMES = {"won": ("won", None), "heart": ("heart", None), "died1": ("died", 1), "died2": ("died", 2), "died3": ("died", 3), "died4": ("died", 4)}


def state(act=2, hp=41, max_hp=70, deck=DECK, label="won", play_id="p"):
    """A human_bosses.py --all-outcomes line."""
    floor, bosses = {1: (16, ["Hexaghost"]), 2: (33, ["Champ"])}[act]
    name, died_act = OUTCOMES[label]
    outcome = {"label": name, "floor_reached": 52, **({"died_act": died_act, "killed_by": "Champ"} if died_act else {})}
    return json.dumps({"play_id": play_id, "character": "THE_SILENT", "ascension": 20, "act": act, "floor": floor, "hp": hp, "max_hp": max_hp, "bosses": bosses, "deck": deck,
                       "relics": ["Ring of the Snake", "Vajra"], "potions": ["Poison Potion"], "outcome": outcome, "gold": 321, "purges": 1})


def scalars(obs, g):
    """Game g's HP, max HP, gold, floor and act; and its keys."""
    return tuple(obs["f_run_scalars"][g, :5].tolist()), obs["f_run_scalars"][g, 7:10].sum()


def deck_of(obs, g):
    names = LAYOUT["card_names"]
    return sorted((names[i - 1], u) for i, u in zip(obs["i_deck"][g].tolist(), obs["f_deck_upgrades"][g].tolist()) if i)


# A game's first observation shows the state; from each game's next episode only, so a reset shows it at once.
lines = [state(act=2, hp=41), state(act=1, hp=52)]
env = RunEnv(4, seed=5, ascension=20, character="silent")
assert env.envs.set_late_starts(lines, 1.0) == 0
assert all(scalars(env.observe(), g)[0][3] == 0 for g in range(4)), "the episodes being played go on"
env.envs.reset()
obs = env.observe()
want = sorted((n, float(up)) for n, (_, up) in zip(NAMES, DECK))
for g in range(4):
    (hp, max_hp, gold, floor, act), keys = scalars(obs, g)
    assert (hp, max_hp, gold, floor, act) in ((41, 70, 321, 33, 2), (52, 70, 321, 16, 1)), (hp, max_hp, gold, floor, act)
    assert keys == 0, "no keys given"
    assert deck_of(obs, g) == want, deck_of(obs, g)
    assert obs["i_screen"][g, 0] == COMBAT
assert list(env.envs.practice_mask()) == [False] * 4 and list(env.envs.late_mask()) == [True] * 4
# What it can't start from is skipped: another character's, what the engine doesn't have, a fight that isn't an act's boss, the Heart.
for bad in (state().replace("THE_SILENT", "DEFECT"), state().replace("Champ", "Hexaghost"), state().replace('"purges"', '"room": "elite", "purges"').replace("Champ", "Slavers"),
            state().replace('"act": 2', '"act": 4').replace("Champ", "The Heart")):
    assert env.envs.set_late_starts([bad, lines[0]], 1.0) == 1, bad
try:
    env.envs.set_late_starts(lines, 1.5)
    raise AssertionError("a share past 1")
except ValueError:
    pass

# Who plays one: the first of the share of the games left, every episode, whichever line. Its next episode after a plain run's is one too.
env = RunEnv(64, seed=7, ascension=20, character="silent", practice=0.25, seeded=0.25, practice_decks="mix", max_steps=3000)
env.envs.set_late_starts(lines, 0.25)
assert scalars(env.observe(), 32)[0][3] == 0
names, starts, decisions = {g: set() for g in range(64)}, {g: [] for g in range(64)}, {"late": 0, None: 0}
for step in range(6000):
    status = env.envs.step_random(step)
    if any(status):
        obs = env.observe()
        for g in np.flatnonzero(status):
            e = json.loads(env.last_episode(int(g)))
            names[g].add(e.get("seeded") if "practice" not in e else "practice")
            starts[g].append(scalars(obs, g)[0][3])
            if g >= 32:
                decisions[e["seeded"]] += e["steps"]
            if e.get("seeded") == "late":
                assert e["floor"] >= 16 and "scenario" not in e and "retry" not in e and e["synergy_picks"] == 0, e
assert all(names[g] == {"practice"} for g in range(16)), [names[g] for g in range(16)]
assert all(names[g] and None not in names[g] and "late" not in names[g] and "practice" not in names[g] for g in range(16, 32))
# (A late game's first episode was a plain run: it was under way when they were set.)
assert all(names[g] == {None, "late"} for g in range(32, 40)), [names[g] for g in range(32, 40)]
assert all(names[g] == {None} for g in range(40, 64)), [names[g] for g in range(40, 64)]
assert list(np.flatnonzero(env.envs.late_mask())) == list(range(32, 40))
# Each episode from a random line: both in about equal shares, over many.
drawn = [f for g in range(32, 40) for f in starts[g]]
assert len(drawn) >= 200 and set(drawn) == {16, 33} and 0.35 < drawn.count(16) / len(drawn) < 0.65, (len(drawn), set(drawn))
assert all(f == 0 for g in range(40, 64) for f in starts[g])
# The share is of games, whatever each one's episodes last, up to a game's rounding (so of decisions too, over many episodes); of the games left after practice and seeds.
assert 0.2 < decisions["late"] / (decisions["late"] + decisions[None]) < 0.3, decisions
for n, practice, seeded, share, late in ((64, 0, 0, 0.25, 16), (10, 0, 0, 0.25, 3), (512, 0.3, 0, 0.25, 90), (512, 0.3, 0.2, 0.25, 64), (8, 0, 0, 1.0, 8), (8, 0, 0, 0.0, 0)):
    env = RunEnv(n, seed=3, ascension=20, character="silent", practice=practice, seeded=seeded, practice_decks="mix" if practice else None)
    env.envs.set_late_starts(lines, share)
    env.envs.reset()
    obs = env.observe()
    counted = int(((obs["f_run_scalars"][:, 3] > 0) & ~np.asarray(env.envs.practice_mask())).sum())
    assert counted == late, (n, practice, seeded, share, counted)


# Ordinary games are unchanged seed by seed: with a share of 0 every game, with a share the others.
def stream(env, steps=500):
    """Each step's rows, each game playing moves of its own (its own random stream reading only its own rows)."""
    rngs = [np.random.default_rng(g) for g in range(len(env.envs))]
    ints, floats, status = [], [], []
    for _ in range(steps):
        obs = env.observe()
        ints.append(obs["ints"])
        floats.append(obs["floats"])
        status.append(env.step([rngs[g].choice(np.flatnonzero(obs["mask"][g])) for g in range(len(env.envs))]))
    return np.stack(ints), np.stack(floats), np.stack(status), [env.envs.episode_seed(g) for g in range(len(env.envs))]


def make(share=None):
    env = RunEnv(8, seed=11, ascension=20, character="silent")
    if share is not None:
        assert env.envs.set_late_starts(lines, share) == 0
        env.envs.reset()
    return env


plain, none, some = stream(make()), stream(make(0.0)), stream(make(0.25))
assert (plain[2] > 0).sum() >= 8, "some episodes ended and others began"
assert all(np.array_equal(a, b) for a, b in zip(plain[:3], none[:3])) and plain[3] == none[3]
assert all(np.array_equal(a[:, 2:], b[:, 2:]) for a, b in zip(plain[:3], some[:3])) and plain[3][2:] == some[3][2:]
assert (some[1][0, :2, RS + 3] > 0).all() and (plain[1][0, :2, RS + 3] == 0).all(), "the first two are late"
# The same run, move for move, as analysis/play_on.py's from the same line.
late_env, on_env = RunEnv(4, seed=13, ascension=20, character="silent"), RunEnv(4, seed=13, ascension=20, character="silent", practice=1.0)
late_env.envs.set_late_starts(lines[:1], 1.0)
late_env.envs.reset()
on_env.envs.set_scenarios(lines[:1], 1.0, True, True)
assert all(np.array_equal(a, b) for a, b in zip(stream(late_env, 200)[:3], stream(on_env, 200)[:3]))


# The rewards, from the training code's own rollouts, every legal move alike.
class Uniform:
    def eval(self):
        return self

    def __call__(self, ints, floats, practice):
        return torch.zeros(len(ints), ACTIONS), torch.zeros(len(ints))


def collector(flags, late, envs=8, steps=128):
    args = parse_args(["--name", "test", "--character", "silent", "--ascension", "20", "--envs", str(envs), "--steps", str(steps), "--eval-runs", "0", "--gamma", "1", *flags])
    return args, Collector(args, torch.device("cpu"), contextlib.nullcontext, 0, late_lines=late)


def record(col):
    """Keeps what Collector._step_rewards gives each step, and what it read to give it."""
    rows, inner = [], col._step_rewards

    def step(env, before, obs, status, acts, start_mon, prog, psi, weights, acc):
        rew, done, new_prog, new_psi = inner(env, before, obs, status, acts, start_mon, prog, psi, weights, acc)
        rows.append({"rew": rew.copy(), "done": done.copy(), "psi": np.array(psi), "prog": np.array(prog), "floor": before["floats"][:, RS + 3].copy(), "act": before["floats"][:, RS + 4].copy(),
                     "keys": before["floats"][:, RS + 7:RS + 10].sum(1), "habits": np.asarray(env.envs.habits()).reshape(len(acts), -1),
                     "eps": {int(g): json.loads(env.last_episode(int(g))) for g in np.flatnonzero(done)}})
        return rew, done, new_prog, new_psi

    col._step_rewards = step
    return rows


def episodes(rows, games):
    """Each finished episode's summary, where it started, its first and last rewards, and all of its rewards added up."""
    out = []
    for g in range(games):
        start = 0
        for t, r in enumerate(rows):
            if r["done"][g]:
                part = rows[start:t + 1]
                out.append({"ep": r["eps"][g], "floor": part[0]["floor"][g], "act": part[0]["act"][g], "keys": part[0]["keys"][g], "psi": part[0]["psi"][g], "first": part[0]["rew"][g],
                            "total": float(sum(float(x["rew"][g]) for x in part)), "last": r["rew"][g], "prog_last": r["prog"][g], "psi_last": r["psi"][g], "added": part[0]["habits"][g, 0],
                            "removed": part[0]["habits"][g, 1], "moves": len(part)})
                start = t + 1
    return out


def owed(e, args):
    """What a run has earned from where it started (gamma 1, so shaping adds up to minus its first potential): the floors, acts and bosses since; the
    Act 3 boss and the Heart, with or without the keys; a key only by the rule, and only one it took itself."""
    ep = e["ep"]
    keys = ep["keys"] if ep["act"] >= 4 or not args.key_bonus_act4 else 0
    return (args.act3_reward * (ep["won"] or ep["act"] >= 4) + args.heart_reward * (ep["won"] and ep["act"] >= 4) + W_FLOOR * (ep["floor"] - e["floor"]) + W_ACT * (ep["act"] - e["act"])
            + args.boss_reward * (min(ep["act"] - 1, 2) - min(e["act"] - 1, 2)) + args.key_bonus * (keys - e["keys"]) - e["psi"])


def play(col, rows, enough, rollouts=80, policy=Uniform()):
    """Rollouts until `enough` of the finished episodes."""
    out = rollout_buffers(col.args.steps, col.args.envs, "cpu")
    for u in range(rollouts):
        col.collect(u, policy, out)
        eps = episodes(rows, col.args.envs)
        if enough(eps):
            return eps
    raise AssertionError(f"{len(eps)} episodes in {rollouts} rollouts: {[e['ep'] for e in eps][:5]}")


def wrong(eps, args):
    return [(e["total"], owed(e, args), e["ep"]) for e in eps if abs(e["total"] - owed(e, args)) >= 2e-3]


# Plain runs first (nothing set yet), late starts from each game's next episode: the jump in floor and act between the two is never paid.
RUN_REWARDS = ["--act3-reward", "3", "--heart-reward", "9", "--key-bonus", "0.15", "--key-bonus-act4", "--boss-reward", "0.2"]
torch.manual_seed(0)
args, col = collector(RUN_REWARDS, None)
rows = record(col)
col.env.envs.set_late_starts([state(act=2, hp=41)], 1.0)
eps = play(col, rows, lambda eps: sum(e["ep"].get("seeded") == "late" for e in eps) >= 16)
assert not wrong(eps, args), wrong(eps, args)[:3]
late = [e for e in eps if e["ep"].get("seeded") == "late"]
ordinary = [e for e in eps if e["ep"].get("seeded") is None]
assert ordinary and all(e["floor"] == 0 and e["act"] == 1 for e in ordinary)
assert all((e["floor"], e["act"], e["keys"]) == (33, 2, 0) for e in late)
# Not the 0.96 of floor 33, Act 2 and a boss: only the potential's change.
assert all(abs(e["first"]) < 0.3 for e in eps), [e["first"] for e in eps if abs(e["first"]) >= 0.3][:3]
# A death at the boss it started at earns only the end of the shaping: neither plus nor minus where it stood.
at_start = [e for e in late if e["ep"]["floor"] == 33]
assert at_start and all(abs(e["total"] + e["psi"]) < 1e-3 for e in at_start), [(e["total"], e["psi"]) for e in at_start][:3]

# On HP no hallway gets through, and a deck that wins with moves at random: to the end of Act 3. A win there earns the Act 3 boss with no keys, and keys it took are the rule's.
STRONG = state(act=2, hp=100000, max_hp=100000, deck=[["Slice", 1]] * 30)
torch.manual_seed(1)
args, col = collector([*RUN_REWARDS, "--late-share", "1"], [STRONG])
rows = record(col)
eps = play(col, rows, lambda eps: sum(e["ep"]["won"] for e in eps) >= 3 and any(e["ep"]["keys"] and e["ep"]["act"] < 4 for e in eps))
assert not wrong(eps, args), wrong(eps, args)[:3]
won = [e for e in eps if e["ep"]["won"] and e["ep"]["act"] == 3]
assert won and all(e["ep"]["seeded"] == "late" and e["ep"]["floor"] == 52 for e in won)
# 3 for the Act 3 boss, 19 floors at 0.02, an act and a boss on top: what a run from floor 0 is paid for those same floors. (Less any key the rule took back: it paid and took it.)
assert all(abs(e["total"] + e["psi"] - (3 + 0.02 * 19 + 0.1 + 0.2)) < 2e-3 for e in won), [(e["total"], e["psi"], e["ep"]["keys"]) for e in won]
print(f"to the end from Act 2's boss: {len(eps)} runs, {len(won)} won, longest {max(e['moves'] for e in eps)} moves")

# With the keys in hand (the evaluations' switch: late starts themselves get none) the same run goes on: the Heart earns its reward on top, and the keys held from the start pay nothing.
torch.manual_seed(2)
args, col = collector([*RUN_REWARDS, "--late-share", "1"], [STRONG])
col.env.envs.set_free_keys(True)
col.env.envs.reset()
col.obs = col.env.observe()
col.start_mon[:] = monster_hp(col.obs["floats"])
rows = record(col)
eps = play(col, rows, lambda eps: any(e["ep"]["won"] and e["ep"]["act"] == 4 for e in eps))
hearts = [e for e in eps if e["ep"]["won"] and e["ep"]["act"] == 4]
assert all(e["keys"] == 3 for e in eps) and not wrong(eps, args), wrong(eps, args)[:3]
assert all(e["total"] > 3 + 9 for e in hearts), [e["total"] for e in hearts]
print(f"with the keys: {len(eps)} runs, {len(hearts)} beat the Heart")


class KeyTaker(Uniform):
    """Every legal move alike, but a key whenever one is on offer and the path to a burning elite when it's next."""
    KEYS = torch.tensor([1 + LAYOUT["option_kinds"].index(k) for k in ("emerald_key", "sapphire_key", "recall")])
    PATH = 1 + LAYOUT["option_kinds"].index("path")
    KINDS, OPTS = RUN_INT_AT["opt_kind"], RUN_FLOAT_AT["opt"][0]

    def __call__(self, ints, floats, practice):
        logits, value = super().__call__(ints, floats, practice)
        kinds = ints[:, self.KINDS[0]:self.KINDS[1]]
        burning = (kinds == self.PATH) & (floats[:, self.OPTS + 6:self.OPTS + 19 * kinds.shape[1]:19] > 0)  # (19 numbers an option, the seventh the key)
        logits[:, PICK:PICK + kinds.shape[1]] += 50.0 * (torch.isin(kinds, self.KEYS) | burning).float()
        return logits, value


# No keys given, and the three taken on the way: Act 4 from either boss, with the key rule's bonus kept, and with it given back by those that took some and didn't get there.
torch.manual_seed(6)
args, col = collector([*RUN_REWARDS, "--late-share", "1"], [STRONG, state(act=1, hp=100000, max_hp=100000, deck=[["Slice", 1]] * 30)], envs=16)
rows = record(col)
eps = play(col, rows, lambda eps: any(e["ep"]["act"] == 4 for e in eps) and any(0 < e["ep"]["keys"] and e["ep"]["act"] < 4 for e in eps), policy=KeyTaker())
assert all(e["keys"] == 0 for e in eps) and not wrong(eps, args), wrong(eps, args)[:3]
act4 = [e for e in eps if e["ep"]["act"] == 4]
assert all(e["ep"]["keys"] == 3 and e["total"] > 3 + 3 * args.key_bonus for e in act4), [(e["ep"]["keys"], e["total"]) for e in act4]
print(f"keys on the way: {len(eps)} runs, {len(act4)} reached Act 4, from floors {sorted({int(e['floor']) for e in act4})}")

# Habit rewards on: none fires on the setup, neither at a late run's first step nor on the step that ended the episode before it.
HABITS = ["--deck-weight", "0.5", "--deck-free", "0", "--strike-keep-cost", "0.1", "--curse-keep-cost", "0.1", "--strike-weight", "0.3", "--curse-weight", "0.3", "--synergy-bonus", "1", "--archetype-bonus", "0.5",
          "--smith-bonus", "0.2", "--elite-bonus", "0.2", "--rest-penalty", "0.2", "--recall-penalty", "0.2", "--removal-skip-penalty", "0.2", "--strike-upgrade-penalty", "0.2", "--hallway-potion-penalty", "0.2"]
torch.manual_seed(3)
args, col = collector(HABITS, None)
rows = record(col)
col.env.envs.set_late_starts([state(act=2, hp=41)], 1.0)
eps = play(col, rows, lambda eps: sum(e["ep"].get("seeded") == "late" for e in eps) >= 8)
late = [e for e in eps if e["ep"].get("seeded") == "late"]
assert all(abs(e["first"]) < 0.3 and e["added"] == 0 and e["removed"] == 0 for e in late), [(e["first"], e["added"], e["removed"]) for e in late][:3]
# The last step is the win if any, and the end of the progress and shaping: nothing on top.
for e in eps:
    last = args.win_reward * e["ep"]["won"] + W_FLOOR * e["ep"]["floor"] + W_ACT * (e["ep"]["act"] - 1) - e["prog_last"] - e["psi_last"]
    assert abs(e["last"] - last) < 1e-3, (e["last"], last, e["ep"])

# The habit counts are the other games': half of these are late starts, adding cards and climbing floors, and only the plain runs are counted, per decision of theirs.
torch.manual_seed(7)
args, col = collector(["--late-share", "0.5"], [STRONG], envs=8)
rows = record(col)
ro = col.collect(0, Uniform(), rollout_buffers(args.steps, args.envs, "cpu"))
added = np.sum([r["habits"][:, 0] for r in rows], axis=0)
climbed = np.sum([np.maximum(b["floor"] - a["floor"], 0) * ~a["done"] for a, b in zip(rows, rows[1:])], axis=0)
assert added[:4].sum() > 0 and climbed[:4].sum() > 0, (added, climbed)
assert ro["habit_rows"] == args.steps * 4 and ro["habits"]["cards_added"] == added[4:].sum(), (ro["habit_rows"], ro["habits"]["cards_added"], added)
args, col = collector(["--late-share", "1"], [STRONG], envs=8)
ro = col.collect(0, Uniform(), rollout_buffers(args.steps, args.envs, "cpu"))
assert ro["habit_rows"] == 0 and not any(ro["habits"].values()) and ro["strikes_held"] == 0, ro["habits"]
args, col = collector([], None, envs=8)
assert col.collect(0, Uniform(), rollout_buffers(args.steps, args.envs, "cpu"))["habit_rows"] == args.steps * args.envs

# With the fixed-shape passes (uncompiled here) and the reward scale.
torch.manual_seed(4)
net = RunNet(d=16, layers=1, heads=2)
args, col = collector(["--fixed-shapes", "--scale-rewards", "--late-share", "0.5"], lines, envs=16, steps=48)
out = rollout_buffers(args.steps, args.envs, "cpu")
ro = col.collect(0, FixedShapes(net, compile=False), out)
assert (out["mask"].gather(2, out["act"][..., None]).squeeze(-1)).all() and not out["prac"].any()
assert torch.isfinite(out["adv"]).all() and torch.isfinite(out["ret"]).all()
assert ro["seeded_now"] and all(e["seeded"] == "late" for e in ro["seeded_now"]) and all(e["seeded"] is None for e in ro["runs_now"]) and not ro["practice_now"]

# The pool: the groups by weight at each boss, the same on a second reading, from every file passed, what the engine can't play left out.
tmp = tempfile.mkdtemp()
with open(f"{tmp}/a.jsonl", "w") as a, gzip.open(f"{tmp}/b.jsonl.gz", "wt") as b:
    for i in range(600):
        label = ("won", "heart", "died3", "died4", "died2", "died2", "died1")[i % 7]
        for act in (1, 2) if label != "died1" else (1,):  # (those who died in Act 1 fought no Act 2 boss)
            (a if i % 2 else b).write(state(act=act, label=label, play_id=f"p{i}") + "\n")
    a.write(state(act=1, label="won", play_id="bad").replace("Hexaghost", "Champ") + "\n")
paths = f"{tmp}/a.jsonl,{tmp}/b.jsonl.gz"


def pool(*flags):
    return late_start_pool(parse_args(["--late-starts", paths, *flags]))


def by_group(pool_lines):
    groups = {"won": "winners", "heart": "winners", "died3": "finishers", "died4": "finishers", "died2": "act2-deaths", "died1": "act1-deaths"}
    counts = {}
    for x in pool_lines:
        s = json.loads(x)
        key = (s["act"], groups[s["outcome"]["label"] + str(s["outcome"].get("died_act", ""))])
        counts[key] = counts.get(key, 0) + 1
    return counts


pool_lines, about = pool("--late-limit", "120", "--practice", "0.3")
assert by_group(pool_lines) == {(1, "winners"): 30, (1, "finishers"): 15, (1, "act2-deaths"): 15, (2, "winners"): 30, (2, "finishers"): 15, (2, "act2-deaths"): 15}, by_group(pool_lines)
assert pool("--late-limit", "120", "--practice", "0.3")[0] == pool_lines
assert "120 human boss states" in about and "other finishers 15 of 172" in about and "17.5% of all" in about, about
assert all(json.loads(x)["play_id"] != "bad" for x in pool("--late-limit", "1000", "--late-mix", "winners:1")[0])
# A group short of its share holds the others back with it: the weights stay.
pool_lines, _ = pool("--late-limit", "4000")
assert by_group(pool_lines) == {(1, "winners"): 172, (1, "finishers"): 86, (1, "act2-deaths"): 86, (2, "winners"): 172, (2, "finishers"): 86, (2, "act2-deaths"): 86}, by_group(pool_lines)
# Another mix, and a group with no lines at one of the bosses: the rest there share its part.
pool_lines, _ = pool("--late-limit", "60", "--late-mix", "winners:1,act1-deaths:1")
assert by_group(pool_lines) == {(1, "winners"): 15, (1, "act1-deaths"): 15, (2, "winners"): 30}, by_group(pool_lines)
for flags in (["--late-mix", "winners"], ["--late-mix", "champions:1"], ["--late-mix", "winners:0"], ["--late-kinds", "boss4"], ["--late-kinds", "elite2"], ["--late-kinds", "boss3"],
              ["--late-kinds", "boss2", "--late-mix", "winners:1,act1-deaths:1"]):
    try:
        pool(*flags)
        raise AssertionError(flags)
    except SystemExit as e:
        assert "--late" in str(e), (flags, e)
with open(f"{tmp}/old.jsonl", "w") as f:
    f.write(json.dumps({k: v for k, v in json.loads(state()).items() if k != "outcome"}) + "\n")
try:
    late_start_pool(parse_args(["--late-starts", f"{tmp}/old.jsonl"]))
    raise AssertionError("a file without outcomes")
except SystemExit as e:
    assert "--all-outcomes" in str(e), e


# The trainer, a tiny network for a few updates: the log counts them apart from the runs, a resume carries on, seeded runs lump (and say so), the evals' own games
# stay plain, and it trains either side of the split on them: nothing there stands in their way.
def train(name, *flags):
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        main(["--name", f"{tmp}/{name}", "--character", "silent", "--ascension", "20", "--d", "16", "--layers", "1", "--heads", "2", "--envs", "8", "--steps", "96", "--minibatch", "64", "--epochs", "1",
              "--device", "cpu", "--port", "0", "--save-every", "0", "--eval-every", "0", "--eval-runs", "0", "--practice-eval", "0", "--scale-rewards", "--late-starts", paths, "--late-share", "0.5", "--late-limit", "60", *flags])
    with open(f"{tmp}/{name}/log.csv") as f:
        return said.getvalue(), list(csv.DictReader(f))


said, log = train("run", "--updates", "2")
assert "58 human boss states" in said and "late won" in said and "warning" not in said, said
assert len(log) == 2 and int(log[0]["seeded_runs"]) + int(log[1]["seeded_runs"]) > 0 and float(log[-1]["seeded_floor"]) >= 16, log[-1]
assert float(log[-1]["seeded_win"]) >= 0 and float(log[-1]["floor"]) < 16, "ordinary runs from the bottom"
with open(f"{tmp}/run/args.json") as f:
    assert json.load(f)["late_starts"] == paths
said, log = train("run", "--updates", "3", "--resume")
assert len(log) == 3 and "late starts" in said
said, log = train("seeded", "--updates", "1", "--seeded", "0.25", "--seeds", "poison")
assert "lumps" in said and "seeded won" in said and "late won" in said, said
said, log = train("practice", "--updates", "1", "--practice", "0.25", "--practice-decks", "mix", "--eval-every", "1", "--eval-runs", "4")
assert "37.5% of all" in said, said
with open(f"{tmp}/practice/eval.csv") as f:
    row = next(csv.DictReader(f))
assert row["set"] == "runs" and int(row["runs"]) == 4 and float(row["floor"]) < 16, row
torch.manual_seed(5)
torch.save({"net": RunNet(d=16, layers=1, heads=2).state_dict(), "args": {"d": 16, "layers": 1, "heads": 2}, "update": 7}, f"{tmp}/plain.pt")
torch.save({"net": RunNet(d=16, layers=1, heads=2, fight_outcome=True).state_dict(), "args": {"d": 16, "layers": 1, "heads": 2, "fight_outcome_head": True}, "update": 7}, f"{tmp}/outcome.pt")
for name, flags in (("deck-net", ["--deck-net", f"{tmp}/plain.pt"]), ("fight-net", ["--fight-net", f"{tmp}/plain.pt"]), ("lookup", ["--fight-net", f"{tmp}/outcome.pt", "--lookup", "--lookup-draws", "1"])):
    said, log = train(name, "--updates", "1", *flags)
    assert int(log[0]["seeded_runs"]) > 0, (name, log[0])
shutil.rmtree(tmp)

print("ok")
