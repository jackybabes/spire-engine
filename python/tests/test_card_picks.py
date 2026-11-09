"""Checks for the card picks tests (notes/silent-replan-2026-11-06.md, stage 2; plain asserts, no pytest needed):

    python python/tests/test_card_picks.py

core.split.CardPicker: on positions from random runs, the card reward rows from its floor on get the card network's logits and
every other row, the fights above all, the base policy's. search_eval's CardChoices: picks are that network's top pick or a draw
from its softmax, drawn the same however the runs are shared between games; the floors below --card-from-floor stay the base
policy's; a reward left unopened counts as a skip. analysis/card_picks.py: rules by act and by deck, a forced pick takes the card
if offered under its rule and otherwise leaves the network's choice, a split checkpoint fights with its fight network, a run that dies
in Act 4 counts as having beaten the Act 3 bosses, the human counts leave out runs that look cheated; both scripts' command lines,
end to end, with tiny networks, and the flags that would measure nothing refused."""
import contextlib
import gzip
import io
import json
import os
import sys
import tempfile
import zlib

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from analysis import card_picks
from core.lookup import CARD_REWARD
from core.run_model import RunNet
from core.split import FLOOR_AT, SCREEN_AT, CardPicker, SplitPolicy, fight_rows
from core.sts_env import ACTIONS, LAYOUT, RunEnv
from search import search_eval
from search.search_eval import CARD_BUCKETS, CONFIRM, PICK_AT, REWARDS, CardChoices, CardTally, act3_line, beat_act3, card_bucket, picks_lines, play

CPU = torch.device("cpu")
SILENT_CARDS = ["Neutralize", "Survivor", "Blade Dance", "Dagger Spray", "Dagger Throw", "Quick Slash", "Slice", "Backflip", "Acrobatics",
                "Deflect", "Prepared", "Poisoned Stab", "Deadly Poison", "Dodge and Roll", "Sucker Punch", "Bane", "Flying Knee", "Outmaneuver",
                "Piercing Wail", "Cloak and Dagger", "Sneaky Strike"]


class Moves:
    """A stand-in for a network: logits that depend on the row alone, so a position gets the same move whatever the batch
    (`salt`: another such policy). `screens`: {screen id: slot}, the row's move on those screens."""

    def __init__(self, salt=0, screens=None):
        self.salt, self.screens = salt, screens or {}

    def eval(self):
        return self

    def __call__(self, ints, floats, practice=None):
        rows = []
        for i, f in zip(ints.numpy(), floats.numpy()):
            g = torch.Generator().manual_seed(zlib.crc32(i.tobytes() + f.tobytes()) + self.salt)
            rows.append(torch.randn(ACTIONS, generator=g))
            if int(i[SCREEN_AT]) in self.screens:
                rows[-1][self.screens[int(i[SCREEN_AT])]] = 50.0
        return torch.stack(rows), torch.zeros(len(rows))


def runs(policy, n, batch, sample=None, rule=None, seed=300_000):
    env = RunEnv(batch, seed=seed, ascension=0, max_steps=1500, character="silent")
    out, _ = play(policy, CPU, contextlib.nullcontext, env, lambda g, e: e * batch + g < n, None, 0.0, "test",
                  cards=CardChoices(batch, sample=sample, rule=rule))
    return sorted(out, key=lambda e: e["seed"])


def rows_of(out):
    """The rewards a run looked at."""
    return [row for e in out for row in e["picks"] if row[3] != "unopened"]


def test_card_picker():
    env = RunEnv(32, seed=9, ascension=0, character="silent")
    ints, floats = [], []
    for k in range(400):
        obs = env.observe()
        ints.append(obs["ints"].copy())
        floats.append(obs["floats"].copy())
        for g in np.flatnonzero(obs["ints"][:, SCREEN_AT] == CARD_REWARD):
            assert json.loads(env.envs.observe(int(g)))["floor"] == obs["floats"][g, FLOOR_AT], "the floor's place in the floats"
        env.envs.step_random(500 + k)
    ints, floats = torch.from_numpy(np.concatenate(ints).astype(np.int64)), torch.from_numpy(np.concatenate(floats))
    practice = torch.zeros(len(ints), dtype=torch.bool)
    base, cards = RunNet(d=16, layers=1, heads=2).eval(), RunNet(d=16, layers=1, heads=2).eval()
    # (Neow's card rewards, on floor 0, aren't a floor's: --card-from-floor's default leaves them to the base policy.)
    at_reward = (ints[:, SCREEN_AT] == CARD_REWARD) & (floats[:, FLOOR_AT] >= 1)
    assert at_reward.sum() > 10 and fight_rows(ints).any()
    with torch.no_grad():
        lb, vb = base(ints, floats, practice)
        lc, vc = cards(ints[at_reward], floats[at_reward], practice[at_reward])
        lp, vp = CardPicker(base, cards)(ints, floats, practice)
        assert torch.equal(lp[at_reward], lc) and torch.equal(vp[at_reward], vc)
        assert torch.equal(lp[~at_reward], lb[~at_reward]) and torch.equal(vp[~at_reward], vb[~at_reward]), "the fights and other choices"
        # From a floor on: rows below it stay the base policy's.
        floor = int(floats[at_reward, FLOOR_AT].max())
        picker = CardPicker(base, cards, floor)
        late = at_reward & (floats[:, FLOOR_AT] >= floor)
        assert torch.equal(picker.rows(ints, floats), late) and late.sum() < at_reward.sum()
        lf, _ = picker(ints, floats, practice)
        assert torch.equal(lf[late], cards(ints[late], floats[late], practice[late])[0]) and torch.equal(lf[~late], lb[~late])
        # With a last floor: only the rows between.
        early = CardPicker(base, cards, 1, floor - 1)
        between = at_reward & (floats[:, FLOOR_AT] < floor)
        assert torch.equal(early.rows(ints, floats), between) and between.any()
        le, _ = early(ints, floats, practice)
        assert torch.equal(le[between], cards(ints[between], floats[between], practice[between])[0]) and torch.equal(le[~between], lb[~between])
        # On a split, the fights are the fight network's.
        fight = RunNet(d=16, layers=1, heads=2).eval()
        combat = fight_rows(ints)
        ls, _ = CardPicker(SplitPolicy(fight, base), cards)(ints, floats, practice)
        assert torch.equal(ls[combat], fight(ints[combat], floats[combat], practice[combat])[0])
        assert torch.equal(ls[at_reward], lc)
    print(f"CardPicker: {int(at_reward.sum())} card reward rows of {len(ints)}, its network's; the rest the base policy's")


def test_picks():
    base = Moves()
    alone = runs(base, 16, 16)
    assert len(rows_of(alone)) > 10, "random runs reach card rewards"
    # With itself as the card network, or from a floor no run reaches: the same runs.
    for policy in (CardPicker(base, base), CardPicker(base, Moves(1), 1000)):
        assert [(e["seed"], e["floor"], e["steps"], e["picks"]) for e in runs(policy, 16, 16)] == [(e["seed"], e["floor"], e["steps"], e["picks"]) for e in alone]
    # Its top pick, from its floor on; below it the base policy's.
    pick_first, skip = Moves(screens={CARD_REWARD: PICK_AT}), Moves(1, screens={CARD_REWARD: CONFIRM})
    rows = rows_of(runs(CardPicker(pick_first, skip, 3), 32, 16))
    assert any(r[0] >= 3 for r in rows) and any(r[0] < 3 for r in rows)
    assert all(r[2] is None if r[0] >= 3 else r[2] == r[1][0] for r in rows), rows[:5]
    print(f"card picks: {len(rows)} rewards, skipped from floor 3 on, the base policy's first card below it")


def test_sampling():
    flat = lambda ints, floats, practice=None: (torch.zeros(len(ints), ACTIONS), torch.zeros(len(ints)))  # noqa: E731
    flat.eval = lambda: flat
    base = Moves(screens={CARD_REWARD: PICK_AT})
    picker = CardPicker(base, flat)
    sample = lambda s: (picker, s, lambda g, e, b=8: 300_000 + e * b + g)  # noqa: E731
    one = runs(picker, 32, 8, sample(1))
    rows = rows_of(one)
    slots = [r[1].index(r[2]) if r[2] else -1 for r in rows]
    # A uniform softmax over three cards and the skip (a bowl's slot aside).
    counts = [slots.count(k) for k in (-1, 0, 1, 2)]
    assert len(rows) > 30 and min(counts) > 0.1 * len(rows), counts
    # Reproducible, and the same however the runs are shared out; another seed, other draws.
    assert one == runs(picker, 32, 8, sample(1))
    other = runs(picker, 32, 4, (picker, 1, lambda g, e: 300_000 + e * 4 + g))
    assert [e["picks"] for e in other] == [e["picks"] for e in one], "by run, not by game"
    assert [e["picks"] for e in runs(picker, 32, 8, sample(2))] != [e["picks"] for e in one]
    # The draw follows the logits: the skip three times as likely as each of three cards is half of the picks.
    def uneven(ints, floats, practice=None):
        lg = torch.zeros(len(ints), ACTIONS)
        lg[:, CONFIRM] = np.log(3.0)
        return lg, torch.zeros(len(ints))

    uneven.eval = lambda: uneven
    leaning = CardPicker(base, uneven)
    drawn = [r for r in rows_of(runs(leaning, 64, 8, (leaning, 1, lambda g, e: 300_000 + e * 8 + g))) if len(r[1]) == 3]
    assert len(drawn) > 50 and 0.35 < sum(r[2] is None for r in drawn) / len(drawn) < 0.65, len(drawn)
    # Only the picker's rows draw.
    late = CardPicker(base, flat, 1000)
    rows = rows_of(runs(late, 16, 8, (late, 1, lambda g, e: e * 8 + g)))
    assert all(r[2] == r[1][0] for r in rows) and rows
    print(f"--card-sample: {counts} skips and first, second, third cards; the same draws at 4 and 8 games")


def test_unopened():
    leave = Moves(screens={REWARDS: CONFIRM})
    unopened = [row for e in runs(leave, 16, 16) for row in e["picks"] if row[3] == "unopened"]
    assert unopened and all(r[2] is None and len(r[1]) >= 3 for r in unopened)
    t = CardTally()
    for r in unopened:
        t.add(r[0], r[1], r[2])
    assert t.skip_rate() == 1.0
    print(f"{len(unopened)} card rewards left unopened count as skips")


def test_rules():
    for name in card_picks.SHIV_MAKERS:
        assert name in card_picks.CARD_NAMES, name
    assert card_picks.parse_cell("own") == ("own", [])
    poison, late = ["Noxious Fumes", "Bouncing Flask", "Corpse Explosion"], ["Backflip", "Acrobatics", "Footwork"]
    assert card_picks.parse_cell("a") == ("a", [(["act1"], poison)])
    assert card_picks.parse_cell("b") == ("b", [(["fewshiv"], ["Catalyst"])])
    assert card_picks.parse_cell("c") == ("c", [(["act2+"], late)])
    assert card_picks.parse_cell("d") == ("d", [(["act1"], poison), (["fewshiv"], ["Catalyst"]), (["act2+"], late)])
    assert card_picks.parse_cell("x=act2+&noshiv:Well-Laid Plans, Footwork") == ("x", [(["act2+", "noshiv"], ["Well-Laid Plans", "Footwork"])])
    for bad in ("nope", "x=act1:Noxious Fume", "x=act9:Catalyst", "x=Catalyst", "x=act1:", "x=act1:Cloak And Dagger"):
        try:
            card_picks.parse_cell(bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad} parsed")

    def view(act, offered, deck=("Strike", "Defend")):
        return {"act": act, "floor": 1, "deck": [{"name": c} for c in deck], "screen": {"cards": [{"name": c} for c in offered]}}

    pick = LAYOUT["actions"]["pick"][0]
    _, a = card_picks.parse_cell("a")
    assert card_picks.forced_slot(a, view(1, ["Slice", "Bouncing Flask", "Noxious Fumes+"])) == pick + 2, "the rule's order, an upgraded card too"
    assert card_picks.forced_slot(a, view(2, ["Slice", "Noxious Fumes"])) is None, "Act 1 only"
    assert card_picks.forced_slot(a, view(1, ["Slice", "Bane"])) is None, "not offered: the network's choice"
    _, c = card_picks.parse_cell("c")
    assert card_picks.forced_slot(c, view(1, ["Backflip"])) is None and card_picks.forced_slot(c, view(2, ["Slice", "Backflip"])) == pick + 1
    assert card_picks.forced_slot(c, view(3, ["Footwork"])) == pick and card_picks.forced_slot(c, view(4, ["Footwork"])) == pick
    _, b = card_picks.parse_cell("b")
    assert card_picks.forced_slot(b, view(1, ["Catalyst"])) == pick and card_picks.forced_slot(b, view(3, ["Catalyst"], ("Bane",))) == pick
    _, none = card_picks.parse_cell("x=noshiv:Catalyst")
    for maker in card_picks.SHIV_MAKERS:
        assert card_picks.forced_slot(none, view(2, ["Catalyst"], ("Strike", maker))) is None, maker
        assert card_picks.forced_slot(none, view(2, ["Catalyst"], ("Strike", maker + "+"))) is None, maker
        assert card_picks.forced_slot(b, view(2, ["Catalyst"], ("Strike", maker))) == pick, "one Shiv maker is still a non-Shiv deck (shiv0-1)"
        assert card_picks.forced_slot(b, view(2, ["Catalyst"], ("Strike", maker, maker + "+"))) is None, maker
    _, d = card_picks.parse_cell("d")
    assert card_picks.forced_slot(d, view(1, ["Noxious Fumes", "Catalyst"])) == pick, "a cell's rules in order"
    assert card_picks.forced_slot(d, view(2, ["Noxious Fumes", "Catalyst", "Backflip"])) == pick + 1
    assert card_picks.forced_slot(d, view(2, ["Noxious Fumes", "Catalyst", "Backflip"], ("Accuracy", "Blade Dance"))) == pick + 2

    # In real runs: the base policy always skips, so every forced pick shows, and only where its rule holds.
    skip = Moves(screens={CARD_REWARD: CONFIRM})
    _, act1 = card_picks.parse_cell("x=act1:" + ",".join(SILENT_CARDS))
    rows = rows_of(runs(skip, 24, 12, rule=lambda v, _: card_picks.forced_slot(act1, v)))
    forced = [r for r in rows if r[3] == "forced"]
    assert forced and all(r[2] in SILENT_CARDS for r in forced)
    assert all(r[2] is None and not set(r[1]) & set(SILENT_CARDS) for r in rows if r[3] == "own"), "offered under the rule, taken"
    # The first in the rule's order.
    assert all(r[2] == min(set(r[1]) & set(SILENT_CARDS), key=SILENT_CARDS.index) for r in forced)
    _, later = card_picks.parse_cell("x=act2+:" + ",".join(SILENT_CARDS))
    rows = rows_of(runs(skip, 24, 12, rule=lambda v, _: card_picks.forced_slot(later, v)))
    assert rows and all(r[3] == "own" and r[2] is None for r in rows), "these runs stay in Act 1"
    # The starting deck has no Shiv makers, read from the run's own observation.
    env = RunEnv(1, seed=5, ascension=0, character="silent")
    assert card_picks.CONDITIONS["noshiv"](json.loads(env.envs.observe(0)))
    print(f"rules: {len(forced)} picks forced in Act 1 of random runs, none from Act 2 in runs that stay in Act 1")


def test_buckets():
    for floor, b in ((0, None), (1, 0), (8, 0), (9, 1), (16, 1), (18, 2), (33, 2), (35, 3), (51, 3), (53, 3)):
        assert card_bucket(floor) == b, floor
    t = CardTally()
    t.add(0, ["Slice", "Bane", "Dash"], "Slice")
    t.add(3, ["Slice", "Bane", "Dash"], "Slice")
    t.add(20, ["Slice", "Backflip", "Dash"], None)
    assert t.rewards[None] == 3 and t.rewards[0] == 1 and t.rewards[2] == 1 and t.skips[2] == 1 and t.skip_rate() == 1 / 3
    assert t.offered[None]["Slice"] == 3 and t.taken[0]["Slice"] == 1 and t.offered[2]["Backflip"] == 1
    lines = picks_lines([{"picks": [[3, ["Slice", "Bane"], "Slice", "own"], [20, ["Bane"], None, "unopened"]]}])
    assert len(lines) == 1 + len(CARD_BUCKETS) and "Slice 100% of 1" in lines[1] and "skipped 100%" in lines[3]


def test_act3_counts():
    """A run with the keys that beat the Act 3 bosses and died in Act 4 isn't `won`, and counts as having beaten them."""
    def run(seed, act, won):
        return {"seed": seed, "act": act, "won": won, "floor": 40, "steps": 10, "picks": []}

    own = [run(1, 3, False), run(2, 3, False), run(3, 2, False), run(4, 3, True)]
    res = [run(1, 4, False), run(2, 3, True), run(3, 3, False), run(4, 4, True)]
    assert [beat_act3(e) for e in res] == [True, True, False, True] and not beat_act3(own[0])
    line = card_picks.cell_line("x", res, own)
    assert "won  50.0%" in line and "the Heart 25.0%" in line and "Act 3 100%" in line and "Act 4  50%" in line, line
    assert "beaten 75% of those" in line, line
    assert "against own: Act 3 +25.0%" in line and "its bosses +50.0%" in line, line
    assert act3_line(res) == "  Act 3 reached 100.0%, its bosses beaten 75.0% of runs, 75% of arrivals", act3_line(res)


def tiny_checkpoint(path, **args):
    torch.save({"net": RunNet(d=16, layers=1, heads=2).state_dict(), "update": 1, "args": dict(d=16, layers=1, heads=2, character="silent", **args)}, path)


def script(main, *argv):
    """A script's main with these arguments; its output."""
    old, out = sys.argv, io.StringIO()
    sys.argv = ["script", *argv]
    try:
        with contextlib.redirect_stdout(out):
            main()
    finally:
        sys.argv = old
    return out.getvalue()


def test_scripts():
    with tempfile.TemporaryDirectory() as d:
        plain, fight, split, cards = (os.path.join(d, f"{n}.pt") for n in ("plain", "fight", "split", "cards"))
        tiny_checkpoint(plain)
        tiny_checkpoint(fight)
        tiny_checkpoint(cards)
        tiny_checkpoint(split, fight_net=fight, lookup=False)
        # A split checkpoint fights with its fight network.
        policy, lookup, _ = card_picks.load_policy(split, CPU, contextlib.nullcontext)
        assert isinstance(policy, SplitPolicy) and lookup is None
        saved = torch.load(fight, weights_only=False)["net"]
        assert all(torch.equal(v, saved[k]) for k, v in policy.fight.state_dict().items()), "the fight network's weights, not the checkpoint's own"
        assert isinstance(card_picks.load_policy(plain, CPU, contextlib.nullcontext)[0], RunNet)
        small = ["--runs", "6", "--batch", "3", "--ascension", "0", "--device", "cpu"]

        out = script(card_picks.main, split, "--cells", "own", "a", "x=act2+&noshiv:Catalyst", "--seed-offset", "1024", "--buckets", "--dump", os.path.join(d, "cell"), *small)
        lines = [line for line in out.splitlines() if " won " in line and "Act 2" in line]
        assert len(lines) == 3 and "against own" not in lines[0] and "against own: Act 3" in lines[1] and "skipped" in lines[0], out
        assert "floors 9-16" in out
        dumped = [json.loads(line) for line in open(os.path.join(d, "cell.a.jsonl"))]
        assert sorted(e["seed"] for e in dumped) == list(range(301_024, 301_030)) and "picks" in dumped[0]

        # The same runs through search_eval, with a card network and without: --no-search, --dump, a fighter on a split.
        prefix = os.path.join(d, "dump")
        out = script(search_eval.main, plain, "--character", "silent", "--set", "runs", "--no-search", "--runs", "6", "--batch", "3", "--ascension", "0", "--device", "cpu",
                     "--seed-offset", "1024", "--card-net", cards, "--card-from-floor", "2", "--card-sample", "--dump", prefix)
        assert "card picks: " in out and "floors 1-8" in out and "runs A0 greedy: won" in out and "its bosses beaten" in out, out
        dumped = [json.loads(line) for line in open(prefix + ".0.jsonl")]
        assert len(dumped) == 6 and all("picks" in e for e in dumped)
        out = script(search_eval.main, split, "--character", "silent", "--set", "runs", "--no-search", "--runs", "3", "--batch", "3", "--ascension", "0", "--device", "cpu",
                     "--fighter", fight, "--card-net", cards)
        assert "floors 1-8" in out

        # The counts against a human record's, from the dump (no playing), and from a run of its own.
        human = os.path.join(d, "human")
        os.mkdir(human)
        with gzip.open(os.path.join(human, "a.jsonl.gz"), "wt") as f:
            for rec in ({"kind": "card", "won": True, "heart": True, "ascension": 20, "floor": 3, "offered": ["Blade Dance", "Cloak And Dagger+1"], "taken": "Cloak And Dagger+1"},
                        {"kind": "card", "won": True, "heart": False, "ascension": 20, "floor": 20, "offered": ["Blade Dance"], "taken": "SKIP"},
                        {"kind": "card", "won": False, "heart": False, "ascension": 20, "floor": 20, "offered": ["Blade Dance"], "taken": "Blade Dance"},
                        {"kind": "rest", "won": True, "ascension": 20, "floor": 20}):
                f.write(json.dumps(rec) + "\n")
        t = card_picks.human_picks(human, 20, 10)
        assert t.rewards[None] == 2 and t.skips[2] == 1 and t.taken[0]["Cloak and Dagger"] == 1 and t.offered[0]["Blade Dance"] == 1
        assert card_picks.human_picks(human, 20, 10, heart=True).rewards[None] == 1
        # Runs that look cheated are out, their earlier records too; a winner's late gold is no sign.
        dirty = os.path.join(d, "dirty")
        os.mkdir(dirty)
        with gzip.open(os.path.join(dirty, "a.jsonl.gz"), "wt") as f:
            card = {"kind": "card", "won": True, "ascension": 20, "offered": ["Blade Dance"], "taken": "SKIP", "relics": ["a"], "gold": 100}
            for rec in (dict(card, play_id="relics", floor=5), {"kind": "rest", "play_id": "relics", "floor": 40, "relics": ["a"] * 30, "gold": 0},
                        dict(card, play_id="gold", floor=2, gold=1001), dict(card, play_id="gold", floor=9),
                        dict(card, play_id="clean", floor=30, gold=5000), dict(card, play_id="clean", floor=3, gold=1000)):
                f.write(json.dumps(rec) + "\n")
        t = card_picks.human_picks(dirty, 20, 10)
        assert t.rewards[None] == 2 and t.left_out == 2 and t.rewards[2] == 1 and t.rewards[0] == 1, (t.rewards, t.left_out)
        for args in (["--from-dump", prefix], ["--from-dump", os.path.join(d, "cell.a")], [plain, *small]):
            out = script(card_picks.main, *args, "--human", human, "--human-min-asc", "20", "--buckets", "--min-offered", "1")
            assert "human winners A20+: 2 card rewards, skipped 50%" in out and "Act 3:" in out, out
        # A --cells --dump's prefix alone would mix its cells: an error, as is a flag that would do nothing or a lookup nothing fills.
        lookup = os.path.join(d, "lookup.pt")
        tiny_checkpoint(lookup, lookup=True)
        eval_args = [plain, "--character", "silent", "--set", "runs", "--no-search", "--runs", "3", "--batch", "3", "--ascension", "0", "--device", "cpu"]
        for main, bad in ((card_picks.main, ["--from-dump", os.path.join(d, "cell"), "--human", human]),
                          (search_eval.main, [*eval_args, "--card-from-floor", "17"]),
                          (search_eval.main, [*eval_args, "--card-sample"]),
                          (search_eval.main, [*eval_args, "--fighter", fight]),
                          (search_eval.main, [*eval_args, "--card-net", lookup])):
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    script(main, *bad)
            except SystemExit as e:
                assert e.code == 2, bad
                continue
            raise AssertionError(f"{bad} ran")
        # The blunt test still plays its four variants.
        out = script(card_picks.main, plain, "--force", "--take", "Backflip", "--avoid", "Slice", *small)
        assert all(n in out for n in ("network's own", "take and avoid"))
    print("the scripts' command lines: search_eval --card-net --dump, card_picks --cells --buckets --from-dump --force")


if __name__ == "__main__":
    test_card_picker()
    test_picks()
    test_sampling()
    test_unopened()
    test_rules()
    test_buckets()
    test_act3_counts()
    test_scripts()
    print("ok")
