"""The deck look-ahead (notes/split-networks-plan.md: the deck network v1, asking the fight network), measured at play
time first: at each card reward, each option (each card offered, and skipping) is judged by test fights the fight
network plays. A test fight is the run as it stands with that card added (its deck, relics and potions, at full HP,
or at its HP now with --hp now) against this act's boss (a scenario, crates/sts-py/src/scenarios.rs, as
human_bosses.py writes them), on --k copies with their own randomness. A fight scores its practice reward
(train_runs.practice_reward: 0.5 + 0.5 x the HP share left for a win, 0.25 for an A20 Act 3 pair lost to the second
boss, else 0). The option with the best average is taken if it beats the deck network's own pick by --margin-se
standard errors of the difference; every other decision is the networks' own, greedy.

    python python/search/deck_lookahead.py python/runs/watcher-runs-1-stance-seeds/update_3600.pt --runs 256
    python python/search/deck_lookahead.py ... --plain          # the same runs without the look-ahead

The runs are the first --runs of train_runs.py's fixed eval runs (seeds from EVAL_SEED, one a game). The checkpoint
plays the fights (the main runs' and the test fights); --deck CKPT makes the choices between fights instead (the
split, core.split), by default the same network. Prints the runs' summary as train_runs.py --eval does, and how
often the look-ahead was asked and changed the pick, by act.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

from core.fight_model import masked, pick_device  # noqa: E402
from core.split import SplitPolicy  # noqa: E402
from core.sts_env import LAYOUT, RunEnv  # noqa: E402
from training.train_runs import EVAL_SEED, SCREEN_AT, load, practice_reward, summarize  # noqa: E402

CARD_REWARD = 1 + LAYOUT["screens"].index("card_reward")
PICK, CONFIRM = LAYOUT["actions"]["pick"][0], LAYOUT["actions"]["confirm"]
BOSS_FLOOR = {1: 16, 2: 33, 3: 50, 4: 55}
LOOKAHEAD_SEED = 900_000


def load_net(path, device):
    net, ck = load(path, device)
    net.load_state_dict(ck["net"])
    net.eval()
    return net


def option_lines(o, character, ascension, k, hp):
    """The test fights for a card reward (`o`: the run's observation, Envs.observe): `k` scenario lines for each
    option, the cards in order and then skipping. Returns the lines and the number of options."""
    deck = [[c["id"], int(c["upgraded"])] for c in o["deck"]]
    act = o["act"]
    base = {"character": character, "ascension": ascension, "act": act, "floor": BOSS_FLOOR.get(act, o["floor"]),
            "room": "boss", "hp": o["max_hp"] if hp == "full" else o["hp"], "max_hp": o["max_hp"], "bosses": [o["boss"]],
            "relics": o["relics"], "potions": [p for p in o["potions"] if p]}
    options = [[c["id"], int(c["upgraded"])] for c in o["screen"]["cards"]] + [None]
    lines = []
    for card in options:
        lines += [json.dumps({**base, "deck": deck + ([card] if card else [])})] * k
    return lines, len(options)


def fight_scores(net, device, amp, lines, character, seed):
    """Each scenario line played once, greedily, by `net`: its practice reward."""
    n = len(lines)
    env = RunEnv(n, seed=seed, practice=1.0, max_steps=4000, character=character)
    skipped = env.envs.set_scenarios(lines, 1.0, True)
    if skipped:
        raise ValueError(f"{skipped} test fights couldn't be set up, e.g. {lines[0][:200]}")
    out = [None] * n
    prac = torch.ones(n, dtype=torch.bool, device=device)
    while any(e is None for e in out):
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
        status = env.step(masked(logits.float(), mask).argmax(-1).cpu().numpy())
        for g in np.flatnonzero(status):
            if out[g] is None:
                out[g] = json.loads(env.last_episode(int(g)))
    return np.array([practice_reward(e) for e in out])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="the fight network (and the deck network unless --deck)")
    p.add_argument("--deck", help="another network for the choices between fights (the split)")
    p.add_argument("--runs", type=int, default=256)
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--character", default="watcher", choices=["ironclad", "silent", "defect", "watcher"])
    p.add_argument("--k", type=int, default=8, help="test fights per option")
    p.add_argument("--hp", choices=["full", "now"], default="full", help="the test fights' HP: max HP, or the run's HP now")
    p.add_argument("--margin-se", type=float, default=2.0)
    p.add_argument("--keep-skips", action="store_true",
                   help="the deck network's skips stand; the look-ahead only chooses which card when it takes one (the first "
                        "256-run test, 2026-10-29: test fights against this act's boss favour taking a card, which skipping "
                        "can't show the worth of, and turned 158 skips into picks: the Heart 3.9%% against 6.6%%)")
    p.add_argument("--plain", action="store_true", help="no look-ahead: the same runs as the networks play them")
    p.add_argument("--device", default="auto")
    args = p.parse_args()
    device = pick_device(args.device)
    amp = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if device.type == "cuda" else (lambda: torch.autocast("cpu", enabled=False))
    fight = load_net(args.checkpoint, device)
    deck = load_net(args.deck, device) if args.deck else fight
    policy = SplitPolicy(fight, deck) if deck is not fight else fight
    R = args.runs
    env = RunEnv(R, seed=EVAL_SEED, ascension=args.ascension, max_steps=8000, character=args.character)
    out = [None] * R
    st = {"asked": 0, "changed": 0, "fights": 0, **{f"asked_act{a}": 0 for a in (1, 2, 3, 4)},
          **{f"changed_act{a}": 0 for a in (1, 2, 3, 4)}, "to_skip": 0, "from_skip": 0}
    calls, t0, t_look = 0, time.time(), 0.0
    while any(e is None for e in out):
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad(), amp():
            logits, _ = policy(ints, floats)
        acts = masked(logits.float(), mask).argmax(-1).cpu().numpy()
        games = [] if args.plain else [g for g in range(R) if out[g] is None and obs["ints"][g, SCREEN_AT] == CARD_REWARD
                                       and not (args.keep_skips and acts[g] == CONFIRM)]
        if games:
            t1 = time.time()
            views, lines, spans = [], [], []
            for g in games:
                o = json.loads(env.envs.observe(g))
                ls, n_opt = option_lines(o, args.character, args.ascension, args.k, args.hp)
                spans.append((len(lines), n_opt))
                lines += ls
                views.append(o)
            scores = fight_scores(fight, device, amp, lines, args.character, LOOKAHEAD_SEED + calls * 4096)
            calls += 1
            st["fights"] += len(lines)
            for g, o, (at, n_opt) in zip(games, views, spans):
                own = acts[g] - PICK if PICK <= acts[g] < PICK + n_opt - 1 else (n_opt - 1 if acts[g] == CONFIRM else None)
                if own is None or (args.keep_skips and own == n_opt - 1):
                    continue  # (Singing Bowl's max HP, or a skip that stands)
                s = scores[at:at + n_opt * args.k].reshape(n_opt, args.k)
                mean, var = s.mean(1), s.var(1, ddof=1)
                best = int(mean[:n_opt - 1].argmax()) if args.keep_skips else int(mean.argmax())
                act = o["act"]
                st["asked"] += 1
                st[f"asked_act{act}"] += 1
                se = np.sqrt((var[best] + var[own]) / args.k)
                if best != own and mean[best] - mean[own] > args.margin_se * se:
                    acts[g] = CONFIRM if best == n_opt - 1 else PICK + best
                    st["changed"] += 1
                    st[f"changed_act{act}"] += 1
                    st["to_skip"] += int(best == n_opt - 1)
                    st["from_skip"] += int(own == n_opt - 1)
            t_look += time.time() - t1
        status = env.step(acts)
        for g in np.flatnonzero(status):
            if out[g] is None:
                out[g] = json.loads(env.last_episode(int(g)))
    r = summarize(out)
    print(f"{args.checkpoint}{f' (choices: {args.deck})' if args.deck else ''}, {'plain' if args.plain else f'deck look-ahead k {args.k}, hp {args.hp}, margin {args.margin_se} SE{", skips kept" if args.keep_skips else ""}'}: "
          f"floor {r['floor']:.1f}, reached A2 {r['reached_act2']:.0%} A3 {r['reached_act3']:.0%} A4 {r['reached_act4']:.0%}, "
          f"won {r['win_rate']:.1%} of {R}, the Heart {r['win_heart']:.1%} ({time.time() - t0:.0f} s, {t_look:.0f} s in the look-ahead)", flush=True)
    if not args.plain:
        by_act = ", ".join(f"Act {a} {st[f'changed_act{a}']}/{st[f'asked_act{a}']}" for a in (1, 2, 3, 4) if st[f"asked_act{a}"])
        print(f"card rewards looked at {st['asked']}, pick changed {st['changed']} ({st['changed'] / max(st['asked'], 1):.1%}; by act {by_act}); "
              f"to a skip {st['to_skip']}, from a skip {st['from_skip']}; {st['fights']:,} test fights", flush=True)


if __name__ == "__main__":
    main()
