"""Writes <run>/live.html from <run>/log.csv, <run>/eval.csv and (runs) <run>/choices.csv: a chart page that reloads itself
every 15 seconds, so training can be watched in a browser.

    .venv/bin/python python/training/live_chart.py python/runs/<name>     # rebuild the page by hand

The trainers call `write(run_dir)` after every update and `serve(RUNS, port)` at the start,
so the page is at http://<machine>:<port>/<name>/live.html from another computer too. The page
is self-contained (data inlined), so it also opens straight from disk. The server renders each
page from the csv files when it is asked for, so a server of newer code (`live_chart.py <run
folder> <port>`, which serves the runs folder) shows every run with its pages, the ones training
included. A chart, series or tile with no data is left out, so the run page's list covers every
kind of run (2026-10-24: the Ironclad's archetype, block and seeded-run charts went).

The server's home page, http://<machine>:<port>/, is the run training now (the one whose files
changed last, among those changed in the last 10 minutes), and every page has a bar linking to the
other runs training and to /runs, the list of all runs (2026-10-24).
"""
import csv
import functools
import html
import json
import os
import socket
import sys
import threading
import time
import urllib.parse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
LOSSES = ["policy_loss", "value_loss", "entropy", "approx_kl", "lr", "explained_var"]


def _baselines(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def _common_charts():
    """Speed and the PPO health charts, shared by fights and runs."""
    one = lambda key, label, fmt="num", **kw: {"title": label, "source": "log", "series": [{"key": key, "label": label}], "fmt": fmt, **kw}
    health = lambda key, label, fmt="num", **kw: one(key, label, fmt, group="Training health", **kw)
    return [
        one("steps_per_s", "Steps per second", "int", min=0, group="Speed and memory"),
        health("policy_loss", "Policy loss"),
        health("value_loss", "Value loss", min=0),
        health("entropy", "Entropy", min=0),
        health("approx_kl", "Approximate KL", min=0),
        health("lr", "Learning rate", "exp", min=0),
        health("explained_var", "Explained variance of the value", max=1),
    ]


def fight_spec():
    b = _baselines("baselines.json")
    sc, sa = b.get("random_decks_all_encounters/scripted"), b.get("starter_deck_act1/scripted")
    ref = lambda r, key, label: [{"y": r[key], "label": label}] if r else []
    acts = [{"key": f"win_act{k}", "label": f"Act {k}"} for k in (1, 2, 3, 4)]
    return {
        "title": "Fight training", "counts": [{"key": "fights", "label": "fights"}],
        "tiles": [
            {"label": "Training win rate (last 2,000)", "source": "log", "key": "win_rate", "fmt": "pct"},
            {"label": "Eval win rate, random decks", "source": "eval", "key": "random_decks_all_encounters:win_rate", "fmt": "pct"},
            {"label": "Scripted baseline", "source": "const", "value": sc and sc["win_rate"], "fmt": "pct"},
            {"label": "HP lost per fight", "source": "log", "key": "hp_lost", "fmt": "num"},
            {"label": "Steps / s", "source": "log", "key": "steps_per_s", "fmt": "int"},
        ],
        "charts": [
            {"title": "Win rate, training fights (last 2,000 fights)", "source": "log",
             "series": [{"key": "win_rate", "label": "All"}] + acts, "refs": ref(sc, "win_rate", "scripted, all"), "fmt": "pct", "min": 0, "max": 1},
            {"title": "Eval win rate, fixed fights, greedy", "source": "eval",
             "series": [{"key": "random_decks_all_encounters:win_rate", "label": "Random decks, all encounters"},
                        {"key": "starter_deck_act1:win_rate", "label": "Starter deck, Act 1"}],
             "refs": ref(sc, "win_rate", "scripted, random decks") + ref(sa, "win_rate", "scripted, starter"), "fmt": "pct", "min": 0, "max": 1},
            {"title": "HP lost per fight (of 80; a loss counts all of it)", "source": "log",
             "series": [{"key": "hp_lost", "label": "HP lost"}], "refs": ref(sc, "hp_lost", "scripted"), "fmt": "num", "min": 0},
        ] + _common_charts(),
        "table": ["update", "steps", "win_rate", "win_act1", "win_act2", "win_act3", "win_act4", "hp_lost", "steps_per_s"] + LOSSES,
        "formats": {**{k: "pct" for k in ("win_rate", "win_act1", "win_act2", "win_act3", "win_act4")}, "steps_per_s": "int", "lr": "exp"},
    }


# train_runs.py's habit rewards: a run with none of them has no habit charts.
HABITS = ("deck_weight", "rest_penalty", "smith_bonus", "strike_keep_cost", "strike_weight", "curse_keep_cost", "curse_weight",
          "strike_upgrade_penalty", "removal_skip_penalty", "hallway_potion_penalty", "recall_penalty", "elite_bonus",
          "synergy_bonus", "energy_penalty", "forced_purge", "forced_purge_curse", "block_weight", "practice_potion_bonus")
BANDS = ((0, 5), (6, 10), (11, 15), (16, 20))


def run_spec(args=None):
    """train_runs.py's page, for every character, from the run's args.json (`args`). The page
    leaves out the charts, series and tiles with no data (practice-only runs have no runs, runs
    without --scenarios no human fights), so the list here can cover every kind of run; the args
    leave out the rest a run doesn't use (2026-10-26): the habit charts without habit rewards, the
    wins by ascension for a run of one ascension band. With --heart-reward a win is beating the
    Heart; `win_rate` (which counts a win at the Act 3 boss without the keys too) is shown beside it
    as that."""
    args = args or {}
    heart = bool(args.get("heart_reward"))
    habits = any(args.get(k) for k in HABITS)
    lo, hi = args.get("ascension_range") or (args.get("ascension", 0), args.get("ascension", 0))
    band = lambda a: next(i for i, (b0, b1) in enumerate(BANDS) if b0 <= a <= b1)  # noqa: E731
    asc_bands = band(lo) != band(hi) or bool(args.get("ascension_low_share"))
    pct = {"fmt": "pct", "min": 0, "max": 1}
    acts = lambda pre: [{"key": f"{pre}act{a}", "label": f"Act {a}"} for a in (1, 2, 3, 4)]
    kinds = lambda pre, all_key: [{"key": all_key, "label": "all"}] + [{"key": f"{pre}{k}", "label": k} for k in ("grown", "engine", "random")]
    g = lambda group, charts: [{**c, "group": group} for c in charts]
    won = "win_heart" if heart else "win_rate"
    won_label = "won (beat the Heart)" if heart else "won"
    any_win = [{"key": "win_rate", "label": "won at the Act 3 boss without the keys, or the Heart"}] if heart else []
    evals = [("runs", "first eval ascension")] + [(f"runs_a{a}", f"A{a}") for a in (5, 10, 15, 20)]
    spec = {
        "title": "Run training", "counts": [{"key": "runs", "label": "runs"}],
        "tiles": [
            {"label": "Runs that beat the Heart (last 500)" if heart else "Runs won (last 500)", "source": "log", "key": won, "fmt": "pct"},
            {"label": "Mean floor (last 500 runs)", "source": "log", "key": "floor", "fmt": "floor"},
            {"label": "Eval mean floor", "source": "eval", "key": "runs:floor", "fmt": "floor"},
            {"label": "Practice fights won (last 1,000)", "source": "log", "key": "practice_win", "fmt": "pct"},
            {"label": "Eval: practice fights won", "source": "eval", "key": "practice:win_rate", "fmt": "pct"},
            {"label": "Eval: boss fights won", "source": "eval", "key": "practice_boss:win_rate", "fmt": "pct"},
            {"label": "Eval: human boss fights won", "source": "eval", "key": "bosses:win_rate", "fmt": "pct"},
            {"label": "HP lost per practice fight won (last 1,000)", "source": "log", "key": "practice_hp_lost", "fmt": "num"},
            {"label": "Steps / s", "source": "log", "key": "steps_per_s", "fmt": "int"},
        ],
        "charts": g("Runs", [
            {"title": "Floor reached, training runs (last 500)", "source": "log",
             "series": [{"key": "floor", "label": "mean floor"}], "fmt": "floor", "min": 0},
            {"title": "How far training runs get (last 500)", "source": "log",
             "series": [{"key": "reached_act2", "label": "Act 2"}, {"key": "reached_act3", "label": "Act 3"},
                        {"key": "reached_act4", "label": "Act 4"}, {"key": won, "label": won_label}] + any_win
                       + ([] if heart else [{"key": "win_heart", "label": "beat the Heart"}]), **pct},
            *([] if not asc_bands else [{"title": "Training runs won by ascension (last 500)", "source": "log",
             "series": [{"key": f"win_a{b0}_{b1}", "label": f"A{b0}-{b1}"} for b0, b1 in BANDS], **pct}]),
            {"title": "Keys held when training runs end (of 3; all three open Act 4)", "source": "log",
             "series": [{"key": "keys", "label": "keys"}], "fmt": "num", "min": 0},
            {"title": "Decisions per run", "source": "log", "series": [{"key": "run_steps", "label": "decisions"}], "fmt": "int", "min": 0},
            {"title": "Eval: floor reached, fixed runs, greedy", "source": "eval",
             "series": [{"key": "runs:floor", "label": "mean floor"}], "fmt": "floor", "min": 0},
            {"title": "Eval: how far fixed runs get, greedy", "source": "eval",
             "series": [{"key": "runs:reached_act2", "label": "Act 2"}, {"key": "runs:reached_act3", "label": "Act 3"},
                        {"key": "runs:reached_act4", "label": "Act 4"}, {"key": f"runs:{won}", "label": won_label}]
                       + [{"key": f"runs:{x['key']}", "label": x["label"]} for x in any_win], **pct},
            {"title": "Eval: fixed runs that beat the Heart, by ascension, greedy" if heart else "Eval: fixed runs won by ascension, greedy",
             "source": "eval", "series": [{"key": f"{s}:{won}", "label": label} for s, label in evals], **pct},
        ]) + g("Practice fights (--practice)", [
            {"title": "Practice fights won by deck kind, training (last 1,000)", "source": "log",
             "series": kinds("practice_win_", "practice_win"), **pct},
            {"title": "Practice fights won by act, training (last 1,000)", "source": "log", "series": acts("practice_win_"), **pct},
            {"title": "Eval: fixed practice fights won by deck kind, greedy", "source": "eval",
             "series": kinds("practice:win_", "practice:win_rate"), **pct},
            {"title": "Eval: fixed practice fights won by act, greedy", "source": "eval", "series": acts("practice:win_"), **pct},
            {"title": "Practice fights won by room, training (last 1,000)", "source": "log",
             "series": [{"key": f"practice_win_room_{r}", "label": r} for r in ("hallway", "elite", "boss")], **pct},
            {"title": "Eval: fixed practice fights won by room, greedy", "source": "eval",
             "series": [{"key": f"practice:win_room_{r}", "label": r} for r in ("hallway", "elite", "boss")], **pct},
            {"title": "Eval: fixed boss fights won by act, greedy (each act's boss, the Heart in Act 4; A0-20)", "source": "eval",
             "series": [{"key": "practice_boss:win_rate", "label": "all"}] + acts("practice_boss:win_"), **pct},
            {"title": "Eval: fixed boss fights won by deck kind, greedy", "source": "eval",
             "series": kinds("practice_boss:win_", "practice_boss:win_rate"), **pct},
            {"title": "HP lost per practice fight won, by act, training (last 1,000; HP at the start less HP after)", "source": "log",
             "series": [{"key": "practice_hp_lost", "label": "all"}] + acts("practice_hp_lost_"), "fmt": "num", "min": 0},
            {"title": "Eval: HP lost per fixed practice fight won, by act, greedy", "source": "eval",
             "series": [{"key": "practice:hp_lost", "label": "all"}] + acts("practice:hp_lost_"), "fmt": "num", "min": 0},
        ]) + g("Human fights (--scenarios: decks, relics, potions and HP from winning human runs)", [
            {"title": "Human boss fights won, training (last 1,000 practice fights)", "source": "log",
             "series": [{"key": "practice_win_boss1", "label": "Act 1 boss"}, {"key": "practice_win_boss2", "label": "Act 2 boss"},
                        {"key": "practice_win_boss3", "label": "Act 3, both bosses"}, {"key": "practice_win_boss4", "label": "the Heart"}], **pct},
            {"title": "Human elite fights won, training (last 1,000 practice fights)", "source": "log",
             "series": [{"key": f"practice_win_elite{a}", "label": f"Act {a}"} for a in (1, 2, 3)], **pct},
            {"title": "Eval: held-out human boss fights won, greedy", "source": "eval",
             "series": [{"key": "bosses:win_boss1", "label": "Act 1 boss"}, {"key": "bosses:win_boss2", "label": "Act 2 boss"},
                        {"key": "bosses:win_boss3", "label": "Act 3, both bosses"}, {"key": "bosses:first_boss", "label": "Act 3, first boss"},
                        {"key": "bosses:win_boss4", "label": "the Heart"}], **pct},
            {"title": "Eval: held-out human elite fights won, greedy", "source": "eval",
             "series": [{"key": f"bosses:win_elite{a}", "label": f"Act {a}"} for a in (1, 2, 3)], **pct},
        ]) + ([] if not habits else g("Habits in training runs, per 1,000 decisions (--deck-weight, --rest-penalty, --recall-penalty, --elite-bonus)", [
            {"title": "Deck changes", "source": "log",
             "series": [{"key": "cards_added", "label": "cards added"}, {"key": "cards_removed", "label": "cards removed"},
                        {"key": "removals_skipped", "label": "shops left with a removal unused"}], "fmt": "num", "min": 0},
            {"title": "Rest sites", "source": "log",
             "series": [{"key": "rests", "label": "rests"}, {"key": "rests_high", "label": "rests at 80%+ HP"},
                        {"key": "smiths", "label": "upgrades"}], "fmt": "num", "min": 0},
            {"title": "Elites, potions and keys", "source": "log",
             "series": [{"key": "elites_won", "label": "elites won"}, {"key": "hallway_potions", "label": "potions in hallway fights"},
                        {"key": "recalls_wasted", "label": "Recalls without Sapphire"}], "fmt": "num", "min": 0},
        ])) + g("Choices between fights, eval runs (choice_stats.py)", [
            {"title": "Turns ended with energy left and a card it could play", "source": "choices",
             "series": [{"key": "end_turn_energy", "label": "share of turns"}], "fmt": "pct", "min": 0},
            {"title": "Potions per run", "source": "choices",
             "series": [{"key": "potion_drinks", "label": "drunk"}, {"key": "potion_discards", "label": "discarded"},
                        {"key": "potion_rewards_left", "label": "potion rewards left"}], "fmt": "num", "min": 0},
            {"title": "Rest sites, share of visits", "source": "choices",
             "series": [{"key": "rest", "label": "rest"}, {"key": "smith", "label": "upgrade"},
                        {"key": "smith_hp85", "label": "upgrade at 85%+ HP"}, {"key": "smith_hp70", "label": "upgrade at 70-85% HP"},
                        {"key": "smith_hp50", "label": "upgrade at 50-70% HP"}], **pct},
            {"title": "Card removal", "source": "choices",
             "series": [{"key": "removal_at_shop", "label": "shop visits with a removal"},
                        {"key": "removed_curse", "label": "removed: curse"}, {"key": "removed_strike", "label": "removed: Strike"},
                        {"key": "removed_defend", "label": "removed: Defend"}], **pct},
            {"title": "Cards removed per run (shops, events, Neow...)", "source": "choices",
             "series": [{"key": "removals_per_run", "label": "removals"}], "fmt": "num", "min": 0},
            {"title": "Deck at the end of the run", "source": "choices",
             "series": [{"key": "deck_size", "label": "cards"}, {"key": "deck_upgraded", "label": "upgraded"},
                        {"key": "deck_strikes", "label": "Strikes"}, {"key": "deck_defends", "label": "Defends"},
                        {"key": "deck_curses", "label": "curses"}], "fmt": "num", "min": 0},
            {"title": "Card rewards skipped, elites taken when offered", "source": "choices",
             "series": [{"key": "card_skip", "label": "card rewards skipped"}, {"key": "elite_taken", "label": "elites taken"}], **pct},
        ]) + g("Imitation", [
            {"title": "Human imitation (--human-data): agreement with human picks, training batches", "source": "log",
             "series": [{"key": "human_acc", "label": "agreement"}, {"key": "human_acc_card", "label": "card rewards"}], **pct},
        ]) + _common_charts()[:1] + g("Speed and memory", [
            {"title": "Update time, seconds (with --overlap the rollout plays beside the training)", "source": "log",
             "series": [{"key": "update_s", "label": "whole update"}, {"key": "ppo_s", "label": "training (PPO)"},
                        {"key": "prox_s", "label": "proximal pass"}, {"key": "wait_s", "label": "waiting for the rollout"},
                        {"key": "rollout_s", "label": "rollout"}],
             "fmt": "num", "min": 0},
            {"title": "GPU memory, GB (allocated, and held by PyTorch's cache)", "source": "log",
             "series": [{"key": "gpu_alloc_gb", "label": "trainer"}, {"key": "gpu_reserved_gb", "label": "trainer held"},
                        {"key": "worker_gpu_alloc_gb", "label": "rollout worker"}, {"key": "worker_gpu_reserved_gb", "label": "worker held"}],
             "fmt": "num", "min": 0},
        ]) + _common_charts()[1:],
        "notes": [{"label": "Most removed (last eval)", "source": "choices", "key": "top_removed"},
                  {"label": "Most upgraded at rest sites and events (last eval)", "source": "choices", "key": "top_upgraded"}],
        "table": ["update", "steps", "floor", "reached_act2", "reached_act3", "reached_act4", "win_rate", "run_steps", "practice_win",
                  "practice_win_grown", "practice_win_engine", "practice_win_random", "practice_hp_lost", "steps_per_s"] + LOSSES + ["epochs", "stopped_at"],
        "formats": {**{k: "pct" for k in ("win_rate", "reached_act2", "reached_act3", "reached_act4", "practice_win", "practice_win_grown",
                                           "practice_win_engine", "practice_win_random")},
                    "floor": "floor", "run_steps": "int", "steps_per_s": "int", "lr": "exp", "stopped_at": "pct", "epochs": "int"},
    }
    # Training charts of features the run doesn't use (its args say so; a run without args.json
    # keeps them, and the page drops them for want of data).
    off = (("practice fights", "human fights") if args and not args.get("practice") else ()) + (("imitation",) if args and not args.get("human_data") else ())
    spec["charts"] = [c for c in spec["charts"] if not (c["source"] == "log" and c.get("group", "").lower().startswith(off))]
    return spec


def pretrain_spec():
    """pretrain_human.py: supervised training on human choices (eval rows have set "pre")."""
    kinds = [("card", "card rewards"), ("rest", "rest sites"), ("grid", "card upgraded / removed"), ("boss", "boss relics"),
             ("purge", "shop removals")]
    return {
        "title": "Human pretraining", "counts": [{"key": "samples", "label": "samples"}],
        "tiles": [
            {"label": "Agreement with humans (held out)", "source": "eval", "key": "pre:val_acc", "fmt": "pct"},
            {"label": "Eval win rate", "source": "eval", "key": "pre:win_rate", "fmt": "pct"},
            {"label": "Eval mean floor", "source": "eval", "key": "pre:floor", "fmt": "floor"},
            {"label": "Agreement, training batches", "source": "log", "key": "human_acc", "fmt": "pct"},
            {"label": "Passes over the data", "source": "log", "key": "passes", "fmt": "num"},
        ],
        "charts": [
            {"title": "Agreement with held-out human choices", "source": "eval",
             "series": [{"key": "pre:val_acc", "label": "all"}] + [{"key": f"pre:val_acc_{k}", "label": l} for k, l in kinds],
             "fmt": "pct", "min": 0, "max": 1},
            {"title": "Eval: how far fixed runs get, greedy", "source": "eval",
             "series": [{"key": "pre:reached_act2", "label": "Act 2"}, {"key": "pre:reached_act3", "label": "Act 3"},
                        {"key": "pre:win_rate", "label": "won"}], "fmt": "pct", "min": 0, "max": 1},
            {"title": "Agreement and loss, training batches", "source": "log",
             "series": [{"key": "human_acc", "label": "agreement"}], "fmt": "pct", "min": 0, "max": 1},
            {"title": "Human loss (cross-entropy)", "source": "log", "series": [{"key": "human_loss", "label": "loss"}], "fmt": "num", "min": 0},
            {"title": "Drift from the starting network on screens kept as they were", "source": "log",
             "series": [{"key": "keep_kl", "label": "policy KL"}, {"key": "keep_value", "label": "value error"}], "fmt": "num", "min": 0},
            {"title": "Eval: rest sites, share of visits", "source": "choices",
             "series": [{"key": "rest", "label": "rest"}, {"key": "smith", "label": "upgrade"}], "fmt": "pct", "min": 0, "max": 1},
            {"title": "Eval: card rewards skipped, elites taken when offered", "source": "choices",
             "series": [{"key": "card_skip", "label": "card rewards skipped"}, {"key": "elite_taken", "label": "elites taken"}],
             "fmt": "pct", "min": 0, "max": 1},
            {"title": "Eval: deck at the end of the run", "source": "choices",
             "series": [{"key": "deck_size", "label": "cards"}, {"key": "deck_upgraded", "label": "upgraded"},
                        {"key": "deck_curses", "label": "curses"}], "fmt": "num", "min": 0},
            {"title": "Learning rate", "source": "log", "series": [{"key": "lr", "label": "lr"}], "fmt": "exp", "min": 0},
        ],
        "notes": [{"label": "Most removed (last eval)", "source": "choices", "key": "top_removed"},
                  {"label": "Most upgraded (last eval)", "source": "choices", "key": "top_upgraded"}],
        "table": ["update", "samples", "passes", "human_loss", "human_acc", "keep_kl", "keep_value", "lr"],
        "formats": {"human_acc": "pct", "lr": "exp", "samples": "int"},
    }


def _read(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(f)]


def _num(v):
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return v


def spec_for(run_dir):
    """The page for the run in `run_dir`, from its log's columns: train_runs.py, pretrain_human.py
    or train_fights.py."""
    try:
        with open(os.path.join(run_dir, "log.csv")) as f:
            head = f.readline().strip().split(",")
    except OSError:
        head = []
    try:
        with open(os.path.join(run_dir, "args.json")) as f:
            args = json.load(f)
    except (OSError, ValueError):
        args = {}
    return run_spec(args) if "floor" in head else pretrain_spec() if "passes" in head else fight_spec()


def compare_data(folder):
    """The comparison page's data (/compare), from <folder>/compare.json: {"title", "runs": [{"name",
    "label"}]}, runs trained alike but for one thing (the Watcher's network sizes, 2026-10-26). Each
    measure is a chart with a line per run, against steps (the runs log the same steps an update)
    and the floor against hours of training too."""
    with open(os.path.join(folder, "compare.json")) as f:
        cfg = json.load(f)
    keys = ("floor", "reached_act2", "reached_act3", "reached_act4", "win_heart", "win_rate", "keys", "run_steps",
            "value_loss", "explained_var", "entropy", "approx_kl", "steps_per_s", "gpu_reserved_gb")
    by_steps, hours, tiles = {}, [], []
    for r in cfg["runs"]:
        log = [x for x in _read(os.path.join(folder, r["name"], "log.csv")) if x.get("steps") is not None]
        for x in log:
            m = by_steps.setdefault(x["steps"], {"steps": x["steps"], "update": x.get("update"), "seconds": 0})
            m["seconds"] = max(m["seconds"], x.get("seconds") or 0)
            m.update({f"{r['name']}:{k}": x.get(k) for k in keys})
            hours.append({"hours": (x.get("seconds") or 0) / 3600, "update": x.get("update"), "steps": x["steps"],
                          f"{r['name']}:floor": x.get("floor")})
        floors = [x["floor"] for x in log[-100:] if x.get("floor") is not None]
        if floors:
            tiles.append({"label": f"{r['label']}: mean floor over its last 100 updates (to update {int(log[-1]['update'])})",
                          "source": "const", "value": sum(floors) / len(floors), "fmt": "floor"})
    pct = {"fmt": "pct", "min": 0, "max": 1}
    line = lambda key, title, fmt="num", **kw: {"title": title, "source": "log", "fmt": fmt,  # noqa: E731
                                                "series": [{"key": f"{r['name']}:{key}", "label": r["label"]} for r in cfg["runs"]], **kw}
    g = lambda group, charts: [{**c, "group": group} for c in charts]  # noqa: E731
    spec = {
        "title": cfg.get("title", "Comparison"), "about": cfg.get("about", ""), "tiles": tiles, "table": [], "formats": {},
        "charts": g("Runs", [
            line("floor", "Floor reached (last 500 runs), by steps", "floor", min=0),
            {**line("floor", "Floor reached (last 500 runs), by hours of training", "floor", min=0), "source": "hours", "x": "hours"},
            line("reached_act2", "Runs that reach Act 2", **pct),
            line("reached_act3", "Runs that reach Act 3", **pct),
            line("reached_act4", "Runs that reach Act 4", **pct),
            line("win_heart", "Runs that beat the Heart", **pct),
            line("win_rate", "Runs won (the Heart, or the Act 3 boss without the keys)", **pct),
            line("keys", "Keys held when runs end", min=0),
            line("run_steps", "Decisions per run", "int", min=0),
        ]) + g("Training health", [
            line("value_loss", "Value loss", min=0),
            line("explained_var", "Explained variance of the value", max=1),
            line("entropy", "Entropy", min=0),
            line("approx_kl", "Approximate KL", min=0),
        ]) + g("Speed and memory", [
            line("steps_per_s", "Steps per second", "int", min=0),
            line("gpu_reserved_gb", "GPU memory held, GB", min=0),
        ]),
    }
    return {"name": cfg.get("name", "compare"), "log": sorted(by_steps.values(), key=lambda m: m["steps"]), "eval": [],
            "choices": [], "hours": sorted(hours, key=lambda h: h["hours"]), "spec": spec}


def render(run_dir, spec=None, data=None):
    """The page for `run_dir` from its csv files as they are now; `spec` (fight_spec() by default)
    lists its tiles and charts. With `data` (compare_data's), that page instead."""
    if data is not None:
        return PAGE.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    data = {
        "name": os.path.basename(os.path.normpath(run_dir)),
        "log": _read(os.path.join(run_dir, "log.csv")),
        "eval": _read(os.path.join(run_dir, "eval.csv")),
        "choices": _read(os.path.join(run_dir, "choices.csv")),
        "spec": spec or fight_spec(),
    }
    return PAGE.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))


def write(run_dir, spec=None):
    """Writes <run_dir>/live.html (`render`), for opening from disk; the server renders its pages
    itself on each request."""
    tmp = os.path.join(run_dir, "live.html.tmp")
    with open(tmp, "w") as f:
        f.write(render(run_dir, spec))
    os.replace(tmp, os.path.join(run_dir, "live.html"))


# A run counts as training while its files changed in the last this many seconds.
RUNNING_S = 600


def runs(folder):
    """The runs under `folder` (folders with a live.html), newest change first: name, when its
    files last changed, whether it is training, and its last log and eval rows."""
    out = []
    for name in os.listdir(folder):
        d = os.path.join(folder, name)
        page = os.path.join(d, "live.html")
        if not os.path.isfile(page):
            continue
        changed = max(os.path.getmtime(os.path.join(d, f)) for f in ("live.html", "log.csv", "eval.csv") if os.path.exists(os.path.join(d, f)))
        args = {}
        try:
            with open(os.path.join(d, "args.json")) as f:
                args = json.load(f)
        except (OSError, ValueError):
            pass
        last = lambda rows: rows[-1] if rows else {}  # noqa: E731
        out.append({"name": name, "changed": changed, "running": time.time() - changed < RUNNING_S, "args": args,
                    "log": last(_read(os.path.join(d, "log.csv"))), "eval": last(_read(os.path.join(d, "eval.csv")))})
    return sorted(out, key=lambda r: -r["changed"])


def _ago(t):
    s = time.time() - t
    if s < 3600:
        return f"{max(int(s // 60), 0)} min ago"
    if s < 86400:
        return f"{s / 3600:.1f} h ago"
    return time.strftime("%Y-%m-%d", time.localtime(t))


def _nav(all_runs, current=None, compare=False):
    """The bar on every page: the runs training now, the list of all runs and (with a
    compare.json) the comparison."""
    live = [r for r in all_runs if r["running"]]
    links = " · ".join(
        f'<a href="/{urllib.parse.quote(r["name"])}/live.html"{" class=\"on\"" if r["name"] == current else ""}>{html.escape(r["name"])}</a>' for r in live)
    return (f'<nav class="runs"><a href="/">Home</a> · Training now: {links or "none"} · '
            f'<a href="/runs">All runs ({len(all_runs)})</a>' + (' · <a href="/compare">Comparison</a>' if compare else "") + '</nav>')


NAV_CSS = """<style>
nav.runs { font: 13px/1.4 system-ui, sans-serif; padding: 8px 20px; background: #e9e8e4; color: #52514e; }
nav.runs a { color: #2a78d6; text-decoration: none; } nav.runs a.on { font-weight: 600; color: inherit; }
@media (prefers-color-scheme: dark) { nav.runs { background: #222220; color: #c3c2b7; } nav.runs a { color: #3987e5; } }
</style>"""


def _with_nav(page, nav):
    return page.replace("<body>", "<body>" + NAV_CSS + nav, 1)


def index_page(all_runs, compare=False):
    """/runs: every run, newest change first, with what it is and how it was doing."""
    def row(r):
        a, log, ev = r["args"], r["log"], r["eval"]
        what = ", ".join(x for x in (a.get("character", ""), "practice" if a.get("practice", 0) >= 1 else "",
                                     f"A{a['ascension_range'][0]}-{a['ascension_range'][1]}" if a.get("ascension_range") else "") if x)
        won = f"{100 * float(ev['win_rate']):.1f}% ({ev.get('set', '')})" if ev.get("win_rate") not in (None, "") else ""
        update = log.get("update")
        update = f"{int(update)}" if isinstance(update, (int, float)) else ""
        return (f"<tr><td><a href=\"/{urllib.parse.quote(r['name'])}/live.html\">{html.escape(r['name'])}</a></td>"
                f"<td>{'training' if r['running'] else ''}</td><td>{_ago(r['changed'])}</td><td>{update}</td>"
                f"<td>{html.escape(what)}</td><td>{html.escape(won)}</td></tr>")
    rows = "".join(row(r) for r in all_runs)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Training runs</title>{NAV_CSS}<style>
body {{ margin: 0; font: 13px/1.4 system-ui, sans-serif; background: #f3f3f1; color: #0b0b0b; }}
main {{ padding: 16px 20px; }} h1 {{ font-size: 18px; margin: 0 0 12px; }}
table {{ border-collapse: collapse; font-variant-numeric: tabular-nums; }}
th, td {{ padding: 4px 12px 4px 0; text-align: left; border-bottom: 1px solid #e6e5e1; }} th {{ color: #52514e; font-weight: 600; }}
a {{ color: #2a78d6; text-decoration: none; }}
@media (prefers-color-scheme: dark) {{ body {{ background: #111110; color: #f0efec; }} td, th {{ border-color: #2b2b29; }} th {{ color: #c3c2b7; }} a {{ color: #3987e5; }} }}
</style></head><body>{_nav(all_runs, compare=compare)}<main><h1>Training runs</h1>
<table><tr><th>Run</th><th></th><th>Last change</th><th>Update</th><th>What</th><th>Last eval won</th></tr>{rows}</table></main></body></html>"""


class _Quiet(SimpleHTTPRequestHandler):
    """The runs folder, plus the home page (the run training now), /runs (all runs) and a bar of
    links on every live.html. Run pages are rendered from the csv files on each request, so a
    server started with newer code shows every run, running ones included, with its pages."""

    def log_message(self, *args):
        pass

    def _send(self, body, ctype="text/html; charset=utf-8"):
        data = body.encode()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        path = urllib.parse.unquote(urllib.parse.urlsplit(self.path).path)
        folder = self.directory
        compare = os.path.isfile(os.path.join(folder, "compare.json"))
        if path in ("/", "/index.html", "/runs", "/runs/"):
            all_runs = runs(folder)
            live = [r for r in all_runs if r["running"]]
            if path in ("/", "/index.html") and live:
                return self._send(_with_nav(render(os.path.join(folder, live[0]["name"]), spec_for(os.path.join(folder, live[0]["name"]))),
                                            _nav(all_runs, live[0]["name"], compare)))
            return self._send(index_page(all_runs, compare))
        if path in ("/compare", "/compare/") and compare:
            return self._send(_with_nav(render(None, data=compare_data(folder)), _nav(runs(folder), None, compare)))
        parts = path.strip("/").split("/")
        if len(parts) == 2 and parts[1] == "live.html" and os.path.isfile(os.path.join(folder, parts[0], "live.html")) and "." not in parts[0][:1]:
            run_dir = os.path.join(folder, parts[0])
            return self._send(_with_nav(render(run_dir, spec_for(run_dir)), _nav(runs(folder), parts[0], compare)))
        return super().do_GET()


def serve(folder, port, host="0.0.0.0"):
    """Serves `folder` over HTTP from a background thread; returns the base URL, or None if the
    port is taken (another run already serving the same folder is fine). Its home page is the run
    training now; /runs lists them all."""
    try:
        server = ThreadingHTTPServer((host, port), functools.partial(_Quiet, directory=folder))
    except OSError as e:
        print(f"chart server: port {port} unavailable ({e}); use --port to pick another")
        return None
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://{_lan_ip()}:{port}"


def _lan_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 9))  # no packet is sent; this just picks the outgoing interface
            return s.getsockname()[0]
    except OSError:
        return "localhost"


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>Training</title>
<style>
.viz-root {
  color-scheme: light;
  --surface-0: #f3f3f1; --surface-1: #fcfcfb;
  --text-primary: #0b0b0b; --text-secondary: #52514e; --text-muted: #8a8984;
  --grid: #e6e5e1; --axis: #c9c8c2;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100; --series-5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) .viz-root {
    color-scheme: dark;
    --surface-0: #111110; --surface-1: #1a1a19;
    --text-primary: #f0efec; --text-secondary: #c3c2b7; --text-muted: #8f8e86;
    --grid: #2b2b29; --axis: #44443f;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
  }
}
body { margin: 0; font: 13px/1.4 system-ui, sans-serif; }
.viz-root { background: var(--surface-0); color: var(--text-primary); min-height: 100vh; padding: 16px 20px; box-sizing: border-box; }
h1 { font-size: 18px; margin: 0 0 4px; font-weight: 600; }
.sub { color: var(--text-secondary); margin-bottom: 14px; }
.tiles { display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 14px; }
.tile { background: var(--surface-1); border-radius: 8px; padding: 10px 14px; min-width: 120px; }
.tile .v { font-size: 22px; font-weight: 600; font-variant-numeric: tabular-nums; }
.tile .k { color: var(--text-secondary); font-size: 12px; }
.notes { margin: -4px 0 14px; color: var(--text-secondary); }
.notes b { color: var(--text-primary); font-weight: 600; }
.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(420px, 1fr)); gap: 12px; }
.card { background: var(--surface-1); border-radius: 8px; padding: 10px 12px 6px; position: relative; }
.card h2 { font-size: 13px; font-weight: 600; margin: 0 0 2px; }
.legend { display: flex; gap: 12px; flex-wrap: wrap; color: var(--text-secondary); font-size: 12px; min-height: 16px; }
.legend i { display: inline-block; width: 12px; height: 2px; vertical-align: middle; margin-right: 4px; border-radius: 1px; }
.legend i.dash { background: none !important; border-top: 2px dashed var(--text-muted); height: 0; }
svg { display: block; width: 100%; height: 190px; overflow: visible; }
svg text { fill: var(--text-muted); font-size: 11px; font-variant-numeric: tabular-nums; }
.tip { position: absolute; pointer-events: none; background: var(--surface-0); border: 1px solid var(--axis); border-radius: 6px;
  padding: 6px 8px; font-size: 12px; display: none; white-space: nowrap; z-index: 2; color: var(--text-primary); }
.tip .s { color: var(--text-secondary); }
details { margin-top: 16px; }
table { border-collapse: collapse; font-variant-numeric: tabular-nums; font-size: 12px; margin-top: 8px; }
th, td { padding: 3px 10px; text-align: right; border-bottom: 1px solid var(--grid); }
th { color: var(--text-secondary); font-weight: 600; }
.empty { color: var(--text-secondary); padding: 40px 0; }
h3.group { grid-column: 1 / -1; font-size: 14px; font-weight: 600; margin: 10px 0 -2px; color: var(--text-secondary); }
</style></head>
<body><div class="viz-root">
<h1 id="title"></h1><div class="sub" id="sub"></div>
<div class="tiles" id="tiles"></div>
<div class="notes" id="notes"></div>
<div class="grid" id="charts"></div>
<details><summary>Table: last 30 updates</summary><div id="table"></div></details>
</div>
<script>
const D = /*DATA*/null;
const css = v => getComputedStyle(document.querySelector('.viz-root')).getPropertyValue(v).trim();
const pct = v => v == null ? '–' : (100 * v).toFixed(1) + '%';
const f3 = v => v == null ? '–' : Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(3);
const msteps = v => (v / 1e6).toFixed(v < 1e7 ? 2 : 1) + 'M';
const NS = 'http://www.w3.org/2000/svg';
const el = (tag, attrs, parent) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); if (parent) parent.appendChild(e); return e; };
const log = D.log, ev = D.eval, choices = D.choices || [];
const last = log[log.length - 1];

document.getElementById('title').textContent = D.spec.title + ': ' + D.name;
const counts = (D.spec.counts || []).filter(c => last && last[c.key]).map(c => `${Math.round(last[c.key]).toLocaleString()} ${c.label}`);
document.getElementById('sub').textContent = last
  ? [`update ${last.update}`, `${msteps(last.steps)} steps`, ...counts, `${(last.seconds / 3600).toFixed(2)} h`,
     ev.length ? `last eval at update ${ev[ev.length - 1].update}` : D.hours ? D.spec.about : 'no evals yet',
     `page written ${new Date().toLocaleTimeString()}, reloads every 15 s`].filter(Boolean).join(' · ')
  : 'no updates logged yet';

const FMT = {
  pct, num: f3, int: v => v == null ? '–' : Math.round(v).toLocaleString(),
  exp: v => v == null ? '–' : v.toExponential(1), floor: v => v == null ? '–' : v.toFixed(1),
};
// Eval rows pivoted to one row per checkpoint, with "set:field" keys.
const evRows = [...new Set(ev.map(r => r.steps))].map(st => {
  const row = {steps: st};
  for (const r of ev.filter(r => r.steps === st)) {
    row.update = r.update;
    for (const k in r) if (!['steps', 'update', 'set'].includes(k)) row[`${r.set}:${k}`] = r[k];
  }
  return row;
});
const lastEv = evRows[evRows.length - 1];
const lastCh = choices[choices.length - 1];
const SP = D.spec;
for (const n of SP.notes || []) {
  const src = n.source === 'choices' ? lastCh : n.source === 'eval' ? lastEv : last;
  const v = src && src[n.key];
  if (v != null && v !== '') document.getElementById('notes').insertAdjacentHTML('beforeend',
    `<div>${n.label} (update ${src.update}): <b>${String(v).replace(/[<>&]/g, '')}</b></div>`);
}
for (const t of SP.tiles) {
  const src = t.source === 'eval' ? lastEv : t.source === 'choices' ? lastCh : t.source === 'log' ? last : null;
  const v = t.source === 'const' ? t.value : src && src[t.key];
  if (v == null || v === '') continue;  // nothing to show for this kind of run
  document.getElementById('tiles').insertAdjacentHTML('beforeend', `<div class="tile"><div class="v">${v == null ? '–' : FMT[t.fmt](v)}</div><div class="k">${t.label}</div></div>`);
}

// chart(title, rows, series [{key, label, color}], refs [{y, label}], fmt, opts)
function chart(title, rows, series, refs, fmt, opts = {}) {
  // opts.legend: name the series even when there is one (the chart's others had no data).
  const card = document.createElement('div'); card.className = 'card';
  card.innerHTML = `<h2>${title}</h2>`;
  const legend = document.createElement('div'); legend.className = 'legend'; card.appendChild(legend);
  if (series.length > 1 || opts.legend) for (const s of series) legend.insertAdjacentHTML('beforeend', `<span><i style="background:${css(s.color)}"></i>${s.label}</span>`);
  for (const r of refs) legend.insertAdjacentHTML('beforeend', `<span><i class="dash"></i>${r.label}</span>`);
  const svg = el('svg', {}, card);
  const tip = document.createElement('div'); tip.className = 'tip'; card.appendChild(tip);
  document.getElementById('charts').appendChild(card);
  if (!rows.length) { el('text', {x: 0, y: 20}, svg).textContent = 'no data yet'; return; }

  const W = svg.clientWidth, H = 190, L = 46, R = 10, T = 8, Bm = 22;
  // opts.x: the x axis's key, steps (the default) or hours of training.
  const xk = opts.x || 'steps', xf = xk === 'hours' ? (v => v.toFixed(1) + ' h') : msteps;
  const xs = rows.map(r => r[xk]);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  let vals = [];
  for (const s of series) for (const r of rows) if (r[s.key] != null) vals.push(r[s.key]);
  for (const r of refs) vals.push(r.y);
  let y0 = opts.min ?? Math.min(...vals), y1 = opts.max ?? Math.max(...vals);
  if (y1 - y0 < 1e-9) { y1 += 1; y0 -= opts.min != null ? 0 : 1; }
  const pad = (y1 - y0) * 0.05; if (opts.min == null) y0 -= pad; if (opts.max == null) y1 += pad;
  const X = v => L + (v - x0) / (x1 - x0) * (W - L - R), Y = v => T + (1 - (v - y0) / (y1 - y0)) * (H - T - Bm);

  for (let i = 0; i <= 4; i++) {
    const v = y0 + (y1 - y0) * i / 4, y = Y(v);
    el('line', {x1: L, x2: W - R, y1: y, y2: y, stroke: css(i ? '--grid' : '--axis'), 'stroke-width': 1}, svg);
    el('text', {x: L - 6, y: y + 4, 'text-anchor': 'end'}, svg).textContent = fmt(v);
  }
  for (let i = 0; i <= (Math.min(...xs) === Math.max(...xs) ? 0 : 4); i++) {
    const v = x0 + (x1 - x0) * i / 4;
    el('text', {x: X(v), y: H - 6, 'text-anchor': i ? (i === 4 ? 'end' : 'middle') : 'start'}, svg).textContent = xf(v);
  }
  for (const r of refs) el('line', {x1: L, x2: W - R, y1: Y(r.y), y2: Y(r.y), stroke: css('--text-muted'), 'stroke-width': 1.5, 'stroke-dasharray': '5 4'}, svg);
  for (const s of series) {
    const pts = rows.filter(r => r[s.key] != null).map(r => `${X(r[xk]).toFixed(1)},${Y(r[s.key]).toFixed(1)}`);
    el('polyline', {points: pts.join(' '), fill: 'none', stroke: css(s.color), 'stroke-width': 2, 'stroke-linejoin': 'round', 'stroke-linecap': 'round'}, svg);
    if (opts.markers) for (const r of rows) if (r[s.key] != null)
      el('circle', {cx: X(r[xk]), cy: Y(r[s.key]), r: 4, fill: css(s.color), stroke: css('--surface-1'), 'stroke-width': 2}, svg);
  }

  const cross = el('line', {y1: T, y2: H - Bm, stroke: css('--axis'), 'stroke-width': 1, visibility: 'hidden'}, svg);
  const dots = series.map(s => el('circle', {r: 4, fill: css(s.color), stroke: css('--surface-1'), 'stroke-width': 2, visibility: 'hidden'}, svg));
  const hit = el('rect', {x: L, y: 0, width: Math.max(W - L - R, 1), height: H, fill: 'transparent'}, svg);
  hit.addEventListener('mousemove', e => {
    const box = svg.getBoundingClientRect(), mx = e.clientX - box.left;
    let best = rows[0];
    for (const r of rows) if (Math.abs(X(r[xk]) - mx) < Math.abs(X(best[xk]) - mx)) best = r;
    const cx = X(best[xk]);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.setAttribute('visibility', 'visible');
    series.forEach((s, i) => {
      const v = best[s.key];
      dots[i].setAttribute('visibility', v == null ? 'hidden' : 'visible');
      if (v != null) { dots[i].setAttribute('cx', cx); dots[i].setAttribute('cy', Y(v)); }
    });
    tip.innerHTML = `<div class="s">${xk === 'hours' ? xf(best.hours) + ' · ' : ''}update ${best.update} · ${msteps(best.steps)} steps</div>` +
      series.map(s => `<div><i style="display:inline-block;width:8px;height:8px;border-radius:4px;background:${css(s.color)};margin-right:6px"></i>${s.label} <b>${fmt(best[s.key])}</b></div>`).join('') +
      refs.map(r => `<div class="s">${r.label} ${fmt(r.y)}</div>`).join('');
    tip.style.display = 'block';
    const left = cx + 70 + tip.offsetWidth > W ? cx - tip.offsetWidth + 2 : cx + 24;
    tip.style.left = left + 'px'; tip.style.top = '40px';
  });
  hit.addEventListener('mouseleave', () => { tip.style.display = 'none'; cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); });
}

// A series with no value but 0 or nothing is left out, and a chart with no series left: most
// charts belong to one kind of run (runs, practice fights, human fights...). Eval and choice
// charts wait for the first eval. Colours follow the series' place in the spec, so they stay put.
const S = ['--series-1', '--series-2', '--series-3', '--series-4', '--series-5'];
let group;
for (const c of SP.charts) {
  const rows = c.source === 'eval' ? evRows : c.source === 'choices' ? choices : c.source === 'hours' ? (D.hours || []) : log;
  const series = c.series.map((x, i) => ({...x, color: S[i % S.length]})).filter(x => rows.some(r => r[x.key] != null && r[x.key] !== 0));
  if (!series.length) continue;
  if (c.group && c.group !== group)
    document.getElementById('charts').insertAdjacentHTML('beforeend', `<h3 class="group">${c.group.replace(/[<>&]/g, '')}</h3>`);
  group = c.group;
  chart(c.title, rows, series, c.refs || [], FMT[c.fmt], {min: c.min, max: c.max, markers: c.source === 'eval' || c.source === 'choices', legend: c.series.length > 1, x: c.x});
}

const shown = log.slice(-30).reverse();
const cols = SP.table.filter(c => shown.some(r => r[c] != null && r[c] !== 0));
const fmtCol = (c, v) => v == null ? '–' : c === 'steps' ? msteps(v) : c === 'update' ? v : FMT[SP.formats[c] || 'num'](v);
document.getElementById('table').innerHTML = '<table><tr>' + cols.map(c => `<th>${c}</th>`).join('') + '</tr>' +
  shown.map(r => '<tr>' + cols.map(c => `<td>${fmtCol(c, r[c])}</td>`).join('') + '</tr>').join('') + '</table>';

let hovering = false;
document.addEventListener('mousemove', e => { hovering = !!e.target.closest('svg'); });
setInterval(() => { if (!hovering) location.reload(); }, 15000);
</script></body></html>
"""

if __name__ == "__main__":
    # live_chart.py <run folder> [port]: rebuild the page; with a port, also serve the runs
    # folder until stopped.
    run_dir = sys.argv[1]
    write(run_dir, spec_for(run_dir))
    if len(sys.argv) > 2:
        url = serve(os.path.dirname(os.path.abspath(os.path.normpath(sys.argv[1]))), int(sys.argv[2]))
        if url:
            print(f"{url}/{os.path.basename(os.path.normpath(sys.argv[1]))}/live.html  (Ctrl+C to stop)")
            threading.Event().wait()
