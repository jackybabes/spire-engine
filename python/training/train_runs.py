"""PPO on whole runs (milestone 4): the Ironclad (or --character), final act on, any ascension
(--ascension, --ascension-range). docs/models/ironclad-v1.md describes the released network's recipe.
Practice fights are the Ironclad's archetype decks by default; --practice-decks makes them any
character's grown, engine or random decks in Acts 1-4 (crates/sts-py practice.rs build_deck; with
--practice 1 and --eval-runs 0 a network learns to fight from scratch, as the Defect's defect-fights-0 does,
notes/defect-training-plan.md). Another character's practice fights may also be its human boss
scenarios (--scenarios, --scenario-share 1), with its own human data (human_extract.py
--character, human_dataset.py --min-asc, human_bosses.py --character). Seeded runs (--seeded)
exist for the Ironclad, the Silent (crates/sts-py practice.rs SILENT_ARCHETYPES: poison, shiv,
discard, dex, after the wiki's Silent strategy) and the Watcher (the stance engine: Mental Fortress+
and Rushdown).

    .venv/bin/python python/training/train_runs.py --name ironclad-runs-1 --init python/runs/ironclad-fights-2/latest.pt
    .venv/bin/python python/training/train_runs.py --name ironclad-runs-1 --resume
    .venv/bin/python python/training/train_runs.py --eval python/runs/ironclad-runs-1/latest.pt --ascension 20 --eval-runs 1024

The network is run_model.RunNet; --init starts its fight parts from a fight checkpoint
(train_fights.py), or its whole self from a run checkpoint. Each update plays `--envs` runs for
`--steps` steps (runs last about a thousand decisions, so they span several updates; finished
runs restart at once), then runs clipped PPO over the batch like train_fights.py.

Reward: +1 for winning the run, 0.02 per floor climbed and 0.1 per act boss beaten (progress),
plus potential-based shaping on HP:
    psi = hp_weight * HP / max HP - 0.05 * monster HP / monster HP at the fight's start
(the monster term only in combat; --hp-weight, default 0.2), paid as gamma * psi' - psi with psi
= 0 once a run ends. That form provably leaves the best policy unchanged: it only makes HP lost
or healed felt at once instead of when the run ends. --shaping-end with --fade-updates scales
psi linearly down to that factor (e.g. 0, leaving only progress and wins). --legacy-shaping is
the formula ironclad-runs-1 and ironclad-runs-2 trained with (one potential holding floor, act and HP, paid as
phi' - phi, with HP kept at a win); resume those runs with it. A run that dies on floor 10
collects about 0.2; one that beats the Heart about 2.1.

--curse-weight adds -curse_weight * (curses in the deck) to psi, so removing a curse is felt at
once (+curse_weight) and gaining one costs the same (--strike-weight does the same for basic
Strikes: the keeping costs below pay off only floor by floor); like the HP term it leaves the best policy
unchanged, and --curse-fade fades it out. --forced-purge-curse is the --forced-purge chance for a
shop visited with a curse in the deck, for practice at the removal that matters most.

With --scale-rewards, rewards are divided by a running estimate of the spread of the discounted
return (as in SB3's VecNormalize), so the value loss doesn't shrink to nothing next to the
policy loss.

Added since (see --help for each):
- --win-reward sets the win's reward (default 1; 3 lifted A20 from about 11% to 17%).
- Habit rewards on what each run step did (crates/sts-py `Habit`, not potential-based on
  purpose): --deck-weight and --deck-free (a growing cost per card added, earned back per card
  removed), --rest-penalty and --rest-from, --smith-bonus, --hallway-potion-penalty,
  --removal-skip-penalty, --recall-penalty, --elite-bonus, --strike-keep-cost (each floor climbed
  costs W per basic Strike in the deck), --curse-keep-cost (the same per removable curse; set it
  above the Strike's so a curse goes first), --strike-upgrade-penalty (each basic Strike upgraded
  on a rest site's or event's grid costs W); also --energy-penalty and, for seeded
  runs (--seeded), --synergy-bonus.
- Practice fights (--practice; crates/sts-py practice.rs) with their own value head, and boss
  fights from winning human runs among them (--scenarios, --scenario-share; human_bosses.py);
  --practice-potion-bonus pays for potions kept.
- Imitation of human choices between fights (--human-data, --human-weight, --human-kinds).
- A trained network as a teacher (--teacher): a KL to its move probabilities on practice-fight moves,
  or on every move (--teacher-moves all) and its value too (--teacher-value), fading out over
  --teacher-fade updates: kickstarting a new network (a larger one) from a trained one while PPO runs.
- Search as a teacher (--search-weight): during each rollout, --search-share of the fight moves
  are searched with the fight look-ahead (fight_search.py: the network's likeliest moves and
  ending the turn, each played out on redrawn copies by the network itself and scored by its
  value). A move's gain is its score minus the network's own move's, shrunk towards 0 by
  --search-shrink standard errors (the copies' luck: every move plays on the same copies, so the
  gap's spread copy by copy measures it). The moves played are still the policy's own, so PPO
  stays on-policy. Two targets:
  - soft: over the candidates, the rollout policy's own probabilities times exp(gain /
    --search-temp), pulled towards by a KL. It raised the policy's entropy and lost (ironclad-search-teach-2, ironclad-search-teach-3:
    0.30 -> 0.49 in 25 updates, where ironclad-runs-habits-12 held 0.26): re-anchored on the current policy every
    rollout, it keeps multiplying the odds of whatever alternatives the search favours, and
    noise favours some in every position.
  - lift: only clear wins are taught. Where a move's shrunk gain is positive, its probability is
    lifted to --search-floor (loss max(0, log floor - log p)); nothing is asked anywhere else.
  The search's best move as a plain one-move target (cross-entropy) flattened the policy at
  once (ironclad-search-teach-1): the network gave most of the moves the search changed almost no probability.
  Lifting clear wins did too (ironclad-search-teach-4): they had about 1 in 10,000.
  --search-play instead plays the search's clear win in the rollout (where a move's shrunk gain
  is positive) in place of the sampled move, and PPO learns from what follows like any other
  move (logged with the policy's own probability of it): its ratio clipping lets that move's
  probability grow by at most --clip an update, and only if its advantage is positive. Works
  with or without --search-weight. With --fight-net (the split) the fight network searches and
  plays its clear wins (--search-play only: nothing trains the fights), so the deck network's
  choices are judged by the stronger play they get at evals (search_eval.py); --search-rooms
  limits the search to some rooms' fights (the bosses: where it gains most per move searched),
  --search-scale-from names the run whose reward scale the fight network's values are in.
- --overlap plays the next rollout in a worker process while the trainer trains on the last;
  --grow-layers deepens an --init network with pass-through layers.
- Late starts (--late-starts; jack, 2026-11-06: "the fight network needs to be able to win with those cards"): a share of the
  games neither practice fights nor seeded start every episode from a human boss state (human_bosses.py --all-outcomes,
  Envs.set_late_starts) and play on as normal runs, no keys given: the fights practise those decks in a run's economy, and the
  value sees what they're worth at Act 3 and the Heart. Their rewards are a run's from where they start (`progress` is read from
  the state, so no floor or act behind it is paid), and they're kept out of the run stats and the habit counts and counted in the
  seeded_* ones.

Everything goes to python/runs/<name>/ as in train_fights.py; the chart page is served at
http://<this machine>:8000/<name>/live.html.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import gzip
import json
import math
import multiprocessing
import os
import queue
import sys
import time
import traceback
from collections import deque

# Less GPU memory lost to fragmentation between the compiled passes and the usual one
# (2026-10-30: ironclad-fights-6x256-pieces peaked at 19.0 GB without it, 8.0 GB with it).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
import torch.nn.functional as F

import analysis.block_eval as block_eval
import training.choice_stats as choice_stats
import training.live_chart as live_chart
from search.fight_search import FightSearch, reward_std
from core.fight_model import FLOAT_AT, HAND_FEATS, INT_AT, MAX_HAND, MAX_MONSTERS, MAX_OPTIONS, MONSTER_FEATS, masked, pick_device
from core.fixed_shapes import FixedShapes, kind_name, kinds, minibatches
from core.ppo import choice_gae, explained_variance, gae, ppo_update
from core.run_model import (COMBAT, MOVES_PER_MONSTER, PICK, RUN_FLOAT_AT, RUN_INT_AT, FightOutcomeHead, OutcomeHead, RunNet, fight_outcome_loss,
                            from_fight_checkpoint, outcome_loss,
                            upgrade_state)
from core.lookup import AGREE as LOOKUP_AGREE, Lookup, trained_events
from core.split import SplitPolicy
from core.sts_env import ACTIONS, LAYOUT, RUN_FLOAT_LEN, RUN_INT_LEN, RunEnv
from training.train_fights import RUNS, append_csv
import sts

W_FLOOR, W_ACT = 0.02, 0.1
W_HP, W_MONSTER = 0.2, 0.05  # --hp-weight and --monster-weight set them
# Curse cards by encoded card id (id + 1; 0 = no card), for --curse-weight.
IS_CURSE = np.array([False] + [n in choice_stats.CURSES for n in LAYOUT["card_names"]])
# Every character's basic Strike, by encoded card id, for --strike-keep-cost.
IS_STRIKE = np.array([False] + [n == "Strike" for n in LAYOUT["card_names"]])
# Curses a run can get rid of, for --curse-keep-cost (Ascender's Bane, Curse of the Bell and
# Necronomicurse can't be removed: no point charging for them).
UNREMOVABLE = {"Ascender's Bane", "Curse of the Bell", "Necronomicurse"}
IS_REMOVABLE_CURSE = np.array([False] + [n in choice_stats.CURSES - UNREMOVABLE for n in LAYOUT["card_names"]])
# Cards that are never worth playing (statuses and curses), by encoded card id, for --energy-penalty.
IS_DEAD_CARD = np.array([False] + [n in choice_stats.CURSES | choice_stats.UNPLAYABLE for n in LAYOUT["card_names"]])
END_TURN = LAYOUT["actions"]["end_turn"]
HAND_AT, SCALARS_AT = INT_AT["hand_card"][0], FLOAT_AT["scalars"][0]
HAND_FLOATS_AT = FLOAT_AT["hand"][0]
EVAL_SEED = 300_000
PRACTICE_EVAL_SEED = 600_000
ARCHETYPES = ["block", "exhaust", "strength", "self-damage"]  # crates/sts-py practice.rs ARCHETYPES (the Ironclad's)
# Every character's engines (practice.rs archetypes_for: --practice-decks' engine decks carry the engine's name): the others'
# besides the Ironclad's, so their practice wins are logged by engine too.
ENGINES = {c: list(sts.archetypes(c)) for c in ("ironclad", "silent", "defect", "watcher")}
OTHER_ENGINES = sorted({e for names in ENGINES.values() for e in names} - set(ARCHETYPES))
# --practice-decks fights: the deck kinds (an engine deck is logged under its engine's name and
# counted here as "engine") and the acts.
DECK_KINDS = ["grown", "engine", "random"]
DECK_ACTS = [1, 2, 3, 4]
# Practice fights from human_bosses.py scenarios: Act 2, Act 3 (both bosses at A20), Act 1 bosses, each act's elites.
BOSS_KINDS = ["boss2", "boss3", "boss1", "elite1", "elite2", "elite3", "boss4", "elite4"]
# Practice fights by room (their summary's "room").
ROOMS = {"Monster": "hallway", "Elite": "elite", "Boss": "boss"}
SCENARIO_EVAL_SEED = 700_000
HABITS = 13  # Envs.habits() values per game (crates/sts-py Habit::FIELDS)
LATE_KINDS = ("boss1", "boss2", "boss3")  # the states a run goes on from (--late-kinds); the Heart's would hold keys
SEARCH_CANDS = 8  # first moves a search target holds at most (--search-top + ending the turn)
# Counted over each rollout, per 1,000 decisions (log.csv), late starts' left out of both.
HABIT_FIELDS = ["cards_added", "cards_removed", "rests", "rests_high", "recalls_wasted", "elites_won", "smiths", "hallway_potions", "removals_skipped",
                "strikes_upgraded", "arch_in", "arch_off"]
IRONCLAD_SEEDS = ("exhaust", "block", "self-damage")  # crates/sts-py SEEDS

LOG_FIELDS = [
    "update", "steps", "runs", "seconds", "steps_per_s", "floor", "reached_act2", "reached_act3", "reached_act4", "win_rate", "win_heart", "keys",
    "run_steps", "timed_out", "reward", "reward_scale", "shaping", "forced_purge", "curse_weight", "policy_loss", "value_loss", "entropy",
    "kl_uniform", "next_move_loss", "next_move_acc", "next_move_rows",
    # --fight-outcome: the head's win cross-entropy and HP-share squared error, the share of fights called right, the win
    # chance's Brier score, and the rows with a label in the rollout.
    "fight_outcome_ce", "fight_outcome_hp_err", "fight_outcome_acc", "fight_outcome_brier", "fight_outcome_rows", "approx_kl", "clip_frac", "explained_var",
    # PPO's KL by row kind (--fixed-shapes: fights, fights with a card selection pending, the rest)
    # and, for an update stopped early, the share of that epoch's minibatches run and the kind of the last.
    "approx_kl_fight", "approx_kl_options", "approx_kl_other", "stopped_at", "stop_kind",
    # --proximal: the seconds of the pass at the update's start.
    "prox_s",
    # The gradient's norm before clipping (mean over the minibatches) and the share of minibatches clipped.
    "grad_norm", "grad_clipped",
    # --value-epochs: the epochs run, the value loss over them, the policy's KL from where PPO's epochs left it, seconds.
    "value_epochs_run", "value_loss_after", "value_kl", "value_s",
    # --human-outcome: the outcome head's loss and its parts, and how often it calls a human run's win right.
    "outcome_loss", "outcome_floors", "outcome_won", "outcome_heart", "outcome_won_acc",
    # Seconds of the update: in all, waiting for the rollout (--overlap: the worker's results and the
    # copy into the batch), PPO's training (its aux losses included), and the rollout itself (in the
    # worker, beside the training, with --overlap).
    "update_s", "wait_s", "ppo_s", "rollout_s",
    "epochs", "lr", "practice", "practice_fights", "practice_win", *[f"practice_win_{a}" for a in ARCHETYPES + BOSS_KINDS],
    *[f"practice_win_{k}" for k in DECK_KINDS], *[f"practice_win_act{a}" for a in DECK_ACTS],
    *[f"practice_win_{e}" for e in OTHER_ENGINES],
    *[f"practice_win_room_{r}" for r in ROOMS.values()],
    # HP lost in won practice fights (hp_lost below).
    "practice_hp_lost", "practice_hp_lost_share", *[f"practice_hp_lost_act{a}" for a in DECK_ACTS],
    # --retry-losses: replays of lost practice fights this update, the share of the recent ones won, and the share of
    # recent replayed fights that some replay won.
    "retry_fights", "retry_win", "retry_saved",
    "teacher_kl", "teacher_agree", "teacher_value_err", "teacher_fade", "seeded_runs", "seeded_win_exhaust", "seeded_win_block", "seeded_win_self-damage", "seeded_synergy",
    "seeded_win", "seeded_heart", "seeded_floor",
    *[f"deck_{a}" for a in ARCHETYPES], "seeded_deck_block", "seeded_deck_exhaust", "human_loss", "human_acc", "human_acc_card", "human_weight",
    "ascension_hi", *[f"win_a{lo}_{hi}" for lo, hi in ((0, 5), (6, 10), (11, 15), (16, 20))], *HABIT_FIELDS,
    "searches", "search_changed", "search_s", "search_loss", "search_agree", "search_played", "strikes_held", "curses_held",
    # Where a rollout's time goes (seconds over the update's steps; the network's part runs until the
    # sampled moves are back on the CPU, so it includes waiting for the GPU) and each process's
    # peak GPU memory in the update (PyTorch's allocations and what its allocator held, in GB).
    *[f"rollout_{k}_s" for k in ("to_gpu", "network", "engine", "observe", "glue")],
    # --lookup's part of "observe" (core/lookup.py Lookup.times): the engine's probes, the head, the writing; probe rows.
    "lookup_probe_s", "lookup_head_s", "lookup_write_s", "lookup_rows",
    # How often the deck network took the lookup's best option, by screen (core/lookup.py AGREE).
    *[f"lookup_agree_{k}" for k in LOOKUP_AGREE],
    "gpu_alloc_gb", "gpu_reserved_gb", "worker_gpu_alloc_gb", "worker_gpu_reserved_gb",
]
EVAL_FIELDS = ["update", "steps", "set", "runs", "floor", "reached_act2", "reached_act3", "reached_act4", "win_rate", "win_heart", "keys",
               "run_steps", "timed_out",
               "hp_left", *[f"deck_{a}" for a in ARCHETYPES], *block_eval.FIELDS[5:], *[f"win_{a}" for a in ARCHETYPES + BOSS_KINDS],
               *[f"hp_left_{a}" for a in ARCHETYPES + BOSS_KINDS], "first_boss",
               *[f"win_{k}" for k in DECK_KINDS], *[f"win_act{a}" for a in DECK_ACTS], *[f"win_{e}" for e in OTHER_ENGINES],
               "hp_lost", "hp_lost_share", *[f"hp_lost_act{a}" for a in DECK_ACTS], *[f"win_room_{r}" for r in ROOMS.values()], "eval_s"]
CHOICE_FIELDS = ["update", "steps"] + choice_stats.FIELDS

SCREEN_AT = RUN_INT_AT["screen"][0]
ROOM_AT = RUN_INT_AT["room"][0]
SEARCH_ROOMS = {1: "hallway", 2: "elite", 7: "boss"}  # encode_run.rs room_id + 1 (--search-rooms)
POTION_AT, END_TURN_AT = LAYOUT["actions"]["potion"][0], LAYOUT["actions"]["end_turn"]  # --search-potions
ACT_AT = RUN_FLOAT_AT["run_scalars"][0] + 4  # hp, max hp, gold, floor, act


def trained_fight_rows(ints, floats, args):
    """--deck-net: which rows of `ints`/`floats` (torch, any leading shape) this network trains on: fight rows, of the
    --fight-rooms rooms and --fight-acts acts only if set."""
    rows = ints[..., SCREEN_AT] == COMBAT
    if args.fight_rooms:
        ids = torch.tensor([k for k, v in SEARCH_ROOMS.items() if v in args.fight_rooms.split(",")], device=ints.device)
        rows = rows & torch.isin(ints[..., ROOM_AT].long(), ids)
    if args.fight_acts:
        rows = rows & torch.isin(floats[..., ACT_AT].round().long(), torch.tensor(args.fight_acts, device=ints.device))
    return rows
SHOP = 1 + LAYOUT["screens"].index("shop")
CARD_REWARD = 1 + LAYOUT["screens"].index("card_reward")
RS = RUN_FLOAT_AT["run_scalars"][0]  # hp, max hp, gold, floor, act, ...
FS = FLOAT_AT["scalars"][0]  # the fight's hp, max hp, ...
MON_AT = FLOAT_AT["monster"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="runs")
    p.add_argument("--resume", action="store_true", help="carry on from <run>/latest.pt")
    p.add_argument("--init", metavar="CKPT", help="start from a fight or run checkpoint's weights")
    p.add_argument("--grow-layers", type=int, default=0,
                   help="with --init: grow the encoder to this many layers, the new ones starting as pass-throughs (same play)")
    p.add_argument("--eval", metavar="CKPT", help="only score this checkpoint on the fixed runs")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1 << 41, help="training run seeds (far from the eval seeds)")
    p.add_argument("--ascension", type=int, default=0)
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"],
                   help="whose runs to play (archetype practice fights and seeded runs are the Ironclad's only)")
    p.add_argument("--ascension-range", type=int, nargs=2, metavar=("LO", "HI"),
                   help="training: each run draws its ascension evenly from LO..HI (practice fights and seeded runs too)")
    p.add_argument("--ascension-low-share", type=float, default=0.0,
                   help="with --ascension-range: this share of runs draws its ascension evenly from 0..LO-1 instead "
                        "(keeps play below the range from drifting; practice fights take --practice-ascension if set)")
    p.add_argument("--ascension-widen", type=int, default=0,
                   help="with --ascension-range: raise HI linearly to 20 over this many updates (from this run's start)")
    p.add_argument("--eval-ascensions", default=None,
                   help="evaluate the fixed runs at these ascensions (comma-separated; default: --ascension). The first "
                        "is logged as set 'runs' (choice stats in choices.csv), the others as 'runs_a<N>' (choices_a<N>.csv)")
    p.add_argument("--d", type=int, default=128, help="network width (a network trained from scratch)")
    p.add_argument("--layers", type=int, default=3, help="network depth (from scratch)")
    p.add_argument("--heads", type=int, default=4, help="attention heads (from scratch)")
    p.add_argument("--unblended", action="store_true",
                   help="a token per relic, power and distinct pile card instead of their sums (from scratch; fight_model.py)")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=128, help="steps per env per update")
    p.add_argument("--updates", type=int, default=0, help="stop after this many updates (0 = run until stopped)")
    p.add_argument("--stop-flat", type=float, default=0.0,
                   help="stop once the floor levels off: every 50 updates, the mean floor of the last 100 updates against the "
                        "100 before; when it gained less than this many floors twice in a row, the run stops (0 = never)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-end", type=float, default=None, help="decay the learning rate linearly to this over --updates")
    p.add_argument("--gamma", type=float, default=0.999)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.005, help="entropy bonus (combat decisions, and all others unless --ent-choices)")
    p.add_argument("--ent-choices", type=float, default=None,
                   help="entropy bonus for decisions outside combat (map, rewards, cards, rest sites, shops, events...)")
    p.add_argument("--ent-act1", type=float, default=None,
                   help="entropy bonus for Act 1's card rewards (jack, 2026-11-07: explore Act 1's picks, then build on what is found; the "
                        "networks put 90%% or more on one option at 80%% of Act 1's rewards); the others as --ent-choices")
    p.add_argument("--ent-shop", type=float, default=None,
                   help="entropy bonus for shop decisions (default: as the other choices), to keep trying card removal")
    p.add_argument("--zero-avoid", type=float, default=0.0,
                   help="weight of the KL from the uniform policy over the legal moves (Jaxcalibur's zero-avoiding term, "
                        "notes/ideas-from-other-game-ais.md) for combat decisions, and all others unless --zero-avoid-choices: "
                        "keeps a move the policy has given up on (Mental Fortress at a card reward) from reaching zero")
    p.add_argument("--zero-avoid-choices", type=float, default=None, help="--zero-avoid for decisions outside combat")
    p.add_argument("--zero-avoid-shop", type=float, default=None, help="--zero-avoid for shop decisions (default: as the other choices)")
    p.add_argument("--vf", type=float, default=0.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--proximal", action="store_true",
                   help="anchor PPO's clip and KL on the network at each update's start (one more forward pass over the batch), the "
                        "rollout's policy weighting the advantages (decoupled PPO): with --overlap the rollout is an update behind, "
                        "and anchoring on it made defect-fights-0's policy swing back and forth (2026-10-24)")
    p.add_argument("--prox-cap", type=float, default=2.0, help="--proximal: the cap on each row's advantage weight (start / rollout policy)")
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--value-epochs", type=int, default=0,
                   help="after PPO's epochs, this many more passes over the batch training only the value (phasic policy gradient's "
                        "auxiliary phase): the value learns more from each rollout without larger policy steps")
    p.add_argument("--value-clone", type=float, default=1.0,
                   help="--value-epochs: weight of the KL that holds the policy where PPO's epochs left it (the network is shared)")
    p.add_argument("--hp-weight", type=float, default=0.2, help="weight of HP / max HP in the shaping potential")
    p.add_argument("--monster-weight", type=float, default=0.05, help="weight of monster HP in the shaping potential (fights)")
    p.add_argument("--shaping-end", type=float, default=None, help="scale the HP shaping linearly down to this factor ...")
    p.add_argument("--fade-updates", type=int, default=0, help="... over this many updates (counted from this run's start)")
    p.add_argument("--legacy-shaping", action="store_true", help="the reward ironclad-runs-1 and ironclad-runs-2 trained with")
    p.add_argument("--scale-rewards", action="store_true", help="divide rewards by the running std of the discounted return")
    p.add_argument("--value-warmup", type=int, default=0, help="train only the value for this many updates first")
    p.add_argument("--max-steps", type=int, default=8000, help="a run past this many decisions ends as a loss")
    p.add_argument("--forced-purge", type=float, default=0.0,
                   help="training only: chance that arriving at a shop opens its card removal (paid; the network picks the card)")
    p.add_argument("--forced-purge-curse", type=float, default=0.0,
                   help="training only: the --forced-purge chance for a shop visited with a curse in the deck (the larger applies)")
    p.add_argument("--block-weight", type=float, default=0.0,
                   help="weight of block under Barricade (as a share of max HP) in the shaping potential, like --hp-weight")
    p.add_argument("--curse-weight", type=float, default=0.0, help="weight of -(curses in the deck) in the shaping potential")
    p.add_argument("--curse-fade", type=int, default=0,
                   help="fade --curse-weight linearly to 0 over this many updates (counted from this run's start; 0 = keep it)")
    p.add_argument("--forced-purge-fade", type=int, default=0,
                   help="fade --forced-purge linearly to 0 over this many updates (counted from this run's start; 0 = keep it)")
    p.add_argument("--practice", type=float, default=0.0,
                   help="training only: share of the games that play practice fights (archetype decks partway "
                        "through an act; see crates/sts-py/src/practice.rs) instead of runs")
    p.add_argument("--practice-archetype", choices=ARCHETYPES, default=None,
                   help="practice fights only of this archetype (block decks then hold Barricade 3 times in 4); with block, "
                        "each eval also runs block_eval.py's fixed block fights (eval.csv set 'block')")
    p.add_argument("--practice-ascension", type=int, nargs=2, metavar=("LO", "HI"),
                   help="practice fights draw their ascension evenly from LO..HI (harder fights punish sloppy play)")
    p.add_argument("--practice-decks", choices=["mix", *DECK_KINDS], default=None,
                   help="practice fights from crates/sts-py practice.rs build_deck, for any character: the starting deck "
                        "grown, an engine deck or a random deck (mix: a third each), sized from human runs, in Acts 1-4")
    p.add_argument("--practice-boss", type=float, default=None,
                   help="chance a practice fight is the act's boss (default 15%%)")
    p.add_argument("--practice-elite", type=float, default=None,
                   help="--practice-decks fights: the chance one not sent to the act's boss (--practice-boss) is an elite of its act "
                        "(Act 4's Shield and Spear); the rest roll as usual (jack, 2026-10-30: \"do More elites\")")
    p.add_argument("--retry-losses", type=int, default=0,
                   help="practice fights: a lost one is played again, the same fight (deck, relics, potions, HP, encounter) with "
                        "other draws and monster rolls, until a replay wins or it has been replayed this many times; the replays "
                        "train as usual and stay out of the practice win rates (logged as retry_*; jack, 2026-10-30: \"replay each "
                        "lost fight up to 8 times while practising and then train off the winning ones\", then \"relay the fights "
                        "with a differnt seed\")")
    p.add_argument("--retry-same-draws", action="store_true",
                   help="--retry-losses: replays with the fight's own draws and monster rolls (6%% of them won: the network "
                        "played much the same moves)")
    p.add_argument("--retry-temp", type=float, default=1.0,
                   help="--retry-losses: replays sample their moves from the policy with its logits divided by this (above 1: "
                        "flatter, other lines through the same fight; the fight's own randomness is unchanged). PPO anchors on "
                        "the policy's own probabilities and weights each row by them over the ones the move was sampled with "
                        "(decoupled PPO, capped at --prox-cap; jack, 2026-10-30: \"yes build it the temprature bit\": with the "
                        "policy's own sampling 6%% of replays won)")
    p.add_argument("--seeds", default=None,
                   help="--seeded runs only from these of the character's seeds (comma-separated names: mantra; default all of them, the Silent's control apart)")
    p.add_argument("--practice-act-weights", type=float, nargs=4, default=None, metavar=("ACT1", "ACT2", "ACT3", "ACT4"),
                   help="--practice-decks fights: the chance of each act (any scale; default 30/30/30/10%%)")
    p.add_argument("--practice-boss-weights", default=None,
                   help="--practice-decks fights: a weight per boss when an act's boss is drawn (\"Time Eater=3,Awakened One=2\"; 1 for the others)")
    p.add_argument("--scenarios", default=None,
                   help="boss fights from winning human runs (python/human/human_bosses.py's train.jsonl.gz) for practice fights")
    p.add_argument("--scenario-share", type=float, default=1.0, help="share of practice fights that play a --scenarios fight")
    p.add_argument("--scenario-acts", type=int, nargs="+", choices=[1, 2, 3, 4], default=None, help="--scenarios of these acts only")
    p.add_argument("--scenario-eval", default=None,
                   help="held-out scenarios (human_bosses.py's eval.jsonl.gz), each played once greedily at every eval")
    p.add_argument("--energy-penalty", type=float, default=0.0,
                   help="reward penalty for ending a fight turn with energy left and a playable card (not a status or "
                        "curse) in hand; a floor climbed is worth 0.02")
    p.add_argument("--seed-cards", default=None, help="--seeded runs get these cards in place of a named seed: comma-separated, \"+\" for upgraded (Envs.set_seed_cards)")
    p.add_argument("--seed-act", type=int, default=1, help="--seeded runs get their cards on this act's first map screen (1: from the start)")
    p.add_argument("--seeded", type=float, default=0.0,
                   help="training only: share of the games (after the practice ones) whose runs start with an engine in "
                        "the deck (Corruption+ and Feel No Pain, or Barricade+, Impervious and Body Slam)")
    p.add_argument("--late-starts", default=None, metavar="PATH[,PATH]",
                   help="training only: human boss states (human_bosses.py --all-outcomes files; not the eval file, which stays held out) "
                        "that --late-share of the games start every episode from, as normal runs from there (see the module docs)")
    p.add_argument("--late-share", type=float, default=0.25,
                   help="--late-starts: the share of the games neither practice fights nor seeded, not of all games (with --practice "
                        "0.3, 0.25 is 17.5%% of them); of games, not episodes, as --practice is")
    p.add_argument("--late-kinds", default="boss1,boss2", help=f"--late-starts: which bosses' states, of {','.join(LATE_KINDS)}, each an equal part of the pool")
    p.add_argument("--late-mix", default="winners:2,finishers:1,act2-deaths:1",
                   help="--late-starts: the pool's groups (analysis/play_on.py's) and their weights, at each boss")
    p.add_argument("--late-limit", type=int, default=12000, help="--late-starts: lines in the pool at most (1 KB or so of RAM each)")
    p.add_argument("--win-reward", type=float, default=1.0,
                   help="a won run's reward on top of its progress (0.02 a floor, 0.1 an act: about 1.2 by the last boss)")
    p.add_argument("--act3-reward", type=float, default=None,
                   help="in place of --win-reward: beating the Act 3 boss earns this, whether the run ends there (no keys) or goes on to Act 4")
    p.add_argument("--heart-reward", type=float, default=0.0, help="with --act3-reward: beating the Heart earns this on top")
    p.add_argument("--boss-reward", type=float, default=0.0,
                   help="beating the Act 1 boss and the Act 2 boss each earns this on top of the act's 0.1 (jack, 2026-10-30: the deck "
                        "network was paid for reaching a boss, 0.02 a floor, more than for beating it; half its runs died to them)")
    p.add_argument("--key-bonus", type=float, default=0.0,
                   help="each key held is worth this in the run's progress (kept at the end: a reward, not only shaping)")
    p.add_argument("--key-bonus-fade", type=int, default=0, help="--key-bonus fades to 0 over this many updates (0: it stays)")
    p.add_argument("--key-bonus-pair", action="store_true",
                   help="with --key-bonus-act4: keys also keep for a run that beat the Act 3 bosses with fewer than three, so that partial keys get some credit where they don't cost the run")
    p.add_argument("--free-keys", action="store_true",
                   help="every run, a late start's too, holds the three keys from the start (Envs.set_free_keys), so that it plays the Heart and is paid for it with keys' cost aside. "
                        "A run that doesn't reach Act 4 never plays the Heart, so without keys nothing tells a deck that beats the Heart from one that beats the Act 3 bosses")
    p.add_argument("--key-rule", action="store_true", help="a hand-written rule gets the real keys in Act 3 and the network isn't told (RunConfig::key_rule), so that it plays on after them; in place of --free-keys")
    p.add_argument("--key-bonus-act4", action="store_true",
                   help="--key-bonus only keeps for a run that reaches Act 4 (one that dies before gives it back): keys pay only where they lead")
    p.add_argument("--deck-weight", type=float, default=0.0,
                   help="habits: each card added to the deck costs W x (0.2 + (deck size - --deck-free) / 10, from 0), and each card "
                        "removed earns the same back: a growing cost for a bloated deck (not potential-based on purpose)")
    p.add_argument("--deck-free", type=int, default=15, help="--deck-weight: the deck size its cost starts growing from")
    p.add_argument("--rest-penalty", type=float, default=0.0,
                   help="habits: resting costs W x (HP share - --rest-from) / (0.9 - --rest-from): nothing at --rest-from, all of W "
                        "at 90%% HP and over")
    p.add_argument("--rest-from", type=float, default=0.6, help="--rest-penalty: the HP share it starts at")
    p.add_argument("--smith-bonus", type=float, default=0.0, help="habits: each card upgraded at a rest site earns W")
    p.add_argument("--strike-keep-cost", type=float, default=0.0,
                   help="habits: each floor climbed costs W per basic Strike in the deck (human A20 winners keep about 1 of 5)")
    p.add_argument("--strike-weight", type=float, default=0.0,
                   help="shaping: like --curse-weight for basic Strikes (removing one is felt at once, +W; potential-based)")
    p.add_argument("--curse-keep-cost", type=float, default=0.0,
                   help="habits: each floor climbed costs W per removable curse in the deck (above --strike-keep-cost: curses go first)")
    p.add_argument("--strike-upgrade-penalty", type=float, default=0.0,
                   help="habits: each basic Strike upgraded on a rest site's or event's grid costs W (human A20 winners almost never do)")
    p.add_argument("--archetype-eval", action="store_true",
                   help="--practice-eval's archetype practice fights (by archetype) even with --scenario-eval")
    p.add_argument("--archetype-bonus", type=float, default=0.0,
                   help="habits: each card added that belongs to the deck's leading archetype (practice.rs ARCHETYPES; a core card "
                        "of any if it has none) earns W while that archetype isn't complete (two core and four support), at most "
                        "--archetype-cap a run")
    p.add_argument("--archetype-cap", type=float, default=8, help="--archetype-bonus: cards rewarded at most a run")
    p.add_argument("--archetype-off-penalty", type=float, default=0.0,
                   help="habits: once the deck's leading archetype is complete, each card added outside it costs W")
    p.add_argument("--removal-skip-penalty", type=float, default=0.0,
                   help="habits: leaving a shop without the card removal it offered and could pay for costs W")
    p.add_argument("--hallway-potion-penalty", type=float, default=0.0,
                   help="habits: each potion drunk in a fight that isn't an elite or a boss costs W (keep them for those)")
    p.add_argument("--recall-penalty", type=float, default=0.0,
                   help="habits: Recall (the Ruby key) without the Sapphire key costs W (the keys only pay with all three)")
    p.add_argument("--elite-bonus", type=float, default=0.0, help="habits: each elite fight won earns W")
    p.add_argument("--practice-potion-bonus", type=float, default=0.0,
                   help="a practice fight won earns W more for each potion still held (it otherwise drinks every one on turn 1)")
    p.add_argument("--synergy-bonus", type=float, default=0.0,
                   help="seeded runs: reward per card of the seed's archetype added to the deck (block or exhaust core "
                        "and support in crates/sts-py/src/practice.rs), at most 6 a run; a floor climbed is worth 0.02")
    p.add_argument("--teacher", metavar="CKPT",
                   help="distillation: on practice-fight moves, also pull the policy towards this network's (KL from the "
                        "teacher's move probabilities), e.g. a block-deck network from block_eval.py's runs")
    p.add_argument("--teacher-weight", type=float, default=1.0, help="weight of the --teacher KL loss")
    p.add_argument("--teacher-batch", type=int, default=2048,
                   help="practice moves per minibatch for the --teacher loss (--teacher-moves all uses PPO's own minibatches)")
    p.add_argument("--fight-net", metavar="CKPT",
                   help="the split (notes/split-networks-plan.md, D2): this network plays the fights (practice fights too), held "
                        "fixed; this run's network makes the choices between fights and PPO trains it on those only, with "
                        "returns from choice to choice (--choice-lam). With --eval: the fights of the evaluated runs too")
    p.add_argument("--choice-lam", type=float, default=0.95,
                   help="--fight-net: GAE's lambda per choice (the steps between two choices are one transition)")
    p.add_argument("--teacher-moves", choices=["practice", "all"], default="practice",
                   help="the moves the --teacher loss covers: practice-fight moves, or every move of the rollout (kickstarting: "
                        "a new network, e.g. a larger one, learning from a trained one while PPO runs; Schmitt et al. 2018)")
    p.add_argument("--teacher-value", type=float, default=0.0,
                   help="weight of a squared error pulling the value towards the --teacher's on the same moves (its scale: "
                        "the same rewards)")
    p.add_argument("--teacher-fade", type=int, default=0,
                   help="the --teacher losses fall linearly from full at update 0 to nothing at update N (0: full throughout)")
    p.add_argument("--teacher-floor", type=float, default=0.0,
                   help="the --teacher-fade never takes the losses below this share (a light pull kept for good: "
                        "watcher-runs-wide-kickstart lost what it had learned once the teacher faded out)")
    p.add_argument("--teacher-target-kl", type=float, default=None,
                   help="PPO's early stop while the --teacher losses are on (they move the policy further an update than "
                        "PPO alone; default --target-kl)")
    p.add_argument("--search-weight", type=float, default=0.0,
                   help="search as a teacher: weight of a KL pulling the policy towards the fight look-ahead's target (see the "
                        "module docs; fight_search.py) on the fight moves searched during each rollout (0 = off)")
    p.add_argument("--search-temp", type=float, default=0.03,
                   help="--search-weight: the target multiplies a move's odds by e for each this much its score beats the "
                        "network's move's (value units)")
    p.add_argument("--search-play", action="store_true",
                   help="play the search's clear wins in the rollouts instead of the sampled moves (see the module docs)")
    p.add_argument("--search-target", choices=["soft", "lift"], default="soft",
                   help="--search-weight: the soft target over the candidates, or lifting clear wins to --search-floor (see the module docs)")
    p.add_argument("--search-floor", type=float, default=0.6, help="--search-target lift: the probability a clear win is lifted to")
    p.add_argument("--search-shrink", type=float, default=1.0,
                   help="--search-weight: score gains shrink towards 0 by this many standard errors (the copies' luck)")
    p.add_argument("--search-share", type=float, default=0.25, help="--search-weight: share of fight moves searched")
    p.add_argument("--search-k", type=int, default=8, help="--search-weight: redrawn copies per first move")
    p.add_argument("--search-top", type=int, default=4, help="--search-weight: the network's likeliest moves tried (and ending the turn)")
    p.add_argument("--search-turns", type=int, default=1, help="--search-weight: turns each playout runs")
    p.add_argument("--search-max-moves", type=int, default=30, help="--search-weight: moves at most in a playout")
    p.add_argument("--search-batch", type=int, default=1024, help="--search-weight: searched moves per PPO minibatch")
    p.add_argument("--search-value", choices=["run", "fight"], default="run", help="--search-weight: how a run's fights are scored (fight_search.py)")
    p.add_argument("--search-potion-value", type=float, default=None,
                   help="--search-weight: a potion kept outside boss fights, in rewards (fight_search.py)")
    p.add_argument("--search-rooms", default="all",
                   help="--search-weight / --search-play: the rooms whose fights are searched, comma-separated (hallway, elite, "
                        "boss), or all")
    p.add_argument("--search-potions", action="store_true",
                   help="--search-weight / --search-play: search only where the network's likeliest move is a potion")
    p.add_argument("--search-scale-from", default=None,
                   help="--search-play with --fight-net: the run checkpoint whose reward scale the fight network's values are in "
                        "(as search_eval.py --scale-from; default the fight network's own)")
    p.add_argument("--search-roots", type=int, default=96, help="--search-weight: games searched at once (RAM: about 40 copies of a run each)")
    p.add_argument("--search-pass-rows", type=int, default=2048, help="--search-weight: copies per network pass (GPU memory)")
    p.add_argument("--human-data", metavar="DIR",
                   help="human decisions (human_dataset.py's output folder): add an imitation loss on them to each PPO step")
    p.add_argument("--human-weight", type=float, default=0.05, help="weight of the human-imitation cross-entropy")
    p.add_argument("--human-kinds", default="card",
                   help="which human decisions to imitate, of card,rest,grid,boss,purge,event (comma-separated)")
    p.add_argument("--human-kind-weight", default=None,
                   help="per-kind weights on top of --human-weight, e.g. card=3 (the rest 1)")
    p.add_argument("--human-every", type=int, default=1,
                   help="imitate on every Nth PPO minibatch only, at N x --human-weight (the same pull, a Nth of the cost)")
    p.add_argument("--human-batch", type=int, default=1024, help="human decisions read per PPO minibatch")
    p.add_argument("--human-outcome", type=float, default=0.0,
                   help="weight of the human-outcome head's loss (run_model.OutcomeHead: how human runs went from positions between "
                        "fights), beside PPO's like the imitation (--human-every, --human-batch); the network gains the head")
    p.add_argument("--next-move", type=float, default=0.0,
                   help="weight of the next-move head's loss (run_model RunNet.next_move_pred; Jaxcalibur's opponent's-next-action "
                        "loss, notes/ideas-from-other-game-ais.md): each monster's move on the next turn of the fight, labelled from "
                        "the rollout (next_move_targets), beside PPO's; the network gains the head")
    p.add_argument("--next-move-batch", type=int, default=1024, help="fight rows per --next-move loss")
    p.add_argument("--fight-outcome", type=float, default=0.0,
                   help="weight of the fight-outcome head's loss (run_model RunNet.fight_outcome_pred: the chance the network wins "
                        "the practice fight and its HP share at the end, from any fight position, labelled by how the rollout's "
                        "practice fights ended); the head reads the network detached, so play is unchanged; the network gains it")
    p.add_argument("--fight-outcome-batch", type=int, default=2048, help="labelled fight rows per --fight-outcome loss")
    p.add_argument("--deck-net", default=None,
                   help="the split the other way round: this network plays the fights (and practice fights) and trains on them "
                        "only, the network in this checkpoint makes the choices between fights, held fixed; the fight steps' returns "
                        "run from fight move to fight move, the choices between counted as part of the game (jack, 2026-10-30: the "
                        "practice-trained fight network halved ironclad-v1's wins in whole runs: this teaches it the run's worth of "
                        "HP and potions)")
    p.add_argument("--lookup", action="store_true",
                   help="with --fight-net (one with a fight-outcome head): the deck network sees the lookup at each card reward "
                        "(core/lookup.py: each card's gain in the fight network's win chance and HP against this act's boss and "
                        "the next act's bosses; jack, 2026-10-30: \"add the lookup as input to the deck network\")")
    p.add_argument("--lookup-draws", type=int, default=4, help="--lookup: openings each probe fight is read from")
    p.add_argument("--lookup-lr-mult", type=float, default=1.0,
                   help="--lookup: the lookup's input layers (RunNet.lookup_opt, lookup_run: zero at the start) learn at this many "
                        "times the learning rate, the rest of the network at it (a deck network from ironclad-v1 at 3e-5 moves "
                        "them about 0.2 in 100 updates)")
    p.add_argument("--lookup-pick-lr-mult", type=float, default=None,
                   help="the lookup's pick weights (RunNet.lookup_pick: a few numbers per screen) at this many times the learning "
                        "rate instead of --lookup-lr-mult's (at 10 they reached about 0.05 in 50 updates: a hundredth of a logit "
                        "for a 2-point gain)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="keep a running average of the weights, decay D an update (0 off), saved beside each checkpoint as "
                        "ema/update_<n>.pt (and ema/latest.pt): the deck runs from ironclad-v1 swing about 4 points from "
                        "checkpoint to checkpoint")
    p.add_argument("--train-only", default=None,
                   help="only the parameters whose names start with one of these comma-separated prefixes learn, e.g. "
                        "lookup_,run_value: the lookup's layers steering a network held fixed (the deck runs from ironclad-v1 drift "
                        "once their whole network trains); the multiples of --lookup-lr-mult and --lookup-pick-lr-mult still apply")
    p.add_argument("--lookup-net", default=None,
                   help="--deck-net with --lookup: the fixed fight network (with an outcome head) the deck network's lookup reads")
    p.add_argument("--fight-rooms", default=None,
                   help="--deck-net: train on these rooms' fights only, comma-separated (hallway, elite, boss)")
    p.add_argument("--fight-acts", type=int, nargs="+", default=None, help="--deck-net: train on these acts' fights only")
    p.add_argument("--lookup-shaping", type=float, default=0.0,
                   help="--lookup: potential-based reward shaping, this times the change in the lookup's win chance against "
                        "this act's boss and the next act's (their mean; read at probed screens, 0 once a run is over)")
    p.add_argument("--lookup-every", action="store_true",
                   help="--lookup at every choice that changes what a fight starts with (rests, upgrades, removals, shops, boss "
                        "relics, reward relics and potions), not only card rewards (core/lookup.py)")
    p.add_argument("--fight-outcome-every", type=int, default=4, help="the --fight-outcome loss on every Nth PPO minibatch only")
    p.add_argument("--head-only", action="store_true",
                   help="with --fight-outcome: only the fight-outcome head learns, the rest of the network held as it is (an outcome "
                        "head for a network whose play shouldn't change, e.g. ironclad-v1 as the split's fight network)")
    p.add_argument("--next-move-every", type=int, default=16,
                   help="the --next-move loss on every Nth PPO minibatch only, at N x its weight (every 4th at 512 rows doubled "
                        "the PPO step's time, 9.5 s to 19.5 s, 2026-10-28: an uncompiled pass and a wait for the GPU each)")
    p.add_argument("--human-outcome-batch", type=int, default=None,
                   help="positions read per --human-outcome loss (default --human-batch)")
    p.add_argument("--human-outcome-data", metavar="DIR", default=None,
                   help="the outcome head's dataset (human_dataset.py from 2026-10-27, every run: wins and losses; default --human-data)")
    p.add_argument("--human-floor", type=float, default=0.0,
                   help="with --human-data: only lift a human's choice the policy gives less than this probability "
                        "(loss max(0, log floor - log p)); 0 = full imitation (cross-entropy)")
    p.add_argument("--practice-eval", type=int, default=384,
                   help="practice fights in each eval (fixed seeds, greedy; 0 = none); with --practice-decks, this many of "
                        "each deck kind in each act. The Ironclad's archetype ones only without --scenario-eval")
    p.add_argument("--gpu-gb", type=float, default=None,
                   help="cap this process's GPU memory at this many GB (fails instead of spilling into shared memory)")
    p.add_argument("--worker-gpu-gb", type=float, default=None, help="--overlap: the same cap for the rollout worker")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--overlap", action="store_true",
                   help="play the next rollout in a worker process while training on the last (each rollout is then one "
                        "update behind the network trained on it)")
    p.add_argument("--compile", action="store_true", help="torch.compile the network for the rollout and PPO passes")
    p.add_argument("--fixed-shapes", action="store_true",
                   help="fight rows through passes of fixed shape compiled with CUDA graphs (core/fixed_shapes.py: about twice "
                        "as fast a training pass), each PPO minibatch one kind of row; choices between fights through the usual pass")
    p.add_argument("--compile-backend", default=None, help="torch.compile backend (default inductor, which needs the Python headers)")
    p.add_argument("--port", type=int, default=8000, help="serve python/runs on this port while training (0 = off)")
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--eval-every", type=int, default=50,
                   help="score on the fixed runs every this many updates (0 = never: run eval_worker.py beside the trainer instead, "
                        "which evaluates each --save-every checkpoint on the CPU without stopping training)")
    p.add_argument("--eval-runs", type=int, default=512)
    return p.parse_args(argv)


class ReturnScale:
    """Running variance of each env's discounted return; rewards are divided by its std."""

    def __init__(self, n, gamma):
        self.ret = np.zeros(n)
        self.gamma = gamma
        self.count, self.mean, self.var = 1e-4, 0.0, 1.0

    def update(self, rew, done):
        self.ret = self.ret * self.gamma + rew
        x = self.ret
        b_mean, b_var, b_n = x.mean(), x.var(), len(x)
        delta, tot = b_mean - self.mean, self.count + b_n
        self.mean += delta * b_n / tot
        self.var = (self.var * self.count + b_var * b_n + delta ** 2 * self.count * b_n / tot) / tot
        self.count = tot
        self.ret[done] = 0.0

    @property
    def std(self):
        return float(np.sqrt(self.var + 1e-8))


def monster_hp(floats):
    a, b = MON_AT
    return floats[:, a:b].reshape(-1, MAX_MONSTERS, MONSTER_FEATS)[:, :, 0].sum(1)


def in_combat(ints):
    return ints[:, SCREEN_AT] == COMBAT


def wasted_energy(ints, floats, acts):
    """Per game: ending a fight turn with energy left and a card it could play (not a status or
    curse) in hand (see --energy-penalty and choice_stats' end_turn_energy)."""
    ending = in_combat(ints) & (acts == END_TURN)
    energy = floats[:, SCALARS_AT + 3]
    hand = ints[:, HAND_AT:HAND_AT + MAX_HAND]
    rows = floats[:, HAND_FLOATS_AT:HAND_FLOATS_AT + MAX_HAND * HAND_FEATS].reshape(-1, MAX_HAND, HAND_FEATS)
    playable = (hand > 0) & (rows[:, :, 5] > 0) & (rows[:, :, 0] <= energy[:, None]) & ~IS_DEAD_CARD[hand]
    return ending & (energy > 0) & playable.any(1)


# The fight's turn among its scalars (encode.rs: hp, max hp, block, energy, energy per turn, turn, ...).
TURN_AT = FLOAT_AT["scalars"][0] + 5


def next_move_targets(ints, floats, done, moves):
    """The next-move head's labels (`--next-move`): for each fight row, each monster slot's move on
    the next turn of the same fight, the move it makes after the one its intent shows now, as its
    place among its monster's moves (0-15); -1 where the rollout has none: the next turn starts past
    the rollout's end, the fight or the episode ends first, or another monster holds the slot then.
    The label is the move the slot shows at the next turn's first row (`Envs.monster_next_moves`).
    `ints` (T, N, RUN_INT_LEN), `floats` (T, N, RUN_FLOAT_LEN), `done` (T, N) bool (the episode ended
    after the step), `moves` (T, N, MAX_MONSTERS). Returns (T * N, MAX_MONSTERS) long, rows in the
    rollout's order (step major), on their device."""
    T, N = done.shape
    dev = ints.device
    # Each game's rows in turn (game major), so a fight's turns are consecutive.
    def g(x):
        return x.transpose(0, 1).reshape(N * T, *x.shape[2:])
    combat = g(ints[..., RUN_INT_AT["screen"][0]] == COMBAT)
    turn, floor = g(floats[..., TURN_AT]), g(floats[..., RS + 3])
    ep = g(torch.cumsum(torch.cat([torch.zeros_like(done[:1]), done[:-1]]).long(), 0))
    mon = g(ints[..., INT_AT["monster"][0]:INT_AT["monster"][1]]).long()
    nxt_moves = g(moves).long()
    game = torch.arange(N, device=dev).repeat_interleave(T)
    # Segments: runs of rows in one game, episode, floor and turn (and in or out of a fight).
    change = torch.ones(N * T, dtype=torch.bool, device=dev)
    change[1:] = ((game[1:] != game[:-1]) | (ep[1:] != ep[:-1]) | (floor[1:] != floor[:-1]) | (turn[1:] != turn[:-1])
                  | (combat[1:] != combat[:-1]))
    seg = torch.cumsum(change.long(), 0) - 1
    first = torch.nonzero(change).squeeze(1)
    nxt_seg = seg + 1
    has = nxt_seg < first.numel()
    row = first[nxt_seg.clamp(max=first.numel() - 1)]  # the next segment's first row
    ok = (has & combat & combat[row] & (game[row] == game) & (ep[row] == ep) & (floor[row] == floor)
          & (turn[row] == turn + 1))
    m = nxt_moves[row]
    lab = torch.where(ok[:, None] & (mon[row] == mon) & (mon > 0) & (m > 0), (m - 1) % MOVES_PER_MONSTER, torch.full_like(m, -1))
    return lab.reshape(N, T, -1).transpose(0, 1).reshape(T * N, -1)


def progress(floats, w_key=0.0, w_boss=0.0):
    """The reward-bearing part of the run's state: floors climbed, acts reached, (--boss-reward) the
    Act 1 and 2 bosses beaten, and (--key-bonus) keys held."""
    keys = floats[:, RS + 7:RS + 10].sum(1) if w_key else 0.0
    act = floats[:, RS + 4]
    return W_FLOOR * floats[:, RS + 3] + W_ACT * (act - 1) + w_boss * np.clip(act - 1, 0, 2) + w_key * keys


def end_progress(ep, w_key=0.0, keys_act4=False, w_boss=0.0, keys_pair=False):
    """A finished run's progress. With `keys_act4` (--key-bonus-act4) its keys only count if it
    reached Act 4: a run that dies first gives the key bonus back (shaping only). With `keys_pair` (--key-bonus-pair) they
    also count for a run that beat the Act 3 bosses with fewer than three, which ends there."""
    keys = ep.get("keys", 0) if not keys_act4 or ep["act"] >= 4 or (keys_pair and ep["won"]) else 0
    return W_FLOOR * ep["floor"] + W_ACT * (ep["act"] - 1) + w_boss * min(max(ep["act"] - 1, 0), 2) + w_key * keys


def end_reward(ep, args):
    """A finished run's reward on top of its progress: --win-reward for a win; with --act3-reward,
    that for beating the Act 3 boss (the run won there without the keys, or went on to Act 4) and
    --heart-reward on top for beating the Heart."""
    if args.act3_reward is None:
        return args.win_reward if ep["won"] else 0.0
    heart = ep["won"] and ep["act"] >= 4
    return args.act3_reward * (ep["won"] or ep["act"] >= 4) + args.heart_reward * heart


BARRICADE = 9  # PowerId::Barricade + 1 (crates/sts-core/src/powers.rs)


def hp_shaping(ints, floats, start_mon, w_hp, w_mon, w_curse=0.0, w_block=0.0, w_strike=0.0):
    """psi per game (see the module docstring); `start_mon` is each fight's starting monster HP.
    `w_block` counts the player's block under Barricade (kept from turn to turn, so worth about as
    much as HP) like HP, up to max HP. `w_strike` counts each basic Strike in the deck against psi
    like a curse (--strike-weight)."""
    combat = in_combat(ints)
    hp = np.where(combat, floats[:, FS], floats[:, RS])
    max_hp = np.maximum(np.where(combat, floats[:, FS + 1], floats[:, RS + 1]), 1.0)
    mon = np.where(combat, np.clip(monster_hp(floats) / np.maximum(start_mon, 1.0), 0.0, 2.0), 0.0)
    psi = w_hp * hp / max_hp - w_mon * mon
    if w_block:
        a, b = INT_AT["player_power"]
        barricade = combat & (ints[:, a:b] == BARRICADE).any(1)
        psi = psi + np.where(barricade, w_block * np.minimum(floats[:, FS + 2], max_hp) / max_hp, 0.0)
    if w_curse:
        a, b = RUN_INT_AT["deck"]
        psi = psi - w_curse * IS_CURSE[ints[:, a:b]].sum(1)
    if w_strike:
        a, b = RUN_INT_AT["deck"]
        psi = psi - w_strike * IS_STRIKE[ints[:, a:b]].sum(1)
    return psi


def deck_kind(label):
    """A --practice-decks fight's kind from its label (an engine deck carries its engine's name)."""
    if label in ("grown", "random"):
        return label
    return "" if label in ARCHETYPES + BOSS_KINDS or not label else "engine"


def win_by(eps, key, prefix=""):
    """Win rates of `eps` grouped by `key(e)`: {prefix + group: rate}."""
    groups = {}
    for e in eps:
        groups.setdefault(key(e), []).append(e["won"])
    return {f"{prefix}{g}": float(np.mean(v)) for g, v in groups.items()}


def hp_lost(eps, prefix=""):
    """HP lost in the won practice fights of `eps`, from their starting HP to their HP after the
    fight (end-of-fight healing included; losses are left out, the win rates count them): the
    mean in HP and as a share of max HP, and by act in HP."""
    won = [e for e in eps if e["won"] and "hp_start" in e]
    lost = lambda es: float(np.mean([e["hp_start"] - e["hp"] for e in es])) if es else ""  # noqa: E731
    out = {f"{prefix}hp_lost": lost(won),
           f"{prefix}hp_lost_share": float(np.mean([(e["hp_start"] - e["hp"]) / max(e["max_hp"], 1) for e in won])) if won else ""}
    out.update({f"{prefix}hp_lost_act{a}": lost([e for e in won if e["act"] == a]) for a in DECK_ACTS})
    return out


def retry_summary(now, recent):
    """--retry-losses: replays this update; of the recent replays, the share won; of the recent fights lost the first
    time, the share some replay won (a fight still being replayed counts once it is decided)."""
    replays = [e for e in recent if e.get("retry")]
    out = {"retry_fights": sum(1 for e in now if e.get("retry")), "retry_win": float(np.mean([e["won"] for e in replays])) if replays else ""}
    # A replay that wins ends the fight's replays; so does a loss on the last replay. Fights decided: wins on a replay, and
    # losses that started no further replay (the next episode of the game was a new fight), counted from the summaries.
    saved = sum(1 for e in replays if e["won"])
    first_losses = sum(1 for e in recent if not e.get("retry") and not e["won"])
    out["retry_saved"] = saved / first_losses if first_losses else ""
    return out


def practice_summary(eps):
    fields = [f"practice_win_{a}" for a in ARCHETYPES + BOSS_KINDS + DECK_KINDS] + [f"practice_win_act{a}" for a in DECK_ACTS]
    fields += [f"practice_win_room_{r}" for r in ROOMS.values()]
    out = {"practice_win": float(np.mean([e["won"] for e in eps])) if eps else "", **{f: "" for f in fields}, **hp_lost(eps, "practice_")}
    if not eps:
        return out
    out.update(win_by([e for e in eps if e.get("room") in ROOMS], lambda e: ROOMS[e["room"]], "practice_win_room_"))
    for a in ARCHETYPES + BOSS_KINDS:
        sub = [e["won"] for e in eps if e.get("practice") == a]
        out[f"practice_win_{a}"] = float(np.mean(sub)) if sub else ""
    decks = [e for e in eps if deck_kind(e.get("practice"))]
    out.update(win_by(decks, lambda e: deck_kind(e["practice"]), "practice_win_"))
    out.update(win_by(decks, lambda e: f"act{e['act']}", "practice_win_"))
    out.update(win_by([e for e in eps if e.get("practice") in OTHER_ENGINES], lambda e: e["practice"], "practice_win_"))
    return out


def summarize(eps):
    if not eps:
        return {k: "" for k in ("floor", "reached_act2", "reached_act3", "reached_act4", "win_rate", "win_heart", "keys", "run_steps",
                                *[f"deck_{a}" for a in ARCHETYPES])}
    return {
        "floor": float(np.mean([e["floor"] for e in eps])),
        **{f"reached_act{a}": float(np.mean([e["act"] >= a for e in eps])) for a in (2, 3, 4)},
        "win_rate": float(np.mean([e["won"] for e in eps])),
        # Runs that beat the Heart, and the keys held when the run ended.
        "win_heart": float(np.mean([e["won"] and e["act"] >= 4 for e in eps])),
        "keys": float(np.mean([e.get("keys", 0) for e in eps])),
        "run_steps": float(np.mean([e["steps"] for e in eps])),
        # Cards of each archetype (practice.rs core and support) in the deck when the run ended.
        **{f"deck_{a}": float(np.mean([e.get("archetype_cards", {}).get(a, 0) for e in eps])) for a in ARCHETYPES},
    }


def practice_reward(ep, potion_bonus=0.0):
    """A practice fight's return: 0.5 for winning plus 0.5 x the share of max HP left (and
    `potion_bonus` for each potion still held); 0 for a loss, except 0.25 for an A20 Act 3 scenario
    lost to its second boss after beating the first. An escape (a Smoke Bomb) is still a loss, but it
    keeps 0.5 x the share of max HP it escaped with: in a run an escape survives (the 2026-10-30 potion
    analysis scored it a loss at -45 points; 2026-11-05, the Silent's plan)."""
    if ep["won"]:
        return 0.5 + 0.5 * ep["hp"] / max(ep["max_hp"], 1) + potion_bonus * ep.get("potions", 0)
    if ep.get("escaped"):
        return 0.5 * ep.get("hp_escaped", 0) / max(ep["max_hp"], 1)
    return 0.25 * min(ep.get("bosses_beaten", 0), 1)


def teacher_targets(teacher, amp, ints, floats, mask, practice, rows, chunk=8192):
    """The --teacher's log-probabilities over the legal moves and its value on `rows` of a rollout's
    flat tensors (practice rows on its practice value head), or (None, None) for no rows."""
    if len(rows) == 0:
        return None, None
    logp, value = [], []
    with torch.no_grad(), amp():
        for a in range(0, len(rows), chunk):
            r = rows[a:a + chunk]
            lg, v = teacher(ints[r].long(), floats[r], practice[r])
            logp.append(F.log_softmax(masked(lg.float(), mask[r]), -1))
            value.append(v.float())
    return torch.cat(logp), torch.cat(value)


def teacher_loss(net, amp, ints, floats, mask, practice, rows, t_logp, t_value):
    """KL(teacher || net) over the legal moves on `rows`, the squared error of the net's value
    against the teacher's, and how often their likeliest moves agree (`t_logp`, `t_value`: the
    teacher's on the same rows, teacher_targets)."""
    with amp():
        lg, v = net(ints[rows].long(), floats[rows], practice[rows])
    s_logp = F.log_softmax(masked(lg.float(), mask[rows]), -1)
    kl = (t_logp.exp() * (t_logp - s_logp)).nan_to_num(0.0).sum(-1).mean()
    v_err = (v.float() - t_value).pow(2).mean()
    agree = (s_logp.argmax(-1) == t_logp.argmax(-1)).float().mean()
    return kl, v_err, agree


def teacher_fade(args, update):
    """The share of the --teacher losses at `update`: 1 throughout, or with --teacher-fade N falling
    linearly from 1 at update 0 to 0 at update N, but never below --teacher-floor."""
    if not args.teacher_fade:
        return 1.0
    return max(getattr(args, "teacher_floor", 0.0), 1.0 - update / args.teacher_fade)


def boss_weights(spec):
    """--practice-boss-weights ("Time Eater=3,Awakened One=2") as [(boss, weight)], or None."""
    if not spec:
        return None
    out = []
    for part in spec.split(","):
        name, _, w = part.rpartition("=")
        out.append((name.strip(), float(w)))
    return out


def read_scenarios(path, acts=None):
    """human_bosses.py's scenario lines (JSON strings, for Envs.set_scenarios), of `acts` only if set."""
    with (gzip.open if path.endswith(".gz") else open)(path, "rt") as f:
        lines = [line.strip() for line in f if line.strip()]
    return [line for line in lines if not acts or json.loads(line)["act"] in acts]


def late_start_pool(args):
    """--late-starts: the pool's lines, and what it holds, to print. Each boss of --late-kinds gets an equal part of --late-limit, shared
    among --late-mix's groups by weight (the groups with lines at that boss: nobody dies in Act 2 at the Act 3 bosses); a group short
    of its share holds the others at that boss back with it, so the weights stay the weights. Which lines goes by a hash of the play id,
    so a resume has the same pool."""
    from analysis.play_on import GROUPS, NAMES, choose, read_lines  # (here, not at the top: play_on imports this module)
    kinds = args.late_kinds.split(",")
    if not set(kinds) <= set(LATE_KINDS):
        sys.exit(f"--late-kinds: some of {', '.join(LATE_KINDS)}, not {args.late_kinds}")
    if args.late_limit < 1:
        sys.exit(f"--late-limit: at least 1, not {args.late_limit}")
    try:
        mix = {group: float(weight) for group, weight in (part.split(":") for part in args.late_mix.split(","))}
    except ValueError:
        sys.exit(f"--late-mix: group:weight,group:weight..., not {args.late_mix}")
    if not all(g in GROUPS and w > 0 for g, w in mix.items()):
        sys.exit(f"--late-mix: weights above 0 on groups of {', '.join(GROUPS)}, not {args.late_mix}")
    at_boss = args.late_limit / len(kinds)
    cells, counts = read_lines(args.late_starts.split(","), set(kinds), math.ceil(at_boss))
    lines, rows, seen = [], [], set()
    for kind in kinds:
        have = {g: w for g, w in mix.items() if any(cells.get((kind, label)) for label in GROUPS[g])}
        if not have:
            sys.exit(f"--late-starts: no lines of {', '.join(mix)} at {kind}")
        per_weight = at_boss / sum(have.values())
        found = {g: choose(cells, counts, kind, g, math.ceil(per_weight * w)) for g, w in have.items()}
        playable = {g: [x for x in found[g][0] if not sts.scenario_problem(x)] for g in have}
        scale = min(1.0, *(len(playable[g]) / (per_weight * w) for g, w in have.items()))
        parts = {g: playable[g][:max(int(scale * per_weight * w + 1e-6), 1)] for g, w in have.items()}
        lines += [x for part in parts.values() for x in part]
        seen |= {g for g, part in parts.items() if part}
        rows.append(f"  {kind}: " + ", ".join(f"{NAMES[g]} {len(part):,} of {found[g][1]:,}" for g, part in parts.items()))
    if set(mix) - seen:
        sys.exit(f"--late-starts: no lines of {', '.join(sorted(set(mix) - seen))} at {', '.join(kinds)}")
    of_all = args.late_share * (1 - min(args.practice + args.seeded, 1))
    return lines, "\n".join([f"late starts: {len(lines):,} human boss states from {args.late_starts}, {args.late_share:.0%} of the games neither practice "
                             f"fights nor seeded ({of_all:.1%} of all):", *rows])


def evaluate_scenarios(net, device, lines, amp, batch=128, character="ironclad"):
    """Greedy play of every scenario in `lines` once (the held-out human boss fights, the same
    seeds every time): win rate and HP left, by kind; for Act 3, also the share that beat the
    first boss."""
    net.eval()
    env = RunEnv(batch, seed=SCENARIO_EVAL_SEED, practice=1.0, max_steps=4000, character=character)
    env.envs.set_scenarios(lines, 1.0, True)
    n = len(lines)
    # Game g's episode e plays scenario e * batch + g (Envs.set_scenarios, sequential).
    episode = np.zeros(batch, dtype=int)
    out = []
    while (episode * batch + np.arange(batch) < n).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        prac = torch.ones(batch, dtype=torch.bool, device=device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
        status = env.step(masked(logits.float(), mask).argmax(-1).cpu().numpy())
        for g in np.flatnonzero(status):
            if episode[g] * batch + g < n:
                out.append(json.loads(env.last_episode(int(g))))
            episode[g] += 1
    net.train()
    row = {"set": "bosses", "runs": len(out), "win_rate": float(np.mean([e["won"] for e in out])),
           "hp_left": float(np.mean([practice_reward(e) for e in out])), "timed_out": sum(e.get("timed_out", False) for e in out)}
    for a in BOSS_KINDS:
        eps = [e for e in out if e.get("practice") == a]
        row[f"win_{a}"] = float(np.mean([e["won"] for e in eps])) if eps else ""
        row[f"hp_left_{a}"] = float(np.mean([practice_reward(e) for e in eps])) if eps else ""
    act3 = [e for e in out if e.get("practice") == "boss3"]
    row["first_boss"] = float(np.mean([e.get("bosses_beaten", 0) >= 1 for e in act3])) if act3 else ""
    return row


def evaluate_practice(net, device, fights, amp, ascension=0, batch=128, character="ironclad", decks=None, bosses=False):
    """Greedy play of fixed practice fights (the same ones every time): win rate and HP left,
    overall and by archetype. With `decks` (--practice-decks), `fights` of each deck kind in each
    act (Acts 1-4), and the win rates by kind, act and room. With `bosses`, every one of them at
    the act's boss (the Heart in Act 4), A0-20 drawn per fight: the set "practice_boss"."""
    if decks:
        out = []
        for i, kind in enumerate(DECK_KINDS):
            for act in DECK_ACTS:
                env = RunEnv(min(batch, fights), seed=PRACTICE_EVAL_SEED + 16 * i + act + (1 << 20 if bosses else 0), ascension=ascension,
                             practice=1.0, max_steps=2000, character=character, practice_decks=kind, practice_act=act,
                             practice_boss=1.0 if bosses else None, ascension_range=(0, 20) if bosses else None)
                out += play_practice(net, device, env, fights, amp)
        row = {"set": "practice_boss" if bosses else "practice", "runs": len(out), "win_rate": float(np.mean([e["won"] for e in out])),
               "hp_left": float(np.mean([practice_reward(e) for e in out])), "timed_out": sum(e.get("timed_out", False) for e in out)}
        row.update(win_by(out, lambda e: deck_kind(e["practice"]), "win_"))
        row.update(win_by(out, lambda e: f"act{e['act']}", "win_"))
        # Engine decks by engine (the Ironclad's engines are named as his archetypes and logged under them already).
        row.update(win_by([e for e in out if e["practice"] in OTHER_ENGINES], lambda e: e["practice"], "win_"))
        row.update(win_by([e for e in out if e.get("room") in ROOMS], lambda e: ROOMS[e["room"]], "win_room_"))
        row.update(hp_lost(out))
        return row
    env = RunEnv(batch, seed=PRACTICE_EVAL_SEED, ascension=ascension, practice=1.0, max_steps=2000, character=character)
    out = play_practice(net, device, env, fights, amp)
    row = {"set": "practice", "runs": len(out), "win_rate": float(np.mean([e["won"] for e in out])),
           "hp_left": float(np.mean([practice_reward(e) for e in out])), "timed_out": sum(e.get("timed_out", False) for e in out)}
    for a in ARCHETYPES:
        eps = [e for e in out if e.get("practice") == a]
        row[f"win_{a}"] = float(np.mean([e["won"] for e in eps])) if eps else ""
        row[f"hp_left_{a}"] = float(np.mean([practice_reward(e) for e in eps])) if eps else ""
    return row


def first_episodes(out, n):
    """The first `n` of `out` ((episode number in its game, game), summary) in game order: every
    game's first episode, then second ones... Never in the order they ended, which would keep the
    shortest (a run won is the longest kind: 64 of 256 games gave A5 0% for 40%; fixed 2026-10-25)."""
    return [e for _, e in sorted(out, key=lambda x: x[0])[:n]]


def play_practice(net, device, env, fights, amp):
    """`fights` practice fights of `env` played greedily: their episode summaries."""
    net.eval()
    batch = len(env.envs)
    per_game = -(-fights // batch)
    done = np.zeros(batch, dtype=int)
    out = []
    while (done < per_game).any():
        obs = env.observe()
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        prac = torch.ones(batch, dtype=torch.bool, device=device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
        status = env.step(masked(logits.float(), mask).argmax(-1).cpu().numpy())
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                out.append(((done[g], g), json.loads(env.last_episode(int(g)))))
            done[g] += 1
    net.train()
    return first_episodes(out, fights)


def evaluate(net, device, runs, amp, ascension=0, batch=256, max_steps=8000, stats=None, character="ironclad", fight_net=None,
             lookup=None):
    """Greedy play on the fixed runs (the seeds run_baselines.py uses). With `stats` (a
    choice_stats.ChoiceStats for `min(batch, runs)` games), also counts the choices between fights.
    With `fight_net` (the split, core.split) that network plays the fights and `net` the rest; with
    `lookup` (core.lookup.Lookup), the deck network sees the lookup at card rewards."""
    policy = SplitPolicy(fight_net, net) if fight_net is not None else net
    policy.eval()
    batch = min(batch, runs)
    env = RunEnv(batch, seed=EVAL_SEED, ascension=ascension, max_steps=max_steps, character=character)
    per_game = -(-runs // batch)
    done = np.zeros(batch, dtype=int)
    out = []
    while (done < per_game).any():
        obs = env.observe()
        if lookup is not None:
            obs = lookup.fill(env.envs, obs)
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad(), amp():
            logits, _ = policy(ints, floats)
        acts = masked(logits.float(), mask).argmax(-1).cpu().numpy()
        if stats:
            stats.step(obs, acts, done < per_game)
        status = env.step(acts)
        for g in np.flatnonzero(status):
            if done[g] < per_game:
                out.append(((done[g], g), json.loads(env.last_episode(int(g)))))
                if stats:
                    stats.end(g)
            done[g] += 1
    net.train()
    out = first_episodes(out, runs)
    return {"set": "runs", "runs": len(out), "timed_out": sum(e.get("timed_out", False) for e in out), **summarize(out)}


def grow_layers(net, layers, device):
    """`net` with its encoder grown to `layers` layers, playing exactly as before: the new layers
    come after the old ones, each with its attention and feed-forward output projections at zero,
    so a (pre-norm) layer passes its input through unchanged until training gives it a use."""
    old = len(net.encoder.layers)
    heads = net.encoder.layers[0].self_attn.num_heads
    grown = RunNet(d=net.d, layers=layers, heads=heads, unblended=net.unblended, outcome=net.outcome is not None,
                   next_move=net.move_head is not None, fight_outcome=net.fight_outcome_head is not None).to(device)
    missing, unexpected = grown.load_state_dict(net.state_dict(), strict=False)
    assert not unexpected and all(k.startswith("encoder.layers.") for k in missing), (missing, unexpected)
    with torch.no_grad():
        for layer in grown.encoder.layers[old:]:
            for lin in (layer.self_attn.out_proj, layer.linear2):
                lin.weight.zero_()
                lin.bias.zero_()
    print(f"grew the encoder from {old} to {layers} layers ({sum(p.numel() for p in grown.parameters()):,} parameters)")
    return grown


def load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    upgrade_state(ck["net"])
    a = ck["args"]
    net = RunNet(d=a.get("d", 128), layers=a.get("layers", 3), heads=a.get("heads", 4), unblended=a.get("unblended", False),
                 outcome=a.get("outcome", False), next_move=a.get("move_head", False),
                 fight_outcome=a.get("fight_outcome_head", False)).to(device)
    return net, ck


def load_fight_net(path, device):
    """The split's fight network (--fight-net), held fixed: in eval mode, no gradients."""
    net, ck = load(path, device)
    net.load_state_dict(ck["net"])
    net.eval()
    for q in net.parameters():
        q.requires_grad_(False)
    return net


def rollout_buffers(T, N, device, shared=False):
    """A rollout's tensors (T steps x N games), on `device`, in shared memory with `shared`. The
    integer inputs are vocabulary ids, kept as int16 (a third less memory than int32)."""
    assert max(LAYOUT["vocab"].values()) < 2 ** 15
    spec = {"ints": ((T, N, RUN_INT_LEN), torch.int16), "floats": ((T, N, RUN_FLOAT_LEN), torch.float32),
            "mask": ((T, N, ACTIONS), torch.bool), "act": ((T, N), torch.long), "logp": ((T, N), torch.float32),
            # --retry-temp: the taken moves' log-probabilities under the policy itself ("logp": the ones sampled from).
            "logp_pi": ((T, N), torch.float32),
            "val": ((T, N), torch.float32), "prac": ((T, N), torch.bool), "adv": ((T, N), torch.float32),
            "ret": ((T, N), torch.float32), "search_slot": ((T, N, SEARCH_CANDS), torch.long),
            "search_p": ((T, N, SEARCH_CANDS), torch.float32),
            # --next-move: each monster slot's next move (Envs.monster_next_moves) and the episode ends.
            "next_move": ((T, N, MAX_MONSTERS), torch.int16), "done": ((T, N), torch.bool),
            # --fight-outcome: how each row's practice fight ended (won, HP share left), and whether it ended in this rollout.
            "fo_won": ((T, N), torch.float32), "fo_hp": ((T, N), torch.float32), "fo_has": ((T, N), torch.bool)}
    out = {k: torch.zeros(shape, dtype=dt, device=device) for k, (shape, dt) in spec.items()}
    if shared:
        for t in out.values():
            t.share_memory_()
    return out


class RolloutCounts:
    """What a rollout counts as it goes: the episodes that ended, the reward, habits, and the floors
    climbed with basic Strikes and curses in the deck."""

    def __init__(self):
        self.runs_now, self.practice_now, self.seeded_now = [], [], []
        self.reward_sum = 0.0
        self.habits = dict.fromkeys(HABIT_FIELDS, 0.0)
        self.habit_rows = 0
        self.strike_floors = self.curse_floors = self.floors_climbed = 0.0
        self.agree, self.agree_n = dict.fromkeys(LOOKUP_AGREE, 0), dict.fromkeys(LOOKUP_AGREE, 0)

    def count_lookup(self, acts, src, best, nothing, screens):
        """The moves `acts` (every game's) against the lookup's best options (Lookup.last) of the games `src`."""
        slot = acts[src] - PICK
        option = (slot >= 0) & (slot < MAX_OPTIONS)
        hit = np.where(option, best[np.arange(len(src)), np.clip(slot, 0, MAX_OPTIONS - 1)], nothing)
        for k, sid in LOOKUP_AGREE.items():
            m = screens == sid
            self.agree[k] += int(hit[m].sum())
            self.agree_n[k] += int(m.sum())

    def summary(self):
        return {"runs_now": self.runs_now, "practice_now": self.practice_now, "seeded_now": self.seeded_now,
                "reward_sum": self.reward_sum, "habits": self.habits, "habit_rows": self.habit_rows,
                "strikes_held": self.strike_floors / max(self.floors_climbed, 1.0),
                "curses_held": self.curse_floors / max(self.floors_climbed, 1.0),
                "lookup_agree": {k: self.agree[k] / self.agree_n[k] if self.agree_n[k] else "" for k in LOOKUP_AGREE}}



class Collector:
    """The games, and what a rollout carries over to the next: the observations, each fight's
    starting monster HP and the reward scale. Lives in the training process, or with --overlap in
    the rollout worker."""

    def __init__(self, args, device, amp, first_update, scaler_state=None, quiet=False, late_lines=None):
        self.args, self.device, self.amp, self.first_update = args, device, amp, first_update
        # (reuse_obs: the rollout keeps the last step's observation and no other.)
        self.env = env = RunEnv(args.envs, seed=args.seed + (first_update << 24), reuse_obs=True, ascension=args.ascension, max_steps=args.max_steps,
                                forced_purge=args.forced_purge, forced_purge_curse=args.forced_purge_curse, practice=args.practice,
                                seeded=args.seeded, ascension_range=tuple(args.ascension_range) if args.ascension_range else None,
                                practice_archetype=args.practice_archetype,
                                practice_ascension_range=tuple(args.practice_ascension) if args.practice_ascension else None,
                                practice_boss=args.practice_boss, character=args.character, practice_decks=args.practice_decks)
        if args.ascension_low_share:
            env.envs.set_ascension_low_share(args.ascension_low_share)
        if args.seeds:
            env.envs.set_seeds(args.seeds.split(","))
        if args.seed_cards:
            env.envs.set_seed_cards(args.seed_cards.split(","))
        if args.seed_act != 1:
            env.envs.set_seed_act(args.seed_act)
        if args.free_keys or args.key_rule:
            env.envs.set_free_keys(args.free_keys)
            env.envs.set_key_rule(args.key_rule)
            env.envs.reset()  # (from the first episodes)
        if args.practice_act_weights or args.practice_boss_weights or args.practice_elite is not None:
            env.envs.set_practice_focus(args.practice_act_weights, boss_weights(args.practice_boss_weights), args.practice_elite)
        if args.retry_losses:
            env.envs.set_retry_losses(args.retry_losses, not args.retry_same_draws)
        if args.scenarios:
            lines = read_scenarios(os.path.expanduser(args.scenarios), args.scenario_acts)
            skipped = env.envs.set_scenarios(lines, args.scenario_share, False)
            if not quiet:
                print(f"scenarios: {len(lines) - skipped:,} from {args.scenarios} ({skipped} skipped), {args.scenario_share:.0%} of practice fights")
        if late_lines:
            # (Only here, in the training games: the evals' games are their own.)
            skipped = env.envs.set_late_starts(late_lines, args.late_share)
            if skipped:
                raise ValueError(f"--late-starts: {skipped} of {len(late_lines):,} lines aren't {args.character}'s, or aren't act bosses'")
            env.envs.reset()  # (from the first episodes)
        self.obs = env.observe()
        combat = in_combat(self.obs["ints"])
        self.start_mon = np.where(combat, monster_hp(self.obs["floats"]), 0.0).astype(np.float32)
        self.scaler = ReturnScale(args.envs, args.gamma)
        if scaler_state is not None:
            self.scaler.count, self.scaler.mean, self.scaler.var = scaler_state
        self.search_rng = np.random.default_rng(args.seed + first_update)
        # --lookup: core.lookup.Lookup on the fight network, set by the trainer once that is loaded.
        self.lookup = None
        # --search-play with --fight-net: (the fight network, its FixedShapes or None, its reward scale, its potion bonus),
        # set by the trainer.
        self.search_net = None
        # --lookup-shaping: each game's potential, the lookup's last win chance against the bosses (0 at a run's start).
        self.lookup_phi = np.zeros(args.envs, np.float32)
        # --archetype-bonus: each game's archetype cards rewarded so far this run.
        self.arch_picks = np.zeros(args.envs, np.float32)

    def _step_rewards(self, env, before, obs, status, acts, start_mon, prog, psi, weights, acc):
        """One step's rewards for `env`'s games, moved from `before` to `obs` by `acts` (`status`:
        which episodes ended): progress and shaping, finished episodes (into `acc`), habits. Updates
        `start_mon` (each fight's starting monster HP) in place and `acc`'s counts; returns (rewards
        before scaling, done, progress and shaping potential at `obs`)."""
        args = self.args
        w_hp, w_mon, w_curse, w_key = weights
        N = len(acts)
        wasted = wasted_energy(before["ints"], before["floats"], acts) if args.energy_penalty else None
        done = status != 0
        combat = in_combat(obs["ints"])
        # A fight starts: remember its monsters' HP.
        started = combat & (~in_combat(before["ints"]) | done)
        start_mon[started] = monster_hp(obs["floats"][started])
        new_prog = progress(obs["floats"], w_key, args.boss_reward)
        new_psi = hp_shaping(obs["ints"], obs["floats"], start_mon, w_hp, w_mon, w_curse, args.block_weight, args.strike_weight)
        discount = 1.0 if args.legacy_shaping else args.gamma
        rew = (new_prog - prog + discount * new_psi - psi).astype(np.float32)
        for g in np.flatnonzero(done):
            ep = json.loads(env.last_episode(int(g)))
            if ep.get("practice"):
                # A practice fight: its own reward, nothing from run progress.
                acc.practice_now.append(ep)
                rew[g] = practice_reward(ep, args.practice_potion_bonus) - psi[g]
                continue
            if ep.get("seeded"):
                # A seeded run: rewarded like any run, but kept out of the run stats.
                acc.seeded_now.append(ep)
            else:
                acc.runs_now.append(ep)
            # psi is 0 once a run is over (legacy: HP still counted after a win).
            end_psi = w_hp * ep["hp"] / max(ep["max_hp"], 1) if args.legacy_shaping and ep["won"] else 0.0
            rew[g] = end_reward(ep, args) + end_progress(ep, w_key, args.key_bonus_act4, args.boss_reward, args.key_bonus_pair) - prog[g] + end_psi - psi[g]
        if wasted is not None:
            # Not potential-based on purpose: unused energy with a card to play is (almost) always a waste.
            rew -= args.energy_penalty * wasted.astype(np.float32)
        if args.synergy_bonus:
            # Not potential-based on purpose: a nudge towards building around the seed.
            rew += args.synergy_bonus * np.asarray(env.envs.synergy_picks(), dtype=np.float32)
        # Habits (crates/sts-py Habit): what this step did to the deck, a rest, a Recall, an elite.
        added, removed, size, rest_hp, recall, elite, smith, hall_pot, skip_removal, strikes_up, arch_in, arch_off, arch_complete = (
            np.asarray(env.envs.habits(), dtype=np.float32).reshape(N, HABITS).T)
        # Floors climbed this step (none for a finished run: the next one starts afresh) and
        # the basic Strikes taken along.
        climbed = np.where(done, 0.0, np.clip(obs["floats"][:, RS + 3] - before["floats"][:, RS + 3], 0, None))
        a, b = RUN_INT_AT["deck"]
        strikes = IS_STRIKE[obs["ints"][:, a:b]].sum(1)
        curses = IS_REMOVABLE_CURSE[obs["ints"][:, a:b]].sum(1)
        # The counts are the network's own runs': a late start earns the rewards below for what it does, but its deck started
        # as a human's and it plays only the later acts, so it stays out of them, as it does of the run stats.
        own = ~np.asarray(env.envs.late_mask(), dtype=bool)
        acc.habit_rows += int(own.sum())
        acc.strike_floors += float((strikes * climbed)[own].sum())
        acc.curse_floors += float((curses * climbed)[own].sum())
        acc.floors_climbed += float(climbed[own].sum())
        rested = rest_hp >= 0
        acc.habits["cards_added"] += float(added[own].sum())
        acc.habits["cards_removed"] += float(removed[own].sum())
        acc.habits["rests"] += float(rested[own].sum())
        acc.habits["rests_high"] += float((rest_hp >= 0.8)[own].sum())
        acc.habits["recalls_wasted"] += float(recall[own].sum())
        acc.habits["elites_won"] += float(elite[own].sum())
        acc.habits["smiths"] += float(smith[own].sum())
        acc.habits["hallway_potions"] += float(hall_pot[own].sum())
        acc.habits["removals_skipped"] += float(skip_removal[own].sum())
        acc.habits["strikes_upgraded"] += float(strikes_up[own].sum())
        acc.habits["arch_in"] += float(arch_in[own].sum())
        acc.habits["arch_off"] += float(arch_off[own].sum())
        if args.deck_weight:
            rew -= args.deck_weight * (0.2 + np.maximum(size - args.deck_free, 0) / 10) * (added - removed)
        if args.rest_penalty:
            rew -= args.rest_penalty * np.clip((rest_hp - args.rest_from) / (0.9 - args.rest_from), 0, 1) * rested
        if args.recall_penalty:
            rew -= args.recall_penalty * recall
        if args.elite_bonus:
            rew += args.elite_bonus * elite
        if args.smith_bonus:
            rew += args.smith_bonus * smith
        if args.strike_keep_cost:
            rew -= args.strike_keep_cost * strikes * climbed
        if args.curse_keep_cost:
            rew -= args.curse_keep_cost * curses * climbed
        if args.strike_upgrade_penalty:
            rew -= args.strike_upgrade_penalty * strikes_up
        if args.hallway_potion_penalty:
            rew -= args.hallway_potion_penalty * hall_pot
        if args.removal_skip_penalty:
            rew -= args.removal_skip_penalty * skip_removal
        if args.archetype_bonus or args.archetype_off_penalty:
            # Not potential-based on purpose: a nudge to keep building the deck's leading archetype (crates/sts-py
            # leading_archetype; a core card of any starts one), at most --archetype-cap cards a run, and once it is
            # complete to stop adding cards outside it.
            complete = arch_complete > 0
            gain = np.where(complete, 0.0, np.minimum(arch_in, np.maximum(args.archetype_cap - self.arch_picks, 0)))
            self.arch_picks += gain
            rew += args.archetype_bonus * gain - args.archetype_off_penalty * arch_off * complete
            self.arch_picks[done] = 0.0
        return rew, done, new_prog, new_psi

    def shaping_at(self, u):
        a = self.args
        if a.shaping_end is None:
            return 1.0
        return 1.0 + (a.shaping_end - 1.0) * min(u / a.fade_updates, 1.0)

    def forced_purge_at(self, u):
        a = self.args
        if not a.forced_purge_fade:
            return 1.0
        return max(1.0 - u / a.forced_purge_fade, 0.0)

    # The schedules below count updates from the run's start (update 0, also after --init), so a
    # --resume carries on where it was (they once restarted with the process: defect-runs-6's ascension
    # widening fell back from A19.5 to A18 when it resumed at update 170).

    def key_at(self, u):
        a = self.args
        if not a.key_bonus_fade:
            return a.key_bonus
        return a.key_bonus * max(1.0 - u / a.key_bonus_fade, 0.0)

    def curse_at(self, u):
        a = self.args
        if not a.curse_fade:
            return a.curse_weight
        return a.curse_weight * max(1.0 - u / a.curse_fade, 0.0)

    def ascension_hi_at(self, u):
        a = self.args
        lo, hi = a.ascension_range
        if not a.ascension_widen:
            return hi
        return int(hi + (20 - hi) * min(u / a.ascension_widen, 1.0))

    def collect(self, u, policy, out):
        """One rollout of every game (`args.steps` steps) with `policy` and the schedules of update
        `u`, written into `out` (`rollout_buffers`: observations, moves, values, advantages and
        returns, on the GPU or in shared memory). Returns the episodes that ended and the stats."""
        args, env, scaler, start_mon, device, amp = self.args, self.env, self.scaler, self.start_mon, self.device, self.amp
        T, N = args.steps, args.envs
        obs = self.obs
        t0 = time.time()
        asc_hi = None
        if args.ascension_range:
            asc_hi = self.ascension_hi_at(u)
            env.envs.set_ascension_range((args.ascension_range[0], asc_hi))
        fade = self.forced_purge_at(u)
        purge = args.forced_purge * fade
        env.envs.set_forced_purge(purge, args.forced_purge_curse * fade)
        w_curse = self.curse_at(u)
        shape = self.shaping_at(u)
        w_hp, w_mon = W_HP * shape, W_MONSTER * shape
        w_key = self.key_at(u)
        # Kept here: values, rewards and dones (for the advantages); the rest goes straight to `out`.
        b_val = torch.zeros(T, N, device=device)
        b_rew = torch.zeros(T, N, device=device)
        b_done = torch.zeros(T, N, device=device)
        on_gpu = out["ints"].is_cuda
        # Shaping weights only change between updates; within one, psi' and psi use the same.
        prog = progress(obs["floats"], w_key, args.boss_reward)
        psi = hp_shaping(obs["ints"], obs["floats"], start_mon, w_hp, w_mon, w_curse, args.block_weight, args.strike_weight)
        policy.eval()
        if self.lookup is not None:
            obs = self.lookup.fill(env.envs, obs)
        acc = RolloutCounts()
        search = None
        searched = {"n": 0, "changed": 0, "played": 0, "s": 0.0}
        if self.search_net is not None:
            # The split: the fixed fight network searches, in its own run's reward scale.
            fnet, ffixed, fscale, fbonus = self.search_net
            search = FightSearch(fnet, device, amp, k=args.search_k, top=args.search_top, fixed=ffixed, turns=args.search_turns,
                                 seed=args.seed + u, scale=fscale, potion_bonus=fbonus, max_moves=args.search_max_moves,
                                 value=args.search_value, potion_value=args.search_potion_value, roots_per_call=args.search_roots,
                                 batch=args.search_pass_rows)
        elif args.search_weight or args.search_play:
            # The network's own module (a compiled one would recompile for every batch size).
            search = FightSearch(getattr(policy, "_orig_mod", policy), device, amp, k=args.search_k, top=args.search_top,
                                 fixed=policy if isinstance(policy, FixedShapes) else None,
                                 turns=args.search_turns, seed=args.seed + u, scale=scaler.std if args.scale_rewards else 1.0,
                                 potion_bonus=args.practice_potion_bonus, max_moves=args.search_max_moves, value=args.search_value,
                                 potion_value=args.search_potion_value, roots_per_call=args.search_roots, batch=args.search_pass_rows)
        search_rooms = None if args.search_rooms == "all" else np.array(
            [k for k, v in SEARCH_ROOMS.items() if v in args.search_rooms.split(",")])
        phase = dict.fromkeys(("to_gpu", "network", "engine", "observe", "glue"), 0.0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        if args.fight_outcome:
            # --fight-outcome: each row's practice fight's ending, once it ends in this rollout (the rows from each
            # game's current fight's first row in the rollout, `fo_start`).
            fo_won, fo_hp = np.zeros((T, N), np.float32), np.zeros((T, N), np.float32)
            fo_has, fo_start = np.zeros((T, N), bool), np.zeros(N, dtype=int)
        for t in range(T):
            p0 = time.perf_counter()
            ints = torch.from_numpy(obs["ints"]).to(device, non_blocking=True)
            floats = torch.from_numpy(obs["floats"]).to(device, non_blocking=True)
            mask = torch.from_numpy(obs["mask"]).to(device, non_blocking=True)
            prac_np = np.asarray(env.envs.practice_mask(), dtype=bool)
            prac = torch.from_numpy(prac_np).to(device)
            out["prac"][t] = prac
            if args.next_move:
                out["next_move"][t] = torch.from_numpy(np.asarray(env.envs.monster_next_moves(), dtype=np.int16).reshape(N, MAX_MONSTERS))
            p1 = time.perf_counter()
            with torch.no_grad(), amp():
                logits, value = policy(ints.long(), floats, prac)
            logits, value = logits.float(), value.float()
            ml = masked(logits, mask)
            hot = None
            if args.retry_temp != 1.0 and args.retry_losses:
                # --retry-temp: games replaying a lost practice fight sample from the flattened policy.
                hot = torch.from_numpy(np.asarray(env.envs.retries(), dtype=np.int64) > 0).to(device)
            dist = torch.distributions.Categorical(logits=ml if hot is None else ml / torch.where(hot, args.retry_temp, 1.0)[:, None])
            a = dist.sample()
            if search is not None:
                # Targets for some fight moves from the look-ahead (the move played is still sampled): over
                # the candidates, the policy's own probabilities times exp(score gain / temperature).
                slots = np.full((N, SEARCH_CANDS), -1, dtype=np.int64)
                probs = np.zeros((N, SEARCH_CANDS), dtype=np.float32)
                games = np.flatnonzero(in_combat(obs["ints"]) & (self.search_rng.random(N) < args.search_share)
                                       & (True if search_rooms is None else np.isin(obs["ints"][:, ROOM_AT], search_rooms)))
                if args.search_potions and len(games):
                    # Only where the network's likeliest move is a potion.
                    own_moves = ml[torch.from_numpy(games).to(device)].argmax(-1).cpu().numpy()
                    games = games[(own_moves >= POTION_AT) & (own_moves < END_TURN_AT)]
                if len(games):
                    ts = time.time()
                    lg = masked(logits, mask).cpu().numpy()
                    for g, (cands, q, se) in search.scores(env.envs, obs["ints"], obs["mask"], lg, games, prac_np).items():
                        own = int(lg[g].argmax())
                        gain = q - q[list(cands).index(own)]
                        gain = np.sign(gain) * np.maximum(np.abs(gain) - args.search_shrink * se, 0.0)
                        searched["n"] += 1
                        if args.search_play:
                            # The clear win is played instead of the sampled move.
                            if gain.max() > 0:
                                a[g] = int(cands[gain.argmax()])
                                searched["played"] += 1
                            if not args.search_weight:
                                continue
                        if args.search_target == "lift":
                            # Only a clear win is taught: the move with the largest gain beyond the luck.
                            if gain.max() > 0:
                                slots[g, 0] = cands[gain.argmax()]
                                probs[g, 0] = 1.0
                                searched["changed"] += 1
                            continue
                        z = lg[g][cands] + gain / args.search_temp
                        z = np.exp(z - z.max())
                        slots[g, :len(cands)] = cands
                        probs[g, :len(cands)] = z / z.sum()
                        searched["changed"] += int(cands[z.argmax()] != own)
                    searched["s"] += time.time() - ts
                out["search_slot"][t] = torch.from_numpy(slots)
                out["search_p"][t] = torch.from_numpy(probs)
            if on_gpu:
                out["ints"][t], out["floats"][t], out["mask"][t] = ints, floats, mask
            else:
                out["ints"][t] = torch.from_numpy(obs["ints"])
                out["floats"][t] = torch.from_numpy(obs["floats"])
                out["mask"][t] = torch.from_numpy(obs["mask"])
            out["act"][t], out["logp"][t], b_val[t] = a, dist.log_prob(a), value
            out["logp_pi"][t] = out["logp"][t] if hot is None else F.log_softmax(ml, -1).gather(1, a[:, None]).squeeze(1)

            acts_np = a.cpu().numpy()
            if self.lookup is not None and self.lookup.last is not None:
                acc.count_lookup(acts_np, *self.lookup.last)
            p2 = time.perf_counter()
            before = obs
            status = env.step(acts_np)
            p3 = time.perf_counter()
            obs = env.observe()
            if self.lookup is not None:
                obs = self.lookup.fill(env.envs, obs)
            p4 = time.perf_counter()
            rew, done, new_prog, new_psi = self._step_rewards(env, before, obs, status, acts_np, start_mon, prog, psi, (w_hp, w_mon, w_curse, w_key), acc)
            prog, psi = new_prog, new_psi
            if args.lookup_shaping and self.lookup is not None:
                # Potential-based shaping from the lookup: the potential is its win chance against this act's boss and
                # the next act's (the mean), as last read at a probed screen; 0 once a run is over.
                phi = self.lookup_phi.copy()
                if self.lookup.last is not None:
                    src = self.lookup.last[0]
                    ra = RUN_FLOAT_AT["run_lookup"][0]
                    phi[src] = 0.5 * (obs["floats"][src, ra] + obs["floats"][src, ra + 2])
                phi[done] = 0.0
                rew = rew + args.lookup_shaping * (args.gamma * phi - self.lookup_phi)
                self.lookup_phi = phi
            if args.fight_outcome:
                for g in np.flatnonzero(done):
                    ep = json.loads(env.last_episode(int(g)))
                    if ep.get("practice"):
                        s0 = fo_start[g]
                        fo_won[s0:t + 1, g], fo_hp[s0:t + 1, g] = float(ep["won"]), ep["hp"] / max(ep["max_hp"], 1)
                        fo_has[s0:t + 1, g] = True
                    fo_start[g] = t + 1
            acc.reward_sum += float(rew.sum())
            if args.scale_rewards:
                scaler.update(rew, done)
                rew = rew / scaler.std
            b_rew[t] = torch.from_numpy(rew.astype(np.float32)).to(device)
            b_done[t] = torch.from_numpy(done.astype(np.float32)).to(device)
            out["done"][t] = torch.from_numpy(done.astype(bool))
            p5 = time.perf_counter()
            for k, dt in zip(phase, (p1 - p0, p2 - p1, p3 - p2, p4 - p3, p5 - p4)):
                phase[k] += dt

        with torch.no_grad():
            ints = torch.from_numpy(obs["ints"]).to(device)
            floats = torch.from_numpy(obs["floats"]).to(device)
            prac = torch.tensor(env.envs.practice_mask(), dtype=torch.bool, device=device)
            with amp():
                next_val = policy(ints.long(), floats, prac)[1].float()
            if args.fight_net:
                # The split: the deck network's returns run from choice to choice.
                choice = (out["ints"][:, :, SCREEN_AT] != COMBAT).to(device)
                adv, ret = choice_gae(b_rew, b_val, b_done, next_val, choice, args.gamma, args.choice_lam)
            elif args.deck_net:
                # The split the other way round (--deck-net): this network's returns run from fight move to fight move (of
                # the --fight-rooms / --fight-acts fights only, if set).
                fight = trained_fight_rows(out["ints"], out["floats"], args).to(device)
                adv, ret = choice_gae(b_rew, b_val, b_done, next_val, fight, args.gamma, args.lam)
            else:
                adv, ret = gae(b_rew, b_val, b_done, next_val, args.gamma, args.lam)
            out["val"].copy_(b_val)
            out["adv"].copy_(adv)
            out["ret"].copy_(ret)
            if args.fight_outcome:
                for k, a in (("fo_won", fo_won), ("fo_hp", fo_hp), ("fo_has", fo_has)):
                    out[k].copy_(torch.from_numpy(a))
        self.obs = obs
        if search is not None:
            searched.update(search.stats, times=dict(search.times))
        return {**acc.summary(), "rollout_s": time.time() - t0, "asc_hi": asc_hi, "purge": purge, "w_curse": w_curse, "shape": shape,
                "reward_std": scaler.std if args.scale_rewards else 1.0, "scaler_state": (scaler.count, scaler.mean, scaler.var),
                "searched": searched, "phase_s": phase, "gpu_gb": gpu_peak_gb(device),
                "lookup": self.lookup.take_times() if self.lookup is not None else {}}


def gpu_peak_gb(device):
    """This process's peak GPU memory since the last reset: (allocated, held by the allocator) in GB."""
    if device.type != "cuda":
        return ("", "")
    return (torch.cuda.max_memory_allocated(device) / 2 ** 30, torch.cuda.max_memory_reserved(device) / 2 ** 30)


def cap_gpu_memory(device, gb):
    """Caps this process's GPU memory (PyTorch's allocations, not the CUDA context's ~0.45 GB) at
    `gb`: past it the process fails out of memory instead of spilling into Windows' shared GPU
    memory (never wanted)."""
    if device.type == "cuda" and gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)


def next_job(q):
    """A worker process's next message on `q`; None once the trainer is gone (killed before it
    could stop its workers, which would otherwise wait forever)."""
    parent = multiprocessing.parent_process()
    while True:
        try:
            return q.get(timeout=5)
        except queue.Empty:
            if parent is not None and not parent.is_alive():
                return None


def rollout_worker(args, device_type, first_update, scaler_state, out, commands, results, late_lines=None):
    """--overlap: plays the rollouts in a process of its own (its own Python lock), each into the
    shared buffers `out`, with the weights the trainer sends for it: (update, state dict) on
    `commands`, None to stop. Each rollout's stats go back on `results` once `out` is written."""
    global W_HP, W_MONSTER
    try:
        W_HP, W_MONSTER = args.hp_weight, args.monster_weight
        device = torch.device(device_type)
        cap_gpu_memory(device, args.worker_gpu_gb)

        def amp():
            return torch.autocast(device_type, dtype=torch.bfloat16, enabled=not args.no_amp and device_type == "cuda")

        net = RunNet(d=args.d, layers=args.layers, heads=args.heads, unblended=args.unblended, outcome=args.outcome,
                     next_move=args.move_head, fight_outcome=args.fight_outcome_head).to(device)
        for q in net.parameters():
            q.requires_grad_(False)
        # Loading a state copies into the same parameters, so captured CUDA graphs stay valid.
        policy = FixedShapes(net, cuda_graphs=device.type == "cuda") if args.fixed_shapes else net
        col = Collector(args, device, amp, first_update, scaler_state, quiet=True, late_lines=late_lines)
        while (msg := next_job(commands)) is not None:
            u, state = msg
            net.load_state_dict(state)
            r = col.collect(u, policy, out)
            r["update"] = u  # which weights it played with (the trainer checks)
            results.put(r)
        parent = multiprocessing.parent_process()
        if parent is not None and not parent.is_alive():
            # The trainer was killed: nobody reads `results`, so its last rollout's stats (more than
            # a pipe holds) would keep the process waiting to flush them at exit, GPU memory held.
            results.cancel_join_thread()
    except Exception:
        results.put({"error": traceback.format_exc()})





def ema_checkpoint(ck, ema):
    """Checkpoint `ck` with the running average's weights in place of the network's (no optimizer state)."""
    net = {k: (ema[k].to(v.dtype) if k in ema else v) for k, v in ck["net"].items()}
    return {**{k: v for k, v in ck.items() if k != "opt"}, "net": net, "ema": True}


def freeze_others(net, opt):
    """--train-only: every parameter outside the optimizer stops taking gradients (set before the compiled passes are built).
    Otherwise their gradients pile up across minibatches, which CUDA graphs refuse (2026-10-31), and cost a backward pass."""
    kept = {id(q) for g in opt.param_groups for q in g["params"]}
    for q in net.parameters():
        if id(q) not in kept:
            q.requires_grad_(False)

def optimizer_groups(net, args):
    """The optimizer's parameter groups: --head-only the fight-outcome head alone; --lookup-lr-mult the lookup's layers in a
    group of their own whose learning rate is scaled ("lr_mult": the update loop sets each group's), --lookup-pick-lr-mult its
    pick weights in a third; else one group. --train-only keeps only the parameters it names (empty groups dropped)."""
    if args.head_only:
        return [{"params": list(net.fight_outcome_head.parameters())}]
    if args.train_only:
        prefixes = tuple(x for x in args.train_only.split(",") if x)
        allowed = {id(q) for n, q in net.named_parameters() if n.startswith(prefixes)}
        groups = optimizer_groups(net, argparse.Namespace(**{**vars(args), "train_only": None}))
        return [g2 for g2 in ({**g, "params": [q for q in g["params"] if id(q) in allowed]} for g in groups) if g2["params"]]
    pick_mult = args.lookup_pick_lr_mult
    if args.lookup_lr_mult != 1.0 or pick_mult is not None:
        # (each group's own learning-rate multiple; the pick weights with the lookup's layers unless given their own)
        pick = {id(q) for q in net.lookup_pick.parameters()} if pick_mult is not None else set()
        fast = {id(q) for m in (net.lookup_opt, net.lookup_run, net.lookup_pick) for q in m.parameters()} - pick
        groups = [{"params": [q for q in net.parameters() if id(q) not in fast | pick]},
                  {"params": [q for q in net.parameters() if id(q) in fast], "lr_mult": args.lookup_lr_mult}]
        if pick:
            groups.append({"params": [q for q in net.parameters() if id(q) in pick], "lr_mult": pick_mult})
        return groups
    return [{"params": list(net.parameters())}]

def head_update(step, opt, n):
    """--head-only's update: `n` optimizer steps on the fight-outcome head's loss (`step()`: the loss and its stats) and
    nothing else; PPO's stats as zeros (the policy and value don't train)."""
    stats = {k: 0.0 for k in ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "grad_clipped")}
    if step is None:
        return stats
    n, sums = max(n, 1), {}
    for _ in range(n):
        loss, st = step()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        for k, v in st.items():
            sums[k] = sums.get(k, 0.0) + v
    return {**stats, **{k: v / n for k, v in sums.items()}}

def main(argv=None):
    global W_HP, W_MONSTER
    args = parse_args(argv)
    W_HP, W_MONSTER = args.hp_weight, args.monster_weight
    device = pick_device(args.device)
    cap_gpu_memory(device, args.gpu_gb)
    if not args.overlap:
        cap_gpu_memory(device, args.worker_gpu_gb and (args.gpu_gb or 0) + args.worker_gpu_gb)
    # TF32 for whatever still runs in float32 (most of the network runs in bfloat16).
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp and device.type == "cuda")

    if args.eval:
        net, ck = load(args.eval, device)
        net.load_state_dict(ck["net"])
        fight_path = args.fight_net or ck["args"].get("fight_net")
        fight_net = load_fight_net(fight_path, device) if fight_path else None
        lookup = (Lookup(fight_net, device, amp, draws=ck["args"].get("lookup_draws", 4), every=ck["args"].get("lookup_every", False))
                  if ck["args"].get("lookup") and fight_net is not None else None)
        r = evaluate(net, device, args.eval_runs, amp, args.ascension, max_steps=args.max_steps, character=args.character,
                     fight_net=fight_net, lookup=lookup)
        if fight_net is not None:
            print(f"fights played by {fight_path}")
        print(f"{args.eval}: update {ck['update']}: floor {r['floor']:.1f}, reached A2 {r['reached_act2']:.0%} A3 {r['reached_act3']:.0%} "
              f"A4 {r['reached_act4']:.0%}, won {r['win_rate']:.1%} of {r['runs']}, the Heart {r['win_heart']:.1%} ({r['timed_out']} timed out)")
        return
    if args.lr_end is not None and not args.updates:
        sys.exit("--lr-end needs --updates")
    if args.shaping_end is not None and not args.fade_updates:
        sys.exit("--shaping-end needs --fade-updates")
    if args.init and args.resume:
        sys.exit("pass only one of --init and --resume")
    if args.character != "ironclad":
        # Archetype practice fights (and their eval) and seeded runs are the Ironclad's only so far;
        # another character's practice fights are all its human boss scenarios.
        decks = args.practice_decks is not None
        bad = [f for f, on in (("--seeded (no Defect seeds)", args.seeded and args.character == "defect"),
                                ("--practice-eval without --practice-decks", args.practice_eval and not decks),
                                ("--practice-archetype", args.practice_archetype),
                                ("--practice without --practice-decks, or --scenarios and --scenario-share 1",
                                 args.practice and not decks and not (args.scenarios and args.scenario_share >= 1.0))) if on]
        if bad:
            sys.exit(f"{args.character}: archetype practice fights and seeded runs are the Ironclad's only so far ({', '.join(bad)})")
    if args.search_weight and args.search_top + 1 > SEARCH_CANDS:
        sys.exit(f"--search-top: at most {SEARCH_CANDS - 1} (the search's targets hold {SEARCH_CANDS} first moves)")
    fight_net = None
    if args.fight_net:
        # The split: the fights are the fixed fight network's, so whatever trains fights has nothing to train.
        bad = [f for f, on in (("--practice", args.practice), ("--teacher", args.teacher), ("--search-weight", args.search_weight),
                                ("--next-move", args.next_move), ("--fight-outcome", args.fight_outcome),
                                ("--value-epochs", args.value_epochs),
                                ("--proximal", args.proximal), ("--overlap", args.overlap)) if on]
        if bad:
            sys.exit(f"--fight-net: the fights are the fight network's, held fixed; drop {', '.join(bad)}")
        fight_net = load_fight_net(args.fight_net, device)
        if args.lookup and fight_net.fight_outcome_head is None:
            sys.exit(f"--lookup: {args.fight_net} has no fight-outcome head (train_runs.py --fight-outcome)")
        print(f"fight network: {args.fight_net} (held fixed); this network trains on the choices between fights only "
              f"(returns from choice to choice, lambda {args.choice_lam})")
    deck_net = None
    if args.deck_net:
        # The split the other way round: the choices are the fixed deck network's, so whatever trains them has nothing to train.
        bad = [f for f, on in (("--fight-net", args.fight_net), ("--lookup without --lookup-net", args.lookup and not args.lookup_net),
                                ("--human-data", args.human_data), ("--seeded", args.seeded), ("--overlap", args.overlap),
                                ("--proximal", args.proximal)) if on]
        if bad:
            sys.exit(f"--deck-net: the choices between fights are that network's, held fixed; drop {', '.join(bad)}")
        deck_net = load_fight_net(args.deck_net, device)
        print(f"deck network: {args.deck_net} (held fixed); this network trains on the fights only (returns from fight move to "
              f"fight move, lambda {args.lam})")
    late_lines = None
    if args.late_starts:
        if not 0 <= args.late_share <= 1:
            sys.exit(f"--late-share: a share in 0..1, not {args.late_share}")
        late_lines, late_about = late_start_pool(args)
        print(late_about)
        if args.seeded:
            print("warning: --seeded with --late-starts: seeded_* in log.csv lumps the two kinds of run together")

    run_dir = os.path.join(RUNS, args.name)
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "latest.pt")
    update, total_steps, total_runs, elapsed = 0, 0, 0, 0.0
    if args.resume:
        net, ck = load(latest, device)
        net.load_state_dict(ck["net"])
        update, total_steps, total_runs, elapsed = ck["update"], ck["steps"], ck["runs"], ck["seconds"]
    elif os.path.exists(latest):
        sys.exit(f"{run_dir} already has a run: pass --resume, or pick another --name")
    elif args.init:
        init = torch.load(args.init, map_location=device, weights_only=False)
        if "runs" in init:  # a run checkpoint
            net, init = load(args.init, device)
            net.load_state_dict(init["net"])
        else:
            net = from_fight_checkpoint(args.init, device)
        print(f"starting from {args.init}")
        if args.grow_layers and args.grow_layers > len(net.encoder.layers):
            net = grow_layers(net, args.grow_layers, device)
    else:
        net = RunNet(d=args.d, layers=args.layers, heads=args.heads, unblended=args.unblended).to(device)
    if args.human_outcome and net.outcome is None:  # (a network from before gains the head, fresh)
        net.outcome = OutcomeHead(net.d).to(device)
    if args.next_move and net.move_head is None:  # (the same for the next-move head)
        net.move_head = torch.nn.Linear(net.d, MOVES_PER_MONSTER).to(device)
    if args.fight_outcome and net.fight_outcome_head is None:  # (and the fight-outcome head)
        net.fight_outcome_head = FightOutcomeHead(net.d).to(device)
    # Fused on CUDA: one kernel for all parameters (about 5% faster updates).
    # --head-only: only the fight-outcome head learns (the policy and value play and score exactly as they did).
    if args.head_only and net.fight_outcome_head is None:
        sys.exit("--head-only: the network has no fight-outcome head (add --fight-outcome)")
    opt = torch.optim.Adam(optimizer_groups(net, args), lr=args.lr, eps=1e-5, fused=device.type == "cuda")
    if args.train_only:
        freeze_others(net, opt)
    # --compile: the rollout and PPO passes go through torch.compile (same parameters; checkpoints and
    # evals use the plain module).
    fwd = torch.compile(net, dynamic=False, **({"backend": args.compile_backend} if args.compile_backend else {})) if args.compile else net
    # --fixed-shapes: fight rows through the fixed passes (compiled, CUDA graphs), the rest through `fwd`.
    fixed = FixedShapes(net, cuda_graphs=device.type == "cuda") if args.fixed_shapes else None
    if args.resume:
        saved = ck["opt"]
        if len(saved["param_groups"]) != len(opt.param_groups):
            sys.exit(f"--resume: the run's optimizer had {len(saved['param_groups'])} parameter groups, now {len(opt.param_groups)} "
                     "(--lookup-lr-mult changed): start a new run")
        n_now = len(opt.param_groups[0]["params"])
        if len(saved["param_groups"][0]["params"]) < n_now:
            # The network gained parameters since (the practice value head, registered last):
            # their Adam state starts fresh.
            saved["param_groups"][0]["params"] = list(range(n_now))
        opt.load_state_dict(saved)
        for q in net.parameters():
            st = opt.state.get(q, {})
            for k in ("exp_avg", "exp_avg_sq"):
                if k in st and st[k].shape != q.shape:
                    # An embedding grew on load (fight_model.grow_embeddings): new rows start fresh.
                    grown = torch.zeros_like(q)
                    grown[:st[k].shape[0]] = st[k]
                    st[k] = grown
        print(f"resumed {latest}: update {update}, {total_steps:,} steps")
    # The network's shape, for loading its checkpoints.
    args.d, args.layers, args.heads = net.d, len(net.encoder.layers), net.encoder.layers[0].self_attn.num_heads
    args.unblended = net.unblended
    args.outcome = net.outcome is not None
    args.move_head = net.move_head is not None
    args.fight_outcome_head = net.fight_outcome_head is not None
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    spec = live_chart.run_spec(vars(args))
    live_chart.write(run_dir, spec)
    if args.port:
        url = live_chart.serve(RUNS, args.port)
        if url:
            print(f"live chart: {url}/{args.name}/live.html  (or ssh -L {args.port}:localhost:{args.port} <this machine> "
                  f"and http://localhost:{args.port}/{args.name}/live.html)")
    print(f"device {device}, {sum(p.numel() for p in net.parameters()):,} parameters, "
          f"{args.envs} runs x {args.steps} steps per update, run folder {run_dir}")

    torch.manual_seed(args.seed + update)
    teacher = None
    if args.teacher:
        teacher, t_ck = load(args.teacher, device)
        teacher.load_state_dict(t_ck["net"])
        teacher.eval()
        for q in teacher.parameters():
            q.requires_grad_(False)
        print(f"teacher: {args.teacher} (update {t_ck['update']}), weight {args.teacher_weight}, value weight {args.teacher_value}, "
              f"on {'practice-fight' if args.teacher_moves == 'practice' else 'all'} moves"
              f"{f', fading to nothing at update {args.teacher_fade}' if args.teacher_fade else ''}")
    human_aux = None
    if args.human_data:
        # Imitation of human choices (card rewards by default) next to PPO: keeps choices the
        # policy has written off (engine cards) alive while play decides whether they pay.
        from human.human_data import KIND_NAMES, Human
        human = Human(os.path.join(args.human_data, "train"), np.random.default_rng(args.seed + update))
        human_kinds = [KIND_NAMES.index(k) for k in args.human_kinds.split(",")]
        # --human-kind-weight: a weight per kind on top of --human-weight (1 for the rest).
        kind_w = np.ones(len(KIND_NAMES), dtype=np.float32)
        for kv in (args.human_kind_weight or "").split(","):
            if kv:
                k, v = kv.split("=")
                kind_w[KIND_NAMES.index(k)] = float(v)
        if set(human_kinds) != set(range(len(KIND_NAMES))) and human.kinds_bytes(human_kinds) < (1 << 30):
            # Only the kinds imitated, in memory (the rest were read and computed for no loss). Not a subset of a gigabyte or
            # more (2026-10-31: the A20 Ironclad dataset's five kinds, 3.3 GB, swapped the training machine to a standstill): those stream from
            # the files with the rest, the other kinds weighted to nothing (`keep`).
            human.select_kinds(human_kinds)
        else:
            human.start_prefetch(args.human_batch)

        human_calls = [0]

        def human_aux():
            human_calls[0] += 1
            if (human_calls[0] - 1) % args.human_every:
                return torch.zeros((), device=device), {}
            ints_h, floats_h, mask_h, slot_h, _, kind_h, _ = human.next(args.human_batch, device)
            keep = torch.from_numpy(np.isin(kind_h, human_kinds)).to(device)
            w_h = torch.from_numpy(kind_w[kind_h]).to(device)
            with amp():
                logits_h, _ = net(ints_h, floats_h)
            logp_h = F.log_softmax(masked(logits_h.float(), mask_h), -1)
            nll = -logp_h.gather(1, slot_h[:, None]).squeeze(1)
            n = keep.sum().clamp(min=1)
            if args.human_floor > 0:
                # Keep-alive: no pull once the human's choice has at least --human-floor.
                loss_h = (F.relu(nll - float(-np.log(args.human_floor))) * keep * w_h).sum() / n
            else:
                loss_h = (nll * keep * w_h).sum() / n
            acc = ((logp_h.argmax(-1) == slot_h) & keep).sum() / n
            card = keep & (torch.from_numpy(kind_h).to(device) == KIND_NAMES.index("card"))
            acc_card = ((logp_h.argmax(-1) == slot_h) & card).sum() / card.sum().clamp(min=1)
            return args.human_every * args.human_weight * loss_h, {"human_loss": loss_h.item(), "human_acc": acc.item(),
                                                                   "human_acc_card": acc_card.item()}

        print(f"human imitation: {human.n:,} decisions from {args.human_data}, kinds {args.human_kinds}, weight {args.human_weight}"
              + (f", per kind {args.human_kind_weight}" if args.human_kind_weight else "")
              + (f", floor {args.human_floor}" if args.human_floor > 0 else " (full cross-entropy)"))
    outcome_aux = None
    if args.human_outcome:
        # The human-outcome head: how human runs went from positions between fights, a small loss
        # beside PPO's that teaches the shared network what makes a run strong (never the PPO value).
        from human.human_data import Human
        odata = Human(os.path.join(args.human_outcome_data or args.human_data, "train"), np.random.default_rng(args.seed + update + 1))
        if odata.outcome is None:
            sys.exit(f"{args.human_outcome_data or args.human_data}: no outcome arrays (human_dataset.py from 2026-10-27)")
        odata.start_prefetch(args.human_outcome_batch or args.human_batch)
        outcome_calls = [0]

        def outcome_aux():
            outcome_calls[0] += 1
            if (outcome_calls[0] - 1) % args.human_every:
                return torch.zeros((), device=device), {}
            b = odata.next(args.human_outcome_batch or args.human_batch, device)
            with amp():
                pred = net.outcome_pred(b[0], b[1])
            loss_o, st = outcome_loss(pred, b[6])
            return args.human_every * args.human_outcome * loss_o, {"outcome_loss": loss_o.item(), **st}

        print(f"human outcome: {odata.n:,} positions from {args.human_outcome_data or args.human_data}, weight {args.human_outcome}")
    scenario_eval = read_scenarios(os.path.expanduser(args.scenario_eval), args.scenario_acts) if args.scenario_eval else []
    N, T = args.envs, args.steps
    first_update = update
    scaler_state = None
    if args.resume and "reward_scale" in ck:
        scaler_state = ck["reward_scale"]
    elif args.init and "reward_scale" in init:
        # Same rewards as the run it starts from: keep its scale, so its value still fits.
        scaler_state = init["reward_scale"]
    elif teacher is not None and args.teacher_value and "reward_scale" in t_ck:
        # A new network pulled towards the teacher's value: the teacher's scale, so the value's two
        # targets (the returns and the teacher's) agree.
        scaler_state = t_ck["reward_scale"]

    def lr_at(u):
        if args.lr_end is None:
            return args.lr
        return args.lr + (args.lr_end - args.lr) * min(u / args.updates, 1.0)

    recent = deque(maxlen=500)
    recent_dt = deque(maxlen=10)
    recent_practice = deque(maxlen=1000)
    recent_seeded = deque(maxlen=500)
    last_save = time.time()
    eval_ascensions = [int(a) for a in args.eval_ascensions.split(",")] if args.eval_ascensions else [args.ascension]

    # The rollout's buffers on the GPU, for training. With --overlap a worker process plays the next
    # rollout (into shared memory, copied here when it's done) while this one trains on the last:
    # the games and the network's moves on one side, PPO on the other, each with its own Python
    # lock. Each rollout is then one update behind the network trained on it (PPO's ratio and KL
    # stop allow for that). (CUDA memory can't be shared between processes under WSL, hence the
    # copy: about 3.4 GB at 2,048 games, a fraction of a second.)
    buf = rollout_buffers(T, N, device)
    collector = worker = None
    if args.overlap:
        shared = rollout_buffers(T, N, "cpu", shared=True)
        # (Not page-locked: under WSL, Windows counts page-locked memory as shared GPU memory.)
        ctx = torch.multiprocessing.get_context("spawn")
        commands, results = ctx.Queue(), ctx.Queue()
        worker = ctx.Process(target=rollout_worker, args=(args, device.type, first_update, scaler_state, shared, commands, results, late_lines),
                             daemon=True)
        worker.start()

        def weights():
            return {k: v.detach().to("cpu", copy=True) for k, v in net.state_dict().items()}

        commands.put((update, weights()))
    else:
        collector = Collector(args, device, amp, first_update, scaler_state, late_lines=late_lines)
    late_lines = None  # (the games hold their own)
    rollout_policy = fixed or fwd
    if deck_net is not None:
        deck_fixed = FixedShapes(deck_net, cuda_graphs=device.type == "cuda") if fixed is not None else deck_net
        rollout_policy = SplitPolicy(rollout_policy, deck_fixed, trains="fight")
        if args.lookup:
            # The deck network's lookup as it trained, from a fixed fight network with the outcome head (--lookup-net),
            # not the one training here; at events and Neow only if the deck network trained with them there.
            lookup_net = load_fight_net(args.lookup_net, device)
            deck_state = torch.load(args.deck_net, map_location="cpu", weights_only=False)["net"]
            collector.lookup = Lookup(lookup_net, device, amp, draws=args.lookup_draws, seed=args.seed, every=args.lookup_every,
                                      events=trained_events(deck_state))
    if fight_net is not None:
        fight_fixed = FixedShapes(fight_net, cuda_graphs=device.type == "cuda") if fixed is not None else fight_net
        rollout_policy = SplitPolicy(fight_fixed, rollout_policy)
        if args.lookup:
            # (the fight network's compiled passes for the head, with --fixed-shapes)
            collector.lookup = Lookup(fight_net, device, amp, draws=args.lookup_draws, seed=args.seed, every=args.lookup_every,
                                      fixed=fight_fixed if isinstance(fight_fixed, FixedShapes) else None)
        if args.search_play:
            scale_ck = torch.load(args.search_scale_from or args.fight_net, map_location="cpu", weights_only=False)
            collector.search_net = (fight_net, fight_fixed if isinstance(fight_fixed, FixedShapes) else None, reward_std(scale_ck),
                                    scale_ck["args"].get("practice_potion_bonus", 0.0))
    reward_scale = scaler_state or (1e-4, 0.0, 1.0)
    # --stop-flat: each update's floor (the last 500 runs'), this run's log included on a resume.
    floors = [r["floor"] for r in live_chart._read(os.path.join(run_dir, "log.csv")) if r.get("floor") not in (None, "")] if args.resume else []
    # --ema: the running average of the weights (float32), from the network as it starts (or the run's saved average).
    ema = None
    if args.ema:
        os.makedirs(os.path.join(run_dir, "ema"), exist_ok=True)
        saved_ema = os.path.join(run_dir, "ema", "latest.pt")
        ema_start = torch.load(saved_ema, map_location=device, weights_only=False)["net"] if args.resume and os.path.exists(saved_ema) else net.state_dict()
        ema = {k: v.detach().float().clone() for k, v in ema_start.items() if v.is_floating_point()}
    flat_checks, levelled = 0, False

    while not args.updates or update < args.updates:
        t0 = time.time()
        lr = lr_at(update)
        for g in opt.param_groups:
            g["lr"] = lr * g.get("lr_mult", 1.0)
        if args.overlap:
            ro = results.get()
            if "error" in ro:
                raise RuntimeError("rollout worker failed:\n" + ro["error"])
            if ro.get("update", update) != update:
                print(f"  the rollout for update {update} was played with update {ro['update']}'s weights", flush=True)
            for k in buf:
                buf[k].copy_(shared[k], non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize()
            if not args.updates or update + 1 < args.updates:
                # The next rollout starts now, with the network before this update's training.
                commands.put((update + 1, weights()))
        else:
            ro = collector.collect(update, rollout_policy, buf)
        wait_s = time.time() - t0
        b_ints, b_floats, b_mask, b_act, b_logp, b_val, b_prac = buf["ints"], buf["floats"], buf["mask"], buf["act"], buf["logp"], buf["val"], buf["prac"]
        adv, ret, runs_now, practice_now, seeded_now = buf["adv"], buf["ret"], ro["runs_now"], ro["practice_now"], ro["seeded_now"]
        reward_sum, rollout_s, asc_hi, purge, w_curse, shape = ro["reward_sum"], ro["rollout_s"], ro["asc_hi"], ro["purge"], ro["w_curse"], ro["shape"]
        reward_scale = ro["scaler_state"]
        ent_coef = screen = None

        net.train()
        B = T * N
        ent_coef = None
        if args.ent_choices is not None or args.ent_shop is not None or args.ent_act1 is not None:
            screen = b_ints.reshape(B, -1)[:, SCREEN_AT]
            choices = args.ent if args.ent_choices is None else args.ent_choices
            ent_coef = torch.where(screen == COMBAT, args.ent, choices)
            if args.ent_shop is not None:
                ent_coef = torch.where(screen == SHOP, args.ent_shop, ent_coef)
            if args.ent_act1 is not None:
                act1 = b_floats.reshape(B, -1)[:, RS + 4] < 1.5
                ent_coef = torch.where((screen == CARD_REWARD) & act1, args.ent_act1, ent_coef)
        za_coef = None
        za_choices = args.zero_avoid if args.zero_avoid_choices is None else args.zero_avoid_choices
        za_shop = za_choices if args.zero_avoid_shop is None else args.zero_avoid_shop
        if args.zero_avoid or za_choices or za_shop:
            screen = b_ints.reshape(B, -1)[:, SCREEN_AT]
            za_coef = torch.where(screen == COMBAT, args.zero_avoid, za_choices)
            za_coef = torch.where(screen == SHOP, za_shop, za_coef).float()
        teacher_aux = search_aux = None
        if args.search_weight:
            # The search's targets over its candidates, on the fight moves it searched in this rollout.
            s_slot, s_p = buf["search_slot"].reshape(B, SEARCH_CANDS), buf["search_p"].reshape(B, SEARCH_CANDS)
            s_rows = torch.nonzero(s_slot[:, 0] >= 0).squeeze(1)
            s_ints, s_floats, s_prac, s_mask = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_prac.reshape(B), b_mask.reshape(B, -1)
            log_floor = float(np.log(args.search_floor))

            def search_aux():
                if len(s_rows) == 0:
                    return torch.zeros((), device=device), {}
                r = s_rows[torch.randint(len(s_rows), (min(args.search_batch, len(s_rows)),), device=device)]
                with amp():
                    lg, _ = net(s_ints[r].long(), s_floats[r], s_prac[r])
                slot, p = s_slot[r], s_p[r]
                if args.search_target == "lift":
                    logp = F.log_softmax(masked(lg.float(), s_mask[r]), -1).gather(1, slot[:, :1]).squeeze(1)
                    lift = F.relu(log_floor - logp).mean()
                    agree = (logp >= log_floor).float().mean()
                    return args.search_weight * lift, {"search_loss": lift.item(), "search_agree": agree.item()}
                valid = slot >= 0
                # The policy over the same candidates.
                logq = F.log_softmax(lg.float().gather(1, slot.clamp(min=0)).masked_fill(~valid, -1e9), -1)
                kl = torch.where(valid & (p > 0), p * (p.clamp(min=1e-12).log() - logq), torch.zeros_like(p)).sum(-1).mean()
                agree = (logq.argmax(-1) == p.argmax(-1)).float().mean()
                return args.search_weight * kl, {"search_loss": kl.item(), "search_agree": agree.item()}
        fade = teacher_fade(args, update)
        ppo_teacher = None
        if teacher is not None and fade > 0 and args.teacher_moves == "all":
            # Every move: the teacher's targets for each row, and PPO's own minibatches carry the losses.
            f_ints, f_floats, f_mask, f_prac = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_mask.reshape(B, -1), b_prac.reshape(B)
            t_logp, t_val = teacher_targets(teacher, amp, f_ints, f_floats, f_mask, f_prac, torch.arange(B, device=device))
            ppo_teacher = (t_logp, t_val, fade * args.teacher_weight, fade * args.teacher_value)
        elif teacher is not None and fade > 0:
            # The teacher's move probabilities (and value) on this rollout's practice-fight moves, once;
            # each minibatch then samples some of them for the KL.
            f_ints, f_floats, f_mask, f_prac = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_mask.reshape(B, -1), b_prac.reshape(B)
            rows = torch.nonzero(f_prac & (f_ints[:, SCREEN_AT] == COMBAT)).squeeze(1)
            t_logp, t_val = teacher_targets(teacher, amp, f_ints, f_floats, f_mask, f_prac, rows)

            def teacher_aux():
                if t_logp is None:
                    return torch.zeros((), device=device), {}
                k = torch.randint(len(rows), (min(args.teacher_batch, len(rows)),), device=device)
                kl, v_err, agree = teacher_loss(net, amp, f_ints, f_floats, f_mask, f_prac, rows[k], t_logp[k], t_val[k])
                kl_v, v_err_v, agree_v = torch.stack([kl.detach(), v_err.detach(), agree]).tolist()  # (one wait for the GPU)
                return fade * (args.teacher_weight * kl + args.teacher_value * v_err), \
                    {"teacher_kl": kl_v, "teacher_agree": agree_v, "teacher_value_err": v_err_v}
        move_aux = None
        if args.next_move:
            # The next-move head's labels from this rollout (next_move_targets), on a sample of the fight
            # rows that have one, every --next-move-every minibatches.
            m_lab = next_move_targets(b_ints, b_floats, buf["done"].to(device), buf["next_move"].to(device))
            m_rows = torch.nonzero((m_lab >= 0).any(1)).squeeze(1)
            move_calls = [0]
            if m_rows.numel():
                def move_aux():
                    move_calls[0] += 1
                    if (move_calls[0] - 1) % args.next_move_every:
                        return torch.zeros((), device=device), {}
                    pick = m_rows[torch.randint(m_rows.numel(), (min(args.next_move_batch, m_rows.numel()),), device=device)]
                    with amp():
                        pred = net.next_move_pred(b_ints.reshape(B, -1)[pick].long(), b_floats.reshape(B, -1)[pick]).float()
                    lab = m_lab[pick]
                    has = lab >= 0
                    ce = F.cross_entropy(pred[has], lab[has])
                    acc = (pred.argmax(-1)[has] == lab[has]).float().mean()
                    ce_v, acc_v = torch.stack([ce.detach(), acc]).tolist()  # (one wait for the GPU, not two)
                    return args.next_move_every * args.next_move * ce, {"next_move_loss": ce_v, "next_move_acc": acc_v,
                                                                         "next_move_rows": float(m_rows.numel())}
        fo_aux = None
        if args.fight_outcome:
            # The fight-outcome head on a sample of the fight rows whose fight ended in this rollout, every
            # --fight-outcome-every minibatches (it reads the network detached: only the head learns).
            fo_flat = {k: buf[k].reshape(B).to(device) for k in ("fo_won", "fo_hp", "fo_has")}
            fo_rows = torch.nonzero(fo_flat["fo_has"] & (b_ints.reshape(B, -1)[:, SCREEN_AT] == COMBAT)).squeeze(1)
            fo_calls = [0]
            if fo_rows.numel():
                def fo_step():
                    pick = fo_rows[torch.randint(fo_rows.numel(), (min(args.fight_outcome_batch, fo_rows.numel()),), device=device)]
                    with amp():
                        pred = net.fight_outcome_pred(b_ints.reshape(B, -1)[pick].long(), b_floats.reshape(B, -1)[pick])
                    loss, st = fight_outcome_loss(pred, fo_flat["fo_won"][pick], fo_flat["fo_hp"][pick])
                    vals = torch.stack([v.detach() for v in st.values()]).tolist()  # (one wait for the GPU)
                    return args.fight_outcome * loss, {**dict(zip(st, vals)), "fight_outcome_rows": float(fo_rows.numel())}

                def fo_aux():
                    fo_calls[0] += 1
                    if (fo_calls[0] - 1) % args.fight_outcome_every:
                        return torch.zeros((), device=device), {}
                    loss, st = fo_step()
                    return args.fight_outcome_every * loss, st
        parts = [f for f in (teacher_aux, search_aux, human_aux, outcome_aux, move_aux, fo_aux) if f is not None]
        aux = None
        if parts:
            def aux():
                loss, st = torch.zeros((), device=device), {}
                for f in parts:
                    l2, s2 = f()
                    loss, st = loss + l2, {**st, **s2}
                return loss, st
        # The rows PPO trains on: all of them, or with the split the choices between fights only.
        p_ints, p_floats, p_mask, p_prac = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_mask.reshape(B, -1), b_prac.reshape(B)
        p_act, p_logp, p_adv, p_ret, p_ent, p_za = b_act.reshape(B), b_logp.reshape(B), adv.reshape(B), ret.reshape(B), ent_coef, za_coef
        p_val = b_val.reshape(B)
        rows = None
        if fight_net is not None or deck_net is not None:
            # The split: this network's own rows only (the choices with --fight-net, the fights with --deck-net).
            mine = p_ints[:, SCREEN_AT] != COMBAT if fight_net is not None else trained_fight_rows(p_ints, p_floats, args)
            rows = torch.nonzero(mine).squeeze(1)
            p_ints, p_floats, p_mask, p_prac, p_act, p_logp, p_adv, p_ret, p_val = (
                x[rows] for x in (p_ints, p_floats, p_mask, p_prac, p_act, p_logp, p_adv, p_ret, p_val))
            p_ent = p_ent[rows] if p_ent is not None else None
            p_za = p_za[rows] if p_za is not None else None
        batches = forward = None
        if fixed is not None:
            row_kinds = kinds(p_ints)
            batches, forward = (lambda: minibatches(row_kinds, args.minibatch)), fixed.forward
        def taken_logp():
            """The taken moves' log-probabilities under the network now, without gradients."""
            with torch.no_grad(), amp():  # (train mode kept: no dropout, and the compiled passes guard on it)
                fi, ff, fp = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_prac.reshape(B)  # (ints made long a chunk at a time)
                fm, fa = b_mask.reshape(B, -1), b_act.reshape(B)
                if fixed is not None:
                    return fixed.logp(fi, ff, fp, fm, fa, args.minibatch)
                out = torch.empty(B, device=device)
                for i in range(0, B, args.minibatch):
                    s = slice(i, i + args.minibatch)
                    lg = fwd(fi[s].long(), ff[s], fp[s])[0]
                    out[s] = F.log_softmax(masked(lg.float(), fm[s]), -1).gather(1, fa[s, None]).squeeze(1)
                return out

        # --proximal: the taken moves' log-probabilities under the network at this update's start.
        b_prox, prox_s = None, ""
        if args.proximal:
            t1 = time.time()
            b_prox = taken_logp()
            prox_s = time.time() - t1
        elif args.retry_temp != 1.0 and args.retry_losses:
            # --retry-temp: the policy that played, before the temperature (the rows PPO trains on).
            b_prox = buf["logp_pi"].reshape(B) if rows is None else buf["logp_pi"].reshape(B)[rows]
        t1 = time.time()
        ppo_args = args
        if teacher is not None and fade > 0 and args.teacher_target_kl is not None:
            ppo_args = argparse.Namespace(**{**vars(args), "target_kl": args.teacher_target_kl})
        if args.head_only:
            # --head-only: no PPO pass at all (the policy and value don't move, and a backward pass through the compiled
            # passes would only pile up gradients nothing uses): the head's steps alone, as many as PPO's minibatches
            # would have given it.
            n_steps = args.epochs * max(1, len(p_act) // args.minibatch) // args.fight_outcome_every
            stats, epochs = head_update(fo_step if fo_aux is not None else None, opt, n_steps), args.epochs
        elif rows is not None and len(p_act) < args.minibatch:
            # Too few rows of this network's to train on (--fight-rooms / --fight-acts before the runs reach those fights):
            # no step this update, PPO's stats as zeros.
            stats, epochs = head_update(None, opt, 0), 0
        else:
            stats, epochs = ppo_update(net if fixed is not None else fwd, opt, ppo_args, amp, p_ints, p_floats, p_mask, p_act, p_logp, p_adv, p_ret,
                                       policy=update >= args.value_warmup, ent_coef=p_ent, practice=p_prac, aux=aux,
                                       batches=batches, forward=forward, logp_prox=b_prox, zero_avoid=p_za, teacher=ppo_teacher)
        if device.type == "cuda":
            torch.cuda.synchronize()
        stats["prox_s"], stats["ppo_s"] = prox_s, time.time() - t1
        if args.value_epochs and update >= args.value_warmup:
            # --value-epochs: value-only passes, the policy held by a KL penalty to where PPO's epochs left it.
            t1 = time.time()
            b_keep = taken_logp()
            vstats, vepochs = ppo_update(net if fixed is not None else fwd, opt, argparse.Namespace(**{**vars(args), "epochs": args.value_epochs}),
                                         amp, b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_mask.reshape(B, -1), b_act.reshape(B),
                                         b_logp.reshape(B), adv.reshape(B), ret.reshape(B), policy=False, practice=b_prac.reshape(B),
                                         batches=batches, forward=forward, logp_prox=b_keep, clone=args.value_clone)
            if device.type == "cuda":
                torch.cuda.synchronize()
            stats.update(value_epochs_run=vepochs, value_loss_after=vstats["value_loss"], value_kl=vstats["approx_kl"], value_s=time.time() - t1)
        # The KL by kind of minibatch, the choices' kinds together as "other".
        by_name = {}
        for key in [key for key in stats if key.startswith("approx_kl_") and key[len("approx_kl_"):].isdigit()]:
            by_name.setdefault(kind_name(int(key[len("approx_kl_"):])), []).append(stats.pop(key))
        for name in ("fight", "options", "other"):
            stats[f"approx_kl_{name}"] = float(np.mean(by_name[name])) if name in by_name else ""
        if "stop_kind" in stats:
            stats["stop_kind"] = kind_name(stats["stop_kind"])

        update += 1
        total_steps += B
        total_runs += len(runs_now) + len(seeded_now)
        recent.extend(runs_now)
        recent_practice.extend(practice_now)
        recent_seeded.extend(seeded_now)
        dt = time.time() - t0
        recent_dt.append(dt)
        elapsed += dt
        row = {
            "update": update, "steps": total_steps, "runs": total_runs, "seconds": round(elapsed, 1),
            "update_s": dt, "wait_s": wait_s, "rollout_s": rollout_s,
            # Over the last 10 updates: with --overlap one update can take half a second (PPO stopped
            # early at the KL limit, the next rollout already waiting) and the next twenty.
            "steps_per_s": B * len(recent_dt) / sum(recent_dt),
            **summarize(list(recent)),
            "timed_out": sum(e.get("timed_out", False) for e in runs_now),
            "reward": reward_sum / N,
            "reward_scale": ro["reward_std"],
            "shaping": shape,
            "forced_purge": purge,
            "curse_weight": w_curse,
            "practice": float(b_prac.float().mean()),
            "practice_fights": len(practice_now),
            **practice_summary([e for e in recent_practice if not e.get("retry")]),
            **retry_summary(practice_now, recent_practice),
            "seeded_runs": len(seeded_now),
            "human_weight": args.human_weight if args.human_data else 0.0,
            "ascension_hi": asc_hi if asc_hi is not None else "",
            **{f"win_a{lo}_{hi}": (float(np.mean([e["won"] for e in recent if lo <= e.get("ascension", 0) <= hi]))
                                   if any(lo <= e.get("ascension", 0) <= hi for e in recent) else "")
               for lo, hi in ((0, 5), (6, 10), (11, 15), (16, 20))},
            **{f"seeded_win_{k}": (float(np.mean([e["won"] for e in recent_seeded if e["seeded"] == k]))
                                   if any(e["seeded"] == k for e in recent_seeded) else "") for k in IRONCLAD_SEEDS},
            # Every seed's runs together (any character): won, the Heart beaten, floor.
            "seeded_win": float(np.mean([e["won"] for e in recent_seeded])) if recent_seeded else "",
            "seeded_heart": float(np.mean([e["won"] and e["act"] >= 4 for e in recent_seeded])) if recent_seeded else "",
            "seeded_floor": float(np.mean([e["floor"] for e in recent_seeded])) if recent_seeded else "",
            "seeded_synergy": float(np.mean([e.get("synergy_picks", 0) for e in recent_seeded])) if recent_seeded else "",
            # A seeded run's own archetype in its final deck (the seed's cards included).
            **{f"seeded_deck_{k}": (float(np.mean([e.get("archetype_cards", {}).get(k, 0) for e in recent_seeded if e["seeded"] == k]))
                                    if any(e["seeded"] == k for e in recent_seeded) else "") for k in IRONCLAD_SEEDS},
            **stats,
            "teacher_fade": fade if teacher is not None else "",
            "explained_var": explained_variance(p_ret, p_val),
            "epochs": epochs, "lr": lr,
            **{k: 1000 * ro["habits"][k] / max(ro["habit_rows"], 1) for k in HABIT_FIELDS},
            "searches": ro["searched"]["n"],
            "search_changed": ro["searched"]["changed"] / max(ro["searched"]["n"], 1) if args.search_weight else "",
            "search_played": ro["searched"]["played"] / max(ro["searched"]["n"], 1) if args.search_play else "",
            "search_s": ro["searched"]["s"],
            # Basic Strikes in the deck, averaged over the floors climbed in this rollout.
            "strikes_held": ro["strikes_held"],
            **{f"rollout_{k}_s": v for k, v in ro["phase_s"].items()},
            **{f"lookup_{k}": v for k, v in ro.get("lookup", {}).items()},
            **{f"lookup_agree_{k}": v for k, v in ro.get("lookup_agree", {}).items()},
            # With --overlap the rollout's process is the worker; else the trainer's peak covers both.
            **dict(zip(("gpu_alloc_gb", "gpu_reserved_gb"), gpu_peak_gb(device))),
            **(dict(zip(("worker_gpu_alloc_gb", "worker_gpu_reserved_gb"), ro["gpu_gb"])) if args.overlap else {}),
            "curses_held": ro["curses_held"],
        }
        append_csv(os.path.join(run_dir, "log.csv"), LOG_FIELDS, [row])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        pct = lambda k: f"{row[k]:.0%}" if row[k] != "" else "-"
        gb = lambda k: f"{row[k]:.1f}" if row.get(k, "") != "" else "-"
        floor = f"{row['floor']:.1f}" if row["floor"] != "" else "-"
        print(
            f"upd {update:5d}  {total_steps / 1e6:7.2f}M steps  {row['steps_per_s']:5.0f}/s (rollout {rollout_s:4.1f}s)  "
            f"runs {total_runs}  floor {floor}  reached A2 {pct('reached_act2')} A3 {pct('reached_act3')} "
            + (f"A4 {pct('reached_act4')} won (the Heart) {pct('win_heart')} (any {pct('win_rate')})  " if args.heart_reward
               else f"A4 {pct('reached_act4')} won {pct('win_rate')}  ")
            + f"pg {row['policy_loss']:+.3f} v {row['value_loss']:.3f} ent {row['entropy']:.2f} kl {row['approx_kl']:.3f} "
            f"ev {row['explained_var']:.2f} ep {epochs}" + (f" (stopped {row['stopped_at']:.0%} in, {row['stop_kind']})" if row.get("stopped_at", "") != "" else "")
            + f" lr {lr:.2g}"
            + (f"  practice {row['practice']:.0%} won {pct('practice_win')}" if args.practice else "")
            + (f" HP lost {row['practice_hp_lost']:.1f}" if args.practice and row.get("practice_hp_lost", "") != "" else "")
            + (f"  replays {row['retry_fights']} won {pct('retry_win')} saved {pct('retry_saved')}" if args.retry_losses else "")
            + (f"  fight outcome right {row['fight_outcome_acc']:.0%} brier {row['fight_outcome_brier']:.3f}"
               if args.fight_outcome and row.get("fight_outcome_acc", "") != "" else "")
            + ("  " + " ".join(f"{k} {pct(f'practice_win_{k}')}" for k in DECK_KINDS + [f"act{a}" for a in DECK_ACTS]) if args.practice_decks else "")
            + (f"  seeded won exhaust {pct('seeded_win_exhaust')} block {pct('seeded_win_block')} self-damage {pct('seeded_win_self-damage')}"
               if args.seeded and args.character == "ironclad" else "")
            + ("".join(f"  {name} won {np.mean([e['won'] for e in eps]):.0%} the Heart {np.mean([e['won'] and e['act'] >= 4 for e in eps]):.1%} floor {np.mean([e['floor'] for e in eps]):.1f}"
                       for name, eps in (("seeded", [e for e in recent_seeded if e["seeded"] != "late"]), ("late", [e for e in recent_seeded if e["seeded"] == "late"])) if eps)
               if args.late_starts or (args.seeded and args.character != "ironclad") else "")
            + (f"  search {row['searches']} changed {pct('search_changed')} agree {row.get('search_agree', 0):.0%} "
               f"({row['search_s']:.0f}s)" if args.search_weight else "")
            + (f"  search {row['searches']} played {pct('search_played')} ({row['search_s']:.0f}s)" if args.search_play and not args.search_weight else "")
            + (f"  A0-{asc_hi}: " + " ".join(f"A{lo}-{hi} {pct(f'win_a{lo}_{hi}')}" for lo, hi in ((0, 5), (6, 10), (11, 15), (16, 20))) if asc_hi is not None else "")
            + "  rollout " + " ".join(f"{k} {row[f'rollout_{k}_s']:.1f}" for k in ("to_gpu", "network", "engine", "observe", "glue")) + "s"
            + (f" (lookup probes {row['lookup_probe_s']:.1f} head {row['lookup_head_s']:.1f} write {row['lookup_write_s']:.1f}s, "
               f"{row['lookup_rows'] / 1e3:.0f}k rows)" if row.get("lookup_rows") else "")
            + f"  GPU GB trainer {gb('gpu_alloc_gb')}/{gb('gpu_reserved_gb')}" + (f" worker {gb('worker_gpu_alloc_gb')}/{gb('worker_gpu_reserved_gb')}" if args.overlap else ""),
            flush=True,
        )

        if ema is not None:
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    if k in ema:
                        ema[k].mul_(args.ema).add_(v.float(), alpha=1 - args.ema)
        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "args": vars(args), "update": update,
              "steps": total_steps, "runs": total_runs, "seconds": elapsed,
              "reward_scale": reward_scale}
        if args.save_every and update % args.save_every == 0:
            torch.save(ck, os.path.join(run_dir, f"update_{update}.pt"))
            if ema is not None:
                torch.save(ema_checkpoint(ck, ema), os.path.join(run_dir, "ema", f"update_{update}.pt"))
        if update % 10 == 0 or time.time() - last_save > 120 or update == args.updates:
            torch.save(ck, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
            if ema is not None:
                torch.save(ema_checkpoint(ck, ema), os.path.join(run_dir, "ema", "latest.pt.tmp"))
                os.replace(os.path.join(run_dir, "ema", "latest.pt.tmp"), os.path.join(run_dir, "ema", "latest.pt"))
            last_save = time.time()
        if row["floor"] != "":
            floors.append(row["floor"])
        if args.stop_flat and update % 50 == 0 and len(floors) >= 200:
            now, before = float(np.mean(floors[-100:])), float(np.mean(floors[-200:-100]))
            flat_checks = flat_checks + 1 if now - before < args.stop_flat else 0
            levelled = flat_checks >= 2
            print(f"  levelling off: mean floor {now:.2f} over the last 100 updates, {now - before:+.2f} on the 100 before "
                  f"({flat_checks} of 2 checks under {args.stop_flat})" + (": stopping" if levelled else ""), flush=True)
        if args.eval_every and update % args.eval_every == 0:
            t1 = time.time()
            for k, asc in enumerate(eval_ascensions if args.eval_runs else []):
                t1 = time.time()
                stats = choice_stats.ChoiceStats(min(256, args.eval_runs))
                r = evaluate(net, device, args.eval_runs, amp, asc, max_steps=args.max_steps, stats=stats, character=args.character)
                if k > 0:
                    r["set"] = f"runs_a{asc}"
                append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, [{"update": update, "steps": total_steps, **r}])
                # The first eval ascension's choices go to choices.csv (the live chart's), others to
                # choices_a<N>.csv.
                choices = "choices.csv" if k == 0 else f"choices_a{asc}.csv"
                append_csv(os.path.join(run_dir, choices), CHOICE_FIELDS, [{"update": update, "steps": total_steps, **stats.summary()}])
                print(f"  eval A{asc}: floor {r['floor']:.1f}, reached A2 {r['reached_act2']:.0%} A3 {r['reached_act3']:.0%} A4 {r['reached_act4']:.0%}, "
                      f"won {r['win_rate']:.1%} ({r['timed_out']} timed out, {time.time() - t1:.0f}s)", flush=True)
            if args.practice_archetype:
                # Fixed fights of the archetype (block_eval.py; its habit counts are block's).
                t1 = time.time()
                arch = args.practice_archetype
                br = block_eval.block_eval(net, device, args.practice_eval or 512, amp, args.ascension, archetype=arch)
                append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, [{"update": update, "steps": total_steps, "set": arch, "runs": br["fights"], **br}])
                print(f"  {arch} eval: {block_eval.describe(br)}, {time.time() - t1:.0f}s", flush=True)
            if scenario_eval:
                t1 = time.time()
                sr = evaluate_scenarios(net, device, scenario_eval, amp, character=args.character)
                append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, [{"update": update, "steps": total_steps, **sr}])
                print(f"  boss eval: won {sr['win_rate']:.1%} (" + ", ".join(f"{a} {sr[f'win_{a}']:.0%}" for a in BOSS_KINDS if sr[f"win_{a}"] != "")
                      + (f", first boss {sr['first_boss']:.0%}" if sr["first_boss"] != "" else "") + f"), {time.time() - t1:.0f}s", flush=True)
            if args.practice_eval and (args.practice_decks or not scenario_eval or args.archetype_eval):
                t1 = time.time()
                pr = evaluate_practice(net, device, args.practice_eval, amp, args.ascension, character=args.character, decks=args.practice_decks)
                append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, [{"update": update, "steps": total_steps, **pr}])
                groups = DECK_KINDS + [f"act{a}" for a in DECK_ACTS] if args.practice_decks else ARCHETYPES
                print(f"  practice eval: won {pr['win_rate']:.1%} (" + ", ".join(
                    f"{a} {pr[f'win_{a}']:.0%}" for a in groups if pr.get(f"win_{a}", "") != "") + f"), {time.time() - t1:.0f}s", flush=True)
        live_chart.write(run_dir, spec)
        if levelled:
            # Checks fall on --save-every's multiples of 50 too, so update_<n>.pt is this update.
            torch.save(ck, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
            print(f"levelled off at update {update} ({total_steps / 1e6:.1f}M steps): stopped", flush=True)
            break


if __name__ == "__main__":
    main()
