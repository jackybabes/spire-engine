"""Evaluates a training run's checkpoints on the CPU, beside the trainer, so training never waits
for its evals (jack, 2026-10-24). The trainer runs with --eval-every 0 and saves update_N.pt every
--save-every updates; this process picks up each new one, plays the evals greedily and appends
their rows to the run's eval.csv (and choices*.csv), which the live page reads.

    .venv/bin/python python/training/eval_worker.py python/runs/defect-fights-0 > python/runs/defect-fights-0/eval_worker.out 2>&1 &

It keeps out of the trainer's way: it never touches the GPU, runs at a lower priority (--nice) and
on a few CPU threads (--threads, for PyTorch and the engine alike). If it falls behind, it skips to
the newest checkpoint. The evals are smaller than the trainer's own were (2026-10-24: the training machine's CPU
plays about 1,200 positions a second on 2 threads, so the Silent's 1,536 runs and 900 fights would
take 20 minutes), and since 2026-10-25 every 50 updates by default (the run must save checkpoints
on them) with half the fights:
- practice fights (runs with --practice-decks): --practice-fights of each deck kind in each act,
  at the run's --ascension (A0 for defect-fights-0), and with --practice-boss the same number at each act's
  boss (the Heart in Act 4), A0-20 drawn per fight (set "practice_boss");
- whole runs (runs that play them): --runs at each eval ascension (the run's --eval-ascensions);
- the held-out human fights (the run's --scenario-eval): --scenario-fights of them (256), spread
  evenly over the file.
Each row records how long its eval took (eval_s). Precise numbers (1,024 runs) are for chosen
checkpoints afterwards (`train_runs.py --eval`).
"""
import argparse
import json
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--threads", type=int, default=2, help="CPU threads for the network and the engine")
    p.add_argument("--nice", type=int, default=10, help="lower this process's priority by this much")
    p.add_argument("--every", type=int, default=None,
                   help="evaluate checkpoints whose update is a multiple of this (default 50, or the run's --save-every if that is larger; "
                        "a multiple of --save-every)")
    p.add_argument("--practice-fights", type=int, default=None,
                   help="practice fights of each deck kind in each act (default 16 for a --practice-decks run, else none)")
    p.add_argument("--runs", type=int, default=None, help="whole runs at each eval ascension (default 128 if the run plays runs, else none)")
    p.add_argument("--scenario-fights", type=int, default=256,
                   help="held-out human fights to play, spread evenly over the run's --scenario-eval (every other one of the 512; 0: all)")
    p.add_argument("--poll", type=float, default=30.0, help="seconds between looks for a new checkpoint")
    p.add_argument("--once", action="store_true", help="evaluate the newest checkpoint not yet evaluated, then stop")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    # Before torch and the engine start their thread pools: CPU only, a few threads.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["RAYON_NUM_THREADS"] = str(a.threads)
    os.environ["OMP_NUM_THREADS"] = str(a.threads)
    if a.nice:
        os.nice(a.nice)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

    import contextlib
    import csv
    import json
    import re
    import time

    import torch

    import training.choice_stats as choice_stats
    from training.train_fights import append_csv
    from training.train_runs import (CHOICE_FIELDS, DECK_ACTS, EVAL_FIELDS, evaluate, evaluate_practice, evaluate_scenarios, load,
                                     load_fight_net, read_scenarios)

    torch.set_num_threads(a.threads)
    device, amp = torch.device("cpu"), contextlib.nullcontext
    from core.lookup import Lookup
    run_dir = a.run_dir
    with open(os.path.join(run_dir, "args.json")) as f:
        args = json.load(f)
    # Every 50 updates, with half the fights (jack, 2026-10-25: every 10 took 130-300 s against
    # 70-110 s of training, so it fell behind and evaluated every 16-25).
    every = a.every or max(50, args.get("save_every", 25) or 25)
    decks = args.get("practice_decks")
    character = args.get("character", "ironclad")
    plays_runs = args.get("practice", 0.0) < 1.0
    practice_fights = a.practice_fights if a.practice_fights is not None else (16 if decks else 0)
    runs = a.runs if a.runs is not None else (128 if plays_runs else 0)
    ascensions = [int(x) for x in str(args.get("eval_ascensions") or args.get("ascension", 0)).split(",")]
    scenarios = []
    if args.get("scenario_eval"):
        scenarios = read_scenarios(os.path.expanduser(args["scenario_eval"]), args.get("scenario_acts"))
        if a.scenario_fights and a.scenario_fights < len(scenarios):
            # An even spread: the file is in act and ascension order (sample_scenarios.py).
            step = len(scenarios) / a.scenario_fights
            scenarios = [scenarios[int(i * step)] for i in range(a.scenario_fights)]
    fight_net = None
    if args.get("fight_net"):
        # The split (train_runs.py --fight-net): runs are played with the fixed fight network fighting; the
        # practice and human fights would measure only that network, the same at every update.
        fight_net = load_fight_net(args["fight_net"], device)
        practice_fights, scenarios = 0, []
        print(f"fights played by {args['fight_net']} (held fixed): runs only", flush=True)
    # train_runs.py --deck-net: runs are played with that network choosing between fights and the evaluated one fighting (its
    # practice and human fights are evaluated as usual).
    deck_net = load_fight_net(args["deck_net"], device) if args.get("deck_net") else None
    if deck_net is not None:
        print(f"runs: {args['deck_net']} choosing between fights (held fixed), the evaluated network fighting", flush=True)
    # train_runs.py --lookup: the deck network sees the lookup at card rewards here too.
    lookup = (Lookup(fight_net, device, amp, draws=args.get("lookup_draws", 4), every=args.get("lookup_every", False))
              if args.get("lookup") and fight_net is not None else None)
    print(f"eval worker for {run_dir}: every {every} updates on {a.threads} CPU threads; practice {practice_fights} per kind and act, "
          f"runs {runs} at A{','.join(map(str, ascensions))}, human fights {len(scenarios)}", flush=True)

    eval_csv = os.path.join(run_dir, "eval.csv")
    done = set()
    if os.path.exists(eval_csv):
        with open(eval_csv, newline="") as f:
            done = {int(r["update"]) for r in csv.DictReader(f) if r.get("update")}
    last = max(done, default=-1)

    def updates():
        """The run's --updates, read again each time: a run resumed with more updates rewrites args.json
        (2026-10-27: read once, the worker stopped at watcher-fights-1's first budget, 300 of 600)."""
        try:
            with open(os.path.join(run_dir, "args.json")) as f:
                return json.load(f).get("updates")
        except (OSError, ValueError):
            return args.get("updates")

    def trainer_stopped():
        """latest.pt untouched for 5 minutes (the trainer saves it every 10 updates): a run resumed with more updates
        rewrites args.json only once it starts, and a worker started a moment before it took the old budget's end
        for the run's (2026-10-28)."""
        p = os.path.join(run_dir, "latest.pt")
        return not os.path.exists(p) or time.time() - os.path.getmtime(p) > 300

    def pending():
        found = []
        for name in os.listdir(run_dir):
            m = re.fullmatch(r"update_(\d+)\.pt", name)
            if m and int(m.group(1)) > last and int(m.group(1)) % every == 0:
                path = os.path.join(run_dir, name)
                # The trainer may still be writing it.
                if time.time() - os.path.getmtime(path) > 10:
                    found.append((int(m.group(1)), path))
        return sorted(found)

    while True:
        todo = pending()
        if not todo:
            if a.once or (updates() and last >= updates() and trainer_stopped()):
                return
            time.sleep(a.poll)
            continue
        update, path = todo[-1]
        if len(todo) > 1:
            print(f"skipping updates {', '.join(str(u) for u, _ in todo[:-1])} (behind)", flush=True)
        try:
            net, ck = load(path, device)
            net.load_state_dict(ck["net"])
        except Exception as e:  # half written: try again next time
            print(f"update {update}: couldn't load ({e}); trying again", flush=True)
            time.sleep(a.poll)
            continue
        net.eval()
        base = {"update": update, "steps": ck.get("steps", "")}
        t0 = time.time()
        if practice_fights:
            t1 = time.time()
            r = evaluate_practice(net, device, practice_fights, amp, args.get("ascension", 0), character=character, decks=decks)
            append_csv(eval_csv, EVAL_FIELDS, [{**base, **r, "eval_s": time.time() - t1}])
            lost = f", {r['hp_lost']:.1f} HP lost a fight won" if r.get("hp_lost", "") != "" else ""
            print(f"update {update}: practice won {r['win_rate']:.1%} of {r['runs']}{lost} ({time.time() - t1:.0f}s)", flush=True)
        if practice_fights and decks and args.get("practice_boss"):
            # A run training on more bosses (--practice-boss): the same fights, every one at the act's boss.
            t1 = time.time()
            r = evaluate_practice(net, device, practice_fights, amp, character=character, decks=decks, bosses=True)
            append_csv(eval_csv, EVAL_FIELDS, [{**base, **r, "eval_s": time.time() - t1}])
            print(f"update {update}: practice bosses won {r['win_rate']:.1%} of {r['runs']} (" +
                  ", ".join(f"Act {a} {r.get(f'win_act{a}', 0):.0%}" for a in DECK_ACTS) + f", {time.time() - t1:.0f}s)", flush=True)
        for k, asc in enumerate(ascensions if runs else []):
            t1 = time.time()
            stats = choice_stats.ChoiceStats(min(256, runs))
            if deck_net is not None:
                r = evaluate(deck_net, device, runs, amp, asc, max_steps=args.get("max_steps", 8000), stats=stats, character=character,
                             fight_net=net)
            else:
                r = evaluate(net, device, runs, amp, asc, max_steps=args.get("max_steps", 8000), stats=stats, character=character,
                             fight_net=fight_net, lookup=lookup)
            if k > 0:
                r["set"] = f"runs_a{asc}"
            append_csv(eval_csv, EVAL_FIELDS, [{**base, **r, "eval_s": time.time() - t1}])
            append_csv(os.path.join(run_dir, "choices.csv" if k == 0 else f"choices_a{asc}.csv"), CHOICE_FIELDS, [{**base, **stats.summary()}])
            if args.get("heart_reward"):
                # A win is beating the Heart; the Act 3 boss beaten (a win there without the keys, or on to Act 4) apart.
                act3 = r["win_rate"] - r["win_heart"] + r["reached_act4"]
                print(f"update {update}: A{asc} won (beat the Heart) {r['win_heart']:.1%} of {runs}, Act 4 {r['reached_act4']:.0%}, "
                      f"beat the Act 3 boss {act3:.0%}, floor {r['floor']:.1f}, keys {r['keys']:.1f} ({time.time() - t1:.0f}s)", flush=True)
            else:
                print(f"update {update}: A{asc} won {r['win_rate']:.1%} of {runs}, floor {r['floor']:.1f}, Act 4 {r['reached_act4']:.0%}, "
                      f"the Heart {r['win_heart']:.0%}, keys {r['keys']:.1f} ({time.time() - t1:.0f}s)", flush=True)
        if scenarios:
            t1 = time.time()
            r = evaluate_scenarios(net, device, scenarios, amp, character=character)
            append_csv(eval_csv, EVAL_FIELDS, [{**base, **r, "eval_s": time.time() - t1}])
            kinds = ", ".join(f"{k} {r[f'win_{k}']:.0%}" for k in ("boss1", "boss2", "boss3", "boss4") if r.get(f"win_{k}", "") != "")
            print(f"update {update}: human fights won {r['win_rate']:.1%} of {r['runs']} ({kinds}; {time.time() - t1:.0f}s)", flush=True)
        print(f"update {update}: evaluated in {time.time() - t0:.0f}s", flush=True)
        last = update
        if a.once:
            return


if __name__ == "__main__":
    main()
