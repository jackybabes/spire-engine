"""Supervised training of the run network on human choices, keeping its fights as they were.

    .venv/bin/python python/training/pretrain_human.py --data ~/sts-data/human-won \\
        --init python/runs/ironclad-runs-7/latest.pt --name ironclad-human-1

With a split deck network (an --init from `train_runs.py --fight-net`, or --fight-net) the fights are the fight
network's, held fixed: keep states and evals are played with it fighting, only the states between fights are kept, and
card rewards carry the lookup if the deck network trained with one (core/lookup.py; the human data should then be built
with `human_dataset.py --lookup`).

The human data (`human_dataset.py`) covers choices between fights only: card rewards, rest sites
and their grids, boss relics and shop removals. Fights share the network's encoder and card and
relic embeddings with those choices, so learning the choices alone would change how it fights.
Each step therefore has two parts:
- human: cross-entropy of the player's choice, each sample weighted 1 + asc_weight x ascension
  (players who win at higher ascension are the better teachers);
- keep: on states from the engine (runs played by the starting network), the KL from the starting
  network's policy on every screen the human data doesn't cover (fights, the map, events, shops,
  reward screens, Neow, other grids...) and the squared error to its value on every state, so
  those choices and the value stay where they were. (Keeping only fights was not enough: the
  first try won 22% of its runs instead of 80% after 2,500 steps. The screens share outputs, so
  learning that humans skip 23% of card rewards also taught it to leave reward screens.)

Every --eval-every steps: agreement with held-out human choices (by kind), the keep losses, and
the win rate on the evaluation runs (`train_runs.evaluate`). The keep states and evaluation runs
are the --init checkpoint's character's (or --character's). Checkpoints are in train_runs'
format, so `train_runs.py --init <run dir>/latest.pt` carries on with PPO from them.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders
import training.choice_stats as choice_stats  # noqa: E402
import training.live_chart as live_chart  # noqa: E402
from core.fight_model import masked, pick_device  # noqa: E402
from core.run_model import RUN_INT_AT, OutcomeHead, outcome_loss  # noqa: E402
from core.sts_env import ACTIONS, LAYOUT, RunEnv  # noqa: E402
from training.train_fights import RUNS, append_csv  # noqa: E402
from core.lookup import Lookup  # noqa: E402
from core.split import SplitPolicy  # noqa: E402
from training.train_runs import CHOICE_FIELDS, evaluate, in_combat, load, load_fight_net  # noqa: E402

SCREEN = {name: i + 1 for i, name in enumerate(LAYOUT["screens"])}
# Grid purposes as encode_run.rs numbers them (1 + its match): Smith, Toke, ShopPurge.
HUMAN_GRIDS = (4, 7, 8)

from human.human_data import KIND_NAMES, Human  # noqa: E402
LOG_FIELDS = ["update", "steps", "step", "samples", "passes", "seconds", "human_loss", "human_acc", "keep_kl", "keep_value", "lr",
              "outcome_loss", "outcome_won_acc"]
EVAL_FIELDS = ["update", "steps", "set", "step", "val_acc", *[f"val_acc_{k}" for k in KIND_NAMES], "val_loss", "keep_kl", "keep_value", "val_outcome_loss", "val_outcome_won_acc",
               "runs", "floor", "reached_act2", "reached_act3", "reached_act4", "win_rate"]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--init", required=True, help="run checkpoint to start from (and keep fights like)")
    p.add_argument("--fight-net", default=None,
                   help="the split's fight network (default: the --init checkpoint's, train_runs.py --fight-net): keep states and evals "
                        "are played with it fighting, and only the states between fights are kept")
    p.add_argument("--name", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--keep-batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--asc-weight", type=float, default=0.1)
    p.add_argument("--keep-kl", type=float, default=1.0, help="weight of the fight-policy KL")
    p.add_argument("--keep-value", type=float, default=0.5, help="weight of the value error")
    p.add_argument("--keep-states", type=int, default=150_000, help="engine states collected for the keep losses")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-runs", type=int, default=512)
    p.add_argument("--val-samples", type=int, default=50_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--port", type=int, default=8000, help="serve the live chart (0: don't)")
    p.add_argument("--outcome", type=float, default=0.0,
                   help="weight of the human-outcome head's loss (run_model.OutcomeHead): the network gains the head and learns "
                        "how human runs went from positions between fights")
    p.add_argument("--outcome-data", default=None,
                   help="the outcome head's dataset (every run, wins and losses; human_dataset.py from 2026-10-27; default --data)")
    p.add_argument("--ascension", type=int, default=0,
                   help="the ascension of the runs played for the keep states and the evals (20 for the Watcher's A20-only network)")
    p.add_argument("--character", default=None,
                   help="whose runs to play for the keep states and the evals (default: the --init checkpoint's; the data has its own)")
    return p.parse_args()


@torch.no_grad()
def collect_states(net, device, n, amp, seed, character="ironclad", ascension=0, fight_net=None, lookup=None):
    """Engine states (int16 / float16 / mask) from runs the network plays, sampling its policy. With `fight_net` (the
    split, core.split) that network fights and only the states between fights are kept (the deck network's); with
    `lookup` (core.lookup.Lookup) card rewards carry the lookup, as in training."""
    env = RunEnv(256, seed=seed, character=character, ascension=ascension)
    policy = SplitPolicy(fight_net, net) if fight_net is not None else net
    ints, floats, masks = [], [], []
    got = 0
    net.eval()
    while got < n:
        obs = env.observe()
        if lookup is not None:
            obs = lookup.fill(env.envs, obs)
        keep = ~in_combat(obs["ints"]) if fight_net is not None else np.ones(len(obs["mask"]), dtype=bool)
        ints.append(obs["ints"][keep].astype(np.int16))
        floats.append(np.clip(obs["floats"][keep], -65000, 65000).astype(np.float16))
        masks.append(obs["mask"][keep].copy())
        got += int(keep.sum())
        i = torch.from_numpy(obs["ints"]).to(device).long()
        fl = torch.from_numpy(obs["floats"]).to(device)
        m = torch.from_numpy(obs["mask"]).to(device)
        with torch.no_grad(), amp():
            logits, _ = policy(i, fl)
        acts = torch.distributions.Categorical(logits=masked(logits.float(), m)).sample().cpu().numpy()
        env.step(acts)
    net.train()
    return np.concatenate(ints)[:n], np.concatenate(floats)[:n], np.concatenate(masks)[:n]


def human_loss(net, b, asc_weight, amp):
    ints, floats, mask, slot, asc = b[:5]
    with amp():
        logits, _ = net(ints, floats)
    logp = F.log_softmax(masked(logits.float(), mask), -1)
    nll = -logp.gather(1, slot[:, None]).squeeze(1)
    w = 1.0 + asc_weight * asc
    return (w * nll).sum() / w.sum(), (logp.argmax(-1) == slot)


def human_screen(ints):
    """Rows on a screen the human data teaches: card rewards, rest sites, boss relics, and the
    smith, toke and shop-removal grids."""
    screen = ints[:, RUN_INT_AT["screen"][0]]
    purpose = ints[:, RUN_INT_AT["grid_purpose"][0]]
    grid = (screen == SCREEN["grid"]) & ((purpose == HUMAN_GRIDS[0]) | (purpose == HUMAN_GRIDS[1]) | (purpose == HUMAN_GRIDS[2]))
    return (screen == SCREEN["card_reward"]) | (screen == SCREEN["rest"]) | (screen == SCREEN["boss_relic"]) | grid


def keep_loss(net, teacher, ints, floats, mask, amp):
    with torch.no_grad(), amp():
        t_logits, t_value = teacher(ints, floats)
    with amp():
        logits, value = net(ints, floats)
    keep = ~human_screen(ints)
    kl = torch.zeros((), device=ints.device)
    if keep.any():
        tl = F.log_softmax(masked(t_logits.float()[keep], mask[keep]), -1)
        sl = F.log_softmax(masked(logits.float()[keep], mask[keep]), -1)
        kl = (tl.exp() * (tl - sl)).sum(-1).mean()
    return kl, F.mse_loss(value.float(), t_value.float())


def main():
    a = parse_args()
    device = pick_device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    rng = np.random.default_rng(a.seed)
    torch.manual_seed(a.seed)
    net, ck = load(a.init, device)
    net.load_state_dict(ck["net"])
    if a.outcome and net.outcome is None:
        net.outcome = OutcomeHead(net.d).to(device)
    a.character = a.character or ck["args"].get("character", "ironclad")
    # The split (train_runs.py --fight-net): a deck network's fights are another network's, held fixed; its keep states and
    # evals are played with that network fighting (and the lookup at card rewards, if it trained with one).
    fight_path = a.fight_net or ck["args"].get("fight_net")
    fight_net = load_fight_net(fight_path, device) if fight_path else None
    lookup = (Lookup(fight_net, device, amp, draws=ck["args"].get("lookup_draws", 4), every=ck["args"].get("lookup_every", False))
              if fight_net is not None and ck["args"].get("lookup") else None)
    if fight_net is not None:
        print(f"fights played by {fight_path} (held fixed); lookup {lookup is not None}", flush=True)
    teacher, _ = load(a.init, device)
    teacher.load_state_dict(ck["net"])
    teacher.eval()
    for p_ in teacher.parameters():
        p_.requires_grad_(False)
    run_dir = os.path.join(RUNS, a.name)
    if os.path.exists(os.path.join(run_dir, "latest.pt")):
        sys.exit(f"{run_dir} already has a run; pick another --name")
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(a), f, indent=2)

    data = Human(os.path.join(a.data, "train"), rng)
    held = Human(os.path.join(a.data, "val"), rng)
    if a.outcome:
        odata = Human(os.path.join(a.outcome_data or a.data, "train"), rng)
        oheld = Human(os.path.join(a.outcome_data or a.data, "val"), rng)
        if odata.outcome is None:
            sys.exit(f"{a.outcome_data or a.data}: no outcome arrays (human_dataset.py from 2026-10-27)")
        print(f"outcome head: {odata.n:,} positions ({oheld.n:,} held out), weight {a.outcome}", flush=True)
    n_val = min(a.val_samples, held.n)
    print(f"{data.n:,} human samples for training, {held.n:,} held out ({n_val:,} used); "
          f"collecting {a.keep_states:,} engine states with {a.init}", flush=True)
    t0 = time.time()
    k_ints, k_floats, k_mask = collect_states(teacher, device, a.keep_states, amp, seed=a.seed + 7_000_000, character=a.character,
                                              ascension=a.ascension, fight_net=fight_net, lookup=lookup)
    taught = human_screen(torch.from_numpy(k_ints.astype(np.int64))).numpy()
    print(f"  {in_combat(k_ints).mean():.0%} of them in fights, {taught.mean():.1%} on screens the human data teaches "
          f"({time.time() - t0:.0f} s)", flush=True)
    keep_val = rng.choice(len(k_ints), 8192, replace=False)

    def keep_batch(idx):
        return (torch.from_numpy(k_ints[idx].astype(np.int64)).to(device),
                torch.from_numpy(k_floats[idx].astype(np.float32)).to(device),
                torch.from_numpy(k_mask[idx]).to(device))

    @torch.no_grad()
    def validate():
        net.eval()
        right, kinds, losses = [], [], []
        for s in range(0, n_val, 4096):
            b = held.rows(s, min(s + 4096, n_val), device)
            loss, ok = human_loss(net, b, 0.0, amp)
            right.append(ok.cpu().numpy())
            kinds.append(b[5])
            losses.append(float(loss) * len(ok))
        kl, v = [], []
        for s in range(0, len(keep_val), 2048):
            k1, k2 = keep_loss(net, teacher, *keep_batch(np.sort(keep_val[s:s + 2048])), amp)
            kl.append(float(k1))
            v.append(float(k2))
        net.train()
        right, kinds = np.concatenate(right), np.concatenate(kinds)
        row = {"val_acc": float(right.mean()), "val_loss": sum(losses) / len(right),
               "keep_kl": float(np.mean(kl)), "keep_value": float(np.mean(v))}
        if a.outcome:
            ol, oa, m = [], [], min(a.val_samples, oheld.n)
            for s in range(0, m, 4096):
                b = oheld.rows(s, min(s + 4096, m), device)
                with amp():
                    pred = net.outcome_pred(b[0], b[1])
                loss, st = outcome_loss(pred, b[6])
                ol.append(float(loss) * len(pred))
                oa.append(st["outcome_won_acc"] * len(pred))
            row.update(val_outcome_loss=sum(ol) / m, val_outcome_won_acc=sum(oa) / m)
        for i, k in enumerate(KIND_NAMES):
            row[f"val_acc_{k}"] = float(right[kinds == i].mean()) if (kinds == i).any() else ""
        return row

    def eval_row(step):
        row = {"update": step, "steps": step * a.batch, "set": "pre", "step": step, **validate()}
        t1 = time.time()
        stats = choice_stats.ChoiceStats(min(256, a.eval_runs))
        r = evaluate(net, device, a.eval_runs, amp, ascension=a.ascension, stats=stats, character=a.character, fight_net=fight_net,
                     lookup=lookup)
        row.update({k: r[k] for k in ("runs", "floor", "reached_act2", "reached_act3", "reached_act4", "win_rate")})
        append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, [row])
        append_csv(os.path.join(run_dir, "choices.csv"), CHOICE_FIELDS,
                   [{"update": step, "steps": step * a.batch, **stats.summary()}])
        live_chart.write(run_dir, spec)
        print(f"eval step {step}: human agreement {row['val_acc']:.1%} (" +
              ", ".join(f"{k} {row[f'val_acc_{k}']:.0%}" for k in KIND_NAMES if row[f"val_acc_{k}"] != "") +
              f"), fight KL {row['keep_kl']:.4f}, value err {row['keep_value']:.4f}; "
              f"won {row['win_rate']:.1%} of {row['runs']}, floor {row['floor']:.1f} ({time.time() - t1:.0f} s)", flush=True)

    def save(step):
        args = dict(ck["args"])
        args.update({"pretrain": vars(a), "outcome": net.outcome is not None})
        out = {"net": net.state_dict(), "args": args, "update": 0, "steps": 0, "runs": 0, "seconds": 0.0,
               "pretrain_step": step}
        torch.save(out, os.path.join(run_dir, f"step_{step}.pt"))
        torch.save(out, os.path.join(run_dir, "latest.pt.tmp"))
        os.replace(os.path.join(run_dir, "latest.pt.tmp"), os.path.join(run_dir, "latest.pt"))

    spec = live_chart.pretrain_spec()
    live_chart.write(run_dir, spec)
    if a.port:
        url = live_chart.serve(RUNS, a.port)
        if url:
            print(f"live chart: {url}/{a.name}/live.html", flush=True)
    eval_row(0)
    opt = torch.optim.Adam(net.parameters(), lr=a.lr, eps=1e-5)
    t0, last = time.time(), time.time()
    acc_h, loss_h, kl_h, v_h, out_h, outacc_h = [], [], [], [], [], []
    for step in range(1, a.steps + 1):
        lr = a.lr * min(1.0, step / 500) * (0.1 + 0.9 * (1 - step / a.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        b = data.next(a.batch, device)
        h, ok = human_loss(net, b, a.asc_weight, amp)
        kl, v = keep_loss(net, teacher, *keep_batch(np.sort(rng.choice(len(k_ints), a.keep_batch, replace=False))), amp)
        loss = h + a.keep_kl * kl + a.keep_value * v
        if a.outcome:
            ob = odata.next(a.batch, device)
            with amp():
                pred = net.outcome_pred(ob[0], ob[1])
            o_loss, o_st = outcome_loss(pred, ob[6])
            loss = loss + a.outcome * o_loss
            out_h.append(float(o_loss.detach()))
            outacc_h.append(o_st["outcome_won_acc"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        loss_h.append(float(h.detach()))
        acc_h.append(float(ok.float().mean()))
        kl_h.append(float(kl))
        v_h.append(float(v))
        if step % 200 == 0:
            row = {"update": step, "steps": step * a.batch, "step": step, "samples": step * a.batch, "passes": data.passes, "seconds": time.time() - t0, "human_loss": np.mean(loss_h),
                   "human_acc": np.mean(acc_h), "keep_kl": np.mean(kl_h), "keep_value": np.mean(v_h), "lr": lr,
                   "outcome_loss": np.mean(out_h) if out_h else "", "outcome_won_acc": np.mean(outacc_h) if outacc_h else ""}
            append_csv(os.path.join(run_dir, "log.csv"), LOG_FIELDS, [row])
            live_chart.write(run_dir, spec)
            print(f"step {step:6d}  {row['samples'] / 1e6:6.2f}M samples ({row['passes']:.2f} passes)  human loss {row['human_loss']:.3f} "
                  f"acc {row['human_acc']:.1%}  fight KL {row['keep_kl']:.4f}  value err {row['keep_value']:.4f}  "
                  f"lr {lr:.2g}  {200 / (time.time() - last):.1f} steps/s", flush=True)
            acc_h, loss_h, kl_h, v_h, out_h, outacc_h = [], [], [], [], [], []
            last = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            save(step)
            eval_row(step)
            last = time.time()


if __name__ == "__main__":
    sys.exit(main())
