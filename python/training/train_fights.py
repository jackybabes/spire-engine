"""PPO on single fights (milestone 4, stage 4): random decks against every encounter.

    .venv/bin/python python/training/train_fights.py --name first            # train, printing a line per update
    .venv/bin/python python/training/train_fights.py --name first --resume   # carry on from the last checkpoint
    .venv/bin/python python/training/train_fights.py --eval python/runs/ironclad-fights-1/latest.pt   # score a checkpoint
    .venv/bin/python python/training/train_fights.py --name second --init python/runs/ironclad-fights-1/update_300.pt \
        --updates 300 --lr-end 0                                    # a new run from trained weights

Each update plays `--envs` fights for `--steps` steps (the env auto-resets finished fights), then
runs `--epochs` passes of clipped PPO over the batch, stopping early when the policy moves too far
(approximate KL past `--target-kl`). Rollouts live on the GPU.

Reward: +1 for a win, plus shaping by a potential each step,
    phi = HP / max HP - 0.3 * (monster HP / monster HP at the fight's start),
so a fight that loses at full monster HP sums to about -1 and a win without HP lost to 1.3.

Everything goes to python/runs/<name>/: log.csv (one row per update), eval.csv (checkpoint scores
on the fixed fight sets of baselines.py), live.html (a chart page that reloads itself) and
checkpoints (latest.pt, plus update_<n>.pt every --save-every updates). While training runs, a
small web server (--port, default 8000) serves python/runs, so the chart can be watched from
another computer: http://<this machine>:8000/<name>/live.html.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import csv
import json
import os
import sys
import time
from collections import deque

import numpy as np
import torch

import training.live_chart as live_chart
from analysis.baselines import ACT1, ACT_OF, evaluate
from core.ppo import explained_variance, gae, ppo_update
from core.fight_model import FLOAT_AT, MAX_MONSTERS, MONSTER_FEATS, FightNet, masked, pick_device, policy_fn
from core.sts_env import ACTIONS, FLOAT_LEN, INT_LEN, FightEnv

RUNS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "runs")
MONSTER_WEIGHT = 0.3

LOG_FIELDS = [
    "update", "steps", "fights", "seconds", "steps_per_s", "win_rate", "win_act1", "win_act2", "win_act3",
    "win_act4", "hp_lost", "turns", "timed_out", "reward", "policy_loss", "value_loss", "entropy", "approx_kl",
    "clip_frac", "explained_var", "epochs", "lr",
]
EVAL_FIELDS = ["update", "steps", "set", "fights", "win_rate", "win_act1", "win_act2", "win_act3", "win_act4", "hp_lost", "turns"]

# Fixed evaluation sets: the same seeds and batch as baselines.py, so scores compare directly.
EVAL_SETS = {
    "random_decks_all_encounters": dict(seed=100_000),
    "starter_deck_act1": dict(seed=200_000, random_decks=False, encounters=ACT1),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="fights", help="run folder under python/runs")
    p.add_argument("--resume", action="store_true", help="carry on from <run>/latest.pt")
    p.add_argument("--init", metavar="CKPT", help="start a new run from this checkpoint's weights (fresh optimizer)")
    p.add_argument("--eval", metavar="CKPT", help="only score this checkpoint on both fight sets")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=1 << 40, help="training fight seeds (far from the eval seeds)")
    p.add_argument("--envs", type=int, default=512)
    p.add_argument("--steps", type=int, default=128, help="steps per env per update")
    p.add_argument("--updates", type=int, default=0, help="stop after this many updates (0 = run until stopped)")
    p.add_argument("--lr", type=float, default=2.5e-4)
    p.add_argument("--lr-end", type=float, default=None,
                   help="decay the learning rate linearly from --lr to this over --updates (default: constant)")
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent", type=float, default=0.01)
    p.add_argument("--vf", type=float, default=0.5, help="value loss weight")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--max-grad", type=float, default=0.5)
    p.add_argument("--no-amp", action="store_true", help="run the network in float32 instead of bfloat16 autocast")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--port", type=int, default=8000, help="serve python/runs on this port while training (0 = off)")
    p.add_argument("--save-every", type=int, default=50, help="keep update_<n>.pt every this many updates")
    p.add_argument("--eval-every", type=int, default=50, help="score on the fixed fights every this many updates (0 = never)")
    p.add_argument("--eval-fights", type=int, default=1024)
    return p.parse_args(argv)


def potential(floats, start_monster_hp):
    """phi per game from the float rows: HP / max HP - 0.3 * monster HP / monster HP at the start."""
    hp, max_hp = floats[:, 0], np.maximum(floats[:, 1], 1.0)
    return hp / max_hp - MONSTER_WEIGHT * monster_part(floats, start_monster_hp)


def monster_hp(floats):
    a, b = FLOAT_AT["monster"]
    return floats[:, a:b].reshape(-1, MAX_MONSTERS, MONSTER_FEATS)[:, :, 0].sum(1)


def monster_part(floats, start_monster_hp):
    return np.clip(monster_hp(floats) / np.maximum(start_monster_hp, 1.0), 0.0, 2.0)


def act_rates(episodes):
    out = {}
    for act in (1, 2, 3, 4):
        won = [e["won"] for e in episodes if ACT_OF.get(e.get("encounter")) == act]
        out[f"win_act{act}"] = float(np.mean(won)) if won else ""
    return out


def score(net, device, fights, update=None, steps=None, sets=EVAL_SETS):
    """Greedy play on the fixed fight sets; returns eval.csv rows."""
    net.eval()
    rows = []
    for name, kwargs in sets.items():
        r = evaluate(policy_fn(net, device), fights, **kwargs)
        row = {"update": update, "steps": steps, "set": name, "fights": r["fights"], "win_rate": r["win_rate"],
               "hp_lost": r["hp_lost"], "turns": r["turns"]}
        for act in (1, 2, 3, 4):
            row[f"win_act{act}"] = r["win_rate_by_act"].get(str(act), "")
        rows.append(row)
    net.train()
    return rows


def append_csv(path, fields, rows):
    """Appends rows; an existing file without some of `fields` is rewritten once with them added
    (as empty in its rows so far), so a resumed run logs what newer code logs."""
    new = not os.path.exists(path)
    if not new:
        with open(path, newline="") as f:
            old = list(csv.DictReader(f))
            f.seek(0)
            have = next(csv.reader(f), [])
        missing = [k for k in fields if k not in have]
        fields = have + missing
        if missing:
            with open(path + ".tmp", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(old)
            os.replace(path + ".tmp", path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        for row in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in row.items()})


def make_net(args, device):
    return FightNet(d=args.d, layers=args.layers, heads=args.heads).to(device)


def eval_only(args, device):
    ck = torch.load(args.eval, map_location=device, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    net = make_net(a, device)
    net.load_state_dict(ck["net"])
    print(f"{args.eval}: update {ck['update']}, {ck['steps']:,} steps")
    for row in score(net, device, args.eval_fights, ck["update"], ck["steps"]):
        acts = "  ".join(f"A{k} {row[f'win_act{k}']:.0%}" for k in (1, 2, 3, 4) if row[f"win_act{k}"] != "")
        print(f"{row['set']:28} win {row['win_rate']:6.1%}  hp lost {row['hp_lost']:5.1f}  turns {row['turns']:4.1f}  [{acts}]")


def main(argv=None):
    args = parse_args(argv)
    device = pick_device(args.device)
    if args.eval:
        return eval_only(args, device)
    if args.lr_end is not None and not args.updates:
        sys.exit("--lr-end needs --updates (the decay runs over that many updates)")
    if args.init and args.resume:
        sys.exit("--init starts a new run; --resume continues one: pass only one")
    init = None
    if args.init:
        init = torch.load(args.init, map_location=device, weights_only=False)
        # The network's shape comes from the checkpoint.
        for k in ("d", "layers", "heads"):
            setattr(args, k, init["args"][k])

    run_dir = os.path.join(RUNS, args.name)
    os.makedirs(run_dir, exist_ok=True)
    latest = os.path.join(run_dir, "latest.pt")
    net = make_net(args, device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    update, total_steps, total_fights, elapsed = 0, 0, 0, 0.0
    if args.resume:
        ck = torch.load(latest, map_location=device, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        update, total_steps, total_fights, elapsed = ck["update"], ck["steps"], ck["fights"], ck["seconds"]
        print(f"resumed {latest}: update {update}, {total_steps:,} steps")
    elif os.path.exists(latest):
        sys.exit(f"{run_dir} already has a run: pass --resume, or pick another --name")
    if init:
        net.load_state_dict(init["net"])
        print(f"starting from {args.init}: update {init['update']}, {init['steps']:,} steps")
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    live_chart.write(run_dir)
    if args.port:
        url = live_chart.serve(RUNS, args.port)
        if url:
            print(f"live chart: {url}/{args.name}/live.html  (or ssh -L {args.port}:localhost:{args.port} <this machine> "
                  f"and http://localhost:{args.port}/{args.name}/live.html)")
    print(f"device {device}, {sum(p.numel() for p in net.parameters()):,} parameters, "
          f"{args.envs} envs x {args.steps} steps per update, run folder {run_dir}")

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=not args.no_amp and device.type == "cuda")

    torch.manual_seed(args.seed + update)
    # A resumed run starts fresh fights (seeds past every earlier update's).
    env = FightEnv(args.envs, seed=args.seed + (update << 24))
    N, T = args.envs, args.steps
    obs = env.observe()
    start_mon = monster_hp(obs["floats"]).astype(np.float32)
    phi = potential(obs["floats"], start_mon)

    b_ints = torch.zeros(T, N, INT_LEN, dtype=torch.int32, device=device)
    b_floats = torch.zeros(T, N, FLOAT_LEN, dtype=torch.float32, device=device)
    b_mask = torch.zeros(T, N, ACTIONS, dtype=torch.bool, device=device)
    b_act = torch.zeros(T, N, dtype=torch.long, device=device)
    b_logp = torch.zeros(T, N, device=device)
    b_val = torch.zeros(T, N, device=device)
    b_rew = torch.zeros(T, N, device=device)
    b_done = torch.zeros(T, N, device=device)

    recent = deque(maxlen=2000)  # finished fights, for the rolling stats
    last_save = time.time()
    def lr_at(u):
        if args.lr_end is None:
            return args.lr
        return args.lr + (args.lr_end - args.lr) * min(u / args.updates, 1.0)

    while not args.updates or update < args.updates:
        t0 = time.time()
        lr = lr_at(update)
        for g in opt.param_groups:
            g["lr"] = lr
        net.eval()
        fights_now = []
        reward_sum = 0.0
        for t in range(T):
            ints = torch.from_numpy(obs["ints"]).to(device, non_blocking=True)
            floats = torch.from_numpy(np.ascontiguousarray(obs["floats"])).to(device, non_blocking=True)
            mask = torch.from_numpy(obs["mask"]).to(device, non_blocking=True)
            with torch.no_grad(), amp():
                logits, value = net(ints.long(), floats)
                logits, value = logits.float(), value.float()
                dist = torch.distributions.Categorical(logits=masked(logits, mask))
                a = dist.sample()
            b_ints[t], b_floats[t], b_mask[t] = ints, floats, mask
            b_act[t], b_logp[t], b_val[t] = a, dist.log_prob(a), value

            status = env.step(a.cpu().numpy())
            obs = env.observe()
            done = status != 0
            new_phi = potential(obs["floats"], start_mon)
            rew = np.zeros(N, dtype=np.float32)
            if done.any():
                for g in np.flatnonzero(done):
                    ep = json.loads(env.last_episode(int(g)))
                    fights_now.append(ep)
                    # The fight is over: its last potential comes from the summary (monsters are
                    # all dead after a win; after a loss, keep the last monster HP seen).
                    mon = 0.0 if ep["won"] else monster_part(b_floats[t, g][None].cpu().numpy(), start_mon[g:g + 1])[0]
                    end_phi = ep["hp"] / max(ep["max_hp"], 1) - MONSTER_WEIGHT * mon
                    rew[g] = (1.0 if ep["won"] else 0.0) + end_phi - phi[g]
                start_mon[done] = monster_hp(obs["floats"][done])
                new_phi[done] = potential(obs["floats"][done], start_mon[done])
            rew[~done] += (new_phi - phi)[~done]
            phi = new_phi
            reward_sum += float(rew.sum())
            b_rew[t] = torch.from_numpy(rew).to(device)
            b_done[t] = torch.from_numpy(done.astype(np.float32)).to(device)

        with torch.no_grad():
            ints = torch.from_numpy(obs["ints"]).to(device)
            floats = torch.from_numpy(np.ascontiguousarray(obs["floats"])).to(device)
            with amp():
                next_val = net(ints.long(), floats)[1].float()
            adv, ret = gae(b_rew, b_val, b_done, next_val, args.gamma, args.lam)
        rollout_s = time.time() - t0

        net.train()
        B = T * N
        f_ints, f_floats, f_mask = b_ints.reshape(B, -1), b_floats.reshape(B, -1), b_mask.reshape(B, -1)
        f_act, f_logp, f_adv, f_ret, f_val = b_act.reshape(B), b_logp.reshape(B), adv.reshape(B), ret.reshape(B), b_val.reshape(B)
        stats, epochs = ppo_update(net, opt, args, amp, f_ints, f_floats, f_mask, f_act, f_logp, f_adv, f_ret)

        update += 1
        total_steps += B
        total_fights += len(fights_now)
        recent.extend(fights_now)
        dt = time.time() - t0
        elapsed += dt
        row = {
            "update": update, "steps": total_steps, "fights": total_fights, "seconds": round(elapsed, 1),
            "steps_per_s": B / dt,
            "win_rate": float(np.mean([e["won"] for e in recent])) if recent else "",
            **act_rates(recent),
            "hp_lost": float(np.mean([e["max_hp"] - e["hp"] for e in recent])) if recent else "",
            "turns": float(np.mean([e["turns"] for e in recent])) if recent else "",
            "timed_out": sum(e.get("timed_out", False) for e in fights_now),
            "reward": reward_sum / max(len(fights_now), 1),
            **stats,
            "explained_var": explained_variance(f_ret, f_val),
            "epochs": epochs, "lr": lr,
        }
        append_csv(os.path.join(run_dir, "log.csv"), LOG_FIELDS, [row])
        acts = " ".join(f"{row[f'win_act{k}']:.0%}" if row[f"win_act{k}"] != "" else "-" for k in (1, 2, 3, 4))
        print(
            f"upd {update:5d}  {total_steps / 1e6:7.2f}M steps  {row['steps_per_s']:6.0f}/s "
            f"(rollout {rollout_s:4.1f}s)  win {row['win_rate']:.1%} [A1-4 {acts}]  hp lost {row['hp_lost']:4.1f}  "
            f"pg {row['policy_loss']:+.3f}  v {row['value_loss']:.3f}  ent {row['entropy']:.2f}  "
            f"kl {row['approx_kl']:.3f}  ev {row['explained_var']:.2f}  ep {epochs}  lr {lr:.2g}",
            flush=True,
        )

        ck = {"net": net.state_dict(), "opt": opt.state_dict(), "args": vars(args), "update": update,
              "steps": total_steps, "fights": total_fights, "seconds": elapsed}
        if args.save_every and update % args.save_every == 0:
            torch.save(ck, os.path.join(run_dir, f"update_{update}.pt"))
        if update % 10 == 0 or time.time() - last_save > 120 or update == args.updates:
            torch.save(ck, latest + ".tmp")
            os.replace(latest + ".tmp", latest)
            last_save = time.time()
        if args.eval_every and update % args.eval_every == 0:
            t1 = time.time()
            rows = score(net, device, args.eval_fights, update, total_steps)
            append_csv(os.path.join(run_dir, "eval.csv"), EVAL_FIELDS, rows)
            for r in rows:
                print(f"  eval {r['set']:28} win {r['win_rate']:6.1%}  hp lost {r['hp_lost']:5.1f}  ({time.time() - t1:.0f}s)")
        live_chart.write(run_dir)


if __name__ == "__main__":
    main()
