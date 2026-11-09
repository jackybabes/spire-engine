"""Training and rollout in one process: a benchmark (2026-10-24; branch one-process, merged and deleted the same day).

    .venv/bin/python python/training/bench_one_process.py python/runs/defect-fights-0 [--updates 6] [--modes sequential,thread]

With --overlap the rollout plays in a process of its own, and the two processes take turns on the
GPU (with defect-fights-0's settings a rollout takes 12.7 s beside training against 6.5 s alone, an update
about 12.4 s). Here both run in one process: the rollout in a thread of its own, on its own CUDA
stream with its own copy of the network (refreshed on the GPU at each update, behind an event),
training on the main thread on another stream, each with its own GPU batch (one filling while the
other trains), so their kernels can run at the same time instead of taking turns.

Loads the run's latest.pt and settings and times `--updates` updates (after a first one that
compiles and captures the CUDA graphs) of each mode: `sequential` (the rollout, then the
training: each alone) and `thread` (the next update's rollout beside this one's training, one
update behind as with --overlap), sampling the GPU's use (nvidia-smi) meanwhile. The network
trains as in train_runs.py (fixed shapes, PPO), but nothing is saved.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import argparse
import json
import queue
import subprocess
import threading
import time

import numpy as np
import torch

from core.fixed_shapes import FixedShapes, kinds, minibatches
from core.ppo import ppo_update
from core.run_model import RunNet
from training.train_runs import Collector, cap_gpu_memory, load, parse_args, rollout_buffers

NVIDIA_SMI = "/usr/lib/wsl/lib/nvidia-smi"


class GpuSampler:
    """nvidia-smi every 0.5 s in the background: the GPU's busy share and memory in use."""

    def __init__(self):
        self.rows, self.proc = [], None

    def start(self):
        self.rows = []
        cmd = [NVIDIA_SMI if os.path.exists(NVIDIA_SMI) else "nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
               "--format=csv,noheader,nounits", "-lms", "500"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self.proc.stdout:
            try:
                self.rows.append([float(x) for x in line.split(",")])
            except ValueError:
                pass

    def stop(self):
        self.proc.terminate()
        rows = np.array(self.rows) if self.rows else np.zeros((1, 2))
        return {"gpu_busy": float(rows[:, 0].mean()), "gpu_mem_gb_max": float(rows[:, 1].max()) / 1024}


def run_args(run_dir):
    """The run's settings (args.json) over train_runs' defaults."""
    args = parse_args(["--name", "bench"])
    with open(os.path.join(run_dir, "args.json")) as f:
        vars(args).update(json.load(f))
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir")
    p.add_argument("--updates", type=int, default=6, help="timed updates of each mode (after one to warm up)")
    p.add_argument("--modes", default="sequential,thread")
    p.add_argument("--gpu-gb", type=float, default=14.0, help="the process's GPU memory cap")
    a = p.parse_args()
    args = run_args(a.run_dir)
    device = torch.device("cuda")
    cap_gpu_memory(device, a.gpu_gb)
    torch.backends.cuda.matmul.allow_tf32 = True

    def amp():
        return torch.autocast("cuda", dtype=torch.bfloat16)

    net, ck = load(os.path.join(a.run_dir, "latest.pt"), device)
    net.load_state_dict(ck["net"])
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5, fused=True)
    opt.load_state_dict(ck["opt"])
    for g in opt.param_groups:
        g["lr"] = args.lr_end or args.lr
    fixed = FixedShapes(net, cuda_graphs=True)
    roll_net = RunNet(d=args.d, layers=args.layers, heads=args.heads).to(device)
    for q in roll_net.parameters():
        q.requires_grad_(False)
    roll_policy = FixedShapes(roll_net, cuda_graphs=True)
    T, N = args.steps, args.envs
    B = T * N
    update = ck["update"]
    col = Collector(args, device, amp, update)
    bufs = [rollout_buffers(T, N, device), rollout_buffers(T, N, device)]
    s_train, s_roll = torch.cuda.Stream(), torch.cuda.Stream()
    print(f"{a.run_dir} update {update}: {N} games x {T} steps, minibatch {args.minibatch}, {args.epochs} epochs; "
          f"two batches {2 * sum(t.numel() * t.element_size() for t in bufs[0].values()) / 2 ** 30:.1f} GB", flush=True)

    def copy_weights():
        """The rollout's network := the trainer's, on the rollout stream once training's queued
        work is done; training waits for the copy before it changes the weights again."""
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream())
        s_roll.wait_event(done)
        with torch.no_grad(), torch.cuda.stream(s_roll):
            for (_, dst), (_, src) in zip(roll_net.state_dict().items(), net.state_dict().items()):
                dst.copy_(src, non_blocking=True)
        copied = torch.cuda.Event()
        copied.record(s_roll)
        torch.cuda.current_stream().wait_event(copied)

    def train(buf):
        net.train()
        b = {k: v.reshape(B, *v.shape[2:]) for k, v in buf.items() if k in ("ints", "floats", "mask", "act", "logp", "adv", "ret", "prac")}
        row_kinds = kinds(b["ints"])
        t0 = time.time()
        stats, epochs = ppo_update(net, opt, args, amp, b["ints"], b["floats"], b["mask"], b["act"], b["logp"], b["adv"], b["ret"],
                                   practice=b["prac"], batches=lambda: minibatches(row_kinds, args.minibatch), forward=fixed.forward)
        torch.cuda.current_stream().synchronize()
        return time.time() - t0, epochs

    # The rollout thread: one for the whole run (torch.compile's CUDA-graph trees are per thread).
    jobs, results = queue.Queue(), queue.Queue()

    def rollout_thread():
        while (job := jobs.get()) is not None:
            u, buf = job
            try:
                with torch.cuda.stream(s_roll):
                    t0 = time.time()
                    ro = col.collect(u, roll_policy, buf)
                    s_roll.synchronize()
                    ro["wall_s"] = time.time() - t0
                results.put(ro)
            except Exception as e:  # noqa: BLE001
                import traceback
                results.put({"error": traceback.format_exc() or repr(e)})

    worker = threading.Thread(target=rollout_thread, daemon=True)
    worker.start()

    def rollout(u, buf):
        jobs.put((u, buf))
        ro = results.get()
        if "error" in ro:
            raise RuntimeError(ro["error"])
        return ro

    # Warm-up: one rollout and one training, each alone (compiling and capturing their graphs).
    t0 = time.time()
    copy_weights()
    ro = rollout(update, bufs[0])
    with torch.cuda.stream(s_train):
        train_s, _ = train(bufs[0])
    print(f"warm-up: rollout {ro['wall_s']:.1f} s, training {train_s:.1f} s ({time.time() - t0:.0f} s)", flush=True)
    update += 1
    report = {}
    gpu = GpuSampler()
    for mode in a.modes.split(","):
        times, roll_s, train_t, phases = [], [], [], []
        gpu.start()
        cur = 0
        if mode == "thread":
            copy_weights()
            jobs.put((update, bufs[cur]))
            first = results.get()
            if "error" in first:
                raise RuntimeError(first["error"])
        for k in range(a.updates):
            t0 = time.time()
            if mode == "sequential":
                copy_weights()
                ro = rollout(update, bufs[0])
                with torch.cuda.stream(s_train):
                    tr, _ = train(bufs[0])
            else:
                # The next rollout (with the weights before this training) beside this training.
                with torch.cuda.stream(s_train):
                    copy_weights()
                jobs.put((update + 1, bufs[1 - cur]))
                with torch.cuda.stream(s_train):
                    tr, _ = train(bufs[cur])
                ro = results.get()
                if "error" in ro:
                    raise RuntimeError(ro["error"])
                cur = 1 - cur
            dt = time.time() - t0
            times.append(dt)
            roll_s.append(ro["wall_s"])
            train_t.append(tr)
            phases.append(ro["phase_s"])
            update += 1
            print(f"  {mode} {k + 1}: update {dt:.1f} s ({B / dt:,.0f} steps/s), rollout {ro['wall_s']:.1f} s, training {tr:.1f} s; "
                  + " ".join(f"{p} {v:.1f}" for p, v in ro["phase_s"].items()), flush=True)
        g = gpu.stop()
        report[mode] = {"update_s": float(np.mean(times)), "steps_per_s": B / float(np.mean(times)), "rollout_s": float(np.mean(roll_s)),
                        "training_s": float(np.mean(train_t)), **g,
                        "phase_s": {p: float(np.mean([x[p] for x in phases])) for p in phases[0]},
                        "reserved_gb": torch.cuda.max_memory_reserved(device) / 2 ** 30}
        print(f"{mode}: {json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in report[mode].items() if k != 'phase_s'})}", flush=True)
    jobs.put(None)
    with open(os.path.join(a.run_dir, "bench_one_process.json"), "w") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
