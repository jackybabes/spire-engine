"""Checks for train_runs.py's --teacher losses (plain asserts, no pytest needed):

    python python/tests/test_teacher.py

--teacher-fade N takes the losses from full at update 0 to nothing at update N (or to
--teacher-floor, kept for good), and leaves them full without it. On positions from random Watcher runs (practice rows among them), the teacher
loss pulls a new, larger network towards a small one, as kickstarting a larger network does
(--teacher-moves all --teacher-value W): the KL and the value's error fall a long way and their
likeliest moves come to agree; and the same through PPO's own minibatches (`ppo_update`'s `teacher`,
which --teacher-moves all uses): with no advantage to follow, its updates alone bring a new network
to the teacher, and report the teacher KL, value error and agreement."""
import argparse
import contextlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch

from core.fight_model import masked
from core.ppo import ppo_update
from core.run_model import RunNet
from core.sts_env import RunEnv
from training.train_runs import teacher_fade, teacher_loss, teacher_targets


def test_fade():
    a = argparse.Namespace(teacher_fade=300)
    assert teacher_fade(a, 0) == 1.0
    assert abs(teacher_fade(a, 150) - 0.5) < 1e-12
    assert teacher_fade(a, 300) == 0.0 and teacher_fade(a, 1000) == 0.0
    assert teacher_fade(argparse.Namespace(teacher_fade=0), 5000) == 1.0
    held = argparse.Namespace(teacher_fade=300, teacher_floor=0.3)
    assert abs(teacher_fade(held, 150) - 0.5) < 1e-12 and teacher_fade(held, 250) == 0.3 and teacher_fade(held, 5000) == 0.3


def positions(n_games=64, steps=12):
    env = RunEnv(n_games, seed=3, character="watcher")
    ints, floats, mask = [], [], []
    for k in range(steps):
        obs = env.observe()
        ints.append(obs["ints"].copy())
        floats.append(obs["floats"].copy())
        mask.append(obs["mask"].copy())
        env.envs.step_random(100 + k)
    ints = torch.from_numpy(np.concatenate(ints).astype(np.int64))
    floats = torch.from_numpy(np.concatenate(floats))
    mask = torch.from_numpy(np.concatenate(mask))
    practice = torch.arange(len(ints)) % 5 == 0  # some rows on the practice value head
    return ints, floats, mask, practice


def test_kickstart():
    torch.manual_seed(0)
    ints, floats, mask, practice = positions()
    amp = contextlib.nullcontext
    teacher = RunNet(d=32, layers=1, heads=2).eval()
    student = RunNet(d=64, layers=2, heads=4)
    rows = torch.arange(len(ints))
    t_logp, t_val = teacher_targets(teacher, amp, ints, floats, mask, practice, rows, chunk=256)
    assert t_logp.shape == (len(ints), mask.shape[1]) and t_val.shape == (len(ints),)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    first = None
    for _ in range(300):
        k = torch.randint(len(rows), (256,))
        kl, v_err, agree = teacher_loss(student, amp, ints, floats, mask, practice, rows[k], t_logp[k], t_val[k])
        if first is None:
            first = (kl.item(), v_err.item(), agree.item())
        opt.zero_grad()
        (kl + v_err).backward()
        opt.step()
    with torch.no_grad():
        kl, v_err, agree = teacher_loss(student, amp, ints, floats, mask, practice, rows, t_logp, t_val)
    print(f"KL {first[0]:.4f} -> {kl.item():.4f}, value error {first[1]:.4f} -> {v_err.item():.5f}, "
          f"agreement {first[2]:.2f} -> {agree.item():.2f}")
    assert kl.item() < 0.2 * first[0]
    assert v_err.item() < 0.2 * first[1]
    assert agree.item() > 0.9


def test_ppo_teacher():
    torch.manual_seed(1)
    ints, floats, mask, practice = positions()
    amp = contextlib.nullcontext
    teacher = RunNet(d=32, layers=1, heads=2).eval()
    student = RunNet(d=64, layers=2, heads=4)
    B = len(ints)
    t_logp, t_val = teacher_targets(teacher, amp, ints, floats, mask, practice, torch.arange(B), chunk=256)
    opt = torch.optim.Adam(student.parameters(), lr=1e-3)
    args = argparse.Namespace(epochs=2, minibatch=128, clip=0.2, vf=0.0, ent=0.0, max_grad=0.5, target_kl=10.0, prox_cap=2.0)
    seen = []
    for _ in range(25):
        with torch.no_grad():
            logp_all = torch.log_softmax(masked(student(ints, floats, practice)[0].float(), mask), -1)
        act = torch.distributions.Categorical(logits=logp_all).sample()
        logp = logp_all.gather(1, act[:, None]).squeeze(1)
        stats, _ = ppo_update(student, opt, args, amp, ints, floats, mask, act, logp, torch.zeros(B), torch.zeros(B),
                              practice=practice, teacher=(t_logp, t_val, 1.0, 1.0))
        seen.append((stats["teacher_kl"], stats["teacher_value_err"], stats["teacher_agree"]))
    print(f"through PPO: KL {seen[0][0]:.4f} -> {seen[-1][0]:.4f}, value error {seen[0][1]:.4f} -> {seen[-1][1]:.5f}, "
          f"agreement {seen[0][2]:.2f} -> {seen[-1][2]:.2f}")
    assert seen[-1][0] < 0.2 * seen[0][0]
    assert seen[-1][1] < 0.2 * seen[0][1]
    assert seen[-1][2] > 0.9


if __name__ == "__main__":
    test_fade()
    test_kickstart()
    test_ppo_teacher()
    print("ok")
