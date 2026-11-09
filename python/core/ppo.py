"""PPO pieces shared by train_fights.py and train_runs.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import numpy as np
import torch
import torch.nn.functional as F

from core.fight_model import masked


def gae(rew, val, done, next_val, gamma, lam):
    """Advantages and returns for (T, N) rollouts; `done[t]` ends the episode after step t."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    last = torch.zeros_like(next_val)
    for t in reversed(range(T)):
        nv = next_val if t == T - 1 else val[t + 1]
        live = 1.0 - done[t]
        delta = rew[t] + gamma * nv * live - val[t]
        last = delta + gamma * lam * live * last
        adv[t] = last
    return adv, adv + val


def choice_gae(rew, val, done, next_val, choice, gamma, lam):
    """Advantages and returns over the choice steps only (`choice` (T, N) true), for a network that
    makes only those (the split's deck network, train_runs.py --fight-net): the steps between two
    choices are one transition, whose reward is their rewards discounted from the first choice
    (gamma a step) and whose discount is gamma to the power of the steps between. `lam` is per choice.
    `done[t]` ends the episode after step t; `next_val` is the value after the last step (the rest of
    the run from there). Other steps get advantage and return 0."""
    T = rew.shape[0]
    adv = torch.zeros_like(rew)
    ret = torch.zeros_like(rew)
    acc = torch.zeros_like(next_val)  # the rewards from the step after a choice to the next choice, discounted
    disc = torch.ones_like(next_val)  # gamma to the power of the steps from there to the next choice (0 past the episode's end)
    nv, na = next_val, torch.zeros_like(next_val)  # the next choice's value and advantage
    for t in reversed(range(T)):
        live = 1.0 - done[t]
        acc = rew[t] + gamma * live * acc
        disc = gamma * live * disc
        c = choice[t].bool()
        delta = acc + disc * nv - val[t]
        a = delta + lam * disc * na
        adv[t] = torch.where(c, a, adv[t])
        ret[t] = torch.where(c, a + val[t], ret[t])
        # A choice is the next one for the steps before it.
        nv, na = torch.where(c, val[t], nv), torch.where(c, a, na)
        acc, disc = torch.where(c, torch.zeros_like(acc), acc), torch.where(c, torch.ones_like(disc), disc)
    return adv, ret


def ppo_update(net, opt, args, amp, ints, floats, mask, act, logp_old, adv, ret, policy=True, ent_coef=None, practice=None, aux=None,
               batches=None, forward=None, logp_prox=None, clone=0.0, zero_avoid=None, teacher=None):
    """`args.epochs` passes of clipped PPO over flat (B, ...) tensors, stopping when the
    approximate KL passes `args.target_kl`. With `policy=False` only the value is trained (to
    warm it up after the rewards change scale). `ent_coef` (B,) gives each row its own entropy
    bonus (default `args.ent` for all). `practice` (B,) marks practice-fight rows for the run
    network's value head. `aux()` (optional) returns an extra loss for each minibatch and a dict of
    stats to average (the human-imitation loss). Its gradients are taken before the minibatch's
    own pass and add up with them, so the two batches' activations aren't held at once (GPU
    memory). `batches()` (optional) gives each epoch's minibatches as (rows, kind) and
    `forward(kind, ints, floats, practice)` runs one (`core.fixed_shapes`: minibatches of one shape
    each, for CUDA graphs); by default each epoch shuffles every row into `args.minibatch`es for
    `net`. The early stop watches the KL of the whole policy: each kind's latest minibatch KL
    weighted by the kind's share of the rows (kinds not reached yet this epoch count as 0), which
    for mixed minibatches is the latest minibatch's own. One kind's minibatch alone stopped updates
    before (2026-10-24: defect-fights-0's card-selection rows, 3% of them, move faster than the rest). Returns
    (mean stats, epochs run): the stats include each kind's mean KL (`approx_kl_<kind>`) and, if
    the update stopped early, the share of that epoch's minibatches run (`stopped_at`) and the
    stopping minibatch's kind (`stop_kind`). `logp_prox` (B,), if given, anchors the clip and the
    KL on the network at the update's start instead of the policy that played the rollout (an
    update older with --overlap, which made defect-fights-0's policy swing back and forth, 2026-10-24); the
    rollout's policy then weights each row's advantage by prox / played, capped at `args.prox_cap`
    (decoupled PPO: Hilton et al., "Batch size-invariance for policy optimization", 2021).
    With `policy=False`, `clone` weighs a penalty on the KL from the anchor (`logp_prox`, or the
    rollout's policy), so value-only epochs on the shared network hold the policy where it was
    (`train_runs.py --value-epochs`; phasic policy gradient's behaviour cloning, Cobbe et al. 2020).
    The stats include the gradient's norm before clipping (`grad_norm`) and the share of minibatches
    it was clipped in (`grad_clipped`). `zero_avoid` (B,), if given, weighs each row's KL from the
    uniform policy over its legal moves, added to the loss (Jaxcalibur's "zero-avoiding" term,
    notes/ideas-from-other-game-ais.md): its pull on a move's logit is about 1/n - p however unlikely
    the move has become, where the entropy bonus's (about p (log p + H)) vanishes with p, so a move
    the policy has given up on still gets tried. The stats include that KL's mean (`kl_uniform`).
    `teacher` (optional), (log-probabilities (B, actions), values (B,), KL weight, value weight): a
    teacher network's on every row, so each minibatch's loss also pulls the policy towards the
    teacher's (KL(teacher || policy) over the legal moves) and the value towards its value, on the
    minibatch's own rows (kickstarting, train_runs.py --teacher-moves all; no extra pass). The stats
    include `teacher_kl`, `teacher_value_err` and `teacher_agree` (their likeliest moves the same).

    On a GPU nothing waits for it within an epoch (2026-10-24: five reads back a minibatch held the
    CPU until each minibatch was done, so the GPU idled while the next was queued): the stats stay on
    the GPU until the epoch ends, and each minibatch's KL comes back through a small page-locked
    buffer and is checked once the next minibatch is queued, so an early stop comes a minibatch late
    (`stopped_at` counts the minibatches run)."""
    B = ints.shape[0]
    stats = {"policy_loss": [], "value_loss": [], "entropy": [], "approx_kl": [], "clip_frac": [], "grad_norm": [], "grad_clipped": [],
             "kl_uniform": []}
    names = ("policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "grad_norm", "grad_clipped", "kl_uniform")
    cuda = ints.is_cuda
    epochs = 0
    for _ in range(args.epochs):
        if batches is not None:
            plan = batches()
        else:
            perm = torch.randperm(B, device=ints.device)
            plan = [(perm[i:i + args.minibatch], None) for i in range(0, B, args.minibatch)]
        stop = False
        rows = {}
        for idx, kind in plan:
            rows[kind] = rows.get(kind, 0) + len(idx)
        total, latest = sum(rows.values()), {}
        on_gpu = []  # each minibatch's (policy loss, value loss, entropy, KL, clip fraction), until the epoch ends
        t_on_gpu = []  # with `teacher`: each minibatch's (teacher KL, value error, agreement)
        host = torch.zeros(len(plan), pin_memory=cuda)
        arrived = [None] * len(plan)

        def check(j):
            """Minibatch j's KL is back: the early stop's sums."""
            nonlocal stop, stopped
            if arrived[j] is not None:
                arrived[j].synchronize()
            kl = float(host[j])
            j_kind = plan[j][1]
            latest[j_kind] = kl
            if not stop and sum(rows[k] * v for k, v in latest.items()) / total > args.target_kl:
                stop, stopped = True, {"stop_kind": j_kind}

        stopped = {}
        for i, (idx, kind) in enumerate(plan):
            opt.zero_grad(set_to_none=True)
            if aux is not None:
                a_loss, a_stats = aux()
                if a_loss.requires_grad:  # (none on a minibatch the aux skips: --human-every)
                    a_loss.backward()
                for k, v in a_stats.items():
                    stats.setdefault(k, []).append(v)
            with amp():
                if forward is not None:
                    logits, value = forward(kind, ints[idx].long(), floats[idx], practice[idx])
                elif practice is None:
                    logits, value = net(ints[idx].long(), floats[idx])
                else:
                    logits, value = net(ints[idx].long(), floats[idx], practice[idx])
            logits, value = logits.float(), value.float()
            logp_all = F.log_softmax(masked(logits, mask[idx]), dim=-1)
            logp = logp_all.gather(1, act[idx][:, None]).squeeze(1)
            row_entropy = -(logp_all.exp() * logp_all).sum(-1)
            entropy = row_entropy.mean()
            ent_bonus = args.ent * entropy if ent_coef is None else (ent_coef[idx] * row_entropy).mean()
            # KL(uniform over the legal moves || policy) = -mean of the legal moves' log p - log n.
            legal = mask[idx]
            n_legal = legal.sum(-1).clamp(min=1).float()
            row_kl_u = -torch.where(legal, logp_all, torch.zeros_like(logp_all)).sum(-1) / n_legal - n_legal.log()
            za_loss = (zero_avoid[idx] * row_kl_u).mean() if zero_avoid is not None else 0.0
            anchor = logp_old[idx] if logp_prox is None else logp_prox[idx]
            log_ratio = logp - anchor
            ratio = log_ratio.exp()
            mb_adv = adv[idx]
            if len(idx) > 1:
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            else:
                # One row (a kind's leftover in --fixed-shapes' one-kind minibatches): its std is
                # undefined (NaN, which put NaN in defect-runs-9's weights at update 444); no policy signal.
                mb_adv = torch.zeros_like(mb_adv)
            if logp_prox is not None:
                mb_adv = mb_adv * (anchor - logp_old[idx]).exp().clamp(max=args.prox_cap)
            pg = torch.max(-mb_adv * ratio, -mb_adv * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
            v_loss = 0.5 * (value - ret[idx]).pow(2).mean()
            kl = ((ratio - 1) - log_ratio).mean()
            loss = args.vf * v_loss + ((pg - ent_bonus + za_loss) if policy else clone * kl if clone else 0.0)
            if teacher is not None and policy:
                t_logp = teacher[0][idx]
                t_kl = (t_logp.exp() * (t_logp - logp_all)).nan_to_num(0.0).sum(-1).mean()
                t_err = (value - teacher[1][idx]).pow(2).mean()
                loss = loss + teacher[2] * t_kl + teacher[3] * t_err
                t_on_gpu.append(torch.stack([t_kl.detach(), t_err.detach(), (logp_all.argmax(-1) == t_logp.argmax(-1)).float().mean()]))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(net.parameters(), args.max_grad).float()
            opt.step()
            with torch.no_grad():
                kl = kl.detach()
                on_gpu.append(torch.stack([pg.detach(), v_loss.detach(), entropy.detach(), kl, ((ratio - 1).abs() > args.clip).float().mean(),
                                           norm, (norm > args.max_grad).float(), row_kl_u.detach().mean()]))
                host[i:i + 1].copy_(kl.reshape(1), non_blocking=cuda)
                if cuda:
                    arrived[i] = torch.cuda.Event()
                    arrived[i].record()
            if not cuda:
                check(i)
            elif i > 0:
                check(i - 1)
            if stop:
                stopped["stopped_at"] = (i + 1) / len(plan)
                break
        if cuda and not stop:
            check(len(plan) - 1)
            if stop:
                stopped["stopped_at"] = 1.0
        done = len(on_gpu)
        for row in torch.stack(on_gpu).cpu().tolist():
            for name, v in zip(names, row):
                stats[name].append(v)
        if t_on_gpu:
            for row in torch.stack(t_on_gpu).cpu().tolist():
                for name, v in zip(("teacher_kl", "teacher_value_err", "teacher_agree"), row):
                    stats.setdefault(name, []).append(v)
        for j in range(done):
            if plan[j][1] is not None:
                stats.setdefault(f"approx_kl_{plan[j][1]}", []).append(float(host[j]))
        epochs += 1
        if stop:
            break
    out = {k: float(np.mean(v)) for k, v in stats.items()}
    if stop:
        out.update(stopped)
    return out, epochs


def explained_variance(ret, val):
    var = ret.var().item()
    return 1 - (ret - val).var().item() / var if var > 0 else 0.0
