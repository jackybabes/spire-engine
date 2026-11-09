"""Greedy play with the fight look-ahead (fight_search.py) against plain greedy play, on the
fixed eval runs and on the held-out human boss fights (train_runs.py's evals, same seeds).

At every fight move outside a card selection, the network's `--top` likeliest moves and ending
the turn are each played out on `--k` redrawn copies by the network's own greedy moves until the
player's next turn (`--turns`), scored by the network's value and averaged; the best-scoring
move is played if it beats the network's own choice by `--margin`. Elsewhere the network picks
as usual.

    python python/search/search_eval.py python/runs/ironclad-runs-habits-12/update_200.pt --set runs --ascension 20 --runs 1024 --procs 4
    python python/search/search_eval.py python/runs/ironclad-runs-habits-12/update_200.pt --set bosses --procs 4

Several deck networks' checkpoints, comma-separated, play as one (their mean logits; the first's settings). A deck network's
checkpoint (train_runs.py --fight-net) plays as it trained: it makes the choices between fights (with the
lookup, if it trained with one), its fight network fights, and the look-ahead searches with the fight network
(`--scale-from`: the checkpoint whose reward scale the fight network's values are in; default the fight network's own).

--card-net: another network's picks at card rewards, from --card-from-floor on (core.split.CardPicker; its top pick, or with
--card-sample a draw from its masked softmax). Every other row stays the checkpoint's, plain or a deck network's, the fights
above all. --picks (on with --card-net) prints the skips and the cards taken by CARD_BUCKETS, and puts each run's card rewards in
its summary (--dump; analysis/card_picks.py --from-dump compares them with human winners').

    python python/search/search_eval.py python/runs/silent-runs-5/update_1200.pt --character silent --set runs --no-search --seed-offset 1024 \
        --card-net python/runs/silent-human-1/latest.pt --card-from-floor 17 --card-sample --dump cards17

--card-route: a network's picks for each of the Silent's archetypes, from --card-from-floor on (core.split.ArchetypeRouter). Each card reward
goes to the picker for the archetype of that run's deck as it stands (analysis/archetypes.py). A deck with no picker goes by
--card-route-default: max, the pickers together (each card at its largest probability over them, the skip at its smallest), so a run takes
a defining card when it's offered and goes to that card's picker from then on; or base, the checkpoint's own. --card-sample draws from
those logits as it does for --card-net; a reward left to the checkpoint keeps its top pick. --picks is on, each row of a run's "picks" says
who decided it, and the end prints each picker's count by CARD_BUCKETS, and any rewards the checkpoint left unopened, which reach no picker (the
rewards screen stays the checkpoint's). The look-ahead works as it does with --card-net: it only plays fights, with the fight network. --decks
with --dump gives analysis/archetypes.py --dump what became of each archetype (--min-floor 17: the runs that got as far as the routes).

    python python/search/search_eval.py python/runs/silent-runs-9/update_300.pt --character silent --set runs --no-search --seed-offset 1024 --free-keys --decks --dump routed --card-from-floor 17 \
        --card-route grand-finale=python/runs/silent-arch-grand-finale/latest.pt,wraith-form=python/runs/silent-arch-wraith-form/latest.pt,discard=python/runs/silent-arch-discard/latest.pt,poison=python/runs/silent-arch-poison/latest.pt
    python3 python/analysis/archetypes.py --dump routed --min-floor 17

--procs N splits the runs (and the boss fights) between N processes, each with its own copy of
the network on the GPU. They play the same runs one process would (run j has seed EVAL_SEED + j
either way), about N times as fast while the GPU has room: one process keeps it about a quarter
busy, held up by its own Python loop. With a 16 GB GPU and 15.6 GB of RAM (WSL) RAM is the tighter
limit: each process holds about 0.7 GB of libraries plus its games and the search's copies of
them (about 40 whole runs per game searched at once, --roots-per-call). 4 processes of 256 runs
took 10.6 GB of RAM in all and 6.2 GB of GPU memory; 8 of 128 ran out of RAM (swapping) at
11.3 GB of GPU memory. --pass-rows bounds the GPU memory each one takes, and
--gpu-gb caps it: past the cap a process fails (out of memory) instead of spilling into Windows'
shared GPU memory.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import os

# Less GPU memory held in the cache between the search's batches of different sizes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse  # noqa: E402
import collections  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from analysis.archetypes import ORDER  # noqa: E402
from core.fight_model import masked, pick_device  # noqa: E402
from core.fixed_shapes import FixedShapes  # noqa: E402
from core.lookup import CARD_REWARD, SCREEN_AT, Lookup, trained_events  # noqa: E402
from core.split import ArchetypeRouter, BossFighter, CardPicker, Ensemble, SplitPolicy  # noqa: E402
from search.fight_search import FightSearch, reward_std  # noqa: E402
from core.run_model import RUN_FLOAT_AT, RUN_INT_AT  # noqa: E402
from core.sts_env import LAYOUT, RunEnv  # noqa: E402
from training.train_runs import BOSS_KINDS, EVAL_SEED, SCENARIO_EVAL_SEED, in_combat, load, load_fight_net, practice_reward, read_scenarios, summarize  # noqa: E402

SEARCH_STATS = ("searches", "playouts", "playout_moves")
POTION, END_TURN = LAYOUT["actions"]["potion"][0], LAYOUT["actions"]["end_turn"]
ROOM_AT = RUN_INT_AT["room"][0]
ROOMS = {1: "hallway", 2: "elite", 7: "boss"}  # encode_run.rs room_id + 1
BOSS_ROOM_ID = 7
HALLWAY_ROOM_ID = 1
PICK_AT, MAX_OPTIONS_AT = LAYOUT["actions"]["pick"]
# --habits: what each run's choices add up to (Envs.habits rows summed; rest_hp is the HP shares rested at, summed).
HABIT_KEYS = ["cards_added", "cards_removed", "deck", "rests", "rest_hp", "elites_won", "smiths", "hallway_potions",
              "removals_skipped", "strikes_upgraded", "cards_taken", "cards_skipped"]
ACT_AT = RUN_FLOAT_AT["run_scalars"][0] + 4  # hp, max hp, gold, floor, act
REWARDS = LAYOUT["screens"].index("rewards") + 1  # encode_run.rs screen_id
CONFIRM = LAYOUT["actions"]["confirm"]
CARDS_OPTION = LAYOUT["option_kinds"].index("cards") + 1  # encode_run.rs kind
OPT_KIND_AT, OPT_KIND_END = RUN_INT_AT["opt_kind"]
CARD_BUCKETS = ("floors 1-8", "floors 9-16", "Act 2", "Act 3")


def card_bucket(floor):
    """A card reward's place in CARD_BUCKETS by its floor, as human records have it (a boss's chest closes its act; Act 4's few
    rewards count as Act 3's). None for Neow's, on floor 0."""
    if floor < 1:
        return None
    return 0 if floor <= 8 else 1 if floor <= 17 else 2 if floor <= 34 else 3


def card_names(cards):
    """The names of a screen's cards as the game displays them, without an upgrade's +."""
    return [c["name"].rstrip("+") for c in cards]


def beat_act3(e):
    """Whether a run beat the Act 3 boss (both of them at A20): `won`, or Act 4 reached. With the three keys a run goes on and
    `won` waits for the Heart (analysis/run_deaths.py counts the same)."""
    return bool(e["won"]) or e["act"] >= 4


class CardChoices:
    """`play`'s `cards`: each run's card rewards, chosen and written down. In its summary, "picks": [floor, the cards offered, the
    card taken or None, how]: "own", the policy's choice (a sampled one too), or who decided for it with `router` (a
    core.split.ArchetypeRouter: an archetype's name, "max"); "forced", `rule`'s where it differs; "unopened", a reward left on the
    rewards screen, which a human record counts as a skip too.
    `sample` (picker, seed, run_seed(game, episode)): the picker's rows (a CardPicker's, a router's) draw from its masked softmax. Each
    draw has its own stream (the seed, the run's seed, the run's card rewards so far), so a run gets the same draws however the runs
    are shared between games and processes. `rule(view, logits)`: the action slot to play at this card reward (`view`: the game's
    observation; `logits`: this row's, masked), or None for the policy's choice."""

    def __init__(self, n, sample=None, rule=None, router=None):
        self.sample, self.rule, self.router = sample, rule, router
        self.picks = [[] for _ in range(n)]
        self.seen = [0] * n

    def step(self, env, obs, ints, floats, logits, acts, episode, wanted):
        """Changes `acts` in place and writes down the games' card rewards."""
        screen = obs["ints"][:, SCREEN_AT]
        at_reward = np.flatnonzero(screen == CARD_REWARD)
        drawn = self.sample[0].rows(ints, floats).cpu().numpy() if self.sample and len(at_reward) else None
        decided = self.router.deciders(ints, floats) if self.router and len(at_reward) else {}
        for g in at_reward:
            if not wanted(g, episode[g]):
                continue
            if drawn is not None and drawn[g]:
                _, seed, run_seed = self.sample
                p = torch.softmax(logits[g].cpu().double(), -1).numpy()
                acts[g] = np.random.default_rng([seed, int(run_seed(g, episode[g])), self.seen[g]]).choice(len(p), p=p / p.sum())
            self.seen[g] += 1
            view = json.loads(env.envs.observe(int(g)))
            cards, how = card_names(view["screen"]["cards"]), decided.get(int(g), "own")
            slot = self.rule(view, logits[g]) if self.rule else None
            if slot is not None and slot != acts[g]:
                acts[g], how = slot, "forced"
            a = int(acts[g])
            self.picks[g].append([view["floor"], cards, cards[a - PICK_AT] if PICK_AT <= a < PICK_AT + len(cards) else None, how])
        # Leaving the rewards screen with a card reward not looked at yet (one already skipped isn't a legal option).
        unopened = (screen == REWARDS) & (acts == CONFIRM) & ((obs["ints"][:, OPT_KIND_AT:OPT_KIND_END] == CARDS_OPTION)
                                                              & obs["mask"][:, PICK_AT:PICK_AT + MAX_OPTIONS_AT]).any(1)
        for g in np.flatnonzero(unopened):
            if wanted(g, episode[g]):
                view = json.loads(env.envs.observe(int(g)))
                for k, item in enumerate(view["screen"]["items"]):
                    if item["type"] == "cards" and obs["mask"][g, PICK_AT + k]:
                        self.picks[g].append([view["floor"], card_names(item["cards"]), None, "unopened"])

    def finish(self, g):
        """Game `g`'s run is over: returns its picks, ready for its next run."""
        picks, self.picks[g], self.seen[g] = self.picks[g], [], 0
        return picks


class CardTally:
    """Card rewards counted overall (bucket None) and by CARD_BUCKETS: how many, how many skipped, each card's offers and takes."""

    def __init__(self):
        self.rewards, self.skips = collections.Counter(), collections.Counter()
        self.offered, self.taken = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)

    def add(self, floor, cards, taken):
        for b in {None, card_bucket(floor)}:
            self.rewards[b] += 1
            self.offered[b].update(cards)
            if taken is None:
                self.skips[b] += 1
            else:
                self.taken[b][taken] += 1

    def add_runs(self, out):
        for e in out:
            for floor, cards, taken, _ in e["picks"]:
                self.add(floor, cards, taken)
        return self

    def skip_rate(self, b=None):
        return self.skips[b] / max(self.rewards[b], 1)


def most_taken(t, b, top):
    """A CardTally's cards taken most in bucket `b`, with each one's share of its offers."""
    most = sorted(t.taken[b].items(), key=lambda kv: -kv[1])[:top]
    return ", ".join(f"{c} {n / t.offered[b][c]:.0%} of {t.offered[b][c]}" for c, n in most)


def picks_lines(out, top=8):
    """--picks: the runs' skips by CARD_BUCKETS and the cards they took most (each one's share of its offers)."""
    t = CardTally().add_runs(out)
    unopened = sum(how == "unopened" for e in out for *_, how in e["picks"])
    lines = [f"  card rewards: {t.rewards[None] / max(len(out), 1):.1f} a run, skipped {t.skip_rate():.0%} ({unopened} of {t.skips[None]} left unopened)"]
    for b, name in enumerate(CARD_BUCKETS):
        lines.append(f"  {name}: {t.rewards[b] / max(len(out), 1):.1f} a run, skipped {t.skip_rate(b):.0%}; most taken: " + most_taken(t, b, top))
    return lines


def route_lines(out, deciders, top=5):
    """--card-route: how many of the runs' card rewards each of `deciders` decided (a row's "how"), by CARD_BUCKETS, with its skips and
    the cards it took most in each. A reward left unopened was nobody's: the rewards screen is the checkpoint's, so those never reach a
    picker, and a last line says how many there were and where."""
    tally, unopened = {who: CardTally() for who in deciders}, CardTally()
    for e in out:
        for floor, cards, taken, how in e["picks"]:
            (unopened if how == "unopened" else tally[how]).add(floor, cards, taken)
    total = sum(t.rewards[None] for t in tally.values())

    def counts(t):
        return "".join(f", {name} {t.rewards[b]}" for b, name in enumerate(CARD_BUCKETS) if t.rewards[b])

    lines = []
    for who, t in tally.items():
        lines.append(f"  decided by {who}: {t.rewards[None]} of {total} card rewards" + counts(t) + f"; skipped {t.skip_rate():.0%}")
        lines += [f"    {name}: skipped {t.skip_rate(b):.0%}; most taken: " + most_taken(t, b, top) for b, name in enumerate(CARD_BUCKETS) if t.rewards[b]]
    if unopened.rewards[None]:
        lines.append(f"  left unopened by the checkpoint, no picker's: {unopened.rewards[None]}" + counts(unopened))
    return lines


def act3_line(out):
    """--picks: what a test is judged by, on every run: Act 3 reached, and its bosses beaten (beat_act3)."""
    arrived, beat = sum(e["act"] >= 3 for e in out), sum(beat_act3(e) for e in out)
    return f"  Act 3 reached {arrived / max(len(out), 1):.1%}, its bosses beaten {beat / max(len(out), 1):.1%} of runs, {beat / max(arrived, 1):.0%} of arrivals"


def parse_card_route(text):
    """--card-route's NAME=PATH,NAME=PATH to {archetype: path}, names as analysis.archetypes.ORDER with - for spaces. Raises
    ValueError on a name that isn't one (nothing would ever be routed to it) or that comes twice."""
    names = {a.replace(" ", "-"): a for a in ORDER}
    routes = {}
    for part in text.split(","):
        name, _, path = part.partition("=")
        if name not in names or not path:
            raise ValueError(f"--card-route: {part!r} isn't ARCHETYPE=PATH, with ARCHETYPE one of {', '.join(names)}")
        if names[name] in routes:
            raise ValueError(f"--card-route: {name} named twice")
        routes[names[name]] = os.path.expanduser(path)
    return routes


def cap_gpu(device, gb):
    """Past `gb` GB a process fails (out of memory) instead of spilling into Windows' shared GPU memory."""
    if device.type == "cuda" and gb:
        index = torch.cuda.current_device() if device.index is None else device.index
        torch.cuda.set_per_process_memory_fraction(min(gb * 2 ** 30 / torch.cuda.get_device_properties(index).total_memory, 1.0), index)


def play(net, device, amp, env, wanted, search, margin, label, margin_se=0.0, lookup=None, rooms=None, potion_bias=None,
         potions_only=False, boss_search=None, peek=None, boss_acts=None, habits=False, cards=None, save_wins=None, hallway_from=1):
    """Plays `env` until `wanted(game, episode)` is false for every game's next episode; returns the
    episodes it wanted and the search's stats. `lookup`: filled in before each move (a deck network's). `rooms`: room ids
    (ROOMS' keys) whose fights the search plays, or None for every fight. `potion_bias`: {room id: added to the potion moves'
    logits in that room's fights} (before the greedy move and the search's candidates are read). `potions_only`: the search
    plays only the moves where the network would drink a potion. `boss_search`: another search for the boss fights (stronger
    settings), `search` for the rest. `habits`: each run's choices counted into its summary ("habits": cards added and removed,
    the deck at the end, rests and the HP share rested at, smiths, elites won, hallway potions, removals skipped, card rewards
    taken and skipped). `peek` (margin, deck weight, deck free, reward scale): at card rewards, each option
    taken on a copy (Envs.peek_after) and judged by the deck network's value where it leads, less the deck-size penalty it
    trained with (train_runs.py --deck-weight); the best replaces the network's pick if ahead by `margin` (value units).
    `cards`: a CardChoices, for the choices at card rewards and each run's "picks". `save_wins`: a folder for each Heart win's log.
    `hallway_from`: the first act whose hallway fights the search plays (if `rooms` has them)."""
    n = env.n
    episode = np.zeros(n, dtype=int)
    out = []
    st = {"changed": 0, "gap": [], **{f"potions_{r}": 0 for r in ROOMS.values()}}
    hab = np.zeros((n, len(HABIT_KEYS)))
    t0, t_search = time.time(), 0.0
    while any(wanted(g, episode[g]) for g in range(n)):
        obs = env.observe()
        if lookup is not None:
            obs = lookup.fill(env.envs, obs)
        ints = torch.from_numpy(obs["ints"]).to(device).long()
        floats = torch.from_numpy(obs["floats"]).to(device)
        mask = torch.from_numpy(obs["mask"]).to(device)
        prac_np = np.asarray(env.envs.practice_mask(), dtype=bool)
        prac = torch.from_numpy(prac_np).to(device)
        with torch.no_grad(), amp():
            logits, _ = net(ints, floats, prac)
        logits = masked(logits.float(), mask)
        if potion_bias:
            room = torch.from_numpy(obs["ints"][:, ROOM_AT]).to(device)
            fight = torch.from_numpy(in_combat(obs["ints"])).to(device)
            for r, bias in potion_bias.items():
                logits[fight & (room == r), POTION:END_TURN] += bias
        acts = logits.argmax(-1).cpu().numpy()
        if peek is not None:
            peek_card_rewards(net, device, amp, env, obs, acts, peek, st, [g for g in range(n) if wanted(g, episode[g])])
        if cards is not None:
            cards.step(env, obs, ints, floats, logits, acts, episode, wanted)
        if search:
            t = time.time()
            games = [g for g in np.flatnonzero(in_combat(obs["ints"])) if wanted(g, episode[g])
                     and (rooms is None or obs["ints"][g, ROOM_AT] in rooms) and (not potions_only or POTION <= acts[g] < END_TURN)
                     and (obs["ints"][g, ROOM_AT] != HALLWAY_ROOM_ID or round(float(obs["floats"][g, ACT_AT])) >= hallway_from)]
            lg_np = logits.cpu().numpy()
            if boss_search is None:
                found = search.scores(env.envs, obs["ints"], obs["mask"], lg_np, games, prac_np)
            else:
                boss = [g for g in games if obs["ints"][g, ROOM_AT] == BOSS_ROOM_ID
                        and (not boss_acts or round(float(obs["floats"][g, ACT_AT])) in boss_acts)]
                found = search.scores(env.envs, obs["ints"], obs["mask"], lg_np, [g for g in games if g not in set(boss)], prac_np)
                found.update(boss_search.scores(env.envs, obs["ints"], obs["mask"], lg_np, boss, prac_np))
            for g, (cands, q, se) in found.items():
                net_q = q[list(cands).index(acts[g])]
                # The best move once each gain is shrunk by `margin_se` standard errors.
                gain = q - net_q
                shrunk = np.sign(gain) * np.maximum(np.abs(gain) - margin_se * se, 0.0)
                best = int(shrunk.argmax()) if margin_se else int(q.argmax())
                st["gap"].append(float(q[best] - net_q))
                if shrunk[best] > margin:
                    st["changed"] += 1
                    acts[g] = cands[best]
            t_search += time.time() - t
        drink = in_combat(obs["ints"]) & (acts >= POTION) & (acts < END_TURN)
        for g in np.flatnonzero(drink):
            if wanted(g, episode[g]) and obs["ints"][g, ROOM_AT] in ROOMS:
                st[f"potions_{ROOMS[obs['ints'][g, ROOM_AT]]}"] += 1
        if habits:
            at_reward = obs["ints"][:, SCREEN_AT] == CARD_REWARD
            picked = (acts >= PICK_AT) & (acts < PICK_AT + MAX_OPTIONS_AT)
            hab[:, HABIT_KEYS.index("cards_taken")] += at_reward & picked
            hab[:, HABIT_KEYS.index("cards_skipped")] += at_reward & ~picked
        status = env.step(acts)
        if habits:
            h = np.asarray(env.envs.habits(), dtype=np.float64).reshape(n, -1)
            hab[:, 0] += h[:, 0]
            hab[:, 1] += h[:, 1]
            hab[:, 2] = np.where(h[:, 2] > 0, h[:, 2], hab[:, 2])
            rested = h[:, 3] >= 0
            hab[:, 3] += rested
            hab[:, 4] += np.where(rested, h[:, 3], 0.0)
            hab[:, 5:10] += h[:, [5, 6, 7, 8, 9]]
        for g in np.flatnonzero(status):
            if wanted(g, episode[g]):
                out.append(json.loads(env.last_episode(int(g))))
                log = out[-1].pop("log", None)  # (--decks records the run for its deck: its log, kept, is 3 GB of RAM on 1,024 runs and nothing reads it)
                if save_wins and out[-1]["won"] and out[-1]["act"] >= 4:
                    os.makedirs(save_wins, exist_ok=True)
                    with open(os.path.join(save_wins, f"{out[-1]['seed']}.json"), "w") as f:
                        json.dump(log, f, indent=1)
                if habits:
                    out[-1]["habits"] = dict(zip(HABIT_KEYS, hab[g].tolist()))
                if cards is not None:
                    out[-1]["picks"] = cards.finish(g)
                if len(out) % 128 == 0:
                    print(f"  {label}: {len(out)} done, {time.time() - t0:.0f} s, won {np.mean([e['won'] for e in out]):.1%}", flush=True)
            episode[g] += 1
            hab[g] = 0
    st["search_seconds"] = t_search
    if search:
        st.update({k: search.stats[k] + (boss_search.stats[k] if boss_search else 0) for k in SEARCH_STATS})
    return out, st


def peek_card_rewards(net, device, amp, env, obs, acts, peek, st, games):
    """`play`'s `peek`: changes `acts` in place where the deck network's value of an option's result beats its own pick's."""
    margin, deck_weight, deck_free, scale = peek
    games = [g for g in games if obs["ints"][g, SCREEN_AT] == CARD_REWARD]
    if not games:
        return
    roots = [(int(g), [int(a) for a in np.flatnonzero(obs["mask"][g])]) for g in games]
    pi, pf, pm, meta = env.envs.peek_after(roots)
    k = len(meta)
    ints = torch.from_numpy(np.frombuffer(pi, dtype="<i4").reshape(k, -1).copy()).to(device).long()
    floats = torch.from_numpy(np.frombuffer(pf, dtype="<f4").reshape(k, -1).copy()).to(device)
    with torch.no_grad():
        _, value = net(ints, floats, torch.zeros(k, dtype=torch.bool, device=device))
    value = value.float().cpu().numpy()
    a, b = RUN_INT_AT["deck"]
    size_now = {g: int((obs["ints"][g, a:b] > 0).sum()) for g in games}
    size_after = (ints[:, a:b] > 0).sum(1).cpu().numpy()
    for r, (g, slots) in enumerate(roots):
        rows = [i for i, m in enumerate(meta) if m[0] == r]
        added = size_after[rows] - size_now[g]
        score = value[rows] - deck_weight * (0.2 + np.maximum(size_after[rows] - deck_free, 0) / 10) * added / scale
        own = [i for i, m in enumerate(rows) if meta[m][1] == acts[g]]
        if not own:
            continue
        best = int(score.argmax())
        st["peeks"] = st.get("peeks", 0) + 1
        if score[best] - score[own[0]] > margin:
            acts[g] = meta[rows[best]][1]
            st["peek_changed"] = st.get("peek_changed", 0) + 1


def shard_episodes(args, name, use_search, shard, lines):
    """Process `shard` of `args.procs`: loads the network and plays its share of `name` ("runs":
    the fixed runs, "bosses": the boss fights in `lines`), with the look-ahead if `use_search`.
    Returns (episodes, stats)."""
    device = pick_device(args.device)
    cap_gpu(device, args.gpu_gb)
    # Several checkpoints (comma-separated): their mean logits (core.split.Ensemble); the first's settings.
    nets = []
    for path in args.checkpoint.split(","):
        n, c = load(path, device)
        n.load_state_dict(c["net"])
        nets.append((n.eval(), c))
    (net, ck), events = nets[0], all(trained_events(c["net"]) for _, c in nets)
    if args.lookup_events != "auto":
        events = args.lookup_events == "on"
    if len(nets) > 1:
        net = Ensemble(n for n, _ in nets)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda")

    # A deck network: its fight network fights and searches; it chooses (with its lookup).
    policy, fighter, lookup, scale_ck = net, net, None, ck
    if ck["args"].get("fight_net"):
        fight_path = ck["args"]["fight_net"]
        fighter = load_fight_net(fight_path, device)
        player = fighter
        if args.fighter:
            # Other fight networks play the fights (several: their mean logits); the lookup stays the deck network's.
            fs = [load_fight_net(f, device) for f in args.fighter.split(",")]
            player = Ensemble(fs) if len(fs) > 1 else fs[0]
        policy = SplitPolicy(player, net)
        boss_net = None
        if args.boss_fighter:
            # A boss specialist plays the boss fights (and searches them, in its own run's reward scale).
            boss_net = load_fight_net(args.boss_fighter, device)
            policy = BossFighter(policy, boss_net, args.boss_fighter_acts)
        if ck["args"].get("lookup"):
            lookup = Lookup(fighter, device, amp, draws=args.lookup_draws or ck["args"].get("lookup_draws", 4),
                            every=ck["args"].get("lookup_every", False), events=events)
        scale_ck = torch.load(args.scale_from or fight_path, map_location="cpu", weights_only=False)
    picker = None
    if args.card_early:
        policy = CardPicker(policy, load_fight_net(args.card_early, device), 1, args.card_from_floor - 1)
    if args.card_net:
        # Another network's picks at card rewards (the search and the lookup keep the fight network above).
        policy = picker = CardPicker(policy, load_fight_net(args.card_net, device), args.card_from_floor)
    elif args.card_route:
        # The picker for each run's deck as it stands (the search and the lookup keep the fight network here too).
        nets = {path: load_fight_net(path, device) for path in set(args.card_route.values())}
        policy = picker = ArchetypeRouter(policy, {a: nets[path] for a, path in args.card_route.items()}, args.card_from_floor, args.card_route_default)
    search = None
    if use_search:
        if args.fighter:
            fighter = load_fight_net(args.fighter.split(",")[0], device)
        search = FightSearch(fighter, device, amp, k=args.k, top=args.top, turns=args.turns, seed=args.seed, scale=reward_std(scale_ck),
                             potion_bonus=scale_ck["args"].get("practice_potion_bonus", 0.0), max_moves=args.max_moves, value=args.value,
                             potion_value=args.potion_value, roots_per_call=args.roots_per_call, batch=args.pass_rows,
                             fixed=FixedShapes(fighter, cuda_graphs=device.type == "cuda") if args.fixed else None)
    boss_search = None
    if use_search and args.boss_fighter:
        boss_ck = torch.load(args.boss_fighter, map_location="cpu", weights_only=False)
        boss_search = FightSearch(boss_net, device, amp, k=args.boss_k or args.k, top=args.boss_top or args.top,
                                  turns=args.boss_turns or args.turns, seed=args.seed, scale=reward_std(boss_ck),
                                  potion_bonus=boss_ck["args"].get("practice_potion_bonus", 0.0), max_moves=args.max_moves,
                                  value=args.value, potion_value=args.potion_value, roots_per_call=args.roots_per_call,
                                  batch=args.pass_rows, fixed=FixedShapes(boss_net, cuda_graphs=device.type == "cuda") if args.fixed else None)
    elif use_search and (args.boss_k or args.boss_top or args.boss_turns):
        # The boss fights' own settings (the rest as above), sharing the network's compiled passes.
        boss_search = FightSearch(fighter, device, amp, k=args.boss_k or args.k, top=args.boss_top or args.top,
                                  turns=args.boss_turns or args.turns, seed=args.seed, scale=reward_std(scale_ck),
                                  potion_bonus=scale_ck["args"].get("practice_potion_bonus", 0.0), max_moves=args.max_moves,
                                  value=args.value, potion_value=args.potion_value, roots_per_call=args.roots_per_call,
                                  batch=args.pass_rows, fixed=search.fixed)
    label = f"{name}{f'[{shard}]' if args.procs > 1 else ''}"
    if name == "runs":
        # Run j of the whole set has seed EVAL_SEED + j: game g's episode e here is run first + e * batch + g.
        per = -(-args.runs // args.procs)
        first = shard * per
        count = max(min(per, args.runs - first), 0)
        b = max(min(args.batch, count), 1)
        env = RunEnv(b, seed=EVAL_SEED + args.seed_offset + first, ascension=args.ascension, max_steps=8000, character=args.character,
                     record=args.decks or bool(args.save_wins), detail=args.decks)
        if args.free_keys:
            env.envs.set_free_keys(True)
            env.envs.reset()  # (it holds from the next episode, and the first ones are already dealt)
        if args.key_rule:
            env.envs.set_key_rule(True, *(int(x) for x in args.key_rule.split(",")))
            env.envs.reset()
        rooms = None if args.search_rooms == "all" else {k for k, v in ROOMS.items() if v in args.search_rooms.split(",")}
        bias = {k: float(dict(x.split(":") for x in args.potion_bias.split(",")).get(v, 0.0)) for k, v in ROOMS.items()} \
            if args.potion_bias else None
        cards = None
        if args.picks:
            cards = CardChoices(b, (picker, args.seed, lambda g, e: EVAL_SEED + args.seed_offset + first + e * b + g) if args.card_sample else None,
                                router=picker if args.card_route else None)
        out, st = play(policy, device, amp, env, lambda g, e: e * b + g < count, search, args.margin, label, args.margin_se, lookup,
                       rooms, bias, args.search_potions, boss_search,
                       (args.peek_margin, ck["args"].get("deck_weight", 0.0), ck["args"].get("deck_free", 0), reward_std(ck))
                       if args.peek_margin is not None else None, args.boss_fighter_acts if args.boss_fighter else None,
                       args.habits, cards, args.save_wins, args.search_hallway_from)
        if args.dump:
            # Every run's summary (its seed, how it ended, where it died) for analysis (analysis/run_deaths.py).
            with open(f"{args.dump}.{shard}.jsonl", "w") as f:
                for e in out:
                    f.write(json.dumps(e) + "\n")
        return out, st
    # Boss fight j has seed SCENARIO_EVAL_SEED + j (Envs.set_scenarios, sequential).
    per = -(-len(lines) // args.procs)
    mine = lines[shard * per:(shard + 1) * per]
    if not mine:
        return [], {"changed": 0, "gap": [], "search_seconds": 0.0, **{f"potions_{r}": 0 for r in ROOMS.values()}}
    b = min(args.boss_batch, len(mine))
    env = RunEnv(b, seed=SCENARIO_EVAL_SEED + shard * per, practice=1.0, max_steps=4000, character=args.character)
    env.envs.set_scenarios(mine, 1.0, True)
    return play(fighter, device, amp, env, lambda g, e: e * b + g < len(mine), search, args.margin, label, args.margin_se)


def worker(args, name, use_search, shard, lines, results):
    try:
        results.put((shard, shard_episodes(args, name, use_search, shard, lines)))
    except Exception:
        results.put((shard, traceback.format_exc()))


def run_set(args, name, use_search, lines):
    """Plays `name` in `args.procs` processes; returns the episodes and the stats, merged."""
    t0 = time.time()
    if args.procs == 1:
        parts = [shard_episodes(args, name, use_search, 0, lines)]
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        results = ctx.Queue()
        procs = [ctx.Process(target=worker, args=(args, name, use_search, i, lines, results)) for i in range(args.procs)]
        for p in procs:
            p.start()
        got = dict(results.get() for _ in procs)
        for p in procs:
            p.join()
        for i, r in got.items():
            if isinstance(r, str):
                raise RuntimeError(f"{name} shard {i} failed:\n{r}")
        parts = [got[i] for i in range(args.procs)]
    out = [e for o, _ in parts for e in o]
    st = {"seconds": time.time() - t0, "changed": sum(s["changed"] for _, s in parts), "gap": [x for _, s in parts for x in s["gap"]],
          "search_seconds": sum(s["search_seconds"] for _, s in parts) / len(parts)}
    for k in SEARCH_STATS + tuple(f"potions_{r}" for r in ROOMS.values()) + ("peeks", "peek_changed"):
        st[k] = sum(s.get(k, 0) for _, s in parts)
    return out, st


def describe(st, use_search, episodes):
    potions = "potions a run: " + " ".join(f"{r} {st[f'potions_{r}'] / max(episodes, 1):.2f}" for r in ROOMS.values())
    if st.get("peeks"):
        potions += f"; card picks changed by the value peek {st['peek_changed']} of {st['peeks']}"
    if not use_search:
        return f"{potions} ({st['seconds']:.0f} s)"
    n = max(st["searches"], 1)
    gap = np.array(st["gap"]) if st["gap"] else np.zeros(1)
    return (f"{potions} ({st['seconds']:.0f} s, {st['search_seconds']:.0f} of a process's in the search; {st['searches']} searches, "
            f"{st['playouts'] / n:.0f} playouts and {st['playout_moves'] / max(st['playouts'], 1):.1f} moves each; changed the move "
            f"{st['changed'] / n:.1%}; best - network score p50 {np.percentile(gap, 50):.4f} p90 {np.percentile(gap, 90):.4f})")


def runs_line(out, st, use_search):
    r = summarize(out)
    line = (f"won {r['win_rate']:.1%} of {len(out)} (the Heart {r['win_heart']:.1%}), floor {r['floor']:.1f}, reached A2 {r['reached_act2']:.0%} "
            f"A3 {r['reached_act3']:.0%} A4 {r['reached_act4']:.0%}, " + describe(st, use_search, len(out)))
    if out and "habits" in out[0]:
        m = {k: np.mean([e["habits"][k] for e in out]) for k in HABIT_KEYS}
        line += (f"; a run: cards taken {m['cards_taken']:.1f} skipped {m['cards_skipped']:.1f}, added {m['cards_added']:.1f} removed "
                 f"{m['cards_removed']:.1f}, deck {m['deck']:.1f}, rests {m['rests']:.1f} (at HP {m['rest_hp'] / max(m['rests'], 1e-9):.2f}), "
                 f"smiths {m['smiths']:.1f}, elites won {m['elites_won']:.1f}, hallway potions {m['hallway_potions']:.2f}, "
                 f"removals skipped {m['removals_skipped']:.2f}, Strikes upgraded {m['strikes_upgraded']:.2f}")
    return line


def bosses_line(out, st, use_search):
    parts = [f"won {np.mean([e['won'] for e in out]):.1%} of {len(out)}, HP left {np.mean([practice_reward(e) for e in out]):.3f}"]
    for a in BOSS_KINDS:
        eps = [e for e in out if e.get("practice") == a]
        if eps:
            parts.append(f"{a} {np.mean([e['won'] for e in eps]):.1%} of {len(eps)}")
    act3 = [e for e in out if e.get("practice") == "boss3"]
    if act3:
        parts.append(f"first boss {np.mean([e.get('bosses_beaten', 0) >= 1 for e in act3]):.1%}")
    return ", ".join(parts) + ", " + describe(st, use_search, len(out))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint")
    p.add_argument("--set", choices=["runs", "bosses", "both"], default="both")
    p.add_argument("--runs", type=int, default=1024)
    p.add_argument("--batch", type=int, default=256, help="runs at once in a process (train_runs.py's evals use 256)")
    p.add_argument("--boss-batch", type=int, default=128, help="boss fights at once in a process (as train_runs.py)")
    p.add_argument("--ascension", type=int, default=20)
    p.add_argument("--character", default="ironclad", choices=["ironclad", "silent", "defect", "watcher"], help="whose runs and boss fights (--scenarios: that character's, e.g. ~/sts-data/bosses-watcher/eval.jsonl.gz)")
    p.add_argument("--scenarios", default="~/sts-data/bosses/eval.jsonl.gz", help="held-out boss fights (human_bosses.py)")
    p.add_argument("--k", type=int, default=8, help="redrawn copies per first move")
    p.add_argument("--top", type=int, default=4, help="the network's likeliest moves tried (and ending the turn)")
    p.add_argument("--turns", type=int, default=1, help="turns each playout runs")
    p.add_argument("--max-moves", type=int, default=30, help="moves at most in a playout")
    p.add_argument("--value", choices=["run", "fight", "outcome"], default="run", help="how a run's fights are scored (fight_search.py)")
    p.add_argument("--potion-value", type=float, default=None, help="a potion kept outside boss fights, in rewards (fight_search.py)")
    p.add_argument("--seed", type=int, default=1, help="seed for redrawing the copies (and for --card-sample's draws)")
    p.add_argument("--margin", type=float, default=0.0, help="play the search's move only if it beats the network's by this much")
    p.add_argument("--margin-se", type=float, default=0.0,
                   help="... after shrinking each move's gain by this many standard errors (the copies' luck; fight_search.py)")
    p.add_argument("--no-baseline", action="store_true", help="skip plain greedy play")
    p.add_argument("--no-search", action="store_true", help="plain greedy play only (a fast eval of a checkpoint on the GPU)")
    p.add_argument("--free-keys", action="store_true", help="runs start with the three keys: what its decks would do at the Heart, keys' cost aside (no effect on --set bosses)")
    p.add_argument("--key-rule", nargs="?", const="101,60", default=None, metavar="ACT2,ACT3",
                   help="for a network trained with keys free: a hand-written rule gets the real keys (RunConfig::key_rule); from what percent of HP it takes Recall in Act 2 and Act 3. Results with it say so: the keys aren't the network's")
    p.add_argument("--no-fixed", dest="fixed", action="store_false",
                   help="the search's copies through the usual pass (default: the compiled fixed-shape passes, fight_search.BUCKETS)")
    p.add_argument("--procs", type=int, default=1, help="processes, each playing a share of the runs (see above)")
    p.add_argument("--pass-rows", type=int, default=4096, help="copies per network pass in the search (GPU memory)")
    p.add_argument("--roots-per-call", type=int, default=256, help="games searched at once (RAM: each holds about 40 copies of its run)")
    p.add_argument("--gpu-gb", type=float, default=None,
                   help="GPU memory cap per process, GB (PyTorch's own, without the CUDA context's ~0.45 GB; 0 = none). "
                        "Default: 11 GB shared out, at most 3 GB a process, so the total stays under 16 GB")
    p.add_argument("--device", default="auto")
    p.add_argument("--lookup-draws", type=int, default=0,
                   help="a deck network's lookup: the openings each probe reads (default as it trained)")
    p.add_argument("--lookup-events", choices=["auto", "on", "off"], default="auto",
                   help="a deck network's lookup at events and Neow: auto, as it trained (its pick weights for them non-zero)")
    p.add_argument("--fighter", default=None,
                   help="a deck network's checkpoint: other fight networks' checkpoints to play its fights, comma-separated (their "
                        "mean logits; the look-ahead searches with the first); its lookup keeps the fight network it trained with")
    p.add_argument("--search-hallway-from", type=int, default=1, metavar="ACT",
                   help="runs: the look-ahead plays hallway fights from this act on (in Act 1 it cost HP the elites then took: 2026-11-08)")
    p.add_argument("--search-rooms", default="all",
                   help="runs: the rooms whose fights the look-ahead plays, comma-separated (hallway, elite, boss), or all")
    p.add_argument("--peek-margin", type=float, default=None,
                   help="runs: at card rewards, take the option whose result the deck network values most if ahead of its own "
                        "pick by this much (value units; Envs.peek_after)")
    p.add_argument("--boss-fighter", default=None,
                   help="a deck network's checkpoint: another fight network's checkpoint for the boss fights only (a boss specialist)")
    p.add_argument("--boss-fighter-acts", type=int, nargs="+", default=None,
                   help="--boss-fighter: in these acts' boss fights only (default every act's)")
    p.add_argument("--boss-k", type=int, default=0, help="runs: the look-ahead's copies in boss fights (default --k)")
    p.add_argument("--boss-top", type=int, default=0, help="runs: the look-ahead's candidate moves in boss fights (default --top)")
    p.add_argument("--boss-turns", type=int, default=0, help="runs: the look-ahead's turns in boss fights (default --turns)")
    p.add_argument("--decks", action="store_true", help="runs: each run's deck and relics at the end in its summary (--dump)")
    p.add_argument("--save-wins", default=None, help="runs: a folder for each Heart win's log, <seed>.json, to watch with tools/trace/watch.sh or in the viewer")
    p.add_argument("--habits", action="store_true", help="runs: count each run's choices into its summary (and the line)")
    p.add_argument("--seed-offset", type=int, default=0,
                   help="runs: run j has seed EVAL_SEED + this + j (e.g. 1024: seeds no checkpoint was chosen on)")
    p.add_argument("--search-potions", action="store_true",
                   help="runs: the look-ahead plays only the moves where the network would drink a potion")
    p.add_argument("--potion-bias", default=None,
                   help="runs: added to the potion moves' logits by room, e.g. hallway:-2,elite:-1,boss:1 (a rule for when to drink)")
    p.add_argument("--card-net", default=None,
                   help="runs: another network's checkpoint picks at card rewards (its logits on those rows only; the fights and every "
                        "other choice stay the checkpoint's); implies --picks")
    p.add_argument("--card-route", default=None,
                   help="runs: a network's checkpoint for each archetype, e.g. grand-finale=PATH,wraith-form=PATH,discard=PATH,poison=PATH (analysis/archetypes.py's names, - for "
                        "spaces): each card reward goes to the one for that run's deck as it stands (the fights and every other choice stay the checkpoint's); implies --picks")
    p.add_argument("--card-route-default", choices=["base", "max"], default=None,
                   help="--card-route: a deck whose archetype isn't named: max (the default), the pickers together (each card its largest probability over them, the skip its "
                        "smallest); or base, the checkpoint's own choice")
    p.add_argument("--card-from-floor", type=int, default=1,
                   help="--card-net, --card-route: at card rewards on this floor or later (the default, 1, leaves Neow's, on floor 0, to the checkpoint)")
    p.add_argument("--card-early", default=None,
                   help="another network's picks at card rewards below --card-from-floor (from floor 1), before --card-net's or --card-route's (jack, 2026-11-06: "
                        "\"take some good damage in first act, then build\"); with neither, the checkpoint's from there on")
    p.add_argument("--card-sample", action="store_true",
                   help="--card-net, --card-route: each pick drawn from its masked softmax (seeded by --seed and the run), not its top pick")
    p.add_argument("--picks", action="store_true",
                   help="runs: print the skips and the cards taken by CARD_BUCKETS; each run's card rewards in its summary (--dump)")
    p.add_argument("--dump", default=None, help="write every run's summary to <DUMP>.<process>.jsonl (runs only)")
    p.add_argument("--scale-from", default=None,
                   help="a deck network's checkpoint: the run checkpoint whose reward scale its fight network's values are in "
                        "(default the fight network's own; a --head-only outcome head's run: the run it started from)")
    args = p.parse_args()
    if args.gpu_gb is None:
        args.gpu_gb = min(3.0, 11.0 / args.procs)
    if args.card_net and args.peek_margin is not None:
        p.error("--card-net and --peek-margin both choose at card rewards")
    if args.card_route is not None:
        try:
            args.card_route = parse_card_route(args.card_route)
        except ValueError as e:
            p.error(str(e))
        if args.card_net or args.peek_margin is not None:
            p.error("--card-route with --card-net or --peek-margin: both would choose at card rewards")
        if args.character != "silent":
            p.error("--card-route: the archetypes are the Silent's (--character silent)")
        if args.set == "bosses":
            p.error("--card-route picks in whole runs: with --set bosses it would do nothing")
    elif args.card_route_default:
        p.error("--card-route-default chooses how --card-route picks: without it, it would do nothing")
    args.card_route_default = args.card_route_default or "max"
    if args.card_early and args.card_from_floor <= 1:
        sys.exit("--card-early needs a --card-from-floor above 1: the floor its picks stop")
    if (args.card_sample or args.card_from_floor != 1) and not (args.card_net or args.card_route or args.card_early):
        p.error("--card-sample and --card-from-floor choose how --card-net or --card-route picks: without them they would do nothing")
    args.picks = args.picks or bool(args.card_net or args.card_route)
    ck = torch.load(args.checkpoint.split(",")[0], map_location="cpu", weights_only=False)
    if args.fighter and not ck["args"].get("fight_net"):
        p.error("--fighter only acts on a deck network's checkpoint (args fight_net): this one would play its own fights")
    if args.card_net and not ck["args"].get("lookup") and torch.load(args.card_net, map_location="cpu", weights_only=False)["args"].get("lookup"):
        p.error("--card-net: this network trained with a lookup and nothing would fill its inputs (only a checkpoint with its own lookup does)")
    route_steps = {}
    for a, path in (args.card_route or {}).items():
        route = torch.load(path, map_location="cpu", weights_only=False)
        route_steps[a] = route.get("pretrain_step", route.get("update"))  # (a latest.pt moves while its picker trains)
        if not ck["args"].get("lookup") and route["args"].get("lookup"):
            p.error(f"--card-route: {path} trained with a lookup and nothing would fill its inputs (only a checkpoint with its own lookup does)")
    print(f"{args.checkpoint} (update {ck['update']}); search: top {args.top} + end turn, {args.k} copies, {args.turns} turn(s), "
          f"at most {args.max_moves} moves, {args.value} value, potion value {args.potion_value}, margin {args.margin} + {args.margin_se} SE; reward scale {reward_std(ck):.3f}; "
          f"{args.procs} process(es)", flush=True)
    if args.card_net:
        print(f"card picks: {args.card_net} from floor {args.card_from_floor}, {'sampled' if args.card_sample else 'top pick'}", flush=True)
    if args.card_route:
        print(f"card picks by the deck's archetype from floor {args.card_from_floor}: " + ", ".join(f"{a} {path} (step {route_steps[a]})" for a, path in args.card_route.items())
              + f"; any other deck: {args.card_route_default}; {'sampled' if args.card_sample else 'top pick'}", flush=True)
    del ck
    sets = ["runs", "bosses"] if args.set == "both" else [args.set]
    lines = read_scenarios(os.path.expanduser(args.scenarios)) if "bosses" in sets else []
    for name in sets:
        for use in ([False] if not args.no_baseline else []) + ([True] if not args.no_search else []):
            out, st = run_set(args, name, use, lines)
            line = runs_line(out, st, use) if name == "runs" else bosses_line(out, st, use)
            print(f"{name} {f'A{args.ascension} ' if name == 'runs' else ''}{'search' if use else 'greedy'}: {line}", flush=True)
            if name == "runs" and args.picks:
                print("\n".join([act3_line(out), *picks_lines(out)]), flush=True)
                if args.card_route:
                    print("\n".join(route_lines(out, [*args.card_route, *(["max"] if args.card_route_default == "max" else []), "own"])), flush=True)


if __name__ == "__main__":
    main()
