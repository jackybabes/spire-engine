"""What a run network chooses between fights, counted from the arrays it sees (no JSON, so it
is cheap enough to run inside every training eval). train_runs.py writes `summary()` to
<run>/choices.csv at each eval, and live_chart.py charts it.

    stats = ChoiceStats(n)
    stats.step(obs, actions, counted)   # before env.step, `counted` = games whose episode counts
    stats.end(g)                        # when game g's counted episode ends
    row = stats.summary()
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # python/: the sts module and these folders

import collections

import numpy as np

from core.sts_env import LAYOUT

SCREENS = ["-"] + LAYOUT["screens"]
KINDS = ["-"] + LAYOUT["option_kinds"]
NAMES = ["-"] + LAYOUT["card_names"]
REST, SHOP, CARD_REWARD, GRID, MAP = (SCREENS.index(s) for s in ("rest", "shop", "card_reward", "grid", "map"))
PICK, _ = LAYOUT["actions"]["pick"]
CONFIRM = LAYOUT["actions"]["confirm"]
POTION, N_POTIONS, TARGETS = LAYOUT["actions"]["potion"]
END_TURN = LAYOUT["actions"]["end_turn"]
COMBAT, BOSS_RELIC, REWARDS = (SCREENS.index(s) for s in ("combat", "boss_relic", "rewards"))
RELICS = ["-"] + LAYOUT["relic_java_ids"]
UNPLAYABLE = {"Wound", "Dazed", "Burn", "Slimed", "Void"}
K = {k: KINDS.index(k) for k in ("rest", "smith", "shop_purge", "grid_remove", "grid_upgrade") if k in KINDS}
ELITE = 2  # i_opt_room: 1 monster, 2 elite, ...

CURSES = {"Regret", "Shame", "Doubt", "Pain", "Parasite", "Writhe", "Decay", "Normality", "Injury", "Clumsy",
          "Necronomicurse", "Curse of the Bell", "Ascender's Bane", "Pride"}
ENGINES = {
    "exhaust": {"Corruption", "Feel No Pain", "Dark Embrace"},
    "block": {"Barricade", "Entrench", "Body Slam", "Juggernaut"},
    "strength": {"Demon Form", "Limit Break", "Inflame", "Spot Weakness"},
}

FIELDS = ["rest_visits", "rest", "smith", "smith_hp85", "smith_hp70", "smith_hp50", "shop_visits", "removals_per_run",
          "removal_at_shop", "removed_curse", "removed_strike", "removed_defend", "card_rewards", "card_skip",
          "elite_taken", "deck_size", "deck_strikes", "deck_defends", "deck_curses", "deck_upgraded",
          "engine_exhaust", "engine_block", "engine_strength", "top_removed", "top_upgraded",
          "end_turn_energy", "potion_discards", "potion_drinks", "potion_rewards_left", "top_boss_relics"]


class ChoiceStats:
    def __init__(self, n):
        self.c = collections.Counter()
        self.removed = collections.Counter()
        self.upgraded = collections.Counter()
        self.rest_hp = []  # (HP fraction, chose smith)
        self.last_deck = [None] * n
        self.last_up = [None] * n
        self.decks = []
        self.boss_relics = collections.Counter()

    def step(self, obs, acts, counted):
        screen = obs["i_screen"][:, 0]
        self.c["decisions"] += int(counted.sum())
        is_potion = (acts >= POTION) & (acts < POTION + N_POTIONS * TARGETS)
        for g in np.flatnonzero(counted & (screen == COMBAT)):
            a = acts[g]
            if is_potion[g]:
                self.c["potion_drinks"] += 1
            elif a == END_TURN:
                # Ending a turn with energy left and a card it could play (not a status or curse).
                energy = obs["f_scalars"][g][3]
                hand = obs["i_hand_card"][g]
                f = obs["f_hand"][g].reshape(len(hand), -1)
                self.c["end_turns"] += 1
                self.c["end_turn_energy"] += bool(energy > 0 and any(
                    hand[h] and f[h][5] > 0 and f[h][0] <= energy and NAMES[hand[h]] not in UNPLAYABLE | CURSES for h in range(len(hand))))
        for g in np.flatnonzero(counted & (screen != SCREENS.index("combat"))):
            self.last_deck[g] = obs["i_deck"][g].copy()
            self.last_up[g] = obs["f_deck_upgrades"][g].copy()
            a, sc = acts[g], screen[g]
            kind = obs["i_opt_kind"][g, a - PICK] if PICK <= a < CONFIRM else 0
            card = NAMES[obs["i_opt_card"][g, a - PICK]] if PICK <= a < CONFIRM else None
            if is_potion[g]:
                if (a - POTION) % TARGETS == 1:
                    self.c["potion_discards"] += 1
                else:
                    self.c["potion_drinks"] += 1
                continue
            if sc == BOSS_RELIC and PICK <= a < CONFIRM:
                self.boss_relics[RELICS[obs["i_opt_relic"][g, a - PICK]]] += 1
            if sc == REWARDS and a == CONFIRM and any(obs["i_opt_potion"][g] > 0):
                self.c["potion_rewards_left"] += 1
            if sc == REST:
                s = obs["f_run_scalars"][g]
                self.c["rest_visits"] += 1
                self.c["rest"] += kind == K["rest"]
                self.c["smith"] += kind == K["smith"]
                self.rest_hp.append((s[0] / max(s[1], 1), kind == K["smith"]))
            elif sc == SHOP:
                self.c["shop_visits"] += a == CONFIRM
                self.c["shop_removals"] += kind == K["shop_purge"]
            elif sc == CARD_REWARD:
                self.c["card_rewards"] += 1
                self.c["card_skip"] += a == CONFIRM
            elif sc == GRID and kind == K.get("grid_remove") and card:
                self.removed[card] += 1
            elif sc == GRID and kind == K.get("grid_upgrade") and card:
                self.upgraded[card] += 1
            elif sc == MAP:
                rooms = obs["i_opt_room"][g]
                if any(rooms[j] == ELITE and obs["mask"][g, PICK + j] for j in range(len(rooms))):
                    self.c["elite_offered"] += 1
                    self.c["elite_taken"] += PICK <= a < CONFIRM and rooms[a - PICK] == ELITE

    def end(self, g):
        if self.last_deck[g] is not None:
            ids, up = self.last_deck[g], self.last_up[g]
            self.decks.append([(NAMES[i], u > 0) for i, u in zip(ids, up) if i > 0])
        self.last_deck[g] = self.last_up[g] = None

    def summary(self):
        c, runs = self.c, max(len(self.decks), 1)
        share = lambda a, b: c[a] / c[b] if c[b] else None
        def smith_share(lo, hi):
            sub = [s for f, s in self.rest_hp if lo <= f < hi]
            return float(np.mean(sub)) if sub else None
        removed = sum(self.removed.values())
        base = lambda n: n.rstrip("+")
        rem_share = lambda test: sum(v for k, v in self.removed.items() if test(base(k))) / removed if removed else None
        decks = self.decks or [[]]
        names = [[base(n) for n, _ in d] for d in decks]
        mean = lambda f: float(np.mean([f(d, n) for d, n in zip(decks, names)]))
        top = lambda ctr: "; ".join(f"{k} {v}" for k, v in ctr.most_common(6))
        return {
            "rest_visits": c["rest_visits"], "rest": share("rest", "rest_visits"), "smith": share("smith", "rest_visits"),
            "smith_hp85": smith_share(0.85, 2), "smith_hp70": smith_share(0.7, 0.85), "smith_hp50": smith_share(0.5, 0.7),
            "shop_visits": c["shop_visits"], "removals_per_run": removed / runs,
            "removal_at_shop": share("shop_removals", "shop_visits"),
            "removed_curse": rem_share(lambda k: k in CURSES), "removed_strike": rem_share(lambda k: k == "Strike"),
            "removed_defend": rem_share(lambda k: k == "Defend"),
            "card_rewards": c["card_rewards"], "card_skip": share("card_skip", "card_rewards"),
            "elite_taken": share("elite_taken", "elite_offered"),
            "deck_size": mean(lambda d, n: len(d)), "deck_strikes": mean(lambda d, n: n.count("Strike")),
            "deck_defends": mean(lambda d, n: n.count("Defend")), "deck_curses": mean(lambda d, n: sum(x in CURSES for x in n)),
            "deck_upgraded": mean(lambda d, n: sum(u for _, u in d)),
            **{f"engine_{e}": mean(lambda d, n, cards=cards: len(set(n) & cards) >= 2) for e, cards in ENGINES.items()},
            "top_removed": top(self.removed), "top_upgraded": top(self.upgraded),
            "end_turn_energy": share("end_turn_energy", "end_turns"),
            "potion_discards": c["potion_discards"] / runs, "potion_drinks": c["potion_drinks"] / runs,
            "potion_rewards_left": c["potion_rewards_left"] / runs, "top_boss_relics": top(self.boss_relics),
        }
