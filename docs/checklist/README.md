# Port checklist: everything in the game

Every card, relic, potion, power, monster, event, orb, stance and character in the decompiled game
(`reference/`), with the engine's status and two boxes per item. It is the tracker for porting
the whole game to Rust.

| File | What |
|---|---|
| [cards.md](cards.md) | 375 cards by color, plus curses, status, temporary and option cards |
| [relics.md](relics.md) | 186 relics by character |
| [potions.md](potions.md) | 42 potions by character |
| [powers.md](powers.md) | 161 powers (the Watcher's in their own group) |
| [monsters.md](monsters.md) | 68 monster classes by act |
| [events.md](events.md) | 52 events and shrines by act |
| [other.md](other.md) | characters, orbs, stances; blights (Endless) and daily mods, optional |

## Status on 2026-10-19 (the Silent, the Defect and the Watcher added the same day)

| | Ironclad | Silent | Defect | Watcher | Shared |
|---|---|---|---|---|---|
| Cards | 75 of 75 | 75 of 75 | 75 of 75 (Impulse, a 76th, is in no library: n/a) | 75 of 75 (Discipline and Unraveling are in no library: n/a) | colorless 39, curses 14, status 5: all ported. Temporary cards 9 of 9 (Shiv, the Silent's; the rest the Watcher's), option cards 5 of 5 (the Watcher's) |
| Relics | 11 of 11 | 11 of 11 | 9 of 9 | 9 of 9 | 139 of 139 |
| Potions | 3 of 3 | 3 of 3 | 3 of 3 | 3 of 3 | 30 of 30 |
| Powers | | 18 | 17 | 21 of 26 (5 are created by nothing: n/a) | 112 of 123 general powers (the Silent's, the Defect's and the Watcher's Collect among them); 11 are created by nothing (n/a) |
| Mechanics | done | done: poison, Shivs and Accuracy, discard triggers and counts, retain, Night Terror | done: orbs (all 5), Focus, channel and evoke, Lock-On | done: stances (all 4), Mantra and Divinity, Scry, Retain, Miracles, choose-one cards | |
| Characters | runs and fights | runs and fights; 65 recorded fights and 4 recorded runs match (2026-10-19) | runs and fights; 55 recorded fights and 5 recorded runs match (2026-10-19) | runs and fights; 67 recorded fights and 7 of 8 recorded runs match (2026-10-19) | |

Monsters and events are all ported (the "n/a" rows are classes no game reaches: Hexaghost's drawn
parts, an encounter fallback, the victory room). Blights and daily mods belong to
Endless mode and daily climbs, which training doesn't need; they are listed so nothing is
forgotten.

Statuses:
- **ported**: the engine has it and its logic;
- **data only**: in the generated data tables, logic not written yet;
- **missing**: not in the engine;
- **gap**: in the engine but incomplete (with the reason);
- **n/a**: in the game's code but never reached by a game (with the reason).

## The boxes

- **Unit test**: a test in `crates/sts-core` sets up the situation (a fight with the card in hand,
  the relic held, the power applied...) and checks the result against what the game's code does:
  damage and block numbers, upgrades, the action order, every hook the item has (start and end of
  turn, on play, on attacked, on exhaust...), and the RNG streams it draws from. Tick it when that
  test exists and passes.
- **Real game**: the item has been exercised in the real game and matched: a recording replayed by
  `sts-cli replay`/`replay-run`, a playback or a soak. Tick it when the match is confirmed for that
  item (not just "a run containing it matched somewhere").

Every ported item has its unit-test box ticked (2026-10-19). The real-game boxes are ticked by
`tools/trace/tick_coverage.py` from `sts-cli coverage` over the recordings and matched soak runs
(what counts as exercised is in `crates/sts-cli/src/coverage.rs`): a card played, a potion drunk,
a power present, a monster that took a turn, an event with an option taken, and a relic whose
effect showed (replayed without it, the recording no longer matches or the room plays out
differently).

## Regenerating

```bash
python3 tools/extract/port_checklist.py
```

The statuses are worked out from the engine each time (its data tables and id maps in
`crates/sts-core`); ticked boxes are kept, matched by ID. Run it after porting something and commit
the result.
