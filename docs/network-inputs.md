# What the network sees in a fight

Encoding version 2 (2026-10-24, `ENCODING_VERSION` in `crates/sts-encode/src/encode.rs`). Every
decision becomes one row of integers (ids into learned embeddings, 0 = empty) and one row of
floats (raw game numbers; the network scales each into symlog(x) and x / 100). This page covers
the fight's part; between fights the run encoding (`encode_run.rs`) lists the screen's options.

Code: `crates/sts-encode/src/encode.rs` (`encode_fight`, `hand_row`, `play_damage_against`),
`crates/sts-core/src/combat/damage.rs` (`Combat::card_display`, `card_damage_against`,
`card_hits`, `x_energy`), `python/core/fight_model.py` (tokens, `pair_features`, `Scorer`),
`crates/sts-net` (the same network in Rust, for the browser), `crates/sts-py/src/advise.rs` (the
same fields from the real game's state, for the in-game advisor). Tests: `encode.rs`'s tests,
`crates/sts-core/tests/hits.rs`, `python/tests/test_upgrade.py`.

## Tokens

The network is a transformer over tokens, one per thing in the fight:

- **global**: the scalars below, the character, a pending card selection's kind and pile, the
  type of the last card played this combat (Follow-Up, Crush Joints, Sash Whip and Sanctity look
  at it);
- **player**: the scalars, the stance (Neutral, Calm, Wrath, Divinity), the powers with their
  amounts (Strength, Focus, Mantra...);
- **relics** (pooled, with counters), **draw, discard and exhaust piles** (pooled cards; the draw
  pile sorted, as its order is hidden);
- **10 hand cards** (the row below), **6 monsters**, **5 potions**;
- **orb slots**, front first (the next one evoked), up to 10 (the game's most): each the orb's
  kind (or Empty), its passive and evoke numbers as the game shows them (Focus applied, a Dark
  orb's grown evoke) and its position, so the network knows which orb Multicast, Dualcast,
  Recursion, Loop and Gold-Plated Cables act on, and whether channeling into full slots evokes;
- **64 selection options** (the cards of a pending choice).

A card is scored against each target from its token and the target's (`play(h, t)`), plus the
pair features below.

## Scalars (20)

HP, max HP, block, energy, energy per turn, turn, hand / draw / discard / exhaust sizes, gold,
ascension, cards played this turn, a selection's count, whether it takes any number, cards
picked; attacks played this turn (Finisher), cards discarded this turn (Sneaky Strike,
Eviscerate), orb slots, an extra turn coming (Vault).

## Hand row (`HAND_FEATS = 15`, one row per hand slot)

| # | Feature | Notes |
|---|---------|-------|
| 0 | cost this turn | `cost_for_turn` (Corruption's 0, Enlightenment, Snecko and so on) |
| 1 | times upgraded | |
| 2 | damage as shown | one hit, the number on the card in game, before the target (below); 0 for non-attacks |
| 3 | block as shown | the number on the card in game |
| 4 | magic number | unchanged |
| 5 | playable | 1 if a legal action plays it |
| 6 | needs a target | |
| 7 | exhausts | the card's Exhaust, an exhaust-once flag (e.g. from a potion), or a Skill under Corruption |
| 8 | hits | how many times one play hits (`Combat::card_hits`); 1 for other attacks, 0 for non-attacks |
| 9 | every enemy | each hit lands on every enemy (Cleave, Whirlwind, Dagger Spray...) |
| 10 | random enemies | each hit lands on a random enemy (Sword Boomerang, Rip and Tear, Thunder Strike, Ragnarok) |
| 11 | X | the energy an X-cost card would spend, 2 more with Chemical X; else 0 |
| 12 | free | costs nothing this time though its cost says otherwise (free once, Swivel's Free Attack) |
| 13 | ethereal | exhausted at the end of the turn if still in hand |
| 14 | kept | stays in hand at the end of the turn: Retain, or (unless Ethereal) Runic Pyramid or Equilibrium |

### Hits (column 8)

As the cards' `use` counts them: Twin Strike, Dagger Spray, Glass Knife, Flying Sleeves 2,
Eviscerate 3, Riddle with Holes 5; the magic number for Pummel, Tantrum, Expunger, Sword Boomerang,
Rip and Tear and Ragnarok; X for Whirlwind and Skewer; and from the fight: Finisher (attacks
played before it this turn), Flechettes (skills in hand), Barrage (orbs channeled), Bowling Bash
(monsters alive), Fiend Fire (the other cards in hand), Thunder Strike (Lightning channeled this
combat). Bane hits a poisoned target twice: 1 in the hand row, 2 against that monster below. Each
count is tested by playing the card (`tests/hits.rs`).

### What "as shown" includes (columns 2 and 3)

Only attacks show damage and only cards with base block show block. The same calculation the game
runs to colour the card's numbers (`applyPowers`):

- Damage dealt: Strength, Vigor, Weakened (×0.75), Pen Nib (×2), the stance (Wrath ×2, Divinity ×3).
- Heavy Blade: Strength counted 3 times (5 upgraded). Body Slam: the player's block. Perfected
  Strike: +2 (+3) per Strike in the deck. Strike Dummy: +3 on Strikes. Brilliance: the Mantra made
  this combat. Mind Blast: the draw pile. Blizzard: the Frost channeled.
- Block: Dexterity, Frail (×0.75); No Block makes it 0. Stack, Spirit Shield: their own counts.

## Damage against each monster

Two float fields of 10 × 6, for every hand card and monster slot:

- `hand_target_damage`: one hit (`card_damage_against`), so it includes what the target adds:
  Vulnerable (×1.5, ×1.75 with Paper Frog), Slow, Flight, Intangible.
- `hand_target_total`: one whole play (`play_damage_against`): that times its hits there; random
  hits spread evenly over the monsters they can land on. Judgement, a skill, counts as the
  monster's HP and block when it would kill it (HP at or below its number), else 0.

In the network (`pair_features`) each (card, target) pair gets 8 numbers:

- a monster (t = 1-6): one hit's damage (two numbers), whether it kills (damage ≥ HP + block) and
  the share of HP + block it removes (capped at 2); then the same four for the whole play;
- no target (t = 0), for attacks on every enemy or random enemies: four zeros, then the damage
  that lands summed over the monsters (each capped at its HP + block; two numbers), how many it
  kills and whether it kills them all (ends the fight). Zeros for any other card.

## Monsters

Each monster token: its id, its intent, HP, max HP, block, the intent's damage per hit (with its
powers) and hits, targetable, half dead, escaping, its powers with amounts, and its **moves on its
previous two turns**, the last first (`moves_seen`; its own vocabulary per monster, id = monster
× 16 + move). A player remembers what each monster did and knows its patterns (Jaw Worm never
Chomps twice running, the Guardian's modes); the move it is about to make is its intent. Runic
Dome hides intents (and their damage), not the moves already seen.

## Older networks

A network trained on encoding 1 can't read version 2 rows. Released networks keep playing on the
engine they were released with (the site loads it, `notes/hosting.md`). Networks for encoding 2
are trained from scratch (jack, 2026-10-24). An encoding 1 checkpoint can still be loaded, to
measure it on this build or as a baseline: `run_model.upgrade_state` carries it over: every old weight stays
in place, Feel No Pain's and Dark Embrace's hand columns (8 and 9 in encoding 1: the block and
cards drawn from exhausting the card) are dropped as too detailed, the weights on old tokens' new
inputs start at zero and the orb tokens' own weights fresh. It plays as before until training
uses the new inputs, except where Feel No Pain or Dark Embrace mattered and in fights with orbs
(`python/tests/test_upgrade.py`).

Encoding 1 (`hand-info`, 2026-10-16) had brought the displayed numbers, the exhaust flags and the
one-hit damage against each monster; before it, a hand card was its id, cost, upgrades and base
numbers.

## Not represented yet

Candidates if the networks still misplay these:

- The magic number (column 4) means something different on every card: Bash's Vulnerable,
  Shrug It Off's draw, Heavy Blade's Strength multiplier. Next: split it into what it does (draw,
  energy, Strength, Vulnerable, Weak, HP lost, ...).
- Cards outside the hand show only their id and upgrades: changed costs (Madness, Streamline) and
  grown or shrunk numbers (Claw, Rampage, Ritual Dagger, Genetic Algorithm, Glass Knife, Steam
  Barrier, Windmill Strike) don't show in the piles, nor in the deck between fights.
- What playing a card costs: Sharp Hide, Thorns per hit, Beat of Death, Time Warp, Gremlin Nob's
  Enrage (the powers and their amounts are there; the sum isn't).
- Cards played twice: Double Tap, Burst, Amplify, Echo Form, Duplication, Necronomicon (only the
  powers and the relic show it).
- Rage, Juggernaut, Flame Barrier's returns; Sentinel's and Dropkick's conditional energy.
- The draw order under Frozen Eye (the draw pile is always sorted).
- From the real game (the in-game advisor): the cards and attacks played this turn, the last card
  type played, Retain given during a fight and the Lightning channeled are not in CommunicationMod's
  state, so the advisor leaves them at 0.
