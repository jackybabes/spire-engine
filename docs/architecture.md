# Architecture

How the pieces fit together and why they're built the way they are: the engine, the CLI and
Python bindings, the modded real game used as the reference, and the tools that check one
against the other (recordings, replays, playback and the soak). The port's status item by item
is in `docs/checklist/`; the released network is described in `docs/models/`.

## The goal shapes everything

A neural network has to play millions of Ironclad runs, so the engine must be fast, headless and
deterministic. And it's only useful if it plays the *real* game, so every rule is ported from the
decompiled source and checked against the running game, down to RNG counters. Most design
choices below come from holding both at once: be fast enough to train on, and be checkable
against the game at every step.

```
            decompiled game (reference/, kept privately)
                 │  ported by hand, data tables generated (tools/extract)
                 ▼
  ┌──────────────────────────────┐        ┌─────────────────────────────────────┐
  │ sts-core (engine, pure Rust) │◄──────►│ real game + patched CommunicationMod │
  │  Combat, Run, 13 RNG streams │ checks │  (harness/, kept privately)          │
  │  commands.rs: decision→game  │        └─────────────────────────────────────┘
  └──────┬───────────────┬───────┘          ▲ recordings (game→engine): record*.py → replay*
         │               │                  │ playback (engine→game): playback.py, soak.py
     sts-cli          sts-py (PyO3)
  serve/play/run/    Envs, encoders ──► python/ training (PPO on a GPU)
  replay/commands
```

## The engine (`crates/sts-core`)

- **Pure logic, no I/O, one owned state.** The game reads globals everywhere
  (`AbstractDungeon.player`, `getCurrRoom()`); the engine holds everything in `Run` (and its
  `Combat`) and passes it by `&mut`. `Run` is `Clone`, so a search can copy a state and try moves.
- **Closed enums and `match`, not trait objects.** Cards, relics, powers, monsters, events and
  actions are enums with data tables (`cards/data.rs` etc., generated from `reference/` by
  `tools/extract`). Fast, cloneable, serialisable, and the compiler finds every place a new
  variant needs handling.
- **Decision points only.** The game runs on frame timers and `update()` ticks; the engine
  collapses each action to one step and stops only where the player decides (play a card, end
  turn, pick a reward, choose a path...). Those decisions are `PlayerAction` (in combat) and
  `RunAction` (everything else), offered as a list of legal actions: what an RL agent needs.
- **Bit-exact randomness.** `crates/sts-rng` ports libGDX `RandomXS128`, the game's counting
  `Random` wrapper, `java.util.Random` and `Collections.shuffle`, golden-tested against the real
  jar (`tools/golden`). `rngs.rs` keeps the game's 13 named streams with their reseed rules.
  HashMap iteration orders that leak into game behaviour were computed on the game's JRE
  (`tools/data`). The few unseeded calls in the game go through a dedicated `rng.unseeded`.
- **Determinism contract.** Same seed and same decisions give the same state, and the same as
  the real game wherever the game itself is seeded.
- **The run is a state machine.** `Run.screen` is the current `Screen` (map, combat, rewards,
  shop, event, boss relic...); `step(RunAction)` moves between them. Sub-screens a relic opens
  (Cauldron's rewards, Dolly's Mirror's grid) are pushed on `screen_stack` and return where they
  came from. Screens the game shows but that hold no decision (Neow's "Talk", the merchant room,
  an unopened chest, act transitions) don't exist in the engine.
- **No meta-progression.** Everything unlocked, ascensions 0-20, four Neow blessings, final act
  on. Don't add unlock logic.

## Interfaces

- **`sts-cli`** (`crates/sts-cli`): `serve` (JSON lines on stdin/stdout, see README.md), `play`
  and `run` (by hand; `run --save` records a run to watch in the game), benchmarks,
  `replay`/`replay-run` (check recordings, below) and `commands` (below).
- **`sts-py`** (`crates/sts-py`, built by `python/build.sh` into `python/sts.abi3.so`): batched
  environments (`sts.Envs`) stepped in parallel without the GIL, plus fixed-size array encodings
  of fights and runs (`encode*.rs`, `python/core/sts_env.py`) for the network. The encoding names
  every field (`sts.layout()`) and must not assume Ironclad: other characters come later.
- **Training** (`python/`): PPO for fights and whole runs, on a CUDA GPU (a 16 GB RTX 5070 Ti
  under WSL for the released network). With `--overlap`, `train_runs.py` plays the next rollout in
  a worker process (its own `sts.Envs` and a copy of the network) while the trainer trains on the
  last; the rollout reaches the trainer through shared memory, since CUDA memory can't be shared
  between processes under WSL. `build.sh` renames the new module into place, so rebuilding doesn't disturb a running
  training process.
- **`sts-encode` and `sts-net`**: the network's encoder and action slots live in their own crate,
  so training (`sts-py`) and the browser (`sts-wasm`) encode a position identically; `sts-net`
  is the run network's forward pass in plain Rust (no ML runtime: 1M parameters, a few
  milliseconds a decision in WebAssembly), loaded from the release's safetensors file, whose
  metadata carries the encoder's signature so a network never reads another encoding's inputs.
  Tokens PyTorch pads out of attention are left out instead, which gives the same outputs; ties
  between equal logits go to the first slot, as `argmax` does, or the browser's runs drift from
  PyTorch's (which copy of a card is played changes later shuffles).
- **`sts-wasm` and `web/`** (milestone 6): the same engine compiled to WebAssembly behind a JSON
  command interface (plain exports, no wasm-bindgen), and a static page that draws its
  observations. The engine keeps a run as its config and decisions and reaches any position by
  replaying them (snapshots every 64 decisions), the same way saved runs are played back in the
  game, so the viewer, play mode and `watch.sh` share one file format.
- **The fight look-ahead** (`crates/sts-py/src/search.rs`, `python/search/fight_search.py`): Rust holds
  the copies (`Envs.playout_start` clones and redraws each game once per candidate move and
  copy, `playout_observe`/`playout_step` step them, `playout_results` reports how each ended)
  and Python runs the network on all of them at once, a pass per move. The network passes are
  the cost (one process keeps the GPU about a quarter busy), so `search_eval.py --procs` splits
  the runs between processes; there WSL's RAM, not the GPU, limits how many (each holds about
  40 copies of every run it searches at once). `train_runs.py --search-*` uses it in rollouts.

## The real game as the reference (`harness/`, `tools/trace/`)

`tools/trace/setup.sh` builds ModTheSpire, BaseMod and CommunicationMod from source into
`harness/` and applies `tools/trace/*.patch`. CommunicationMod runs a script and talks to it over
stdin/stdout: it sends the game state as JSON whenever the game is ready for a command, and
reads commands (`play 2 0`, `choose 1`, `proceed`, `skip`...). The patch adds:

- a `harness` command: set deck, HP, relics, potions; start an encounter; abandon the run;
- extra state: RNG counters, monster internals, pools, the master deck;
- a frame-time speed-up (record at speed 6: at 10 the game drops animation-timed effects);
- no music (fast fights crash OpenAL), and discovery choices that roll once
  (`HarnessDiscoveryPatch`; the unmodded game rerolls them every frame);
- Note For Yourself always offering an unupgraded Iron Wave (`HarnessNotePatch`), like the engine:
  the unmodded game offers the card the profile saved at the end of an earlier run, so in a soak
  each run would get the card the run before it left;
- The Courier's restock picking its card from the engine's `rng.courier` stream
  (`HarnessCourierPatch`; the unmodded game picks it unseeded, so it differs on every play);
- a curse waiting on a transform grid previewed without cardRng (`HarnessTransformPreviewPatch`;
  the game re-rolls the preview every 0.1 s from cardRng, so its cardRng depended on timing);
- Secret Portal eligible whatever the playtime (`HarnessSecretPortalPatch`), like the engine;
- above speed 1, a heal still queued when the battle ends applied first
  (`HarnessEndBattleHealPatch`): at harness speed the last death animation could end the battle
  before Reaper's heal ran;
- fixes to CommunicationMod's "is the game ready" check, each of which left the harness waiting
  for ever: a reward taken after a fight won on the monsters' turn (gold and claimed rewards count
  out of combat whatever the action manager says), a `wait` during a choice at the end of the
  turn (Nilry's Codex), and a shop removal that the game finishes only back in the shop (blocked
  only for the shop's own removal grid, and only when no rewards are left to reopen).

The harness profile is fully unlocked, final act included. Nothing in `harness/` or
`reference/` is ever committed.

## Two directions of checking

The engine is checked against the game in both directions, because each catches what the other
can't.

### Game → engine: recordings and replays

`tools/trace/record.py` (fights) and `record_run.py` (whole seeded runs) play the real game with
a random policy and log every state and command (`harness/traces/`). `sts-cli replay` and
`replay-run` rebuild each one in the engine from its seed, map every game command to an engine
decision (`run_replay.rs`, `replay.rs`) and compare the full state at every decision point:
piles in order, HP, block, energy, powers, monster moves, gold, deck, relics, pools and RNG
counters. A difference points at the exact decision where the engine's logic is wrong.

This is the precise check, but it only covers what a random recorder does. It plays by its own
rules (avoids some potions, abandons Match and Keep), rarely skips a boss relic, and never gets
far enough to see much of Acts 3 and 4.

`sts-cli coverage <recordings and soak folders>` says which items the recordings and soak runs
that match the game exercised (cards played, potions drunk, powers present, monster moves,
event options; a relic when replaying without it changes the outcome), and
`tools/trace/tick_coverage.py` ticks docs/checklist's real-game boxes from it. `record.py
--cards/--relics/--potions` and `record_run.py --relics` put chosen items in every fight or run,
for what is left.

### Engine → game: playback and the soak

The other way round: the engine (or the network playing in it) decides, and the game is told to
do the same. Each decision becomes CommunicationMod commands in **`sts_core::commands`**
(`commands.rs`); `tools/trace/playback.py` sends them and checks that the game's floor and HP
match the engine's before every decision.

- `sts-cli run --save` and `python/watch/watch_runs.py` save a run; `tools/trace/watch.sh` plays it
  in the game to watch (fights: `watch_fights.py` and `playback_fights.py`).
- **The soak** (`tools/trace/soak.sh`, `soak.py`) plays a batch of network runs
  (`python/watch/soak_runs.py`, made on the training machine) unattended overnight. It stops each run at the first
  difference and logs it: `differs` (floor or HP), `refused` (the game keeps rejecting a
  command), `stuck` (no progress for 3 minutes), `ended_early`, `result_differs`, `crashed`,
  `engine_changed`. `soak.sh` keeps the computer awake and restarts the game if it freezes.
  Every difference a soak found was explained and fixed, or recorded as a known deviation.

Playback checks less per step (floor and HP only), but it checks what the network actually does,
and the network finds corners random play doesn't: it buys Cauldron, skips boss relics, spins
the Wheel of Change. It also tests the command translation itself, which recordings can't.

### Who owns what in playback

**The engine owns the translation from a decision to game commands.** Everything about "which
clicks carry out this decision in the game" lives in `commands.rs`, in Rust, with unit tests,
written from the game's source: the wheel's spin is two clicks, skipping the boss relic closes
the chest so the room's proceed must follow, a labelled reward screen (Cauldron, Orrery, Tiny
House) leaves by the cancel button. One translation serves the soak, `watch.sh` and
`sts-cli run --save`, so they can't drift apart.

**Playback stays simple.** `playback.py` only does what the engine can't know:

- click through screens the engine doesn't have (`click_through`: the merchant room, an unopened
  chest, Neow's talk, leaving a rest site);
- wait for the game: CommunicationMod reports "ready" before some effects finish (an obtained
  card is added, and Darkstone Periapt fires, only when `ShowCardAndObtainEffect` ends), so on a
  floor or HP difference the soak lets the game run up to 5 × 30 frames before calling it real.

When a playback problem turns up, first ask whether the engine's commands are wrong (fix
`commands.rs`) or whether it's a screen or timing the engine rightly doesn't model (fix
`playback.py`). Engine game logic is fixed in the engine, never worked around in playback.

**Commands are worked out at playback time, not trusted from files.** A saved run or batch keeps
the seed and every decision, with the engine's floor and HP and the commands as they were then.
`sts-cli commands` replays the decisions in the current engine and writes the steps back with
fresh commands and the floor and HP the engine has now (which is what the game is compared
with), and reports runs whose floor or HP changed since they were saved. `soak.sh` (into
`<out dir>/batch.json`) and `watch.sh` (into `harness/watch/`) run it first, so a fix to the
engine's commands or game logic applies to every old batch at once. A run whose decisions the
engine can't take any more (it reaches a different screen, or refuses a move) is caught and
marked `engine_changed` instead of producing confusing failures in the game.

## What can't be matched exactly

- **Timing in the game.** Some game behaviour depends on frames, not seeds: card-discovery
  rerolls (patched to roll once), effects that land after CommunicationMod says the game is
  ready (handled by the settle wait, and for Neow's curse and Cursed Key's chest curse by waits
  in the engine's commands), and a heal queued as the battle ends (patched, above).
- **Unseeded randomness** in the game (`MathUtils.random`) goes through `rng.unseeded` in the
  engine; the game isn't deterministic there.
- **Choices the engine merges.** After skipping a boss relic the game lets you reopen the chest;
  the engine doesn't, since that only shows the same relics again. Leaving out decisions that
  reach no new state keeps the network's choices meaningful.
- Known gap: Secret Portal skips its playtime check (the harness patch `HarnessSecretPortalPatch`
  makes the modded game skip it too).

## Adding to the engine

1. Port the rule from `reference/` (keep game order for hooks and iteration; `f32` damage math).
2. Unit-test it in `crates/sts-core/tests` against what the game code does.
3. If the game needs more than one click for the decision, or a different command than its
   default, add it to `commands.rs` with a test.
4. Check against the game: a short recording (`record.py`/`record_run.py`, then `replay`/
   `replay-run`) for logic; a short playback or soak of affected runs for commands. Keep
   real-game sessions short unless a long one is wanted.
