# Slay the Spire headless engine (Rust)

A headless port of Slay the Spire's game logic, written in Rust so a neural network can play
millions of games and learn the game. Its randomness matches the real game step for step: the
same seed and the same choices give the same run, which recordings of the real game check.

- **The engine** plays whole runs for all four characters, from Neow to the Corrupt Heart:
  every card, relic, potion, power, monster, event, orb and stance, each with a unit test and
  checked against recordings of the real game ([docs/checklist/](docs/checklist/README.md)).
- **A network** trained on it plays the Ironclad: it wins 93% of runs at ascension 0, 75% at
  A10 and 22% at A20 ([docs/models/ironclad-v1.md](docs/models/ironclad-v1.md)).
- **A browser viewer** steps through saved runs, lets you play any run, and shows the network's
  advice at every decision; the engine and the network both run in the page (WebAssembly).

[docs/architecture.md](docs/architecture.md) explains how the pieces fit.

This is an unofficial fan project, not affiliated with Mega Crit. Slay the Spire and its names
belong to them; you need the game to play it. The engine was written by reading the game's
code, which is not part of this repository.

## Build

```bash
cargo build --release
cargo test
```

## CLI

```bash
# Play a combat by hand (type the number of an action)
./target/release/sts-cli play --seed 1 --encounter "Gremlin Nob"

# Play a whole run by hand (--character silent or defect; the Ironclad by default)
./target/release/sts-cli run --seed 1
./target/release/sts-cli run --seed 1 --character silent

# Random-agent benchmark over whole runs
./target/release/sts-cli bench-runs --games 20000

# Random-agent benchmark over every encounter (add --random-decks to fuzz every card)
./target/release/sts-cli bench --games 20000

# List encounter names
./target/release/sts-cli encounters
```

### JSON-lines protocol (`sts-cli serve`)

One JSON command per input line, one JSON reply per output line:

```json
{"cmd":"reset","seed":1,"encounter":"Jaw Worm","ascension":0,"deck":["Strike","Bash+","Whirlwind"],"hp":80}
{"cmd":"step","index":0}
{"cmd":"step","action":{"type":"play_card","hand_index":0,"target":0}}
{"cmd":"step","action":{"type":"end_turn"}}
{"cmd":"step","action":{"type":"choose","indices":[2]}}
{"cmd":"observe"}
{"cmd":"new_run","seed":1,"ascension":0,"neow_full":true,"character":"silent"}
{"cmd":"step","action":{"type":"path","x":3}}
{"cmd":"step","action":{"type":"combat","action":{"type":"end_turn"}}}
```

`new_run` switches the session to a whole run; its steps are `RunAction`s (`neow`, `path`,
`combat`, `take_reward`, `pick_card`, `skip_cards`, `select`, `rest`, `smith`, `recall`, `boss_relic`,
`proceed`) and its observation is the run (act, floor, HP, gold, deck, relics, keys, map, the
current screen) with `legal_actions`. `new_run` takes `"final_act": false` to play without the
keys and Act 4 (the default is on).

Replies are `{"ok":true,"done":false,"obs":{...}}`. The observation holds the player, hand
(with current damage/block), pile contents, monsters (stable `id`, HP, block, powers,
intent and intent damage), any pending card selection, and `legal_actions`. `index` picks
from that list. Omitted reset fields default to the character's starting deck, HP and relic
(`"character"`, the Ironclad by default: 80 HP and Burning Blood).

## Watching a run in the real game

```bash
./target/release/sts-cli run --seed 5 --save harness/playback/run.json   # play by hand; saved after every step
tools/trace/watch.sh harness/playback/run.json 0.4                         # watch it in the game
```

`watch.sh` (on macOS, with the modded game that `tools/trace/setup.sh` builds) points
CommunicationMod at `tools/trace/playback.py`, launches the game, and restores CommunicationMod's
previous command when the game closes. The playback starts the same seed and ascension and makes
the same choices, pausing the given seconds between commands (default 0.8). It stops where the
saved run stopped, even part way through, logs to `<file>.playback.log`, and warns if the game's
floor or HP ever differs from the engine's. A network's runs: `python/watch/watch_runs.py <checkpoint>
out.json` saves the best of 64 fixed runs (`python/results/watch/` has examples, including an A20
win, `a20_win_L7WC.json`).

## Watching and playing in the browser

```bash
web/serve.sh            # builds web/sts.wasm if needed; http://localhost:8765/
web/serve.sh --lan      # the same, reachable from the local network (http://<this machine>:8765/)
```

The engine compiled to WebAssembly (`crates/sts-wasm`) runs in the page (`web/`, plain HTML and
JavaScript). `web/serve.py` serves only `web/` and the saved runs in `python/results/watch/` (at
`/runs/`), never the rest of the repo (`reference/` and `harness/` hold the decompiled game). The
start screen opens the featured A20 win, a run of the gallery (`web/gallery.json` describes the
saved runs), a file (or drop one on the page), or a new run (character, ascension, seed). A link
such as `http://localhost:8765/?run=runs/a20_win_L7WC.json&at=120` opens a run at a decision.
Step through it: back and forward (arrow keys), autoplay (space), jump to a floor or click any
decision in the log (grouped by floor); `?` lists the keys. Hover over or tap a card, relic,
potion, power or intent to see what it does (hand-written text in `web/text.js` and
`web/text-cards.js` with the engine's numbers, no game text). Play by clicking the moves or the
next room on the map; choosing differently from a run that goes on starts a variation there, as
on a chess site: the run keeps its later decisions, the log shows the variation under the
decision it replaces (variations can branch again), and you play on in it, or let the AI. A run
you play, and the variations played on a gallery run, are kept in the browser. **Save run**
downloads the run in the format `watch.sh` plays in the real game, its variations in a
`variations` list beside its `steps`. `web/build.sh` rebuilds the engine after changes to the Rust.

The AI's advice needs the network's weights, which aren't in git; they come with this
repository's release:

```bash
gh release download ironclad-v1 -p ironclad-v1.safetensors -D web/models
```

In an Ironclad run (watched or played) the panel above the decisions shows the network's three
likeliest decisions with their probabilities and its chance to win, and outlines its move on the
screen; click a row to take it, or let the AI take this move (A), play to the next floor or to
the end. **Hide until I choose** keeps the advice back until you have chosen, then says what the
AI would have done. The network runs in the page (`crates/sts-net` compiled into `sts.wasm`), the
same as in training: in the browser it plays fixed runs to the same ends as PyTorch.

## The network's advice in the real game

```bash
python/build.sh                                  # the sts module, with sts.advise
tools/advisor/play.sh python/runs/dh12/update_200.pt   # a run checkpoint; later runs reuse it
```

`play.sh` starts the harness game with CommunicationMod running `tools/advisor/advisor.py`. Play an
Ironclad run as usual: after every change in the game a box in the top-left corner shows the
network's three most likely choices with their probabilities (cards to play and their targets,
potions, end turn; card rewards, the map, rest sites, upgrade and removal grids, boss relics,
shops, events and Neow). F9 hides it. The advisor never sends an action. While it is connected,
the harness's changes to the game (Note For Yourself, the Courier's restock, discovery rolling
once) are off. `sts.advise` (crates/sts-py/src/advise.rs) turns the game's JSON into the network's
inputs; fights match the engine's own encoding except that cards show their base damage, block and
magic. Log: `harness/advisor/advisor.log`.

## Python (for training)

`python/build.sh` builds the `sts` module into `python/` (no maturin or pip installs needed):

```python
import sts
envs = sts.Envs(256, mode="fight", seed=0)   # or mode="run" for whole runs
counts = envs.num_legal()                     # legal actions per game
status = envs.step([0] * len(envs))           # 0 running, 1 won, 2 lost
envs.legal_actions(0)                         # game 0's actions, as JSON
envs.observe(0)                               # game 0's full view, as JSON
envs.last_episode(0)                          # summary of the last finished episode
```

`step` takes one index per game into its legal actions and runs every game in parallel without
the GIL; a finished game starts its next episode at once. Fights pick an encounter from
`encounters=[...]` (default: all) with a random Ironclad deck (`random_decks=False` for the
starter deck); runs take `ascension` and `final_act`. Random play runs at about a million steps
per second from Python (`python3 python/tests/bench.py fight|run`); `python3 python/tests/test_sts.py`
checks it.

For a network, fights also come as arrays (numpy, in the project venv: `python3 -m venv .venv
&& .venv/bin/pip install numpy`):

```python
import sys; sys.path.insert(0, "python")   # from the repository root
from core.sts_env import FightEnv
env = FightEnv(256, seed=0)
obs = env.observe()        # "i_*" ID fields, "f_*" number fields, "mask" of legal slots
status = env.step(slots)   # one action slot per game (171 slots: play card x target,
                           # potion x target, end turn, pick option k, confirm)
```

`sts.layout()` names every field and gives the embedding vocabulary sizes; card selections
(Exhaust, Armaments, Headbutt...) are made one pick per step. `python/tests/test_encode.py` checks it.
Whole runs come as arrays too (`RunEnv`, same action slots; `python/tests/test_run_encode.py`).

Training wants a CUDA GPU: `python/training/train_runs.py` trains the run network (PPO on whole
runs; `--help` lists the reward, practice-fight and speed options), `python/training/train_fights.py`
the earlier fight-only network. [python/README.md](python/README.md) says what each script is for;
the model card has the recipe of the released network.

## No progression

The engine has no profile: every card and relic is unlocked, all ascensions 0-20 can be
played (`--ascension N`, or `"ascension": N` in `new_run`), Neow always offers four blessings,
every boss counts as seen, and the final act is available (keys, Act 4 and the Heart; turn
it off with `"final_act": false`). Ascension effects across acts are in: the A5 heal between
acts, A12's lower card upgrade chance, the A20 double boss.

## Layout

- `crates/sts-rng` — bit-exact ports of libGDX `RandomXS128`, the game's `Random` wrapper,
  and `java.util.Random` + `Collections.shuffle` (golden-tested against the real jar).
- `crates/sts-core` — the engine: action queue, cards, powers, monsters and encounters for
  all four acts, whole runs (map, Neow, rewards, shop, events, rest sites, keys).
- `crates/sts-cli` — the driver above.
- `crates/sts-encode` — the network's array encodings and action slots, shared by training and
  the browser.
- `crates/sts-net` — the run network's forward pass in plain Rust (from a release weights file).
- `crates/sts-py` — the `sts` Python module: batched environments, practice fights and boss
  scenarios, the advisor's encoder.
- `crates/sts-wasm` + `web/` — the engine and the network in WebAssembly, and the browser viewer
  and play mode with the AI's advice.
- `python/` — training (fight and run networks, PPO), evals, human-run data, analysis tools,
  in folders by purpose ([python/README.md](python/README.md)).
- `tools/` — golden-value harnesses, extraction scripts that read the decompiled code,
  `tools/trace/` for recording real-game fights and runs and diffing them (`sts-cli replay
  <traces>`, `sts-cli replay-run <runs>`), playing runs back (`watch.sh`, the overnight
  `soak.sh`), and `tools/advisor/`.
- `docs/` — architecture, what the network sees (`network-inputs.md`), the model card
  (`models/`) and the port checklist (`checklist/`).
