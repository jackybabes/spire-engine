# spire-engine

Slay the Spire's rules rewritten in Rust, AI players trained on it, and a website where you can
watch them play, play runs yourself and ask the AI what it would do.

**Try it: https://spire-viewer.pages.dev**. Everything runs in your browser, with nothing to install.

An unofficial fan project, not affiliated with Mega Crit. Slay the Spire belongs to them, and if you
like it, buy the game. No art, music or text from the game is included: the card and relic
descriptions on the site were written for this project.

## On the site

- **Watch** the AI's best runs from the gallery, among them an A20 Ironclad win and the Watcher
  beating the Heart at A20.
- **Play** a new run with any character at ascension 0-20 on any seed. Click a card, then its
  target, or the next room on the map.
- **Ask the AI.** Open the AI panel to see the three moves it likes best, with how sure it is of
  each, and its estimate of your chance to win the run. It can play the next move for you, the
  rest of the floor or the rest of the run. Tick **Hide until I choose** to choose first and then
  see what it would have done.
- **Try something else.** Go back to any decision and choose differently, and the run goes on from
  there as a variation, like analysing a game on a chess site. The original line is kept.
- **Look things up.** Hover over or tap any card, relic, potion, power or monster intent.
- **Save** a run with its variations to open again later, or copy a link to one decision.

Keys: arrow keys step through a run, space plays or pauses, A lets the AI take the move, and `?`
lists the rest.

## The AI players

There's one network per character. Each one plays whole runs on its own: Neow, the map, card
rewards, shops, events, rest sites and every turn of every fight. These are their win rates on a
fixed set of seeds:

| AI | Ascension 0 | A10 | A20 | Counted as a win |
|---|---|---|---|---|
| [Ironclad v1](docs/models/ironclad-v1.md) | 93% | 75% | 22% | beating the Act 3 boss |
| [Silent v1](docs/models/silent-v1.md) | 76% | 48% | 3% | beating the Act 3 boss |
| [Defect v1](docs/models/defect-v1.md) | 28% | 10% | 0.9% | beating the Heart |
| [Watcher v1](docs/models/watcher-v1.md) | 44% | 26% | 8% | beating the Heart |

The Ironclad and Silent networks were trained to beat the Act 3 boss and don't collect the keys, so
their runs end there. The Defect and Watcher networks go for the keys and the Heart.

The AI knows only what you would know: your hand, the draw pile's contents but not its order, the
monsters' intents, your relics, potions and deck. It doesn't search ahead or peek at the random
numbers.

**How the networks learned.** Each one played hundreds of thousands of runs against the engine,
rewarded for getting further and winning (reinforcement learning, PPO), on a single consumer GPU.
Some learned from human runs as well: they imitated choices made by human winners in Mega Crit's
2020 data, and practised the boss fights those winners faced, with the winners' decks. Each model
card has the details.

## How faithful is the engine?

It plays like the real game, down to the random numbers. The game's random number generators are
ported bit for bit, and the engine draws from them the way the game does, so the same seed and the
same choices give the same run: the same map, card rewards, shuffles and monster moves. Every
card, relic, potion, power, monster, event, orb and stance for all four characters is in, through
Act 4 and the Heart.

To check it, a modded copy of the game recorded hundreds of fights and whole runs, which were then
replayed in the engine to find the first point where the two differed. Runs played by the networks
were also played back in the real game. Every item has a unit test and has been seen in a
recording that matches the game ([docs/checklist/](docs/checklist/README.md)). A few known
differences remain, mostly where the real game depends on frame timing or on your saved profile.

There is no unlock progression. Every card and relic is available, every ascension can be played,
Neow always offers all four blessings, and the final act is on.

## Run it yourself

You need [Rust](https://rustup.rs). Then:

```bash
cargo build --release
```

### Play in the terminal

```bash
./target/release/sts-cli run --seed 1 --character silent --ascension 10
./target/release/sts-cli play --encounter "Gremlin Nob"          # a single fight
./target/release/sts-cli encounters                              # the fight names
```

Characters: `ironclad` (the default), `silent`, `defect`, `watcher`. Type the number of a move to
make it.

### Run the website locally

```bash
web/serve.sh        # builds the engine for the browser if needed, then http://localhost:8765/
```

The AI files aren't in this repository, so locally the site runs without the AI's advice.

### Write your own bot

`sts-cli serve` reads one JSON command a line and answers with one JSON line. Every answer holds
the game state and a `legal_actions` list:

```json
{"cmd":"new_run","seed":1,"ascension":20,"character":"watcher"}
{"cmd":"step","index":0}
{"cmd":"observe"}
{"cmd":"reset","encounter":"Jaw Worm","deck":["Strike","Bash+"],"hp":80}
```

- `new_run` starts a whole run.
- `step` makes a move, either by its `index` in `legal_actions` or as an explicit `action`.
- `reset` starts a single fight with the deck and HP you give it.

Add `"final_act": false` to `new_run` to play without the keys and Act 4.

From Python, `python/build.sh` builds the `sts` module, which runs many games in parallel. Random
play runs at about a million moves a second:

```python
import sys; sys.path.insert(0, "python")
import sts
envs = sts.Envs(256, mode="run", seed=0, character="defect")
status = envs.step([0] * len(envs))   # a move for every game; 0 still going, 1 won, 2 lost
envs.legal_actions(0)                 # game 0's moves, as JSON
envs.observe(0)                       # game 0's state, as JSON
```

`python/training/train_runs.py` is the training script (PPO, and it needs a CUDA GPU).
[python/README.md](python/README.md) describes the other scripts.

## What's in the repository

- `crates/sts-rng`: the game's random number generators.
- `crates/sts-core`: the engine.
- `crates/sts-cli`: the command-line player and the JSON interface.
- `crates/sts-encode`: how the game is turned into the networks' inputs and moves.
- `crates/sts-net`: the networks in plain Rust, for the browser.
- `crates/sts-py`: the Python module.
- `crates/sts-wasm` and `web/`: the engine compiled to WebAssembly, and the website.
- `python/`: training, evaluation and analysis.
- `tools/`: tests against the real game's code, data extraction, and the harness that records the
  real game, plays runs back in it and shows the AI's advice over it. The harness needs your own
  copy of the game, and the setup assumes a Mac. The extraction scripts read a decompiled copy of
  the game, which isn't included.
- `docs/`: [how it fits together](docs/architecture.md), [what the networks
  see](docs/network-inputs.md), the model cards and the checklist.

## Licence

The code is under the MIT licence ([LICENSE](LICENSE)). Slay the Spire and everything in it belong
to Mega Crit.
