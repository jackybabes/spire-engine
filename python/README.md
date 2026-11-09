# python/

Training and evaluating the networks that play the Rust engine. The engine reaches Python as the
`sts` module (`crates/sts-py`), which `build.sh` builds into `python/sts.abi3.so`:

```bash
python/build.sh                                  # the sts module (needs the Rust toolchain)
.venv/bin/python python/tests/test_sts.py         # checks it
```

Run any script by its path from the repository root, e.g.
`.venv/bin/python python/training/train_runs.py --help`. Each script puts `python/` on the import
path, so the folders import each other by name (`from core.run_model import RunNet`).

## The folders

| Folder | What it holds |
|---|---|
| `core/` | What everything else builds on: `sts_env.py` (numpy views of the engine's arrays: `RunEnv`, `FightEnv`), `fight_model.py` and `run_model.py` (the networks), `ppo.py`. |
| `training/` | `train_runs.py` trains the run network (PPO on whole runs; the main script), `train_fights.py` the earlier fight-only network, `pretrain_human.py` imitation of human choices. `live_chart.py` draws a run's chart, `choice_stats.py` counts its choices between fights. |
| `human/` | Mega Crit's 2020 metrics dump of human runs turned into training data: `human_extract.py` (one character's runs and decisions), `human_dataset.py` (the network's inputs for imitation), `human_bosses.py` (boss and elite fights as practice scenarios), `human_runs.py` and `human_data.py` (shared). |
| `search/` | The fight look-ahead: `fight_search.py` plays each candidate move out on redrawn copies of the fight; `search_eval.py` compares play with and without it. |
| `analysis/` | Looking at a checkpoint: `run_choices.py` (what it picks between fights), `run_probe.py` (what it would win by choosing otherwise), `synergy.py` (how one card or relic changes its card picks: its synergies), `loss_report.py` (which fights it loses), `seed_rates.py` (what each seed is worth in whole runs, against a control seed), `fights_by_card.py` (whether a fight network uses a card: wins, play rates, removal, on human fights), `play_on.py` (whether it wins with human players' decks, from their boss states on), `block_eval.py`, and the scripted baselines (`baselines.py`, `run_baselines.py`). |
| `release/` | A network as a release: `export_release.py` writes and reads the weights file (safetensors) the browser loads; `net_fixture.py` and `net_play_check.py` check the Rust network against PyTorch; `win_calibration.py` turns its run value into a chance to win. |
| `watch/` | Saving a network's runs to watch: `watch_runs.py` (in the browser viewer or the real game), `watch_fights.py`, `soak_runs.py` (a batch for the overnight real-game check). |
| `tests/` | Plain-assert checks (`test_*.py`, no pytest needed) and `bench.py`, a speed check. |

## Data and results

- `runs/` (not in git): one folder per training run: checkpoints (`update_<n>.pt`, `latest.pt`),
  `args.json`, `log.csv`, `eval.csv`, `choices*.csv` and `live.html`.
- `results/watch/`: saved runs to watch; the browser viewer serves them at `/runs/`.
- `results/soak/`: batches of network runs played back in the real game.

## A network as a release

```bash
.venv/bin/python python/release/export_release.py python/runs/ironclad-runs-habits-12/update_200.pt ironclad-v1.safetensors
.venv/bin/python python/tests/test_export.py      # the exported network equals the checkpoint
```

The weights file stores the encoder's signature (`ENCODING_VERSION` and the row lengths), and the
loaders in Python and Rust refuse a file made for another encoding.
