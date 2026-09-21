# Survival simulator — baseline model

Nordic AI Cup 2026, Challenge 1. You control every member of a herbivore species at once: eat fruit, conserve energy, avoid predators, and keep the species alive for up to 3000 simulated seconds (30000 ticks).

This README explains our **survival baseline**: a rule-based policy with no learned weights, the harness that scores it, and how to run it. For the full game rules (observation/action tables, energy costs, scoring) see the [organizers' README](https://github.com/amboltio/Nordic-AI-Cup-2026/tree/main/survival-simulator) and the engine notes in [MECHANICS_QA.md](MECHANICS_QA.md).

## The model in one paragraph

Each tick, every agent looks at its own observations and picks the first rule that applies: **flee a predator → eat the nearest fruit → walk to the nearest tree and wait there → wander**. Fruit spawns around trees, so "stand by a tree and eat what drops" is the core survival loop. `heuristic_v0` is exactly that. `heuristic_v1` adds three rules on top that stop the species from piling onto one tree: a per-tree headcount cap, a "leave in a straight line" travel mode, and births that only happen when there is room.

All numbers live in a JSON config in [training/configs/](training/configs/) — nothing is hardcoded in the agent classes.

| Agent | Code | Config | What it is |
|---|---|---|---|
| `dummy_v0` | [agents/dummy.py](agents/dummy.py) | `dummy_v0.json` | Organizers' default policy, as a floor |
| `heuristic_v0` | [agents/heuristic.py](agents/heuristic.py) | `heuristic_v0.json` | The baseline priority rules |
| `heuristic_v1` | [agents/heuristic_v1.py](agents/heuristic_v1.py) | `heuristic_v1.json` | v0 + tree cap, LEAVING mode, birth rules |

Every agent exposes the same interface as the submission endpoint:

```python
agent.act_batch(agent_states: list[dict]) -> list[dict]   # ObservationResponse dicts in, ActionRequest dicts out
```

`agents.make_agent(agents.load_config(path))` builds one from a config.

## heuristic_v0 — the priority rules

Stateless; each agent is decided independently from its own observations.

1. **Predator observed** → move directly away from the nearest one while *turning to face it*. Predators only chase agents that look away or are very close, so facing it matters as much as the distance. Sprint if it is closer than `flee_sprint_dist`, otherwise walk.
2. **Fruit observed** → turn to and walk onto the nearest fruit (never overshooting it).
3. **Tree observed** → walk toward the nearest tree at reduced speed until within `tree_wait_dist`, then stand still and slowly rotate (`scan_turn` per tick) to keep looking for fruit and predators.
4. **Nothing observed** → wander forward at `wander_speed_frac × speed` while scanning.

**Spawning:** request a new agent whenever energy exceeds `spawn_energy_threshold` and no predator is in view.

Two engine facts the rules rely on: observation `angle` and action `move_direction` are both relative to the agent's current heading, and within a tick movement is applied before the turn.

| Parameter | Value | Meaning |
|---|---|---|
| `spawn_energy_threshold` | 175 | Energy above which the agent spawns (spawning costs 100) |
| `wander_speed_frac` | 0.5 | Fraction of walking speed used when wandering / approaching a tree |
| `scan_turn` | π/16 | Rotation per tick while waiting or wandering |
| `tree_wait_dist` | 40 | Distance from the tree at which the agent stops and waits |
| `flee_sprint_dist` | 90 | Predator distance below which the agent sprints |
| `face_predator` | true | Turn to face the predator while backing away |

## heuristic_v1 — anti-overcrowding rules

v0's weakness is that the whole species converges on the same tree, competes for the same fruit, and offers predators one big target. v1 keeps v0's flee / fruit / tree behaviour unchanged and adds a small per-agent memory on the server (`mode` = `WAITING` or `LEAVING`, plus a step counter), dropped when an agent dies.

**Counting agents at a tree.** An agent counts itself plus every other *observed* agent within `tree_radius` of that tree. The other agent's distance to the tree is computed from the observer's own relative observations with the law of cosines, so no absolute positions are needed. Agents currently in `LEAVING` mode are not counted. Modes are snapshotted at the start of each tick so the result doesn't depend on the order agents are processed in.

- **Rule 1 — tree cap.** While waiting at a tree, if the headcount exceeds `tree_cap`, the agent with the *highest energy* (ties broken by lowest id) switches to `LEAVING`. It can best afford the trip; the others keep scanning.
- **Rule 2 — LEAVING travel.** The agent turns its back on the tree and walks in a straight line at full walking speed. For the first `leaving_ignore_steps` ticks it ignores trees and fruit entirely, so it actually gets away instead of turning straight back. After that it joins the least crowded visible tree with fewer than `tree_join_threshold` agents (ties → nearest) and returns to `WAITING`. If a wall or the arena boundary crosses its path within the next step, it turns by `leaving_turn_angle` and carries on. Fleeing a predator still overrides everything, in either mode.
- **Rule 3 — births.** An agent spawns only when energy exceeds `birth_energy_threshold` **and** there is room: at most `birth_max_others_at_tree` other agents at its nearest tree, or, with no tree in sight, at most `birth_max_others_no_tree` agents within `birth_no_tree_radius`. After spawning, the *parent* enters `LEAVING` — the child inherits the tree and the parent goes to find a new one.

If anything in v1's logic raises for an agent, that agent gets the v0 action for that tick instead.

| Parameter | Value | Meaning |
|---|---|---|
| `tree_radius` | 60 | Distance from a tree within which an agent counts as "at" it |
| `tree_cap` | 2 | Max agents at a tree before one must leave |
| `tree_join_threshold` | 2 | A leaving agent only joins trees with fewer agents than this |
| `leaving_ignore_steps` | 30 | Ticks of straight travel before trees are considered again |
| `leaving_turn_angle` | 2π/3 | Turn applied when the path ahead is blocked |
| `birth_energy_threshold` | 230 | Energy required to spawn |
| `birth_max_others_at_tree` | 1 | Max other agents at the nearest tree for a birth |
| `birth_no_tree_radius` / `birth_max_others_no_tree` | 80 / 1 | Same check when no tree is visible |

The remaining parameters are shared with v0. Both configs are marked `provisional`: the values were picked for a first look at the game, not tuned.

## Results

Local headless runs, raw game score (higher is better; a full 3000 s game is 30000 steps).

| Agent | Seeds | Mean score | Std | Mean survival steps | Source |
|---|---|---|---|---|---|
| `dummy_v0` | 0–4 | 18.9 | 3.0 | 189 | `results/index.csv`, C1-E00 |
| `heuristic_v0` | 0–4 | 698.3 | 205.2 | 6947 | `results/index.csv`, C1-E01 |
| `heuristic_v1` | 1–3 | 1059.1 | 48.6 | – | `results/EXTERNAL_seeds1-3.csv` |

The v1 row is from a different seed set and only three games, so read it as "clearly better than v0", not as a precise number. On those same seeds 1–3 the external candidates in [external/](external/) scored about 1460–1590, which is why the live submission currently serves one of them rather than this baseline (see below). The simulation is only deterministic per OS: these are Windows numbers, and the evaluation server runs Linux.

## Running it

Python 3.10+. From this folder:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q tests            # run before every scored run
```

**Scored evaluation** — always a fresh process, never inside a long-lived kernel:

```bash
python -m training.evaluate --experiment C1-E01 --config training/configs/heuristic_v0.json \
    --seeds 0 1 2 3 4 --workers 4 --max-wall-sec 1800
```

This writes one row per seed to `results/<run_id>.csv` and appends a summary row to the append-only `results/index.csv`, including the git revision (flagged `-dirty` if the tree has uncommitted changes — commit first). Add `--dry-run` to smoke-test without writing anything.

**Notebook** — [survival_baseline.ipynb](survival_baseline.ipynb) is an execution trigger only; all logic lives in `agents/` and `training/`. It runs the tests, launches the scored evaluation as a subprocess, shows the results index, renders a game to video for inspection (`training.render.render_episode`, never scored), and runs the comparisons against the external candidates.

**Watching a game locally** — [local_playground.py](local_playground.py) runs the organizers' simulator with rendering.

## Submission server

[submission_server.py](submission_server.py) serves `POST /predict` on port 9052 in the format the evaluation service expects. It currently loads an external candidate policy (`AGENT_CANDIDATE`, default `original-eat-rest-preserved`), not the baseline described above. Every step is wrapped in try/except: if the policy raises, returns an invalid action or skips an agent, those agents get a valid no-op action, and the fallback count is visible on `GET /api`. Set `INJECT_FAILURE=1` to test that path, and `NAIC_API_KEY` for the API key (never commit it). Deployment scripts are in [deploy/](deploy/).

The evaluation server waits at most 10 seconds per response and 600 seconds in total, and the final evaluation runs three games back to back on preset seeds — the server has to stay up through all three.

## Layout

```
agents/        policies (dummy, heuristic v0, heuristic v1) + make_agent/load_config
training/      episode loop, batch runner, evaluate (scored runs), render, diagnostics, configs/
external/      third-party candidate and finalist policies with thin adapters for comparison
results/       per-run CSVs and the append-only index.csv
tests/         agent tests
src/           organizers' simulator (unmodified)
```
