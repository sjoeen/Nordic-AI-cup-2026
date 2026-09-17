# Survival Simulator: Handover

**Challenge:** Nordic AI Cup 2026, Challenge 1 (Survival Simulator)
**Date:** 2026-09-17
**Git state at handover:** branch `main`, latest commit `226598a`. Nothing has been pushed from this work.

This document covers three things:

1. What has been built and run so far.
2. How the game actually works, taken from the simulator source code rather than the README.
3. What we have learned, what is still unknown, and which decisions are open.

Everything here is marked by how sure we are:

- **Verified** means read directly in the code or measured in a run.
- **Derived** means calculated from the code by hand but not measured in a game.
- **Hypothesis** means a plausible explanation that nobody has tested yet.

---

## 1. Status in one paragraph

A working evaluation harness exists and runs end to end from a Jupyter notebook. Two policies have been scored on seeds 0 to 4. The starter dummy policy averages **18.9** points and dies after about 20 simulated seconds. A simple rule-based heuristic averages **698** points and survives 485 to 944 seconds. No policy has survived the full 3000 seconds yet. The heuristic's parameters were picked by the coding agent and have **not been approved**. Nothing has been run on the cluster yet; every result so far comes from a Windows laptop.

---

## 2. What has been done

### 2.1 Timeline (git history)

| Commit | What it did |
|---|---|
| `eb7fe58` | Added the starter competition repository. |
| `7ad90fa` | Added the submission endpoint (still serving the dummy policy), an Azure deploy script and a connection test. |
| `684199c` | The submission server now logs the IP address of each new client, to find where the evaluator connects from. |
| `9ff405b` | Built the baseline harness: agent classes, headless episode runner, scored evaluator, video renderer, tests and the notebook. |
| `f3b7944` | Committed the first scored results: experiments C1-E00 (dummy) and C1-E01 (heuristic). |
| `81557dd` | Harness fixes: MP4 video instead of a 64 MB GIF, package versions logged with every run, result files no longer mark the code as modified, and a dry-run mode. |
| `1918e57` | Wrote the first baseline report to `trial_racecar/report.md` at the repository root. |
| `226598a` | Added a dry-run smoke-test cell to the notebook and fixed garbled characters in it. |

### 2.2 Code layout

All paths are relative to `Nordic-AI-Cup-2026-main/survival-simulator/`. The starter layout was kept as is.

| Path | Purpose |
|---|---|
| `src/` | The simulator itself. **Unmodified starter code.** |
| `agents/__init__.py` | Agent registry. `make_agent(config)` builds an agent from a JSON config. |
| `agents/dummy.py` | The starter random policy, reproduced exactly as the starter server runs it. |
| `agents/heuristic.py` | The rule-based policy. Every number comes from its config file. |
| `training/configs/*.json` | One config per policy version: `dummy_v0.json` and `heuristic_v0.json`. |
| `training/episode.py` | Runs one headless game with the same loop as the official evaluator. |
| `training/evaluate.py` | Scored evaluation over many seeds, run in parallel processes. Writes the results files. |
| `training/render.py` | Renders a game to MP4 so it can be watched inside Jupyter. Scores from here are never logged. |
| `tests/test_agents.py` | 5 tests: configs load, actions are valid, flee logic, no fruit overshoot, spawn threshold, same seed gives the same game. |
| `survival_baseline.ipynb` | The notebook. It only triggers the Python files and holds no logic. |
| `results/index.csv` | Append-only index with one row per scored run. |
| `results/<run_id>.csv` | Per-seed details for one scored run. |
| `logs/` | Videos and executed notebooks. Ignored by git. |
| `submission_server.py` | The competition endpoint (`POST /predict`). Still serves the **dummy** policy. |
| `test_connection.py` | Checks health, latency and fallback behaviour of a running endpoint. |
| `deploy/azure_setup.sh` | Sets the endpoint up as a systemd service on an Azure Ubuntu VM. |
| `agent_server.py`, `simulation_server.py`, `local_playground.py` | Unmodified starter files. |

Two files in the repository root, `main.py` and `test.ipynb`, are PyCharm templates and unused.

### 2.3 How to run it

Open the notebook with its working directory set to `survival-simulator/`, then run the cells in order:

1. **Setup cell.** Prints the Python path and git revision, and warns if package versions differ from `requirements.txt`.
2. **Install cell.** Commented out. Uncomment it once on a new machine.
3. **Tests.** Should say `5 passed`.
4. **Smoke test.** Plays one short game per policy in dry-run mode and writes nothing.
5. **Scored runs.** Each one starts a fresh Python process and appends to `results/index.csv`. **Commit your code before running these**, because the run records the git revision.
6. **Results table.** Shows `results/index.csv`.
7. **Video.** Renders the first 300 simulated seconds of a game to `logs/`.

The same evaluator can be run from a terminal:

```bash
python -m training.evaluate --experiment C1-E02 --config training/configs/heuristic_v0.json \
    --seeds 0 1 2 3 4 --workers 4 --max-wall-sec 1800
```

Add `--dry-run`, or set the environment variable `EVAL_DRY_RUN=1`, to run everything without writing any results.

**Python environments.** Use the venv inside `survival-simulator/.venv`, which has Python 3.13.7 and the pinned package versions. On the laptop it is registered as the Jupyter kernel `naic-survival`. The global Python on the laptop has different versions (numpy 2.2.5 instead of 2.3.5), and **that difference changes scores** (see section 4.2). The `.venv` in the repository root only contains ipykernel and is not used.

### 2.4 Results so far

All runs: seeds 0 to 4, Windows 11, 4 parallel workers.

**C1-E00, dummy_v0** (starter random policy)

| Seed | Score | Survived (s) | Peak agents |
|---|---|---|---|
| 0 | 15.70 | 15.7 | 10 |
| 1 | 21.76 | 21.7 | 10 |
| 2 | 15.70 | 15.7 | 10 |
| 3 | 21.55 | 21.4 | 10 |
| 4 | 19.94 | 19.9 | 10 |

Mean **18.93**, standard deviation 3.03.

**C1-E01, heuristic_v0** (rule-based, provisional parameters)

| Seed | Score | Survived (s) | Peak agents | Agents spawned | Predators at end |
|---|---|---|---|---|---|
| 0 | 657.81 | 652.3 | 57 | 201 | 3 |
| 1 | 531.96 | 536.6 | 65 | 204 | 8 |
| 2 | 828.00 | 854.9 | 47 | 251 | 6 |
| 3 | 981.09 | 944.0 | 68 | 307 | 9 |
| 4 | 492.52 | 485.7 | 35 | 105 | 4 |

Mean **698.28**, standard deviation 205.2. These were run with the global Python, not the pinned venv.

**The same heuristic on the pinned venv** (dry run, not in the index): mean **715.83**. Seed 1 scored 619.71 instead of 531.96, and all other seeds were identical.

**Decision time.** The heuristic takes about 0.1 ms per step for the whole population, with a 95th percentile of 0.3 ms. The evaluator allows 10 s per request and 600 s in total, so the policy is nowhere near the limit. Network time is not included.

**Simulation speed.** The whole run of 5 heuristic games took about 4.5 to 5.5 minutes with 4 workers. The slowest game took 335 s of real time for 944 simulated seconds. Building the map takes about 3 s per game. The simulator is pure Python on the CPU, so the policy is not the bottleneck.

### 2.5 The heuristic_v0 policy

Rules checked in order, for each agent, on each step:

1. **Predator seen:** turn to face the nearest predator and move straight away from it. Sprint if it is closer than 90 units, otherwise walk.
2. **Fruit seen:** turn toward the nearest fruit and walk onto it without overshooting.
3. **Tree seen:** walk toward the nearest tree until within 40 units, then stand still and turn slowly to look around.
4. **Nothing seen:** walk forward at half speed while turning slowly.
5. **Spawning:** ask for a new agent when energy is above 175 and no predator is seen.

| Parameter | Value | Source |
|---|---|---|
| `spawn_energy_threshold` | 175 | Agent's choice: 100 spawn cost plus the 75 energy a child starts with |
| `wander_speed_frac` | 0.5 | Agent's choice |
| `scan_turn` | π/16 rad per step | Agent's choice |
| `tree_wait_dist` | 40 | Agent's choice |
| `flee_sprint_dist` | 90 | From the engine: predators always chase within 90 units |
| `face_predator` | true | From the engine: predators further away only chase agents facing away |

### 2.6 Submission endpoint

- `submission_server.py` answers `POST /predict` in the format the evaluator expects. It **still runs the dummy policy**.
- Every agent's action is wrapped in error handling. If the policy crashes, that agent gets a "do nothing" action instead of breaking the request.
- Setting `INJECT_FAILURE=1` forces the policy to crash, to test that the fallback works.
- `GET /api` shows request counts, fallback counts, response times and the client IPs seen.
- The API key is read from `NAIC_API_KEY`. We don't yet know which header the evaluator sends it in, so it is only logged, not enforced.
- `deploy/azure_setup.sh` installs the server as a service on an Azure VM. Whether a VM is currently live was not checked for this handover.

---

## 3. How the game works

Source: `src/elements/environment.py`, `creature.py`, `predator.py`, `biome.py`, `fruit.py` and `tree.py`. Everything in this section is **verified** from the code unless marked otherwise. Where the README disagrees with the code, the code wins, and the differences are listed in 3.10.

### 3.1 The world and one step

- The map is 1600 × 1200 units, with 80 rectangular obstacles of 30 to 100 units per side.
- A game starts with 5 agents, 50 trees and 32 fruits. Starting agents have 150 energy.
- One step is 0.1 simulated seconds. A game lasts at most 3000 s, which is 30,000 steps. It ends early when no agents are left.
- **Order within one step:**
  1. For each agent, in order: move, then turn, then spawn if asked.
  2. For each agent: age it, charge living cost, remove it if energy ≤ 0, charge old-age cost, compute its observations, then eat any fruit it touches.
  3. Each predator acts and eats any agent it touches.
  4. Fruits grow or rot. Trees may spawn, grow, die or drop fruit.
  5. Time and score each go up by 0.1.
  6. A new predator may spawn.
- The observations you receive were computed **after** your agents moved this step, and before predators moved.
- A seed fixes the map and the starting layout. Later random events depend on what your agents do, so the same seed plays out differently for different policies.

### 3.2 Score

- **+0.1 for every step the species is alive.** Surviving the full game is worth about 3000.
- **+fruit energy ÷ 1000** for every fruit eaten. A fully ripe fruit is worth 0.06.
- **−agent energy ÷ 100** for every agent a predator eats. An agent with 500 energy costs 5.
- In the heuristic runs, fruit bonus minus predator penalty was between −27 and +37 per game. **Score is almost entirely survival time.**

### 3.3 Actions

| Field | Meaning |
|---|---|
| `agent_id` | Which agent the action is for. |
| `move_distance` | Units to move. Negative becomes 0. Anything above `sprint_speed` is cut to `sprint_speed`. |
| `move_direction` | Radians **relative to the agent's current heading**, not absolute. 0 is straight ahead. |
| `turn_angle` | Radians to rotate, applied **after** the move. Any size is allowed. |
| `spawn_agent` | Try to create a child. |

- Moving in any direction is free apart from the distance cost. **You don't need to turn to move sideways or backwards.** Turning only changes where the vision cone points.
- An agent that gets no action does nothing and only pays the living cost.

### 3.4 Energy

| Cost | Exact rule in the code | Per second at full rate |
|---|---|---|
| Living | 0.1 per step. Every biome has a drain multiplier of 1.0, so biome never changes this. | 1 |
| Walking | 0.05 per unit, up to `speed` (10 by default) | 5 at full walking speed |
| Sprinting | Walking cost for the first `speed` units, plus 0.5 per unit above that | 55 at a full sprint of 20 |
| Turning | min(π, \|angle\|) ÷ 2π per step, so at most 0.5 however far you turn | up to 5 |
| Spawning | 100, taken from the parent | - |
| Old age | Once age passes `max_age`, an extra 0.01 × age **per step** | 0.1 × age, so 6 to 12 at age 60 to 120 |

- **Movement is the largest cost.** Walking at full speed costs five times more than simply being alive.
- Distance cost is charged on the distance you **ask for**, before terrain slows you down. Moving through a swamp or river pays full price for less ground.
- Below 20% of max energy an agent **cannot sprint**. The move is cut to walking speed.
- If an obstacle blocks the move, the engine tries other directions in 10° steps. If none work, the agent stays put and still pays.
- Energy is capped at `max_energy`, which is 500 by default.

### 3.5 Life cycle and reproduction

- Every agent gets a random `max_age` between 60 and 120 seconds when it is created.
- **Derived:** after `max_age` the old-age drain grows to 6 to 12 energy per second. That empties even a full 500-energy agent in under a minute. **No single agent lives much longer than about 3 minutes, so the species only survives 3000 s by reproducing continuously.**
- A spawn only happens if the parent has **more than 100 energy after its move and turn this step**. The parent pays 100.
- The child starts with **75 energy**, not a share of the parent's energy. It appears 10 to 30 units away from the parent with age 0 and a new random `max_age`.
- **Mutation:** each of the 6 traits has a 10% chance to be multiplied by a random factor between 0.5 and 1.5. Upper limits apply:

| Trait | Default | Cap |
|---|---|---|
| speed | 10 | 20 |
| sprint_speed | 20 | 40 |
| max_energy | 500 | 1000 |
| hearing_radius | 50 | 100 |
| vision_range | 200 | 400 |
| vision_angle | π/3 (60°) | π/2 (90°) |

There is no lower limit, so traits can drift down as well as up.

### 3.6 Senses and observations

- **Hearing/smell:** everything within `hearing_radius` in any direction, **through walls**.
- **Vision:** things further away but within `vision_range` and inside the vision cone, blocked by walls.
- Obstacle edges are only reported when they are seen.

| Type | Fields |
|---|---|
| `Fruit` | `distance`, `angle` |
| `Tree` | `distance`, `angle` |
| `Agent` | `distance`, `angle`, `rel_dir`, `id` |
| `Predator` | `distance`, `angle`, `rel_dir` |
| `Edge` | `coords`: start and end points, in the agent's own frame (rotated so the heading points along +x) |

- `angle` is relative to the agent's heading and lies between −π and π.
- For a predator, `rel_dir` is where your agent sits relative to the **predator's** heading. A value near 0 means the predator is looking straight at you.
- Observations do **not** include fruit ripeness, tree age, your own position or your heading. A policy that needs those has to track them itself.
- Status fields per agent: `energy`, `biome`, `age`, `speed`, `sprint_speed`, `hearing_radius`, `vision_angle`, `vision_range`, `max_energy`. Note that `max_age` is **not** reported.

### 3.7 Food: fruit and trees

**Fruit**
- Starts at 20 energy and gains 2 energy per second up to 60, which it reaches after 20 s.
- **Rots and disappears 50 s after it appears.**
- Eaten automatically when the agent's centre is within its radius plus the agent's size (5).

**Trees**
- Age 1 per second. Starting trees begin with a random age of 20 to 80.
- Drop fruit only once they are at least 20 s old. The chance per second equals the biome's fruit rate (see 3.8), and the fruit lands 1 to 3 tree radii away. Each drop gets one placement attempt and is lost if the spot is blocked.
- After age 50, a tree dies each step with probability ((age − 50) ÷ 50)². **Derived:** trees live about 58 s on average, and starting trees older than about 55 die within seconds of the game starting.
- **Tree spawning slows down exponentially.** The chance of a new tree per step is 20 ÷ (number of trees) × 0.5^(time ÷ 300), and the tree only appears if a roll against the local biome's tree rate succeeds.

**Derived, not measured: food supply falls steadily over the game.** Setting tree births equal to tree deaths gives roughly:

| Sim time (s) | Trees on the map | Fruit appearing per second, whole map |
|---|---|---|
| 0 | ~80 | ~4.9 |
| 600 | ~40 | ~2.5 |
| 1200 | ~20 | ~1.2 |
| 1800 | ~10 | ~0.6 |
| 3000 | ~2 to 3 | ~0.15 |

The tree count halves about every 600 s. At the end of the game the whole map produces about one fruit every 7 seconds, roughly 9 energy per second at best. That can feed only a handful of agents, even before counting search and old age. **This should be checked with instrumentation before anyone relies on it.**

### 3.8 Biomes

The map has 10 regions, each randomly forest, grassland, swamp or desert, plus one river crossing the map.

| Biome | Movement multiplier | Tree spawn rate | Fruit per second per mature tree |
|---|---|---|---|
| Forest | 1.0 | 1.0 | 0.10 |
| Grassland | 1.0 | 0.5 | 0.10 |
| Swamp | 0.5 | 0.9 | 0.08 |
| Desert | 0.8 | 0.1 | 0.05 |
| River | 0.3 | 0.0 | 0.0 |

- The movement multiplier uses the biome at the creature's current position and applies to predators too.
- The river's "flow speed" setting exists in the code but is **never used**. Rivers don't push anything.
- Energy drain is identical everywhere (see 3.4).

### 3.9 Predators

| Property | Value |
|---|---|
| Size | 10 (agents are 5) |
| Walk speed | 11 |
| Sprint speed | 15 |
| Hearing radius | 60 |
| Vision range | 250 |
| Vision cone | 60° |
| Max energy | 200 |

**Spawning**
- The chance per step is 0.1 × time × 0.0001 ÷ (number of predators), and the spawn fails if the chosen spot is blocked.
- **Predators never die.** Nothing in the code removes them.
- **Derived:** the count grows roughly as 0.01 × time, which is about 10 predators at 1000 s and about 30 at 3000 s. This fits the observed 3 to 9 predators at 485 to 944 s.

**Rest cycle**
- A predator spawns asleep with 0 energy. It regains 30 energy per second and wakes above 100, about 3.4 s later.
- It pays the same movement and turning costs as agents, but **no living cost**. It falls asleep again at 0 energy.
- Eating an agent gives the predator that agent's energy.
- **Derived:** an awake predator that walks and never eats stays awake about 18 s.

**Hunting**, aimed at the nearest agent it perceives:
- **It chases directly if** the agent is facing away from it (more than 90° off the agent's heading) **or** the agent is within 90 units.
- **Otherwise it circles:** it sprints at 45° off the line to the agent, trying to get behind it.
- If no agent is perceived, it steers away from walls or wanders.

**Consequences**
- A walking agent (10) is slower than a chasing predator (15).
- A sprinting agent (20) is faster, but sprinting costs 55 energy per second and is impossible below 20% energy.
- **Facing a predator that is more than 90 units away stops it from chasing directly.** The heuristic relies on this.

### 3.10 Where the README is wrong or unclear

| Topic | README says | Code does |
|---|---|---|
| `move_direction` | Absolute direction | Relative to the agent's heading |
| Old-age cost | "increase by 0.01 × age" | 0.01 × age **per step**, which is 0.1 × age per second |
| Turning cost | abs(angle) / 2 * pi (ambiguous) | min(π, \|angle\|) ÷ 2π per step |
| Biome energy | Living cost depends on the biome | The multiplier is 1.0 in every biome |
| Spawn energy | Not stated | The child gets a fixed 75. The parent needs more than 100 after this step's move. |
| Sprint limit | Not stated | No sprinting below 20% energy |
| Rivers | Not stated | Flow speed is defined but unused |
| Predators | Not stated | They never die, and their number grows with time |

### 3.11 Competition rules that affect the design

- **Validation attempts** are unlimited, use random seeds, and run one at a time in a queue.
- **The final evaluation can only be run once.** It plays 3 games on preset seeds and scores their average, so the server must stay up through all three.
- **Time limits:** the evaluator waits at most 10 s for each response and stops the run once total waiting reaches 600 s. Over 30,000 steps that means an **average round trip below 20 ms**, network included.
- **Determinism:** games are only reproducible on the same operating system. The official evaluator runs on Linux, so Windows scores won't match it exactly.

---

## 4. What we have learned

### 4.1 Verified from runs

1. **The local harness matches the official loop.** On seed 1 the dummy scored 21.7574 in 217 steps both through `simulation_server.py` and through our own runner.
2. **The dummy dies in about 20 s**, from running out of energy: it sprints at random and turns constantly.
3. **Saving energy plus reproducing buys about 35 times more survival.** The heuristic's population peaks at 35 to 68 agents and then collapses between 485 and 944 s.
4. **Results vary a lot between seeds.** The heuristic's standard deviation is about 200 on a mean of about 700. With only 5 seeds, small improvements can't be told apart from noise.
5. **Package versions change results.** One of five heuristic seeds scored differently under numpy 2.2.5 than under 2.3.5. Every scored run now records its interpreter and package versions.
6. **Simulation is CPU-bound and slow for large populations.** A full 3000 s game with a big population is estimated at 10 to 20 minutes per seed. On the cluster's 4 CPUs that means about 4 games in parallel, and the T4 GPU does nothing for the simulation itself.

### 4.2 Hypotheses about why populations collapse (untested)

The cause of death is **not logged yet**, so none of these has been checked:

- **Food runs out.** Tree spawning halves every 300 s, so fruit supply falls through the game (section 3.7).
- **Predators pile up.** Their number grows with time and they never die (section 3.9).
- **Old age forces constant reproduction**, and each child costs 100 energy (section 3.5).
- **Starting conditions differ per seed.** Food-poor maps or maps with a lot of river may collapse early.

The obvious next step is to count deaths by cause (starvation, predator, old age) and log fruit eaten, tree count and predator count over time. That would show which of these dominates.

---

## 5. Open decisions and loose ends

In this project, strategy decisions (algorithms, thresholds, seeds, budgets) belong to the strategy lead (Astra) or the team, not the coding agent. These are waiting on them:

1. **Heuristic parameters.** All six were chosen by the coding agent. Accept `heuristic_v0` as the baseline, or specify new values?
2. **Seed sets.** Seeds 0 to 4 were an ad hoc choice. Define the development seeds, a separate holdout set, and how many seeds a comparison needs.
3. **Re-baselining.** Run the baselines again on the cluster's Linux environment with the pinned packages, since that is closer to the official evaluator.
4. **Cause-of-death logging.** Approve the instrumentation described in 4.2.
5. **Report location.** The report was written to `trial_racecar/report.md` as instructed, but that name looks copied from another challenge. Confirm the right path.
6. **Uncommitted duplicate results.** `results/index.csv` has two extra rows, and there are two extra per-seed files, all from 11:03. They are full scored runs made with the global Python and exactly repeat C1-E00 and C1-E01. They were not written intentionally; the likely cause is a notebook run that was stopped late, or a second agent session working in the same folder. The index is append-only, so they were not deleted. Decide whether to commit them, or add an index row marking them invalid.
7. **Submission endpoint.** It still serves the dummy policy, and the API key header is unknown. Plugging in a real policy and checking latency on the Azure VM is still to do.

## 6. Working rules used on this project

- Every scored run has an experiment ID (`C1-E00`, `C1-E01`, and so on) and fixed seeds.
- `results/index.csv` is append-only. A bad run is invalidated by appending a new row, never by editing or deleting.
- Commit before and after every scored run. Commit messages start with the experiment ID, or `[NONE]` for non-experiment changes. Never force-push, and don't push without being asked.
- Policy logic lives in `.py` files. The notebook only triggers runs, and scored runs always use a fresh Python process.
- Only the submission server gets fallback error handling. Its fallback is tested by forcing a failure with `INJECT_FAILURE=1`.
