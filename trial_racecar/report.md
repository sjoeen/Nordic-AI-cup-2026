# Survival Simulator — First Baselines (C1-E00, C1-E01)

**Date:** 2026-09-17
**Challenge:** Nordic AI Cup 2026, Challenge 1 (Survival Simulator)
**Purpose:** Get the game running end-to-end from a Jupyter notebook and measure a first baseline (requested by the human while the submission server is being set up).

---

## 1. Experiments and code revision

| Experiment | Agent / config | Config hash | Code revision | Results commit |
|---|---|---|---|---|
| C1-E00 | `dummy_v0` — starter random policy, served exactly as in the starter `agent_server.py` | `e27e72fb908f` | `9ff405b` | `f3b7944` |
| C1-E01 | `heuristic_v0` — rule-based policy, **provisional parameters** | `2793444c26b8` | `9ff405b` | `f3b7944` |

- **Seeds:** 0, 1, 2, 3, 4 (both experiments)
- **Environment version:** starter simulator, unmodified (`src/`)
- **OS:** Windows 11. The official evaluator runs on Linux, so exact per-seed scores may differ there (see the note on determinism in the starter README).
- **How it ran:** `survival_baseline.ipynb`, executed top to bottom in a fresh kernel. Each scored run was a fresh subprocess (`python -m training.evaluate`), with 4 workers.
- **Per-game wall-clock cap:** 1800 s. No game came close to it.
- **Revision flag:** the C1-E01 index row says `rev=9ff405b-dirty`. The only uncommitted files at that point were C1-E00's result files, not code. The evaluator was later fixed (`81557dd`) so output files no longer mark the tree dirty.

### Evaluator fidelity check

`training/episode.py` copies the official loop in `simulation_server.py`:
- the first step is taken with no actions
- the agent sees only live agents
- the game ends when no agents remain or sim time passes 3000 s

On seed 1, the dummy agent scores **21.7574 in 217 steps** in both loops, identical to the official loop run earlier.

---

## 2. Policy versions

### C1-E00 — `dummy_v0`
The starter `dummy_agent_policy.action_decision` with a fresh `Random(1)` per request:
- random move distance up to sprint speed
- random turn of up to ±45°
- always requests a spawn

### C1-E01 — `heuristic_v0` (`agents/heuristic.py`, `training/configs/heuristic_v0.json`)
Rules are checked in this order, for each agent on each step:

1. **Predator observed:** turn to face the nearest predator and move directly away from it. Sprint if it is closer than `flee_sprint_dist`, otherwise walk.
2. **Fruit observed:** turn toward the nearest fruit and walk onto it, never overshooting.
3. **Tree observed:** walk toward the nearest tree until within `tree_wait_dist`, then stand still and turn slowly to scan.
4. **Nothing observed:** walk forward at `wander_speed_frac × speed` while turning slowly.
5. **Spawning:** request a spawn when energy > `spawn_energy_threshold` and no predator is observed.

| Parameter | Value | Where it came from |
|---|---|---|
| `spawn_energy_threshold` | 175 | Executor choice: 100 spawn cost + 75, the energy a newborn agent starts with |
| `wander_speed_frac` | 0.5 | Executor choice |
| `scan_turn` | π/16 rad per step | Executor choice |
| `tree_wait_dist` | 40 | Executor choice (fruit spawns 1–3 tree radii from a tree) |
| `flee_sprint_dist` | 90 | Taken from the engine: predators always chase within 1.5 × their hearing radius of 60 |
| `face_predator` | true | Taken from the engine: predators otherwise chase only agents that face away |

> **These parameters were NOT specified by Astra or the human.** They were picked to get a working first look at the game, as the human asked. Treat C1-E01 as provisional until they are signed off or replaced.

---

## 3. Results

"Extinction" means all agents died before 3000 s; the tick is the step on which that happened.

### C1-E00 — dummy_v0

| Seed | Score | Survived 3000 s? | Extinction tick | Peak agents | Agents spawned |
|---|---|---|---|---|---|
| 0 | 15.70 | no | 157 | 10 | 5 |
| 1 | 21.76 | no | 217 | 10 | 5 |
| 2 | 15.70 | no | 157 | 10 | 5 |
| 3 | 21.55 | no | 214 | 10 | 5 |
| 4 | 19.94 | no | 199 | 10 | 5 |

**Mean 18.93, std 3.03, median 19.94, min 15.70, max 21.76 (n = 5).** Wall-clock for all 5 seeds: 8.8 s.

### C1-E01 — heuristic_v0

| Seed | Score | Survived 3000 s? | Extinction tick | Sim time (s) | Peak agents | Agents spawned | Predators at end | Score − time |
|---|---|---|---|---|---|---|---|---|
| 0 | 657.81 | no | 6523 | 652.3 | 57 | 201 | 3 | +5.51 |
| 1 | 531.96 | no | 5366 | 536.6 | 65 | 204 | 8 | −4.64 |
| 2 | 828.00 | no | 8549 | 854.9 | 47 | 251 | 6 | −26.90 |
| 3 | 981.09 | no | 9440 | 944.0 | 68 | 307 | 9 | +37.09 |
| 4 | 492.52 | no | 4857 | 485.7 | 35 | 105 | 4 | +6.82 |

**Mean 698.28, std 205.21, median 657.81, min 492.52, max 981.09 (n = 5).** Mean survival: 6947 steps. Wall-clock for all 5 seeds (4 workers): 269.9 s; slowest single game 269 s.

"Score − time" is the fruit bonus minus the predation penalty.

### Decision latency (heuristic_v0, policy time only, whole population per step)

| Seed | Mean (ms) | p95 (ms) | Max (ms) |
|---|---|---|---|
| 0 | 0.099 | 0.235 | 3.23 |
| 1 | 0.122 | 0.295 | 1.98 |
| 2 | 0.089 | 0.201 | 2.63 |
| 3 | 0.124 | 0.279 | 75.63 |
| 4 | 0.062 | 0.145 | 2.78 |

This is far below the evaluator's 10 s timeout per request and the 600 s total-wait budget. The 75.6 ms maximum on seed 3 was a single spike while 4 workers ran in parallel. Network time is not included.

---

## 4. Comparison against the current best baseline

Before these runs there was no measured baseline (`best_score = NONE`).

| | C1-E00 dummy_v0 | C1-E01 heuristic_v0 | Delta |
|---|---|---|---|
| Mean score | 18.93 | 698.28 | **+679.35 (≈ 36.9×)** |
| Median score | 19.94 | 657.81 | +637.87 |
| Worst seed | 15.70 | 492.52 | +476.82 |
| Full-game survival | 0 / 5 | 0 / 5 | — |

Every heuristic seed beats every dummy seed; the heuristic's worst seed is about 23× the dummy's best. The gap is far larger than the variance of either run (heuristic std 205), so the improvement is real, even with n = 5.

---

## 5. Conclusion

1. **The pipeline works end-to-end from Jupyter:**
   - tests
   - scored multi-seed evaluation in fresh processes
   - the append-only `results/index.csv`
   - per-run CSVs
   - headless video rendering

   The local loop reproduces the official evaluator's score exactly.
2. **The dummy policy dies almost immediately (~20 s)**, mostly from energy exhaustion: it sprints at random and turns constantly.
3. **Simple energy discipline plus reproduction keeps the species alive about 35× longer (~700 s).** That means walking instead of sprinting, going for fruit and trees, and spawning above a threshold. The population peaks at 35–68 agents.
4. **No policy survives the full 3000 s.** Score is dominated by survival time (fruit and predation changes are ±40 at most), so the main lever for a higher score is keeping the population from collapsing late in the game. At the end of heuristic runs 3–9 predators were alive, and the predator spawn chance grows with time. Whether collapse comes from predators, running out of food, or old age has **not** been measured, because deaths are not logged by cause yet.
5. **Variance across seeds is large** (485–944 s survived), so later comparisons need more than 5 seeds to detect smaller improvements.

---

## 6. Flags for Astra

1. **Heuristic parameters need sign-off.** All six `heuristic_v0` parameters were chosen by the executor (table in §2), which goes against the "never invent thresholds" rule. The human's request prompted them. Should `heuristic_v0` be accepted as the working baseline, or should the parameters be re-specified?
2. **Seed set.** Seeds 0–4 were chosen by the executor. Please define the official development seeds, a separate holdout set, and n.
3. **Interpreter mismatch in the scored runs.** The notebook kernel was the global Python 3.13, with numpy 2.2.5, scipy 1.16.2 and fastapi 0.135.1, not the pinned venv (numpy 2.3.5, scipy 1.16.3). The dummy's seed-1 score still matches the pinned venv exactly, so the effect is probably small, but it is not verified for the heuristic. **The runs were not repeated**, per instructions. Since then:
   - each run logs its interpreter and package versions
   - the notebook warns when versions differ from the pins
   - a `naic-survival` kernel pointing at the venv is registered locally

   Decide whether to re-baseline on the cluster's Linux environment, which matches the evaluator's OS.
4. **Cause-of-death logging is missing.** Adding counters for predation, starvation and old-age deaths, plus fruit eaten, would show why populations collapse at ~500–950 s. This is an instrumentation change only. Approve?
5. **Engine facts that affect strategy:**
   - `move_direction` is relative to the agent's heading, not absolute as the README says.
   - Past a random max age of 60–120 s, the extra old-age drain is `0.01 × age` **per step**, not per second.
   - Below 20 % energy, agents cannot sprint.
   - Predators only chase agents facing away from them or within 90 units.
6. **Runtime.** A full 3000 s game with a large population will take roughly 10–20 min of CPU per seed at the observed ~1 min per 250 s simulated. That matters for training budgets and seed counts on the 4-CPU cluster.
7. **Report path.** This file was written to `trial_racecar/report.md` as instructed. That name looks copied from another challenge; please confirm the intended path for C1 reports.

---

## 7. Artifacts

- Notebook trigger: `Nordic-AI-Cup-2026-main/survival-simulator/survival_baseline.ipynb`
- Results index: `Nordic-AI-Cup-2026-main/survival-simulator/results/index.csv`
- Per-seed results:
  - `results/C1-E00_dummy_v0_20260917-104326.csv`
  - `results/C1-E01_heuristic_v0_20260917-104335.csv`
- Code:
  - `agents/heuristic.py`, `agents/dummy.py`
  - `training/episode.py`, `training/evaluate.py`, `training/render.py`
  - `training/configs/*.json`
- Tests: `tests/test_agents.py` (5 passed)
- Video of heuristic_v0, seed 0, first 300 s (inspection only, not scored; not in git): `logs/render_heuristic_v0_seed0.mp4`
