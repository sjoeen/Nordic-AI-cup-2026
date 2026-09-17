# C1-MECH-P01: Predator facing and retreat mechanism study

**Type:** RUN of a synthetic mechanism study. These are not competition scores and select no policy.
**Date:** 2026-09-17, 13:20–13:35 (+02:00)
**Status:** complete. 168/168 main episodes, 7/7 fresh-process repeats and 7/7 uninstrumented repeats ran. 11 smoke checks used 10 episodes of at most 5 steps each. No missing or invalid cases.

## 0. Provenance

| Item | Value |
|---|---|
| Repo / branch | `sjoeen/Nordic-AI-cup-2026`, local checkout `C:\Users\sjoen\PycharmProjects\NordicAI2026`, branch **`main`** (the `challenge-1` branch exists but was not checked out; I did not switch branches) |
| Engine source | commit `d9b5d9c`; `src/` clean. Per-file sha256 in `manifest.json` |
| Harness | `training/predator_mechanics.py`, committed in `7b6805c` before the main matrix and unchanged since |
| Unrelated dirty state (untouched) | staged deletions `MECHANICS_QA.md`, `trial_racecar/report.md`; modified `results/index.csv`, `survival_baseline.ipynb`; untracked `results/C1-E0*_2026-09-17-1*.csv`; `../CLAUDE.md` |
| OS / Python | **Windows 11** (not Linux; no Linux environment was available locally), CPython 3.13.7 global interpreter |
| Packages | numpy 2.2.5, scipy 1.16.2, shapely 2.1.2, pygame 2.6.1, pydantic 2.12.5, matplotlib 3.10.6 (already installed; nothing downloaded) |
| Model | The session ran on **claude-opus-5**. The task specified Sonnet, and the session model could not be changed from inside the session. No subagents were used |
| Budget used | 182 full episodes. Simulation wall-clock ≈ 26 s total (main 23.5 s, repeat 0.33 s, uninstrumented 0.26 s, smoke ≈ 1 s). Task wall-clock ≈ 15 min. 1 CPU worker, no GPU, no paid resources |
| Reproduce | `python -m training.predator_mechanics smoke`, then `main`, then `repeat --tag repeat`, then `repeat --tag nolog` (each in a fresh process), then `python experiments/C1-MECH-P01/analyze.py` |

**Pending seed-1 repeat.** No separate repeat report exists. This checkout has four local heuristic_v0 result files (`results/C1-E01_heuristic_v0_20260917-{104335,110347,120435,121101}.csv`). In all four, seed 1 ended at sim_time 536.6 s with score 531.96, and seeds 0–4 were identical across files. The 625.6 s and 980.7 s seed-1 records mentioned in CLAUDE.md are not in this checkout, so the discrepancy could not be checked here. I reran no full games.

## 1. Source facts (verified in this checkout)

Every value reported in CLAUDE.md matched the source, so the fixtures were not changed.

- **[FACT] Step order** (`src/utils/simulation.py:43-79`):
  1. `agent_step`: move, then turn, then spawn (`environment.py:602-623`).
  2. `non_agent_step`, agent part (`environment.py:635-672`): age += dt; energy −= dt × biome drain (1.0 on grassland); removal if energy ≤ 0; old-age drain; **the controller's observation is computed here, after the agent moves but before the predator moves**.
  3. `non_agent_step`, predator part (`environment.py:675-728`).
  4. Fruit and trees (RNG), then time += dt and score += dt, then the predator-spawn RNG gate (`environment.py:730-762`).
  - dt = 0.1 (`src/core.py:36`).
  - As a result, the observation the controller sees is always one predator move out of date.
- **[FACT] Movement** (`environment.py:496-557`):
  - The requested distance is clipped to [0, sprint_speed].
  - If `energy < max_energy/5` and the distance is above walking speed, it is clamped to walking speed. This check uses energy **before** the move cost.
  - Cost is 0.05 per unit up to walking speed, plus 0.5 per unit above it.
  - The move direction is relative to the heading **before** this tick's turn.
  - Turning (`:559-564`) costs `min(pi,|turn|)/(2pi)` and does not affect this tick's movement.
- **[FACT] Predator tick** (`environment.py:675-728`):
  - If resting with energy > 100 (`max_energy*0.5`), it wakes and **acts on the same tick**. Otherwise it gains +3 energy per tick and does nothing.
  - An awake predator observes, calls `Predator.step`, moves, turns, and then runs the kill check. The kill check removes every local agent with `distance < 10+5 = 15`, and each kill adds that agent's energy to the predator (capped at 200).
  - The predator falls asleep if energy ≤ 0 **after** the kill check, so it can still kill on the tick it runs out.
  - Predators have no idle drain. Their sprint is clamped to 11 when energy < 40.
- **[FACT] Predator perception** (`creature.py:194-279`, `predator.py:13-14`):
  - Hearing: 60 units, no wall test.
  - Vision: 250 units within ±30°, tested against the visibility polygon.
  - `rel_dir` = (bearing from agent to predator) − (agent heading), wrapped.
- **[FACT] Target and mode** (`predator.py:31-62`):
  - The target is the nearest perceived agent, chosen fresh each tick (no target lock).
  - **Chase** if `|rel_dir| > pi/2` **or** `distance < 90`. Chase moves `min(15, distance)` in direction `clip(0.5*angle, ±0.3)` and turns the same amount. If `|angle| ≤ 0.05`, it moves straight along `angle` with no turn.
  - **Circle** otherwise. It moves 15 in direction `angle + pivot_sign*pi/4` with `pivot_sign = -np.sign(rel_dir)`, then turns toward the agent's predicted bearing.
  - **When `rel_dir == 0.0` exactly, `np.sign` returns 0, so "circle" is a straight 15-unit move toward the agent.**
  - If nothing is perceived and no edge is in view, the predator wanders using `rng.uniform(-0.1,0.1)`.
- **[FACT] Agent perception**: hearing 50, vision 200 within ±30°. An agent facing away cannot see a predator more than 50 units behind it.
- **[FACT] What heuristic_v0 does** (`agents/heuristic.py:52-57`, config `flee_sprint_dist=90`, `face_predator=true`): it moves radially away from the observed predator (`angle+pi`), turns to face it (`turn=angle`), and sprints only when the predator is within 90.
- **[FACT] Engine crash found:** `compute_visibility` raises IndexError when the local edge list is empty (`sensing.py:24-27`: `np.array([])` is 1-D). The early return at `:59` for empty edges comes after the crash, so it never runs.

## 2. The fixture

- **Arena:**
  - 6000×6000 world. The agent starts at internal (3000, 3000), reported as (0, 0).
  - The predator starts at (d, 0), heading −x, awake, with the specified energy.
  - The agent heads +x with max_energy 500, speeds 10/20, and max_age 120. Its escape line is −x.
  - Uniform grassland: movement penalty 1.0, drain 1.0.
  - Only the four boundary walls, each at least 2500 units away.
- **Built with `Environment.__new__`**, which skips random map generation and the pygame surfaces used only for drawing.
- **Deviations from a full game** (also in `config.json`):
  1. `chunk_size` is 3000 instead of 400. This works around the empty-edge crash: the far boundary edges are always in the candidate list, and all perception is still filtered by radius and cone.
  2. Instance-level no-ops for `spawn_tree`, `spawn_fruit`, `spawn_fruit_around_tree` and `spawn_predator`. Their RNG gates still run.
  3. The initial observation comes from the same `agent.observe` call used at `environment.py:649-660`, without advancing time, age or drain (verified: time=0, energy unchanged).
  4. The constructors each draw one `rng.uniform` for a random heading, which is then overwritten.
- **Controllers:**
  - Movement is always `wrap(pi − heading_acc)`, where `heading_acc` is 0 plus the sum of issued turns.
  - A arms turn `wrap(pi − heading_acc)`: a one-time −π turn costing 0.5.
  - F and S0 arms turn by the observed predator angle, or 0 if no predator is observed.
  - No spawning. Exactly one action per step.
- **Instrumentation:** wrappers on `update_entity_position`, `update_entity_direction` and `predator.step` record arguments and results, then call the original method unchanged.
  - The mode is **observed**: the emitted signals are matched exactly against each branch's output, recomputed from the same observation.
  - It is then checked against the source predicate.
- **Seeds 0, 1, 2:** Python `random`, NumPy and the environment RNG are all seeded before each fixture.

## 3. Validation

| Check | Result |
|---|---|
| Smoke (11 checks, 10 episodes of ≤ 5 steps) | 10 pass. **1 failure, caused by a wrong expectation in my test, not a harness defect:** I expected a predator with 2.0 energy to sleep on tick 1, but below 40 energy it walks 11 (cost 0.55). It actually fell asleep on tick 4 (energy −0.2), then recharged +3.0 on tick 5 while taking no action. That confirms the source ordering. Other checks confirmed: first-tick −x movement of 10 with a −π turn costing 0.5; energy components 0.5 move + 0.5 turn + 0.1 drain; a 20-unit sprint costs 5.5; the clamp at energy 99 gives 10 units; predicate matches mode; capture on tick 1 at d=20; no time advance at init; instrumented and uninstrumented runs identical over 5 steps (`smoke.json`) |
| Predicate vs observed branch | **6277 decision ticks, 0 mismatches, 0 unidentified** |
| Fresh-process repeat (d120/aE300/pE101, seed 0, 7 arms) | **7/7 identical** per-step digests (state + actions + RNG), outcomes and scores |
| Instrumentation disabled (same 7) | **7/7 identical** per-step digests. Logging does not change state or RNG use |
| Seeds 0–2 | Outcome, event time and minimum separation are identical across seeds for all 56 fixture-arms. 44/56 trajectories are bit-identical apart from RNG state. The other 12 diverge only **after** the predator loses perception, because wandering uses the RNG. **The seeds are not independent samples.** |

## 4. Results

Rows are per fixture and arm. "alive@10s" means censored, not a death at 10 s. Times are in seconds. Where seeds differ, values are shown as s0/s1/s2. "n/o" means not observed before death or the horizon.

| d | aE | pE | arm | outcome | event t | min sep | chase | circle | no-target | first clamp | pred sleep t | agent E at sleep | agent E @10s | sep @10s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 80 | 150 | 101 | S0 | captured | 0.5 | 5 | 0.5 | 0 | 0 | n/o | n/o | n/o | — | — |
| 80 | 150 | 101 | A10 / F10 | captured / captured | 1.4 / 1.4 | 10 / 10 | 1.4 / 1.3 | 0 / 0.1 | 0 | n/o | n/o | n/o | — | — |
| 80 | 150 | 101 | A15 / F15 | captured / captured | 5.0 / 5.5 | 14 / 14 | 5.0 / 3.7 | 0 / 1.8 | 0 | 1.7 / 1.8 | n/o | n/o | — | — |
| 80 | 150 | 101 | A20 / F20 | captured / captured | 6.0 / 6.0 | 14 / 14 | 6.0 / 4.1 | 0 / 1.9 | 0 | 1.0 | n/o | n/o | — | — |
| 80 | 150 | 200 | S0 | captured | 0.5 | 5 | 0.5 | 0 | 0 | n/o | n/o | n/o | — | — |
| 80 | 150 | 200 | A10 / F10 | captured / captured | 1.4 / 1.4 | 10 | 1.4 / 1.3 | 0 / 0.1 | 0 | n/o | n/o | n/o | — | — |
| 80 | 150 | 200 | A15 / F15 | captured / captured | 3.0 / 3.1 | 10 | 3.0 / 1.3 | 0 / 1.8 | 0 | 1.7 / 1.8 | n/o | n/o | — | — |
| 80 | 150 | 200 | A20 / F20 | captured / captured | 3.2 / 3.2 | 10 | 3.2 / 1.3 | 0 / 1.9 | 0 | 1.0 | n/o | n/o | — | — |
| 80 | 300 | 101 | S0 | captured | 0.5 | 5 | 0.5 | 0 | 0 | n/o | n/o | n/o | — | — |
| 80 | 300 | 101 | A10 / F10 | captured / captured | 1.4 | 10 | 1.4 / 1.3 | 0 / 0.1 | 0 | n/o | n/o | n/o | — | — |
| 80 | 300 | 101 | A15 | alive@10s | — | 80 | 8.9/6.3/8.8 | 0 | 0.8/3.3/0.9 | 6.6 | 9.7/9.6/9.7 | 78.8–79.4 | 77.0 | 242.2/290.3/242.8 |
| 80 | 300 | 101 | F15 | alive@10s | — | 80 | 0 | 7.1/6.3/7.1 | 2.6/3.3/2.5 | 6.6 | 9.7/9.6/9.6 | 79.3–79.9 | 77.5 | 267.7/290.3/279.0 |
| 80 | 300 | 101 | A20 / F20 | alive@10s / alive@10s | — | 80 | 2.8 / 0 | 0 / 2.8 | 6.8 | 3.7 | 9.6 | 61.9 / 62.4 | 59.5 / 60.0 | 296.2/308.4/347.2 (both) |
| 80 | 300 | 200 | S0 | captured | 0.5 | 5 | 0.5 | 0 | 0 | n/o | n/o | n/o | — | — |
| 80 | 300 | 200 | A10 / F10 | captured / captured | 1.4 | 10 | 1.4 / 1.3 | 0 / 0.1 | 0 | n/o | n/o | n/o | — | — |
| 80 | 300 | 200 | A15 / F15 | alive@10s / alive@10s | — | 53 | 10.0 / 2.6 | 0 / 7.4 | 0 | 6.6 | n/o | n/o | 77.0 / 77.5 | 53 (all) |
| 80 | 300 | 200 | A20 | alive@10s | — | 80 | 3.0/4.2/3.0 | 0 | 7.0/5.8/7.0 | 3.7 | n/o | n/o | 59.5 | 282.2/178.1/290.8 |
| 80 | 300 | 200 | F20 | alive@10s | — | 80 | 0 | 3.0/4.2/3.0 | 7.0/5.8/7.0 | 3.7 | n/o | n/o | 60.0 | 282.2/228.7/290.8 |
| 120 | 150 | 101 | S0 | captured | 0.8 | 0 | 0.5 | 0.3 | 0 | n/o | n/o | n/o | — | — |
| 120 | 150 | 101 | A10 / F10 | captured / captured | 2.2 | 10 | 2.2 / 1.3 | 0 / 0.9 | 0 | n/o | n/o | n/o | — | — |
| 120 | 150 | 101 | A15 / F15 | captured / captured | 9.0 / 9.5 | 14 | 9.0 / 6.5 | 0 / 3.0 | 0 | 1.7 / 1.8 | n/o | n/o | — | — |
| 120 | 150 | 101 | A20 / F20 | alive@10s / alive@10s | — | **17** | 9.7 / 6.2 | 0 / 3.5 | 0 | 1.0 | 9.7 | 46.3 / 46.8 | 44.5 / 45.0 | 47 (all) |
| 120 | 150 | 200 | S0 | captured | 0.8 | 0 | 0.5 | 0.3 | 0 | n/o | n/o | n/o | — | — |
| 120 | 150 | 200 | A10 / F10 | captured / captured | 2.2 | 10 | 2.2 / 1.3 | 0 / 0.9 | 0 | n/o | n/o | n/o | — | — |
| 120 | 150 | 200 | A15 / F15 | captured / captured | 3.8 / 3.9 | 10 | 3.8 / 1.3 | 0 / 2.6 | 0 | 1.7 / 1.8 | n/o | n/o | — | — |
| 120 | 150 | 200 | A20 / F20 | captured / captured | 4.0 / 4.0 | 10 | 4.0 / 1.3 | 0 / 2.7 | 0 | 1.0 | n/o | n/o | — | — |
| 120 | 300 | 101 | S0 | captured | 0.8 | 0 | 0.5 | 0.3 | 0 | n/o | n/o | n/o | — | — |
| 120 | 300 | 101 | A10 / F10 | captured / captured | 2.2 | 10 | 2.2 / 1.3 | 0 / 0.9 | 0 | n/o | n/o | n/o | — | — |
| 120 | 300 | 101 | A15 / F15 | alive@10s / alive@10s | — | 120 | 5.3 / 0 | 0 / 5.3 | 4.3 | 6.6 | 9.6 | 79.4 / 79.9 | 77.0 / 77.5 | 302.6/324.6/331.3 (both) |
| 120 | 300 | 101 | A20 / F20 | alive@10s / alive@10s | — | 120 | 2.2 / 0 | 0 / 2.2 | 7.8 | 3.7 | n/o (pred E 1.4 at 10 s) | n/o | 59.5 / 60.0 | 303.2/434.9/327.6 (both) |
| 120 | 300 | 200 | S0 | captured | 0.8 | 0 | 0.5 | 0.3 | 0 | n/o | n/o | n/o | — | — |
| 120 | 300 | 200 | A10 / F10 | captured / captured | 2.2 | 10 | 2.2 / 1.3 | 0 / 0.9 | 0 | n/o | n/o | n/o | — | — |
| 120 | 300 | 200 | A15 / F15 | alive@10s / alive@10s | — | 93 | 10.0 / 0 | 0 / 10.0 | 0 | 6.6 | n/o | n/o | 77.0 / 77.5 | 93 (all) |
| 120 | 300 | 200 | A20 / F20 | alive@10s / alive@10s | — | 120 | 2.2 / 0 | 0 / 2.2 | 7.8 | 3.7 | n/o | n/o | 59.5 / 60.0 | 303.2/434.9/327.6 (both) |

Full statistics (n, mean, sample std, median, min, max) are in `stats_by_fixture_arm.csv`. Because the seeds coincide on every metric above except post-perception-loss wandering, std is 0 for all of them. That reflects determinism, not precision.

### Matched comparisons, F − A (`matched_comparisons.csv`, 72 pairs = 8 fixtures × 3 speeds × 3 seeds)

- **Capture outcome differs in 0/72 pairs.** Minimum separation differs in 0/72.
- **Predator path identical in 69/72 pairs.** The maximum predator position difference over common ticks is 0.0, even when the F arm spent up to 10 s labelled "circle" and the A arm the same time labelled "chase".
- **Agent energy at common alive times:** F − A = +0.5, which is exactly the A arm's one-time turn cost. There is one mixed case, below.
- **Capture delayed at speed 15 with aE=150:** by 0.5 s at pE=101 (d80: 5.0→5.5 s; d120: 9.0→9.5 s) and by 0.1 s at pE=200.
  - Cause: the A arm's 0.5 turn cost pushes it below the sprint threshold one tick earlier (first clamp at 1.7 s vs 1.8 s). The F arm gets one extra 15-unit tick, so it has +5 separation and −2.0 energy at t=5 (d120/150/101).
  - The predator's positions are identical in both arms, so **this is an energy effect, not a behaviour effect.**
- **The only real path differences** come in 3/72 pairs, all after the predator lost and re-acquired the agent off-axis. There, `rel_dir ≠ 0`, so the circle pivot was ±1:
  - d80/300/101 speed 15, seed 0: separation at 9.9 s +25.6 for F.
  - Same fixture, seed 2: +36.3.
  - d80/300/200 speed 20, seed 1: +46.9.
  - What happened (`traces/d80_ae300_pe101_F15_s0.json.gz`, from t=7.2 s): the predator re-acquired at about 240 units, moved at 45°, lost the agent from its cone, and wandered. This repeated circle → no-target → circle. Separation stayed near 240. In the A arm, the predator chased straight and closed about 1 unit per tick (it walked 11 while the agent was clamped to 10).

## 5. Answers to the questions

1. **Does facing change the predator's behaviour as expected?**
   - **The mode label changes as the source predicts** (0 mismatches). A facing agent at ≥ 90 gets "circle"; below 90 it is always "chase". Facing did not prevent a single chase decision inside 90.
   - **The movement does not change in this fixture.**
     - 1908 of 1936 circle ticks had `rel_dir == 0.0` exactly, so the pivot sign was 0 and the predator moved straight at the agent at 15. That is the same as chase at distance ≥ 15.
     - The positions stay exactly on the axis because the internal coordinates (3000, 3000) absorb the 1e-16 `sin(pi)` term.
   - [INFERENCE] The predeclared collinear fixture is degenerate for the question "does circling help". It measures the `sign(0)` edge case, not the ±45° orbit. The 28 off-axis circle ticks suggest the orbit can make the predator lose perception and stall, but that evidence comes from three pairs in which the geometry arose by chance.
2. **Does it translate into distance or survival?** Not in the predeclared geometry: 0/72 outcome changes. The F arms kept +0.5 energy by skipping the one-time turn. Up to +47 separation appeared only after perception loss and off-axis re-acquisition, and never changed an outcome within 10 s.
3. **Does walking suffice anywhere?** **No.** All 48 walk-arm episodes (A10/F10, 8 fixtures × 3 seeds) were captured, at 1.4 s (d80) or 2.2 s (d120). All 24 S0 episodes were captured, at 0.5 s or 0.8 s.
4. **When does sprint clamping break escape?**
   - Agents starting at 150 energy fall below 100 at 1.0 s (speed 20) or 1.7–1.8 s (speed 15). They then walk at 10 while the predator moves 15, or 11 once it is below 40 energy. Every aE=150 case was captured except d120/pE101 at speed 20: minimum separation 17, and the predator fell asleep at 9.7 s, about one tick before contact.
   - Agents starting at 300 energy fall below 100 at 3.7 s (speed 20) or 6.6 s (speed 15). All aE=300 cases at speed 15 or 20 were alive at 10 s.
5. **Does a well-fed predator overturn conclusions that held at 101?** **Yes.**
   - With pE=200, the predator never slept within 10 s (0/84 episodes).
   - At aE=150, every run with pE=200 was captured: speed 15 at 3.0–3.9 s (vs 5.0–9.5 s at pE=101), and speed 20 at 3.2–4.0 s (at pE=101, d120 survived).
   - At aE=300, speed 15 survived 10 s but was being chased at the end with separation 53 or 93 and still closing. The predator had 19 energy left, so the outcome after 10 s is unknown (censored).
6. **Survived until predator sleep:** 24 episodes (8 fixture-arm combinations × 3 seeds, all with pE=101). The agent was alive at 10 s in 24/24. **Agent energy at the sleep event was ≥ 20% of max_energy in 0/24** (range 46–80, so no sprint available). No wake was observed within the horizon. [FACT from source] Recharging from 0 to above 100 takes 34 ticks (3.4 s) at +3 per tick.

**One success and one failure, both from the same fixture:**
- **Success:** d120/aE150/pE101/F20, seed 0 (`traces/d120_ae150_pe101_F20_s0.json.gz`). The agent sprinted to about 165 separation, was clamped at 1.0 s, and was chased back down to 17. The predator's energy reached 0 and it slept at 9.7 s. The agent had 46.8 energy at that moment.
- **Failure:** d120/aE150/pE101/F15, seed 0 (`traces/d120_ae150_pe101_F15_s0.json.gz`). The agent was clamped at 1.8 s and captured at 9.5 s while the predator still had about 50 energy.

## 6. Limits of this evidence

- The arena is synthetic, with known collinear geometry. There is no food, terrain, walls in view, other agents, other predators, births or mutations.
- The fixture used `chunk_size` 3000 and suppressed spawning.
- Circling was exercised almost only in its `sign(0)` degenerate form.
- The horizon is 10 s. A predator that sleeps wakes about 3.4 s later with just over 100 energy, and the escaping agent had < 100 energy (no sprint) in every sleep case.
- There is one predator. The target is the nearest perceived agent each tick; attention transfer to other agents was not tested.
- Privileged quantities (predator energy, sleep, mode, positions) were logged only. The controllers saw only observations. The F controllers turned by an angle that is one predator move out of date.
- These are Windows measurements. The seeds are deterministic repeats, not independent samples.
- Nothing here measures competition score or population survival.

## 7. What is still unknown about using a decoy to protect other agents

- Whether facing an off-axis predator, where the ±45° pivot actually applies, reliably holds it off or breaks its perception. This happened only by chance in 3 pairs.
- Whether a decoy that reaches predator sleep can do anything useful afterwards. It had < 20% energy in every case, and the predator needs about 3.4 s to wake.
- Whether a predator switches to a nearer non-decoy agent. There is no target lock, so this depends on geometry.
- Energy transfer to the predator when a decoy is caught: the predator refills by the decoy's energy, up to 200.
- Behaviour against multiple predators, near walls or obstacles (edge avoidance), and with mutated traits.
- Whether these Windows results hold on Linux, and how they interact with the full-game nondeterminism reported for seed 1.

## 8. Files

All paths are under `experiments/C1-MECH-P01/`:

- `config.json`, `manifest.json` (hashes, versions, commands, budget)
- `smoke.json`
- `summary.csv` (168 rows), `stats_by_fixture_arm.csv`, `matched_comparisons.csv`, `analysis_census.json`
- `digests_main.json`, `digests_repeat.json`, `digests_nolog.json`
- `traces/*.json.gz` (168 per-tick traces, 2.2 MB, sha256 in the manifest)
- `plot_distance.png`, `plot_energy.png`, `plot_trajectories.png`. These show seed 0 for all 8 fixtures. The A (dashed) lines are hidden under F wherever the paths coincide.
- `analyze.py`

Nothing was written to `results/index.csv`.

**Harness note:** in `run_episode`, the per-tick `dist_before_kill_check` falls back to the post-move distance when the predator emits no turn, since turning does not move it. This fix was part of `7b6805c`, which is the code that ran the main matrix.

## 9. Contents of this branch copy

This `challenge-1` copy contains only the findings: this report, the summary/statistics/comparison CSVs, the census, config, manifest and plots. The code and raw data were left out: the harness `training/predator_mechanics.py`, `analyze.py`, `smoke.json`, the digest files and the 168 per-tick traces. They remain in the local `main` commits `7b6805c` (harness) and `7c45468` (full results). The hashes in `manifest.json` refer to those commits.
