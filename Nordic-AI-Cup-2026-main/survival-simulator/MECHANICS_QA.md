# Survival Simulator: Answers on Predators, Food, Population and Control

**Date:** 2026-09-17
**Scope:** answers to four open questions, plus observed agent behaviour.

Every answer is tagged with where it comes from:

- **[Code]** read directly in the simulator source (`src/`, unmodified starter code).
- **[Test]** checked with a small script against the real engine.
- **[Measured]** counted during full games (see "How the measurements were made").
- **[Derived]** calculated by hand from the code, not measured.
- **[Unknown]** cannot be answered from what we have. Nothing is guessed.

## How the measurements were made

- Script: `training/diagnostics.py`. It wraps the unmodified engine from the outside and writes `logs/diagnostics_heuristic_v0_seed<N>.json`. Nothing goes into `results/`.
- Policy: `heuristic_v0` (the provisional rule-based baseline), seeds 0 to 4, full games until extinction.
- A check on seed 0 gave the identical score with and without the wrapper, so the wrapper does not change the game.
- **All measured numbers describe this one heuristic.** A different policy will behave differently.

---

## 1. Predators

### How does a predator select a target?

**[Code]** (`src/elements/predator.py`, `environment.py`)

- Every step, an awake predator looks at the agents it perceives and targets **the nearest one**. There is no memory and no lock-on. The target is recomputed from scratch each step.
- **What it perceives:**
  - agents within 60 units in any direction, through walls
  - agents within 250 units inside a 60° cone in front of it, blocked by walls
  - only agents in the 3 × 3 block of 400-unit map chunks around it
- **Chase or circle**, decided only by the nearest agent:
  - It **chases** (turns toward the agent and sprints) if that agent is facing away from it (predator more than 90° off the agent's heading), **or** the agent is closer than 90 units.
  - Otherwise it **circles**: it sprints at 45° off the line to the agent, trying to get behind it, and turns toward where the agent will be.
- If it perceives no agent, it steers away from nearby walls or wanders slowly.
- Predators ignore fruit, trees and each other. Several predators can go after the same agent.

### How does it switch targets?

- **[Code]** It switches whenever a different agent becomes the nearest, or when the current target leaves its perception.
- **[Measured]** In 65% to 78% of target switches, the previous target was still perceived and a different agent had simply become closer.

| Seed | Target switches | Previous target still perceived | Share of steps spent circling (when an agent is perceived) |
|---|---|---|---|
| 0 | 159 | 107 | 39% |
| 1 | 378 | 276 | 37% |
| 2 | 202 | 132 | 31% |
| 3 | 267 | 209 | 32% |
| 4 | 76 | 55 | 29% |

**[Measured]** Awake predators perceived no agent at all in 63% to 82% of their steps.

### Is there a kill cooldown?

**No. [Code]**

- After moving, a predator kills **every** agent whose centre is within 15 units of its own centre (predator size 10 plus agent size 5). All of them die in the same step.
- It gains each victim's energy, capped at 200. Eating makes it stay awake longer, not shorter.
- **The only pause is sleep.** A predator falls asleep when its energy reaches 0 and wakes when it is back above 100, regaining 30 energy per second (about 3.4 s).
- **A sleeping predator does not move and does not kill, even if an agent touches it.** Agents cannot tell whether a predator is asleep: observations don't include that.
- **[Derived]** An awake predator sprinting at 15 units per step pays about 2.6 energy per step. Starting from just over 100 energy, it can chase for about 38 steps (3.8 s), covering about 570 units, before it falls asleep, unless it eats.
- **[Measured]** At the 10-second sample points, 15% to 20% of predators were asleep.

| Seed | Predator kills | Steps where one predator killed 2 or more agents | Most kills by one predator in one step |
|---|---|---|---|
| 0 | 77 | 3 | 2 |
| 1 | 108 | 2 | 2 |
| 2 | 107 | 6 | 3 |
| 3 | 70 | 4 | 2 |
| 4 | 37 | 2 | 2 |

### Is there any protection from grouping?

**No. [Code]** Nothing in the predator code or the kill check depends on how many agents are near.

- Only the **nearest** agent's distance and facing decide chase or circle. Other agents nearby, and where they face, have no effect.
- Every agent within 15 units of the predator dies in the same step, so clustered agents can be killed together. The measured multi-kills above show this happens.
- **[Unknown]** Whether some grouping behaviour helps in practice, for example many agents facing outward so more predators end up circling, has not been tested.

### Other predator facts

- **[Code]** Predators never die and are never removed.
- **[Code]** Their spawn chance per step is 0.1 × time × 0.0001 ÷ (number of predators). The spawn fails if the chosen spot overlaps an obstacle.
- **[Measured]** 3 to 12 predators were alive at the last sample before extinction (485 to 981 s).
- **[Code]** Speeds:
  - predators walk at 11 and sprint at 15
  - agents walk at 10 and sprint at 20 by default
  - sprinting is impossible below 20% of max energy

---

## 2. Food

### Is a fruit consumed by one agent only?

**Yes. [Code]**

- A fruit is removed the moment the first agent touches it: agent centre within 5 + fruit radius. That agent gets all its energy, capped at its max energy; anything above the cap is lost.
- **Tie-break:** agents are processed in the engine's internal list order, oldest spawn first. The order of actions you send does **not** change who eats first.
- One agent can eat several fruits in one step if it touches several.
- **Observation timing:** observations are computed just before eating in the same step. The next response can still list a fruit the agent has just eaten.

### How quickly do trees replenish fruit?

**[Code]**

- **New fruit comes only from trees.** The 32 starting fruits are the only other source.
- Only trees at least 20 s old drop fruit. Each gets one chance per step, equal to 0.1 × the biome's fruit rate:

| Biome | Fruit rate per second, per mature tree |
|---|---|
| Forest | 0.10 |
| Grassland | 0.10 |
| Swamp | 0.08 |
| Desert | 0.05 |
| River | 0 (trees never spawn there) |

- **So a mature forest or grassland tree drops about one fruit every 10 seconds.**
- The fruit lands 1 to 3 tree radii from the tree's centre, which is 20 to 60 units for a full-grown tree. It gets **one** placement attempt and is lost if the spot overlaps an obstacle or the map edge.
- There is **no cap** on fruit per tree or on the map.
- A fruit starts with 20 energy, gains 2 per second up to 60 (reached after 20 s), and **disappears 50 s after it appears**.
- **Tree lifetime:**
  - [Code] after age 50, each step a tree dies with probability ((age − 50) ÷ 50)²
  - [Derived] a typical tree lives about 58 s, which gives about 38 s of fruiting and about 4 fruits per tree lifetime in forest or grassland
- **New trees:** they appear at random places anywhere on the map, not next to existing trees. The spawn chance per step is 20 ÷ (number of trees) × 0.5^(time ÷ 300), and the new tree also needs to pass a roll against the local biome's tree rate.

### Can agents exhaust a local patch?

**Yes, for this heuristic. [Measured]**

- **Fruit is eaten almost as soon as it appears.** The median fruit had 24.7 to 25.6 energy when eaten, which means it was eaten about 2.5 s after appearing. Agents waiting at trees pick them clean.
- A patch around one tree yields at most about one fruit per 10 s [Code], and the tree itself dies within about a minute [Derived].

| Seed | Fruit eaten | Fruit rotted | Rotted fruit with an agent within 50 units | Rotted fruit with no agent within 200 units | Fruit on map at extinction | Trees at extinction |
|---|---|---|---|---|---|---|
| 0 | 1766 | 106 | 3 | 76 | 80 | 38 |
| 1 | 2312 | 448 | 5 | 373 | 53 | 33 |
| 2 | 1789 | 332 | 8 | 291 | 60 | 28 |
| 3 | 2270 | 698 | 6 | 611 | 78 | 40 |
| 4 | 806 | 365 | 3 | 332 | 89 | 44 |

- **[Measured]** 72% to 91% of rotted fruit had no agent within 200 units. Fruit rots where agents aren't, not next to them.
- **[Measured]** **Food was not gone when the population died.** 53 to 89 fruits and 28 to 44 trees were on the map at extinction, in all 5 games.
- **[Derived]** Tree numbers should keep falling after 1000 s, because tree spawning halves every 300 s. **[Unknown]** No game lasted past 981 s, so late-game food supply has never been measured.

---

## 3. Population

### What exactly causes energy drain?

**[Code]** These are all the energy costs in the engine:

| Drain | Rule |
|---|---|
| Living | 0.1 per step. The biome multiplier is 1.0 everywhere, so biome never changes it. |
| Walking | 0.05 per unit, up to the agent's `speed` |
| Sprinting | Walking cost up to `speed`, plus 0.5 per unit above it |
| Turning | min(π, \|turn angle\|) ÷ 2π per step, so at most 0.5 |
| Spawning | 100, paid by the parent |
| Old age | Once age exceeds `max_age`: an extra 0.01 × age per step |

- Movement cost is charged on the distance **requested** (after capping at sprint speed), before terrain slows the agent. Swamps and rivers cost full price for less ground.
- **[Code]** `max_age` is drawn uniformly between 60 and 120 s for every agent, including children. It is **not** inherited and **not** reported to the policy.

### What exactly causes death?

**[Code]** There are only two ways to die:

1. **Energy at or below 0.** Checked at the start of the agent's update each step. Starvation and old age both kill this way, because old age only adds drain.
2. **Touched by an awake predator.**

There is no hard age limit, no damage from terrain, and no other death.

**[Measured]** Deaths by cause across all 5 games (1219 deaths):

| Cause | Share, all games | Range per game | Median age at death |
|---|---|---|---|
| Starved before reaching `max_age` | 39% | 27% to 52% | 37 to 45 s |
| Eaten by a predator | 33% | 22% to 42% | 26 to 46 s |
| Starved after passing `max_age` | 28% | 22% to 35% | 91 to 98 s |

**[Measured]** Energy stayed low throughout:
- The population's mean energy at the 10-second sample points had a median of 77 to 84, against a max energy of 500.
- Agents were at their max energy in only 4 agent-steps across all 5 games.
- **Derived from these numbers:** most agents spend most of the game below 100 energy, which is 20% of the default 500. **Below that level they cannot sprint away from predators.**

### Can reproduction sustainably replace dying agents?

- **With heuristic_v0: no. [Measured]**
  - In all 5 games the population peaked at 35 to 67 agents between 80 and 100 s, drifted down, and went extinct between 485 and 981 s.
  - There were 105 to 330 births per game.
- **[Code]** The energy arithmetic of one birth:
  - The parent needs **more than 100 energy after its move and turn that step**, and pays 100.
  - The child starts with **75 energy**, whatever its max energy. So every birth destroys 25 energy.
  - A child is below the 100-energy (20%) sprint threshold from birth. It needs food before it can flee at a sprint or reproduce.
  - The child appears 10 to 30 units from the parent, or exactly on the parent if that spot overlaps an obstacle. It shows up in the very next response ([Test]).
- **[Unknown]** Whether **any** policy can sustain the population for 3000 s. Nothing we have run shows either answer.

---

## 4. Control

### Can we keep memory between requests?

- **Yes, on our side. [Code]**
  - Each request is stateless, but the endpoint is our own long-running Python process.
  - Nothing in the request/response format or the starter code stops the server from keeping state in memory. Our `submission_server.py` already keeps counters across requests.
- **Detecting a new game. [Code]**, starter `simulation_server.py`:
  - The first request of a game has `sim_time` 0 and an empty agent list.
  - Agent IDs restart at 0 in every game. They are never reused within a game.
- **[Unknown]** Whether the official evaluator behaves exactly like the starter `simulation_server.py`, for example sending that empty first request. We only have the starter version.
- Remember that the final evaluation plays 3 games in a row against the same server, so any memory must be reset between games.

### Can we coordinate across agents?

**Yes. [Code]** One request contains every living agent's full state, and one response returns all their actions. The policy can decide for all agents together.

- **What links agents to each other:** an `Agent` observation includes the other agent's `id`, plus `distance`, `angle` and `rel_dir` ([Test]). So we know which agent sees which, and roughly where they are relative to each other.
- **What is missing:** no absolute positions and no headings. Any shared map would have to be built from relative observations.
  - **[Derived]** Tracking position by adding up our own moves would drift. Terrain slows movement, obstacles deflect it in 10° steps, and none of that is reported back.

### What self-state is supplied?

**[Code]/[Test]** Per agent, every step:

| Field | Meaning |
|---|---|
| `agent_id` | Unique within the game |
| `energy` | Current energy |
| `age` | Seconds alive |
| `biome` | `forest`, `grassland`, `swamp`, `desert` or `river` at the agent's position |
| `speed`, `sprint_speed` | Current, possibly mutated, traits |
| `hearing_radius`, `vision_angle`, `vision_range` | Current traits |
| `max_energy` | Current trait |
| `observations` | List of seen or heard objects (next table) |

**Not supplied:** position, heading, `max_age`, body size, parent ID.

| Observation | Fields | Not included |
|---|---|---|
| `Fruit` | `distance`, `angle` | Energy, age, time until it rots |
| `Tree` | `distance`, `angle` | Age, whether it is old enough to fruit |
| `Agent` | `distance`, `angle`, `rel_dir`, `id` | Its energy |
| `Predator` | `distance`, `angle`, `rel_dir` | Asleep or awake, its energy, an ID |
| `Edge` | `coords`: start and end points, in the agent's own rotated frame | |

The response also carries `score`, `sim_time` and `n_agents` for the whole game.

### Action handling details

- **[Code]** Actions are applied in the order sent. Each agent moves, then turns, then spawns if asked.
- **[Test]** An action for an agent ID that doesn't exist is silently ignored.
- **[Test] Duplicate actions are applied twice.** If the same agent ID appears twice in one response, the starter engine applies both: in the test the agent moved 10 units instead of 5 and paid twice. This looks unintended. **[Unknown]** Whether the official evaluator removes duplicates. Relying on it would be a rules question for the team, not an engineering one.

---

## 5. Observed behaviour: bunching, circling, stuck, abandoning food

All **[Measured]** with heuristic_v0 unless marked otherwise. **Nobody has watched the rendered videos closely.** These are counts, not visual observations.

### What agents spend their time doing (share of agent-steps)

| Rule that fired | Share |
|---|---|
| Standing at a tree, turning slowly | 37% to 44% |
| Walking toward a tree | 25% to 33% |
| Going to a fruit | 15% to 21% |
| Fleeing a predator | 4% to 9% |
| Nothing in sight, wandering | 5% to 9% |

### Bunching

- **Yes, often.** At the 10-second sample points, the median share of agents with another agent within 15 units was 20% to 31%.
- For comparison, 15 units is the predator's kill distance measured from its own centre.
- **[Code] Reasons bunching is built in:**
  - children appear only 10 to 30 units from their parent
  - many agents wait at the same tree

### Circling

- **[Derived from the heuristic code]** When nothing is in sight, an agent walks 5 units forward and turns π/16 every step. That traces a closed circle about 25 units in radius every 3.2 s, so **wandering agents explore almost nothing**.
- **[Measured]** This rule fired in only 5% to 9% of agent-steps.
- Separately, predators spend 29% to 39% of their agent-in-sight steps circling (section 1).

### Getting stuck

- **Rare.** A move of 1 unit or more produced no motion at all in 0 to 114 of 25,357 to 71,992 move requests per game, under 0.2%.
- Partial blocking, where an obstacle deflects an agent sideways, was not measured.

### Abandoning food

- **For predators:** agents fled a predator while a fruit was in view in 417 to 4213 agent-steps per game.
- **Losing sight:** in 7.1% to 8.8% of the steps where an agent was heading for a fruit, its next observation contained no fruit and it had not eaten one. Possible reasons include another agent eating it or the fruit leaving hearing and vision; which one was not measured.

---

## 6. Engine quirks found while answering

1. **[Code]/[Measured] Some agents skip their update.** When an agent starves, the engine removes it from the list it is iterating over. The next agent in the list is then skipped for that step: it doesn't age, doesn't pay living cost, doesn't get a new observation (it keeps the previous one), and doesn't eat. This happened 73 to 237 times per game. The same pattern affects fruit growth and tree updates when a fruit or tree is removed.
2. **[Test] Duplicate actions are applied twice** (section 4).
3. **[Pending] Seed 1 may not be deterministic.** heuristic_v0 on seed 1 has now ended at three different times:

   | Run | Ended at (s) |
   |---|---|
   | Global Python, scored run | 536.6 |
   | Project venv, dry run | 625.6 |
   | Project venv, diagnostics | 980.7 |

   Seeds 0, 2, 3 and 4 gave identical results in every run. The handover blamed the seed 1 difference on numpy versions, but the last two runs used the same environment. A repeat test (3 identical runs of seed 1) is running, and this section will be updated with the result.

---

## 7. What still cannot be answered

- Whether any policy can survive 3000 s, and what food supply looks like after 1000 s. No game has lasted that long.
- Whether the official evaluator matches the starter `simulation_server.py`: first request, duplicate-action handling, how the API key is sent.
- Whether grouping or facing strategies reduce predation. The code shows no built-in protection, but nothing has been tested.
- What the games look like visually. The videos have not been reviewed.

## Where to find the source

On GitHub, branch `challenge-1`, repository `sjoeen/Nordic-AI-cup-2026`, under `Nordic-AI-Cup-2026-main/survival-simulator/`:

| What | Path |
|---|---|
| Simulator | `src/elements/environment.py`, `predator.py`, `creature.py`, `agent.py`, `fruit.py`, `tree.py`, `biome.py` |
| Current heuristic | `agents/heuristic.py`, `training/configs/heuristic_v0.json` |
| Full handover | `HANDOVER.md` |
| This document | `MECHANICS_QA.md` |
| Measurement script | `training/diagnostics.py` |
