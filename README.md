# Eat-Rest v1 (challenge 1 survival simulation)

## Achievements

Eat-Rest v1 achieved a **top 10 score overall** and ranked **2nd in Norway** in the **Nordic AI Cup**. The competition included **112 teams overall**, with a Discord community of approximately **300 members**.

| Achievement | Placement |
| --- | --- |
| Overall score | **Top 10** |
| Norwegian ranking | **2nd** |

A stateful, rule-based controller for a multi-agent survival simulation. Each tick, it combines the agents' public observations, assigns fruit targets, chooses movement, and decides which agents should reproduce.

The central idea is to **eat until sufficiently supplied, rest while that reserve is useful, and resume foraging before it runs too low**. Predator evasion can interrupt either activity. Reproduction maintains a replacement generation, while target ownership and newborn dispersal reduce competition for the same food.

This document describes the original Eat-Rest v1 configuration implemented by `AppetitePolicy` in `survival_agent.py`.

## Contents

- [Running the controller](#running-the-controller)
- [Information and memory](#information-and-memory)
- [Decision order](#decision-order)
- [Eating and resting](#eating-and-resting)
- [Fruit tracking and allocation](#fruit-tracking-and-allocation)
- [Short fruit-route planning](#short-fruit-route-planning)
- [Predator detection and escape](#predator-detection-and-escape)
- [Searching, dispersal, and walls](#searching-dispersal-and-walls)
- [Reproduction and retirement](#reproduction-and-retirement)
- [Configuration reference](#configuration-reference)
- [Implementation structure](#implementation-structure)
- [Assumptions and tradeoffs](#assumptions-and-tradeoffs)

## Running the controller

The submission is a standalone Python file exposing a FastAPI application. Its external dependencies are FastAPI, Pydantic v2, and Uvicorn; the remaining imports are from the Python standard library.

```bash
python -m pip install fastapi 'pydantic>=2' uvicorn
python survival_agent.py
```

The script listens on port `9052`. Alternatively, choose the listening address and port explicitly:

```bash
python -m uvicorn survival_agent:app --host 0.0.0.0 --port 9052
```

| Endpoint | Purpose |
| --- | --- |
| `GET /` | Reports service status and the strategy name, `appetite`. |
| `POST /predict` | Accepts a simulation step and returns an `actions` array. |

Each action contains `agent_id`, `move_distance`, `move_direction`, `turn_angle`, and `spawn_agent`. Angles are in radians. Movement direction is relative to the agent's observation frame. Movement distance is a requested distance for the current tick, not a speed in world units per second.

The request contains `game_status`, `score`, `agent_status`, and optionally `sim_time` and `n_agents`. The controller uses the actual agent list for population decisions. It does not use the supplied score to choose behavior.

The service retains memory between requests. A lock serializes access to that memory, and an identical repeated request receives the cached response without advancing policy state. Keep consecutive steps of a game on the same controller process; independent games should use separate instances.

## Information and memory

The policy reads each agent's public energy, age, biome, movement and sensing traits, and relative observations of fruit, trees, predators, teammates, and edges. It imports no simulator internals and does not read hidden world coordinates, exact fruit energy, or an agent's hidden maximum age.

Per-agent memory includes:

- A persistent fruit target and locally generated fruit identities.
- Previous fruit, predator, and static-landmark positions.
- Previous movement and turn requests.
- Estimated heading and a persistent exploration heading.
- Eating/resting state and newborn dispersal state.
- A private pseudorandom generator seeded by agent ID.

Species memory includes merged fruit identities, inferred fruit birth times, selected parents, retired agents, and a simulation clock.

### Estimating movement

The previous movement request is an initial estimate of translation. The policy corrects it by matching stationary fruit, trees, and edge endpoints between observations. Two or more close landmark matches support a median translation estimate. A separate collision-recovery path looks for a translation supported by at least three landmarks when normal matching fails.

When an agent's reported age repeats, `stale_correction=True` treats its observations as potentially stale and adjusts their bearings using accumulated movement and turns. These are observation-based estimates; they are not absolute position measurements.

### Time and resets

The supplied simulation time is preferred. If it is missing, or is serialized as zero while agents are already aging, the policy advances its own clock by `0.1` per request. That fallback assumes the normal tick interval.

State is reset when the API detects a new game through time rollback or restarted IDs, or when the game status is not `ok` or the agent list is empty. The policy also detects age rollback and removes dead agents from its active memory.

## Decision order

Each tick performs observation processing and species-level allocation before returning individual actions:

1. Update memory, estimate movement, and identify threats.
2. Calculate the desired young generation, retirement, and normal birth requests.
3. Share current fruit observations, update eating/resting decisions, reserve fruit targets, and refine those targets with short route planning.
4. Choose each agent's movement using the priorities below.
5. Apply the emergency reproduction check to the completed actions.

| Priority | Condition | Action |
| --- | --- | --- |
| 1 | An active predator threat exists | Use the final `CROWD_ESCAPE` controller. |
| 2 | The agent is still dispersing and is not escaping | Follow its migration heading at walking distance. |
| 3 | The agent is resting, including retirement | Request no translation and normally scan. |
| 4 | An assigned fruit target exists | Walk toward that fruit, shortening the step near it. |
| 5 | None of the above | Explore with a persistent heading and reduced movement distance. |

Reproduction is a flag on an action, not a separate movement mode. A parent can move and request a birth in the same tick. An emergency birth is added after movement selection, so it does not recompute that tick's rest or movement decision.

## Eating and resting

The appetite rule applies to agents that are neither retired nor dispersing. It uses the agent's actual maximum energy:

```python
level = min(max(130, 250 + 0 * max(0, age - 50)), 0.87 * max_energy)
```

The zero is `appetite_age_slope`: v1 does not raise the appetite target with age. For agents with enough energy capacity, `level` is 250; smaller-capacity agents use the capacity-based limit.

### Starting a rest

An active agent rests when its energy is **strictly above** `level`, unless it has already been selected for a normal birth. Resting agents are excluded from normal fruit targeting.

### Staying at rest

If the agent was already resting and there is no nearby fruit, its threshold becomes:

```python
threshold = level - 70
```

It continues resting while energy is strictly above that threshold. With a level of 250, this means entering rest above 250 and resuming activity at 180 or below.

This difference between the entry and exit thresholds is **hysteresis**: it prevents small energy changes from repeatedly starting and stopping a food trip.

### Nearby fruit exception

A currently known fruit strictly between 10 and 80 units away removes the 70-energy resting discount. The threshold returns to `level`, allowing a resting agent below that level to pursue a nearby opportunity.

The nearby-fruit check includes shared observations. It happens before exclusive target allocation, so a nearby fruit can end rest even if another agent ultimately receives that fruit. It also does not force an agent above `level` to eat.

### What resting does

Normal rest requests zero movement and a `0.2`-radian scan. A nearby observed predator can change the gaze direction, and an active threat overrides rest entirely. Rest therefore still has living and turning costs.

There is no explicit eat command. The policy moves into collection range and lets the simulator handle consumption.

## Fruit tracking and allocation

### Tracking fruit without engine IDs

Fruit observations farther than six units are converted from relative bearings to Cartesian positions. Previous fruit tracks are transformed into the new frame and matched within two units. Unmatched fruit receive a local identity `(agent_id, sequence_number)`.

Targets remain associated with tracked fruit rather than being rebuilt from an average attraction vector every tick.

### Sharing observations

Observed teammates provide relative positions and orientations. A graph traversal constructs a shared coordinate frame for each connected group of agents. Fruit observations within one unit in the same reconstructed frame are merged into a common identity.

Current fruit observations are then shared across that connected group, including fruit outside an individual recipient's own view. Disconnected groups cannot infer each other's fruit positions. The policy does not maintain a persistent global food map.

### Assigning targets

Allocation first preserves valid existing targets. If multiple agents claim the same merged fruit identity, the closer agent keeps it. Remaining agent–fruit pairs are considered in increasing distance order, assigning at most one fruit per agent and one owner per fruit identity.

A target is released when the agent is escaping, is ineligible for food, or the fruit is no longer currently known. Ineligibility includes normal appetite rest and retirement. Consumption generally removes the target through subsequent observations.

The route planner can subsequently replace a target with a more valuable route. Persistence is therefore a preference rather than an unconditional lock.

## Short fruit-route planning

The planner processes eligible, non-escaping agents from lowest to highest energy. It considers up to 12 nearby fruit that are not owned by another agent, and evaluates the first fruit plus one possible follow-up fruit.

### Travel cost estimate

Let `walk = max(0.1, min(speed, sprint_speed))`. The estimated cost per world unit is:

```python
aging = clamp((age - 60) / 60, 0, 1) * 0.1 * age
unit_cost = (0.05 + (1.32 + aging) * 0.1 / walk) / terrain
travel_cost = max(0, fruit_distance - 10) * unit_cost
```

Terrain is `0.3` for river, `0.5` for swamp, `0.8` for desert, and `1.0` otherwise. The aging term is an estimate based on public age, not knowledge of the individual aging threshold.

A first-leg candidate must have a clear straight segment through the observed edges and satisfy `travel_cost + 1 < energy`.

**Scope of that check:** it filters route-planner candidates. If no candidate survives, the implementation retains the target from initial allocation. It is not a universal guarantee that every pursued fruit is affordable or unobstructed.

### Fruit value and route utility

Fruit value defaults to 42 because exact energy is not observed. When a fruit appears well inside the previous hearing area where it was absent, the controller records an inferred birth time. Its estimated value then becomes:

```python
estimated_value = min(60, 20 + 2 * inferred_age)
```

The first fruit contributes its estimated value minus travel cost. A second fruit may add a discounted positive contribution if it is less than 180 units from the first and their connecting segment is clear:

```text
route utility = first fruit value − first travel cost
              + 0.7 × max(0, second fruit value − second travel cost)
              + 7 if the first fruit is the existing target
```

The follow-up is a planning bonus, not a reservation of a second fruit or a commitment to finish a route. The policy replans next tick. Equal utilities prefer the shorter first leg, with fruit identity providing the final ordering.

### Approaching the target

The agent moves directly toward its assigned fruit, applying wall sliding if needed. Requested distance is capped at:

```python
min(walk, max(0, fruit_distance - 6) / terrain)
```

This shortens the final approach instead of repeatedly requesting a full step past the fruit. Ordinary food pursuit turns toward the movement direction by at most `0.6` radians per tick, subject to the predator-facing rule.

## Predator detection and escape

### Threat detection

Predator heading comes from its observation. Velocity is estimated by matching successive predator positions after correcting for the agent's own motion. A match is accepted when displacement is at most 18 units. Predators do not have stable observation IDs, so this matching is approximate.

A predator normally triggers escape when it is within 100 units. For a measured predator moving less than `0.5` units per tick, the radius is 45 units instead. A farther predator also triggers escape when estimated closing movement exceeds two units per tick and `distance / closing < 7`.

That last ratio is in estimated movement steps, not seven seconds.

### Shelter behind edges

The controller retains up to 60 edge observations from the previous two seconds, transformed into the current frame. If an edge blocks the direct segment to a predator, it estimates the detour through the edge endpoints. A predator farther than 22 units can be excluded as an escape trigger when this detour measure exceeds 65 units.

This affects threat detection, not the existence of the predator observation. Such predators remain available to other rules, including normal reproduction safety checks and the escape evaluator if another predator triggers escape.

### Final escape controller

With `crowd_escape=True`, the final movement choice is made by `AdaptivePolicy.steer`. It evaluates up to five predators, prioritizing those classified as threats.

Candidate directions comprise:

- Thirteen directions at 15-degree intervals across a 180-degree fan centered on an inverse-distance-weighted direction away from predators.
- The direction proposed by the underlying escape rule.
- The direction toward the nearest known fruit when `hunger_feed=True`.

Every direction is adjusted for walls. Walking distance is always considered. If energy exceeds `0.2 * max_energy + 5` and sprint distance exceeds walking distance, the controller also considers the midpoint between walking and sprinting, and full sprint distance.

For each candidate, the policy projects three movement steps. Measured predator velocity is used when available; otherwise it assumes 15 units per step, adjusted for terrain, along the observed heading. Its calculation includes a one-step predator offset to account for observation timing.

The utility strongly penalizes predicted separation below 25 units, applies an additional penalty below the 45-unit clearance margin, rewards final separation up to 150 units, penalizes movement energy, and can reward progress toward food:

```text
utility = −1000 × max(0, 25 − minimum separation)
          −1.5 × max(0, 45 − minimum separation)
          +0.15 × min(final separation, 150)
          −5 × movement cost
          +0.5 × hunger-weighted food progress
```

The hunger weight is `min(1, 75 / max(50, energy))`. The highest-utility candidate supplies movement distance and direction, while gaze turns toward the first prioritized predator.

The movement-cost model is:

```python
0.05 * min(distance, speed) + 0.5 * max(0, distance - speed)
```

Sprinting is therefore a choice in the search, not an automatic response to every threat. Food progress can influence evasion, but it does not bypass the clearance penalties. The predictions are short approximations, not guaranteed safe trajectories.

## Searching, dispersal, and walls

### Searching

An active agent without a fruit target searches at `0.4 * walk`, normally scanning by `0.2` radians per tick. Its exploration heading persists and receives a small random perturbation between `−0.015` and `+0.015` radians each tick.

If fruit is known but no target is available and a teammate is observed, search is directed away from the nearest teammate. This helps separate agents competing for the same local food. The controller remembers the movement heading after wall deflection, reducing repeated attempts to walk into the same edge.

### Facing predators

With `face_predators=True`, an observed predator within 180 units can override the normal scan or food-facing turn when there is no target, or the target is inside hearing range. The turn is bounded by `gaze_turn=π`. The final crowd-escape and dispersal methods have their own turn behavior.

### Newborn dispersal

An agent first observed below age `0.5` and below 100 energy starts as a disperser. It receives a random migration heading and moves at walking distance to leave its birth area.

Dispersal ends when any condition holds:

- Age exceeds six seconds.
- Energy falls below 40.
- Estimated distance from the birth location exceeds 100 units.

An active threat takes priority over migration. Otherwise, if a predator is within 160 units and the migration direction points sufficiently toward it, the migration heading is redirected away. Wall sliding still applies, and the migration turn is limited to `±0.6` radians.

Dispersal movement overrides ordinary rest and food pursuit. Because allocation is layered, a disperser can still acquire a fruit reservation in lower-level allocation; having a target does not terminate its migration early.

### Wall sliding

For each observed edge, the controller compares clearance with the planned movement toward it. If that step approaches the edge too closely, movement is redirected along the edge with a small outward component.

It chooses the tangent aligned with the desired movement. If that is ambiguous, it uses the previous movement, then agent-ID parity as a deterministic fallback. The clearance reference is seven units; the outward component is `0.2` inside that clearance and `0.04` otherwise.

Terrain-adjusted movement is also used in prediction and memory: river multiplies translation by `0.3`, swamp by `0.5`, desert by `0.8`, and other biomes by `1.0`.

## Reproduction and retirement

### Replacement target

The desired number of young agents decreases with simulation time:

```python
target = max(2, round(6 * 0.5 ** (sim_time / 900)))
young = count(agent.age < 65)
birth_budget = max(0, target - young)
```

This uses Python's `round`. It is a target for replacement agents, **not a hard cap on total population**. Older agents remain alive, and additional birth rules can request offspring outside the normal deficit.

### Avoiding a long gap between generations

When total population is below twice the target, the policy examines the youngest agent with energy above 40. If that agent is older than 20 seconds—or no agent meets the energy condition—it permits at least one normal birth slot.

This is a check on the age of the youngest sufficiently supplied agent, not a strict timer since the last birth. It still requires an eligible parent.

### Normal parent eligibility

A normal parent must:

- Have energy strictly above `max(112, min(135, 0.75 * max_energy))`.
- Have no observed predator at or within 130 units.
- Not currently be retired.

Eligible parents are ranked by:

```python
quality = min(speed, sprint_speed) + 0.015 * hearing_radius + 0.003 * vision_range
```

Higher quality is preferred, followed by higher energy and then lower agent ID. The species selects up to the available birth budget. This ranking favors movement and sensing traits; it does not directly reward maximum energy capacity.

No nearby-tree requirement is imposed. Normal spawning uses an energy threshold rather than a complete movement-adjusted budget; the simulator determines whether the request can actually execute.

In the standard simulator, a successful birth deducts 100 energy from the parent and a newborn starts with 75. The nominal 135 threshold leaves a 35-energy buffer before other costs; the 112 floor permits a smaller buffer for agents with lower capacity. Reproduction therefore trades stored energy for a replacement agent, rather than creating energy for the species.

### Emergency reproduction

After movement selection, an agent without an existing birth request can request one if:

- No raw observed predator is at or within 55 units.
- Energy is strictly above `112 + estimated movement cost + abs(turn_angle)/(2π)`.
- Either total population is below three, or the parent is older than 75 and fewer than two agents are younger than 75.

This pass uses its own young-age threshold of 75, rather than the normal 65. Each added emergency birth increments its internal young count. The low-total-population condition uses the current population throughout the pass, so it may allow several parents to spawn on the same tick.

Emergency births do not require a tree, food target, normal birth slot, or non-retired status. They can accompany an escape action when the separate 55-unit safety test permits it. Their energy check includes the selected movement and turn; the movement itself is retained.

The fixed 112 in the emergency condition is written directly in that method. It is not automatically changed by editing the configurable normal `birth_floor`.

### Retirement

An agent older than 105 is retired when the young count is at least `max(2, target // 2)`. Retirement removes it from normal reproduction and active food allocation and makes it rest when not escaping.

Retirement is recalculated each tick. If the younger generation shrinks, an older agent can become active again. It does not kill or remove the agent, and it does not prevent the separate emergency birth rule from firing.

## Configuration reference

The factory selects the appetite policy with these explicit overrides:

```python
CONFIG = {"appetite_release": 70.0, "appetite_near": 80.0}

def make_policy():
    return AppetitePolicy("appetite", **CONFIG)
```

Other values come from the inherited constructors. The tables below describe their effective role in this configuration.

### Active behavior settings

| Group | Setting | Value | Meaning |
| --- | --- | --- | --- |
| Appetite | `appetite_level` | 250 | Nominal energy at which active agents begin resting. |
| Appetite | `appetite_release` | 70 | Reduction in the threshold while already resting. |
| Appetite | `appetite_near` | 80 | Nearby fruit removes that reduction; lower distance bound is 10. |
| Appetite | `idle_energy` | 0.87 | Caps the appetite level as a fraction of maximum energy. |
| Appetite | `appetite_age_slope` | 0 | No age-dependent increase in appetite. |
| Appetite | `appetite_start` | 0 | Appetite rules apply from the start. |
| Harvest | `harvest_depth` | 2 | Evaluate the first fruit plus one follow-up. |
| Harvest | `harvest_discount` | 0.7 | Discount the follow-up contribution. |
| Harvest | `harvest_switch` | 7 | Utility bonus for retaining the current target. |
| Harvest | `harvest_range` | 180 | Maximum gap considered for a follow-up fruit. |
| Harvest | `harvest_food_value` | 42 | Default estimated fruit value. |
| Harvest | `harvest_start` | 0 | Route planning applies from the start. |
| Search | `search_speed` | 0.4 | Fraction of walking distance requested during search. |
| Search | `scan` | 0.2 | Normal scan turn during search and rest. |
| Threats | `escape_radius` | 100 | Normal distance trigger. |
| Threats | `rest_radius` | 45 | Distance trigger for measured near-stationary predators. |
| Threats | `blocked_threat_distance` | 65 | Shelter detour threshold. |
| Threats | `face_predators` | True | Allow gaze to follow nearby predators. |
| Threats | `gaze_turn` | π | Turn limit for the underlying predator-facing rule. |
| Escape | `crowd_escape` | True | Enable the final multi-predator candidate search. |
| Escape | `crowd_margin` | 45 | Additional clearance penalty threshold. |
| Escape | `hunger_feed` | True | Include food-directed options and food progress in escape. |
| Escape | `sprint_ceiling` | None | No additional global sprint-distance cap. |
| Dispersal | `dispersal_distance` | 100 | Distance from birth origin that ends migration. |
| Dispersal | `dispersal_time` | 6 | Age threshold that ends migration. |
| Memory | `stale_correction` | True | Correct bearings when age indicates stale observations. |
| Reproduction | `birth_energy` | 135 | Normal parent energy threshold before capacity adjustment. |
| Reproduction | `birth_floor` | 112 | Lower bound on that threshold. |
| Reproduction | `population` | 6 | Initial young-generation target. |
| Reproduction | `population_floor` | 2 | Minimum young-generation target. |
| Reproduction | `population_decay` | 900 | Time constant for halving the target before rounding. |
| Reproduction | `young_age` | 65 | Age cutoff for normal replacement counting. |
| Reproduction | `maximum_birth_gap` | 20 | Age-of-youngest check that can create a birth slot. |
| Reproduction | `emergency_birth` | True | Enable the final emergency birth pass. |
| Reproduction | `emergency_birth_distance` | 55 | Predator exclusion distance for emergency births. |
| Reproduction | `renewal_age` | 75 | Parent-age and young-count threshold in that pass. |
| Retirement | `retirement` | 105 | Retirement age when replacements are present. |

Several constants are written directly in the methods rather than exposed as settings: food approach distance, matching tolerances, ordinary birth safety distance, emergency energy reserve, candidate counts, and the escape utility coefficients. Their values are described in the relevant rules above.

### Inherited settings with limited or inactive roles

| Settings | v1 values | Effective role |
| --- | --- | --- |
| `escape` | `direct` | Supplies the underlying away-direction candidate; final escape movement comes from crowd escape. |
| `sprint_radius`, `emergency_fraction` | 55, 0.4 | Belong to the underlying escape-speed rule. Crowd escape replaces the final speed; these are not its sprint eligibility rule. |
| `hard_idle_energy` | 250 | Used by an earlier rest calculation. Appetite logic replaces that calculation for ordinary active agents. |
| `ripen_seconds`, `fresh_ripen`, `patch_wait` | 0, 0, 0 | Explicit fruit-ripening waits and tree-patch waiting are disabled. |
| `reserve`, `ripen_age_limit` | 80, 65 | Used by those disabled waiting branches. |
| `prediction_horizon`, `prediction_angles` | 4, 6 | Used by the separate `escape='predict'` implementation, not the active crowd-escape search. |
| `threat_memory`, `threat_cooldown` | 0, 0 | No persistent unseen-threat behavior or positive birth cooldown. |
| `boundary_bias`, `search_reorient` | False, 0 | No extra boundary bias or periodic large exploration turns. |
| `generation_mode` | `baseline` | Use the young-count deficit, not a weighted generation model. |
| `generation_fraction`, `generation_lifetime` | 0.65, 90 | Parameters of the inactive linear generation model. |
| `birth_spacing`, `elder_birth_priority` | 0, False | No extra normal-birth spacing or elder-first parent ranking. |
| `selective_breeding`, `quality_mode` | 0, `basic` | No alternative fitness-based threshold adjustment. Effective parent ranking uses the explicit movement-and-sensing formula. |
| `fertile_population`, `fitness_population` | False, False | Young counts are not filtered or weighted by those optional mechanisms. |
| `weighted_allocation`, `young_energy_weight` | False, 0 | Present in configuration but not read by an active decision path in this implementation. |

The older `SpeciesFields.steer` also contains attraction/separation vectors, tangential evasion, and age-based movement scaling. `SurvivalPolicy.steer` replaces that method; those older movement rules should not be mistaken for the final v1 behavior. The active policy retains tracking, shared allocation, and wall sliding from that base class.

The normal reproduction code also lowers the threshold to at most 160 for parents older than 60 when young agents are scarce. With v1's threshold already at most 135, that branch does not lower it further.

## Implementation structure

The active class chain is:

```text
AppetitePolicy → HarvestPolicy → GenerationPolicy → DispersalPolicy
              → RefugePolicy → AdaptivePolicy → SurvivalPolicy → SpeciesFields
```

| Component | Responsibility |
| --- | --- |
| `SpeciesFields.observe` | Local tracks, motion estimation, coordinate conversion, and stale-observation correction. |
| `SpeciesFields.shared_frames` | Teammate frame alignment and sharing current fruit observations. |
| `SpeciesFields.allocate` | Persistent target ownership and greedy matching. |
| `SpeciesFields.slide` | Movement along walls. |
| `SurvivalPolicy.observe` | Fruit birth inference and initial threat processing. |
| `SurvivalPolicy.decide_all` | Tick orchestration, initial population logic, and retirement. |
| `SurvivalPolicy.steer` | Ordinary rest, food pursuit, search, and underlying escape action. |
| `AdaptivePolicy.steer` | Final crowd-escape direction and speed. |
| `AdaptivePolicy.decide_all` | Emergency reproduction after movement selection. |
| `RefugePolicy.observe` | Shelter-aware final threat classification. |
| `DispersalPolicy` | Birth-origin tracking and migration movement. |
| `GenerationPolicy.allocate` | Effective normal birth budget, youngest-agent gap check, and parent selection. |
| `HarvestPolicy.allocate` | Short fruit-route evaluation after initial assignment. |
| `AppetitePolicy.shared_frames` | Final appetite thresholds and rest hysteresis after fruit sharing. |

Method names alone do not describe the entire order: for example, `SpeciesFields.allocate` calls the overridden `shared_frames`, which is where the final appetite decision is applied. Changing an earlier inherited default may have no effect if a later layer replaces its decision.

## Assumptions and tradeoffs

- **Rest preserves energy but reduces exploration.** The release threshold keeps a reserve while nearby fruit can justify resuming activity sooner.
- **Target persistence reduces switching but is not absolute.** Route utility can justify changing targets, and escaping immediately releases ownership.
- **Shared allocation depends on geometric estimates.** Frame reconstruction, fruit identity matching, and predator matching can be imperfect.
- **Food value and aging are estimates.** Neither exact fruit energy nor individual maximum age is available to the controller.
- **The second fruit is an opportunity, not a guaranteed meal.** Other agents, predators, and subsequent observations can change the route.
- **Predator absence means absence from observations.** A safety check does not establish that unseen space is safe.
- **Replacement targets permit overlapping generations.** Normal budgets, gap births, emergency births, and retirement have distinct roles and thresholds.
- **Dispersal spends energy to separate newborns.** Its time, distance, and low-energy exits bound that commitment.
- **Changes to movement affect future observations and encounters.** The implementation deliberately records actual chosen movement for the next tick's estimates.
- **Policy memory is per game.** Separate processes or explicit resets are required to avoid mixing independent game histories.
