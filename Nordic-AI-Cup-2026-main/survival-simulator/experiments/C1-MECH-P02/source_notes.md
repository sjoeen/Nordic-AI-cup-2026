# C1-MECH-P02 — source verification notes

Checkout: worktree at `.../scratchpad/wt-c1`, branch `challenge-1`, HEAD `4ea9ca3`
(`[C1-MECH-P01] findings: predator facing/retreat mechanism study`). `src/` on
`challenge-1` is identical to `main`'s source revision used by P01
(`d9b5d9c`) — no diff was needed; only `training/predator_mechanics.py`
(harness commit `7b6805c`) had to be copied over, since `challenge-1` never
had it. Interpreter/deps verified in this environment: CPython 3.13.7,
numpy 2.2.5, scipy 1.16.2, shapely 2.1.2, pygame 2.6.1 (SDL 2.28.4),
pydantic 2.12.5, matplotlib 3.10.6 — matches the P02 task-prompt environment
statement and P01's record.

Local-testing constraints checked: `README.md` only states "Python 3.10 or
newer" and recommends a venv (`README.md:124`); no separate local-testing
rules file or official-endpoint documentation was found in this checkout.
No official/evaluator endpoints are used by this harness.

All line numbers below are from the files as they exist in this checkout
(`challenge-1`, same content as `main@d9b5d9c`).

## `src/elements/predator.py`

- `Predator.__init__` (L10–14): `speed=11`, `sprint_speed=15`, `max_energy=200`
  defaults; `hearing_radius=60`, `vision_radius=250` are set explicitly here
  (L13–14), overriding the `Creature` defaults. `cone_angle` is **not**
  overridden, so the predator inherits `Creature`'s default `cone_angle=np.pi/3`
  (±30°) — confirms the "±30-degree cone" claim; the 250-unit vision figure is
  explicit, the cone width is inherited, not restated.
- `Predator.step` spans L16–100 (101-line file), matching the spec's claim of
  "lines 16–100".
  - Chase predicate, L37: `abs(agent_looking_dir) > np.pi*1/2 or distance_to_agent < self.hearing_radius * 1.5`.
    `agent_looking_dir` is the observation's `rel_dir` field (how the *agent*
    is facing relative to the bearing agent→predator; supplied by
    `Creature.observe`, see below), not the predator's own facing. With
    `hearing_radius=60`, the distance branch is `< 90`. This matches the
    spec's "chase if distance <90 or target facing angle >90 degrees" but the
    90 is `hearing_radius*1.5`, not a hardcoded literal — it moves if a
    mutated predator's hearing radius ever changes (out of scope here, fixed
    predator defaults are used).
  - Near-zero-angle chase, L39–43: only when `abs(angle_to_agent) > 0.05` does
    the signal dict include a `turn` key (L40); at `|angle| <= 0.05` only
    `move` is signalled (L43) — confirms "chase signal with |angle|<=0.05 has
    no 'turn' key".
  - Circle branch, L46–62: `pivot_sign = -np.sign(agent_looking_dir)` (L48).
    Exactly-zero `agent_looking_dir` (`np.sign(0.0) == 0.0`) makes
    `pivot_sign == 0.0`, collapsing the ±45° pivot to 0 — confirmed. Move
    direction `move_dir = angle_to_agent + pivot_sign*pi/4` (L49); the
    predicted post-move bearing recomputed via `arctan2` (L52–59) becomes the
    `turn` signal (L61). Circle always emits both `move` and `turn`.
  - Edge-avoidance, L65–95: only reached when `agents` is empty; steers away
    from the closest boundary point at `speed=11` (not sprint), both `turn`
    and `move` keys always present.
  - Default wander, L97–99: `turn(self.rng.uniform(-0.1,0.1))`,
    `move(self.speed, None)` — direction `None` means "move forward" (the
    environment substitutes `entity.direction`, see below). Always emits both
    `move` and `turn`.

## `src/elements/environment.py`

- `update_entity_position` (L496–557): distance clamped to
  `[0, entity.sprint_speed]` (L505–510); **then** if
  `entity.energy < entity.max_energy/5` and `distance > entity.speed`, clamp
  to `entity.speed` (L512–513) — the 20%-of-max-energy sprint-disable clamp.
  Cost: walking (`distance<=entity.speed`) costs `distance*0.05` (L515–516);
  sprinting costs `entity.speed*0.05 + (distance-entity.speed)*0.5` (L517–518).
  Direction semantics (L521–524): if `direction is None`, movement uses
  `entity.direction` unchanged; otherwise the *passed* `direction` is added to
  `entity.direction` (`direction = entity.direction + direction`) — i.e. the
  `direction` argument to `move()`/`update_entity_position` is a **relative**
  offset from current heading, not an absolute world angle. This underlies
  the controller's `move_direction = wrap(ESCAPE - heading_acc)` trick (adding
  it to the tracked current heading reproduces the fixed world escape line
  without querying engine truth).
- `update_entity_direction` (L559–564): `entity.direction += turn_angle`
  (absolute increment, unconditional); cost
  `min(pi, abs(turn_angle))/(2*pi)` (capped at 180°).
- `agent_step` (L602–623): calls `update_entity_position` (L614) **before**
  `update_entity_direction` (L618) — move-before-turn confirmed for agents.
- `non_agent_step` (L627–762):
  - Agent block (L635–672) runs first each tick: age/energy drain, death
    check (`energy<=0` → `kill_agent`, L642–644), then **captures this
    tick's observation** (`agent.observe(...)`, L652–659) using the spatial
    grid as it stood at the *start* of this call — i.e. before the predator
    block below moves the predator this tick. Since `agent_step` (the
    controller's move/turn) already ran earlier in `step_environment`
    (L43–79 of `src/utils/simulation.py`) for this same tick, the observation
    stored here is used by the *controller on the next tick*, and reflects
    predator geometry from **before** this tick's predator move — confirms
    the "stale observation" claim.
  - Predator block (L675–728): if `resting`, wakes when
    `energy > max_energy*0.5` (L677) — 100 only because default
    `max_energy=200`; not a hardcoded "100" — else `+= dt*30` and `continue`
    (L680–681, no action while asleep, confirms "never move/kill while
    asleep"). Otherwise: build local observation ignoring fruits/trees/
    obstacles (agents+edges only, L688–694); `predator.step(observation)`
    (L696); handle `move` (position, L699–702) **before** `turn` (direction,
    L705–707) — move-before-turn confirmed for predators too; kill check
    (L710–724): `touching = distances < size_sum` where
    `size_sum = predator.size(10) + agent.size(5) = 15` for default sizes
    (L717–718) — confirms "post-move kill distance <15" is `size_sum`, not a
    hardcoded constant; energy replenishment
    `predator.energy = min(predator.max_energy, predator.energy + agent.energy)`
    (L722) runs **before** `self.kill_agent(agent)` (L724) — so at the moment
    `kill_agent` is invoked, `predator.energy` already reflects the eaten
    agent's energy (capped at `max_energy`); sleep transition
    (`if predator.energy<=0: predator.resting=True`, L726–727) is checked
    **after** the kill block — confirms "kill-before-sleep ordering" and
    pinpoints exactly where pre-kill vs post-eat energy must be captured (see
    harness section below). Predators are never removed anywhere in this
    file — confirmed "predators never die".

## `src/utils/simulation.py`

- `step_environment` (L43–79): applies each queued `(agent_id, action)` via
  `env.agent_step(...)` (L56–65), **then** calls `env.non_agent_step(dt)`
  (L68) once, unconditionally — calling it with an empty `actions` list still
  advances predators/energy/fruit exactly as in a normal tick, which is what
  the post-death continuation phase relies on (no code path skips the
  predator/environment update when there are zero agents).

## `src/utils/sensing.py`

- `compute_visibility` (L1–117): `corners = np.array([pt for edge in edges for
  pt in edge])` (L24) runs **before** the `edges_arr.size == 0` early return
  (L59–65). If the `edges` argument passed in is an empty list, `corners` is
  `np.array([])`, a 1-D array of shape `(0,)`; `dx = corners[:, 0] - x`
  (L27) then raises `IndexError: too many indices for array` — confirms the
  crash. The workaround (enlarging `chunk_size` so the four boundary walls'
  edges are always members of `_get_local_edges`/`_get_local_objects`, never
  literally empty) sidesteps this without touching `sensing.py`.
  - Separately, even when `edges` is non-empty but far away: corners beyond
    `vision_radius` are masked out of the *extra-ray* generation (L30–33),
    but the 5 fixed rays (L45–51) are still tested against every edge in
    `edges_arr` (L67–108). An edge only becomes a `hit_edge` (and thus only
    appears in `Creature.observe`'s output) if its true ray intersection
    distance `t < vision_radius` (L107) — i.e. distance alone, not the
    chunk/candidate-filtering step, is what keeps far boundary edges out of
    observations. This is the property the P02 arena enlargement relies on
    and that must be checked empirically (see `smoke.json`).

## `src/elements/creature.py`

- `observe` (L94–179): `nearby_mask = distances <= hearing_radius` (L131);
  `visible_mask = (~nearby)*(distances<=vision_radius)*(|angle|<=cone/2)`, then
  additionally must be inside the shapely vision polygon (L132, L146–160).
  Predators and agents include `rel_dir` (the *observed* object's own facing
  relative to the bearing back to the observer) only when
  `include_direction=True`, which both `process_objects(agents,...)` (agent
  observing predator — wait, opposite: agents observing predators, and
  predators observing agents, both pass `include_direction=True`, L166/168).
  `angle` (bearing) is always `arctan2(dy,dx) - self.direction`, wrapped to
  `[-pi,pi)` (L90–91) — this is exactly the controller's `beta`.
- `move`/`turn` (L45–51) are pure signal-dict constructors; no state
  mutation, no RNG use — confirms the logger-noninterference requirement is
  about the *environment* wrappers, not these helpers.

## `src/elements/agent.py`

- Defaults (L10–27): `speed=10`, `sprint_speed=20`, `max_energy=500`,
  `hearing_radius=50`, `vision_radius=200`, `cone_angle=pi/3` — all match the
  spec's stated agent perception numbers. `max_age` defaults to
  `60 + U(0,60)`, but both P01 and P02 harnesses pass `max_age=120.0`
  explicitly, per the spec's fixture definition, so the random default is not
  in play here.

## Harness-level deviations from a full game (unchanged from P01, applies to P02 too)

- `Environment.__new__` bypass skips `Map_generator`, pygame surfaces, and
  random biome placement (rendering/world-gen only); `biome_map` is a
  constant-lookup object always returning one `Grassland_biome()`
  (`move_penalty=1.0`, `energy_drain_rate=1.0` — verified by inspection of
  `src/elements/biome.py`'s `Grassland_biome`, not modified for P02).
- `spawn_tree`/`spawn_fruit`/`spawn_fruit_around_tree`/`spawn_predator` are
  instance-level no-ops; `non_agent_step`'s `rng.random()` gates for those
  events still execute and still consume RNG state exactly as in a real game
  (only the *effect* of a triggered spawn is suppressed).
- Boundary obstacles/edges are constructed with the same tuples
  `_create_boundaries`/`spawn_obstacle` would produce (`environment.py`
  L268–276), just without the pygame draw calls.

## New for P02: pre-kill vs post-eat predator energy

P01's wrappers (`env.update_entity_position`, `env.update_entity_direction`,
`predator.step`) do not observe the kill/eat step itself
(`environment.py` L710–724), which runs *between* the direction wrapper
returning and the tick ending. P02's harness adds a fourth instance-level
wrapper, `env.kill_agent`, installed only in `mode="full"`. Because
`predator.energy = min(max_energy, predator.energy+agent.energy)` (L722)
executes immediately before `self.kill_agent(agent)` is called (L724), the
`kill_agent` wrapper observes `pred.energy` already reflecting the eaten
agent — this is captured as `post_kill_energy`. The corresponding pre-kill
value is the predator's energy immediately after its own move/turn cost this
tick and before the kill loop, captured as the last of
`post_move_energy`/`post_turn_energy` recorded by the (also wrapped)
position/direction calls for the predator this tick (turn always follows
move when both are signalled, and the near-zero-angle chase case and the
default-wander/edge-avoid cases always signal `move`, so a "last predator
energy snapshot before kill" is always available whenever the predator
acted this tick).
