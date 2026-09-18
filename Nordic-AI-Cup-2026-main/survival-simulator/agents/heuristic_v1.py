"""
heuristic_v1 = heuristic_v0 plus three rules (tree cap, straight-line LEAVING travel,
birth conditions). Everything not touched by those rules is v0's original behaviour,
unmodified (predator flee, fruit-seeking, tree-approach). See agents/heuristic.py.

Per-agent server-side memory is kept on the instance, keyed by agent_id:
    {"mode": "WAITING" | "LEAVING", "leaving_steps": int}
IDs that disappear (agent died) are dropped every act_batch call.

"Agents at a tree" (used by Rule 1's cap check, Rule 2's re-entry tree choice, and
Rule 3's birth check) = the observing agent itself, plus every other OBSERVED agent
(agents carry an "id" field in their observation, per src/elements/creature.py) whose
distance to that tree -- computed purely from this agent's own relative observations via
the law of cosines, no absolute position needed -- is <= tree_radius, excluding agents
whose mode (as of the start of this step, before any agent's decision this step) is
LEAVING. Other agents' energy is not observable, so ties/comparisons use the batch's own
self-reported energy, looked up by id (the "server's knowledge of all agents' energies").

Engine facts this relies on (see agents/heuristic.py's docstring, same facts): angle
observations and move_direction are relative to current heading; movement is applied
before the turn within a step. Edge observations carry ("Edge", coords=((x1,y1),(x2,y2)))
in the agent's own forward-facing local frame (+x = current heading) -- both obstacles
and the arena boundary appear this way (both are built as Obstacles, see
src/elements/environment.py:_create_boundaries), so a segment crossing the forward axis
within the next step's walking distance means "blocked".
"""
import math

from agents.heuristic import HeuristicAgent, _wrap


def _crosses_forward(x1: float, y1: float, x2: float, y2: float, lookahead: float) -> bool:
    """True if the segment (in the agent's local forward-facing frame) crosses the
    forward ray (+x axis) within [0, lookahead]."""
    if y1 == y2:
        if y1 != 0:
            return False
        xmin, xmax = sorted((x1, x2))
        return xmax >= 0 and xmin <= lookahead
    t = -y1 / (y2 - y1)
    if t < 0 or t > 1:
        return False
    x_cross = x1 + t * (x2 - x1)
    return 0 <= x_cross <= lookahead


class HeuristicAgentV1:
    def __init__(self, wander_speed_frac: float, scan_turn: float, tree_wait_dist: float,
                 flee_sprint_dist: float, face_predator: bool, tree_radius: float, tree_cap: int,
                 tree_join_threshold: int, leaving_ignore_steps: int, leaving_turn_angle: float,
                 birth_energy_threshold: float, birth_max_others_at_tree: int,
                 birth_no_tree_radius: float, birth_max_others_no_tree: int):
        self.wander_speed_frac = wander_speed_frac
        self.scan_turn = scan_turn
        self.tree_wait_dist = tree_wait_dist
        self.flee_sprint_dist = flee_sprint_dist
        self.face_predator = face_predator
        self.tree_radius = tree_radius
        self.tree_cap = tree_cap
        self.tree_join_threshold = tree_join_threshold
        self.leaving_ignore_steps = leaving_ignore_steps
        self.leaving_turn_angle = leaving_turn_angle
        self.birth_energy_threshold = birth_energy_threshold
        self.birth_max_others_at_tree = birth_max_others_at_tree
        self.birth_no_tree_radius = birth_no_tree_radius
        self.birth_max_others_no_tree = birth_max_others_no_tree

        self.memory = {}  # agent_id -> {"mode": ..., "leaving_steps": ...}
        self._mode_snapshot = {}
        # Exception fallback: the real v0 policy, v0's own default params (training/configs/heuristic_v0.json).
        self._fallback = HeuristicAgent(
            spawn_energy_threshold=175.0, wander_speed_frac=wander_speed_frac, scan_turn=scan_turn,
            tree_wait_dist=tree_wait_dist, flee_sprint_dist=flee_sprint_dist, face_predator=face_predator,
        )

    def act_batch(self, agent_states):
        current_ids = {s["agent_id"] for s in agent_states}
        for stale_id in list(self.memory.keys()):
            if stale_id not in current_ids:
                del self.memory[stale_id]
        for s in agent_states:
            self.memory.setdefault(s["agent_id"], {"mode": "WAITING", "leaving_steps": 0})
        # Snapshot every agent's mode as of the START of this step, so "not counting agents
        # already in LEAVING" is order-independent within this batch (not mutated mid-pass).
        self._mode_snapshot = {aid: m["mode"] for aid, m in self.memory.items()}
        by_id = {s["agent_id"]: s for s in agent_states}

        actions = []
        for s in agent_states:
            try:
                actions.append(self._act_one(s, by_id))
            except Exception:
                actions.append(self._fallback.act(s))
        return actions

    def _agents_at_tree(self, tree_angle: float, tree_distance: float, other_agents: list) -> int:
        count = 1  # itself, per the definition, regardless of its own distance to the tree
        for o in other_agents:
            oid = o.get("id")
            if oid is None or self._mode_snapshot.get(oid) == "LEAVING":
                continue
            d = math.sqrt(tree_distance ** 2 + o["distance"] ** 2
                          - 2 * tree_distance * o["distance"] * math.cos(tree_angle - o["angle"]))
            if d <= self.tree_radius:
                count += 1
        return count

    def _straight_action(self, s: dict, edges: list):
        lookahead = s["speed"]
        for e in edges:
            (x1, y1), (x2, y2) = e["coords"]
            if _crosses_forward(x1, y1, x2, y2, lookahead):
                return s["speed"], 0.0, self.leaving_turn_angle
        return s["speed"], 0.0, 0.0

    def _act_one(self, s: dict, by_id: dict) -> dict:
        agent_id = s["agent_id"]
        m = self.memory[agent_id]
        obs = s["observations"]
        predators = [o for o in obs if o["type"] == "Predator"]
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]
        other_agents = [o for o in obs if o["type"] == "Agent"]
        edges = [o for o in obs if o["type"] == "Edge"]

        # Fleeing always keeps v0's priority, in any mode, and does not itself change mode.
        if predators:
            if m["mode"] == "LEAVING":
                m["leaving_steps"] += 1
            p = min(predators, key=lambda o: o["distance"])
            move_direction = _wrap(p["angle"] + math.pi)
            move_distance = s["sprint_speed"] if p["distance"] < self.flee_sprint_dist else s["speed"]
            turn_angle = p["angle"] if self.face_predator else 0.0
            return {"agent_id": agent_id, "move_distance": float(move_distance),
                    "move_direction": float(move_direction), "turn_angle": float(turn_angle),
                    "spawn_agent": False}

        # ---- Rule 2 (LEAVING branch) ----
        if m["mode"] == "LEAVING":
            m["leaving_steps"] += 1
            if m["leaving_steps"] > self.leaving_ignore_steps:
                candidates = []
                for t in trees:
                    cnt = self._agents_at_tree(t["angle"], t["distance"], other_agents)
                    if cnt < self.tree_join_threshold:
                        candidates.append((cnt, t["distance"], t))
                if candidates:
                    candidates.sort(key=lambda c: (c[0], c[1]))
                    t = candidates[0][2]
                    m["mode"] = "WAITING"
                    m["leaving_steps"] = 0
                    if t["distance"] > self.tree_wait_dist:
                        move_direction = t["angle"]
                        move_distance = min(t["distance"] - self.tree_wait_dist, s["speed"] * self.wander_speed_frac)
                        turn_angle = t["angle"]
                    else:
                        move_direction, move_distance, turn_angle = 0.0, 0.0, self.scan_turn
                    return {"agent_id": agent_id, "move_distance": float(move_distance),
                            "move_direction": float(move_direction), "turn_angle": float(turn_angle),
                            "spawn_agent": False}
            move_distance, move_direction, turn_angle = self._straight_action(s, edges)
            return {"agent_id": agent_id, "move_distance": float(move_distance),
                    "move_direction": float(move_direction), "turn_angle": float(turn_angle),
                    "spawn_agent": False}

        # ---- Normal (WAITING) mode: v0's fruit/tree priority, unmodified, plus Rule 1 ----
        move_distance, move_direction, turn_angle = 0.0, 0.0, 0.0
        entering_leaving = False
        leaving_face_tree = None  # angle to face away from; None = no tree to face away from

        if fruits:
            f = min(fruits, key=lambda o: o["distance"])
            move_direction = f["angle"]
            move_distance = min(f["distance"], s["speed"])
            turn_angle = f["angle"]
        elif trees:
            t = min(trees, key=lambda o: o["distance"])
            if t["distance"] > self.tree_wait_dist:
                move_direction = t["angle"]
                move_distance = min(t["distance"] - self.tree_wait_dist, s["speed"] * self.wander_speed_frac)
                turn_angle = t["angle"]
            else:
                # Waiting at this tree: Rule 1 cap check.
                cnt = self._agents_at_tree(t["angle"], t["distance"], other_agents)
                if cnt > self.tree_cap:
                    highest = (s["energy"], -agent_id)
                    is_highest = True
                    for o in other_agents:
                        oid = o.get("id")
                        if oid is None or self._mode_snapshot.get(oid) == "LEAVING":
                            continue
                        d = math.sqrt(t["distance"] ** 2 + o["distance"] ** 2
                                      - 2 * t["distance"] * o["distance"] * math.cos(t["angle"] - o["angle"]))
                        if d <= self.tree_radius:
                            other_state = by_id.get(oid)
                            if other_state is None:
                                continue
                            if (other_state["energy"], -oid) > highest:
                                is_highest = False
                                break
                    if is_highest:
                        entering_leaving = True
                        leaving_face_tree = t["angle"]
                    else:
                        turn_angle = self.scan_turn
                else:
                    turn_angle = self.scan_turn
        else:
            move_distance, move_direction, turn_angle = self._straight_action(s, edges)

        # ---- Rule 3: births (independent of the movement branch above; Rule 1, if it fired,
        # already set entering_leaving, so this can't also fire -- an overcrowded tree can't
        # simultaneously have "at most 1 other agent") ----
        spawn = False
        if not entering_leaving and s["energy"] > self.birth_energy_threshold:
            if trees:
                t = min(trees, key=lambda o: o["distance"])
                cnt = self._agents_at_tree(t["angle"], t["distance"], other_agents)
                others = cnt - 1
                can_spawn = others <= self.birth_max_others_at_tree
                spawn_face_tree = t["angle"] if can_spawn else None
            else:
                others = sum(1 for o in other_agents
                             if o["distance"] <= self.birth_no_tree_radius
                             and self._mode_snapshot.get(o.get("id")) != "LEAVING")
                can_spawn = others <= self.birth_max_others_no_tree
                spawn_face_tree = None
            if can_spawn:
                spawn = True
                entering_leaving = True
                leaving_face_tree = spawn_face_tree

        if entering_leaving:
            m["mode"] = "LEAVING"
            m["leaving_steps"] = 0
            if leaving_face_tree is not None:
                away = _wrap(leaving_face_tree + math.pi)
                move_direction, turn_angle = away, away
            else:
                move_direction, turn_angle = 0.0, 0.0
            move_distance = s["speed"]

        return {"agent_id": agent_id, "move_distance": float(move_distance),
                "move_direction": float(move_direction), "turn_angle": float(turn_angle),
                "spawn_agent": bool(spawn)}
