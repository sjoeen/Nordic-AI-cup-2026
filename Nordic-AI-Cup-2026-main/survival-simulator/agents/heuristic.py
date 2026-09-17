"""
Rule-based baseline. Every numeric choice is a constructor parameter supplied by a
config file in training/configs/ -- nothing is hardcoded here.

Priority per agent, per step:
  1. Predator observed -> face it and back away (predators only chase agents that look
     away or are very close); sprint if it is within flee_sprint_dist.
  2. Fruit observed    -> turn to and walk onto the nearest fruit.
  3. Tree observed     -> walk toward the nearest tree (fruit spawns around trees),
     then wait near it while slowly scanning.
  4. Nothing observed  -> wander forward at a reduced speed while slowly scanning.
Spawning is requested when energy exceeds spawn_energy_threshold and no predator is seen.

Engine facts this relies on (src/elements/environment.py, predator.py):
  - observation "angle" and action "move_direction" are both relative to current heading.
  - movement is applied before the turn within a step.
"""
import math


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class HeuristicAgent:
    def __init__(self, spawn_energy_threshold: float, wander_speed_frac: float, scan_turn: float,
                 tree_wait_dist: float, flee_sprint_dist: float, face_predator: bool):
        self.spawn_energy_threshold = spawn_energy_threshold
        self.wander_speed_frac = wander_speed_frac
        self.scan_turn = scan_turn
        self.tree_wait_dist = tree_wait_dist
        self.flee_sprint_dist = flee_sprint_dist
        self.face_predator = face_predator

    def act_batch(self, agent_states):
        return [self.act(s) for s in agent_states]

    def act(self, s: dict) -> dict:
        obs = s["observations"]
        predators = [o for o in obs if o["type"] == "Predator"]
        fruits = [o for o in obs if o["type"] == "Fruit"]
        trees = [o for o in obs if o["type"] == "Tree"]

        move_distance, move_direction, turn_angle = 0.0, 0.0, 0.0

        if predators:
            p = min(predators, key=lambda o: o["distance"])
            move_direction = _wrap(p["angle"] + math.pi)
            move_distance = s["sprint_speed"] if p["distance"] < self.flee_sprint_dist else s["speed"]
            turn_angle = p["angle"] if self.face_predator else 0.0
        elif fruits:
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
                turn_angle = self.scan_turn
        else:
            move_distance = s["speed"] * self.wander_speed_frac
            turn_angle = self.scan_turn

        spawn = (not predators) and s["energy"] > self.spawn_energy_threshold

        return {
            "agent_id": s["agent_id"],
            "move_distance": float(move_distance),
            "move_direction": float(move_direction),
            "turn_angle": float(turn_angle),
            "spawn_agent": bool(spawn),
        }
