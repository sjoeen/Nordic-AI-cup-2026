"""Run from survival-simulator/:  python -m pytest -q tests"""
import json
import math
from pathlib import Path

from agents import make_agent
from src.utils.DTOs import ActionRequest

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "training" / "configs").glob("*.json"))


def state(observations, energy=150.0, agent_id=7):
    return {"agent_id": agent_id, "observations": observations, "energy": energy, "biome": "grassland",
            "age": 1.0, "speed": 10.0, "sprint_speed": 20.0, "hearing_radius": 50.0,
            "vision_angle": math.pi / 3, "vision_range": 200.0, "max_energy": 500.0}


def heuristic():
    return make_agent(json.loads((ROOT / "training/configs/heuristic_v0.json").read_text()))


def test_configs_load_and_produce_valid_actions():
    assert CONFIGS
    obs_sets = [[], [{"type": "Fruit", "distance": 30.0, "angle": 0.4}],
                [{"type": "Predator", "distance": 40.0, "angle": -1.0, "rel_dir": 0.2}],
                [{"type": "Tree", "distance": 120.0, "angle": 2.0}],
                [{"type": "Edge", "coords": ((1.0, 2.0), (3.0, 4.0))}]]
    for path in CONFIGS:
        agent = make_agent(json.loads(path.read_text()))
        states = [state(o, agent_id=i) for i, o in enumerate(obs_sets)]
        actions = agent.act_batch(states)
        assert [a["agent_id"] for a in actions] == list(range(len(obs_sets)))
        for a in actions:
            ActionRequest(**a)


def test_heuristic_flees_directly_away_and_faces_predator():
    a = heuristic().act(state([{"type": "Predator", "distance": 40.0, "angle": -1.0, "rel_dir": 0.2}]))
    assert math.isclose(a["move_direction"], -1.0 + math.pi, abs_tol=1e-9)
    assert a["turn_angle"] == -1.0
    assert a["move_distance"] == 20.0  # within flee_sprint_dist -> sprint
    assert a["spawn_agent"] is False


def test_heuristic_does_not_overshoot_fruit():
    a = heuristic().act(state([{"type": "Fruit", "distance": 4.0, "angle": 0.3}]))
    assert a["move_distance"] == 4.0 and a["move_direction"] == 0.3


def test_heuristic_spawn_threshold():
    ag = heuristic()
    assert ag.act(state([], energy=ag.spawn_energy_threshold + 1))["spawn_agent"] is True
    assert ag.act(state([], energy=ag.spawn_energy_threshold - 1))["spawn_agent"] is False


def test_episode_seed_is_deterministic():
    from training.episode import run_episode
    r1 = run_episode(heuristic(), seed=3, max_sim_time=20)
    r2 = run_episode(heuristic(), seed=3, max_sim_time=20)
    assert r1["score"] == r2["score"] and r1["steps"] == r2["steps"]
