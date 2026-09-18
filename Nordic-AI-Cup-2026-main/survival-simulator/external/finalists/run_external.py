"""
Runs a finalist controller (agent_dispersal / agent_decoy / agent_refuge) or our own
heuristic_v1 through the same headless game loop as training/batch_runner.py's
_run_one: SimulationCore(seed=seed, starting_predators=0), MAX_SIM_TIME=3000, one
sim.step() per tick. The finalist's own decide_all() is called every step through a
thin adapter (ExternalAdapter) that only converts its ActionRequest objects to dicts --
it does not alter which action is chosen.

This script is not part of the batch runner and does not modify it.
"""
import argparse
import csv
import json
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

SIM_ROOT = Path(__file__).resolve().parents[2]  # .../survival-simulator
FINALISTS_DIR = Path(__file__).resolve().parent / "survival-finalists"
for p in (str(SIM_ROOT), str(FINALISTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

MAX_SIM_TIME = 3000  # matches training/batch_runner.py

MODULES = {"dispersal": "agent_dispersal", "refuge": "agent_refuge", "decoy": "agent_decoy"}


class ExternalAdapter:
    """agents.<x>-style act_batch() wrapper around a finalist's make_policy()/decide_all()."""

    def __init__(self, module_name):
        import importlib
        module = importlib.import_module(module_name)
        self.policy = module.make_policy()

    def act_batch(self, agent_states):
        actions = self.policy.decide_all(agent_states)
        return [a.model_dump() for a in actions]


def make_agent(agent_name):
    if agent_name == "v1":
        from agents.heuristic_v1 import HeuristicAgentV1
        config = json.loads((SIM_ROOT / "training/configs/heuristic_v1.json").read_text())
        return HeuristicAgentV1(**config["params"])
    return ExternalAdapter(MODULES[agent_name])


def run_one(agent_name, seed):
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    agent = make_agent(agent_name)
    sim = SimulationCore(seed=seed, starting_predators=0)
    t0 = time.perf_counter()
    state = sim.step([])
    while state["num_agents"] > 0 and sim.env.time <= MAX_SIM_TIME:
        agent_states = [o for o in state["observations"] if o is not None]
        actions = agent.act_batch(agent_states)
        parsed = [(a["agent_id"], ActionRequest(**a)) for a in actions]
        state = sim.step(parsed)
    row = dict(
        agent=agent_name, seed=seed, score=round(float(state["score"]), 4),
        extinction_time=round(sim.env.time, 2),
        end_reason="extinction" if state["num_agents"] == 0 else "time_limit",
        wall_clock_sec=round(time.perf_counter() - t0, 2),
    )
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--agents", nargs="+", required=True, choices=["dispersal", "refuge", "decoy", "v1"])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()

    jobs = [(agent, seed) for agent in a.agents for seed in a.seeds]
    workers = a.workers or max(1, min(len(jobs), (os.cpu_count() or 2) - 1))

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, agent, seed): (agent, seed) for agent, seed in jobs}
        for fut in as_completed(futures):
            rows.append(fut.result())

    rows.sort(key=lambda r: (r["agent"], r["seed"]))
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "seed", "score", "extinction_time", "end_reason", "wall_clock_sec"])
        writer.writeheader()
        writer.writerows(rows)

    for agent in a.agents:
        scores = [r["score"] for r in rows if r["agent"] == agent]
        print(f"{agent}: mean={statistics.mean(scores):.4f} median={statistics.median(scores):.4f} "
              f"n={len(scores)} scores={scores}")
    print(f"wrote {out_path}")
