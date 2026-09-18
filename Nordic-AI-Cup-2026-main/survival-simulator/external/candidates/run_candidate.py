"""
Runs the original-eat-rest-overcrowding-v4 candidate (external/candidates/
original-eat-rest-overcrowding-v4/survival_agent.py) through the same headless
game loop as training/batch_runner.py's _run_one: SimulationCore(seed=seed,
starting_predators=0), MAX_SIM_TIME=3000, one sim.step() per tick. The
candidate's own make_policy()/decide_all() is called every step through a thin
adapter that only converts its ActionRequest objects to dicts -- it does not
alter which action is chosen. This is a real full game to extinction or the
3000s time limit, not the package's own decision-check suite.

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
CANDIDATE_DIR = Path(__file__).resolve().parent / "original-eat-rest-overcrowding-v4"
for p in (str(SIM_ROOT), str(CANDIDATE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

MAX_SIM_TIME = 3000  # matches training/batch_runner.py


class CandidateAdapter:
    """decide_all(states, sim_time)-style wrapper around survival_agent.make_policy()."""

    def __init__(self):
        import survival_agent
        self.policy = survival_agent.make_policy()

    def decide(self, agent_states, sim_time):
        actions = self.policy.decide_all(agent_states, sim_time)
        return [a.model_dump() for a in actions]


def run_one(seed):
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest

    agent = CandidateAdapter()
    sim = SimulationCore(seed=seed, starting_predators=0)
    t0 = time.perf_counter()
    state = sim.step([])
    while state["num_agents"] > 0 and sim.env.time <= MAX_SIM_TIME:
        agent_states = [o for o in state["observations"] if o is not None]
        actions = agent.decide(agent_states, sim.env.time)
        parsed = [(a["agent_id"], ActionRequest(**a)) for a in actions]
        state = sim.step(parsed)
    row = dict(
        agent="original_eat_rest_overcrowding_v4", seed=seed, score=round(float(state["score"]), 4),
        extinction_time=round(sim.env.time, 2),
        end_reason="extinction" if state["num_agents"] == 0 else "time_limit",
        wall_clock_sec=round(time.perf_counter() - t0, 2),
    )
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=None)
    a = p.parse_args()

    workers = a.workers or max(1, min(len(a.seeds), (os.cpu_count() or 2) - 1))

    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, seed): seed for seed in a.seeds}
        for fut in as_completed(futures):
            rows.append(fut.result())

    rows.sort(key=lambda r: r["seed"])
    out_path = Path(a.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["agent", "seed", "score", "extinction_time", "end_reason", "wall_clock_sec"])
        writer.writeheader()
        writer.writerows(rows)

    scores = [r["score"] for r in rows]
    print(f"original_eat_rest_overcrowding_v4: mean={statistics.mean(scores):.4f} median={statistics.median(scores):.4f} "
          f"n={len(scores)} scores={scores}")
    print(f"wrote {out_path}")
