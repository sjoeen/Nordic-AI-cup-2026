"""
Config sweep for an external candidate WITHOUT modifying it: each variant is the candidate's own
policy class built with keyword overrides on top of its shipped CONFIG. Same headless loop as
run_candidate.py (SimulationCore(seed, starting_predators=0), 3000 s limit, one step per tick).

    python external/candidates/sweep_config.py --seeds 2000:2016 --out logs/sweep_pop.csv \
        --variant base --variant floor4:population_floor=4 --variant "slow:population_decay=1800,population=8"

Mechanism study: writes wherever --out points (use logs/), never results/index.csv.
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

SIM_ROOT = Path(__file__).resolve().parents[2]
CANDIDATES_ROOT = Path(__file__).resolve().parent
if str(SIM_ROOT) not in sys.path:
    sys.path.insert(0, str(SIM_ROOT))
MAX_SIM_TIME = 3000


def parse_seeds(tokens):
    seeds = []
    for token in tokens:
        if ":" in token:
            start, stop = map(int, token.split(":"))
            seeds.extend(range(start, stop))
        else:
            seeds.append(int(token))
    return seeds


def parse_variant(text):
    name, _, body = text.partition(":")
    overrides = {}
    for pair in filter(None, body.split(",")):
        key, value = pair.split("=")
        overrides[key.strip()] = json.loads(value) if value.strip()[0] in '-0123456789tfn"[{' else value.strip()
    return name, overrides


def run_one(job):
    candidate, name, overrides, seed = job
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    sys.path.insert(0, str(CANDIDATES_ROOT / candidate))
    import survival_agent
    base = survival_agent.make_policy()
    unknown = set(overrides) - set(base.config)
    if unknown:
        raise ValueError(f"{name}: not config keys of this candidate: {sorted(unknown)}")
    policy = type(base)("appetite", **{**survival_agent.CONFIG, **overrides})
    sim = SimulationCore(seed=seed, starting_predators=0)
    t0 = time.perf_counter()
    state = sim.step([])
    peak = 0
    while state["num_agents"] > 0 and sim.env.time <= MAX_SIM_TIME:
        states = [o for o in state["observations"] if o is not None]
        peak = max(peak, len(states))
        actions = policy.decide_all(states, sim.env.time)
        state = sim.step([(a.agent_id, ActionRequest(**a.model_dump())) for a in actions])
    row = dict(variant=name, seed=seed, score=round(float(state["score"]), 2), extinction_time=round(sim.env.time, 1),
               peak_agents=peak, predators_at_end=len(sim.env.predators), wall_clock_sec=round(time.perf_counter() - t0, 1))
    print(json.dumps(row), flush=True)
    return row


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", default="original-eat-rest-preserved")
    p.add_argument("--seeds", nargs="+", required=True, help="ints and start:stop ranges (stop exclusive)")
    p.add_argument("--variant", action="append", required=True, help="name[:key=value,key=value]")
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    seeds, variants = parse_seeds(a.seeds), [parse_variant(v) for v in a.variant]
    jobs = [(a.candidate, name, overrides, seed) for name, overrides in variants for seed in seeds]
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run_one, job) for job in jobs]):
            rows.append(future.result())
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    reference = {r["seed"]: r["extinction_time"] for r in rows if r["variant"] == variants[0][0]}
    print(f"\n{'variant':<14}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'vs ' + variants[0][0]:>12}{'better':>8}")
    for name, _ in variants:
        t = {r["seed"]: r["extinction_time"] for r in rows if r["variant"] == name}
        delta = [t[s] - reference[s] for s in seeds]
        print(f"{name:<14}{statistics.mean(t.values()):>8.0f}{statistics.median(t.values()):>8.0f}{min(t.values()):>8.0f}"
              f"{max(t.values()):>8.0f}{statistics.mean(delta):>+12.0f}{sum(d > 0 for d in delta):>5}/{len(seeds)}")
    print(f"wrote {out}")
