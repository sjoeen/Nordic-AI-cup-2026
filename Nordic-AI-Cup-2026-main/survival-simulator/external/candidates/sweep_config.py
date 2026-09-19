"""
Config sweep for an external candidate WITHOUT modifying it: each variant is the candidate's own
policy class built with keyword overrides on top of its shipped CONFIG. Same headless loop as
run_candidate.py (SimulationCore(seed, starting_predators=0), 3000 s limit, one step per tick).

    python external/candidates/sweep_config.py --seeds 2000:2016 --out logs/sweep_pop.csv \
        --variant base --variant floor4:population_floor=4 --variant "slow:population_decay=1800,population=8"

Every game is also a post-mortem (read from the engine from outside, never altering it): births,
deaths by cause over the whole game and over the final 300 s, and the last deaths in detail
(<out>.deaths.jsonl). Causes: eaten = an awake-or-not predator within 45 units at death with energy
left; old_age = past the agent's hidden max_age; newborn_starved = starved younger than 30 s;
starved = the rest.

Mechanism study: writes wherever --out points (use logs/), never results/index.csv.
"""
import argparse
import csv
import json
import math
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
    peak, births, deaths, env = 0, 0, [], sim.env
    genes, next_sample = {}, 250.
    while state["num_agents"] > 0 and env.time <= MAX_SIM_TIME:
        states = [o for o in state["observations"] if o is not None]
        peak = max(peak, len(states))
        actions = policy.decide_all(states, env.time)
        before = {a.agent_id: (a.x, a.y, a.energy, a.age, a.max_age) for a in env.agents}
        predators = [(q.x, q.y) for q in env.predators]
        state = sim.step([(a.agent_id, ActionRequest(**a.model_dump())) for a in actions])
        alive = {a.agent_id for a in env.agents}
        births += len(alive - before.keys())
        if env.time >= next_sample and env.agents:  # colony's mean walking-speed gene, to see selection act
            genes[f"speed_t{int(next_sample)}"] = round(statistics.mean(min(a.speed, a.sprint_speed) for a in env.agents), 2)
            next_sample += 250.
        for aid in before.keys() - alive:
            x, y, energy, age, max_age = before[aid]
            near = min((math.hypot(x - px, y - py) for px, py in predators), default=1e9)
            cause = ("eaten" if near < 45 and energy > 3 else "old_age" if age > max_age
                     else "newborn_starved" if age < 30 else "starved")
            deaths.append(dict(t=round(env.time, 1), cause=cause, energy=round(energy, 1), age=round(age, 1),
                               nearest_predator=round(near), alive_after=len(alive)))
    end = env.time
    causes = ("eaten", "old_age", "newborn_starved", "starved")
    total = {c: sum(d["cause"] == c for d in deaths) for c in causes}
    final = {"final300_" + c: sum(d["cause"] == c and d["t"] > end - 300 for d in deaths) for c in causes}
    row = dict(variant=name, seed=seed, score=round(float(state["score"]), 2), extinction_time=round(end, 1),
               peak_agents=peak, births=births, **total, **final, predators_at_end=len(env.predators),
               trees_at_end=len(env.trees), fruits_at_end=len(env.fruits), wall_clock_sec=round(time.perf_counter() - t0, 1),
               **genes, **{k: v for k, v in policy.metrics.items() if k.startswith(("sel_", "fedbirth_"))})
    print(json.dumps(row), flush=True)
    return row, deaths[-8:]


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
    rows, last_deaths = [], []
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(run_one, job) for job in jobs]):
            row, last = future.result()
            rows.append(row)
            last_deaths.append(dict(variant=row["variant"], seed=row["seed"], extinction_time=row["extinction_time"], last_deaths=last))
    rows.sort(key=lambda r: (r["variant"], r["seed"]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fields = list(dict.fromkeys(key for r in rows for key in r))  # games differ in length, so in sampled columns
        writer = csv.DictWriter(f, fieldnames=fields, restval="")
        writer.writeheader()
        writer.writerows(rows)
    with open(str(out) + ".deaths.jsonl", "w") as f:
        f.writelines(json.dumps(d) + "\n" for d in sorted(last_deaths, key=lambda d: (d["variant"], d["seed"])))
    reference = {r["seed"]: r["extinction_time"] for r in rows if r["variant"] == variants[0][0]}
    print(f"\n{'variant':<14}{'mean':>8}{'median':>8}{'min':>8}{'max':>8}{'vs ' + variants[0][0]:>12}{'better':>8}")
    for name, _ in variants:
        t = {r["seed"]: r["extinction_time"] for r in rows if r["variant"] == name}
        delta = [t[s] - reference[s] for s in seeds]
        print(f"{name:<14}{statistics.mean(t.values()):>8.0f}{statistics.median(t.values()):>8.0f}{min(t.values()):>8.0f}"
              f"{max(t.values()):>8.0f}{statistics.mean(delta):>+12.0f}{sum(d > 0 for d in delta):>5}/{len(seeds)}")
    print(f"\nmean per game{'':<1}{'births':>8}{'eaten':>8}{'old_age':>9}{'newborn_starved':>17}{'starved':>9}   | final 300 s: eaten / old / newborn / starved")
    for name, _ in variants:
        mine = [r for r in rows if r["variant"] == name]
        avg = lambda key: statistics.mean(r[key] for r in mine)
        print(f"{name:<14}{avg('births'):>8.0f}{avg('eaten'):>8.1f}{avg('old_age'):>9.1f}{avg('newborn_starved'):>17.1f}{avg('starved'):>9.1f}"
              f"   | {avg('final300_eaten'):.1f} / {avg('final300_old_age'):.1f} / {avg('final300_newborn_starved'):.1f} / {avg('final300_starved'):.1f}")
    samples = sorted({k for r in rows for k in r if k.startswith("speed_t")}, key=lambda k: int(k[7:]))[:6]
    if samples:
        print("\nmean walking-speed gene of the colony (games still alive at that time)")
        print(f"{'variant':<14}" + "".join(f"{k[6:]:>9}" for k in samples))
        for name, _ in variants:
            mine = [r for r in rows if r["variant"] == name]
            cells = [[r[k] for r in mine if k in r] for k in samples]
            print(f"{name:<14}" + "".join(f"{statistics.mean(v):>9.2f}" if v else f"{'-':>9}" for v in cells))
    print(f"wrote {out}")
