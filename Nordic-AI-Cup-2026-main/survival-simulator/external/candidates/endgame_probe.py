"""
How does eat-rest-v1 die? Plays the untouched baseline (or --set overrides) on fresh seeds and records, per game, a 50 s
timeline of the world and the colony plus every death and birth, then prints what changes in the run-up to extinction.

Diagnostic for choosing the next behaviour change after tune_v1 found the numeric settings flat. Seeds 11000+ (unused by
the tuners: 5000-7063, 8000-10063). Resumable like the tuners: finished games are appended to <out>/games.jsonl.
    python external/candidates/endgame_probe.py --out logs/endgame                 # 64 games, then the report
    python external/candidates/endgame_probe.py --out logs/endgame --seeds 0       # report only, from the stored games
    python external/candidates/endgame_probe.py --out logs/endgame_eg --candidate eat-rest-endgame --compare logs/endgame
--brief skips the long report (progress and the --compare block only). --candidate picks another folder of external/candidates; --compare prints the paired per-seed difference against the
games stored in another --out folder (same seeds), which is how a behaviour change is judged.
Mechanism study: nothing is written to results/.
"""
import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_config import CANDIDATES_ROOT, MAX_SIM_TIME  # noqa: E402
from tune_v1 import CANDIDATE  # noqa: E402


def play(job):
    candidate, overrides, seed = job
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    sys.path.insert(0, str(CANDIDATES_ROOT / candidate))
    import survival_agent
    base = survival_agent.make_policy()
    policy = type(base)("appetite", **{**survival_agent.CONFIG, **overrides})
    sim = SimulationCore(seed=seed, starting_predators=0)
    state = sim.step([])
    env, timeline, deaths, births, next_sample = sim.env, [], [], [], 0.
    sensed, threatened, ticks = 0., 0., 0  # what the agents themselves can sense, averaged over each 50 s window
    while state["num_agents"] > 0 and env.time <= MAX_SIM_TIME:
        states = [o for o in state["observations"] if o is not None]
        actions = policy.decide_all(states, env.time)
        before = {a.agent_id: (a.x, a.y, a.energy, a.age, a.max_age) for a in env.agents}
        counts = [sum(o["type"] == "Predator" for o in s["observations"]) for s in states]
        sensed, threatened, ticks = sensed + sum(counts) / len(counts), threatened + sum(c > 0 for c in counts) / len(counts), ticks + 1
        predators = [(q.x, q.y) for q in env.predators]
        if env.time >= next_sample:
            agents = env.agents
            timeline.append(dict(t=round(env.time), n=len(agents), pred=len(env.predators), awake=sum(not q.resting for q in env.predators),
                                 trees=len(env.trees), fruits=len(env.fruits), fruit_energy=round(sum(f.energy for f in env.fruits)),
                                 energy=round(statistics.mean(a.energy for a in agents), 1),
                                 max_energy=round(statistics.mean(a.max_energy for a in agents), 1),
                                 age=round(statistics.mean(a.age for a in agents), 1),
                                 speed=round(statistics.mean(min(a.speed, a.sprint_speed) for a in agents), 1),
                                 ratio=round(len(env.predators) / len(agents), 2), sensed=round(sensed / ticks, 3),
                                 threatened=round(threatened / ticks, 3), endgame=int(getattr(policy, "endgame", False))))
            sensed, threatened, ticks = 0., 0., 0
            next_sample += 50.
        state = sim.step([(a.agent_id, ActionRequest(**a.model_dump())) for a in actions])
        alive = {a.agent_id for a in env.agents}
        births += [round(env.time, 1)] * len(alive - before.keys())
        for aid in before.keys() - alive:
            x, y, energy, age, max_age = before[aid]
            near = min((math.hypot(x - px, y - py) for px, py in predators), default=1e9)
            cause = ("eaten" if near < 45 and energy > 3 else "old_age" if age > max_age
                     else "newborn_starved" if age < 30 else "starved")
            deaths.append(dict(t=round(env.time, 1), cause=cause, energy=round(energy, 1), age=round(age, 1), alive_after=len(alive)))
    return dict(seed=seed, end=round(env.time, 1), score=round(float(state["score"]), 2), timeline=timeline, deaths=deaths, births=births,
                metrics={k: v for k, v in policy.metrics.items() if k.startswith(("eg_", "emergency_"))})


def report(games):
    mean = statistics.mean
    ends = [g["end"] for g in games]
    print(f"\n{len(games)} games   extinction: mean {mean(ends):.0f}  sd {statistics.pstdev(ends):.0f}  min {min(ends):.0f}  "
          f"median {statistics.median(ends):.0f}  max {max(ends):.0f}   score mean {mean(g['score'] for g in games):.0f}")

    print("\n-- world and colony by clock time (mean over the games still alive then)")
    # ratio = predators alive / agents (simulator truth, invisible to the agents); sensed = predators in an agent's own
    # observations, threatened = share of agents sensing at least one (both averaged over the 50 s before the sample)
    columns = ("n", "pred", "awake", "ratio", "sensed", "threatened", "endgame", "trees", "fruits", "fruit_energy", "energy", "age", "speed")
    cell = lambda rows, c: f"{mean(r[c] for r in rows if c in r):>13.2f}" if any(c in r for r in rows) else f"{'-':>13}"
    print(f"{'t':>6}{'games':>6}" + "".join(f"{c:>13}" for c in columns))
    for t in range(0, MAX_SIM_TIME + 1, 250):
        rows = [p for g in games for p in g["timeline"] if p["t"] // 50 * 50 == t]
        if rows:
            print(f"{t:>6}{len(rows):>6}" + "".join(cell(rows, c) for c in columns))

    print("\n-- the same, counted back from each game's extinction")
    print(f"{'t-end':>6}{'games':>6}" + "".join(f"{c:>13}" for c in columns))
    for back in (600, 450, 300, 200, 150, 100, 50, 0):
        rows = [min(g["timeline"], key=lambda p: abs(p["t"] - (g["end"] - back))) for g in games if g["end"] > back]
        print(f"{-back:>6}{len(rows):>6}" + "".join(cell(rows, c) for c in columns))
    for key in ("ratio", "sensed", "threatened"):  # is there one level that separates "about to die" from "fine"?
        late = [p[key] for g in games for p in g["timeline"] if key in p and g["end"] - p["t"] <= 100]
        early = [p[key] for g in games for p in g["timeline"] if key in p and g["end"] - p["t"] > 400 and p["t"] >= 300]
        if late and early:
            q = lambda v, f: sorted(v)[int(f * (len(v) - 1))]
            print(f"  {key:<11} last 100 s: p25 {q(late, .25):.2f}  median {q(late, .5):.2f}  p75 {q(late, .75):.2f}      "
                  f"more than 400 s out: p25 {q(early, .25):.2f}  median {q(early, .5):.2f}  p75 {q(early, .75):.2f}")

    print("\n-- deaths by cause, per window before extinction (and births in the same window)")
    for far, near in ((float("inf"), 600), (600, 300), (300, 100), (100, 0)):
        inside = lambda g, t: near <= g["end"] - t < far
        causes = Counter(d["cause"] for g in games for d in g["deaths"] if inside(g, d["t"]))
        total = sum(causes.values()) or 1
        born = sum(inside(g, b) for g in games for b in g["births"])
        label = f"{far:.0f}..{near} s" if far < float("inf") else f"before {near} s"
        print(f"  {label:>14}: {total:>5} deaths, {born:>5} births   " + "  ".join(f"{c} {n / total:.0%}" for c, n in causes.most_common()))

    print("\n-- how the colony ends")
    print("  cause of the very last death :", dict(Counter(g["deaths"][-1]["cause"] for g in games).most_common()))
    print("  cause of the last 4 deaths   :", dict(Counter(d["cause"] for g in games for d in g["deaths"][-4:]).most_common()))
    gaps = [g["end"] - g["births"][-1] for g in games if g["births"]]
    print(f"  last birth before extinction : mean {mean(gaps):.0f} s  median {statistics.median(gaps):.0f}  min {min(gaps):.0f}  max {max(gaps):.0f}")
    print(f"  energy of agents eaten in the final 300 s: mean "
          f"{mean([d['energy'] for g in games for d in g['deaths'] if d['cause'] == 'eaten' and d['t'] > g['end'] - 300] or [0]):.0f}")

    print("\n-- per game: the last 5 deaths as cause@seconds-before-end(energy, age)")
    for g in sorted(games, key=lambda g: g["end"]):
        tail = "  ".join(f"{d['cause'][:3]}@{d['t'] - g['end']:+.0f}(E{d['energy']:.0f},a{d['age']:.0f})" for d in g["deaths"][-5:])
        print(f"  seed {g['seed']}  end {g['end']:>6.0f}   {tail}")


def compare(games, reference, label):
    seeds = sorted(set(games) & set(reference))
    if len(seeds) < 2:
        print(f"\n-- compare: fewer than 2 shared seeds with {label}")
        return
    print(f"\n-- paired against {label} on {len(seeds)} shared seeds")
    for key in ("score", "end"):
        delta = [games[s][key] - reference[s][key] for s in seeds]
        se = statistics.stdev(delta) / len(delta) ** .5
        print(f"  {key:>5}: {statistics.mean(games[s][key] for s in seeds):.0f} vs {statistics.mean(reference[s][key] for s in seeds):.0f}   "
              f"delta {statistics.mean(delta):+.0f}  SE {se:.0f}  better on {sum(d > 0 for d in delta)}/{len(delta)}  "
              f"worst {min(delta):+.0f}  best {max(delta):+.0f}")
    print("  per seed (score delta): " + "  ".join(f"{s}:{games[s]['score'] - reference[s]['score']:+.0f}" for s in seeds))
    used = [games[s].get("metrics", {}) for s in seeds]
    for key in sorted({k for m in used for k in m}):
        print(f"  {key:<22} mean {statistics.mean(m.get(key, 0) for m in used):.1f}")


def load(path):
    return {g["seed"]: g for g in map(json.loads, path.read_text().splitlines())} if path.exists() else {}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--seeds", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=11000)
    p.add_argument("--candidate", default=CANDIDATE, help="folder under external/candidates")
    p.add_argument("--brief", action="store_true", help="skip the long report: progress lines and the --compare block only")
    p.add_argument("--compare", help="another --out folder holding the same seeds (usually the baseline's)")
    p.add_argument("--set", default="{}", help='json overrides on top of the shipped config, e.g. \'{"population": 4}\'')
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "games.jsonl"
    games = load(path)
    jobs = [(a.candidate, json.loads(a.set), s) for s in range(a.seed_base, a.seed_base + a.seeds) if s not in games]
    print(f"=== endgame probe: {a.candidate}, {len(jobs)} games to run, {len(games)} stored, overrides {a.set} ===", flush=True)
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=a.workers) as pool, path.open("a") as log:
        for done, future in enumerate(as_completed([pool.submit(play, j) for j in jobs]), 1):
            game = future.result()
            games[game["seed"]] = game
            log.write(json.dumps(game) + "\n")
            log.flush()
            if done % 8 == 0 or done == len(jobs):
                print(f"  [{(time.perf_counter() - t0) / 60:5.1f} min] {done}/{len(jobs)} games", flush=True)
    games = {s: g for s, g in games.items() if a.seed_base <= s < a.seed_base + max(a.seeds, 1) or not a.seeds}
    if not a.brief:
        report(list(games.values()))
    if a.compare:
        compare(games, load(Path(a.compare) / "games.jsonl"), a.compare)
