"""
Noise-aware parameter tuning of eat-rest-v1 WITHOUT modifying it (keyword overrides only).

Local random search around the proven baseline + successive halving on fresh seeds:
  stage 1: baseline + N configs, each perturbing 2-4 settings, on seed block A     (wide and cheap)
  stage 2: the best K of those + baseline on seed block B                          (new seeds)
  stage 3: the best few + baseline on seed block C                                 (new seeds, final estimate)
Every comparison is paired with the baseline on the same seeds. Stage winners are picked on one block
and re-measured on the next, so a lucky config falls back to its true level; only the stage-3 number
(mean paired delta and its standard error) is an honest estimate, and even it is a max over a few configs.
Rule of thumb: adopt only if delta - 2*SE > 0.

Resumable: every finished game is appended to <out>/games.jsonl; rerunning the same command skips them.
    python external/candidates/tune_v1.py --out logs/tune_v1            # full run (~1,200 games)
    python external/candidates/tune_v1.py --out logs/tune_smoke --configs 3 --seeds1 1 --keep2 2 --seeds2 1 --keep3 1 --seeds3 1
Mechanism study: nothing is written to results/.
"""
import argparse
import json
import os
import random
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sweep_config import run_one  # noqa: E402  (same headless game loop + post-mortem)

CANDIDATE = "original-eat-rest-preserved"
# (low, high) around the shipped value; ints are sampled as ints. birth_floor follows birth_energy (see sample()).
SPACE = {
    "appetite_level": (200., 320.), "appetite_release": (40., 110.), "appetite_near": (50., 120.),
    "birth_energy": (120., 260.), "maximum_birth_gap": (12., 35.),
    "population": (5, 7), "population_decay": (700., 1100.),
    "escape_radius": (85., 160.), "rest_radius": (35., 70.), "crowd_margin": (35., 75.), "sprint_radius": (45., 80.),
    "search_speed": (.3, .6), "scan": (.1, .35),
    "retirement": (90., 120.), "young_age": (55., 75.), "renewal_age": (60., 90.),
    "emergency_birth_distance": (45., 80.), "emergency_fraction": (.3, .5),
    "dispersal_distance": (60., 160.), "dispersal_time": (4., 10.),
    "harvest_range": (140., 240.), "harvest_food_value": (30., 55.), "harvest_switch": (4., 10.), "harvest_discount": (.55, .85),
    "idle_energy": (.75, .95), "hard_idle_energy": (220., 320.), "reserve": (60., 110.),
}


def sample(rng):
    overrides = {}
    for key in rng.sample(sorted(SPACE), rng.choice((2, 3, 4))):
        low, high = SPACE[key]
        overrides[key] = rng.randint(low, high) if isinstance(low, int) else round(rng.uniform(low, high), 3)
    if "birth_energy" in overrides:  # keep the floor below the normal threshold, as in the baseline (135 / 112)
        overrides["birth_floor"] = round(max(105., overrides["birth_energy"] - rng.uniform(15., 40.)), 3)
    return overrides


def load_games(path):
    games = {}
    if path.exists():
        for line in path.read_text().splitlines():
            g = json.loads(line)
            games[(g["name"], g["seed"])] = g
    return games


def run_stage(stage, names, configs, seeds, games, path, workers):
    jobs = [(CANDIDATE, n, configs[n], s) for n in names for s in seeds if (n, s) not in games]
    print(f"\n=== stage {stage}: {len(names)} configs x {len(seeds)} seeds = {len(names) * len(seeds)} games "
          f"({len(jobs)} to run) ===", flush=True)
    t0, done = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as log:
        for future in as_completed([pool.submit(run_one, j) for j in jobs]):
            row, _ = future.result()
            game = dict(stage=stage, name=row["variant"], seed=row["seed"], t=row["extinction_time"], eaten=row["eaten"],
                        old_age=row["old_age"], newborn_starved=row["newborn_starved"], starved=row["starved"], births=row["births"])
            games[(game["name"], game["seed"])] = game
            log.write(json.dumps(game) + "\n")
            log.flush()
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"  [{(time.perf_counter() - t0) / 60:5.1f} min] {done}/{len(jobs)} games", flush=True)
    return rank(names, seeds, games)


def rank(names, seeds, games):
    table = []
    for n in names:
        delta = [games[(n, s)]["t"] - games[("base", s)]["t"] for s in seeds]
        se = statistics.stdev(delta) / len(delta) ** .5 if len(delta) > 1 else float("nan")
        table.append(dict(name=n, mean=statistics.mean(games[(n, s)]["t"] for s in seeds), delta=statistics.mean(delta), se=se,
                          better=sum(d > 0 for d in delta), eaten=statistics.mean(games[(n, s)]["eaten"] for s in seeds)))
    return sorted(table, key=lambda r: -r["delta"])


def show(table, configs, top):
    print(f"{'config':<8}{'mean':>7}{'delta':>8}{'SE':>6}{'better':>8}{'eaten':>7}  settings")
    for r in table[:top]:
        print(f"{r['name']:<8}{r['mean']:>7.0f}{r['delta']:>+8.0f}{r['se']:>6.0f}{r['better']:>8}{r['eaten']:>7.1f}  {configs[r['name']]}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--configs", type=int, default=80)
    p.add_argument("--seeds1", type=int, default=8)
    p.add_argument("--keep2", type=int, default=12)
    p.add_argument("--seeds2", type=int, default=24)
    p.add_argument("--keep3", type=int, default=3)
    p.add_argument("--seeds3", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=5000, help="blocks A/B/C start at base, base+1000, base+2000")
    p.add_argument("--search-seed", type=int, default=20260919)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "configs.json"
    if config_path.exists():
        configs = json.loads(config_path.read_text())
    else:
        rng = random.Random(a.search_seed)
        configs = {"base": {}, **{f"c{i:03d}": sample(rng) for i in range(a.configs)}}
        config_path.write_text(json.dumps(configs, indent=1))
    games_path = out / "games.jsonl"
    games = load_games(games_path)
    blocks = [list(range(a.seed_base + 1000 * k, a.seed_base + 1000 * k + n)) for k, n in enumerate((a.seeds1, a.seeds2, a.seeds3))]

    names = list(configs)
    table = run_stage(1, names, configs, blocks[0], games, games_path, a.workers)
    show(table, configs, 15)
    names = ["base"] + [r["name"] for r in table if r["name"] != "base"][:a.keep2]
    table = run_stage(2, names, configs, blocks[1], games, games_path, a.workers)
    show(table, configs, 15)
    names = ["base"] + [r["name"] for r in table if r["name"] != "base"][:a.keep3]
    table = run_stage(3, names, configs, blocks[2], games, games_path, a.workers)
    print("\n=== FINAL (fresh seeds; the only numbers to trust) ===")
    show(table, configs, 10)
    best = next(r for r in table if r["name"] != "base")
    verdict = "ADOPT candidate" if best["delta"] - 2 * best["se"] > 0 else "NOT proven better than the baseline - keep v1"
    print(f"\nbest: {best['name']} delta {best['delta']:+.0f} s, SE {best['se']:.0f}  ->  {verdict}")
    (out / "best.json").write_text(json.dumps(dict(name=best["name"], overrides=configs[best["name"]], delta=best["delta"], se=best["se"],
                                                   seeds=len(blocks[2]), verdict=verdict), indent=1))
    print(f"wrote {out / 'best.json'}")
