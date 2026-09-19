"""
Second tuning round of eat-rest-v1: a narrower search around the round-1 winner (keyword overrides only).

Differences from tune_v1.py, each one a lesson from that run:
  - the search is centred on --center (round 1's best.json), not on v1; every variant is the centre plus 2-3 changed settings
  - the space drops the settings that turned out dead or inert in AppetitePolicy and adds the live ones round 1 never searched
  - --combos adds hand-built configs (round-1 winners merged), which random search would almost never hit
  - fewer configs on more seeds per stage: the remaining gains are smaller, so the noise has to be lower to see them
  - ranks on the competition score (--metric), stores the whole game row, and shows the worst game next to the mean
Paired against the centre on the same seeds ("base" = untouched v1 rides along as a second reference). Fresh seed blocks
(8000/9000/10000; round 1 used 5000/6000/7000). Rule of thumb: adopt only if delta - 2*SE > 0 on the final block.

Resumable like tune_v1: finished games are appended to <out>/games.jsonl; rerunning the same command skips them.
    python external/candidates/tune_v2.py --out logs/tune_v2 --center logs/tune_v1/best.json --combos external/candidates/tune_v2_combos.json
    python external/candidates/tune_v2.py --out logs/tune2_smoke --center logs/tune_v1/best.json --configs 1 --seeds1 1 --keep2 1 --seeds2 1 --keep3 1 --seeds3 1
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
from tune_v1 import CANDIDATE, load_games  # noqa: E402

# (low, high); ints are sampled as ints. birth_floor follows birth_energy (see sample()). v1 ships the value in the comment.
SPACE = {
    "rest_radius": (45., 75.),              # 45   round-1 winners sat at 50-68, old ceiling was 70
    "dispersal_distance": (90., 170.),      # 100  winners 107-138
    "dispersal_time": (4., 10.),            # 6
    "harvest_switch": (3., 11.),            # 7    winners on both sides
    "harvest_depth": (1, 3),                # 2    unsearched in round 1
    "harvest_range": (150., 220.),          # 180
    "appetite_level": (200., 320.),         # 250
    "appetite_age_slope": (-1.5, 1.5),      # 0    unsearched
    "birth_energy": (125., 190.),           # 135
    "population": (5, 7),                   # 6
    "population_decay": (800., 1100.),      # 900
    "maximum_birth_gap": (15., 35.),        # 20
    "birth_spacing": (0., 6.),              # 0    unsearched
    "threat_cooldown": (0., 3.),            # 0    unsearched
    "blocked_threat_distance": (45., 90.),  # 65   unsearched
    "retirement": (95., 118.),              # 105
}
REFERENCES = ("center", "base")  # never eliminated: every delta is against the centre, and v1 stays as the yardstick


def sample(rng, center):
    overrides = dict(center)
    for key in rng.sample(sorted(SPACE), rng.choice((2, 3))):
        low, high = SPACE[key]
        overrides[key] = rng.randint(low, high) if isinstance(low, int) else round(rng.uniform(low, high), 3)
    if "birth_energy" in overrides:  # keep the floor below the normal threshold, as in the baseline (135 / 112)
        overrides["birth_floor"] = round(max(105., overrides["birth_energy"] - rng.uniform(15., 40.)), 3)
    return overrides


def run_stage(stage, names, configs, seeds, games, path, workers, metric):
    jobs = [(CANDIDATE, n, configs[n], s) for n in names for s in seeds if (n, s) not in games]
    print(f"\n=== stage {stage}: {len(names)} configs x {len(seeds)} seeds = {len(names) * len(seeds)} games "
          f"({len(jobs)} to run) ===", flush=True)
    t0, done = time.perf_counter(), 0
    with ProcessPoolExecutor(max_workers=workers) as pool, path.open("a") as log:
        for future in as_completed([pool.submit(run_one, j) for j in jobs]):
            row, _ = future.result()
            game = dict(stage=stage, name=row.pop("variant"), **row)
            games[(game["name"], game["seed"])] = game
            log.write(json.dumps(game) + "\n")
            log.flush()
            done += 1
            if done % 40 == 0 or done == len(jobs):
                print(f"  [{(time.perf_counter() - t0) / 60:5.1f} min] {done}/{len(jobs)} games", flush=True)
    return rank(names, seeds, games, metric)


def rank(names, seeds, games, metric):
    table = []
    for n in names:
        values = [games[(n, s)][metric] for s in seeds]
        delta = [games[(n, s)][metric] - games[("center", s)][metric] for s in seeds]
        se = statistics.stdev(delta) / len(delta) ** .5 if len(delta) > 1 else float("nan")
        table.append(dict(name=n, mean=statistics.mean(values), worst=min(values), delta=statistics.mean(delta), se=se,
                          better=sum(d > 0 for d in delta), eaten=statistics.mean(games[(n, s)]["eaten"] for s in seeds)))
    return sorted(table, key=lambda r: -r["delta"])


def show(table, configs, top):
    print(f"{'config':<8}{'mean':>7}{'worst':>7}{'delta':>8}{'SE':>6}{'better':>8}{'eaten':>7}  settings (delta is vs center)")
    references = [r for r in table[top:] if r["name"] in REFERENCES]  # always visible, wherever they rank
    for r in table[:top] + references:
        print(f"{r['name']:<8}{r['mean']:>7.0f}{r['worst']:>7.0f}{r['delta']:>+8.0f}{r['se']:>6.0f}{r['better']:>8}{r['eaten']:>7.1f}  "
              f"{configs[r['name']]}", flush=True)


def survivors(table, keep):
    return list(REFERENCES) + [r["name"] for r in table if r["name"] not in REFERENCES][:keep]


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--center", help="best.json of the previous round (its 'overrides' become the centre); omit to centre on v1")
    p.add_argument("--combos", help="json file {name: overrides} of hand-built configs to enter in stage 1")
    p.add_argument("--metric", default="score", choices=("score", "extinction_time"))
    p.add_argument("--configs", type=int, default=40)
    p.add_argument("--seeds1", type=int, default=16)
    p.add_argument("--keep2", type=int, default=8)
    p.add_argument("--seeds2", type=int, default=32)
    p.add_argument("--keep3", type=int, default=3)
    p.add_argument("--seeds3", type=int, default=64)
    p.add_argument("--seed-base", type=int, default=8000, help="blocks A/B/C start at base, base+1000, base+2000")
    p.add_argument("--search-seed", type=int, default=20260920)
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    config_path = out / "configs.json"
    if config_path.exists():
        configs = json.loads(config_path.read_text())
    else:
        center = json.loads(Path(a.center).read_text())["overrides"] if a.center else {}
        combos = json.loads(Path(a.combos).read_text()) if a.combos else {}
        clash = set(combos) & set(REFERENCES)
        if clash:
            raise ValueError(f"--combos may not use the reserved names {sorted(clash)}")
        rng = random.Random(a.search_seed)
        configs = {"base": {}, "center": center, **combos, **{f"d{i:03d}": sample(rng, center) for i in range(a.configs)}}
        config_path.write_text(json.dumps(configs, indent=1))
    games_path = out / "games.jsonl"
    games = load_games(games_path)
    blocks = [list(range(a.seed_base + 1000 * k, a.seed_base + 1000 * k + n)) for k, n in enumerate((a.seeds1, a.seeds2, a.seeds3))]

    table = run_stage(1, list(configs), configs, blocks[0], games, games_path, a.workers, a.metric)
    show(table, configs, 15)
    table = run_stage(2, survivors(table, a.keep2), configs, blocks[1], games, games_path, a.workers, a.metric)
    show(table, configs, 15)
    table = run_stage(3, survivors(table, a.keep3), configs, blocks[2], games, games_path, a.workers, a.metric)
    print(f"\n=== FINAL (fresh seeds; the only numbers to trust), metric = {a.metric} ===")
    show(table, configs, 10)
    best = next(r for r in table if r["name"] not in REFERENCES)
    verdict = "ADOPT candidate" if best["delta"] - 2 * best["se"] > 0 else "NOT proven better than the center - keep the center"
    print(f"\nbest: {best['name']} delta {best['delta']:+.0f} vs center, SE {best['se']:.0f}  ->  {verdict}")
    (out / "best.json").write_text(json.dumps(dict(name=best["name"], overrides=configs[best["name"]], metric=a.metric,
                                                   delta=best["delta"], se=best["se"], seeds=len(blocks[2]), verdict=verdict), indent=1))
    print(f"wrote {out / 'best.json'}")
