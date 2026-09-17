"""
Scored multi-seed evaluation. Run as a fresh process (never inside a long-lived kernel):

    python -m training.evaluate --experiment C1-E00 --config training/configs/dummy_v0.json \
        --seeds 0 1 2 3 4 --workers 4 --max-wall-sec 1800

Writes results/<run_id>.csv (one row per seed) and appends one row to results/index.csv.
"""
import argparse
import csv
import datetime
import hashlib
import json
import platform
import statistics
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INDEX = RESULTS / "index.csv"
INDEX_FIELDS = [
    "experiment_id", "run_id", "date", "agent", "config_hash", "seed",
    "mean_raw_score", "std_raw_score", "median_raw_score", "min_raw_score", "max_raw_score",
    "mean_survival_steps", "illegal_moves", "override_count", "wall_clock_sec", "status", "notes",
]


def config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]


def git_revision() -> str:
    try:
        rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--", "."], cwd=ROOT, text=True).strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def _worker(job):
    config, seed, max_wall_sec = job
    from agents import make_agent
    from training.episode import run_episode
    return run_episode(make_agent(config), seed, max_wall_sec=max_wall_sec)


def evaluate(experiment_id: str, config_path: str, seeds, workers: int = 1, max_wall_sec: float = None,
             notes: str = "") -> dict:
    config = json.loads(Path(config_path).read_text())
    chash = config_hash(config)
    rev = git_revision()
    run_id = f"{experiment_id}_{config['name']}_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    RESULTS.mkdir(exist_ok=True)
    if rev.endswith("-dirty"):
        print(f"WARNING: uncommitted changes (revision {rev}); commit before scored runs.", file=sys.stderr)

    print(f"run_id={run_id} config={config_path} hash={chash} rev={rev} seeds={list(seeds)} workers={workers}")
    t0 = time.time()
    jobs = [(config, s, max_wall_sec) for s in seeds]
    rows = []
    if workers > 1:
        with Pool(workers) as pool:
            for r in pool.imap_unordered(_worker, jobs):
                print(f"  seed={r['seed']} score={r['score']:.2f} sim_time={r['sim_time']:.1f} "
                      f"peak_agents={r['peak_agents']} status={r['status']} wall={r['wall_clock_sec']:.0f}s", flush=True)
                rows.append(r)
    else:
        for job in jobs:
            r = _worker(job)
            print(f"  seed={r['seed']} score={r['score']:.2f} sim_time={r['sim_time']:.1f} "
                  f"peak_agents={r['peak_agents']} status={r['status']} wall={r['wall_clock_sec']:.0f}s", flush=True)
            rows.append(r)
    wall = time.time() - t0
    rows.sort(key=lambda r: r["seed"])

    with open(RESULTS / f"{run_id}.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    scores = [r["score"] for r in rows]
    statuses = {r["status"] for r in rows}
    summary = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "date": datetime.date.today().isoformat(),
        "agent": config["name"],
        "config_hash": chash,
        "seed": " ".join(str(s) for s in seeds),
        "mean_raw_score": round(statistics.mean(scores), 4),
        "std_raw_score": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
        "median_raw_score": round(statistics.median(scores), 4),
        "min_raw_score": round(min(scores), 4),
        "max_raw_score": round(max(scores), 4),
        "mean_survival_steps": round(statistics.mean(r["steps"] for r in rows), 1),
        "illegal_moves": 0,  # ActionRequest validation raises on any illegal action, so a finished run has 0
        "override_count": 0,  # no override/emergency layer exists yet
        "wall_clock_sec": round(wall, 1),
        "status": "ok" if statuses == {"ok"} else "+".join(sorted(statuses)),
        "notes": f"rev={rev}; os={platform.system()}; config={config_path}; {notes}".strip("; "),
    }
    new_index = not INDEX.exists()
    with open(INDEX, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_FIELDS)
        if new_index:
            w.writeheader()
        w.writerow(summary)

    print(f"mean={summary['mean_raw_score']} std={summary['std_raw_score']} median={summary['median_raw_score']} "
          f"min={summary['min_raw_score']} max={summary['max_raw_score']} n={len(scores)} wall={wall:.0f}s")
    print(f"wrote {RESULTS / (run_id + '.csv')} and appended {INDEX}")
    return {"summary": summary, "rows": rows}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-wall-sec", type=float, default=None, help="per-game wall-clock cap")
    p.add_argument("--notes", default="")
    a = p.parse_args()
    evaluate(a.experiment, a.config, a.seeds, a.workers, a.max_wall_sec, a.notes)


if __name__ == "__main__":
    main()
