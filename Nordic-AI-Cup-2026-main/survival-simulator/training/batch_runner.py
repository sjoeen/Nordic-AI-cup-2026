"""
Batch runner: full headless games for a policy/config across a list of seeds.

Depends only on the starter simulator (src/) and the policy (agents/) -- no imports
from experiments/ or the predator_mechanics harnesses. The headless game loop below is
copied to the minimum needed rather than importing training/episode.py.

Robustness (overnight task): each game's CSV row is written the moment that game ends
(not batched at the end), a companion .log file gets one line per game plus a start/end
banner, a .DONE marker file is written only once every requested seed has a row in the
CSV, and --resume skips seeds already present in an existing CSV so a killed/relaunched
run picks up where it left off.

Usage:
    python -m training.batch_runner --config training/configs/heuristic_v0.json \
        --seeds 0 1 2 3 4 5 6 7 8 9 [--predators-off] [--workers N] [--out results/foo.csv] \
        [--resume] [--diagnostics]
"""
import argparse
import csv
import datetime
import multiprocessing as mp
import os
import platform
import statistics
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MAX_SIM_TIME = 3000  # matches the official evaluator loop (training/episode.py)

CSV_FIELDS = [
    "seed", "extinction_time", "score", "peak_population", "end_reason", "wall_clock_sec",
    "policy_time_mean_ms", "policy_time_max_ms", "exceptions_caught",
    "deaths_predation", "deaths_energy",
    "share_fleeing", "share_travelling", "share_waiting",
    "predation_below_sprint_share", "predation_unseen_killer_share",
    "mean_fruit_energy_eaten", "births", "tree_occ_avg",
    "rule_fired_shares",  # e.g. "r1:0.0812 r3:0.0210" -- overnight task, heuristic_night only
]

TREE_OCC_RADIUS = 60.0  # matches Rule 1's "agents at the tree" radius; diagnostic-only, ground truth
ENERGY_DEATH_THRESHOLD = 1.0  # see module docstring / overnight_log.md: approximation, not exact engine truth


def _run_one(job):
    """Runs one full headless game. Import here so worker processes (Windows: spawn) pick it up cleanly."""
    config, seed, predators_off, diagnostics = job
    from agents import make_agent
    from src.core import SimulationCore
    from src.utils.DTOs import ActionRequest
    import numpy as np

    agent = make_agent(config)
    t0 = time.perf_counter()
    sim = SimulationCore(seed=seed, starting_predators=0)
    if predators_off:
        # Override inside the runner: no dynamic predator spawning either. src/ is untouched.
        sim.env.spawn_predator = lambda *a, **kw: None

    policy_times_sum = 0.0
    policy_times_max = 0.0
    policy_times_n = 0

    travel_steps = fleeing_steps = total_agent_steps = 0
    tree_occ_sum, tree_occ_count = 0.0, 0
    deaths_predation = deaths_energy = 0
    predation_below_sprint = predation_unseen_killer = 0
    fruit_energy_sum, fruit_events = 0.0, 0
    births = 0

    state = sim.step([])
    peak_population = state["num_agents"]
    next_progress_mark = 100
    prev_by_id = {}  # agent_id -> observation dict, from the step immediately before a death

    while state["num_agents"] > 0 and sim.env.time <= MAX_SIM_TIME:
        agent_states = [o for o in state["observations"] if o is not None]
        by_id = {s["agent_id"]: s for s in agent_states}

        t_policy = time.perf_counter()
        actions = agent.act_batch(agent_states)
        dt_policy_ms = (time.perf_counter() - t_policy) * 1000
        policy_times_sum += dt_policy_ms
        policy_times_max = max(policy_times_max, dt_policy_ms)
        policy_times_n += 1

        if diagnostics:
            mem = getattr(agent, "memory", None)
            any_predator = False
            for st in agent_states:
                total_agent_steps += 1
                has_predator = any(o.get("type") == "Predator" for o in st["observations"])
                has_tree = any(o.get("type") == "Tree" for o in st["observations"])
                leaving = False
                if mem is not None:
                    mm = mem.get(st["agent_id"])
                    leaving = bool(mm and mm.get("mode") == "LEAVING")
                if has_predator:
                    fleeing_steps += 1
                elif leaving or not has_tree:
                    travel_steps += 1
            for a in actions:
                if a.get("spawn_agent") and by_id.get(a["agent_id"], {}).get("energy", 0) > 100:
                    births += 1

        parsed = [(a["agent_id"], ActionRequest(**a)) for a in actions]  # raises on illegal actions
        state = sim.step(parsed)
        peak_population = max(peak_population, state["num_agents"])

        if diagnostics:
            current_ids = {o["agent_id"] for o in state["observations"] if o is not None}
            for dead_id in (set(by_id) - current_ids):
                s = by_id[dead_id]
                last_energy = s["energy"]
                last_max_energy = s["max_energy"]
                saw_predator = any(o.get("type") == "Predator" for o in s["observations"])
                if last_energy <= ENERGY_DEATH_THRESHOLD:
                    deaths_energy += 1
                else:
                    deaths_predation += 1
                    if last_energy < last_max_energy / 5:
                        predation_below_sprint += 1
                    if not saw_predator:
                        predation_unseen_killer += 1
            for o in state["observations"]:
                if o is None:
                    continue
                prev = by_id.get(o["agent_id"])
                if prev is not None and o["energy"] > prev["energy"]:
                    fruit_energy_sum += (o["energy"] - prev["energy"])
                    fruit_events += 1
            if sim.env.trees and sim.env.agents:
                tx = np.array([t.x for t in sim.env.trees]); ty = np.array([t.y for t in sim.env.trees])
                ax_ = np.array([a.x for a in sim.env.agents]); ay_ = np.array([a.y for a in sim.env.agents])
                d = np.hypot(tx[:, None] - ax_[None, :], ty[:, None] - ay_[None, :])
                counts = (d <= TREE_OCC_RADIUS).sum(axis=1)
                occ = counts[counts >= 1]
                tree_occ_sum += float(occ.sum())
                tree_occ_count += int(occ.size)

        if sim.env.time >= next_progress_mark:
            print(f"    [progress] seed={seed} sim_time={sim.env.time:.1f} "
                  f"population={state['num_agents']} score={state['score']:.2f}", flush=True)
            next_progress_mark += 100

    end_reason = "extinction" if state["num_agents"] == 0 else "time_limit"

    row = {
        "seed": seed,
        "extinction_time": round(sim.env.time, 2),
        "score": round(float(state["score"]), 4),
        "peak_population": peak_population,
        "end_reason": end_reason,
        "wall_clock_sec": round(time.perf_counter() - t0, 2),
        "policy_time_mean_ms": round(policy_times_sum / policy_times_n, 4) if policy_times_n else 0.0,
        "policy_time_max_ms": round(policy_times_max, 4),
        "exceptions_caught": getattr(agent, "exceptions_caught", 0),
    }
    if diagnostics:
        total_steps = total_agent_steps
        waiting_steps = total_steps - fleeing_steps - travel_steps
        n_deaths = deaths_predation + deaths_energy
        row.update(
            deaths_predation=deaths_predation, deaths_energy=deaths_energy,
            share_fleeing=round(fleeing_steps / total_steps, 4) if total_steps else 0.0,
            share_travelling=round(travel_steps / total_steps, 4) if total_steps else 0.0,
            share_waiting=round(waiting_steps / total_steps, 4) if total_steps else 0.0,
            predation_below_sprint_share=round(predation_below_sprint / deaths_predation, 4) if deaths_predation else 0.0,
            predation_unseen_killer_share=round(predation_unseen_killer / deaths_predation, 4) if deaths_predation else 0.0,
            mean_fruit_energy_eaten=round(fruit_energy_sum / fruit_events, 4) if fruit_events else 0.0,
            births=births,
            tree_occ_avg=round(tree_occ_sum / tree_occ_count, 4) if tree_occ_count else 0.0,
            # raw aggregates too, so the caller can pool correctly across seeds rather than averaging averages
            _total_agent_steps=total_steps, _fleeing_steps=fleeing_steps, _travel_steps=travel_steps,
            _tree_occ_sum=tree_occ_sum, _tree_occ_count=tree_occ_count,
        )
        rule_fired = getattr(agent, "rule_fired", None)
        agent_total_steps = getattr(agent, "total_agent_steps", 0)
        if rule_fired and agent_total_steps:
            row["rule_fired_shares"] = " ".join(
                f"{k}:{v / agent_total_steps:.4f}" for k, v in rule_fired.items() if v
            )
        else:
            row["rule_fired_shares"] = ""
    return row


def environment_info() -> str:
    from importlib.metadata import version
    pkgs = " ".join(f"{p}={version(p)}" for p in ("numpy", "scipy", "shapely", "pygame", "pydantic"))
    return f"python={platform.python_version()} {pkgs} os={platform.system()}"


def _read_existing_seeds(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {int(row["seed"]) for row in csv.DictReader(f)}


def run_batch(config: dict, seeds, predators_off: bool = False, workers: int = None,
              out_path: Path = None, diagnostics: bool = False, resume: bool = False,
              log_path: Path = None) -> list:
    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)  # leave one CPU core free
    seeds = list(seeds)

    log_f = None
    if out_path is not None:
        log_path = log_path or out_path.with_suffix(".log")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_f = open(log_path, "a")

    def log(msg):
        print(msg, flush=True)
        if log_f:
            log_f.write(msg + "\n")
            log_f.flush()

    log(f"=== batch start {datetime.datetime.now().isoformat()} ===")
    log(environment_info())  # recorded once per batch, not per game
    log(f"config={config.get('name')} seeds={seeds} predators_off={predators_off} workers={workers} "
        f"diagnostics={diagnostics} resume={resume} out={out_path}")

    already = set()
    if resume and out_path is not None:
        already = _read_existing_seeds(out_path)
        if already:
            log(f"resume: {len(already)} seeds already in {out_path}: {sorted(already)}")
    todo_seeds = [s for s in seeds if s not in already]

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not out_path.exists()
        csv_f = open(out_path, "a", newline="")
        writer = csv.DictWriter(csv_f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if new_file:
            writer.writeheader()
            csv_f.flush()
    else:
        csv_f = writer = None

    rows = []
    # Load rows already on disk too, so the returned/aggregated stats cover the full seed set.
    if already:
        with open(out_path, newline="") as f:
            for r in csv.DictReader(f):
                if int(r["seed"]) in already:
                    rows.append({k: (float(v) if k not in ("seed", "end_reason") else
                                      (int(v) if k == "seed" else v)) for k, v in r.items()})

    if todo_seeds:
        jobs = [(config, s, predators_off, diagnostics) for s in todo_seeds]
        t0 = time.time()
        if workers > 1 and len(jobs) > 1:
            with mp.Pool(workers) as pool:
                for r in pool.imap_unordered(_run_one, jobs):
                    rows.append(r)
                    log(f"  seed={r['seed']} score={r['score']:.2f} extinction_time={r['extinction_time']:.1f} "
                        f"peak_population={r['peak_population']} end_reason={r['end_reason']} "
                        f"wall={r['wall_clock_sec']:.1f}s policy_ms(mean/max)={r['policy_time_mean_ms']:.2f}/"
                        f"{r['policy_time_max_ms']:.2f}")
                    if writer:
                        writer.writerow(r)
                        csv_f.flush()
        else:
            for j in jobs:
                r = _run_one(j)
                rows.append(r)
                log(f"  seed={r['seed']} score={r['score']:.2f} extinction_time={r['extinction_time']:.1f} "
                    f"peak_population={r['peak_population']} end_reason={r['end_reason']} "
                    f"wall={r['wall_clock_sec']:.1f}s policy_ms(mean/max)={r['policy_time_mean_ms']:.2f}/"
                    f"{r['policy_time_max_ms']:.2f}")
                if writer:
                    writer.writerow(r)
                    csv_f.flush()
        batch_wall_sec = time.time() - t0
    else:
        batch_wall_sec = 0.0
        log("nothing to do: all requested seeds already present")

    if csv_f:
        csv_f.close()
    rows.sort(key=lambda r: r["seed"])

    scores = [r["score"] for r in rows]
    log(f"mean={statistics.mean(scores):.4f} median={statistics.median(scores):.4f} "
        f"worst={min(scores):.4f} n={len(scores)} batch_wall_sec={batch_wall_sec:.1f}")

    if diagnostics:
        total_steps = sum(r.get("_total_agent_steps", 0) for r in rows)
        total_fleeing = sum(r.get("_fleeing_steps", 0) for r in rows)
        total_travel = sum(r.get("_travel_steps", 0) for r in rows)
        total_occ_sum = sum(r.get("_tree_occ_sum", 0.0) for r in rows)
        total_occ_count = sum(r.get("_tree_occ_count", 0) for r in rows)
        total_pred = sum(r.get("deaths_predation", 0) for r in rows)
        total_energy_deaths = sum(r.get("deaths_energy", 0) for r in rows)
        total_births = sum(r.get("births", 0) for r in rows)
        log(f"diagnostics(pooled): share_fleeing={total_fleeing / total_steps:.4f} "
            f"share_travelling={total_travel / total_steps:.4f} "
            f"share_waiting={(total_steps - total_fleeing - total_travel) / total_steps:.4f} "
            f"tree_occ_avg={(total_occ_sum / total_occ_count) if total_occ_count else 0.0:.4f} "
            f"deaths_predation={total_pred} deaths_energy={total_energy_deaths} births={total_births}"
            if total_steps else "diagnostics(pooled): no agent-steps recorded")

    if out_path is not None:
        log(f"wrote {out_path}")
        done_path = out_path.with_suffix(".DONE")
        if set(seeds) <= {r["seed"] for r in rows}:
            done_path.write_text(f"done {datetime.datetime.now().isoformat()} n={len(rows)}\n")
            log(f"DONE marker: {done_path}")
    log(f"=== batch end {datetime.datetime.now().isoformat()} ===")
    if log_f:
        log_f.close()
    return rows


def default_out_path(config: dict, predators_off: bool) -> Path:
    tag = "noPredators" if predators_off else "predators"
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    return RESULTS / f"BATCH_{config.get('name', 'agent')}_{tag}_{stamp}.csv"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--predators-off", action="store_true", help="no initial predators, no predator spawning")
    p.add_argument("--workers", type=int, default=None, help="default: cpu_count - 1")
    p.add_argument("--out", default=None, help="CSV output path (default: results/BATCH_<name>_<tag>_<ts>.csv)")
    p.add_argument("--no-write", action="store_true", help="print stats only, do not write a CSV")
    p.add_argument("--diagnostics", action="store_true",
                    help="track deaths/cause, fleeing/travelling/waiting shares, fruit/birth stats, tree occupancy")
    p.add_argument("--resume", action="store_true", help="skip seeds already present in --out's CSV")
    a = p.parse_args()

    from agents import load_config
    config = load_config(a.config)
    if a.no_write:
        out_path = None
    else:
        out_path = Path(a.out) if a.out else default_out_path(config, a.predators_off)
    run_batch(config, a.seeds, predators_off=a.predators_off, workers=a.workers, out_path=out_path,
              diagnostics=a.diagnostics, resume=a.resume)


if __name__ == "__main__":
    main()
