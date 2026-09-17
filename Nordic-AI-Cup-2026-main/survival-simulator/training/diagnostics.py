"""
Behaviour diagnostics for a policy. Inspection only: nothing is written to results/.

    python -m training.diagnostics --config training/configs/heuristic_v0.json --seeds 0 1 2 3 4 --workers 4

Runs the same loop as training/episode.py, but instruments the (unmodified) simulator from
the outside to record:
  - deaths by cause (predator / starvation before max_age / starvation after max_age)
  - births, fruit eaten vs rotted (and how far the nearest agent was when a fruit rotted)
  - predator target choice per step: chase vs circle mode, target switches, multi-kills
  - agent behaviour: which heuristic rule fired, stuck moves, net displacement, crowding
  - agents skipped by the engine's list-mutation-during-iteration (age did not advance)
Writes logs/diagnostics_<config>_seed<N>.json.
"""
import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import training.episode  # noqa: F401  (headless SDL env vars)
import numpy as np

from agents import make_agent
from src.core import SimulationCore
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_EVERY = 100  # steps = 10 simulated seconds


def heuristic_branch(s, tree_wait_dist):
    types = {o["type"] for o in s["observations"]}
    if "Predator" in types:
        return "flee_predator"
    if "Fruit" in types:
        return "go_to_fruit"
    if "Tree" in types:
        t = min((o for o in s["observations"] if o["type"] == "Tree"), key=lambda o: o["distance"])
        return "approach_tree" if t["distance"] > tree_wait_dist else "wait_at_tree"
    return "wander"


def run(job):
    config, seed, max_sim_time = job
    agent = make_agent(config)
    tree_wait = config.get("params", {}).get("tree_wait_dist", 40.0)

    sim = SimulationCore(seed=seed)
    env = sim.env
    ev = defaultdict(list)          # event lists
    cnt = Counter()                 # counters
    step_idx = [0]

    # ---- deaths: distinguish caller line inside non_agent_step ----
    starve_line = pred_line = None
    import inspect
    src_lines, first = inspect.getsourcelines(type(env).non_agent_step)
    for i, line in enumerate(src_lines):
        if "self.kill_agent(agent)" in line:
            if starve_line is None:
                starve_line = first + i
            else:
                pred_line = first + i
    orig_kill = env.kill_agent

    def kill(a):
        if a in env.agents:
            fr = sys._getframe(1)
            if fr.f_lineno == pred_line:
                cause = "predator"
                ev["pred_kill"].append((step_idx[0], id(fr.f_locals.get("predator"))))
            elif a.age > a.max_age:
                cause = "starved_after_max_age"
            else:
                cause = "starved_before_max_age"
            ev["death"].append({"t": round(env.time, 1), "cause": cause, "age": round(a.age, 1),
                                "max_age": round(a.max_age, 1), "energy": round(a.energy, 1)})
        orig_kill(a)
    env.kill_agent = kill

    # ---- fruit removal: eaten vs rotted ----
    eaters_this_step = set()
    orig_remove_fruit = env.remove_fruit

    def remove_fruit(f):
        if f in env.fruits:
            if f.age > 100:
                d = min((math.hypot(a.x - f.x, a.y - f.y) for a in env.agents), default=float("inf"))
                ev["rot_nearest_agent"].append(round(d, 1))
            else:
                ev["eaten_energy"].append(round(f.energy, 1))
                eater = sys._getframe(1).f_locals.get("agent")
                if eater is not None:
                    eaters_this_step.add(eater.agent_id)
        orig_remove_fruit(f)
    env.remove_fruit = remove_fruit

    # ---- predator decisions ----
    last_target = {}

    def pred_step(self, observation=None):
        agents_seen = [o for o in observation if o.get("type") == "Agent"]
        if agents_seen:
            c = min(agents_seen, key=lambda o: o["distance"])
            mode = "chase" if (abs(c["rel_dir"]) > math.pi / 2 or c["distance"] < self.hearing_radius * 1.5) else "circle"
            cnt["pred_steps_" + mode] += 1
            prev = last_target.get(id(self))
            if prev is not None and prev != c["id"]:
                cnt["pred_target_switch"] += 1
                if any(o["id"] == prev for o in agents_seen):
                    cnt["pred_target_switch_prev_still_seen"] += 1
            last_target[id(self)] = c["id"]
            cnt["pred_agents_seen_multi"] += len(agents_seen) > 1
        else:
            cnt["pred_steps_no_agent"] += 1
            last_target.pop(id(self), None)
        return Predator._orig_step(self, observation)
    if not hasattr(Predator, "_orig_step"):
        Predator._orig_step = Predator.step
    Predator.step = pred_step

    state = sim.step([])
    step_idx[0] = 1
    seen_ids = {a.agent_id for a in env.agents}
    births = 0
    series = []
    last_pos = {}
    fruit_lost = 0
    fruit_chase_steps = 0

    while state["num_agents"] > 0 and env.time <= 3000 and env.time <= max_sim_time:
        states = [o for o in state["observations"] if o is not None]
        pre = {a.agent_id: (a.x, a.y, a.age) for a in env.agents}
        eaters_this_step.clear()
        actions = agent.act_batch(states)
        req = {a["agent_id"]: a["move_distance"] for a in actions}
        branches = {s["agent_id"]: heuristic_branch(s, tree_wait) for s in states}
        for s in states:
            b = branches[s["agent_id"]]
            cnt["branch_" + b] += 1
            if b == "flee_predator" and any(o["type"] == "Fruit" for o in s["observations"]):
                cnt["flee_while_fruit_seen"] += 1
            if s["energy"] > s["max_energy"] * 0.99:
                cnt["agent_steps_at_max_energy"] += 1

        state = sim.step([(a["agent_id"], ActionRequest(**a)) for a in actions])
        step_idx[0] += 1

        post = {a.agent_id: a for a in env.agents}
        for aid, (x, y, age) in pre.items():
            a = post.get(aid)
            if a is None:
                continue
            if a.age - age < 0.05:
                cnt["agent_updates_skipped_by_engine"] += 1
            if req.get(aid, 0) >= 1.0:
                cnt["move_requests"] += 1
                if math.hypot(a.x - x, a.y - y) < 1e-6:
                    cnt["move_requests_no_motion"] += 1
        new_ids = set(post) - seen_ids
        births += len(new_ids)
        seen_ids |= new_ids

        # fruit chase lost: agent was going to fruit, its next observation has no fruit, and it ate nothing
        next_states = {o["agent_id"]: o for o in state["observations"] if o is not None}
        for aid, b in branches.items():
            if b == "go_to_fruit":
                fruit_chase_steps += 1
                ns = next_states.get(aid)
                if ns is not None and not any(o["type"] == "Fruit" for o in ns["observations"]) and aid not in eaters_this_step:
                    fruit_lost += 1

        if step_idx[0] % SAMPLE_EVERY == 0:
            ags = env.agents
            pos = np.array([[a.x, a.y] for a in ags]) if ags else np.zeros((0, 2))
            nn = None
            near15 = near50 = 0.0
            if len(ags) > 1:
                d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 1] - pos[None, :, 1])
                np.fill_diagonal(d, np.inf)
                m = d.min(axis=1)
                nn = float(np.median(m))
                near15 = float((m < 15).mean())
                near50 = float((m < 50).mean())
            disp = [math.hypot(a.x - last_pos[a.agent_id][0], a.y - last_pos[a.agent_id][1])
                    for a in ags if a.agent_id in last_pos]
            last_pos = {a.agent_id: (a.x, a.y) for a in ags}
            causes = Counter(d["cause"] for d in ev["death"])
            series.append({
                "t": round(env.time, 1), "agents": len(ags), "births_cum": births,
                "deaths_predator_cum": causes["predator"],
                "deaths_starved_before_max_age_cum": causes["starved_before_max_age"],
                "deaths_starved_after_max_age_cum": causes["starved_after_max_age"],
                "predators": len(env.predators), "predators_resting": sum(p.resting for p in env.predators),
                "trees": len(env.trees), "trees_fruiting_age": sum(t.age >= 20 for t in env.trees),
                "fruits": len(env.fruits), "fruit_eaten_cum": len(ev["eaten_energy"]),
                "fruit_rotted_cum": len(ev["rot_nearest_agent"]),
                "mean_energy": round(float(np.mean([a.energy for a in ags])), 1) if ags else None,
                "frac_past_max_age": round(float(np.mean([a.age > a.max_age for a in ags])), 3) if ags else None,
                "median_nn_dist": round(nn, 1) if nn is not None else None,
                "frac_nn_under_15": round(near15, 3), "frac_nn_under_50": round(near50, 3),
                "median_disp_10s": round(float(np.median(disp)), 1) if disp else None,
            })

    Predator.step = Predator._orig_step
    pk = Counter(ev["pred_kill"])
    rot = ev["rot_nearest_agent"]
    deaths = ev["death"]
    by_cause = Counter(d["cause"] for d in deaths)
    out = {
        "seed": seed, "config": config["name"], "end_time": round(env.time, 1), "score": round(env.score, 2),
        "survived": state["num_agents"] > 0, "births": births, "deaths_by_cause": dict(by_cause),
        "median_age_at_death_by_cause": {c: float(np.median([d["age"] for d in deaths if d["cause"] == c]))
                                         for c in by_cause},
        "predator_kills": len(ev["pred_kill"]),
        "predator_steps_with_2plus_kills_same_predator": sum(1 for v in pk.values() if v >= 2),
        "max_kills_one_predator_one_step": max(pk.values(), default=0),
        "fruit_eaten": len(ev["eaten_energy"]), "fruit_rotted": len(rot),
        "median_energy_per_fruit_eaten": float(np.median(ev["eaten_energy"])) if ev["eaten_energy"] else None,
        "rotted_fruit_nearest_agent_within_50": sum(d <= 50 for d in rot),
        "rotted_fruit_nearest_agent_within_200": sum(d <= 200 for d in rot),
        "fruit_chase_steps": fruit_chase_steps, "fruit_chase_lost_sight_no_eat": fruit_lost,
        "counters": dict(cnt), "series": series,
    }
    path = ROOT / "logs" / f"diagnostics_{config['name']}_seed{seed}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"seed={seed} end={out['end_time']} score={out['score']} births={births} deaths={dict(by_cause)} "
          f"fruit eaten/rotted={out['fruit_eaten']}/{out['fruit_rotted']} -> {path}", flush=True)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-sim-time", type=float, default=3000)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    jobs = [(config, s, a.max_sim_time) for s in a.seeds]
    if a.workers > 1:
        with Pool(a.workers) as pool:
            list(pool.imap_unordered(run, jobs))
    else:
        for j in jobs:
            run(j)


if __name__ == "__main__":
    main()
