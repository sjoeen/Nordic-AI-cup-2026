"""
C1-MECH-P01: predator facing / retreat mechanism study (diagnostic harness, not a policy).

One agent + one awake predator in an empty uniform-grassland arena, stepped with the real
engine (src.utils.simulation.step_environment -> Environment.agent_step / non_agent_step).

Fixture deviations from a full game (all harness-side, src/ untouched):
  - Environment built with Environment.__new__ (skips random biome generation and pygame
    surfaces, which are rendering-only); biome_map is a uniform lookup returning one
    Grassland_biome for every pixel; boundary obstacles are created exactly as
    _create_boundaries would (same Obstacle objects / edge tuples) but without drawing.
  - Instance-level no-ops for exogenous world spawning: spawn_tree, spawn_fruit,
    spawn_fruit_around_tree, spawn_predator. The rng.random() gates in non_agent_step still
    run, so RNG consumption per step is unchanged except inside the suppressed methods.
  - Initial observation: computed with the same agent.observe(...) call as
    Environment.non_agent_step lines 649-660, without ageing/drain/time advance.

Instrumentation (mode="full") wraps env.update_entity_position, env.update_entity_direction
and predator.step at instance level; each wrapper records inputs and then calls the original
bound method with identical arguments. mode="digest" installs no wrappers.

Usage:
  python -m training.predator_mechanics smoke
  python -m training.predator_mechanics main
  python -m training.predator_mechanics repeat --tag repeat|nolog
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import hashlib
import json
import math
import random
import struct
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.elements.agent import Agent
from src.elements.biome import Grassland_biome
from src.elements.environment import Environment
from src.elements.obstacle import Obstacle
from src.elements.predator import Predator
from src.utils.DTOs import ActionRequest
from src.utils.simulation import step_environment

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "C1-MECH-P01"

DT = 0.1
MAX_STEPS = 100
WORLD = 6000
CHUNK = 3000  # engine default 400; enlarged so boundary edges are always "local" (sensing.py L24-27 crashes on an empty edge list). All perception is still filtered by radius/cone.
AX0, AY0 = 3000.0, 3000.0  # internal agent start; reporting frame subtracts this
ESCAPE = math.pi           # world direction of initial escape line (-x)

SEPARATIONS = [80, 120]
AGENT_ENERGIES = [150, 300]
PRED_ENERGIES = [101, 200]
SEEDS = [0, 1, 2]
ARMS = {  # name: (requested move distance, facing)
    "S0": (0.0, "toward"),
    "A10": (10.0, "away"), "F10": (10.0, "toward"),
    "A15": (15.0, "away"), "F15": (15.0, "toward"),
    "A20": (20.0, "away"), "F20": (20.0, "toward"),
}
REP_FIXTURE = (120, 300, 101)
REP_SEED = 0


def wrap(a):
    """Canonical wrap to [-pi, pi)."""
    return (a + math.pi) % (2 * math.pi) - math.pi


class UniformBiome:
    def __init__(self, biome):
        self.biome = biome

    def __getitem__(self, idx):
        return self.biome


def _noop(*a, **k):
    return None


def build_env(seed, d, agent_energy, pred_energy):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    env = Environment.__new__(Environment)
    env.rng = rng
    env.width = WORLD
    env.height = WORLD
    env.agents, env.agents_dict, env._next_agent_id = [], {}, 0
    env.fruits, env.fruits_dict, env._next_fruit_id = [], {}, 0
    env.trees, env.obstacles, env.predators = [], [], []
    env.edges = set()
    env.score = 0
    env.time = 0
    env.biome_map = UniformBiome(Grassland_biome())
    env.chunk_size = CHUNK
    env.agent_observations = {}
    t = 30  # same thickness SimulationCore/Environment uses
    for (x, y, w, h) in [(0, 0, WORLD, t), (0, WORLD - t, WORLD, t), (0, 0, t, WORLD), (WORLD - t, 0, t, WORLD)]:
        obs = Obstacle(x, y, width=w, height=h, color=(100, 100, 100))
        env.obstacles.append(obs)
    for obs in env.obstacles:  # identical edge construction to spawn_obstacle
        env.edges.update([
            ((obs.x, obs.y), (obs.x + obs.width, obs.y)),
            ((obs.x + obs.width, obs.y), (obs.x + obs.width, obs.y + obs.height)),
            ((obs.x, obs.y + obs.height), (obs.x + obs.width, obs.y + obs.height)),
            ((obs.x, obs.y), (obs.x, obs.y + obs.height)),
        ])
    for name in ("spawn_tree", "spawn_fruit", "spawn_fruit_around_tree", "spawn_predator"):
        setattr(env, name, _noop)
    env._update_spatial_grid()

    agent = Agent(x=AX0, y=AY0, rng=rng, energy=float(agent_energy), max_age=120.0)
    agent.direction = 0.0
    agent.agent_id = env._next_agent_id
    env._next_agent_id += 1
    env.agents.append(agent)
    env.agents_dict[agent.agent_id] = agent
    pred = Predator(AX0 + d, AY0, rng=rng)
    pred.direction = math.pi
    pred.energy = float(pred_energy)
    pred.resting = False
    env.predators.append(pred)
    env._update_spatial_grid()

    # Initial observation: same call as non_agent_step L649-660, no time advance.
    la, lf, lt, lo, lp, le = env._get_local_objects(agent)
    env.agent_observations[agent.agent_id] = agent.observe(agents=la, fruits=lf, trees=lt, obstacles=lo,
                                                           predators=lp, edges=le)
    state = {"score": env.score, "sim_time": env.time, "num_agents": len(env.agents),
             "observations": [env.get_agent_state(a.agent_id) for a in env.agents]}
    return env, agent, pred, state


def _fhex(v):
    return struct.pack("<d", float(v))


def state_bytes(env, agent, pred, include_rng=True):
    b = bytearray()
    b += _fhex(env.time) + _fhex(env.score)
    alive = agent in env.agents
    b += bytes([alive])
    for v in (agent.x, agent.y, agent.direction, agent.energy, agent.age):
        b += _fhex(v)
    for v in (pred.x, pred.y, pred.direction, pred.energy):
        b += _fhex(v)
    b += bytes([pred.resting])
    if include_rng:
        b += hashlib.sha256(repr(env.rng.getstate()).encode()).digest()
    return bytes(b)


def controller(arm, obs_state, heading_acc):
    """Returns (move_distance, move_direction_rel, turn). Uses only the controller observation
    and the internal heading accumulator (known initial heading + issued turns)."""
    dist, facing = ARMS[arm]
    move_dir = wrap(ESCAPE - heading_acc)
    if facing == "away":
        turn = wrap(ESCAPE - heading_acc)
    else:
        preds = [o for o in obs_state["observations"] if o["type"] == "Predator"]
        turn = min(preds, key=lambda o: o["distance"])["angle"] if preds else 0.0
    return dist, move_dir, turn


def run_episode(d, ae, pe, arm, seed, mode="full"):
    t0 = time.perf_counter()
    env, agent, pred, state = build_env(seed, d, ae, pe)
    init_hash = hashlib.sha256(state_bytes(env, agent, pred)).hexdigest()
    rng_hash = hashlib.sha256(repr(env.rng.getstate()).encode()).hexdigest()
    heading_acc = 0.0
    ticks = []
    step_digests = []
    traj_h = hashlib.sha256()
    traj_h_norng = hashlib.sha256()
    cur = {}

    if mode == "full":
        orig_pos = env.update_entity_position
        orig_dir = env.update_entity_direction
        orig_pstep = pred.step

        def w_pos(entity, distance, direction=None, local_obstacles=None):
            who = "agent" if entity is agent else "predator"
            e0 = entity.energy
            clamp = entity.energy < entity.max_energy / 5 and max(0.0, distance) > entity.speed
            thr = entity.energy < entity.max_energy / 5
            x0, y0 = entity.x, entity.y
            r = orig_pos(entity, distance, direction, local_obstacles)
            cur[who + "_move"] = {"requested": distance, "rel_direction": direction, "energy_before": e0,
                                  "energy_after": entity.energy, "move_cost": e0 - entity.energy,
                                  "below_sprint_threshold": thr, "sprint_clamped": clamp,
                                  "realized": math.hypot(entity.x - x0, entity.y - y0)}
            if who == "predator":
                cur["dist_after_pred_move"] = math.hypot(agent.x - pred.x, agent.y - pred.y) if agent in env.agents else None
            return r

        def w_dir(entity, turn_angle):
            who = "agent" if entity is agent else "predator"
            e0 = entity.energy
            r = orig_dir(entity, turn_angle)
            cur[who + "_turn"] = {"turn": turn_angle, "turn_cost": e0 - entity.energy}
            if who == "predator":
                cur["dist_before_kill_check"] = math.hypot(agent.x - pred.x, agent.y - pred.y) if agent in env.agents else None
            return r

        def w_pstep(observation):
            ags = [o for o in observation if o.get("type") == "Agent"]
            info = {"n_agents_perceived": len(ags),
                    "pred_energy_at_decision": pred.energy,
                    "true_dist_at_decision": math.hypot(agent.x - pred.x, agent.y - pred.y) if agent in env.agents else None,
                    "pred_xy_at_decision": (pred.x - AX0, pred.y - AY0),
                    "agent_xy_at_decision": (agent.x - AX0, agent.y - AY0)}
            sig = orig_pstep(observation)
            if ags:
                c = min(ags, key=lambda f: f["distance"])
                pred_chase = abs(c["rel_dir"]) > np.pi * 1 / 2 or c["distance"] < pred.hearing_radius * 1.5
                ts = max(-0.3, min(0.3, c["angle"] * 0.5))
                # branch observed by exact match of the emitted signals against each branch's
                # output (predator.py L37-62), recomputed from the same observation
                if abs(c["angle"]) > 0.05:
                    chase_sig = {"turn": ts, "move": min(pred.sprint_speed, c["distance"]), "direction": ts}
                else:
                    chase_sig = {"move": min(pred.sprint_speed, c["distance"]), "direction": c["angle"]}
                ps = -np.sign(c["rel_dir"])
                md = c["angle"] + ps * np.pi * 1 / 4
                xa, ya = c["distance"] * np.cos(c["angle"]), c["distance"] * np.sin(c["angle"])
                circle_sig = {"move": pred.sprint_speed, "direction": md,
                              "turn": np.arctan2(ya - pred.sprint_speed * np.sin(md), xa - pred.sprint_speed * np.cos(md))}
                m_ch, m_ci = sig == chase_sig, sig == circle_sig
                observed = "ambiguous" if (m_ch and m_ci) else "chase" if m_ch else "circle" if m_ci else "unidentified"
                info.update({"decision_dist": c["distance"], "angle_to_agent": c["angle"], "rel_dir": c["rel_dir"],
                             "predicate_mode": "chase" if pred_chase else "circle", "observed_mode": observed,
                             "pivot_sign": float(-np.sign(c["rel_dir"])),
                             "chase_reason": ("dist<90" if c["distance"] < 90 else "") + ("|facing_away" if abs(c["rel_dir"]) > np.pi / 2 else "")})
            else:
                info.update({"predicate_mode": "no_target", "observed_mode": "no_target"})
            info["signals"] = {k: (float(v) if v is not None else None) for k, v in sig.items()}
            cur["predator_decision"] = info
            return sig

        env.update_entity_position = w_pos
        env.update_entity_direction = w_dir
        pred.step = w_pstep

    outcome, event_step = "survived_horizon", None
    for step in range(1, MAX_STEPS + 1):
        obs_state = state["observations"][0]
        mv, mdir, turn = controller(arm, obs_state, heading_acc)
        action = {"agent_id": obs_state["agent_id"], "move_distance": float(mv), "move_direction": float(mdir),
                  "turn_angle": float(turn), "spawn_agent": False}
        cur.clear()
        pre = {"agent_energy": agent.energy, "pred_energy": pred.energy, "pred_resting": pred.resting,
               "dist": math.hypot(agent.x - pred.x, agent.y - pred.y),
               "agent_below_sprint_threshold": agent.energy < agent.max_energy / 5}
        state = step_environment(env, [(action["agent_id"], ActionRequest(**action))], DT)
        heading_acc += turn
        alive = agent in env.agents
        sb = state_bytes(env, agent, pred)
        abytes = b"".join(_fhex(v) for v in (mv, mdir, turn))
        dg = hashlib.sha256(sb + abytes).hexdigest()
        step_digests.append(dg)
        traj_h.update(sb + abytes)
        traj_h_norng.update(state_bytes(env, agent, pred, include_rng=False) + abytes)
        if not alive:
            if agent.energy <= 0:  # removed by the drain check (L640-644); a kill needs energy > 0 there
                outcome = "energy_death"
            else:
                outcome = "captured"
        if mode == "full":
            obs_preds = [o for o in obs_state["observations"] if o["type"] == "Predator"]
            pdec = cur.get("predator_decision")
            if pre["pred_resting"] and not pred.resting:
                ptrans = "woke"
            elif not pre["pred_resting"] and pred.resting:
                ptrans = "fell_asleep"
            else:
                ptrans = None
            if pdec is None:
                phase_mode = "sleep"
            else:
                phase_mode = pdec["observed_mode"]
            ticks.append({
                "step": step, "t": round(step * DT, 10), "fixture": [d, ae, pe], "arm": arm, "seed": seed,
                "controller_obs_predators": obs_preds, "action": action, "commanded_heading_acc": heading_acc,
                "agent_xy": [agent.x - AX0, agent.y - AY0], "agent_dir": agent.direction,
                "pred_xy": [pred.x - AX0, pred.y - AY0], "pred_dir": pred.direction,
                "dist_pre_step": pre["dist"],
                "dist_at_decision": (pdec or {}).get("true_dist_at_decision"),
                "dist_before_kill_check": cur.get("dist_before_kill_check", cur.get("dist_after_pred_move")),  # turn does not move; falls back to post-move distance when no turn signal
                "dist_post_step": math.hypot(agent.x - pred.x, agent.y - pred.y),
                "agent_energy_pre": pre["agent_energy"], "agent_energy_post": agent.energy,
                "agent_move": cur.get("agent_move"), "agent_turn": cur.get("agent_turn"),
                "pred_energy_pre": pre["pred_energy"], "pred_energy_post": pred.energy,
                "pred_resting_pre": pre["pred_resting"], "pred_resting_post": pred.resting,
                "pred_transition": ptrans, "pred_move": cur.get("predator_move"), "pred_turn": cur.get("predator_turn"),
                "predator_decision": pdec, "mode": phase_mode,
                "agent_perceived_by_pred": bool(pdec and pdec["n_agents_perceived"] > 0),
                "agent_alive": alive, "score": env.score, "digest": dg,
            })
        if not alive:
            event_step = step
            break
        if step < MAX_STEPS:
            pass
    return {
        "fixture": {"d": d, "agent_energy": ae, "pred_energy": pe}, "arm": arm, "seed": seed, "mode": mode,
        "initial_state_hash": init_hash, "initial_rng_hash": rng_hash,
        "outcome": outcome, "event_step": event_step, "steps_run": len(step_digests),
        "trajectory_digest": traj_h.hexdigest(), "trajectory_digest_norng": traj_h_norng.hexdigest(),
        "step_digests": step_digests, "final_score": env.score, "final_state_digest": step_digests[-1],
        "runtime_sec": time.perf_counter() - t0, "ticks": ticks,
    }


# ---------------------------------------------------------------- summaries
def summarize(ep):
    T = ep["ticks"]
    fx = ep["fixture"]
    s = {"d": fx["d"], "agent_energy0": fx["agent_energy"], "pred_energy0": fx["pred_energy"], "arm": ep["arm"],
         "seed": ep["seed"], "outcome": ep["outcome"],
         "event_time": round(ep["event_step"] * DT, 10) if ep["event_step"] else None,
         "censored_at_10s": ep["outcome"] == "survived_horizon", "steps_run": ep["steps_run"]}
    na = "not observed before death/horizon"
    dists = [(t["t"], t["dist_before_kill_check"] if t["dist_before_kill_check"] is not None else t["dist_post_step"]) for t in T]
    # minimum separation over pre-kill-check and post-step samples
    samples = [(0.0, fx["d"])]
    for t in T:
        if t["dist_before_kill_check"] is not None:
            samples.append((t["t"], t["dist_before_kill_check"]))
        samples.append((t["t"], t["dist_post_step"]))
    mt, md = min(samples, key=lambda x: x[1])
    s["min_separation"], s["min_separation_time"] = md, mt
    for m in ("chase", "circle", "no_target", "sleep", "unidentified"):
        s["time_" + m] = round(sum(DT for t in T if t["mode"] == m), 10)
    s["predicate_mismatches"] = sum(1 for t in T if t["predator_decision"] and t["predator_decision"]["predicate_mode"] != t["predator_decision"]["observed_mode"])
    s["decision_steps"] = sum(1 for t in T if t["predator_decision"])
    first = lambda cond: next((t for t in T if cond(t)), None)
    fl = first(lambda t: t["predator_decision"] and t["predator_decision"]["n_agents_perceived"] == 0)
    s["first_pred_perception_loss"] = fl["t"] if fl else na
    fs = first(lambda t: t["pred_transition"] == "fell_asleep")
    s["first_sleep"] = fs["t"] if fs else na
    fw = first(lambda t: t["pred_transition"] == "woke" and fs and t["t"] > fs["t"])
    s["first_wake_after_sleep"] = fw["t"] if fw else na
    fb = first(lambda t: t["agent_move"] and t["agent_move"]["below_sprint_threshold"])
    s["first_below_sprint_threshold"] = fb["t"] if fb else na
    fc = first(lambda t: t["agent_move"] and t["agent_move"]["sprint_clamped"])
    s["first_sprint_clamp"] = fc["t"] if fc else na
    fam = first(lambda t: t["mode"] == "circle")
    s["first_circle"] = fam["t"] if fam else na
    fch = first(lambda t: t["mode"] == "chase")
    s["first_chase"] = fch["t"] if fch else na
    if fs:
        s["sleep_agent_energy"] = fs["agent_energy_post"]
        s["sleep_separation"] = fs["dist_post_step"]
        s["sleep_escape_displacement"] = -fs["agent_xy"][0]
        s["sleep_agent_energy_ge_20pct"] = fs["agent_energy_post"] >= 0.2 * 500
    else:
        s["sleep_agent_energy"] = s["sleep_separation"] = s["sleep_escape_displacement"] = na
        s["sleep_agent_energy_ge_20pct"] = na
    s["survived_until_pred_sleep"] = bool(fs and fs["agent_alive"])
    last = T[-1]
    s["final_t"] = last["t"]
    s["final_agent_energy"] = last["agent_energy_post"] if last["agent_alive"] else None
    s["final_separation"] = last["dist_post_step"] if last["agent_alive"] else None
    s["final_escape_displacement"] = -last["agent_xy"][0]
    s["final_pred_energy"] = last["pred_energy_post"]
    s["pred_asleep_at_end"] = last["pred_resting_post"]
    s["agent_energy_used"] = fx["agent_energy"] - last["agent_energy_post"]
    s["agent_move_cost"] = sum(t["agent_move"]["move_cost"] for t in T if t["agent_move"])
    s["agent_turn_cost"] = sum(t["agent_turn"]["turn_cost"] for t in T if t["agent_turn"])
    s["agent_drain_derived"] = s["agent_energy_used"] - s["agent_move_cost"] - s["agent_turn_cost"]
    s["pred_displacement"] = math.hypot(last["pred_xy"][0] - fx["d"], last["pred_xy"][1])
    s["pred_displacement_along_escape"] = -(last["pred_xy"][0] - fx["d"])
    s["pred_energy_used"] = fx["pred_energy"] - last["pred_energy_post"]
    s["runtime_sec"] = ep["runtime_sec"]
    s["trajectory_digest_norng"] = ep["trajectory_digest_norng"][:16]
    s["trajectory_digest"] = ep["trajectory_digest"][:16]
    s["validity"] = "valid" if s["predicate_mismatches"] == 0 else "predicate_mismatch"
    return s


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def cmd_main(args):
    import csv, gzip
    OUT.mkdir(parents=True, exist_ok=True)
    tdir = OUT / "traces"
    tdir.mkdir(exist_ok=True)
    rows, digests = [], {}
    t0 = time.perf_counter()
    n = 0
    for d in SEPARATIONS:
        for ae in AGENT_ENERGIES:
            for pe in PRED_ENERGIES:
                for seed in SEEDS:
                    for arm in ARMS:
                        if time.perf_counter() - t0 > args.sim_cap_sec:
                            print("SIM CAP REACHED; stopping", flush=True)
                            break
                        ep = run_episode(d, ae, pe, arm, seed, "full")
                        n += 1
                        name = f"d{d}_ae{ae}_pe{pe}_{arm}_s{seed}.json.gz"
                        with gzip.open(tdir / name, "wt") as f:
                            json.dump(ep, f, default=_jsonable)
                        rows.append(summarize(ep))
                        digests[f"{d}|{ae}|{pe}|{arm}|{seed}"] = {k: ep[k] for k in ("trajectory_digest", "trajectory_digest_norng", "final_state_digest", "initial_state_hash", "initial_rng_hash", "outcome", "final_score")}
    with open(OUT / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    json.dump(digests, open(OUT / "digests_main.json", "w"), indent=1)
    print(f"main episodes={n} sim_wall={time.perf_counter()-t0:.1f}s", flush=True)


def cmd_repeat(args):
    d, ae, pe = REP_FIXTURE
    mode = "digest" if args.tag == "nolog" else "full"
    out = {}
    t0 = time.perf_counter()
    for arm in ARMS:
        ep = run_episode(d, ae, pe, arm, REP_SEED, mode)
        out[arm] = {k: ep[k] for k in ("trajectory_digest", "trajectory_digest_norng", "final_state_digest", "step_digests", "outcome", "event_step", "final_score", "initial_state_hash", "initial_rng_hash")}
    json.dump(out, open(OUT / f"digests_{args.tag}.json", "w"), indent=1)
    print(f"{args.tag} episodes=7 sim_wall={time.perf_counter()-t0:.2f}s", flush=True)


def cmd_smoke(args):
    """<=12 cases, <=5 steps each. Checks harness risks against source-derived expectations."""
    global MAX_STEPS
    MAX_STEPS = 5
    res = []

    def check(name, ok, detail):
        res.append({"case": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail), flush=True)

    # 1 direction conversion: A10 first step moves -x by 10, heading -> -pi (turn cost 0.5)
    ep = run_episode(120, 300, 101, "A10", 0)
    t1 = ep["ticks"][0]
    check("away_first_step_direction", abs(t1["agent_xy"][0] + 10) < 1e-9 and abs(t1["agent_turn"]["turn"] + math.pi) < 1e-12,
          {"agent_xy": t1["agent_xy"], "turn": t1["agent_turn"]})
    check("away_turn_cost_0.5", abs(t1["agent_turn"]["turn_cost"] - 0.5) < 1e-12, t1["agent_turn"])
    # 2 energy accounting: walk 10 -> 0.5; drain 0.1
    check("agent_energy_components", abs(t1["agent_energy_pre"] - t1["agent_energy_post"] - (0.5 + 0.5 + 0.1)) < 1e-9,
          [t1["agent_energy_pre"], t1["agent_energy_post"]])
    # 3 sprint cost 20 -> 10*0.05 + 10*0.5 = 5.5
    ep = run_episode(120, 300, 101, "F20", 0)
    t1 = ep["ticks"][0]
    check("sprint_cost_5.5", abs(t1["agent_move"]["move_cost"] - 5.5) < 1e-9, t1["agent_move"])
    # 4 sprint clamp below 20%: energy 99 < 100
    ep = run_episode(120, 99, 101, "F20", 0)
    t1 = ep["ticks"][0]
    check("sprint_clamp_below_threshold", t1["agent_move"]["sprint_clamped"] and abs(t1["agent_move"]["realized"] - 10) < 1e-9, t1["agent_move"])
    # 5 mode predicate matches observed branch in S0 (facing, 120) and A10 at d=80 (chase)
    ep = run_episode(120, 300, 101, "S0", 0)
    mm = [(t["predator_decision"]["predicate_mode"], t["predator_decision"]["observed_mode"]) for t in ep["ticks"]]
    check("predicate_vs_observed_S0_d120", all(a == b for a, b in mm), mm)
    ep = run_episode(80, 300, 101, "A10", 0)
    mm = [(t["predator_decision"]["predicate_mode"], t["predator_decision"]["observed_mode"]) for t in ep["ticks"]]
    check("predicate_vs_observed_A10_d80", all(a == b == "chase" for a, b in mm), mm)
    # 6 kill ordering: d=20 stationary, predator moves 15 -> within 15 -> captured on step 1
    ep = run_episode(20, 300, 101, "S0", 0)
    check("kill_after_predator_move", ep["outcome"] == "captured" and ep["event_step"] == 1,
          {"outcome": ep["outcome"], "step": ep["event_step"], "dist_before_kill": ep["ticks"][0]["dist_before_kill_check"]})
    # 7 sleep transition: predator energy 2.0, sprinting costs 2.55 -> sleeps on step 1 (after kill check)
    ep = run_episode(120, 300, 2.0, "A10", 0)
    ts = [(t["pred_energy_pre"], t["pred_energy_post"], t["pred_transition"], t["mode"]) for t in ep["ticks"]]
    check("sleep_transition_and_recharge", ts[0][2] == "fell_asleep" and ts[1][3] == "sleep" and abs(ts[1][1] - ts[1][0] - 3.0) < 1e-9, ts)
    # 8 one action per living agent + no time advance at init
    env, agent, pred, st = build_env(0, 120, 300, 101)
    check("init_no_time_advance_one_obs", env.time == 0 and agent.energy == 300 and len(st["observations"]) == 1
          and any(o["type"] == "Predator" for o in st["observations"][0]["observations"]),
          {"time": env.time, "energy": agent.energy, "obs": st["observations"][0]["observations"]})
    # 9 logging does not change state: full vs digest for F15 5 steps
    a = run_episode(120, 300, 101, "F15", 0, "full")
    b = run_episode(120, 300, 101, "F15", 0, "digest")
    check("instrumentation_noninterference_5step", a["step_digests"] == b["step_digests"], [a["trajectory_digest"][:12], b["trajectory_digest"][:12]])
    json.dump(res, open(OUT / "smoke.json", "w"), indent=1, default=_jsonable)
    print("smoke cases:", len(res), "episodes used: 10 (<=5 steps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["smoke", "main", "repeat"])
    p.add_argument("--tag", default="repeat")
    p.add_argument("--sim-cap-sec", type=float, default=240.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"smoke": cmd_smoke, "main": cmd_main, "repeat": cmd_repeat}[a.cmd](a)
