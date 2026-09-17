"""
C1-MECH-P02: deliberate facing-offset retreat / predator sleep mechanism study.

Extends the C1-MECH-P01 diagnostic harness (training/predator_mechanics.py, left
UNCHANGED and imported read-only for the compatibility check) with:
  - a larger arena (16000x16000, chunk_size=8000, agent at internal (8000,8000))
    to give clearance for a 40-second horizon instead of P01's 10-second one;
  - a single generalized controller with a signed facing-offset parameter delta,
    per task_prompt.md Section 2, replacing P01's separate "away"/"toward" arms;
  - a 300-step living phase plus, only if the agent dies first, a 100-step
    post-death continuation via plain step_environment(env, [], dt) calls (no
    agent, no controller action) to observe purely native predator/environment
    behaviour after the decoy disappears (explicitly NOT a survival metric);
  - a fourth instance-level wrapper (env.kill_agent) to distinguish predator
    energy just BEFORE a kill from energy AFTER eating (P01 did not need this
    because its own instrumentation never inspected the kill/eat step itself).

All src/ files are untouched. Fixture deviations are identical in kind to P01's
(Environment.__new__, uniform grassland biome, no-op spawners, boundary-only
obstacles) -- see experiments/C1-MECH-P02/source_notes.md for exact citations.

Usage:
  python -m training.predator_mechanics_p02 emit-config     # writes config.json, 0 episodes
  python -m training.predator_mechanics_p02 smoke           # <=12 episodes, <=5 steps each
  python -m training.predator_mechanics_p02 compat          # 6 episodes, P01 arena+controller
  python -m training.predator_mechanics_p02 main            # 216 episodes (NOT run in phase 1)
  python -m training.predator_mechanics_p02 repeat --tag repeat|nolog   # 9 episodes each
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import argparse
import gzip
import hashlib
import json
import math
import random
import struct
import sys
import time
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
OUT = ROOT / "experiments" / "C1-MECH-P02"
P01_OUT = ROOT / "experiments" / "C1-MECH-P01"
# Historical P01 traces exist only in the main checkout (read-only reference).
P01_MAIN_TRACES = Path(r"C:\Users\sjoen\PycharmProjects\NordicAI2026\Nordic-AI-Cup-2026-main"
                        r"\survival-simulator\experiments\C1-MECH-P01\traces")

DT = 0.1
LIVING_STEPS = 300     # 30 s
POST_DEATH_STEPS = 100  # 10 s, counterfactual continuation, out of survival metrics
WORLD = 16000
CHUNK = 8000  # enlarged so boundary edges are always "local" candidates for
              # compute_visibility (src/utils/sensing.py L24-27 crashes on an
              # empty edge list) -- see source_notes.md. Verified via smoke
              # that no boundary edge is ever close enough to be a hit_edge.
AXY0 = 8000.0  # internal agent start (both x and y); reporting frame subtracts this
ESCAPE = math.pi  # world direction of the fixed initial escape line (-x)

SEPARATIONS = [80, 120]
AGENT_ENERGIES = [150, 300]
PRED_ENERGIES = [101, 200]
SEEDS = [0, 1, 2]
SPEEDS = [10, 15, 20]
DELTAS = [("0", 0.0), ("+15", math.pi / 12), ("-15", -math.pi / 12)]
ARMS = {f"S{sp}_D{tag}": (float(sp), dv) for sp in SPEEDS for tag, dv in DELTAS}
ZERO_ARMS = [a for a in ARMS if a.endswith("_D0")]
SIGNED_ARMS = [a for a in ARMS if not a.endswith("_D0")]
ALL_FIXTURES = [(d, ae, pe) for d in SEPARATIONS for ae in AGENT_ENERGIES for pe in PRED_ENERGIES]
# Representative subset used both for early-validation-before-main-matrix (spec S2/S3)
# and as the two P01-compatibility fixtures (spec S3 "six P01 geometry compatibility runs").
REP_FIXTURES = [(120, 150, 200), (120, 300, 101)]
REP_FIXTURE_FOR_REPEATS = (120, 300, 101)
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
    """Same fixture-construction pattern as P01's build_env, scaled to the P02 arena."""
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
    t = 30  # same thickness Environment._create_boundaries uses
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

    agent = Agent(x=AXY0, y=AXY0, rng=rng, energy=float(agent_energy), max_age=120.0)
    agent.direction = 0.0
    agent.agent_id = env._next_agent_id
    env._next_agent_id += 1
    env.agents.append(agent)
    env.agents_dict[agent.agent_id] = agent
    pred = Predator(AXY0 + d, AXY0, rng=rng)
    pred.direction = math.pi
    pred.energy = float(pred_energy)
    pred.resting = False
    env.predators.append(pred)
    env._update_spatial_grid()

    # Initial observation: same call as non_agent_step's agent block, no time advance.
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


def controller(obs_state, heading_acc, speed, delta):
    """Task spec Section 2, steps 1-3, exactly:
      1. move_direction = wrap(ESCAPE - heading_acc), move_distance = speed (fixed line, no world truth).
      2. If predator present in observation with bearing beta: turn = wrap(beta + delta).
         If absent: turn = 0. No scanning / prediction of hidden position.
      (heading_acc update happens in the caller, step 3.)
    """
    move_dir = wrap(ESCAPE - heading_acc)
    preds = [o for o in obs_state["observations"] if o["type"] == "Predator"]
    if preds:
        beta = min(preds, key=lambda o: o["distance"])["angle"]
        turn = wrap(beta + delta)
    else:
        turn = 0.0
    return speed, move_dir, turn


def run_episode(d, ae, pe, arm, seed, mode="full", living_steps=LIVING_STEPS, post_death_steps=POST_DEATH_STEPS):
    t0 = time.perf_counter()
    speed, delta = ARMS[arm]
    env, agent, pred, state = build_env(seed, d, ae, pe)
    init_hash = hashlib.sha256(state_bytes(env, agent, pred)).hexdigest()
    rng_hash = hashlib.sha256(repr(env.rng.getstate()).encode()).hexdigest()
    heading_acc = 0.0
    ticks = []
    post_ticks = []
    step_digests = []
    traj_h = hashlib.sha256()
    traj_h_norng = hashlib.sha256()
    cur = {}

    if mode == "full":
        orig_pos = env.update_entity_position
        orig_dir = env.update_entity_direction
        orig_pstep = pred.step
        orig_kill = env.kill_agent

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
                cur["pred_energy_pre_kill"] = entity.energy  # last snapshot before any kill-check this tick
            return r

        def w_dir(entity, turn_angle):
            who = "agent" if entity is agent else "predator"
            e0 = entity.energy
            r = orig_dir(entity, turn_angle)
            cur[who + "_turn"] = {"turn": turn_angle, "turn_cost": e0 - entity.energy}
            if who == "predator":
                cur["dist_before_kill_check"] = math.hypot(agent.x - pred.x, agent.y - pred.y) if agent in env.agents else None
                cur["pred_energy_pre_kill"] = entity.energy  # overwrites move-snapshot: turn always follows move
            return r

        def w_kill(victim):
            # env.kill_agent is ALSO called by the agent-side energy-drain check
            # (environment.py L642-644), which runs in the agent block, strictly before the
            # predator block (and its predator.step call) executes this tick. Only treat this as
            # an actual predator kill (and thus log pre-kill/post-eat energy) when this tick's
            # predator decision has already been recorded -- Opus stage-2 review BLOCKING-2:
            # without this guard, an energy death would be misread as a kill with unchanged
            # ("pre==post") predator energy.
            if "predator_decision" in cur:
                cur["post_kill_energy"] = pred.energy
                cur["killed_agent_energy_pre_removal"] = victim.energy
            else:
                cur["energy_death_removal"] = True
            return orig_kill(victim)

        def w_pstep(observation):
            ags = [o for o in observation if o.get("type") == "Agent"]
            edg = [o for o in observation if o.get("type") == "Edge"]
            info = {"n_agents_perceived": len(ags), "n_edges_perceived": len(edg),
                    "pred_energy_at_decision": pred.energy,
                    "true_dist_at_decision": math.hypot(agent.x - pred.x, agent.y - pred.y) if agent in env.agents else None,
                    "pred_xy_at_decision": (pred.x - AXY0, pred.y - AXY0),
                    "agent_xy_at_decision": (agent.x - AXY0, agent.y - AXY0)}
            sig = orig_pstep(observation)
            if ags:
                c = min(ags, key=lambda f: f["distance"])
                pred_chase = abs(c["rel_dir"]) > np.pi * 1 / 2 or c["distance"] < pred.hearing_radius * 1.5
                ts = max(-0.3, min(0.3, c["angle"] * 0.5))
                # Branch identified by exact match of the emitted signal dict against each
                # branch's recomputed output (predator.py L37-62), not inferred from labels.
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
                             "chase_reason": ("dist<90" if c["distance"] < pred.hearing_radius * 1.5 else "") +
                                             ("|facing_away" if abs(c["rel_dir"]) > np.pi / 2 else "")})
            elif edg:
                # predator.py L65-95: edge-avoidance branch. Expected count is 0 (see source_notes.md /
                # B4): boundary is >7850 away, vision radii are 200/250, so this should never fire; if it
                # does, that is a fixture-geometry failure to report, not something to hide.
                info.update({"predicate_mode": "edge_avoid", "observed_mode": "edge_avoid"})
            else:
                info.update({"predicate_mode": "wander", "observed_mode": "wander"})
            info["signals"] = {k: (float(v) if v is not None else None) for k, v in sig.items()}
            cur["predator_decision"] = info
            return sig

        env.update_entity_position = w_pos
        env.update_entity_direction = w_dir
        env.kill_agent = w_kill
        pred.step = w_pstep

    outcome, event_step = "survived_horizon", None
    death_step = None
    for step in range(1, living_steps + 1):
        obs_state = state["observations"][0]
        # desired-offset audit: log delta and the observed bearing beta this tick used
        preds_obs = [o for o in obs_state["observations"] if o["type"] == "Predator"]
        beta_used = min(preds_obs, key=lambda o: o["distance"])["angle"] if preds_obs else None
        mv, mdir, turn = controller(obs_state, heading_acc, speed, delta)
        action = {"agent_id": obs_state["agent_id"], "move_distance": float(mv), "move_direction": float(mdir),
                  "turn_angle": float(turn), "spawn_agent": False}
        cur.clear()
        pre = {"agent_energy": agent.energy, "pred_energy": pred.energy, "pred_resting": pred.resting,
               "dist": math.hypot(agent.x - pred.x, agent.y - pred.y),
               "agent_below_sprint_threshold": agent.energy < agent.max_energy / 5}
        any_edge_in_agent_obs = any(o["type"] == "Edge" for o in obs_state["observations"])
        state = step_environment(env, [(action["agent_id"], ActionRequest(**action))], DT)
        heading_acc = wrap(heading_acc + turn)
        alive = agent in env.agents
        sb = state_bytes(env, agent, pred)
        abytes = b"".join(_fhex(v) for v in (mv, mdir, turn))
        dg = hashlib.sha256(sb + abytes).hexdigest()
        step_digests.append(dg)
        traj_h.update(sb + abytes)
        traj_h_norng.update(state_bytes(env, agent, pred, include_rng=False) + abytes)
        if not alive:
            death_step = step
            if agent.energy <= 0:  # removed by the drain check (L642-644); a kill needs energy > 0 there
                outcome = "energy_death"
            else:
                outcome = "captured"
        if mode == "full":
            pdec = cur.get("predator_decision")
            any_edge_in_pred_obs = bool(pdec and pdec.get("n_edges_perceived", 0) > 0)
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
                "speed": speed, "delta": delta,
                "controller_obs_predators": preds_obs, "beta_used": beta_used, "action": action,
                "commanded_heading_acc": heading_acc,
                "any_edge_in_controller_obs": any_edge_in_agent_obs,
                "any_edge_in_pred_obs": any_edge_in_pred_obs,
                "agent_xy": [agent.x - AXY0, agent.y - AXY0], "agent_dir": agent.direction,
                "pred_xy": [pred.x - AXY0, pred.y - AXY0], "pred_dir": pred.direction,
                "dist_pre_step": pre["dist"],
                "dist_at_decision": (pdec or {}).get("true_dist_at_decision"),
                "true_rel_dir_at_decision": (pdec or {}).get("rel_dir"),
                "desired_offset_delta": delta,
                "dist_before_kill_check": cur.get("dist_before_kill_check", cur.get("dist_after_pred_move")),
                "dist_post_step": math.hypot(agent.x - pred.x, agent.y - pred.y),
                "agent_energy_pre": pre["agent_energy"], "agent_energy_post": agent.energy,
                "agent_move": cur.get("agent_move"), "agent_turn": cur.get("agent_turn"),
                "pred_energy_pre": pre["pred_energy"],
                "pred_energy_pre_kill": cur.get("pred_energy_pre_kill"),
                "pred_energy_post_kill": cur.get("post_kill_energy"),
                "killed_agent_energy_pre_removal": cur.get("killed_agent_energy_pre_removal"),
                "energy_death_removal": cur.get("energy_death_removal", False),
                "pred_energy_post": pred.energy,
                "pred_resting_pre": pre["pred_resting"], "pred_resting_post": pred.resting,
                "pred_transition": ptrans, "pred_move": cur.get("predator_move"), "pred_turn": cur.get("predator_turn"),
                "predator_decision": pdec, "mode": phase_mode,
                "agent_perceived_by_pred": bool(pdec and pdec["n_agents_perceived"] > 0),
                "agent_alive": alive, "score": env.score, "digest": dg,
            })
        if not alive:
            event_step = step
            break

    # Post-death continuation: plain step_environment(env, [], dt) calls, no agent, no
    # controller action. Explicitly counterfactual / out of survival metrics (spec S2).
    if death_step is not None and mode == "full":
        for pstep in range(1, post_death_steps + 1):
            pre_p = {"pred_energy": pred.energy, "pred_resting": pred.resting,
                     "pred_xy": [pred.x - AXY0, pred.y - AXY0]}
            cur.clear()
            step_environment(env, [], DT)
            if pre_p["pred_resting"] and not pred.resting:
                ptrans = "woke"
            elif not pre_p["pred_resting"] and pred.resting:
                ptrans = "fell_asleep"
            else:
                ptrans = None
            post_ticks.append({
                "post_death_step": pstep, "t_since_death": round(pstep * DT, 10),
                "pred_xy": [pred.x - AXY0, pred.y - AXY0], "pred_dir": pred.direction,
                "pred_energy": pred.energy, "pred_resting": pred.resting, "pred_transition": ptrans,
                "dist_to_initial_agent_ref": math.hypot(pred.x - AXY0, pred.y - AXY0),
            })
    elif death_step is not None and mode != "full":
        for pstep in range(post_death_steps):
            step_environment(env, [], DT)

    return {
        "fixture": {"d": d, "agent_energy": ae, "pred_energy": pe}, "arm": arm, "seed": seed, "mode": mode,
        "speed": speed, "delta": delta,
        "initial_state_hash": init_hash, "initial_rng_hash": rng_hash,
        "outcome": outcome, "event_step": event_step, "death_step": death_step, "steps_run": len(step_digests),
        "post_death_steps_run": len(post_ticks),
        "trajectory_digest": traj_h.hexdigest(), "trajectory_digest_norng": traj_h_norng.hexdigest(),
        "step_digests": step_digests, "final_score": env.score, "final_state_digest": step_digests[-1] if step_digests else None,
        "runtime_sec": time.perf_counter() - t0, "ticks": ticks, "post_death_ticks": post_ticks,
    }


# ---------------------------------------------------------------- run-order (predeclared)
def build_main_run_order():
    """216 = 8 fixtures x 9 arms x 3 seeds. Predeclared order per spec S2/S3:
    representative-subset early validation first (both compat fixtures, all 9 arms,
    seed 0 -- zero-offset arms before signed arms within that), then the remaining
    zero-offset controls across the full fixture matrix, then the remaining signed
    -offset cases. Each (fixture, arm, seed) combo appears exactly once.
    """
    order = []
    seen = set()

    def add(d, ae, pe, arm, seed):
        key = (d, ae, pe, arm, seed)
        if key in seen:
            return
        seen.add(key)
        speed, delta = ARMS[arm]
        order.append({"index": len(order), "d": d, "agent_energy": ae, "pred_energy": pe,
                      "arm": arm, "seed": seed, "speed": speed, "delta": delta})

    # Phase 1: representative-subset early validation (both compat fixtures, seed 0, all arms).
    for (d, ae, pe) in REP_FIXTURES:
        for arm in ZERO_ARMS:
            add(d, ae, pe, arm, REP_SEED)
        for arm in SIGNED_ARMS:
            add(d, ae, pe, arm, REP_SEED)
    # Phase 2: remaining zero-offset controls, full fixture matrix x all seeds.
    for (d, ae, pe) in ALL_FIXTURES:
        for arm in ZERO_ARMS:
            for seed in SEEDS:
                add(d, ae, pe, arm, seed)
    # Phase 3: remaining signed-offset cases, full fixture matrix x all seeds.
    for (d, ae, pe) in ALL_FIXTURES:
        for arm in SIGNED_ARMS:
            for seed in SEEDS:
                add(d, ae, pe, arm, seed)
    assert len(order) == 216, f"expected 216 unique main episodes, got {len(order)}"
    return order


# ---------------------------------------------------------------- summaries
def summarize(ep):
    T = ep["ticks"]
    PT = ep["post_death_ticks"]
    fx = ep["fixture"]
    s = {"d": fx["d"], "agent_energy0": fx["agent_energy"], "pred_energy0": fx["pred_energy"], "arm": ep["arm"],
         "speed": ep["speed"], "delta_deg": round(math.degrees(ep["delta"]), 6),
         "seed": ep["seed"], "outcome": ep["outcome"],
         "event_time": round(ep["event_step"] * DT, 10) if ep["event_step"] else None,
         "censored_at_30s": ep["outcome"] == "survived_horizon", "steps_run": ep["steps_run"],
         "post_death_steps_run": ep["post_death_steps_run"]}
    na = "not observed before death/horizon"
    samples = [(0.0, fx["d"])]
    for t in T:
        if t["dist_before_kill_check"] is not None:
            samples.append((t["t"], t["dist_before_kill_check"]))
        samples.append((t["t"], t["dist_post_step"]))
    mt, md = min(samples, key=lambda x: x[1])
    s["min_separation"], s["min_separation_time"] = md, mt
    for m in ("chase", "circle", "edge_avoid", "wander", "sleep", "ambiguous", "unidentified"):
        s["time_" + m] = round(sum(DT for t in T if t["mode"] == m), 10)
    s["predicate_mismatches"] = sum(1 for t in T if t["predator_decision"] and t["predator_decision"]["predicate_mode"] != t["predator_decision"]["observed_mode"])
    s["decision_steps"] = sum(1 for t in T if t["predator_decision"])
    s["eligible_circle_ticks"] = sum(1 for t in T if t["predator_decision"] and t["predator_decision"]["observed_mode"] == "circle")
    s["eligible_edge_avoid_ticks"] = sum(1 for t in T if t["predator_decision"] and t["predator_decision"]["observed_mode"] == "edge_avoid")  # expected 0, see B4
    s["nonzero_pivot_ticks"] = sum(1 for t in T if t["predator_decision"] and t["predator_decision"].get("pivot_sign") not in (None, 0.0))
    s["any_edge_ever_in_controller_obs"] = any(t["any_edge_in_controller_obs"] for t in T)
    s["any_edge_ever_in_pred_obs"] = any(t.get("any_edge_in_pred_obs") for t in T)
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
    if T:
        last = T[-1]
        s["final_t"] = last["t"]
        s["final_agent_energy"] = last["agent_energy_post"] if last["agent_alive"] else None
        s["final_separation"] = last["dist_post_step"] if last["agent_alive"] else None
        s["final_pred_energy"] = last["pred_energy_post"]
        s["pred_asleep_at_end"] = last["pred_resting_post"]
        s["agent_energy_used"] = (fx["agent_energy"] - last["agent_energy_post"]) if last["agent_alive"] else None
        s["agent_move_cost"] = sum(t["agent_move"]["move_cost"] for t in T if t["agent_move"])
        s["agent_turn_cost"] = sum(t["agent_turn"]["turn_cost"] for t in T if t["agent_turn"])
        # displacement from the PREDATOR's own start (d,0) vs. distance from the fixed reference
        # point (the agent's initial position, reporting-frame origin) are kept distinct (Opus B/12).
        s["pred_displacement_along_escape_from_own_start"] = -(last["pred_xy"][0] - fx["d"])
        s["pred_dist_from_ref_final"] = math.hypot(*last["pred_xy"])
    # Pre-kill vs post-eat predator energy, and predator/agent state at the kill tick (spec S4,
    # Opus B3): "before" = last pre-kill snapshot on the tick that killed the agent (after the
    # predator's own move/turn cost, before eating); "after" = same tick, after eating.
    # Guarded by outcome=="captured" (Opus B3/BLOCKING-2 recheck): w_kill now only sets
    # pred_energy_post_kill for an actual predator kill, never for an energy-death removal, but
    # this extra guard keeps summarize() correct even if a future harness edit loosens that.
    kill_tick = (next((t for t in T if t.get("pred_energy_post_kill") is not None), None)
                 if ep["outcome"] == "captured" else None)
    s["pred_energy_at_kill_pre"] = kill_tick["pred_energy_pre_kill"] if kill_tick else None
    s["pred_energy_at_kill_post"] = kill_tick["pred_energy_post_kill"] if kill_tick else None
    s["pred_xy_at_kill"] = kill_tick["pred_xy"] if kill_tick else None
    s["killed_agent_energy_pre_removal"] = kill_tick["killed_agent_energy_pre_removal"] if kill_tick else None
    if PT:
        # Post-death continuation is counterfactual and excluded from survival metrics (spec S2);
        # only predator-side fields are read here, never the (removed) agent (Opus B2).
        snap = {}
        for target_t in (1.0, 5.0, 10.0):
            cand = next((p for p in PT if abs(p["t_since_death"] - target_t) < 1e-9), None)
            snap[f"+{target_t:g}s"] = {"pred_xy": cand["pred_xy"], "pred_energy": cand["pred_energy"],
                                        "pred_resting": cand["pred_resting"],
                                        "dist_to_ref": cand["dist_to_initial_agent_ref"]} if cand else None
        s["post_death_snapshots"] = snap
        s["post_death_final_pred_xy"] = PT[-1]["pred_xy"]
        s["post_death_min_dist_to_ref"] = min(p["dist_to_initial_agent_ref"] for p in PT)
        s["post_death_wake_events"] = sum(1 for p in PT if p["pred_transition"] == "woke")
        s["post_death_sleep_events"] = sum(1 for p in PT if p["pred_transition"] == "fell_asleep")
    s["runtime_sec"] = ep["runtime_sec"]
    s["trajectory_digest_norng"] = ep["trajectory_digest_norng"][:16]
    s["trajectory_digest"] = ep["trajectory_digest"][:16]
    s["validity"] = ("valid" if s["predicate_mismatches"] == 0 and not s["any_edge_ever_in_controller_obs"]
                      and not s["any_edge_ever_in_pred_obs"] and s["eligible_edge_avoid_ticks"] == 0
                      else "INVALID")
    return s


def _jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ---------------------------------------------------------------- commands
def cmd_emit_config(args):
    OUT.mkdir(parents=True, exist_ok=True)
    order = build_main_run_order()
    cfg = {
        "experiment": "C1-MECH-P02",
        "type": "RUN (mechanism study, synthetic fixture; not a competition score)",
        "source_revision": {"src_branch": "challenge-1", "src_commit_note": "identical to main@d9b5d9c",
                             "harness_commit_origin": "7b6805c (training/predator_mechanics.py, copied unmodified)",
                             "worktree_head_at_prep": "4ea9ca3"},
        "environment": {"os": "Windows 11", "python": sys.version.split()[0],
                         "numpy": __import__("numpy").__version__, "scipy": __import__("scipy").__version__,
                         "shapely": __import__("shapely").__version__, "pygame": __import__("pygame").__version__,
                         "pydantic": __import__("pydantic").__version__, "matplotlib": __import__("matplotlib").__version__},
        "fixture_matrix": {
            "initial_separation_d": SEPARATIONS, "initial_agent_energy": AGENT_ENERGIES,
            "initial_predator_energy": PRED_ENERGIES, "seeds": SEEDS,
            "speeds": SPEEDS, "deltas_rad": {tag: dv for tag, dv in DELTAS},
            "arms": {a: {"speed": ARMS[a][0], "delta_rad": ARMS[a][1], "delta_deg": math.degrees(ARMS[a][1])} for a in ARMS},
            "zero_offset_arms": ZERO_ARMS, "signed_offset_arms": SIGNED_ARMS,
            "main_episodes": len(order),
        },
        "controller_definition": {
            "spec_ref": "task_prompt.md Section 2",
            "step1": "move_direction = wrap(ESCAPE - heading_acc); move_distance = arm_speed; ESCAPE = pi (-x)",
            "step2": "if predator in controller observation with bearing beta: turn_angle = wrap(beta + delta); else turn_angle = 0",
            "step3": "heading_acc = wrap(heading_acc + turn_angle), updated AFTER issuing the turn",
            "step4": "move-before-turn native ordering preserved; exactly one action per living agent per step; no spawning; speed requested unchanged even if engine clamps it",
            "note": "delta is an offset from the OLD observed bearing at controller decision time, not the true rel_dir at the predator's decision; both are logged separately per tick",
        },
        "fixture_state": {
            "agent": {"reporting_xy": [0, 0], "internal_xy": [AXY0, AXY0], "heading": 0.0, "age": 0,
                      "speed": 10, "sprint_speed": 20, "max_energy": 500, "max_age": 120, "size": 5,
                      "hearing": 50, "vision": 200, "cone": "pi/3"},
            "predator": {"reporting_xy": ["d", 0], "heading": "pi", "resting": False, "speed": 11,
                         "sprint_speed": 15, "max_energy": 200, "size": 10, "hearing": 60, "vision": 250, "cone": "pi/3"},
            "world": {"width": WORLD, "height": WORLD, "boundary_thickness": 30,
                      "biome": "Grassland (move_penalty 1.0, energy_drain_rate 1.0) everywhere",
                      "chunk_size": CHUNK, "obstacles": "4 boundary walls only", "fruits": 0, "trees": 0},
            "horizon": {"dt": DT, "living_steps": LIVING_STEPS, "post_death_steps": POST_DEATH_STEPS},
        },
        "deviations": [
            "Environment.__new__ used instead of Environment.__init__ (skips random biome map generation and pygame surface pre-rendering; both rendering/world-gen only).",
            "spawn_tree/spawn_fruit/spawn_fruit_around_tree/spawn_predator are instance-level no-ops; their rng.random() gates in non_agent_step still consume RNG identically to a real game.",
            f"chunk_size enlarged to {CHUNK} (vs. simulator default 400) solely so the 4 boundary edges are always members of _get_local_edges/_get_local_objects, avoiding the empty-edge-list IndexError in src/utils/sensing.py compute_visibility (see source_notes.md). Verified by smoke that no edge is ever a hit_edge (boundary is 7850+ units away, vision radii are 200/250).",
            "16000x16000 arena (vs. P01's 6000x6000) to give clearance for the 40s (400-step) horizon; this is the geometry validated against P01 via the compat command before any P02 main-matrix execution, per spec S1.",
            "Post-death continuation (100 native steps, step_environment(env, [], dt)) is explicitly counterfactual relative to the real 1-agent-game extinction rule; its time axis is relative to death and excluded from survival metrics.",
        ],
        "compat_check": {
            "spec_ref": "task_prompt.md Section 3, six P01 geometry compatibility runs",
            "method": "Run ORIGINAL P01 arena/controller (training.predator_mechanics.run_episode) arms F10/F15/F20 "
                      "(move along escape line at speed 10/15/20, turn to observed predator angle -- algebraically "
                      "identical to P02's delta=0 controller) for fixtures d120/ae150/pe200 and d120/ae300/pe101, "
                      "seed 0, <=100 steps/10s. Compare against matching P02 zero-offset runs (S10_D0/S15_D0/S20_D0, "
                      "same fixtures/seed) over the first 10s or common pre-death interval: normalized coordinates "
                      "(subtract each arena's own agent-start origin), identical discrete actions/modes/events, "
                      "|err|<=1e-8 for physical quantities (report actual max error). Also diff against historical "
                      "P01 traces in the main checkout (read-only): " + str(P01_MAIN_TRACES),
            "episodes_budgeted": 6,
        },
        "repeat_check": {"fixture": list(REP_FIXTURE_FOR_REPEATS), "seed": REP_SEED, "arms": list(ARMS.keys()),
                          "episodes_budgeted": {"fresh_process_repeat": 9, "instrumentation_disabled_nolog": 9}},
        "caps": {"max_full_episodes": 240, "max_smoke_episodes": 12, "max_smoke_steps_each": 5,
                 "max_sim_wallclock_sec": 600, "episode_step_cap_main": 400, "episode_step_cap_compat": 100},
        "commands": {
            "emit-config": "python -m training.predator_mechanics_p02 emit-config  (0 episodes, writes this file)",
            "smoke": "python -m training.predator_mechanics_p02 smoke  (<=12 episodes, <=5 steps each)",
            "compat": "python -m training.predator_mechanics_p02 compat  (6 episodes; NOT run in phase 1)",
            "main": "python -m training.predator_mechanics_p02 main  (216 episodes in the run_order below; NOT run in phase 1)",
            "repeat": "python -m training.predator_mechanics_p02 repeat --tag repeat|nolog  (9 episodes each; NOT run in phase 1)",
        },
        "main_run_order": order,
    }
    with open(OUT / "config.json", "w") as f:
        json.dump(cfg, f, indent=1, default=_jsonable)
    print(f"wrote {OUT / 'config.json'} with {len(order)} predeclared main episodes (0 episodes run)")


def cmd_smoke(args):
    """<=12 cases, <=5 steps each. Checks required by spec Section 3."""
    global LIVING_STEPS
    res = []
    used_episodes = [0]

    def ep_run(*a, **k):
        used_episodes[0] += 1
        return run_episode(*a, living_steps=5, post_death_steps=0, **k)

    def check(name, ok, detail):
        res.append({"case": name, "ok": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + " :: " + str(detail), flush=True)

    def _sign(x):
        return int(x > 0) - int(x < 0)

    all_eps = []

    def tick0_dec(ep):
        return ep["ticks"][0]["predator_decision"]

    # 1a/1b. Signed offsets at d=120 and d=80 (Opus checklist item 3): at tick 1 the predator's
    # decision uses the agent's JUST-turned direction (agent_step runs before non_agent_step's
    # predator block), so rel_dir = beta_bearing(0) - delta = -delta exactly; pivot_sign =
    # -sign(rel_dir) = sign(delta); and the predator's move direction pi + sign(delta)*pi/4 makes
    # its y-displacement carry sign -sign(delta) -- i.e. the predator circles to the side OPPOSITE
    # the agent's gaze offset, not the same side (do not assume same-side handedness).
    for d in (120, 80):
        ep_p = ep_run(d, 300, 101, "S10_D+15", 0)
        ep_m = ep_run(d, 300, 101, "S10_D-15", 0)
        all_eps += [ep_p, ep_m]
        dp, dm = tick0_dec(ep_p), tick0_dec(ep_m)
        dp_y, dm_y = ep_p["ticks"][0]["pred_xy"][1], ep_m["ticks"][0]["pred_xy"][1]
        delta_p, delta_m = math.pi / 12, -math.pi / 12
        ok = bool(dp and dm and dp["observed_mode"] == "circle" and dm["observed_mode"] == "circle"
                  and abs(dp["rel_dir"] - (-delta_p)) < 1e-9 and abs(dm["rel_dir"] - (-delta_m)) < 1e-9
                  and dp["pivot_sign"] == _sign(delta_p) and dm["pivot_sign"] == _sign(delta_m)
                  and _sign(dp_y) == -_sign(delta_p) and _sign(dm_y) == -_sign(delta_m))
        check(f"signed_offset_mirrored_pivot_and_ydisp_d{d}", ok,
              {"decision_dist": dp and dp.get("decision_dist"),
               "plus15": {"rel_dir": dp and dp["rel_dir"], "pivot_sign": dp and dp["pivot_sign"], "pred_y": dp_y},
               "minus15": {"rel_dir": dm and dm["rel_dir"], "pivot_sign": dm and dm["pivot_sign"], "pred_y": dm_y}})

    # 2. Zero-offset control stays exactly collinear (rel_dir==0, pivot_sign==0) at tick 1, d=120.
    #    Reused for the move-along-escape-line / move-before-turn checks (spec S4).
    ep0 = ep_run(120, 300, 101, "S10_D0", 0)
    all_eps.append(ep0)
    d0 = tick0_dec(ep0)
    check("zero_offset_control_exact_zero_pivot_d120", bool(d0 and d0["rel_dir"] == 0.0 and d0["pivot_sign"] == 0.0),
          {"pivot_sign": d0 and d0["pivot_sign"], "rel_dir": d0 and d0["rel_dir"]})
    t1 = ep0["ticks"][0]
    check("move_along_escape_line_first_step", abs(t1["agent_xy"][0] + t1["agent_move"]["realized"]) < 1e-6 and abs(t1["agent_xy"][1]) < 1e-9,
          {"agent_xy": t1["agent_xy"], "realized": t1["agent_move"]["realized"]})
    check("move_before_turn_ordering", t1["agent_move"] is not None and t1["agent_turn"] is not None, {"move": t1["agent_move"], "turn": t1["agent_turn"]})

    # 3. Chase rule holds at a decision distance genuinely <90 (d=50+speed10=60), regardless of
    #    offset -- NOT tested at d=80, where decision_dist=d+speed>=90 lands on/inside the circle
    #    side of the threshold (Opus B1/checklist item 3 note), so d=80 is covered above instead.
    ep = ep_run(50, 300, 101, "S10_D+15", 0)
    all_eps.append(ep)
    mm = [(t["predator_decision"]["decision_dist"], t["predator_decision"]["predicate_mode"], t["predator_decision"]["observed_mode"])
          for t in ep["ticks"] if t["predator_decision"] and t["predator_decision"].get("decision_dist") is not None]
    check("chase_rule_holds_under_offset_d50", bool(mm) and all(mode == "chase" for _, mode, _ in mm) and all(m == o for _, m, o in mm), mm)

    # 4. Below-40 predator clamping (max_energy=200 -> 20% = 40), kill-before-sleep ordering,
    #    wake-and-act timing: predator energy set just above sprint-disable threshold near a kill.
    # d=1 (not 20): at d=20 the near-zero-angle chase move (min(sprint_speed, distance), capped
    # further by the agent's own 10/step retreat) only closes ~1 unit/tick and never reaches the
    # <15 kill threshold within 5 steps; d=1 guarantees a tick-1 kill (post-retreat separation 11,
    # within the predator's single-tick move budget) so this case is testable within the 5-step cap.
    ep = ep_run(1, 300, 39.0, "S10_D0", 0)
    all_eps.append(ep)
    t1 = ep["ticks"][0]
    check("kill_before_sleep_and_energy_capture", ep["outcome"] == "captured" and t1["pred_energy_post_kill"] is not None
          and t1["pred_energy_pre_kill"] is not None and t1["pred_energy_post_kill"] > t1["pred_energy_pre_kill"]
          and t1["pred_resting_post"] == (t1["pred_energy_post"] <= 0),
          {"pre_kill": t1["pred_energy_pre_kill"], "post_kill": t1["pred_energy_post_kill"],
           "post_energy": t1["pred_energy_post"], "resting_post": t1["pred_resting_post"]})

    # 5. Logger non-interference: full vs digest mode over 5 steps, same digests.
    a = ep_run(120, 300, 101, "S15_D+15", 0, mode="full")
    b = ep_run(120, 300, 101, "S15_D+15", 0, mode="digest")
    all_eps.append(a)
    check("instrumentation_noninterference_5step", a["step_digests"] == b["step_digests"], [a["trajectory_digest"][:12], b["trajectory_digest"][:12]])

    # 6. Predicate-vs-observed-mode agreement across a full 5-tick arm at d=120 (S15_D0).
    ep = ep_run(120, 300, 101, "S15_D0", 0)
    all_eps.append(ep)
    mm = [(t["predator_decision"]["predicate_mode"], t["predator_decision"]["observed_mode"]) for t in ep["ticks"] if t["predator_decision"]]
    check("predicate_vs_observed_S15_D0_d120", all(a == b for a, b in mm), mm)

    # 7. No boundary edge ever enters either actor's observation, and the edge-avoid branch is
    #    never eligible, across every episode run above (16000 arena / chunk 8000 workaround,
    #    Opus B4) -- audited per tick, not merely asserted from arena geometry.
    edge_ctrl = any(t["any_edge_in_controller_obs"] for e in all_eps for t in e["ticks"])
    edge_pred = any(t.get("any_edge_in_pred_obs") for e in all_eps for t in e["ticks"])
    edge_avoid_ticks = sum(1 for e in all_eps for t in e["ticks"] if t["predator_decision"] and t["predator_decision"]["observed_mode"] == "edge_avoid")
    check("no_edge_in_any_observation_across_smoke_episodes", not edge_ctrl and not edge_pred and edge_avoid_ticks == 0,
          {"any_edge_in_controller_obs": edge_ctrl, "any_edge_in_pred_obs": edge_pred,
           "edge_avoid_eligible_ticks": edge_avoid_ticks, "episodes_audited": len(all_eps)})

    json.dump(res, open(OUT / "smoke.json", "w"), indent=1, default=_jsonable)
    n_fail = sum(1 for r in res if not r["ok"])
    print(f"smoke cases: {len(res)}, failures: {n_fail}, episodes used: {used_episodes[0]} (budget 12, reserve kept: {12 - used_episodes[0]})")


def _capture_env(module, fn, *args, **kwargs):
    """Runs fn (a run_episode) with module.build_env temporarily wrapped so the constructed
    Environment (and its rng) is captured for a post-hoc RNG-state comparison. module.build_env
    is restored unconditionally. fn must call the unqualified name `build_env` at module-global
    scope (true for both training.predator_mechanics.run_episode and this module's run_episode)."""
    orig = module.build_env
    captured = {}

    def wrapped(*a, **k):
        result = orig(*a, **k)
        captured["env"] = result[0]
        return result

    module.build_env = wrapped
    try:
        ep = fn(*args, **kwargs)
    finally:
        module.build_env = orig
    return ep, captured.get("env")


def _map_mode(mode, is_p01):
    # P01 has no edge_avoid/wander split; its single "no_target" corresponds to P02's "wander"
    # (P02's compat fixtures/geometry never reach the edge_avoid branch -- see source_notes.md B4).
    return "wander" if (is_p01 and mode == "no_target") else mode


def _compare_ticks(ticks_a, ticks_b):
    """Per-tick comparison per Opus stage-2 review: |err|<=1e-8 for physical quantities (reports
    actual max error), exact equality for discrete mode/event fields. ticks_a is P01, ticks_b is
    P02 (or historical-vs-fresh, either order is symmetric for these checks)."""
    n = min(len(ticks_a), len(ticks_b))
    max_err = {}

    def bump(key, err):
        max_err[key] = max(max_err.get(key, 0.0), abs(err))

    mismatches = []
    for i in range(n):
        a, b = ticks_a[i], ticks_b[i]
        for k in ("agent_xy", "pred_xy"):
            for j in range(2):
                bump(k, a[k][j] - b[k][j])
        for k in ("agent_dir", "pred_dir", "agent_energy_post", "pred_energy_post"):
            bump(k, a[k] - b[k])
        for k in ("move_distance", "move_direction", "turn_angle"):
            bump("action_" + k, a["action"][k] - b["action"][k])
        pa = a.get("controller_obs_predators") or []
        pb = b.get("controller_obs_predators") or []
        if pa and pb:
            bump("obs_pred_distance", pa[0]["distance"] - pb[0]["distance"])
            bump("obs_pred_angle", pa[0]["angle"] - pb[0]["angle"])
        elif bool(pa) != bool(pb):
            mismatches.append([i, "controller_obs_predators_presence", bool(pa), bool(pb)])
        da, db = a.get("predator_decision"), b.get("predator_decision")
        if da and db:
            for k in ("rel_dir", "decision_dist"):
                if da.get(k) is not None and db.get(k) is not None:
                    bump("decision_" + k, da[k] - db[k])
            ma, mb = _map_mode(da["observed_mode"], True), _map_mode(db["observed_mode"], False)
            if ma != mb:
                mismatches.append([i, "observed_mode", ma, mb])
            if da.get("pivot_sign") != db.get("pivot_sign"):
                mismatches.append([i, "pivot_sign", da.get("pivot_sign"), db.get("pivot_sign")])
        elif bool(da) != bool(db):
            mismatches.append([i, "predator_decision_presence", bool(da), bool(db)])
        ma_mode, mb_mode = _map_mode(a["mode"], True), _map_mode(b["mode"], False)
        if ma_mode != mb_mode:
            mismatches.append([i, "mode", ma_mode, mb_mode])
        if a.get("pred_transition") != b.get("pred_transition"):
            mismatches.append([i, "pred_transition", a.get("pred_transition"), b.get("pred_transition")])
        if a["agent_alive"] != b["agent_alive"]:
            mismatches.append([i, "agent_alive", a["agent_alive"], b["agent_alive"]])
    return {"common_ticks": n, "len_a": len(ticks_a), "len_b": len(ticks_b),
            "truncated_at_earlier_death": len(ticks_a) != len(ticks_b),
            "max_abs_err": max_err, "n_mismatches": len(mismatches), "mismatches": mismatches[:50]}


def _load_historical_trace(d, ae, pe, arm_p01):
    p = P01_MAIN_TRACES / f"d{d}_ae{ae}_pe{pe}_{arm_p01}_s0.json.gz"
    if not p.exists():
        return None
    with gzip.open(p, "rt") as fh:
        return json.load(fh)


def cmd_compat(args):
    """6 P01-arena/controller compatibility episodes. NOT executed in phase 1 -- code path only.
    Fixes Opus stage-2 BLOCKING-1: full per-tick physical/discrete comparison (not just two energy
    fields + a dead-code line), RNG-state equality via _capture_env, and a diff against the
    historical P01 traces in the main checkout (read-only)."""
    import training.predator_mechanics as p01
    self_mod = sys.modules[__name__]
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for (d, ae, pe) in REP_FIXTURES:
        for arm_p01, p02_arm in (("F10", "S10_D0"), ("F15", "S15_D0"), ("F20", "S20_D0")):
            ep_p01, env_p01 = _capture_env(p01, p01.run_episode, d, ae, pe, arm_p01, REP_SEED, "full")
            ep_p02, env_p02 = _capture_env(self_mod, run_episode, d, ae, pe, p02_arm, REP_SEED, "full",
                                            living_steps=100, post_death_steps=0)
            cmp_live = _compare_ticks(ep_p01["ticks"], ep_p02["ticks"])
            rng_p01 = repr(env_p01.rng.getstate()) if env_p01 else None
            rng_p02 = repr(env_p02.rng.getstate()) if env_p02 else None
            events_equal = (ep_p01["outcome"] == ep_p02["outcome"] and ep_p01["event_step"] == ep_p02["event_step"]
                             and len(ep_p01["ticks"]) == len(ep_p02["ticks"]))
            fail = (cmp_live["n_mismatches"] > 0 or any(v > 1e-8 for v in cmp_live["max_abs_err"].values())
                    or not events_equal or rng_p01 != rng_p02)
            entry = {"fixture": [d, ae, pe], "arm_p01": arm_p01, "arm_p02": p02_arm,
                     "outcome_p01": ep_p01["outcome"], "outcome_p02": ep_p02["outcome"],
                     "event_step_p01": ep_p01["event_step"], "event_step_p02": ep_p02["event_step"],
                     "events_equal": events_equal, "rng_state_equal": rng_p01 == rng_p02,
                     "live_comparison": cmp_live, "status": "fail" if fail else "pass"}
            hist = _load_historical_trace(d, ae, pe, arm_p01)
            if hist is not None:
                cmp_hist_p01 = _compare_ticks(hist["ticks"], ep_p01["ticks"])
                cmp_hist_p02 = _compare_ticks(hist["ticks"], ep_p02["ticks"])
                entry["historical_vs_fresh_p01"] = cmp_hist_p01
                entry["historical_vs_p02"] = cmp_hist_p02
                if cmp_hist_p01["n_mismatches"] > 0 or any(v > 1e-8 for v in cmp_hist_p01["max_abs_err"].values()):
                    entry["status"] = "fail"
            else:
                entry["historical_trace"] = f"not found at {P01_MAIN_TRACES} (read-only main checkout)"
            results.append(entry)
            if entry["status"] == "fail":
                print(f"COMPAT FAIL at fixture={(d, ae, pe)} arm={arm_p01}/{p02_arm}; stopping remaining compat matrix, preserving evidence.", flush=True)
                break
        else:
            continue
        break
    json.dump(results, open(OUT / "compat.json", "w"), indent=1, default=_jsonable)
    n_fail = sum(1 for r in results if r["status"] == "fail")
    print(f"compat pairs_run={len(results)} (of 6 planned) failures={n_fail}")


def cmd_main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    tdir = OUT / "traces"
    tdir.mkdir(exist_ok=True)
    order = build_main_run_order()
    rows, digests = [], {}
    t0 = time.perf_counter()
    n = 0
    for item in order:
        if time.perf_counter() - t0 > args.sim_cap_sec:
            print("SIM CAP REACHED; stopping", flush=True)
            break
        ep = run_episode(item["d"], item["agent_energy"], item["pred_energy"], item["arm"], item["seed"], "full")
        n += 1
        name = f"d{item['d']}_ae{item['agent_energy']}_pe{item['pred_energy']}_{item['arm']}_s{item['seed']}.json.gz"
        with gzip.open(tdir / name, "wt") as f:
            json.dump(ep, f, default=_jsonable)
        rows.append(summarize(ep))
        key = f"{item['d']}|{item['agent_energy']}|{item['pred_energy']}|{item['arm']}|{item['seed']}"
        digests[key] = {k: ep[k] for k in ("trajectory_digest", "trajectory_digest_norng", "final_state_digest", "initial_state_hash", "initial_rng_hash", "outcome", "final_score")}
    import csv
    if rows:
        with open(OUT / "summary.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    json.dump(digests, open(OUT / "digests_main.json", "w"), indent=1)
    print(f"main episodes={n} sim_wall={time.perf_counter()-t0:.1f}s", flush=True)


def cmd_repeat(args):
    d, ae, pe = REP_FIXTURE_FOR_REPEATS
    mode = "digest" if args.tag == "nolog" else "full"
    out = {}
    t0 = time.perf_counter()
    for arm in ARMS:
        ep = run_episode(d, ae, pe, arm, REP_SEED, mode)
        out[arm] = {k: ep[k] for k in ("trajectory_digest", "trajectory_digest_norng", "final_state_digest", "step_digests", "outcome", "event_step", "final_score", "initial_state_hash", "initial_rng_hash")}
    json.dump(out, open(OUT / f"digests_{args.tag}.json", "w"), indent=1)
    print(f"{args.tag} episodes={len(ARMS)} sim_wall={time.perf_counter()-t0:.2f}s", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["emit-config", "smoke", "main", "compat", "repeat"])
    p.add_argument("--tag", default="repeat")
    p.add_argument("--sim-cap-sec", type=float, default=240.0)
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"emit-config": cmd_emit_config, "smoke": cmd_smoke, "main": cmd_main,
     "compat": cmd_compat, "repeat": cmd_repeat}[a.cmd](a)
