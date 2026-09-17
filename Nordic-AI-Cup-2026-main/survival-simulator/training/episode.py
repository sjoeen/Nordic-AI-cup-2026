"""
Headless episode runner that mirrors the official evaluator loop (simulation_server.py):
  - first step is taken with no actions,
  - the agent receives only live agents' ObservationResponse dicts,
  - the game ends when no agents remain or sim time exceeds 3000 s,
  - the final score is the score of the last state.
"""
import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")  # headless pygame (cluster has no display)
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import numpy as np

from src.core import SimulationCore
from src.utils.DTOs import ActionRequest

MAX_SIM_TIME = 3000


def run_episode(agent, seed: int, max_wall_sec: float = None, max_sim_time: float = MAX_SIM_TIME,
                on_step=None) -> dict:
    """Run one game. on_step(sim, state, step_idx) is an optional hook (used for rendering).
    Stops early (status='capped') if max_wall_sec or max_sim_time is exceeded."""
    t_start = time.perf_counter()
    sim = SimulationCore(seed=seed)
    t_env = time.perf_counter() - t_start
    n_start = len(sim.env.agents)

    state = sim.step([])
    steps, peak_agents, latencies_ms = 1, state["num_agents"], []
    status = "ok"

    while state["num_agents"] > 0 and sim.env.time <= MAX_SIM_TIME:
        if sim.env.time > max_sim_time:
            status = "capped_sim_time"
            break
        if max_wall_sec is not None and time.perf_counter() - t_start > max_wall_sec:
            status = "capped_wall_clock"
            break

        agent_states = [o for o in state["observations"] if o is not None]
        t0 = time.perf_counter()
        actions = agent.act_batch(agent_states)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        parsed = [(a["agent_id"], ActionRequest(**a)) for a in actions]  # raises on illegal actions
        state = sim.step(parsed)
        steps += 1
        peak_agents = max(peak_agents, state["num_agents"])
        if on_step is not None:
            on_step(sim, state, steps)

    lat = np.array(latencies_ms) if latencies_ms else np.array([0.0])
    return {
        "seed": seed,
        "status": status,
        "score": float(state["score"]),
        "sim_time": float(sim.env.time),
        "steps": steps,
        "survived_full_game": bool(state["num_agents"] > 0),
        "agents_alive_end": state["num_agents"],
        "peak_agents": peak_agents,
        "agents_spawned": sim.env._next_agent_id - n_start,
        "predators_end": len(sim.env.predators),
        "score_minus_time": float(state["score"] - sim.env.time),  # fruit bonus minus predation penalty
        "latency_mean_ms": float(lat.mean()),
        "latency_p95_ms": float(np.percentile(lat, 95)),
        "latency_max_ms": float(lat.max()),
        "env_init_sec": t_env,
        "wall_clock_sec": time.perf_counter() - t_start,
    }
