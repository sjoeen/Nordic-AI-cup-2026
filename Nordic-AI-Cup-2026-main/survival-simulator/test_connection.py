"""
Connection test for the submission endpoint. Run from any machine (e.g. your laptop)
against the public Azure address, or against localhost.

    python test_connection.py http://<AZURE_PUBLIC_IP>:9052            # health + latency
    python test_connection.py http://<AZURE_PUBLIC_IP>:9052 --game     # + full simulated game
    python test_connection.py http://<AZURE_PUBLIC_IP>:9052 --expect-fallback  # after INJECT_FAILURE=1

Budget reminder: the evaluator ends a run when accumulated wait reaches 600 s.
With up to 30000 steps that is a mean round trip below 20 ms.
"""
import argparse
import os
import statistics
import sys
import time

import requests

STEP_LIMIT = 30000
WAIT_BUDGET_S = 600.0


def synthetic_step(n_agents: int = 5) -> dict:
    agent = {
        "energy": 100.0, "biome": "grass", "age": 1.0, "speed": 1.0, "sprint_speed": 2.0,
        "hearing_radius": 5.0, "vision_angle": 1.5, "vision_range": 10.0, "max_energy": 200.0,
        "observations": [{"type": "fruit", "distance": 3.0, "angle": 0.2}],
    }
    return {
        "game_status": "ok", "score": 0.0, "sim_time": 0.0, "n_agents": n_agents,
        "agent_status": [{"agent_id": i, **agent} for i in range(n_agents)],
    }


def check(ok: bool, msg: str) -> bool:
    print(f"[{'PASS' if ok else 'FAIL'}] {msg}")
    return ok


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base_url")
    p.add_argument("-n", type=int, default=200, help="latency samples")
    p.add_argument("--game", action="store_true", help="run a full local simulation against the endpoint")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--expect-fallback", action="store_true")
    args = p.parse_args()
    base = args.base_url.rstrip("/")
    ok = True

    r = requests.get(base + "/", timeout=10)
    ok &= check(r.status_code == 200, f"GET / -> {r.status_code} {r.text[:80]}")

    before = requests.get(base + "/api", timeout=10).json()
    ok &= check(True, f"GET /api -> policy={before.get('policy')} api_key_configured={before.get('api_key_configured')}")

    payload = synthetic_step()
    r = requests.post(base + "/predict", json=payload, timeout=10)
    body = r.json()
    acts = body.get("actions", [])
    keys = {"agent_id", "move_distance", "move_direction", "turn_angle", "spawn_agent"}
    ok &= check(r.status_code == 200 and len(acts) == 5 and all(keys <= set(a) for a in acts),
                f"POST /predict -> {r.status_code}, {len(acts)} valid actions")
    ok &= check(sorted(a["agent_id"] for a in acts) == list(range(5)), "one action per agent_id")

    s = requests.Session()
    lat = []
    for _ in range(args.n):
        t0 = time.perf_counter()
        s.post(base + "/predict", json=payload, timeout=10).raise_for_status()
        lat.append((time.perf_counter() - t0) * 1000)
    lat.sort()
    mean = statistics.mean(lat)
    p95 = lat[int(0.95 * (len(lat) - 1))]
    projected = mean * STEP_LIMIT / 1000
    print(f"latency ms over n={args.n}: mean={mean:.1f} median={statistics.median(lat):.1f} "
          f"p95={p95:.1f} max={lat[-1]:.1f}")
    ok &= check(projected < WAIT_BUDGET_S,
                f"projected wait for {STEP_LIMIT} steps = {projected:.0f} s (budget {WAIT_BUDGET_S:.0f} s)")

    after = requests.get(base + "/api", timeout=10).json()
    new_fallbacks = after["fallback_agents"] - before["fallback_agents"]
    if args.expect_fallback:
        ok &= check(new_fallbacks > 0, f"fallback fired: {new_fallbacks} fallback agent-actions")
    else:
        ok &= check(new_fallbacks == 0, f"no fallbacks during test ({new_fallbacks})")

    if args.game:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from simulation_server import run_local_game  # needs full requirements.txt (pygame etc.)
        t0 = time.time()
        run_local_game(base + "/predict", seed=args.seed)
        print(f"full game wall clock: {time.time() - t0:.1f} s")

    print("ALL PASS" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
