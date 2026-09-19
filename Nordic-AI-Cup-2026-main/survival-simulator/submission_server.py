"""
Submission endpoint for the Survival Simulator (Nordic AI Cup 2026, Challenge 1).

Serves POST /predict in the format the evaluation service expects:
    request:  StepResponse JSON
    response: {"actions": [ActionRequest, ...]}

The policy is an external candidate: external/candidates/<AGENT_CANDIDATE>/survival_agent.py
(default original-eat-rest-preserved, "eat-rest-v1"). Its own /predict handler is called
in-process, so game-restart detection and duplicate-request handling stay exactly as tested.

Safety: every step is wrapped in try/except. If the candidate raises, returns an invalid
action, or skips an agent, the affected agents get a fallback action (no move, no turn, no
spawn), which is valid under the ActionRequest DTO. Fallback counts are exposed on GET /api.

API key: set the NAIC_API_KEY environment variable (never commit the key).
The starter template does not show how the evaluation service sends the key,
so it is NOT enforced by default. On each request we log whether (and in which
header) the key arrived, without logging the key itself. Once that is known,
set REQUIRE_API_KEY=1 to reject requests without it.

Failure-injection test: set INJECT_FAILURE=1 to make the policy raise on every
agent; responses must still be valid and fallback_agents must increase.
"""
import datetime
import importlib.util
import logging
import os
import time
from pathlib import Path

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

from src.utils.DTOs import ActionRequest

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "9052"))
API_KEY = os.environ.get("NAIC_API_KEY", "")
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "0") == "1"
INJECT_FAILURE = os.environ.get("INJECT_FAILURE", "0") == "1"
CANDIDATE = os.environ.get("AGENT_CANDIDATE", "original-eat-rest-preserved")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("submission")

app = FastAPI(title="Survival Simulator Submission Endpoint")
start_time = time.time()
stats = {
    "requests": 0,
    "fallback_agents": 0,
    "failed_requests": 0,
    "key_seen_in_header": None,
    "client_ips": [],
    "last_sim_time": None,
    "last_score": None,
    "total_handler_ms": 0.0,
    "max_handler_ms": 0.0,
}


def fallback_action(agent_id: int) -> dict:
    return ActionRequest(
        agent_id=agent_id, move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False
    ).model_dump()


def load_candidate(name: str):
    path = Path(__file__).resolve().parent / "external" / "candidates" / name / "survival_agent.py"
    spec = importlib.util.spec_from_file_location("submission_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


candidate = load_candidate(CANDIDATE)  # fail at startup, not mid-game, if the candidate is broken


def policy(payload: dict) -> dict:
    """Candidate actions keyed by agent_id, each validated against the ActionRequest DTO."""
    if INJECT_FAILURE:
        raise RuntimeError("injected failure")
    response = candidate.predict(candidate.StepResponse(**payload))
    return {a["agent_id"]: ActionRequest(**a).model_dump() for a in response["actions"]}


def check_api_key(request: Request) -> None:
    if not API_KEY:
        return
    header_name = next((k for k, v in request.headers.items() if API_KEY in v), None)
    if stats["requests"] == 0 or header_name != stats["key_seen_in_header"]:  # log on first request or change only
        log.info("API key %s", f"found in header '{header_name}'" if header_name else "not found in any header")
        stats["key_seen_in_header"] = header_name
    if REQUIRE_API_KEY and header_name is None:
        raise HTTPException(status_code=401, detail="missing API key")


@app.post("/predict")
def predict(request: Request, payload: dict = Body(...)):
    t0 = time.perf_counter()
    client_ip = request.client.host if request.client else None
    if client_ip not in stats["client_ips"]:  # reveals where the evaluator connects from
        stats["client_ips"].append(client_ip)
        log.info("new client %s", client_ip)
    check_api_key(request)
    stats["requests"] += 1
    stats["last_sim_time"], stats["last_score"] = payload.get("sim_time"), payload.get("score")
    ids = [a.get("agent_id") for a in payload.get("agent_status") or [] if isinstance(a, dict)]
    ids = [i for i in ids if isinstance(i, int)]
    try:
        decided = policy(payload)
    except Exception as e:
        stats["failed_requests"] += 1
        log.error("candidate failed at sim_time %s: %r", payload.get("sim_time"), e)
        decided = {}
    missing = [i for i in ids if i not in decided]
    if missing:
        stats["fallback_agents"] += len(missing)
    actions = [decided.get(i) or fallback_action(i) for i in ids]
    ms = (time.perf_counter() - t0) * 1000
    stats["total_handler_ms"] += ms
    stats["max_handler_ms"] = max(stats["max_handler_ms"], ms)
    return {"actions": actions}


@app.get("/api")
def api_status():
    n = stats["requests"]
    return {
        "service": "survival-simulator",
        "policy": CANDIDATE,
        "uptime": str(datetime.timedelta(seconds=time.time() - start_time)),
        "api_key_configured": bool(API_KEY),
        "require_api_key": REQUIRE_API_KEY,
        "inject_failure": INJECT_FAILURE,
        **stats,
        "mean_handler_ms": stats["total_handler_ms"] / n if n else None,
    }


@app.get("/")
def index():
    return {"message": "Agent endpoint running!"}


if __name__ == "__main__":
    log.info("starting on %s:%s with %s (api_key_configured=%s, require=%s, inject_failure=%s)",
             HOST, PORT, CANDIDATE, bool(API_KEY), REQUIRE_API_KEY, INJECT_FAILURE)
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)
