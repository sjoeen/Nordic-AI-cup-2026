"""
Submission endpoint for the Survival Simulator (Nordic AI Cup 2026, Challenge 1).

Serves POST /predict in the format the evaluation service expects:
    request:  StepResponse JSON
    response: {"actions": [ActionRequest, ...]}

The policy is currently the starter dummy policy (connection test only).

Safety: every step is wrapped in try/except. If the policy fails for an agent,
that agent gets a fallback action (no move, no turn, no spawn), which is valid
under the ActionRequest DTO. Fallback counts are exposed on GET /api.

API key: set the NAIC_API_KEY environment variable (never commit the key).
The starter template does not show how the evaluation service sends the key,
so it is NOT enforced by default. On each request we log whether (and in which
header) the key arrived, without logging the key itself. Once that is known,
set REQUIRE_API_KEY=1 to reject requests without it.

Failure-injection test: set INJECT_FAILURE=1 to make the policy raise on every
agent; responses must still be valid and fallback_agents must increase.
"""
import datetime
import logging
import os
import random
import time

import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request

from src.utils.DTOs import ActionRequest, StepResponse
from src.utils.controllers.dummy_agent_policy import action_decision

HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "9052"))
API_KEY = os.environ.get("NAIC_API_KEY", "")
REQUIRE_API_KEY = os.environ.get("REQUIRE_API_KEY", "0") == "1"
INJECT_FAILURE = os.environ.get("INJECT_FAILURE", "0") == "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("submission")

app = FastAPI(title="Survival Simulator Submission Endpoint")
start_time = time.time()
stats = {
    "requests": 0,
    "fallback_agents": 0,
    "failed_requests": 0,
    "key_seen_in_header": None,
    "last_sim_time": None,
    "last_score": None,
    "total_handler_ms": 0.0,
    "max_handler_ms": 0.0,
}


def fallback_action(agent_id: int) -> dict:
    return ActionRequest(
        agent_id=agent_id, move_distance=0.0, move_direction=0.0, turn_angle=0.0, spawn_agent=False
    ).model_dump()


def policy(agent: dict, rng: random.Random) -> dict:
    if INJECT_FAILURE:
        raise RuntimeError("injected failure")
    return action_decision(agent, rng).model_dump()


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
    check_api_key(request)
    stats["requests"] += 1
    actions = []
    try:
        step = StepResponse(**payload)
        stats["last_sim_time"], stats["last_score"] = step.sim_time, step.score
        rng = random.Random(1)  # starter dummy policy seeding
        for agent in step.agent_status:
            try:
                actions.append(policy(agent.model_dump(), rng))
            except Exception as e:
                stats["fallback_agents"] += 1
                log.error("policy failed for agent %s: %r", agent.agent_id, e)
                actions.append(fallback_action(agent.agent_id))
    except Exception as e:
        # Malformed payload: answer with fallbacks for any agent ids we can recover.
        stats["failed_requests"] += 1
        log.error("request handling failed: %r", e)
        ids = [a.get("agent_id") for a in payload.get("agent_status", []) if isinstance(a, dict)]
        actions = [fallback_action(i) for i in ids if isinstance(i, int)]
        stats["fallback_agents"] += len(actions)
    ms = (time.perf_counter() - t0) * 1000
    stats["total_handler_ms"] += ms
    stats["max_handler_ms"] = max(stats["max_handler_ms"], ms)
    return {"actions": actions}


@app.get("/api")
def api_status():
    n = stats["requests"]
    return {
        "service": "survival-simulator",
        "policy": "dummy_agent_policy",
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
    log.info("starting on %s:%s (api_key_configured=%s, require=%s, inject_failure=%s)",
             HOST, PORT, bool(API_KEY), REQUIRE_API_KEY, INJECT_FAILURE)
    uvicorn.run(app, host=HOST, port=PORT, access_log=False)
