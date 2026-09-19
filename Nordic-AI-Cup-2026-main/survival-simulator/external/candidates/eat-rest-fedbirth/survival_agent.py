"""eat-rest-fedbirth: the preserved eat-rest-v1 controller plus ONE rule on reproduction.

The baseline file (../original-eat-rest-preserved/survival_agent.py) is loaded unmodified and
subclassed; nothing in it is edited. Diagnosis on validation seeds showed late-game newborns
(75 energy, ~20 s of fuel) starving at age 13-27 while adults sat at 200+ energy, because births
ignore whether any food is near. Rule: a normal birth is allowed only when the parent currently
sees at least `birth_food_count` fruit within `birth_food_distance`; the parent otherwise keeps
its energy and gives birth later, next to food. Safety valve: if the youngest living agent is older
than `birth_food_patience` seconds the rule is waived, so a generation is never skipped entirely.
Emergency births of the baseline are untouched. With birth_food_distance=0 this IS the baseline.
"""
import importlib.util
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "original-eat-rest-preserved" / "survival_agent.py"
_spec = importlib.util.spec_from_file_location("_eat_rest_v1_base", _BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

STRATEGY_NAME = "appetite+fedbirth"
CONFIG = dict(base.CONFIG, birth_food_distance=150., birth_food_count=1, birth_food_patience=45.,
              birth_food_start=0.)


class FedBirthPolicy(base.AppetitePolicy):
    def __init__(self, preset="appetite", **overrides):
        config = dict(birth_food_distance=0., birth_food_count=1, birth_food_patience=45., birth_food_start=0.)
        config.update(overrides)
        super().__init__(preset, **config)

    def allocate(self, views):
        super().allocate(views)
        c = self.config
        if not c["birth_food_distance"] or self.clock < c["birth_food_start"] or not self.spawn_ids:
            return
        youngest = min((v["state"]["age"] for v in views.values()), default=0.)
        if youngest > c["birth_food_patience"]:
            self.metrics["fedbirth_waived"] += 1
            return
        fed = {aid for aid in self.spawn_ids
               if sum(f["obs"]["distance"] <= c["birth_food_distance"] for f in views[aid]["fruits"]) >= c["birth_food_count"]}
        self.metrics["fedbirth_blocked"] += len(self.spawn_ids) - len(fed)
        self.spawn_ids = fed


def make_policy():
    return FedBirthPolicy("appetite", **CONFIG)


# Reuse the baseline's FastAPI app and /predict handler (restart detection, duplicate-request cache);
# they look up make_policy/_policy in the baseline module, so point those at this policy.
base.make_policy = make_policy
base._policy = make_policy()
base.STRATEGY_NAME = STRATEGY_NAME
app, predict, StepResponse = base.app, base.predict, base.StepResponse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9052)
