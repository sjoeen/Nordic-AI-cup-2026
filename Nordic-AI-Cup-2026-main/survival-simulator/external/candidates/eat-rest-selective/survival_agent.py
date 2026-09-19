"""eat-rest-selective: the preserved eat-rest-v1 controller plus gene-aware breeding.

The baseline file (../original-eat-rest-preserved/survival_agent.py) is loaded unmodified and
subclassed. Only ONE thing changes: some births the baseline would allow are postponed. Nothing
else (foraging, resting, escape, fruit allocation, emergency births, population cap) is touched,
and the rule can only remove parents from the baseline's own birth list, never add any.

Genes: a child copies its parent's traits; each trait mutates with 10% chance by up to +-50%.
Score = walking speed + sel_hearing_weight * hearing radius (predators walk 11 / sprint 15, agents
start at 10, so a fast walker escapes at cheap walking cost; hearing detects predators through walls).

Two mechanisms:
1. STRICT mode - only parents scoring >= sel_strict_fraction x the best breeder-age score may breed.
   It is ON only while ALL of these hold, and any failure turns it OFF for at least sel_cooldown s:
     - clock <= sel_strict_until                    (late game: never veto)
     - living agents >= sel_strict_min_agents        (small colony: never veto)
     - mean energy >= sel_strict_energy              (hungry colony: never veto)
     - no predator seen by anyone for sel_strict_calm s (default 0 = not required: measured on seeds
       2003/2010, somebody sees a predator on 70-90% of birth ticks, so this alone kept strict mode off;
       the baseline already refuses births when the PARENT has a predator within 130 units)
     - youngest agent <= sel_strict_patience s old   (births are flowing; a stall lifts the veto by itself)
     - at least sel_min_fertile_young permitted parents are younger than 60 s (the elite must not be all old)
2. STERILE marking - a newborn whose score is below sel_sterile_fraction x the colony median never
   breeds (it still eats and lives). Ignored whenever fewer than sel_min_fertile_young non-sterile
   agents younger than 60 s exist, so a colony can never sterilise itself to death.
With sel_enabled=0 this IS the baseline (verified action-for-action).
"""
import importlib.util
import statistics
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "original-eat-rest-preserved" / "survival_agent.py"
_spec = importlib.util.spec_from_file_location("_eat_rest_v1_base_selective", _BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

STRATEGY_NAME = "appetite+selective"
DEFAULTS = dict(sel_enabled=1, sel_hearing_weight=.03, sel_strict_fraction=.9, sel_strict_until=800.,
                sel_strict_min_agents=6, sel_strict_energy=150., sel_strict_calm=0., sel_strict_patience=30.,
                sel_cooldown=30., sel_sterile_fraction=.85, sel_min_fertile_young=2)
CONFIG = dict(base.CONFIG, **DEFAULTS)


class SelectivePolicy(base.AppetitePolicy):
    def __init__(self, preset="appetite", **overrides):
        config = dict(DEFAULTS, sel_enabled=0)
        config.update(overrides)
        super().__init__(preset, **config)
        self.sel_sterile = set()
        self.sel_seen = set()
        self.sel_last_threat = -1e9
        self.sel_off_until = -1e9

    def sel_score(self, s):
        return min(s["speed"], s["sprint_speed"]) + self.config["sel_hearing_weight"] * s["hearing_radius"]

    def allocate(self, views):
        super().allocate(views)
        c = self.config
        if not c["sel_enabled"]:
            return
        states = [v["state"] for v in views.values()]
        live = {s["agent_id"] for s in states}
        self.sel_sterile &= live
        self.sel_seen &= live
        scores = {s["agent_id"]: self.sel_score(s) for s in states}
        if any(v["predators"] for v in views.values()):
            self.sel_last_threat = self.clock

        # Newborn screening against the colony it was born into (first tick we see it, and only a real newborn).
        median = statistics.median(scores.values())
        for s in states:
            aid = s["agent_id"]
            if aid not in self.sel_seen:
                self.sel_seen.add(aid)
                if c["sel_sterile_fraction"] and s["age"] < 2. and len(states) > 1 and scores[aid] < c["sel_sterile_fraction"] * median:
                    self.sel_sterile.add(aid)
                    self.metrics["sel_sterile_marked"] += 1
        if not self.spawn_ids:
            return

        fertile_young = [s for s in states if s["agent_id"] not in self.sel_sterile and s["age"] < 60.]
        sterile_active = len(fertile_young) >= c["sel_min_fertile_young"]
        allowed = {aid for aid in self.spawn_ids if not (sterile_active and aid in self.sel_sterile)}
        self.metrics["sel_sterile_veto_ticks"] += len(self.spawn_ids) - len(allowed)

        breeders = [s for s in states if s["age"] <= c["retirement"] and not (sterile_active and s["agent_id"] in self.sel_sterile)]
        cutoff = c["sel_strict_fraction"] * max((scores[s["agent_id"]] for s in breeders), default=0.)
        permitted_young = sum(scores[s["agent_id"]] >= cutoff and s["age"] < 60. for s in breeders)
        comfortable = (self.clock <= c["sel_strict_until"]
                       and len(states) >= c["sel_strict_min_agents"]
                       and sum(s["energy"] for s in states) / len(states) >= c["sel_strict_energy"]
                       and self.clock - self.sel_last_threat >= c["sel_strict_calm"]
                       and min(s["age"] for s in states) <= c["sel_strict_patience"]
                       and permitted_young >= c["sel_min_fertile_young"])
        if not comfortable:
            if self.clock <= c["sel_strict_until"]:
                self.sel_off_until = self.clock + c["sel_cooldown"]
            self.metrics["sel_lenient_ticks"] += 1
        elif self.clock < self.sel_off_until:
            self.metrics["sel_lenient_ticks"] += 1
        else:
            self.metrics["sel_strict_ticks"] += 1
            strict = {aid for aid in allowed if scores[aid] >= cutoff}
            self.metrics["sel_strict_veto_ticks"] += len(allowed) - len(strict)
            allowed = strict
        self.spawn_ids = allowed


def make_policy():
    return SelectivePolicy("appetite", **CONFIG)


# Reuse the baseline's FastAPI app and /predict handler; they look up make_policy/_policy in the baseline module.
base.make_policy = make_policy
base._policy = make_policy()
base.STRATEGY_NAME = STRATEGY_NAME
app, predict, StepResponse = base.app, base.predict, base.StepResponse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9052)
