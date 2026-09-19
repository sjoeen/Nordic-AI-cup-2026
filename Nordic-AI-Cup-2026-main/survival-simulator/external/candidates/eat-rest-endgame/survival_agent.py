"""eat-rest-endgame: the preserved eat-rest-v1 controller plus an endgame mode that buys time.

The baseline file (../original-eat-rest-preserved/survival_agent.py) is loaded unmodified and subclassed.
Before the endgame starts this IS the baseline, action for action; with eg_enabled=0 it is the baseline throughout.

Why (endgame_probe, 16 baseline games): the colony holds 4-5 agents until ~30 s before extinction and then collapses
with ~34 fruits still on the map. In the last 100 s "newborn starved" is 28% of deaths (5% early) and adults starve at
0 energy: with few agents the baseline's emergency births fire at 112 energy with a predator as close as 55 units, which
leaves a parent on ~12 energy and a 75-energy newborn that cannot sprint (<20% of max energy) and burns ~7/s fleeing.
Score is 1 per second while anyone lives, so the endgame goal is lineage survival per unit of energy, not headcount.

Detection. The game never tells an agent how many predators are alive; it only gets its own observations. Three triggers:
  eg_trigger="colony" (default) the state of the colony itself, smoothed over eg_window seconds.
                      ON  when mean energy < eg_energy_on or living agents <= eg_agents_on           (struggling)
                      OFF when mean energy > eg_energy_off and living agents >= eg_agents_off,
                          held for eg_stable seconds without a break                                  (stabilised)
                      Baseline games: mean energy is 209-227 until 300 s before extinction, then 174 / 154 / 141 / 106
                      at -200 / -100 / -50 / 0 s, while the headcount only slips from 5.6 to 4.6.
  eg_trigger="clock"  the simulator spawns predators so that their expected number is clock/100 (measured: 4.8 / 7.2 / 9.0
                      alive at t=500 / 750 / 1000). ON when clock/100 >= eg_ratio * living agents, OFF below eg_release *.
  eg_trigger="sensed" predators in the agents' own observations, per agent, smoothed over eg_sensed_window seconds.
                      ON at eg_sensed_on, OFF below eg_sensed_off. NOT calibrated yet: read the "sensed" column of
                      endgame_probe's report and set the two levels from it.
None switches ON before eg_min_clock: the colony starts as 5 agents on 150 energy, which would read as struggling.

While ON, births the baseline orders (normal and emergency) are vetoed unless
  - the parent keeps eg_birth_reserve energy after paying the 100              (eg_birth_reserve, 0 = off)
  - no predator is observed within eg_birth_safe                                (eg_birth_safe, 0 = off)
  - at least one fruit is observed within eg_birth_food, so the newborn can eat (eg_birth_food, 0 = off)
An agent of eg_legacy_age or older is exempt from the vetoes and is also ORDERED to breed once it can afford it: its
energy is about to be lost to old age anyway, and a newborn restarts the 60-120 s lifespan clock (eg_legacy_age, 0 = off).
eg_overrides is a dict of baseline settings that apply only while the endgame is ON (restored when it turns OFF),
e.g. {"escape_radius": 80, "search_speed": 0.3, "population": 4}.
"""
import importlib.util
from pathlib import Path

_BASE = Path(__file__).resolve().parent.parent / "original-eat-rest-preserved" / "survival_agent.py"
_spec = importlib.util.spec_from_file_location("_eat_rest_v1_base_endgame", _BASE)
base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(base)

STRATEGY_NAME = "appetite+endgame"
DEFAULTS = dict(eg_enabled=1, eg_trigger="colony", eg_min_clock=400., eg_window=20.,
                eg_energy_on=175., eg_agents_on=3, eg_energy_off=205., eg_agents_off=5, eg_stable=30.,
                eg_ratio=1.5, eg_release=1., eg_sensed_window=30., eg_sensed_on=.5, eg_sensed_off=.25,
                eg_birth_reserve=50., eg_birth_safe=130., eg_birth_food=0., eg_legacy_age=95., eg_overrides={})
CONFIG = dict(base.CONFIG, **DEFAULTS)


class EndgamePolicy(base.AppetitePolicy):
    def __init__(self, preset="appetite", **overrides):
        config = dict(DEFAULTS, eg_enabled=0)
        config.update(overrides)
        super().__init__(preset, **config)
        self.endgame = False
        self.eg_saved = {}
        self.eg_last_clock = -1.
        self.eg_sensed = 0.
        self.eg_energy = None
        self.eg_stable_since = None

    def eg_switch(self, on):
        self.endgame = on
        if on:
            self.eg_saved = {k: self.config[k] for k in self.config["eg_overrides"]}
            self.config.update(self.config["eg_overrides"])
            self.eg_stable_since = None
            self.metrics["eg_switch_on"] += 1
            if not self.metrics["eg_first_on"]:
                self.metrics["eg_first_on"] = round(self.clock)
        else:
            self.config.update(self.eg_saved)
            self.eg_saved = {}
            self.metrics["eg_switch_off"] += 1

    def decide_all(self, states, sim_time=None):
        c = self.config
        if c["eg_enabled"] and states:
            clock = float(sim_time) if sim_time is not None else self.clock + .1
            if clock < self.eg_last_clock:  # a new game on the same policy object
                self.eg_sensed, self.eg_energy, self.eg_stable_since = 0., None, None
                if self.endgame:
                    self.eg_switch(False)
            step = max(0., clock - self.eg_last_clock) if self.eg_last_clock >= 0. else .1
            self.eg_last_clock = clock
            now = sum(o["type"] == "Predator" for s in states for o in s["observations"]) / len(states)
            self.eg_sensed += min(1., step / c["eg_sensed_window"]) * (now - self.eg_sensed)
            energy = sum(s["energy"] for s in states) / len(states)
            self.eg_energy = energy if self.eg_energy is None else self.eg_energy + min(1., step / c["eg_window"]) * (energy - self.eg_energy)
            if c["eg_trigger"] == "colony":
                on = self.eg_energy < c["eg_energy_on"] or len(states) <= c["eg_agents_on"]
                stable = self.eg_energy > c["eg_energy_off"] and len(states) >= c["eg_agents_off"]
                self.eg_stable_since = (self.eg_stable_since if self.eg_stable_since is not None else clock) if stable else None
                off = stable and clock - self.eg_stable_since >= c["eg_stable"]
            elif c["eg_trigger"] == "sensed":
                on, off = self.eg_sensed >= c["eg_sensed_on"], self.eg_sensed < c["eg_sensed_off"]
            else:
                on, off = clock / 100. >= c["eg_ratio"] * len(states), clock / 100. < c["eg_release"] * len(states)
            if not self.endgame and on and clock >= c["eg_min_clock"]:
                self.eg_switch(True)
            elif self.endgame and off:
                self.eg_switch(False)
        actions = super().decide_all(states, sim_time)
        if not self.endgame:
            return actions

        self.metrics["eg_ticks"] += 1
        byid = {s["agent_id"]: s for s in states}
        for action in actions:
            s = byid[action.agent_id]
            seen = s["observations"]
            cost = (.05 * min(action.move_distance, s["speed"]) + .5 * max(0., action.move_distance - s["speed"])
                    + abs(action.turn_angle) / 6.283)
            if c["eg_legacy_age"] and s["age"] >= c["eg_legacy_age"]:
                enemies = [o["distance"] for o in seen if o["type"] == "Predator"]
                if not action.spawn_agent and s["energy"] > 104. + cost and (not enemies or min(enemies) > 55.):
                    action.spawn_agent = True
                    self.metrics["eg_legacy_births"] += 1
                continue
            if not action.spawn_agent:
                continue
            veto = None
            if c["eg_birth_reserve"] and s["energy"] - cost - 100. < c["eg_birth_reserve"]:
                veto = "reserve"
            elif c["eg_birth_safe"] and any(o["type"] == "Predator" and o["distance"] < c["eg_birth_safe"] for o in seen):
                veto = "unsafe"
            elif c["eg_birth_food"] and not any(o["type"] == "Fruit" and o["distance"] < c["eg_birth_food"] for o in seen):
                veto = "nofood"
            if veto:
                action.spawn_agent = False
                self.metrics["eg_veto_" + veto] += 1
            else:
                self.metrics["eg_births"] += 1
        return actions


def make_policy():
    return EndgamePolicy("appetite", **CONFIG)


# Reuse the baseline's FastAPI app and /predict handler; they look up make_policy/_policy in the baseline module.
base.make_policy = make_policy
base._policy = make_policy()
base.STRATEGY_NAME = STRATEGY_NAME
app, predict, StepResponse = base.app, base.predict, base.StepResponse

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=9053)
