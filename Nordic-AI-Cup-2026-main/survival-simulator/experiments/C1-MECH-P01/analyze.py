"""C1-MECH-P01 analysis: reads traces/, writes stats, paired comparisons, plots. No simulation."""
import csv
import gzip
import json
import math
import statistics as st
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

D = Path(__file__).resolve().parent
ARMS = ["S0", "A10", "F10", "A15", "F15", "A20", "F20"]
FIX = [(d, ae, pe) for d in (80, 120) for ae in (150, 300) for pe in (101, 200)]
SEEDS = [0, 1, 2]
NA = "not observed before death/horizon"


def load(d, ae, pe, arm, s):
    return json.load(gzip.open(D / "traces" / f"d{d}_ae{ae}_pe{pe}_{arm}_s{s}.json.gz", "rt"))


EP = {(f, arm, s): load(*f, arm, s) for f in FIX for arm in ARMS for s in SEEDS}
SUM = {(int(r["d"]), int(r["agent_energy0"]), int(r["pred_energy0"]), r["arm"], int(r["seed"])): r
       for r in csv.DictReader(open(D / "summary.csv"))}


def stats(vals):
    vals = [float(v) for v in vals]
    if not vals:
        return dict(n=0, mean="", std="", median="", min="", max="")
    return dict(n=len(vals), mean=st.mean(vals), std=st.stdev(vals) if len(vals) > 1 else "",
                median=st.median(vals), min=min(vals), max=max(vals))


# ---- predicate verification / pivot sign census
pred_rows = Counter()
pivot = Counter()
reasons = Counter()
for (f, arm, s), ep in EP.items():
    for t in ep["ticks"]:
        pdec = t["predator_decision"]
        if not pdec or pdec["observed_mode"] == "no_target":
            continue
        pred_rows[(pdec["predicate_mode"] == pdec["observed_mode"])] += 1
        if pdec["observed_mode"] == "circle":
            pivot[(arm[0], pdec["pivot_sign"], pdec["rel_dir"] == 0.0)] += 1
        reasons[(pdec["observed_mode"], pdec["chase_reason"] or "-")] += 1

# ---- per fixture/arm statistics
stat_rows = []
for f in FIX:
    for arm in ARMS:
        rs = [SUM[(*f, arm, s)] for s in SEEDS]
        base = {"d": f[0], "agent_energy0": f[1], "pred_energy0": f[2], "arm": arm,
                "n": len(rs), "captures": sum(r["outcome"] == "captured" for r in rs),
                "energy_deaths": sum(r["outcome"] == "energy_death" for r in rs),
                "censored_alive_at_10s": sum(r["outcome"] == "survived_horizon" for r in rs),
                "pred_sleep_before_agent_death": sum(r["survived_until_pred_sleep"] == "True" for r in rs),
                "distinct_trajectories_norng": len({r["trajectory_digest_norng"] for r in rs})}
        for m, vals in [("capture_time_uncensored", [r["event_time"] for r in rs if r["event_time"]]),
                        ("min_separation", [r["min_separation"] for r in rs]),
                        ("time_chase", [r["time_chase"] for r in rs]),
                        ("time_circle", [r["time_circle"] for r in rs]),
                        ("time_no_target", [r["time_no_target"] for r in rs]),
                        ("time_sleep", [r["time_sleep"] for r in rs]),
                        ("agent_turn_cost", [r["agent_turn_cost"] for r in rs]),
                        ("agent_move_cost", [r["agent_move_cost"] for r in rs]),
                        ("final_agent_energy_survivors", [r["final_agent_energy"] for r in rs if r["final_agent_energy"]]),
                        ("final_separation_survivors", [r["final_separation"] for r in rs if r["final_separation"]])]:
            for k, v in stats(vals).items():
                base[f"{m}_{k}"] = v
        stat_rows.append(base)
with open(D / "stats_by_fixture_arm.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()))
    w.writeheader()
    w.writerows(stat_rows)


# ---- matched comparisons F vs A (and each arm vs S0 is in the stats table)
def tick_at(ep, step):
    return ep["ticks"][step - 1] if step <= len(ep["ticks"]) else None


cmp_rows = []
for f in FIX:
    for v in (10, 15, 20):
        for s in SEEDS:
            a, fe = EP[(f, f"A{v}", s)], EP[(f, f"F{v}", s)]
            ra, rf = SUM[(*f, f"A{v}", s)], SUM[(*f, f"F{v}", s)]
            common = min(len(a["ticks"]), len(fe["ticks"]))
            if a["outcome"] == "captured" or fe["outcome"] == "captured":
                common_alive = common - 1  # the capture tick has no living agent
            else:
                common_alive = common
            max_agent_pos = max((math.dist(tick_at(a, k)["agent_xy"], tick_at(fe, k)["agent_xy"]) for k in range(1, common + 1)), default=0)
            max_pred_pos = max((math.dist(tick_at(a, k)["pred_xy"], tick_at(fe, k)["pred_xy"]) for k in range(1, common + 1)), default=0)
            row = {"d": f[0], "agent_energy0": f[1], "pred_energy0": f[2], "speed": v, "seed": s,
                   "A_outcome": ra["outcome"], "F_outcome": rf["outcome"],
                   "A_event_time": ra["event_time"], "F_event_time": rf["event_time"],
                   "delta_event_time_F_minus_A_if_both_captured": (float(rf["event_time"]) - float(ra["event_time"])) if ra["event_time"] and rf["event_time"] else "",
                   "A_min_sep": ra["min_separation"], "F_min_sep": rf["min_separation"],
                   "delta_min_sep_F_minus_A": float(rf["min_separation"]) - float(ra["min_separation"]),
                   "A_time_chase": ra["time_chase"], "F_time_chase": rf["time_chase"],
                   "A_time_circle": ra["time_circle"], "F_time_circle": rf["time_circle"],
                   "A_turn_cost": ra["agent_turn_cost"], "F_turn_cost": rf["agent_turn_cost"],
                   "common_alive_steps": common_alive,
                   "max_agent_pos_diff_common": max_agent_pos, "max_pred_pos_diff_common": max_pred_pos}
            for tt in (1.0, 2.0, 5.0, 9.9):
                k = round(tt * 10)
                if k <= common_alive:
                    ta, tf = tick_at(a, k), tick_at(fe, k)
                    row[f"dE_agent_F_minus_A_t{tt}"] = tf["agent_energy_post"] - ta["agent_energy_post"]
                    row[f"dSep_F_minus_A_t{tt}"] = tf["dist_post_step"] - ta["dist_post_step"]
                else:
                    row[f"dE_agent_F_minus_A_t{tt}"] = "not both alive"
                    row[f"dSep_F_minus_A_t{tt}"] = "not both alive"
            cmp_rows.append(row)
with open(D / "matched_comparisons.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(cmp_rows[0].keys()))
    w.writeheader()
    w.writerows(cmp_rows)

# ---- plots (seed 0 of each fixture; seeds differ only after perception loss)
colors = {"S0": "#444444", "A10": "#1f77b4", "F10": "#1f77b4", "A15": "#2ca02c", "F15": "#2ca02c", "A20": "#d62728", "F20": "#d62728"}
styles = {"S": "-", "A": "--", "F": "-"}


def grid(title, fn, fname, ylabel):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True)
    for ax, f in zip(axes.flat, FIX):
        for arm in ARMS:
            fn(ax, EP[(f, arm, 0)], arm)
        ax.set_title(f"d={f[0]} agentE={f[1]} predE={f[2]} (seed 0)")
        ax.set_xlabel("time (s)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.3)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(D / fname, dpi=90)
    plt.close(fig)


def p_dist(ax, ep, arm):
    ts = [0] + [t["t"] for t in ep["ticks"]]
    ds = [ep["fixture"]["d"]] + [t["dist_post_step"] for t in ep["ticks"]]
    ax.plot(ts, ds, styles[arm[0]], color=colors[arm], label=arm, lw=1.4)
    for t in ep["ticks"]:
        if t["pred_transition"] == "fell_asleep":
            ax.plot(t["t"], t["dist_post_step"], "k*", ms=10)
    if ep["outcome"] == "captured":
        ax.plot(ts[-1], ds[-1], "x", color=colors[arm], ms=9)
    ax.axhline(15, color="k", lw=.8, ls=":")
    ax.axhline(90, color="purple", lw=.8, ls=":")
    ax.set_ylim(0, 320)


def p_energy(ax, ep, arm):
    ts = [0] + [t["t"] for t in ep["ticks"]]
    ax.plot(ts, [ep["fixture"]["agent_energy"]] + [t["agent_energy_post"] for t in ep["ticks"]], styles[arm[0]], color=colors[arm], label=arm + " agent", lw=1.4)
    ax.plot(ts, [ep["fixture"]["pred_energy"]] + [t["pred_energy_post"] for t in ep["ticks"]], ":", color=colors[arm], lw=1)
    for t in ep["ticks"]:
        if t["pred_transition"] == "fell_asleep":
            ax.plot(t["t"], t["pred_energy_post"], "k*", ms=10)
    ax.axhline(100, color="k", lw=.8, ls="-.")
    ax.axhline(40, color="gray", lw=.8, ls="-.")


def p_traj(ax, ep, arm):
    ax.plot([t["agent_xy"][0] for t in ep["ticks"]], [t["agent_xy"][1] for t in ep["ticks"]], styles[arm[0]], color=colors[arm], label=arm + " agent")
    ax.plot([ep["fixture"]["d"]] + [t["pred_xy"][0] for t in ep["ticks"]], [0] + [t["pred_xy"][1] for t in ep["ticks"]], ":", color=colors[arm], lw=1)


grid("Separation vs time (x=capture, *=predator sleep; dotted black=kill 15, purple=chase radius 90). Dashed=away-facing, solid=toward-facing",
     p_dist, "plot_distance.png", "separation")
grid("Energy vs time: agent solid/dashed, predator dotted (dash-dot 100 = agent sprint threshold, 40 = predator sprint threshold, *=sleep)",
     p_energy, "plot_energy.png", "energy")
fig, axes = plt.subplots(2, 4, figsize=(20, 9))
for ax, f in zip(axes.flat, FIX):
    for arm in ARMS:
        p_traj(ax, EP[(f, arm, 0)], arm)
    ax.set_title(f"d={f[0]} agentE={f[1]} predE={f[2]} (seed 0)")
    ax.set_xlabel("x (agent start = 0; escape line = -x)")
    ax.set_ylabel("y")
    ax.grid(alpha=.3)
axes.flat[0].legend(fontsize=7)
fig.suptitle("Trajectories: agent (line), predator (dotted)")
fig.tight_layout()
fig.savefig(D / "plot_trajectories.png", dpi=90)
plt.close(fig)

out = {"predicate_check": {"match": pred_rows[True], "mismatch": pred_rows[False]},
       "circle_ticks_by_armfamily_pivotsign_reldir_exact0": {str(k): v for k, v in pivot.items()},
       "decision_reasons": {str(k): v for k, v in reasons.items()}}
json.dump(out, open(D / "analysis_census.json", "w"), indent=1)
print(json.dumps(out, indent=1))
