"""
C1-MECH-P02 analysis: reads traces/, writes summary/stats/matched comparisons, a mechanism
census, after-contact and post-death summaries, a repeat/nolog/compat digest comparison, and
AT MOST 3 figures. Adapted from C1-MECH-P01's experiments/C1-MECH-P01/analyze.py (same overall
shape: load traces -> per-episode summary -> per-(fixture,arm) stats -> matched comparisons ->
mechanism census -> plots) but restructured for P02's single delta-parameterized controller (9
arms = 3 speeds x 3 signed offsets, replacing P01's separate A/F arm families) and P02's larger
matrix (8 fixtures x 9 arms x 3 seeds, 30s living horizon + 10s post-death continuation).

NO SIMULATION IS RUN HERE. This module only reads already-produced trace/digest files.

Usage:
  python -m experiments.C1-MECH-P02.analyze                     # reads/writes experiments/C1-MECH-P02/
  python experiments/C1-MECH-P02/analyze.py --dir <path>         # point at a different directory,
                                                                  # e.g. a synthetic test fixture
"""
import argparse
import csv
import gzip
import json
import math
import statistics as st
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

THIS_DIR = Path(__file__).resolve().parent
SIM_ROOT = THIS_DIR.parents[1]  # survival-simulator/
sys.path.insert(0, str(SIM_ROOT))
from training.predator_mechanics_p02 import (  # noqa: E402
    ARMS, ALL_FIXTURES, SEEDS, DT, AXY0, summarize, _jsonable,
    REP_FIXTURE_FOR_REPEATS, REP_SEED, REP_FIXTURES,
)

NA = "not observed before death/horizon"
ARM_NAMES = list(ARMS.keys())
ZERO_ARM_OF_SPEED = {sp: f"S{sp}_D0" for sp in {int(a.split("_")[0][1:]) for a in ARM_NAMES}}
PLUS_ARM_OF_SPEED = {sp: f"S{sp}_D+15" for sp in ZERO_ARM_OF_SPEED}
MINUS_ARM_OF_SPEED = {sp: f"S{sp}_D-15" for sp in ZERO_ARM_OF_SPEED}
SPEEDS = sorted(ZERO_ARM_OF_SPEED.keys())


def stats(vals):
    vals = [float(v) for v in vals if v not in (None, "")]
    if not vals:
        return dict(n=0, mean="", std="", median="", min="", max="")
    return dict(n=len(vals), mean=st.mean(vals), std=st.stdev(vals) if len(vals) > 1 else "",
                median=st.median(vals), min=min(vals), max=max(vals))


# ---------------------------------------------------------------- loading
def load_traces(traces_dir, fixtures=ALL_FIXTURES, arms=ARM_NAMES, seeds=SEEDS):
    """Loads whatever trace files exist under traces_dir; missing combos are simply absent from
    the returned dict (this analysis never invents or reruns missing episodes)."""
    EP = {}
    for f in fixtures:
        for arm in arms:
            for s in seeds:
                p = traces_dir / f"d{f[0]}_ae{f[1]}_pe{f[2]}_{arm}_s{s}.json.gz"
                if p.exists():
                    with gzip.open(p, "rt") as fh:
                        EP[(f, arm, s)] = json.load(fh)
    return EP


def write_summary_csv(EP, out_path):
    """Regenerates summary.csv from traces using the harness's own summarize(), so this file is
    always consistent with whatever traces actually exist (independent of whether cmd_main's own
    incremental summary.csv write completed)."""
    rows = [summarize(ep) for ep in EP.values()]
    if not rows:
        with open(out_path, "w") as fh:
            fh.write("")
        return []
    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            flat = {k: (json.dumps(v, default=_jsonable) if isinstance(v, (dict, list)) else v) for k, v in r.items()}
            w.writerow(flat)
    return rows


# ---------------------------------------------------------------- per-fixture/arm stats
def write_stats_by_fixture_arm(EP, rows, out_path):
    by_key = {(r["d"], r["agent_energy0"], r["pred_energy0"], r["arm"]): [] for r in rows}
    for r in rows:
        by_key[(r["d"], r["agent_energy0"], r["pred_energy0"], r["arm"])].append(r)
    stat_rows = []
    for (d, ae, pe, arm), rs in sorted(by_key.items()):
        base = {"d": d, "agent_energy0": ae, "pred_energy0": pe, "arm": arm,
                "n": len(rs), "captures": sum(r["outcome"] == "captured" for r in rs),
                "energy_deaths": sum(r["outcome"] == "energy_death" for r in rs),
                "censored_alive_at_30s": sum(r["outcome"] == "survived_horizon" for r in rs),
                "distinct_trajectories_norng": len({r["trajectory_digest_norng"] for r in rs})}
        for m, vals in [
            ("capture_time_uncensored", [r["event_time"] for r in rs if r["event_time"]]),
            ("min_separation", [r["min_separation"] for r in rs]),
            ("time_chase", [r["time_chase"] for r in rs]),
            ("time_circle", [r["time_circle"] for r in rs]),
            ("time_edge_avoid", [r["time_edge_avoid"] for r in rs]),
            ("time_wander", [r["time_wander"] for r in rs]),
            ("time_sleep", [r["time_sleep"] for r in rs]),
            ("eligible_circle_ticks", [r["eligible_circle_ticks"] for r in rs]),
            ("nonzero_pivot_ticks", [r["nonzero_pivot_ticks"] for r in rs]),
            ("agent_turn_cost", [r["agent_turn_cost"] for r in rs]),
            ("agent_move_cost", [r["agent_move_cost"] for r in rs]),
            ("final_agent_energy_survivors", [r["final_agent_energy"] for r in rs if r.get("final_agent_energy")]),
            ("final_separation_survivors", [r["final_separation"] for r in rs if r.get("final_separation")]),
            ("pred_displacement_along_escape_from_own_start", [r.get("pred_displacement_along_escape_from_own_start") for r in rs]),
            ("pred_dist_from_ref_final", [r.get("pred_dist_from_ref_final") for r in rs]),
        ]:
            for k, v in stats(vals).items():
                base[f"{m}_{k}"] = v
        stat_rows.append(base)
    if stat_rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stat_rows[0].keys()))
            w.writeheader()
            w.writerows(stat_rows)
    return stat_rows


# ---------------------------------------------------------------- matched comparisons (delta effect + handedness)
def tick_at(ep, step):
    T = ep["ticks"]
    return T[step - 1] if 1 <= step <= len(T) else None


def _common_alive_ticks(a, b):
    common = min(len(a["ticks"]), len(b["ticks"]))
    if a["outcome"] == "captured" or b["outcome"] == "captured":
        common = max(0, common - 1)  # exclude the capture tick itself (no living agent to compare)
    return common


def write_matched_comparisons(EP, out_path, checkpoints=(1.0, 5.0, 10.0, 20.0, 29.0)):
    rows = []
    for f in ALL_FIXTURES:
        for sp in SPEEDS:
            for s in SEEDS:
                z, p, m = ZERO_ARM_OF_SPEED[sp], PLUS_ARM_OF_SPEED[sp], MINUS_ARM_OF_SPEED[sp]
                if (f, z, s) not in EP:
                    continue
                base_ep = EP[(f, z, s)]
                for tag, other_arm in (("plus15_minus_0", p), ("minus15_minus_0", m)):
                    if (f, other_arm, s) not in EP:
                        continue
                    other_ep = EP[(f, other_arm, s)]
                    rows.append(_compare_pair(f, sp, s, tag, "delta0", other_arm.split("_")[1], base_ep, other_ep, checkpoints))
                if (f, p, s) in EP and (f, m, s) in EP:
                    rows.append(_compare_pair(f, sp, s, "plus15_vs_minus15", "D+15", "D-15", EP[(f, p, s)], EP[(f, m, s)], checkpoints))
    if rows:
        with open(out_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return rows


def _compare_pair(f, speed, seed, tag, base_label, other_label, base_ep, other_ep, checkpoints):
    common = _common_alive_ticks(base_ep, other_ep)
    row = {"d": f[0], "agent_energy0": f[1], "pred_energy0": f[2], "speed": speed, "seed": seed,
           "comparison": tag, "base": base_label, "other": other_label,
           "base_outcome": base_ep["outcome"], "other_outcome": other_ep["outcome"],
           "base_event_time": round(base_ep["event_step"] * DT, 10) if base_ep["event_step"] else "",
           "other_event_time": round(other_ep["event_step"] * DT, 10) if other_ep["event_step"] else "",
           "common_alive_ticks": common}
    if base_ep["event_step"] and other_ep["event_step"]:
        row["delta_event_time_other_minus_base"] = round(other_ep["event_step"] * DT - base_ep["event_step"] * DT, 10)
    else:
        row["delta_event_time_other_minus_base"] = ""
    for cp in checkpoints:
        k = round(cp / DT)
        ta, tb = tick_at(base_ep, k), tick_at(other_ep, k)
        if ta and tb and k <= common:
            row[f"dE_agent_t{cp:g}"] = tb["agent_energy_post"] - ta["agent_energy_post"]
            row[f"dSep_t{cp:g}"] = tb["dist_post_step"] - ta["dist_post_step"]
            row[f"dPredDispAlongEscape_t{cp:g}"] = (-(tb["pred_xy"][0] - f[0])) - (-(ta["pred_xy"][0] - f[0]))
        else:
            row[f"dE_agent_t{cp:g}"] = "not both alive"
            row[f"dSep_t{cp:g}"] = "not both alive"
            row[f"dPredDispAlongEscape_t{cp:g}"] = "not both alive"
    return row


# ---------------------------------------------------------------- mechanism census
def mechanism_census(EP):
    predicate_match = Counter()
    pivot_by_speed_sign_zero = Counter()
    rel_dir_vals = []
    pivot_sign_changes = 0
    agent_side_losses, agent_side_reacq = 0, 0
    pred_side_losses, pred_side_reacq = 0, 0
    mode_time = Counter()
    for (f, arm, s), ep in EP.items():
        speed = ep.get("speed")
        prev_pivot = None
        prev_ctrl_saw_pred = None
        prev_pred_saw_agent = None
        for t in ep["ticks"]:
            mode_time[t["mode"]] += DT
            pdec = t.get("predator_decision")
            if pdec:
                predicate_match[pdec["predicate_mode"] == pdec["observed_mode"]] += 1
                if pdec["observed_mode"] == "circle":
                    pivot_by_speed_sign_zero[(speed, pdec["pivot_sign"], pdec["rel_dir"] == 0.0)] += 1
                    if "rel_dir" in pdec:
                        rel_dir_vals.append(pdec["rel_dir"])
                    ps = pdec["pivot_sign"]
                    if prev_pivot is not None and ps != 0.0 and prev_pivot != 0.0 and ps != prev_pivot:
                        pivot_sign_changes += 1
                    if ps != 0.0:
                        prev_pivot = ps
                pred_saw_agent = pdec["n_agents_perceived"] > 0
                if prev_pred_saw_agent is True and pred_saw_agent is False:
                    pred_side_losses += 1
                if prev_pred_saw_agent is False and pred_saw_agent is True:
                    pred_side_reacq += 1
                prev_pred_saw_agent = pred_saw_agent
            ctrl_saw_pred = len(t["controller_obs_predators"]) > 0
            if prev_ctrl_saw_pred is True and ctrl_saw_pred is False:
                agent_side_losses += 1
            if prev_ctrl_saw_pred is False and ctrl_saw_pred is True:
                agent_side_reacq += 1
            prev_ctrl_saw_pred = ctrl_saw_pred
    out = {
        "predicate_vs_observed": {"match": predicate_match[True], "mismatch": predicate_match[False]},
        "circle_ticks_by_speed_pivotsign_reldirzero": {str(k): v for k, v in pivot_by_speed_sign_zero.items()},
        "rel_dir_distribution_deg": stats([math.degrees(v) for v in rel_dir_vals]),
        "pivot_sign_changes_total": pivot_sign_changes,
        "agent_side_perception_losses": agent_side_losses, "agent_side_reacquisitions": agent_side_reacq,
        "predator_side_perception_losses": pred_side_losses, "predator_side_reacquisitions": pred_side_reacq,
        "time_in_mode_sec": dict(mode_time),
        "episodes_censused": len(EP),
    }
    return out


# ---------------------------------------------------------------- after-contact / post-death summaries
def after_contact_and_post_death(EP, rows):
    ac = []
    for r in rows:
        if r.get("first_pred_perception_loss") not in (None, "", NA):
            ac.append({"d": r["d"], "arm": r["arm"], "seed": r["seed"],
                       "first_pred_perception_loss": r["first_pred_perception_loss"],
                       "outcome": r["outcome"]})
    pd_rows = [r for r in rows if r.get("post_death_steps_run", 0)]
    pd_summary = {
        "episodes_with_post_death_phase": len(pd_rows),
        "post_death_min_dist_to_ref": stats([r.get("post_death_min_dist_to_ref") for r in pd_rows]),
        "post_death_wake_events": stats([r.get("post_death_wake_events") for r in pd_rows]),
        "post_death_sleep_events": stats([r.get("post_death_sleep_events") for r in pd_rows]),
    }
    return {"after_contact_perception_loss_events": len(ac), "after_contact_sample": ac[:20],
            "post_death_summary": pd_summary}


# ---------------------------------------------------------------- repeat / nolog / compat digest comparison
def compare_repeat_nolog_compat(exp_dir):
    """Compares digests_main.json (representative fixture entries) vs digests_repeat.json
    (fresh-process repeats) vs digests_nolog.json (instrumentation-disabled) and reads compat.json
    (P01-arena zero-offset compatibility). Reads only; produces no simulation. Any missing file is
    reported as not-yet-run rather than treated as a failure."""
    out = {}
    main_p, rep_p, nolog_p, compat_p = (exp_dir / n for n in
                                         ("digests_main.json", "digests_repeat.json", "digests_nolog.json", "compat.json"))
    rep = json.load(open(rep_p)) if rep_p.exists() else None
    nolog = json.load(open(nolog_p)) if nolog_p.exists() else None
    main = json.load(open(main_p)) if main_p.exists() else None
    compat = json.load(open(compat_p)) if compat_p.exists() else None
    d, ae, pe = REP_FIXTURE_FOR_REPEATS

    if rep is not None:
        if main is not None:
            mismatches = []
            for arm, r in rep.items():
                key = f"{d}|{ae}|{pe}|{arm}|{REP_SEED}"
                if key in main and main[key]["trajectory_digest"] != r["trajectory_digest"]:
                    mismatches.append(arm)
            out["repeat_vs_main"] = {"arms_checked": len(rep), "mismatches": mismatches, "status": "checked"}
        else:
            out["repeat_vs_main"] = {"status": "digests_main.json not yet produced (main not run)"}
    else:
        out["repeat_vs_main"] = {"status": "digests_repeat.json not yet produced (repeat not run)"}

    if nolog is not None and rep is not None:
        mismatches = [arm for arm in rep if arm in nolog and rep[arm]["trajectory_digest_norng"] != nolog[arm]["trajectory_digest_norng"]]
        out["nolog_vs_repeat"] = {"arms_checked": len(set(rep) & set(nolog)), "mismatches": mismatches, "status": "checked"}
    else:
        out["nolog_vs_repeat"] = {"status": "digests_repeat.json and/or digests_nolog.json not yet produced"}

    if compat is not None:
        # compat.json schema (post BLOCKING-1 fix): per-pair {"status": "pass"|"fail"|"blocked",
        # "live_comparison": {"max_abs_err": {...}, "n_mismatches": int}, ...}. Reads that nested
        # shape, not stale top-level keys from the pre-fix vacuous version.
        errs = [v for c in compat for k, v in c.get("live_comparison", {}).get("max_abs_err", {}).items() if not k.endswith("_raw")]
        n_nonpass = sum(1 for c in compat if c.get("status") != "pass")
        n_mismatches = sum(c.get("live_comparison", {}).get("n_mismatches", 0) for c in compat)
        out["compat"] = {"pairs_evaluated": len(compat), "pairs_planned": 6,
                          "max_abs_err_over_all_fields": max(errs) if errs else None,
                          "n_pass": sum(1 for c in compat if c.get("status") == "pass"),
                          "n_nonpass": n_nonpass, "n_live_mismatches_total": n_mismatches,
                          "statuses": [c.get("status") for c in compat], "status": "checked"}
    else:
        out["compat"] = {"status": "compat.json not yet produced (compat not run)"}
    return out


# ---------------------------------------------------------------- plots (<=3 total)
def plot_representative_timeline(EP, out_path, fixture=None, seed=REP_SEED):
    """Figure (a): separation + pivot/perception timeline for the representative fixture, one
    panel per arm-family (D0/D+15/D-15) at a single speed, to keep this to ONE figure."""
    fixture = fixture or REP_FIXTURE_FOR_REPEATS
    speed = SPEEDS[0]
    arms = [ZERO_ARM_OF_SPEED[speed], PLUS_ARM_OF_SPEED[speed], MINUS_ARM_OF_SPEED[speed]]
    present = [a for a in arms if (fixture, a, seed) in EP]
    if not present:
        return False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = {arms[0]: "#444444", arms[1]: "#2ca02c", arms[2]: "#d62728"}
    for arm in present:
        ep = EP[(fixture, arm, seed)]
        ts = [0.0] + [t["t"] for t in ep["ticks"]]
        ds = [fixture[0]] + [t["dist_post_step"] for t in ep["ticks"]]
        ax1.plot(ts, ds, color=colors[arm], label=arm, lw=1.4)
        for t in ep["ticks"]:
            if t["pred_transition"] == "fell_asleep":
                ax1.plot(t["t"], t["dist_post_step"], "k*", ms=10)
        if ep["outcome"] == "captured":
            ax1.plot(ts[-1], ds[-1], "x", color=colors[arm], ms=9)
        pv = [t["predator_decision"]["pivot_sign"] if (t["predator_decision"] and t["predator_decision"]["observed_mode"] == "circle") else 0.0 for t in ep["ticks"]]
        ax2.plot([t["t"] for t in ep["ticks"]], pv, color=colors[arm], label=arm, lw=1.0, drawstyle="steps-post")
    ax1.axhline(15, color="k", lw=.8, ls=":")
    ax1.axhline(90, color="purple", lw=.8, ls=":")
    ax1.set_ylabel("separation")
    ax1.set_title(f"Representative fixture d={fixture[0]} aE={fixture[1]} pE={fixture[2]} speed={speed} seed={seed} "
                  f"(*=predator sleep, x=capture, dotted: kill 15 / chase 90)")
    ax1.legend(fontsize=8)
    ax2.axhline(0, color="k", lw=.6)
    ax2.set_ylabel("pivot sign\n(0 outside circle mode)")
    ax2.set_xlabel("time (s)")
    ax1.grid(alpha=.3)
    ax2.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def plot_trajectories(EP, out_path, fixture=None, seed=REP_SEED):
    """Figure (b): equal-aspect trajectories including separate post-death paths, representative
    fixture, one speed, all three deltas."""
    fixture = fixture or REP_FIXTURE_FOR_REPEATS
    speed = SPEEDS[0]
    arms = [ZERO_ARM_OF_SPEED[speed], PLUS_ARM_OF_SPEED[speed], MINUS_ARM_OF_SPEED[speed]]
    present = [a for a in arms if (fixture, a, seed) in EP]
    if not present:
        return False
    colors = {arms[0]: "#444444", arms[1]: "#2ca02c", arms[2]: "#d62728"}
    fig, ax = plt.subplots(figsize=(7, 7))
    for arm in present:
        ep = EP[(fixture, arm, seed)]
        ax.plot([t["agent_xy"][0] for t in ep["ticks"]], [t["agent_xy"][1] for t in ep["ticks"]],
                 "-", color=colors[arm], label=arm + " agent", lw=1.4)
        ax.plot([fixture[0]] + [t["pred_xy"][0] for t in ep["ticks"]], [0] + [t["pred_xy"][1] for t in ep["ticks"]],
                 ":", color=colors[arm], label=arm + " predator", lw=1.0)
        if ep.get("post_death_ticks"):
            xs = [t["pred_xy"][0] for t in ep["post_death_ticks"]]
            ys = [t["pred_xy"][1] for t in ep["post_death_ticks"]]
            ax.plot(xs, ys, "--", color=colors[arm], lw=1.0, alpha=0.6, label=arm + " predator post-death")
    ax.plot(0, 0, "o", color="black", ms=6, label="agent start / reference")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel("x (agent start = 0; escape line = -x)")
    ax.set_ylabel("y")
    ax.set_title(f"Trajectories: d={fixture[0]} aE={fixture[1]} pE={fixture[2]} speed={speed} seed={seed}")
    ax.legend(fontsize=7)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


def plot_paired_difference(cmp_rows, out_path):
    """Figure (c): paired-difference panel -- dSep at t=5s for plus15_minus_0 and
    minus15_minus_0, one point per (fixture,speed,seed), to show the offset effect directly
    instead of relying on overlapping line styles."""
    if not cmp_rows:
        return False
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, tag in enumerate(("plus15_minus_0", "minus15_minus_0")):
        ys = [r["dSep_t5"] for r in cmp_rows if r["comparison"] == tag and r["dSep_t5"] not in ("not both alive", "")]
        ys = [float(y) for y in ys]
        xs = [i + 1 + 0.08 * j for j in range(len(ys))]
        ax.scatter(xs, ys, label=f"{tag} (n={len(ys)})")
        if ys:
            ax.hlines(st.mean(ys), i + 0.8, i + 1.3, colors="k", lw=2)
    ax.axhline(0, color="gray", lw=.8, ls="--")
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["+15 - 0", "-15 - 0"])
    ax.set_ylabel("d(separation) at t=5s (other - base)")
    ax.set_title("Paired offset effect on separation at t=5s, per (fixture, speed, seed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=100)
    plt.close(fig)
    return True


# ---------------------------------------------------------------- main
def main(base_dir):
    base_dir = Path(base_dir)
    traces_dir = base_dir / "traces"
    EP = load_traces(traces_dir)
    print(f"loaded {len(EP)} traces from {traces_dir}")
    rows = write_summary_csv(EP, base_dir / "summary.csv")
    write_stats_by_fixture_arm(EP, rows, base_dir / "stats_by_fixture_arm.csv")
    cmp_rows = write_matched_comparisons(EP, base_dir / "matched_comparisons.csv")
    census = mechanism_census(EP)
    ac_pd = after_contact_and_post_death(EP, rows)
    json.dump({**census, **ac_pd}, open(base_dir / "analysis_census.json", "w"), indent=1, default=_jsonable)
    rn_compat = compare_repeat_nolog_compat(base_dir)
    json.dump(rn_compat, open(base_dir / "repeat_nolog_compat_report.json", "w"), indent=1, default=_jsonable)
    n_figs = 0
    if plot_representative_timeline(EP, base_dir / "plot_timeline.png"):
        n_figs += 1
    if plot_trajectories(EP, base_dir / "plot_trajectories.png"):
        n_figs += 1
    if plot_paired_difference(cmp_rows, base_dir / "plot_paired_difference.png"):
        n_figs += 1
    print(f"episodes={len(EP)} summary_rows={len(rows)} matched_comparisons={len(cmp_rows)} figures={n_figs} (cap 3)")
    print(json.dumps(census, indent=1, default=_jsonable))
    print(json.dumps(rn_compat, indent=1, default=_jsonable))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(THIS_DIR), help="directory containing traces/ and where outputs are written")
    a = ap.parse_args()
    main(a.dir)
