"""Experiment 3: failure-case library. Signatures are measured; root causes are HYPOTHESES.

No teardown, post-mortem or electrochemical diagnostics exist for these cells, so every
root-cause entry is phrased and labelled as a hypothesis with a literature pointer.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cleaning import QD_MAX_VALID, clean_summary

REFS = {
    "attia2022": "Attia et al., 'Review - Knees in lithium-ion battery aging trajectories', J. Electrochem. Soc. 169, 060517 (2022), doi:10.1149/1945-7111/ac6d13",
    "waldmann2018": "Waldmann, Hogg, Wohlfahrt-Mehrens, 'Li plating as unwanted side reaction in commercial Li-ion cells - a review', J. Power Sources 384, 107-124 (2018), doi:10.1016/j.jpowsour.2018.02.063",
    "birkl2017": "Birkl et al., 'Degradation diagnostics for lithium ion cells', J. Power Sources 341, 373-386 (2017), doi:10.1016/j.jpowsour.2016.12.011",
    "vetter2005": "Vetter et al., 'Ageing mechanisms in lithium-ion batteries', J. Power Sources 147, 269-281 (2005), doi:10.1016/j.jpowsour.2005.01.006",
    "epding2019": "Epding et al., 'Investigation of significant capacity recovery effects due to long rest periods during high current cyclic aging tests...', J. Energy Storage 22, 249-256 (2019), doi:10.1016/j.est.2019.02.015",
    "severson_notes": "Batch notes on the data.matr.io project page (Severson et al. 2019): computer restarts and test pauses, inconsistent chamber temperature control (2017-05-12), thermocouples that fell off or were swapped, one 2017-06-30 cell that dies quickly and is possibly defective",
}

HYPOTHESES = {
    "early_knee": ("Hypothesis: lithium plating during high-rate charging consumes cyclable lithium and "
                   "triggers an early knee; untested here (no post-mortem / no low-rate dV/dQ diagnostics).",
                   ["waldmann2018", "attia2022"]),
    "life_overpredicted": ("Hypothesis: the dominant degradation mode is not yet expressed in the "
                           "cycle-10-to-100 discharge curve change, so the early-life feature misses it.",
                           ["attia2022"]),
    "life_underpredicted": ("Hypothesis: the early-life signal exaggerates degradation for this cell "
                            "(e.g. early capacity drift or measurement noise in cycles 10-100); the cell "
                            "then ages more slowly than the cohort relation implies.", ["attia2022"]),
    "ir_jump": ("Hypothesis: an abrupt internal-resistance step that persists points to a cell-internal "
                "change (interphase growth / contact loss); one that reverts points to a fixture or "
                "measurement artefact. Not distinguishable without EIS on these cells.",
                ["birkl2017", "vetter2005"]),
    "temperature_step": ("Hypothesis: a measurement or environment change rather than a cell fault - "
                         "thermocouple contact loss (the provider notes thermocouples that fell off or were "
                         "swapped) or a shift in chamber baseline temperature (noted as inconsistent for the "
                         "2017-05-12 batch); not evidence of a thermal event.", ["severson_notes"]),
    "noisy_capacity": ("Hypothesis: test interruptions / rests cause temporary capacity recovery spikes, "
                       "or an electronic connection problem adds noise; a data-quality issue rather than "
                       "a cell fault.", ["epding2019", "severson_notes"]),
    "invalid_values": ("Hypothesis: recording glitch (a capacity above 1.2 x nominal is physically "
                       "implausible for this cell); values are masked by the cleaning rule.",
                       ["severson_notes"]),
}


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = 1.4826 * (x - med).abs().median()
    return (x - med) / (mad if mad > 0 else 1e-12)


def signatures(cells: pd.DataFrame, cycles: pd.DataFrame, knees: pd.DataFrame,
               preds: pd.DataFrame) -> pd.DataFrame:
    meta = cells.set_index("cell_id")
    rows = []
    for cid, raw in cycles.groupby("cell_id", sort=False):
        eol = float(meta.loc[cid, "cycle_life"])
        g = clean_summary(raw).sort_values("cycle")
        g = g[g["cycle"] <= eol]
        ir = g["ir"].rolling(11, center=True, min_periods=5).median()
        step = (ir.shift(-5) - ir.shift(5))
        j = int(step.abs().idxmax()) if step.notna().any() else None
        ir_jump = float(step.abs().max() * 1000) if j is not None else np.nan
        # does the IR step persist? compare 50 cycles after vs 50 before the jump
        persist = np.nan
        if j is not None:
            pos = g.index.get_loc(j)
            before = g["ir"].iloc[max(0, pos - 55):max(0, pos - 5)].median()
            after = g["ir"].iloc[pos + 5:pos + 55].median()
            persist = float((after - before) * 1000)
        ta = g["tavg"].rolling(21, center=True, min_periods=10).median()
        tstep = float((ta.shift(-10) - ta.shift(10)).abs().max())
        qm = g["qd"].rolling(11, center=True, min_periods=5).median()
        spikes = int(((g["qd"] - qm).abs() > 0.01).sum())
        # noise = scatter around a local (centred 11-cycle) median, so fast but smooth fade is not noise
        noise = float((g["qd"] - qm).abs().mean() * 1000)
        n_invalid = int(((raw["qd"] > QD_MAX_VALID) & (raw["cycle"] <= eol)).sum())
        rows.append({
            "cell_id": cid, "batch": meta.loc[cid, "batch"], "policy": meta.loc[cid, "policy"],
            "split": meta.loc[cid, "split"], "cycle_life": eol,
            "knee_onset": float(knees.loc[cid, "knee_onset"]),
            "knee_onset_frac": float(knees.loc[cid, "knee_onset"] / eol),
            "pred_variance_model": float(preds.loc[cid, "variance"]),
            "obs_over_pred": float(eol / preds.loc[cid, "variance"]),
            "ir_cycle2_mohm": float(g["ir"].iloc[:5].median() * 1000),
            "ir_growth_mohm": float((g["ir"].iloc[-20:].median() - g["ir"].iloc[1:21].median()) * 1000),
            "ir_max_step_mohm": ir_jump,
            "ir_step_cycle": float(g.loc[j, "cycle"]) if j is not None else np.nan,
            "ir_step_persist_mohm": persist,
            "tavg_max_step_c": tstep, "qd_spikes": spikes, "qd_noise_mah": noise,
            "qd_invalid_values": n_invalid, "carried_over": bool(meta.loc[cid, "carried_over"]),
        })
    return pd.DataFrame(rows).set_index("cell_id")


def flag(sig: pd.DataFrame, z_thr: float = 5.0) -> pd.DataFrame:
    s = sig.copy()
    s["early_knee"] = s["knee_onset_frac"] < 0.30
    s["life_overpredicted"] = s["obs_over_pred"] < 0.70
    s["life_underpredicted"] = s["obs_over_pred"] > 1.40
    s["ir_jump"] = robust_z(s["ir_max_step_mohm"]) > z_thr
    s["temperature_step"] = robust_z(s["tavg_max_step_c"]) > z_thr
    s["noisy_capacity"] = (robust_z(s["qd_noise_mah"]) > z_thr) | (s["qd_spikes"] >= 5)
    s["invalid_values"] = s["qd_invalid_values"] > 0
    return s


def _evidence(r: pd.Series, sig: str, med: pd.Series) -> str:
    if sig == "early_knee":
        return f"knee onset at cycle {r.knee_onset:.0f} = {100*r.knee_onset_frac:.0f}% of life (cohort median {100*med.knee_onset_frac:.0f}%)"
    if sig in ("life_overpredicted", "life_underpredicted"):
        return f"observed {r.cycle_life:.0f} vs variance-model prediction {r.pred_variance_model:.0f} cycles (ratio {r.obs_over_pred:.2f})"
    if sig == "ir_jump":
        return (f"IR step {r.ir_max_step_mohm:.2f} mOhm around cycle {r.ir_step_cycle:.0f} (cohort median "
                f"{med.ir_max_step_mohm:.2f}); 50-cycle after-before change {r.ir_step_persist_mohm:+.2f} mOhm")
    if sig == "temperature_step":
        return f"step of {r.tavg_max_step_c:.1f} C in the 21-cycle median of average temperature (cohort median {med.tavg_max_step_c:.1f} C)"
    if sig == "noisy_capacity":
        return (f"{r.qd_spikes} capacity spikes > 10 mAh; mean scatter around the local median "
                f"{r.qd_noise_mah:.3f} mAh (cohort median {med.qd_noise_mah:.3f})")
    if sig == "invalid_values":
        return f"{r.qd_invalid_values} discharge-capacity value(s) above {QD_MAX_VALID:.2f} Ah"
    return ""


def run(cells: pd.DataFrame, cycles: pd.DataFrame, knees: pd.DataFrame, preds: pd.DataFrame,
        out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(exist_ok=True)
    for old in (out / "plots").glob("*.png"):   # no stale plots from earlier rule versions
        old.unlink()
    sig = flag(signatures(cells, cycles, knees, preds))
    sig.to_csv(out / "all_cell_signatures.csv")
    med = sig.median(numeric_only=True)
    names = list(HYPOTHESES)
    rows = []
    for cid, r in sig.iterrows():
        hits = [n for n in names if bool(r[n])]
        if not hits:
            continue
        _plot_cell(cid, cycles[cycles["cell_id"] == cid], r, out / "plots" / f"{cid}.png")
        rows.append({
            "cell_id": cid, "batch": r.batch, "policy": r.policy, "split": r.split,
            "cycle_life": int(r.cycle_life), "signatures": ";".join(hits),
            "evidence": " | ".join(_evidence(r, h, med) for h in hits)
                        + (" | context: test paused between the 2017-05-12 and 2017-06-30 batches and resumed"
                           if r.carried_over else ""),
            "root_cause_hypothesis": " | ".join(HYPOTHESES[h][0] for h in hits),
            "references": "; ".join(sorted({REFS[k] for h in hits for k in HYPOTHESES[h][1]})),
            "status": "hypothesis - not verified",
            "plot": f"plots/{cid}.png",
        })
    lib = pd.DataFrame(rows)
    lib.to_csv(out / "failure_cases.csv", index=False)
    _write_md(lib, sig, out / "failure_cases.md")
    counts = {n: int(sig[n].sum()) for n in names}
    summary = {"n_cells_screened": int(len(sig)), "n_cases": int(len(lib)), "signature_counts": counts,
               "rules": {"early_knee": "knee onset < 30% of life", "life_overpredicted": "observed/predicted < 0.70",
                         "life_underpredicted": "observed/predicted > 1.40",
                         "ir_jump / temperature_step / noisy_capacity": "cohort robust z > 5 (noisy also: >= 5 spikes)",
                         "invalid_values": "any discharge capacity > 1.2 x nominal"}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def _plot_cell(cid: str, raw: pd.DataFrame, r: pd.Series, path: Path) -> None:
    g = raw.sort_values("cycle")
    c = clean_summary(g)
    fig, ax = plt.subplots(1, 3, figsize=(11, 2.8))
    ax[0].plot(c["cycle"], c["qd"], ".", ms=1.5); ax[0].set_ylabel("Q discharge (Ah)")
    ax[0].axvline(r.knee_onset, ls=":", c="k", lw=1); ax[0].axvline(r.cycle_life, c="k", lw=1)
    ax[1].plot(c["cycle"], c["ir"] * 1000, ".", ms=1.5, c="#d9822b"); ax[1].set_ylabel("IR (mOhm)")
    ax[2].plot(c["cycle"], c["tavg"], ".", ms=1.5, c="#2a9d55"); ax[2].set_ylabel("T avg (C)")
    for a in ax:
        a.set_xlabel("cycle")
    fig.suptitle(f"{cid}  {r.policy}  life {r.cycle_life:.0f}", fontsize=9)
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.close(fig)


def _write_md(lib: pd.DataFrame, sig: pd.DataFrame, path: Path) -> None:
    lines = ["# Failure-case library (Severson et al. 2019 cells)", "",
             "Signatures are measured from the public data. Every root cause below is a **hypothesis**",
             "for follow-up, not a diagnosis: there is no teardown or electrochemical diagnostic data for",
             "these cells. Generated by `run_all.py`; do not edit by hand.", "",
             f"Cells screened: {len(sig)}. Cases: {len(lib)}.", "",
             "| cell | policy | life | signatures | evidence | hypothesis | plot |",
             "|---|---|---|---|---|---|---|"]
    def esc(x) -> str:
        return str(x).replace("|", "\\|")

    for _, r in lib.iterrows():
        lines.append(f"| {r['cell_id']} | {esc(r['policy'])} | {r['cycle_life']} | "
                     f"{r['signatures'].replace(';', ', ')} | {esc(r['evidence'])} | "
                     f"{esc(r['root_cause_hypothesis'])} | [png]({r['plot']}) |")
    lines += ["", "## References", ""]
    for k, v in REFS.items():
        lines.append(f"- [{k}] {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
