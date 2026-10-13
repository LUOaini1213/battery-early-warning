import re

import pandas as pd

from bew import failure_cases as FC
from bew.config import RESULTS


def test_robust_z():
    z = FC.robust_z(pd.Series([1.0, 1.1, 0.9, 1.0, 5.0]))
    assert z.iloc[-1] > 5 and abs(z.iloc[0]) < 1


def _sig(**over):
    base = dict(knee_onset_frac=0.5, obs_over_pred=1.0, ir_max_step_mohm=0.7, tavg_max_step_c=0.5,
                qd_noise_mah=0.1, qd_spikes=0, qd_invalid_values=0)
    rows = [dict(base, ir_max_step_mohm=0.7 + 0.01 * i, tavg_max_step_c=0.5 + 0.01 * i,
                 qd_noise_mah=0.1 + 0.001 * i) for i in range(30)]
    rows.append(dict(base, **over))
    return FC.flag(pd.DataFrame(rows))


def test_flag_rules():
    assert _sig(knee_onset_frac=0.25).iloc[-1]["early_knee"]
    assert not _sig(knee_onset_frac=0.35).iloc[-1]["early_knee"]
    assert _sig(obs_over_pred=0.6).iloc[-1]["life_overpredicted"]
    assert _sig(obs_over_pred=1.5).iloc[-1]["life_underpredicted"]
    assert _sig(ir_max_step_mohm=4.0).iloc[-1]["ir_jump"]
    assert _sig(qd_spikes=6).iloc[-1]["noisy_capacity"]
    assert _sig(qd_invalid_values=1).iloc[-1]["invalid_values"]
    assert not _sig().iloc[:-1][list(FC.HYPOTHESES)].any().any()


def test_every_root_cause_is_worded_as_hypothesis_with_reference():
    for name, (text, refs) in FC.HYPOTHESES.items():
        assert text.startswith("Hypothesis"), name
        assert refs and all(r in FC.REFS for r in refs)


def test_committed_library_is_labelled_hypothesis():
    lib = pd.read_csv(RESULTS / "failure_cases" / "failure_cases.csv")
    assert len(lib) > 0
    assert (lib["status"] == "hypothesis - not verified").all()
    assert lib["root_cause_hypothesis"].str.startswith("Hypothesis").all()
    assert lib["references"].str.len().gt(10).all()


def test_markdown_library_table_is_well_formed():
    md = (RESULTS / "failure_cases" / "failure_cases.md").read_text(encoding="utf-8")
    rows = [l for l in md.splitlines() if l.startswith("| b")]
    assert rows
    for r in rows:
        cells = re.split(r"(?<!\\)\|", r)[1:-1]
        assert len(cells) == 7, r[:80]
        assert "plots/" in cells[-1] and "object at" not in r
