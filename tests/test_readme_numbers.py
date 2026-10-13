"""README numbers are pinned: parse README.md, recover every generated value and check it against
the results files directly (own JSON navigation, independent of bew.report's formatting)."""
import json
import re

from bew import report
from bew.config import REPO, RESULTS

TEMPLATE = (REPO / "README.template.md").read_text(encoding="utf-8")
README = (REPO / "README.md").read_text(encoding="utf-8")


def _parse_readme() -> dict[str, str]:
    parts = report.PLACEHOLDER.split(TEMPLATE)
    literals, names = parts[0::2], parts[1::2]
    pattern = re.escape(literals[0])
    for i, lit in enumerate(literals[1:]):
        pattern += f"(?P<p{i}>.+?)" + re.escape(lit)
    m = re.fullmatch(pattern, README, flags=re.S)
    assert m, "README.md does not match README.template.md - regenerate with run_all.py --readme-only"
    out: dict[str, list[str]] = {}
    for i, name in enumerate(names):
        out.setdefault(name, []).append(m.group(f"p{i}"))
    for name, vals in out.items():
        assert len(set(vals)) == 1, f"placeholder {name} rendered inconsistently: {vals}"
    return {k: v[0] for k, v in out.items()}


def _raw(name):
    f, path, kind = report.SOURCES[name]
    p = RESULTS / f
    if not p.exists():
        return None, kind
    v = json.loads(p.read_text())
    for k in path:
        try:
            v = v[k]
        except (IndexError, KeyError):
            return None, kind
    return v, kind


def test_readme_is_rendered_from_results():
    assert README == report.render(TEMPLATE, report.build_headline(RESULTS))


def test_every_number_in_readme_matches_results():
    parsed = _parse_readme()
    assert len(parsed) > 100
    checked = 0
    for name, shown in parsed.items():
        raw, kind = _raw(name)
        if raw is None:
            assert shown == "n/a", name
            continue
        if kind == "s":
            assert shown == str(raw), name
        elif kind == "len":
            assert int(shown) == len(raw), name
        elif kind == "pct0":
            assert abs(float(shown) - 100 * raw) <= 0.5 + 1e-9, name
        elif kind.startswith("e"):
            assert abs(float(shown) - float(raw)) <= abs(float(raw)) * 0.06 + 1e-300, name
        else:
            dec = {"d0": 0, "f1": 1, "f2": 2, "f3": 3}[kind]
            assert abs(float(shown) - float(raw)) <= 0.5 * 10 ** -dec + 1e-9, (name, shown, raw)
        checked += 1
    assert checked > 100


def test_core_results_present():
    assert "{{" not in README
    hl = report.build_headline(RESULTS)
    core = [n for n in report.SOURCES if not n.startswith(("ch_", "mu_"))]
    missing = [n for n in core if hl[n] == "n/a"]
    assert not missing, missing
