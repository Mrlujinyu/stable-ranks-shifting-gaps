"""Configuration identity and directed-pair regression test (pure standard library).

This software test reads the frozen configuration table and the main task panel,
then verifies that submission identifiers, internal model codes, and manuscript
identifiers are uniquely and bijectively mapped, and that the directed-pair
inversion rule holds. It does not re-run any analysis and is not part of the
paper's Results.
"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def main():
    cfg = read_csv(ROOT / "data" / "table2_configurations.csv")
    primary = [r for r in cfg if r["primary"] == "True"]

    # Uniqueness and completeness within the primary panel.
    assert len(primary) == 14
    submissions = [r["submission"] for r in primary]
    figures = [r["figure_id"] for r in primary]
    internals = [r["model_internal_id"] for r in primary]
    assert len(set(submissions)) == 14, "submission identifiers must be unique"
    assert len(set(figures)) == 14, "manuscript identifiers must be unique"
    assert len(set(internals)) == 14, "internal codes must be unique"

    # Bidirectional round trip: internal <-> manuscript, keyed by submission.
    internal_to_figure = {r["model_internal_id"]: r["figure_id"] for r in primary}
    figure_to_internal = {r["figure_id"]: r["model_internal_id"] for r in primary}
    for k, v in internal_to_figure.items():
        assert figure_to_internal[v] == k, "round-trip mapping broken"

    # The two numbering systems are NOT a fixed offset. Pin three derived facts.
    assert internal_to_figure["A02"] == "A04", "internal A02 must map to manuscript A04"
    assert internal_to_figure["A07"] == "A03", "internal A07 must map to manuscript A03"
    assert internal_to_figure["A08"] == "A10", "internal A08 must map to manuscript A10"

    # Matrix columns are manuscript identifiers in lexicographic submission order.
    lex = sorted(primary, key=lambda r: r["submission"])
    with open(ROOT / "data" / "main_panel.csv", newline="") as f:
        header = next(csv.reader(f))
    score_cols = [c for c in header if c.startswith("A")]
    assert score_cols == [r["figure_id"] for r in lex], "panel columns must be figure IDs in submission order"

    # Pair completeness: 14 choose 2 = 91 unordered pairs, 9 choose 2 = 36 for Pro.
    assert 14 * 13 // 2 == 91
    assert 9 * 8 // 2 == 36

    # The stored 91-pair table contains every unordered pair exactly once.
    pairs = read_csv(ROOT / "data" / "all91_pairs.csv")
    assert len(pairs) == 91
    seen = set()
    for r in pairs:
        key = frozenset((r["agent_a_figure_id"], r["agent_b_figure_id"]))
        assert key not in seen, "duplicate unordered pair in all91_pairs.csv"
        assert r["agent_a_figure_id"] != r["agent_b_figure_id"]
        seen.add(key)
    assert len(seen) == 91

    # Directed inversion rule on the A11-A10 example. The stored orientation is
    # A10--A11 (first-listed minus second-listed); the main text quotes the
    # reversed A11-A10 orientation, so sign and interval endpoints must flip.
    a10a11 = next(r for r in pairs if r["agent_a_figure_id"] == "A10" and r["agent_b_figure_id"] == "A11")
    assert abs(float(a10a11["raw_gap_pp"]) - (-1.8977)) < 1e-3
    assert abs(-float(a10a11["delta_gap_pp"]) - (+2.7140)) < 1e-3
    assert abs(-float(a10a11["delta_ci_high"]) - (-0.2032)) < 1e-3
    assert abs(-float(a10a11["delta_ci_low"]) - (+5.6864)) < 1e-3

    print("test_identity_mapping.py: identity and directed-pair checks PASS")


if __name__ == "__main__":
    main()
