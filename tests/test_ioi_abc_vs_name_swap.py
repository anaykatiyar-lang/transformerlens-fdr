"""Analysis helper tests for the ABC/name-swap IOI comparison runner."""

import csv
import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from benchmarks import ioi_abc_vs_name_swap as runner


def sample_effects():
    return np.arange(144, dtype=float).reshape(12, 12) / 100


def test_rank_heads_orders_by_absolute_effect():
    effects = np.zeros((12, 12))
    effects[1, 3] = -4
    effects[0, 7] = 2
    assert runner.rank_heads(effects, 2) == [(1, 3), (0, 7)]


def test_rank_heads_breaks_ties_by_layer_and_head():
    effects = np.zeros((12, 12))
    effects[2, 1] = -2
    effects[0, 5] = 2
    effects[0, 1] = -2
    assert runner.rank_heads(effects, 3) == [(0, 1), (0, 5), (2, 1)]


@pytest.mark.parametrize("bad_k", [0, 145])
def test_rank_heads_rejects_out_of_range_k(bad_k):
    with pytest.raises(ValueError, match="top-K"):
        runner.rank_heads(sample_effects(), bad_k)


def test_rank_heads_rejects_nonfinite():
    effects = sample_effects()
    effects[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        runner.rank_heads(effects, 9)


def test_compare_effects_absolute_ranks_are_sign_invariant():
    effects = sample_effects()
    result = runner.compare_effects(effects, -effects)
    assert result["spearman_absolute_effects"] == pytest.approx(1)
    assert result["jaccard_top_9"] == 1
    assert result["jaccard_top_26"] == 1


def test_compare_effects_records_topk_disagreement():
    effects = sample_effects()
    result = runner.compare_effects(effects, effects[::-1].copy())
    assert result["jaccard_top_9"] == 0
    assert len(result["top_26_abc"]) == 26
    assert result["spearman_absolute_effects"] < 1


def test_compare_effects_handles_constant_ranks():
    result = runner.compare_effects(np.ones((12, 12)), sample_effects())
    assert result["spearman_absolute_effects"] is None


def test_compare_effects_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="12 x 12"):
        runner.compare_effects(np.ones((12, 12)), np.ones((3, 4)))


def test_select_examples_preserves_template_groups_and_order():
    examples = [{"template": 3, "row": 1}, {"template": 7, "row": 2},
                {"template": 3, "row": 3}]
    result = runner.select_examples(examples, [3])
    assert result == [examples[0], examples[2]]


def test_select_examples_rejects_missing_template():
    with pytest.raises(ValueError, match="Missing"):
        runner.select_examples([{"template": 3}], [3, 5])


def test_write_head_table_records_each_144_head_family(tmp_path):
    effects = sample_effects()
    corrected = {
        "bh": (np.ones((12, 12)), np.zeros((12, 12), dtype=bool)),
        "by": (np.ones((12, 12)), np.zeros((12, 12), dtype=bool)),
    }
    path = tmp_path / "heads.csv"
    result = runner.write_head_table(
        path, effects, -2.0, torch.full((12, 12), 0.5), corrected,
    )
    assert np.allclose(result, effects * -2)
    with path.open() as data:
        rows = list(csv.DictReader(data))
    assert len(rows) == 144
    assert rows[0]["layer"] == "0" and rows[-1]["head"] == "11"
    assert float(rows[1]["unnormalized_effect"]) == pytest.approx(-0.02)


def test_write_layer_table_emits_12_layers(tmp_path):
    path = tmp_path / "layers.csv"
    runner.write_layer_table(path, sample_effects(), -sample_effects())
    with path.open() as data:
        rows = list(csv.DictReader(data))
    assert len(rows) == 12
    assert all(float(row["abs_mean_difference"]) == pytest.approx(0) for row in rows)


def test_write_summary_is_valid_json(tmp_path):
    runner.write_summary(tmp_path, {"run": "smoke", "value": 0.01})
    assert json.loads((tmp_path / "summary.json").read_text())["value"] == 0.01


def test_audit_one_skips_unusable_gap(tmp_path, monkeypatch):
    fake_preflight = SimpleNamespace(
        gap_usable=False,
        relative_gap_tolerance=1e-6,
        resolution_report={"p_min": 0.01},
        format_summary=lambda: "unusable gap",
    )

    class NoAudit:
        def check_setup(self, *args, **kwargs):
            return fake_preflight

        def run_patching_audit(self, *args, **kwargs):
            raise AssertionError("Must not patch with an unusable denominator")

    monkeypatch.setattr(runner, "gap_diagnostics", lambda *args: {
        "mean": 0.0, "std": 1.0, "fraction_positive": 0.5,
    })
    record, result, effects, families = runner.audit_one(
        None, NoAudit(), (None, None, None, None, [0, 1]),
        tmp_path, "name_swap", "selection16",
    )
    assert record["status"] == "unusable_gap"
    assert result is None and effects is None and families is None


def test_fdr_families_uses_two_144_head_corrections(monkeypatch):
    calls = []

    def mock_adjust(p_values, alpha, method):
        calls.append((p_values.shape, alpha, method))
        return torch.ones_like(p_values), torch.zeros_like(p_values, dtype=torch.bool)

    fake_module = types.ModuleType("transformerlens_fdr")
    fake_stats = types.ModuleType("transformerlens_fdr.stats")
    fake_stats.apply_fdr_adjustment = mock_adjust
    monkeypatch.setitem(sys.modules, "transformerlens_fdr", fake_module)
    monkeypatch.setitem(sys.modules, "transformerlens_fdr.stats", fake_stats)
    result = runner.fdr_families(torch.full((12, 12), 0.5))
    assert set(result) == {"bh", "by"}
    assert calls == [
        (torch.Size((12, 12)), 0.05, "fdr_bh"),
        (torch.Size((12, 12)), 0.05, "fdr_by"),
    ]


def test_fdr_families_rejects_wrong_family_size():
    with pytest.raises(ValueError, match="144 heads"):
        runner.fdr_families(torch.ones(288))


def test_discovery_overlap_is_descriptive():
    a = np.zeros((12, 12), dtype=bool)
    b = np.zeros((12, 12), dtype=bool)
    a[0, 1] = a[3, 5] = True
    b[0, 1] = b[4, 7] = True
    families_a = {"bh": (None, a), "by": (None, np.zeros_like(a))}
    families_b = {"bh": (None, b), "by": (None, np.zeros_like(a))}
    result = runner.compare_discoveries(families_a, families_b)
    assert result["bh"]["jaccard"] == pytest.approx(1 / 3)
    assert result["bh"]["shared_heads"] == [[0, 1]]
    assert result["by"]["jaccard"] is None
