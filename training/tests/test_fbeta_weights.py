"""compute_class_fn_fp_rates.compute_weights - the F-beta-based class
reweighting formula introduced this session to replace an earlier fn_rate/
fp_rate ratio that had a real, discovered blind spot (see that module's
docstring: a class with zero true positives could score as "neutral"
under the ratio). Previously untested."""
import pytest

from compute_class_fn_fp_rates import compute_weights


def test_perfect_class_gets_weight_one():
    weights, per_class = compute_weights(tp=[10], fp=[0], fn=[0], beta=2.0, min_weight=0.1, max_weight=10.0)
    assert per_class[0]["f_beta"] == 1.0
    assert weights[0].item() == pytest.approx(1.0)


def test_no_evidence_defaults_to_neutral_not_flagged():
    # tp=fp=fn=0: never seen as GT or prediction - must stay distinguishable
    # from "scored as bad" (weight=1, f_beta=None), not just default to a
    # number that looks the same as a real bad score.
    weights, per_class = compute_weights(tp=[0], fp=[0], fn=[0], beta=2.0, min_weight=0.1, max_weight=10.0)
    assert per_class[0]["f_beta"] is None
    assert weights[0].item() == pytest.approx(1.0)


def test_pure_hallucination_gets_min_weight_not_max():
    # The specific bug this formula was introduced to fix: a class with
    # zero true positives and zero ground truth (never a real instance),
    # but the model still predicts it - the OLD fn_rate/fp_rate ratio
    # rewarded this with a high weight (see the module docstring's "boat"
    # example region for the historical failure). It must be down-weighted,
    # not up-weighted.
    weights, per_class = compute_weights(tp=[0], fp=[3], fn=[0], beta=2.0, min_weight=0.1, max_weight=10.0)
    assert per_class[0]["f_beta"] == 0.0
    assert weights[0].item() == pytest.approx(0.1)


def test_all_missed_class_gets_max_weight():
    # Ground truth exists, model finds none of it (zero recall, zero
    # precision since no predictions at all) - the worst real case,
    # should hit the weight ceiling.
    weights, per_class = compute_weights(tp=[0], fp=[0], fn=[5], beta=2.0, min_weight=0.1, max_weight=10.0)
    assert per_class[0]["f_beta"] == 0.0
    assert weights[0].item() == pytest.approx(10.0)


def test_beta_weights_recall_over_precision():
    # Same tp, but one class has more FN (misses) and the other has the
    # same amount of FP (false alarms) instead - beta=2 should penalize
    # the miss-heavy class harder (higher weight) than the false-alarm-
    # heavy one, since beta>1 is defined to weight recall more than
    # precision.
    weights, _ = compute_weights(tp=[10, 10], fp=[0, 5], fn=[5, 0], beta=2.0, min_weight=0.1, max_weight=10.0)
    miss_heavy_weight, false_alarm_heavy_weight = weights.tolist()
    assert miss_heavy_weight > false_alarm_heavy_weight


def test_weights_clamped_to_bounds():
    weights, _ = compute_weights(tp=[0], fp=[0], fn=[1], beta=2.0, min_weight=0.5, max_weight=2.0)
    assert weights[0].item() == pytest.approx(2.0)
