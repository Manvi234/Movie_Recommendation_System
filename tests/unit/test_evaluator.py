import math
import sys

import pytest

sys.path.insert(0, "src")

from evaluation.evaluator import ndcg_at_k, quality_gate


def test_ndcg_hit_at_position_0():
    assert ndcg_at_k(0, 10) == pytest.approx(1.0 / 1.0)  # log2(0+2) = log2(2) = 1


def test_ndcg_hit_at_position_1():
    assert ndcg_at_k(1, 10) == pytest.approx(1.0 / math.log2(3))


def test_ndcg_miss():
    assert ndcg_at_k(None, 10) == 0.0
    assert ndcg_at_k(10, 10) == 0.0


def test_quality_gate_passes():
    assert quality_gate({"hr": 0.20, "ndcg": 0.15}) is True


def test_quality_gate_fails_hr():
    assert quality_gate({"hr": 0.10, "ndcg": 0.15}) is False


def test_quality_gate_fails_ndcg():
    assert quality_gate({"hr": 0.20, "ndcg": 0.05}) is False
