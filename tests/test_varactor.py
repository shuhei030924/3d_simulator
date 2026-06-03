"""バラクタ（電圧可変容量）C-V・チューニングレンジのテスト。"""
import numpy as np
import pytest

from semisim import metrology as M


def test_zero_bias_capacitance_is_cj0():
    """ゼロバイアスで C=Cj0。"""
    r = M.varactor_cv(cj0_ff=2.0, vr_min=0.0)
    assert r["capacitance_ff"][0] == pytest.approx(2.0, rel=1e-12)


def test_cv_formula():
    """C(Vr)=Cj0/(1+Vr/Vbi)^m に一致。"""
    r = M.varactor_cv(cj0_ff=2.0, vbi=0.7, grading_m=0.5, vr_min=0, vr_max=4)
    k = int(r["reverse_bias_v"].searchsorted(2.0))
    vr = r["reverse_bias_v"][k]
    expected = 2.0 / (1.0 + vr / 0.7) ** 0.5
    assert r["capacitance_ff"][k] == pytest.approx(expected, rel=1e-9)


def test_capacitance_monotonic_decreasing():
    """逆バイアスとともに容量は単調減少。"""
    r = M.varactor_cv()
    assert np.all(np.diff(r["capacitance_ff"]) < 0)


def test_tuning_ratio_formula():
    """容量可変比 TR=(1+Vr_max/Vbi)^m / (1+Vr_min/Vbi)^m。"""
    r = M.varactor_cv(vbi=0.7, grading_m=0.5, vr_min=0.0, vr_max=4.0)
    expected = (1.0 + 4.0 / 0.7) ** 0.5
    assert r["tuning_ratio"] == pytest.approx(expected, rel=1e-9)


def test_freq_tuning_ratio_is_sqrt():
    """周波数同調比 = √(容量可変比)（f∝1/√C）。"""
    r = M.varactor_cv()
    assert r["freq_tuning_ratio"] == pytest.approx(np.sqrt(r["tuning_ratio"]), rel=1e-12)


def test_hyperabrupt_wider_tuning_than_abrupt():
    """超階段接合(m 大)は階段接合(m=0.5)より広いチューニングレンジ。"""
    abrupt = M.varactor_cv(grading_m=0.5, vr_max=4.0)["tuning_ratio"]
    hyper = M.varactor_cv(grading_m=2.0, vr_max=4.0)["tuning_ratio"]
    assert hyper > abrupt


def test_cmax_cmin():
    """c_max は最小逆バイアス側、c_min は最大逆バイアス側。"""
    r = M.varactor_cv()
    assert r["c_max_ff"] == pytest.approx(r["capacitance_ff"][0])
    assert r["c_min_ff"] == pytest.approx(r["capacitance_ff"][-1])
    assert r["c_max_ff"] > r["c_min_ff"]


def test_invalid_inputs_raise():
    """非正パラメータ・不正範囲はエラー。"""
    with pytest.raises(ValueError):
        M.varactor_cv(cj0_ff=0.0)
    with pytest.raises(ValueError):
        M.varactor_cv(grading_m=0.0)
    with pytest.raises(ValueError):
        M.varactor_cv(vr_min=2.0, vr_max=1.0)
