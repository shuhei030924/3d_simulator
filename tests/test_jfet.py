"""接合型 FET（n チャネル JFET, Shichman–Hodges）のテスト。"""
import numpy as np
import pytest

from semisim import metrology as M


def test_idss_at_zero_vgs():
    """Vgs=0・飽和域で Id=Idss。"""
    assert M.jfet_drain_current(0.0, 5.0, idss_a=1e-3, v_pinch_v=-2.0,
                                lambda_per_v=0.0) == pytest.approx(1e-3, rel=1e-9)


def test_cutoff_at_pinchoff():
    """Vgs≤Vp で Id=0（遮断）。"""
    assert M.jfet_drain_current(-2.0, 5.0, v_pinch_v=-2.0) == 0.0
    assert M.jfet_drain_current(-3.0, 5.0, v_pinch_v=-2.0) == 0.0


def test_square_law_saturation():
    """飽和電流は (1−Vgs/Vp)² の二乗則。"""
    idss, vp = 1e-3, -2.0
    vgs = -1.0
    expected = idss * (1.0 - vgs / vp) ** 2
    assert M.jfet_drain_current(vgs, 5.0, idss_a=idss, v_pinch_v=vp,
                                lambda_per_v=0.0) == pytest.approx(expected, rel=1e-9)


def test_triode_saturation_continuity():
    """三極管→飽和は Vds=Vov(=Vgs−Vp) で連続。"""
    vgs, vp = 0.0, -2.0
    vov = vgs - vp
    tri = M.jfet_drain_current(vgs, vov - 1e-6, v_pinch_v=vp, lambda_per_v=0.0)
    sat = M.jfet_drain_current(vgs, vov + 1e-6, v_pinch_v=vp, lambda_per_v=0.0)
    assert tri == pytest.approx(sat, rel=1e-4)


def test_triode_rises_with_vds():
    """三極管域では Vds とともに Id が増える。"""
    i1 = M.jfet_drain_current(0.0, 0.1, v_pinch_v=-2.0, lambda_per_v=0.0)
    i2 = M.jfet_drain_current(0.0, 0.3, v_pinch_v=-2.0, lambda_per_v=0.0)
    assert 0.0 < i1 < i2


def test_channel_length_modulation():
    """λ>0 では飽和域でも Vds とともにわずかに増える。"""
    lo = M.jfet_drain_current(0.0, 3.0, lambda_per_v=0.02)
    hi = M.jfet_drain_current(0.0, 5.0, lambda_per_v=0.02)
    assert hi > lo


def test_transfer_curve_monotonic():
    """伝達特性 Id-Vgs は Vp→0 で単調増加（放物線）。"""
    c = M.jfet_iv_curve()
    assert np.all(np.diff(c["id_transfer"]) >= 0)
    assert c["id_transfer"][0] == pytest.approx(0.0, abs=1e-12)


def test_output_curves_saturate():
    """出力特性は高 Vds で飽和に近づく（Vgs=0 で ≈Idss）。"""
    c = M.jfet_iv_curve(vgs_list=(0.0,), idss_a=1e-3, v_pinch_v=-2.0,
                        lambda_per_v=0.0, vds_max=5.0)
    curve = c["curves"][0.0]
    assert curve[-1] == pytest.approx(1e-3, rel=1e-6)


def test_more_negative_vgs_lowers_current():
    """ゲートをより負にすると飽和電流が下がる。"""
    i0 = M.jfet_drain_current(0.0, 5.0, lambda_per_v=0.0)
    i1 = M.jfet_drain_current(-1.0, 5.0, lambda_per_v=0.0)
    assert i1 < i0


def test_invalid_inputs_raise():
    """正のピンチオフ電圧・負の Idss はエラー。"""
    with pytest.raises(ValueError):
        M.jfet_drain_current(0.0, 1.0, v_pinch_v=2.0)
    with pytest.raises(ValueError):
        M.jfet_drain_current(0.0, 1.0, idss_a=-1.0)
