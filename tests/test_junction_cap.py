"""pn 接合 空乏層容量・ビルトイン電位のテスト。"""
import numpy as np
import pytest

from semisim import metrology as M


def test_vbi_increases_with_doping():
    """ビルトイン電位はドーピングが高いほど大きい（Si で ~0.7〜1.0V）。"""
    v_lo = M.junction_capacitance(1e16, 1e18, 0.0)["vbi_v"]
    v_hi = M.junction_capacitance(1e18, 1e18, 0.0)["vbi_v"]
    assert 0.6 < v_lo < v_hi < 1.1


def test_reverse_bias_widens_depletion_lowers_cj():
    """逆バイアスが深いほど空乏層は広く接合容量は小さい。"""
    d0 = M.junction_capacitance(1e17, 1e17, 0.0)
    d5 = M.junction_capacitance(1e17, 1e17, 5.0)
    assert d5["depletion_width_um"] > d0["depletion_width_um"]
    assert d5["cj_ff_per_um2"] < d0["cj_ff_per_um2"]


def test_one_over_cj2_linear_in_bias():
    """1/Cj² は逆バイアスに対し直線（C-V プロファイリング）。"""
    cv = M.junction_cv_curve(1e17, 1e17, v_max_reverse=5, n_points=11)
    coef = np.polyfit(cv["reverse_bias_v"], cv["one_over_cj2"], 1)
    resid = cv["one_over_cj2"] - np.polyval(coef, cv["reverse_bias_v"])
    r2 = 1 - np.var(resid) / np.var(cv["one_over_cj2"])
    assert r2 > 0.9999  # ほぼ完全な直線


def test_cv_x_intercept_equals_minus_vbi():
    """1/Cj²-V 直線の x 切片は −Vbi に等しい。"""
    cv = M.junction_cv_curve(1e17, 5e17, v_max_reverse=5, n_points=11)
    slope, intercept = np.polyfit(cv["reverse_bias_v"], cv["one_over_cj2"], 1)
    x_intercept = -intercept / slope
    assert x_intercept == pytest.approx(-cv["vbi_v"], rel=1e-3)


def test_total_cap_scales_with_area():
    d = M.junction_capacitance(1e17, 1e17, 1.0, area_um2=4.0)
    assert d["cj_total_ff"] == pytest.approx(d["cj_ff_per_um2"] * 4.0, rel=1e-9)


def test_invalid_params_raise():
    with pytest.raises(ValueError):
        M.junction_capacitance(0.0, 1e17, 0.0)
    with pytest.raises(ValueError):
        M.junction_capacitance(1e17, 1e17, -1.0)  # 逆バイアスは≥0
