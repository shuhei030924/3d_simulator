"""リソ CD のモンテカルロ ばらつき（統計 CDU）のテスト。"""
import pytest

from semisim import litho as L


def test_zero_variation_nominal():
    """ばらつき 0 では CDU=0・歩留り 1・平均 CD=公称。"""
    r = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.0, focus_sigma_um=0.0, n_samples=200)
    assert r["cdu_3sigma_um"] == pytest.approx(0.0, abs=1e-9)
    assert r["yield_in_spec"] == 1.0
    assert r["mean_cd_um"] == pytest.approx(0.10, abs=0.003)


def test_more_variation_larger_cdu_lower_yield():
    """ばらつきが大きいほど CDU(3σ) は増え、規格内歩留りは下がる。"""
    small = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.02, focus_sigma_um=0.03,
                             n_samples=1500, seed=1)
    large = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.08, focus_sigma_um=0.10,
                             n_samples=1500, seed=1)
    assert large["cdu_3sigma_um"] > small["cdu_3sigma_um"]
    assert large["yield_in_spec"] < small["yield_in_spec"]


def test_reproducible_with_seed():
    """同じ seed で結果が再現する。"""
    a = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.04, n_samples=500, seed=7)
    b = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.04, n_samples=500, seed=7)
    assert a["mean_cd_um"] == b["mean_cd_um"]
    assert a["cdu_3sigma_um"] == b["cdu_3sigma_um"]


def test_yield_in_range():
    """歩留りは 0〜1。"""
    r = L.monte_carlo_cd(0.10, 0.005, dose_sigma=0.05, focus_sigma_um=0.06, n_samples=800)
    assert 0.0 <= r["yield_in_spec"] <= 1.0
    assert r["cd_samples"].shape == (800,)
