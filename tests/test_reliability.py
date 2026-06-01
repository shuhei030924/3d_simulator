"""信頼性メトロロジ（電流密度/EM・絶縁破壊・歩留り）のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _wire(nz=30, ny=30, nx=60, pitch=0.05) -> Wafer:
    cfg = WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get("oxide").id
    return w


def test_current_density_matches_definition():
    """J=I/A が断面積から厳密に求まる。"""
    w = _wire()
    p = w.config.pitch_um
    w.grid[10:18, 10:20, :] = materials.get("metal_cu").id  # 8(z)x10(y) 断面
    st = M.current_density_stats(w, "metal_cu", 1.0, "x")
    area = 8 * 10 * p ** 2
    assert st["area_min_um2"] == pytest.approx(area)
    assert st["j_max_a_cm2"] == pytest.approx(1e-3 / (area * 1e-8), rel=1e-6)


def test_current_density_necking_increases_j():
    """断面が細る箇所（ネッキング）で電流密度が上がる。"""
    w = _wire()
    w.grid[10:18, 10:20, :] = materials.get("metal_cu").id
    j_uniform = M.current_density_stats(w, "metal_cu", 1.0, "x")["j_max_a_cm2"]
    # x の一部で y 幅を半減
    w.grid[10:18, 10:15, 30:40] = materials.get("oxide").id
    st = M.current_density_stats(w, "metal_cu", 1.0, "x")
    assert st["j_max_a_cm2"] > j_uniform
    assert 30 <= st["bottleneck_index"] < 40


def test_current_density_open_is_inf():
    """断線（途中で断面 0）は電流密度 inf。"""
    w = _wire()
    w.grid[10:18, 10:20, :30] = materials.get("metal_cu").id  # x 半分のみ → 途切れない
    # x>=30 を空けると present 範囲は 0..29 で連続 → 断線ではない。
    # 真ん中を抜いて断線させる
    w2 = _wire()
    w2.grid[10:18, 10:20, :] = materials.get("metal_cu").id
    w2.grid[10:18, 10:20, 28:32] = materials.get("oxide").id
    assert M.current_density_stats(w2, "metal_cu", 1.0, "x")["j_max_a_cm2"] == float("inf")


def test_em_risk_pass_and_fail():
    """EM: 低電流は許容内、過大電流で限界超過 fail。"""
    w = _wire()
    w.grid[10:14, 10:14, :] = materials.get("metal_al").id  # 細い Al 配線
    safe = M.electromigration_risk(w, "metal_al", 0.01, "x")
    assert not safe["fail"] and safe["margin"] > 1.0
    hot = M.electromigration_risk(w, "metal_al", 100.0, "x")
    assert hot["fail"] and hot["margin"] < 1.0


def test_em_cu_more_robust_than_al():
    """同条件で Cu の方が Al より EM 余裕が大きい（許容 J が高い）。"""
    wa = _wire()
    wa.grid[10:14, 10:14, :] = materials.get("metal_al").id
    wc = _wire()
    wc.grid[10:14, 10:14, :] = materials.get("metal_cu").id
    ma = M.electromigration_risk(wa, "metal_al", 1.0, "x")["margin"]
    mc = M.electromigration_risk(wc, "metal_cu", 1.0, "x")["margin"]
    assert mc > ma


def test_dielectric_breakdown_scales_with_voltage():
    """絶縁破壊: 電界は電圧に比例し、破壊電界超過で fail。"""
    w = _wire()
    w.grid[10:13, :, :] = materials.get("metal_al").id
    w.grid[17:20, :, :] = materials.get("metal_cu").id  # oxide 間隙
    low = M.dielectric_breakdown(w, "metal_al", "metal_cu", 3.0)
    high = M.dielectric_breakdown(w, "metal_al", "metal_cu", 300.0)
    assert high["field_mv_cm"] == pytest.approx(100 * low["field_mv_cm"], rel=1e-6)
    assert not low["fail"]
    assert high["fail"]
    assert low["breakdown_field_mv_cm"] == materials.get("oxide").breakdown_field_mv_cm


def test_dielectric_breakdown_contact_fails():
    """接触（間隔 0）は即破壊。"""
    w = _wire()
    w.grid[10:13, :, :] = materials.get("metal_al").id
    w.grid[13:16, :, :] = materials.get("metal_cu").id  # 接触
    res = M.dielectric_breakdown(w, "metal_al", "metal_cu", 1.0)
    assert res["fail"] and res["gap_um"] == 0.0


def test_yield_models_monotonic():
    """欠陥密度が増えると歩留りは下がり、欠陥 0 で歩留り 1。"""
    for model in ("poisson", "murphy", "seeds"):
        assert M.yield_estimate(0.0, 1.0, model) == 1.0
        y_lo = M.yield_estimate(0.05, 1.0, model)
        y_hi = M.yield_estimate(0.5, 1.0, model)
        assert 0.0 < y_hi < y_lo < 1.0
    # Murphy ≥ Poisson（同 AD でばらつき考慮の方が高歩留り）
    assert M.yield_estimate(0.5, 1.0, "murphy") >= M.yield_estimate(0.5, 1.0, "poisson")


def test_yield_negative_raises():
    with pytest.raises(ValueError):
        M.yield_estimate(-1.0, 1.0)
    with pytest.raises(ValueError):
        M.yield_estimate(0.1, 1.0, "bogus")


def test_killer_defect_count_clean_wafer():
    """欠陥のないきれいなウェハではキラー欠陥 0。"""
    w = _wire()
    w.grid[10:14, :, :] = materials.get("metal_cu").id
    assert M.killer_defect_count(w) == 0
