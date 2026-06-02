"""短チャネル Vth(SCE/DIBL)・接合リーク電流のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _mos() -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def test_vth_rolloff_with_short_channel():
    """チャネル長が短いほど SCE で Vth が下がる（ロールオフ）。"""
    w = _mos()
    v_long = M.short_channel_vth_v(w, "metal_al", channel_length_um=1.0, vds=0.05)
    v_short = M.short_channel_vth_v(w, "metal_al", channel_length_um=0.03, vds=0.05)
    assert v_long["dvth_sce_v"] < v_short["dvth_sce_v"]
    assert v_short["vth_v"] < v_long["vth_v"]
    assert v_long["vth_v"] == pytest.approx(v_long["vth_long_v"], abs=0.01)  # 長Lで≈長ch


def test_dibl_lowers_vth_with_vds():
    """DIBL: ドレイン電圧が高いほど Vth が下がる。"""
    w = _mos()
    lo = M.short_channel_vth_v(w, "metal_al", channel_length_um=0.05, vds=0.05)
    hi = M.short_channel_vth_v(w, "metal_al", channel_length_um=0.05, vds=1.0)
    assert hi["vth_v"] < lo["vth_v"]
    assert hi["dvth_dibl_v"] > lo["dvth_dibl_v"]


def test_short_channel_vth_invalid_length():
    with pytest.raises(ValueError):
        M.short_channel_vth_v(_mos(), "metal_al", channel_length_um=0.0)


def test_junction_leakage_increases_with_temperature():
    """接合リークは温度が高いほど指数的に増える（~10°C で倍増オーダー）。"""
    lo = M.junction_leakage_a(1.0, 27.0)
    hi = M.junction_leakage_a(1.0, 125.0)
    assert hi > lo > 0
    ratio = M.junction_leakage_a(1.0, 35.0) / M.junction_leakage_a(1.0, 27.0)
    assert 1.5 < ratio < 2.5  # 約 8°C で倍増


def test_junction_leakage_scales_with_area():
    """リーク電流は接合面積に比例する。"""
    assert M.junction_leakage_a(4.0, 85.0) == pytest.approx(
        4.0 * M.junction_leakage_a(1.0, 85.0), rel=1e-9)


def test_junction_leakage_negative_area_raises():
    with pytest.raises(ValueError):
        M.junction_leakage_a(-1.0, 27.0)


def test_tlm_degenerate_slope_inf_transfer_length():
    """TLM 回帰の傾きが実質ゼロなら伝送長は inf（巨大ゴミ値を出さない）。"""
    ext = M.tlm_extract([2, 5, 10.0], [40.0, 40.0, 40.0], 10.0)
    assert ext["transfer_length_um"] == float("inf")
