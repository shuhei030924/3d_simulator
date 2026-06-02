"""MOS I-V 特性（Idsat・三極管/飽和・サブスレショルド）のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

VT = 0.025852  # kT/q [V]


def _mos(pitch: float = 0.001) -> Wafer:
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:16, :, :] = materials.get("metal_al").id
    return w


def _vth(w):
    return M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]


def test_idsat_square_law():
    """飽和電流は過剰電圧の二乗に比例（2倍の Vov → 4倍の Id）。"""
    w = _mos()
    vth = _vth(w)
    id1 = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=2.0)
    id2 = M.mos_drain_current(w, "metal_al", vg=vth + 1.0, vd=2.0)
    assert id2 / id1 == pytest.approx(4.0, rel=1e-3)


def test_triode_to_saturation():
    """Vd を上げると三極管→飽和で電流が滑らかに頭打ち（λ=0 でほぼ一定）。"""
    w = _mos()
    vth = _vth(w)
    id_lin = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=0.2)
    id_deep1 = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=1.5)
    id_deep2 = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=2.0)
    assert id_lin < id_deep1                              # 三極管 < 飽和
    assert id_deep2 == pytest.approx(id_deep1, rel=1e-3)  # 深飽和でほぼ一定


def test_channel_length_modulation():
    """λ>0 で飽和電流が Vd と共に緩やかに増える。"""
    w = _mos()
    vth = _vth(w)
    lo = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=0.6, lambda_per_v=0.1)
    hi = M.mos_drain_current(w, "metal_al", vg=vth + 0.5, vd=1.8, lambda_per_v=0.1)
    assert hi > lo


def test_subthreshold_slope():
    """サブスレショルド傾斜 SS ≈ n·(kT/q)·ln10（n=1 で ~60mV/dec）。"""
    w = _mos()
    vth = _vth(w)
    for n in (1.0, 1.5):
        # 深いサブスレショルドで漸近的な SS=n·(kT/q)·ln10 を抽出
        i1 = M.mos_drain_current(w, "metal_al", vg=vth - 0.6, vd=1.0, subthreshold_n=n)
        i2 = M.mos_drain_current(w, "metal_al", vg=vth - 0.5, vd=1.0, subthreshold_n=n)
        ss_mv = 0.1 / (np.log10(i2) - np.log10(i1)) * 1000
        assert ss_mv == pytest.approx(n * VT * np.log(10) * 1000, rel=0.02)


def test_current_scales_with_w_over_l():
    """ドレイン電流は W/L に比例する。"""
    w = _mos()
    vth = _vth(w)
    i1 = M.mos_drain_current(w, "metal_al", vg=vth + 0.8, vd=2.0, w_over_l=5.0)
    i2 = M.mos_drain_current(w, "metal_al", vg=vth + 0.8, vd=2.0, w_over_l=20.0)
    assert i2 / i1 == pytest.approx(4.0, rel=1e-6)


def test_iv_curve_shape():
    """出力特性 Id-Vd 族の形状と単調性。"""
    w = _mos()
    vth = _vth(w)
    iv = M.mos_iv_curve(w, "metal_al", vg_list=(vth + 0.3, vth + 0.8), vd_max=2.0, n_points=21)
    assert iv["vd"].shape == (21,)
    for _vg, idc in iv["curves"].items():
        assert idc[0] == pytest.approx(0.0, abs=1e-12)  # Vd=0 で Id=0
        assert idc[-1] > idc[1]                          # 立ち上がり
