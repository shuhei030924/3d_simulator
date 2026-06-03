"""MOS トランスコンダクタンス効率 gm/Id のテスト。"""
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


def _vth(w):
    return M.threshold_voltage_v(w, "metal_al", doping_cm3=1e17)["vth_v"]


def test_gm_id_equals_gm_over_id():
    """gm/Id = gm ÷ Id。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_gm_id_efficiency(w, "metal_al", vg=vth + 0.2, vd=1.5)
    assert r["gm_id_per_v"] == pytest.approx(r["gm_s"] / r["id_a"], rel=1e-9)


def test_efficiency_highest_in_weak_inversion():
    """弱反転（Vov<0）で gm/Id が最大、過剰電圧とともに低下。"""
    w = _mos()
    vth = _vth(w)
    weak = M.mos_gm_id_efficiency(w, "metal_al", vg=vth - 0.2, vd=1.5)["gm_id_per_v"]
    mod = M.mos_gm_id_efficiency(w, "metal_al", vg=vth + 0.1, vd=1.5)["gm_id_per_v"]
    strong = M.mos_gm_id_efficiency(w, "metal_al", vg=vth + 0.6, vd=1.5)["gm_id_per_v"]
    assert weak > mod > strong


def test_weak_inversion_approaches_ideal_max():
    """深い弱反転で gm/Id → 1/(n·Vt) の理論上限に近づく。"""
    w = _mos()
    vth = _vth(w)
    r = M.mos_gm_id_efficiency(w, "metal_al", vg=vth - 0.3, vd=1.5)
    assert r["gm_id_per_v"] == pytest.approx(r["gm_id_max_ideal"], rel=0.05)
    assert r["gm_id_per_v"] < r["gm_id_max_ideal"]  # 上限は超えない


def test_ideal_max_scales_with_subthreshold_n():
    """理論上限 1/(n·Vt) は n が大きいほど小さい。"""
    w = _mos()
    vth = _vth(w)
    lo_n = M.mos_gm_id_efficiency(w, "metal_al", vg=vth, vd=1.5,
                                  subthreshold_n=1.0)["gm_id_max_ideal"]
    hi_n = M.mos_gm_id_efficiency(w, "metal_al", vg=vth, vd=1.5,
                                  subthreshold_n=1.6)["gm_id_max_ideal"]
    assert lo_n > hi_n


def test_no_gate_zero_efficiency():
    """ゲート無し（Id=0）では gm/Id=0。"""
    cfg = WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.mos_gm_id_efficiency(w, "metal_al", vg=1.0, vd=1.5)
    assert r["gm_id_per_v"] == 0.0
