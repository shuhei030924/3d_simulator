"""ゲート直接トンネルリークと high-k 効果のテスト。"""
import math

import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _gate(diel: str, nvox: int, pitch=0.001) -> Wafer:
    w = Wafer(WaferConfig(nx=20, ny=20, nz=30, pitch_um=pitch, substrate_um=0.0))
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:10 + nvox, :, :] = materials.get(diel).id
    g[10 + nvox:14 + nvox, :, :] = materials.get("metal_al").id
    return w


def test_leakage_exponential_in_thickness():
    """J_g ∝ exp(−t_phys/t_char)。"""
    thin = M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al", vg=1.0)
    thick = M.gate_tunneling_leakage(_gate("oxide", 3), "metal_al", vg=1.0)
    ratio = thick["jg_a_cm2"] / thin["jg_a_cm2"]
    dt_nm = thick["phys_thickness_nm"] - thin["phys_thickness_nm"]
    assert ratio == pytest.approx(math.exp(-dt_nm / 0.3), rel=1e-9)


def test_leakage_quadratic_in_vg():
    """J_g ∝ Vg²。"""
    r1 = M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al", vg=1.0)
    r2 = M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al", vg=2.0)
    assert r2["jg_a_cm2"] == pytest.approx(4.0 * r1["jg_a_cm2"], rel=1e-9)


def test_highk_reduces_leakage_at_same_eot():
    """同じ EOT でも high-k（厚い物理膜）はリークが桁違いに低い。"""
    sio2 = _gate("oxide", 2)     # EOT≈2nm, t_phys=2nm
    hfo2 = _gate("hafnia", 13)   # EOT≈2nm, t_phys=13nm
    g_sio2 = M.mos_gate_capacitance(sio2, "metal_al")
    g_hfo2 = M.mos_gate_capacitance(hfo2, "metal_al")
    # EOT がほぼ同じ
    assert g_hfo2["eot_nm"] == pytest.approx(g_sio2["eot_nm"], rel=0.1)
    # 物理膜厚は high-k が厚い
    assert g_hfo2["phys_thickness_nm"] > g_sio2["phys_thickness_nm"]
    # リークは high-k が圧倒的に低い
    j_sio2 = M.gate_tunneling_leakage(sio2, "metal_al")["jg_a_cm2"]
    j_hfo2 = M.gate_tunneling_leakage(hfo2, "metal_al")["jg_a_cm2"]
    assert j_hfo2 < j_sio2 / 1e6


def test_total_current_is_density_times_area():
    """総リーク電流 = J_g × ゲート面積。"""
    r = M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al")
    area_cm2 = r["gate_area_um2"] * 1e-8
    assert r["ig_total_a"] == pytest.approx(r["jg_a_cm2"] * area_cm2, rel=1e-9)


def test_phys_thickness_reported():
    """物理膜厚が正しく報告される（2 ボクセル @1nm = 2nm）。"""
    r = M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al")
    assert r["phys_thickness_nm"] == pytest.approx(2.0, rel=1e-6)


def test_no_gate_zero_leakage():
    """ゲート誘電体が無ければリーク 0。"""
    w = Wafer(WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.001, substrate_um=0.0))
    w.grid[:10, :, :] = materials.get("silicon").id
    r = M.gate_tunneling_leakage(w, "metal_al")
    assert r["jg_a_cm2"] == 0.0
    assert r["ig_total_a"] == 0.0


def test_invalid_tchar_raises():
    """t_char が非正ならエラー。"""
    with pytest.raises(ValueError):
        M.gate_tunneling_leakage(_gate("oxide", 2), "metal_al", t_char_nm=0.0)
