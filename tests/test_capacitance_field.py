"""2.5D 静電界ソルバによる寄生容量（フリンジ込み）のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

EPS0_FF = 8.854e-3


def _plates(diel: str, gap: int, full: bool = True, pw: int = 12,
            n: int = 24, pitch: float = 0.1) -> Wafer:
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get(diel).id
    if full:
        g[6:9, :, :] = materials.get("metal_al").id
        g[9 + gap:12 + gap, :, :] = materials.get("metal_cu").id
    else:
        a, b = (n - pw) // 2, (n + pw) // 2
        g[6:9, a:b, a:b] = materials.get("metal_al").id
        g[9 + gap:12 + gap, a:b, a:b] = materials.get("metal_cu").id
    return w


def test_field_matches_analytic_full_plates():
    """全面平行平板では静電界ソルバが解析値 ε0·εr·A/d に一致。"""
    n, pitch, gap = 24, 0.1, 4
    w = _plates("oxide", gap, full=True, n=n, pitch=pitch)
    cf = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    area = (n * pitch) ** 2
    sep = (gap + 1) * pitch  # セル中心基準の電極面間隔
    analytic = EPS0_FF * 3.9 * area / sep
    assert cf == pytest.approx(analytic, rel=0.03)


def test_field_scales_linearly_with_permittivity():
    """一様誘電体では容量が εr に厳密比例する。"""
    c_ox = M.parasitic_capacitance_field_ff(_plates("oxide", 4), "metal_al", "metal_cu")
    c_ni = M.parasitic_capacitance_field_ff(_plates("nitride", 4), "metal_al", "metal_cu")
    ratio = materials.get("nitride").rel_permittivity / materials.get("oxide").rel_permittivity
    assert c_ni == pytest.approx(c_ox * ratio, rel=0.02)


def test_field_exceeds_parallel_plate_for_finite_electrodes():
    """有限幅電極ではフリンジ電界により静電界ソルバ > 平行平板近似。"""
    w = _plates("oxide", 6, full=False, pw=12)
    cf = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    cl = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")
    assert cf > cl > 0


def test_field_decreases_with_gap():
    """間隙が広いほど容量は小さい。"""
    c_near = M.parasitic_capacitance_field_ff(_plates("oxide", 4), "metal_al", "metal_cu")
    c_far = M.parasitic_capacitance_field_ff(_plates("oxide", 10), "metal_al", "metal_cu")
    assert c_far < c_near


def test_field_absent_material_zero():
    """片方の導体が無ければ 0。"""
    w = _plates("oxide", 4)
    w.grid[w.grid == materials.get("metal_cu").id] = materials.get("oxide").id
    assert M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu") == 0.0
