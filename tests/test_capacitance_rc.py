"""寄生容量・RC 遅延メトロロジのテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

EPS0_FF = 8.854e-3  # C[fF] = EPS0_FF·εr·A[µm²]/d[µm]


def _two_plates(eps_name: str, gap_vox: int, n: int = 40, pitch: float = 0.1) -> Wafer:
    """誘電体 eps_name 中に金属 Al/Cu の平行平板（全面）を gap_vox 間隔で積む。"""
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get(eps_name).id
    g[10:13, :, :] = materials.get("metal_al").id  # 下板 top 面 z=12
    z2 = 12 + gap_vox
    g[z2:z2 + 3, :, :] = materials.get("metal_cu").id  # 上板 bottom 面 z=z2
    return w


def test_permittivity_values():
    """主要誘電体の比誘電率が物理的に妥当な順序で設定されている。"""
    assert materials.get("air").rel_permittivity == 1.0
    assert materials.get("low_k").rel_permittivity < materials.get("oxide").rel_permittivity
    assert materials.get("oxide").rel_permittivity < materials.get("nitride").rel_permittivity
    assert materials.get("hafnia").rel_permittivity > materials.get("nitride").rel_permittivity
    # 導体は誘電体扱いしない（εr=0）
    assert materials.get("metal_cu").rel_permittivity == 0.0


def test_capacitance_matches_parallel_plate():
    """平行平板の解析値 ε0·εr·A/g に厳密一致する。"""
    n, pitch = 40, 0.1
    a_plate = (n * pitch) ** 2
    for eps_name, gap_vox in [("oxide", 4), ("oxide", 8), ("nitride", 4), ("low_k", 8)]:
        w = _two_plates(eps_name, gap_vox, n, pitch)
        c = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")
        g_um = gap_vox * pitch
        eps_r = materials.get(eps_name).rel_permittivity
        analytic = EPS0_FF * eps_r * a_plate / g_um
        assert c == pytest.approx(analytic, rel=0.02)


def test_capacitance_scales_with_permittivity():
    """誘電体を high-k に替えると εr 比で容量が増える。"""
    c_ox = M.parasitic_capacitance_ff(_two_plates("oxide", 6), "metal_al", "metal_cu")
    c_ni = M.parasitic_capacitance_ff(_two_plates("nitride", 6), "metal_al", "metal_cu")
    ratio = materials.get("nitride").rel_permittivity / materials.get("oxide").rel_permittivity
    assert c_ni == pytest.approx(c_ox * ratio, rel=0.02)


def test_capacitance_decreases_with_gap():
    """間隙が広いほど容量は小さい（C ∝ 1/g）。"""
    c_near = M.parasitic_capacitance_ff(_two_plates("oxide", 4), "metal_al", "metal_cu")
    c_far = M.parasitic_capacitance_ff(_two_plates("oxide", 8), "metal_al", "metal_cu")
    assert c_far < c_near
    assert c_near == pytest.approx(2 * c_far, rel=0.02)  # 間隙2倍で容量半分


def test_capacitance_third_conductor_shields():
    """2 板の間に第 3 導体（接地シールド）を挟むと対向容量が遮蔽され減る。"""
    w = _two_plates("oxide", 12)
    c_open = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")
    # 中間に W のシールド板を挿入
    w.grid[18:20, :, :] = materials.get("tungsten").id
    c_shield = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")
    assert c_shield < c_open
    assert c_shield == pytest.approx(0.0, abs=1e-9)  # 直接対向が完全遮蔽


def test_capacitance_absent_material_zero():
    """片方の導体が存在しなければ容量は 0。"""
    w = _two_plates("oxide", 6)
    w.grid[w.grid == materials.get("metal_cu").id] = materials.get("oxide").id
    assert M.parasitic_capacitance_ff(w, "metal_al", "metal_cu") == 0.0
    assert M.parasitic_capacitance_ff(w, "metal_al", "metal_al") == 0.0


def test_rc_delay_positive_and_open():
    """RC 遅延は導通配線で正の有限値、断線では inf。"""
    # 細い Cu 配線（x 方向）＋ 下に Al 基準面、間は酸化膜
    cfg = WaferConfig(nx=60, ny=20, nz=30, pitch_um=0.1, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[5:8, :, :] = materials.get("metal_al").id          # 基準面
    g[15:18, 8:12, :] = materials.get("metal_cu").id      # 上の配線（x に伸びる）
    rc = M.rc_delay_ps(w, "metal_cu", "metal_al", "x")
    assert np.isfinite(rc) and rc > 0.0
    # 配線を途中で分断 → オープン
    g[15:18, 8:12, 30:33] = materials.get("oxide").id
    assert M.rc_delay_ps(w, "metal_cu", "metal_al", "x") == float("inf")
