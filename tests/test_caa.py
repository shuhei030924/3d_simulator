"""クリティカルエリア解析（CAA）ショート歩留りのテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _wires(gap_vox: int) -> Wafer:
    """z 層に 2 本の並走配線（Cu/W）を gap_vox の間隔で配置。"""
    cfg = WaferConfig(nx=60, ny=60, nz=10, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[4:7, 10:16, :] = materials.get("metal_cu").id
    g[4:7, 16 + gap_vox:22 + gap_vox, :] = materials.get("tungsten").id
    return w


def test_critical_area_increases_with_defect_size():
    """臨界面積は欠陥径が大きいほど増え、間隔未満では 0。"""
    w = _wires(4)  # 間隔 0.2µm
    ac_small = M.critical_area_short_um2(w, "metal_cu", "tungsten", 0.1)
    ac_mid = M.critical_area_short_um2(w, "metal_cu", "tungsten", 0.3)
    ac_big = M.critical_area_short_um2(w, "metal_cu", "tungsten", 0.5)
    assert ac_small == 0.0       # 欠陥径 < 間隔 → ブリッジ不可
    assert 0.0 < ac_mid < ac_big


def test_yield_decreases_with_defect_density():
    """欠陥密度が高いほど歩留りは下がる。"""
    w = _wires(4)
    y_lo = M.caa_short_yield(w, "metal_cu", "tungsten",
                             defect_density_per_cm2=1e4, chip_area_cm2=0.1)["yield"]
    y_hi = M.caa_short_yield(w, "metal_cu", "tungsten",
                             defect_density_per_cm2=2e5, chip_area_cm2=0.1)["yield"]
    assert 0.0 < y_hi < y_lo < 1.0


def test_yield_higher_for_wider_spacing():
    """配線間隔が広いほどショート歩留りは高い。"""
    y_narrow = M.caa_short_yield(_wires(2), "metal_cu", "tungsten",
                                 defect_density_per_cm2=5e4, chip_area_cm2=0.1)["yield"]
    y_wide = M.caa_short_yield(_wires(8), "metal_cu", "tungsten",
                               defect_density_per_cm2=5e4, chip_area_cm2=0.1)["yield"]
    assert y_wide > y_narrow


def test_lambda_proportional_to_chip_area():
    """期待故障数 λ はチップ面積に比例する。"""
    w = _wires(4)
    r1 = M.caa_short_yield(w, "metal_cu", "tungsten",
                           defect_density_per_cm2=5e4, chip_area_cm2=0.1)
    r2 = M.caa_short_yield(w, "metal_cu", "tungsten",
                           defect_density_per_cm2=5e4, chip_area_cm2=0.2)
    assert r2["lambda_faults"] == pytest.approx(2 * r1["lambda_faults"], rel=1e-6)


def test_caa_invalid_params_raise():
    w = _wires(4)
    with pytest.raises(ValueError):
        M.caa_short_yield(w, "metal_cu", "tungsten", x0_um=0.0)
    with pytest.raises(ValueError):
        M.caa_short_yield(w, "metal_cu", "tungsten", xmax_um=0.01, x0_um=0.02)
