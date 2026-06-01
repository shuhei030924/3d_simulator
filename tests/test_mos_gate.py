"""MOS ゲート容量・EOT メトロロジのテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _gate(diel: str, t_vox: int, pitch: float = 0.001) -> Wafer:
    """Si チャネル / 誘電体 t_vox / 金属ゲートの積層（pitch 既定 1nm）。"""
    cfg = WaferConfig(nx=30, ny=30, nz=30, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:10 + t_vox, :, :] = materials.get(diel).id
    g[10 + t_vox:14 + t_vox, :, :] = materials.get("metal_al").id
    return w


def test_sio2_eot_equals_physical_thickness():
    """純 SiO2 ゲートでは EOT = 物理膜厚。"""
    r = M.mos_gate_capacitance(_gate("oxide", 2), "metal_al")
    assert r["eot_nm"] == pytest.approx(2.0, rel=1e-3)
    # Cox = εox/t = 3.9·8.854e-12/2e-9 = 17.27 fF/µm²
    assert r["cox_ff_per_um2"] == pytest.approx(17.27, rel=0.01)


def test_high_k_reduces_eot():
    """high-k（HfO2）は同じ物理膜厚でも EOT が薄い＝Cox が高い。"""
    r_ox = M.mos_gate_capacitance(_gate("oxide", 4), "metal_al")
    r_hf = M.mos_gate_capacitance(_gate("hafnia", 4), "metal_al")
    assert r_hf["eot_nm"] < r_ox["eot_nm"]
    assert r_hf["cox_ff_per_um2"] > r_ox["cox_ff_per_um2"]
    # HfO2 4nm: EOT = 3.9·4/25 = 0.624 nm
    assert r_hf["eot_nm"] == pytest.approx(0.624, rel=0.01)


def test_series_stack_eot_additive():
    """積層ゲート（SiO2+HfO2）の EOT は各層の電気的厚みの和。"""
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:11, :, :] = materials.get("oxide").id     # 1nm SiO2
    g[11:14, :, :] = materials.get("hafnia").id     # 3nm HfO2
    g[14:18, :, :] = materials.get("metal_al").id
    r = M.mos_gate_capacitance(w, "metal_al")
    expect = 3.9 * (1e-9 / 3.9 + 3e-9 / 25.0) * 1e9
    assert r["eot_nm"] == pytest.approx(expect, rel=1e-3)


def test_no_dielectric_returns_zero():
    """ゲート電極がチャネルに直接接触（誘電体無し）なら 0。"""
    cfg = WaferConfig(nx=20, ny=20, nz=20, pitch_um=0.001, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:14, :, :] = materials.get("metal_al").id  # 誘電体無しで直接接触
    r = M.mos_gate_capacitance(w, "metal_al")
    assert r["total_cap_ff"] == 0.0 and r["eot_nm"] == 0.0


def test_total_cap_scales_with_area():
    """総容量はゲート面積に比例する。"""
    r = M.mos_gate_capacitance(_gate("oxide", 2), "metal_al")
    assert r["total_cap_ff"] == pytest.approx(
        r["cox_ff_per_um2"] * r["gate_area_um2"], rel=1e-6)
    assert r["gate_area_um2"] > 0
