"""電流密度プロファイル・TLM 接触抵抗抽出のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _necked_wire() -> Wafer:
    cfg = WaferConfig(nx=80, ny=30, nz=15, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[6:12, 10:20, :] = materials.get("metal_cu").id          # 幅10
    g[6:12, 10:13, 30:50] = materials.get("oxide").id          # くびれ
    g[6:12, 17:20, 30:50] = materials.get("oxide").id          # → 幅4
    return w


def test_current_density_profile_peaks_at_neck():
    """電流密度プロファイルはくびれ（最小断面）でピークになる。"""
    p = M.current_density_profile(_necked_wire(), "metal_cu", 2.0, "x")
    assert p["j_a_cm2"].size == 80
    j = p["j_a_cm2"]
    assert j.max() > 2 * j[0]                       # くびれで倍以上
    peak_pos = p["position_um"][int(j.argmax())]
    assert 1.4 < peak_pos < 2.6                      # くびれ位置(30-50vox)


def test_current_density_profile_matches_stats():
    """プロファイルの最大 J は current_density_stats の j_max と一致。"""
    w = _necked_wire()
    p = M.current_density_profile(w, "metal_cu", 2.0, "x")
    st = M.current_density_stats(w, "metal_cu", 2.0, "x")
    assert p["j_a_cm2"].max() == pytest.approx(st["j_max_a_cm2"], rel=1e-9)


def test_current_density_profile_empty_for_nonconductor():
    w = _necked_wire()
    assert M.current_density_profile(w, "oxide", 1.0, "x")["j_a_cm2"].size == 0


def test_tlm_recovers_known_parameters():
    """合成 TLM データから既知の Rsheet・Rc・Lt を抽出できる。"""
    width, rsheet, rc = 10.0, 50.0, 20.0
    spac = np.array([2, 5, 10, 20, 40.0])
    res = 2 * rc + (rsheet / width) * spac
    ext = M.tlm_extract(spac, res, width)
    assert ext["sheet_resistance_ohm_sq"] == pytest.approx(rsheet, rel=1e-6)
    assert ext["contact_resistance_ohm"] == pytest.approx(rc, rel=1e-6)
    assert ext["transfer_length_um"] == pytest.approx(rc * width / rsheet, rel=1e-6)


def test_tlm_validates_input():
    with pytest.raises(ValueError):
        M.tlm_extract([1.0], [10.0], 5.0)
    with pytest.raises(ValueError):
        M.tlm_extract([1.0, 2.0], [10.0, 20.0], 0.0)
