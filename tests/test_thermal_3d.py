"""完全 3D 熱拡散ソルバのテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _slab(mat: str, n: int = 14, pitch: float = 0.1) -> Wafer:
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get(mat).id
    return w


def test_3d_uniform_matches_1d_resistance():
    """全面トップ発熱の 3D 最大 ΔT は 1D 熱抵抗 P·R_th にほぼ一致。"""
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    p = 0.05
    assert M.peak_temperature_rise_3d(w, src, p) == pytest.approx(
        p * M.thermal_resistance_k_w(w), rel=0.1)


def test_3d_point_source_spreads_in_both_lateral_axes():
    """点状発熱は x・y 両方向に対称に拡散（真の 3D 拡散）。"""
    n = 14
    w = _slab("silicon", n)
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, n // 2, n // 2] = True
    t = M.temperature_field_3d(w, src, 0.05)
    center = t[-1, n // 2, n // 2]
    assert center > t[-1, 0, n // 2]      # y 方向に減衰
    assert center > t[-1, n // 2, 0]      # x 方向に減衰
    assert t[-1, 0, n // 2] == pytest.approx(t[-1, n // 2, 0], rel=0.05)  # 対称


def test_3d_linear_in_power():
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, 7, 7] = True
    t1 = M.peak_temperature_rise_3d(w, src, 0.05)
    t2 = M.peak_temperature_rise_3d(w, src, 0.10)
    assert t2 == pytest.approx(2 * t1, rel=1e-6)


def test_3d_bottom_is_heat_sink():
    """基板最下面はヒートシンク（ΔT=0）。"""
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    t = M.temperature_field_3d(w, src, 0.05)
    assert np.allclose(t[0], 0.0)


def test_3d_high_conductivity_lowers_peak():
    """熱伝導率が高い材料ほど点発熱ピーク温度が低い。"""
    n = 14
    src = np.zeros((n, n, n), dtype=bool)
    src[-1, n // 2, n // 2] = True
    t_si = M.peak_temperature_rise_3d(_slab("silicon", n), src, 0.05)
    t_ox = M.peak_temperature_rise_3d(_slab("oxide", n), src, 0.05)
    assert t_ox > t_si
