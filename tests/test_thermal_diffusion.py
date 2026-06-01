"""2.5D 熱拡散ソルバ（温度分布・ホットスポット）のテスト。"""
import numpy as np
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _slab(mat: str, n: int = 24, pitch: float = 0.1) -> Wafer:
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:] = materials.get(mat).id
    return w


def test_uniform_matches_1d_resistance():
    """全面トップ発熱の最大 ΔT は 1D 熱抵抗の P·R_th にほぼ一致。"""
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    p = 0.1
    expected = p * M.thermal_resistance_k_w(w)
    tmax = M.peak_temperature_rise_k(w, src, p)
    assert tmax == pytest.approx(expected, rel=0.1)


def test_localized_source_spreads_laterally():
    """局所発熱は中心が高く端は低い（横方向ヒートスプレッディング）。"""
    w = _slab("silicon")
    n = w.config.nx
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1] = True
    t = M.temperature_field_2d(w, src, 0.1)
    assert t[-1, n // 2] > t[-1, 0]  # 中心 > 端
    assert (t[0, :] == 0.0).all()    # 底はヒートシンク ΔT=0


def test_temperature_linear_in_power():
    """温度上昇は電力に線形。"""
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    t1 = M.peak_temperature_rise_k(w, src, 0.1)
    t2 = M.peak_temperature_rise_k(w, src, 0.2)
    assert t2 == pytest.approx(2 * t1, rel=1e-6)


def test_high_conductivity_lowers_peak():
    """熱伝導率が高い材料ほど局所発熱のピーク温度が低い（よく拡散）。"""
    n = 24
    src = np.zeros((n, n, n), dtype=bool)
    src[-1, n // 2 - 1:n // 2 + 1, n // 2 - 1:n // 2 + 1] = True
    t_si = M.peak_temperature_rise_k(_slab("silicon", n), src, 0.1)   # k=150
    t_ox = M.peak_temperature_rise_k(_slab("oxide", n), src, 0.1)     # k=1.4
    assert t_ox > t_si  # 低熱伝導の酸化膜の方がはるかに高温


def test_zero_power_zero_rise():
    """発熱 0 なら温度上昇 0。"""
    w = _slab("silicon")
    src = np.zeros(w.grid.shape, dtype=bool)
    src[-1, :, :] = True
    assert M.peak_temperature_rise_k(w, src, 0.0) == 0.0
