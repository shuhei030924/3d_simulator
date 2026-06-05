"""slice_2d の軸指定の堅牢性テスト（pyvista 非依存）。

slice_2d は pyvista 無しでも動くため importorskip しない。無効な軸を黙って
Z 断面にせずエラーにすること、大文字小文字を問わないことを検証する。
"""
import numpy as np
import pytest

from semisim import processes as P
from semisim import visualize
from semisim.grid import Wafer, WaferConfig


def _wafer():
    w = Wafer(WaferConfig(nx=30, ny=20, nz=40, pitch_um=0.1, substrate_um=2.0))
    P.CVD(material="oxide", thickness_um=0.3).apply(w)
    return w


def test_axis_case_insensitive():
    """小文字の軸も同じ断面を返す。"""
    w = _wafer()
    for up, lo in (("X", "x"), ("Y", "y"), ("Z", "z")):
        a, _, _ = visualize.slice_2d(w, up, 10)
        b, _, _ = visualize.slice_2d(w, lo, 10)
        assert np.array_equal(a, b)


@pytest.mark.parametrize("axis", ["Q", "", "yz", "z ", "1", "XY"])
def test_invalid_axis_raises(axis):
    """無効な軸は黙って Z 断面にせず ValueError を投げる。"""
    w = _wafer()
    with pytest.raises(ValueError):
        visualize.slice_2d(w, axis, 0)


def test_valid_axes_shapes():
    """X/Y/Z それぞれ正しい断面形状を返す。"""
    w = _wafer()
    nz, ny, nx = w.grid.shape
    assert visualize.slice_2d(w, "X", 5)[0].shape == (nz, ny)
    assert visualize.slice_2d(w, "Y", 5)[0].shape == (nz, nx)
    assert visualize.slice_2d(w, "Z", 5)[0].shape == (ny, nx)


def test_out_of_range_index_clipped():
    """範囲外 index はクリップされ例外を出さない。"""
    w = _wafer()
    nz, ny, nx = w.grid.shape
    a, _, _ = visualize.slice_2d(w, "Z", -100)
    b, _, _ = visualize.slice_2d(w, "Z", 0)
    assert np.array_equal(a, b)
    c, _, _ = visualize.slice_2d(w, "Z", 10**6)
    d, _, _ = visualize.slice_2d(w, "Z", nz - 1)
    assert np.array_equal(c, d)
