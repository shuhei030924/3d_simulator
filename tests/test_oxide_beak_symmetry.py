"""LOCOS バーズビークの対称性／退化入力ガードの回帰テスト。

露出 Si が無いとき（表面が全て他材料）、~eligible が全 True になり距離変換が
退化して端部に非対称な偽の酸化を生む不具合の回帰テスト。また露出 Si がある
通常の LOCOS では左右対称にバーズビークが侵入することを確認する。
"""

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Oxidation, Photo, Strip
from semisim.recipe import Recipe


def _cfg():
    # nx 奇数で中心対称
    return WaferConfig(nx=121, ny=21, nz=90, pitch_um=0.05, substrate_um=2.0)


def _cmask(w):
    lo = (1 - w) / 2
    return Mask(shapes=[Shape("rect", {"x0": lo, "y0": 0.0, "x1": 1 - lo, "y1": 1.0})])


def _asym(grid):
    return int((grid != grid[:, :, ::-1]).sum())


def test_beak_degenerate_no_exposed_si_is_symmetric():
    """露出 Si が無い対称入力では、バーズビークが偽の非対称酸化を生まない。"""
    r = Recipe(config=_cfg())
    r.add(CVD(material="oxide", thickness_um=0.1))
    r.add(CVD(material="poly", thickness_um=0.5))
    r.add(Photo(mask=_cmask(0.3), thickness_um=0.8, polarity="negative"))
    r.add(DryEtch(targets=["poly"], depth_um=0.5))
    r.add(Strip())
    r.add(Oxidation(thickness_um=0.2, beak_fraction=0.5))
    assert _asym(r.simulate().grid) == 0


def test_locos_beak_symmetric_and_grows():
    """通常の LOCOS（窒化膜マスク＋露出 Si）では左右対称にビークが侵入する。"""
    r = Recipe(config=_cfg())
    r.add(CVD(material="nitride", thickness_um=0.2))
    r.add(Photo(mask=_cmask(0.3), thickness_um=0.8, polarity="negative"))
    r.add(DryEtch(targets=["nitride"], depth_um=0.2))
    r.add(Strip())
    r.add(Oxidation(thickness_um=0.3, beak_fraction=0.6))
    g = r.simulate().grid
    assert _asym(g) == 0
    assert int((g == materials.get("oxide").id).sum()) > 0
