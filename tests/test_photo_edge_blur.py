"""Photo エッジ丸め（光学解像度有限による角の丸め）のテスト。"""
from __future__ import annotations

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import Photo
from semisim.recipe import Recipe


def _square_recipe(edge_blur_sigma_um):
    cfg = WaferConfig(nx=60, ny=60, nz=30, pitch_um=0.1, substrate_um=0.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    r = Recipe(config=cfg)
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive",
                edge_blur_sigma_um=edge_blur_sigma_um))
    return r


def _resist_cols(wafer):
    res = materials.get("photoresist").id
    return int((wafer.grid[6] == res).sum())


def test_blur_rounds_corner():
    """角丸めで開口の角にレジストが残る（鋭角が削れて開口が縮む）。"""
    sharp = _square_recipe(0.0).simulate()
    rounded = _square_recipe(0.5).simulate()
    res = materials.get("photoresist").id
    # 開口の角(18,18)は blur 無しでは除去、blur 有りでは残る
    assert sharp.grid[6, 18, 18] != res
    assert rounded.grid[6, 18, 18] == res
    # 角が丸まる分、残存レジスト列が増える
    assert _resist_cols(rounded) > _resist_cols(sharp)


def test_no_blur_keeps_sharp():
    """edge_blur=0 では従来どおり鋭い矩形開口。"""
    w = _square_recipe(0.0).simulate()
    res = materials.get("photoresist").id
    # 開口内部は完全にレジスト除去
    assert w.grid[6, 30, 30] != res


def test_roundtrip():
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    p = Photo(mask=mask, thickness_um=0.8, polarity="positive",
              edge_blur_sigma_um=0.3)
    d = p.params_dict()
    q = Photo._from_params(d)
    assert q.edge_blur_sigma_um == 0.3


def test_negative_blur_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=30, pitch_um=0.1, substrate_um=0.5)
    r = Recipe(config=cfg)
    r.add(Photo(mask=Mask(), thickness_um=0.5, edge_blur_sigma_um=-0.2))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の edge_blur で ValueError が出るべき")
