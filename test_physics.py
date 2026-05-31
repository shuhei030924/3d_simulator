"""PVD段差被覆とDryEtchオーバーエッチの検証。"""
from __future__ import annotations

import numpy as np

from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, PVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe
from semisim import materials


def make_trench():
    """中央に深いトレンチを作ったウェハを返すレシピ。"""
    cfg = WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    m = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.4, "x1": 0.6, "y1": 0.6})])
    r.add(Photo(mask=m, thickness_um=1.2, polarity="positive"))  # 中央だけ開口
    r.add(DryEtch(targets=["oxide"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    return cfg, r


def test_pvd_shadowing():
    cfg, r = make_trench()
    # 被覆100%
    r.add(PVD(material="metal_al", thickness_um=0.3, step_coverage=1.0))
    w_full = r.simulate()
    al = materials.BY_NAME["metal_al"].id
    # トレンチは x,y in [0.4,0.6] → vox 16..23。壁は vox15/24。
    # 壁に隣接する底の列(16, cy)はシャドーイングで薄くなる。
    cy = cfg.ny // 2
    wall_col = 16
    near_full = int((w_full.grid[:, cy, wall_col] == al).sum())

    # 被覆20%で作り直し
    cfg, r2 = make_trench()
    r2.add(PVD(material="metal_al", thickness_um=0.3, step_coverage=0.2))
    w_sc = r2.simulate()
    near_sc = int((w_sc.grid[:, cy, wall_col] == al).sum())

    print(f"壁際の底のAl膜厚: 被覆100%={near_full}vox, 被覆20%={near_sc}vox")
    assert near_sc < near_full, "シャドーイングで壁際の底が薄くなるはず"
    # 平坦部(端)は被覆率の影響をほぼ受けない
    edge_full = int((w_full.grid[:, 1, 1] == al).sum())
    edge_sc = int((w_sc.grid[:, 1, 1] == al).sum())
    assert edge_full == edge_sc, f"平坦部は不変のはず {edge_full} vs {edge_sc}"
    print("PVDシャドーイング OK")


def test_overetch():
    cfg = WaferConfig(nx=30, ny=30, nz=50, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    # オーバーエッチ無し: oxide だけ削れて silicon は残る
    r.add(DryEtch(targets=["oxide"], depth_um=0.5, overetch_pct=0.0))
    w0 = r.simulate()
    si = materials.BY_NAME["silicon"].id
    si0 = int((w0.grid == si).sum())

    # オーバーエッチ50%: silicon も少し削れる
    r2 = Recipe(config=cfg)
    r2.add(CVD(material="oxide", thickness_um=0.5))
    r2.add(DryEtch(targets=["oxide"], depth_um=0.5, overetch_pct=50.0))
    w1 = r2.simulate()
    si1 = int((w1.grid == si).sum())

    print(f"silicon残量: OE0%={si0}, OE50%={si1}")
    assert si1 < si0, "オーバーエッチで下層siliconが削れるはず"
    print("オーバーエッチ OK")


def test_serialization():
    p = PVD(material="metal_al", thickness_um=0.3, step_coverage=0.4)
    p2 = PVD._from_params(p.to_dict())
    assert abs(p2.step_coverage - 0.4) < 1e-9
    d = DryEtch(targets=["oxide"], depth_um=0.5, overetch_pct=30.0)
    d2 = DryEtch._from_params(d.to_dict())
    assert abs(d2.overetch_pct - 30.0) < 1e-9
    print("シリアライズ往復 OK")


if __name__ == "__main__":
    test_pvd_shadowing()
    test_overetch()
    test_serialization()
    print("\nすべて成功")
