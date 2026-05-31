"""スナップショットキャッシュの正当性検証（増分計算 == 全計算）。"""
from __future__ import annotations

import numpy as np

from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, CMP, DryEtch, Photo, Strip, WetEtch, Oxidation
from semisim.recipe import Recipe


def fresh_grid(steps, config):
    r = Recipe(config=config)
    for s in steps:
        r.steps.append(s)
    # キャッシュを使わず毎回 0 から計算するため、毎回新規 Recipe で up_to 指定
    return r.simulate().grid


def main():
    cfg = WaferConfig(nx=40, ny=40, nz=40, pitch_um=0.1, substrate_um=1.5)
    r = Recipe(config=cfg)
    m = Mask(shapes=[Shape("grating", {"angle": 20.0, "period": 0.3, "width": 0.15})])
    steps = [
        CVD(material="oxide", thickness_um=0.3),
        Photo(mask=m, thickness_um=0.8),
        DryEtch(targets=["oxide"], depth_um=0.4),
        Strip(material="photoresist"),
        Oxidation(thickness_um=0.2),
        WetEtch(targets=["oxide"], depth_um=0.1),
        CMP(remove_um=0.2),
    ]
    for s in steps:
        r.add(s)

    # 1) プレビューで段階的に呼んでキャッシュを育てる
    for i in range(1, len(steps) + 1):
        g_cached = r.simulate(up_to=i).grid
        g_fresh = fresh_grid(steps[:i], cfg)
        assert np.array_equal(g_cached, g_fresh), f"段{i}でキャッシュ不一致"
    print("段階プレビューのキャッシュ一致 OK")

    # 2) 中間工程を編集 → 以降のキャッシュが破棄され正しく再計算されるか
    r.replace(2, DryEtch(targets=["oxide"], depth_um=0.6))
    new_steps = list(steps)
    new_steps[2] = DryEtch(targets=["oxide"], depth_um=0.6)
    g_cached = r.simulate().grid
    g_fresh = fresh_grid(new_steps, cfg)
    assert np.array_equal(g_cached, g_fresh), "編集後のキャッシュ不一致"
    print("編集後の再計算 OK")

    # 3) 並べ替え後
    r.move(5, -2)
    moved = list(new_steps)
    proc = moved.pop(5)
    moved.insert(3, proc)
    g_cached = r.simulate().grid
    g_fresh = fresh_grid(moved, cfg)
    assert np.array_equal(g_cached, g_fresh), "並べ替え後のキャッシュ不一致"
    print("並べ替え後の再計算 OK")

    # 4) 複製後
    nr = r.duplicate(0)
    assert nr == 1
    g_cached = r.simulate().grid
    dup = list(moved)
    dup.insert(1, dup[0])
    g_fresh = fresh_grid(dup, cfg)
    assert np.array_equal(g_cached, g_fresh), "複製後のキャッシュ不一致"
    print("複製後の再計算 OK")

    print("\nすべて成功")


if __name__ == "__main__":
    main()
