"""角度指定マスクと角度断面の検証（GUI なし、オフスクリーン）。"""
from __future__ import annotations

import os

import numpy as np
import pyvista as pv

pv.OFF_SCREEN = True

from semisim import visualize
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def test_mask_angles():
    # 45度回転矩形
    s = Shape("rect", {"x0": 0.3, "y0": 0.45, "x1": 0.7, "y1": 0.55, "angle": 45.0})
    arr = s.rasterize(80, 80)
    assert arr.any(), "回転矩形が空"
    # 角度0の帯と90度の帯は転置に近い関係
    h = Shape("stripe", {"cx": 0.5, "cy": 0.5, "angle": 0.0, "width": 0.2}).rasterize(80, 80)
    v = Shape("stripe", {"cx": 0.5, "cy": 0.5, "angle": 90.0, "width": 0.2}).rasterize(80, 80)
    assert h.any() and v.any()
    # 0度の帯は行方向に一様（各行が全True/全False）に近い
    print("回転矩形 True数:", int(arr.sum()))
    print("帯0度 True数:", int(h.sum()), "帯90度 True数:", int(v.sum()))


def test_angle_clip():
    cfg = WaferConfig(nx=60, ny=60, nz=60, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.4))
    m = Mask(shapes=[Shape("stripe", {"cx": 0.5, "cy": 0.5, "angle": 30.0, "width": 0.25})])
    r.add(Photo(mask=m, thickness_um=1.0))
    r.add(DryEtch(targets=["oxide"], depth_um=0.6))
    r.add(Strip(material="photoresist"))
    wafer = r.simulate()
    mesh = visualize.solid_unstructured(wafer, include_resist=True)
    assert mesh.n_cells > 0

    # 斜め法線でクリップ
    b = mesh.bounds
    cx = 0.5 * (b[0] + b[1]); cy = 0.5 * (b[2] + b[3]); cz = 0.5 * (b[4] + b[5])
    az = np.deg2rad(30); el = np.deg2rad(20)
    normal = (np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el))
    clipped = mesh.clip(normal=normal, origin=(cx, cy, cz))
    print("斜め断面セル数:", clipped.n_cells)
    assert clipped.n_cells > 0

    cmap, clim = visualize.material_colormap()
    p = pv.Plotter(off_screen=True)
    p.add_mesh(clipped, scalars="material", cmap=cmap, clim=clim, show_scalar_bar=False)
    p.view_isometric()
    out = os.path.join(os.path.dirname(__file__), "preview_angle.png")
    p.show(screenshot=out)
    p.close()
    print("保存:", out, os.path.exists(out))


def test_serialize():
    m = Mask(shapes=[
        Shape("rect", {"x0": 0.2, "y0": 0.2, "x1": 0.6, "y1": 0.4, "angle": 30.0}),
        Shape("stripe", {"cx": 0.5, "cy": 0.5, "angle": 60.0, "width": 0.15}),
    ])
    r = Recipe()
    r.add(Photo(mask=m, thickness_um=1.0))
    import tempfile
    path = os.path.join(tempfile.gettempdir(), "angle_recipe.json")
    r.save(path)
    r2 = Recipe.load(path)
    a = r.simulate().grid
    bb = r2.simulate().grid
    assert np.array_equal(a, bb)
    print("角度パラメータの保存/読込 往復OK")


if __name__ == "__main__":
    test_mask_angles()
    test_angle_clip()
    test_serialize()
    print("\nすべて成功")
