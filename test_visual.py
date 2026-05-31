"""可視化パイプラインのオフスクリーン検証（GUI なし）。"""
from __future__ import annotations

import os

import pyvista as pv

from semisim import visualize
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, PVD, DryEtch, Photo, Strip, Diffusion
from semisim.recipe import Recipe

pv.OFF_SCREEN = True


def main():
    cfg = WaferConfig(nx=60, ny=60, nz=60, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.4))
    m = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.3, "x1": 0.7, "y1": 0.7})])
    r.add(Photo(mask=m, thickness_um=1.0))
    r.add(DryEtch(targets=["oxide"], depth_um=0.6))
    r.add(Strip(material="photoresist"))
    r.add(Diffusion(dopant="doped_n", depth_um=0.5))
    r.add(PVD(material="metal_al", thickness_um=0.4))

    wafer = r.simulate()
    mesh = visualize.solid_unstructured(wafer, include_resist=True)
    print("固体セル数:", mesh.n_cells)
    assert mesh.n_cells > 0

    # 垂直断面（Y法線）で層構造を確認
    b = mesh.bounds
    cy = b[2] + 0.5 * (b[3] - b[2])
    clipped = mesh.clip(normal=(0, 1, 0), origin=(0, cy, 0))
    print("Yクリップ後セル数:", clipped.n_cells)
    assert clipped.n_cells > 0

    cmap, clim = visualize.material_colormap()
    p = pv.Plotter(off_screen=True)
    p.add_mesh(clipped, scalars="material", cmap=cmap, clim=clim, show_scalar_bar=False)
    p.camera_position = "xz"
    out = os.path.join(os.path.dirname(__file__), "preview_offscreen.png")
    p.show(screenshot=out)
    p.close()
    print("スクリーンショット保存:", out, "存在:", os.path.exists(out))
    print("OK")


if __name__ == "__main__":
    main()
