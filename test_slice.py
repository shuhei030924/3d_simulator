"""visualize.slice_2d の検証。"""
from __future__ import annotations

import numpy as np

from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe
from semisim import materials, visualize


def main():
    cfg = WaferConfig(nx=30, ny=24, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.3))
    wafer = r.simulate()

    si = materials.BY_NAME["silicon"].id
    ox = materials.BY_NAME["oxide"].id

    # Y断面 (XZ面): 縦=Z, 横=X
    plane, w_um, h_um = visualize.slice_2d(wafer, "Y", cfg.ny // 2)
    assert plane.shape == (cfg.nz, cfg.nx), plane.shape
    assert abs(w_um - cfg.nx * cfg.pitch_um) < 1e-9
    assert abs(h_um - cfg.nz * cfg.pitch_um) < 1e-9
    # 底はシリコン、その上に酸化膜
    col = plane[:, cfg.nx // 2]
    assert col[0] == si
    assert ox in col, "酸化膜が断面に現れるはず"
    print("Y断面 OK shape=", plane.shape)

    # X断面 (YZ面): 縦=Z, 横=Y
    planex, wx, hx = visualize.slice_2d(wafer, "X", cfg.nx // 2)
    assert planex.shape == (cfg.nz, cfg.ny), planex.shape
    print("X断面 OK shape=", planex.shape)

    # Z断面 (XY面): 縦=Y, 横=X。基板内なら全面シリコン
    planez, wz, hz = visualize.slice_2d(wafer, "Z", 0)
    assert planez.shape == (cfg.ny, cfg.nx), planez.shape
    assert np.all(planez == si), "基板底は全面シリコン"
    print("Z断面 OK shape=", planez.shape)

    # レジスト除外オプション
    plane_nr, _, _ = visualize.slice_2d(wafer, "Y", cfg.ny // 2, include_resist=False)
    assert plane_nr.shape == plane.shape

    # カラーマップ生成
    cmap, norm = visualize.material_listed_cmap()
    assert cmap.N >= len(materials.BY_ID)

    print("すべて成功")


if __name__ == "__main__":
    main()
