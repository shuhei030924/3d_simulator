"""エンジンの簡易動作確認スクリプト（GUI 不要）。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.masks import Mask, Shape
from semisim.processes import CVD, PVD, DryEtch, WetEtch, Photo, Diffusion, Strip, CMP, Oxidation
from semisim.recipe import Recipe
from semisim.grid import WaferConfig


def count_by_material(wafer):
    out = {}
    for m in materials.all_materials():
        c = int((wafer.grid == m.id).sum())
        if c:
            out[m.name] = c
    return out


def main():
    cfg = WaferConfig(nx=60, ny=60, nz=60, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)

    # 1) 酸化膜をコンフォーマル成膜
    r.add(CVD(material="oxide", thickness_um=0.5))
    # 2) フォトリソ（中央に四角い開口）
    mask = Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0.35, "x1": 0.65, "y1": 0.65})])
    r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
    # 3) ドライエッチで酸化膜に穴
    r.add(DryEtch(targets=["oxide"], depth_um=0.6))
    # 4) レジスト剥離
    r.add(Strip(material="photoresist"))
    # 5) 拡散
    r.add(Diffusion(dopant="doped_n", depth_um=0.5))
    # 6) 金属を指向性成膜
    r.add(PVD(material="metal_al", thickness_um=0.4))
    # 7) ウェットエッチ
    r.add(WetEtch(targets=["metal_al"], depth_um=0.2))
    # 8) 熱酸化（露出シリコンがあれば酸化）
    r.add(Oxidation(thickness_um=0.2))
    # 9) 周期ラインでフォトリソ
    grating = Mask(shapes=[Shape("grating", {"angle": 0.0, "period": 0.25, "width": 0.12})])
    r.add(Photo(mask=grating, thickness_um=0.8, polarity="positive"))
    # 10) CMP 平坦化
    r.add(CMP(remove_um=0.3))

    for i in range(1, len(r.steps) + 1):
        w = r.simulate(up_to=i)
        print(f"[{i}] {r.steps[i-1].summary()}")
        print("    ", count_by_material(w))

    # 断面が空洞でないことの確認（中央 XZ 断面の固体率）
    w = r.simulate()
    mid_y = w.grid.shape[1] // 2
    sl = w.grid[:, mid_y, :]
    solid_ratio = float((sl != 0).mean())
    print(f"\n中央XZ断面の固体率: {solid_ratio:.2%}")

    # CMP 後に上面が平坦になっているか確認
    z_top = w.top_surface_z()
    flat = z_top[z_top >= 0]
    print(f"CMP後の上面 z 範囲: {int(flat.min())}..{int(flat.max())} (平坦なら差が小)")

    # 保存/読込の往復確認
    import tempfile, os
    path = os.path.join(tempfile.gettempdir(), "test_recipe.json")
    r.save(path)
    r2 = Recipe.load(path)
    w2 = r2.simulate()
    assert np.array_equal(w.grid, w2.grid), "保存/読込で結果が一致しません"
    print("保存/読込 往復OK:", path)
    print("\nすべて成功")


if __name__ == "__main__":
    main()
