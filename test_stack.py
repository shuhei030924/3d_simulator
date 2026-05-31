"""Wafer.column_stack の検証。"""
from __future__ import annotations

from semisim.grid import WaferConfig
from semisim.processes import CVD, Photo
from semisim.masks import Mask, Shape
from semisim.recipe import Recipe
from semisim import materials


def main():
    cfg = WaferConfig(nx=20, ny=20, nz=60, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.3))
    r.add(CVD(material="nitride", thickness_um=0.2))
    wafer = r.simulate()

    cx, cy = cfg.nx // 2, cfg.ny // 2
    stack = wafer.column_stack(cx, cy)  # 下→上
    names = [materials.BY_ID[mid].name for mid, _ in stack]
    print("層構成(下→上):", [(materials.BY_ID[mid].name, round(t, 2)) for mid, t in stack])

    assert names == ["silicon", "oxide", "nitride"], names
    # 基板1.0 + oxide0.3 + nitride0.2
    si_t = dict((materials.BY_ID[mid].name, t) for mid, t in stack)
    assert abs(si_t["silicon"] - 1.0) < 1e-6, si_t
    assert abs(si_t["oxide"] - 0.3) < 1e-6, si_t
    assert abs(si_t["nitride"] - 0.2) < 1e-6, si_t

    # 空気しかない列（穴）でも安全に空リスト
    empty = wafer.column_stack(0, 0)
    assert isinstance(empty, list)

    print("column_stack OK")
    print("すべて成功")


if __name__ == "__main__":
    main()
