"""Fill アスペクト比依存キーホール空隙のテスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Fill, Photo, Strip
from semisim.recipe import Recipe


def _buried_air(wafer) -> int:
    """金属に上下を挟まれた埋没空気（=ボイド）のボクセル数。"""
    cu = materials.get("metal_cu").id
    g = wafer.grid
    cnt = 0
    for y in range(g.shape[1]):
        for x in range(g.shape[2]):
            col = g[:, y, x]
            air = np.where(col == materials.AIR)[0]
            cu_z = np.where(col == cu)[0]
            if cu_z.size and air.size:
                cnt += int(np.count_nonzero((air > cu_z.min()) & (air < cu_z.max())))
    return cnt


def _narrow_trench_recipe(void_ar):
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.45, "y0": 0.0, "x1": 0.55, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.5))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.5))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=void_ar))
    return r


def test_void_forms_in_narrow_trench():
    """高 AR の狭いトレンチでキーホール空隙が生じる。"""
    assert _buried_air(_narrow_trench_recipe(2.0).simulate()) > 0


def test_no_void_when_disabled():
    """void_ar=0 では空隙が生じない（完全充填）。"""
    assert _buried_air(_narrow_trench_recipe(0.0).simulate()) == 0


def test_no_void_in_wide_trench():
    """幅広（低 AR）トレンチでは閾値を超えず空隙が生じない。"""
    cfg = WaferConfig(nx=60, ny=20, nz=70, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.2, "y0": 0.0, "x1": 0.8, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=1.0))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=5.0))
    assert _buried_air(r.simulate()) == 0


def test_void_roundtrip():
    p = Fill(material="metal_cu", overfill_um=0.1, void_ar=8.0)
    d = p.params_dict()
    assert d["void_ar"] == 8.0
    assert Fill._from_params(d).void_ar == 8.0


def test_keyhole_is_teardrop_not_floating():
    """キーホール空隙は底の充填頂点から口元付近まで縦に伸びるティアドロップ。

    中空に浮いた短い線でなく、トレンチ高さの大半を占め、中段で最大幅を持つ
    （ハーフ幅 1 ボクセルの細線でない）ことを確認する。
    """
    cfg = WaferConfig(nx=80, ny=12, nz=80, pitch_um=0.05, substrate_um=1.0)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.4, "y0": 0.0, "x1": 0.6, "y1": 1.0})])
    depth = 1.6
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=depth))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=depth))
    r.add(Strip())
    r.add(Fill(material="metal_cu", overfill_um=0.2, void_ar=1.5))
    g = r.simulate().grid
    cu = materials.get("metal_cu").id
    y = cfg.ny // 2
    sl = g[:, y, :]
    pitch = cfg.pitch_um
    # 各 z での埋没空気（上下を Cu で挟まれた）の幅
    width_by_z = {}
    void_z = []
    for x in range(cfg.nx):
        col = sl[:, x]
        air = np.where(col == materials.AIR)[0]
        cu_z = np.where(col == cu)[0]
        if cu_z.size and air.size:
            buried = air[(air > cu_z.min()) & (air < cu_z.max())]
            for z in buried.tolist():
                width_by_z[z] = width_by_z.get(z, 0) + 1
                void_z.append(z)
    assert void_z, "キーホール空隙が検出されない"
    seam_lo, seam_hi = min(void_z) * pitch, max(void_z) * pitch
    trench_top = 1.0 + depth
    trench_bot = trench_top - depth
    # トレンチ高さの半分以上を占める連続空隙（浮いた短い線でない）
    assert (seam_hi - seam_lo) > 0.5 * depth
    # 上端は口元近くまで達する / 下端は底の充填頂点より上（底は埋まる）
    assert seam_hi > trench_top - 0.25
    assert seam_lo > trench_bot
    # ティアドロップ: 最大幅が複数ボクセル（ハーフ幅 1 の細線でない）
    assert max(width_by_z.values()) >= 4
    # 中段の幅 > 上端付近の幅（上ほど細るピンチオフ形状）
    zs = sorted(width_by_z)
    mid_w = width_by_z[zs[len(zs) // 2]]
    top_w = width_by_z[zs[-2]] if len(zs) >= 2 else width_by_z[zs[-1]]
    assert mid_w > top_w
    # 下端は底の充填頂点より上（底は埋まっている）
    assert seam_lo > trench_bot, "シームが底まで開いている"


def test_void_negative_raises():
    cfg = WaferConfig(nx=20, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Fill(material="metal_cu", overfill_um=0.1, void_ar=-1.0))
    try:
        r.simulate()
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("負の void_ar で ValueError が出るべき")
