"""ベクトル化カラムエッチ（_etch_columns_topdown）の逐次参照実装との一致テスト。

DryEtch の主エッチは旧来「1 反復 = 各列の最上面 1 ボクセル」の逐次ループ
だった。新実装は z 方向累積和の閉形式（O(N)）であり、ここでは旧逐次
アルゴリズムをそのまま参照実装として残し、ランダム構造×パラメータ行列で
結果が一致することを恒久的に検証する。
"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import materials
from semisim.grid import Wafer, WaferConfig
from semisim.processes import DryEtch, _resolve_targets, _top_material


def _reference_dry_etch_main(wafer, proc: DryEtch) -> None:
    """旧実装の主エッチ部（逐次・最上面ループ）。選択比/eligible/深さ上限込み。

    テーパ/ARDE は depth_cap 経由で同じ式になるため、ここでは
    depth_cap=depth（既定）のケースを参照する。
    """
    grid = wafer.grid
    ny, nx = grid.shape[1:]
    target_ids = set(_resolve_targets(proc.targets))
    depth = wafer.um_to_vox(proc.depth_um)
    _, top_id = _top_material(wafer)
    eligible = np.isin(top_id, list(target_ids))
    depth_cap = np.full((ny, nx), float(depth))
    etched = np.zeros((ny, nx), dtype=float)
    budget = np.where(eligible, float(depth), 0.0)
    rate_lut = np.ones(max(materials.BY_ID) + 1)
    for name, rv in proc.selectivity.items():
        rate_lut[materials.get(name).id] = rv
    for _ in range(depth):
        z_top, top_id = _top_material(wafer)
        top_is_target = np.isin(top_id, list(target_ids))
        rate = rate_lut[top_id]
        cost = np.where(rate > 0, 1.0 / np.where(rate > 0, rate, 1.0), np.inf)
        do = (
            eligible & top_is_target & (z_top >= 0)
            & (budget >= cost) & (etched < depth_cap)
        )
        if not do.any():
            break
        ys, xs = np.nonzero(do)
        grid[z_top[ys, xs], ys, xs] = materials.AIR
        budget[do] -= cost[do]
        etched[do] += 1.0


def _random_wafer(seed: int) -> Wafer:
    """ランダムな多材料積層＋空隙を持つ構造ウェハ（air 上空き大）。"""
    rng = np.random.default_rng(seed)
    cfg = WaferConfig(nx=24, ny=20, nz=40, pitch_um=0.1, substrate_um=1.0)
    w = Wafer(cfg)
    ids = [materials.get(n).id for n in
           ("silicon", "oxide", "nitride", "poly", "photoresist", "metal_al")]
    # 列ごとにランダム高さまでランダム材料の層を積む（時々空隙を混ぜる）
    heights = rng.integers(5, 28, size=(cfg.ny, cfg.nx))
    for y in range(cfg.ny):
        for x in range(cfg.nx):
            z = 10
            while z < heights[y, x]:
                run = int(rng.integers(1, 5))
                mat = int(rng.choice(ids)) if rng.random() > 0.15 else materials.AIR
                w.grid[z : min(z + run, cfg.nz), y, x] = mat
                z += run
    return w


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
@pytest.mark.parametrize(
    "kw",
    [
        dict(targets=["oxide"], depth_um=0.8),
        dict(targets=["oxide", "nitride", "poly"], depth_um=1.5),
        dict(targets=["oxide", "poly"], depth_um=1.2,
             selectivity={"oxide": 0.33, "poly": 0.5}),
        dict(targets=[], depth_um=0.6),
        dict(targets=["oxide", "nitride"], depth_um=2.0,
             selectivity={"nitride": 0.0}),
    ],
)
def test_vectorized_matches_sequential_reference(seed, kw):
    """ランダム構造×パラメータで新旧の主エッチ結果が完全一致する。"""
    w_new = _random_wafer(seed)
    w_ref = _random_wafer(seed)
    proc = DryEtch(**kw)
    proc.apply(w_new)
    _reference_dry_etch_main(w_ref, proc)
    assert np.array_equal(w_new.grid, w_ref.grid)
