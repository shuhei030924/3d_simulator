"""商用級モジュールプリセット（STI/FinFET/W プラグ）の構造テスト。"""
from __future__ import annotations

import numpy as np

from semisim import materials, presets


def _id(name: str) -> int:
    return materials.get(name).id


def test_sti_isolation_structure():
    """STI: 酸化膜が元の Si 表面より下（トレンチ内）に埋まり、窒化膜は残らない。"""
    r = presets.build("STI 素子分離")
    w = r.simulate()
    sub_top = w.um_to_vox(w.config.substrate_um) - 1
    oxide = w.grid == _id("oxide")
    # トレンチ充填: 元の基板上面より下に酸化膜が存在する（素子分離）
    assert oxide[: sub_top - 2].any()
    # CMP ストップに使った窒化膜は最後に全て除去されている
    assert not (w.grid == _id("nitride")).any()
    # アクティブ領域（トレンチ外）のシリコンはほぼ保たれている
    # （パッド酸化が表面を 1 ボクセル程度消費するのは物理的に正しい）
    si_cols = (w.grid == _id("silicon")).sum(axis=0)
    assert int(si_cols.max()) >= sub_top + 1 - w.um_to_vox(0.1)


def test_finfet_fin_structure():
    """FinFET: フィンが STI 酸化膜上に突出し、high-k がフィンとゲートを隔てる。"""
    r = presets.build("FinFET フィン形成")
    w = r.simulate()
    si = w.grid == _id("silicon")
    nz, ny, nx = w.grid.shape
    # フィン: シリコン上面の高低差 = フィン高さ（リセス量以上）
    si_top = np.where(si.any(axis=0), nz - 1 - np.argmax(si[::-1], axis=0), -1)
    fin_h = int(si_top.max() - si_top.min())
    assert fin_h >= w.um_to_vox(0.3)
    # high-k とメタルゲートが存在する
    hk, tin = w.grid == _id("hafnia"), w.grid == _id("tin")
    assert hk.any() and tin.any()
    # フィン中心列は上から TiN → HfO2 → Si の順（ゲート積層がフィンを被覆）
    fin_x = int(np.unravel_index(si_top.argmax(), si_top.shape)[1])
    col = w.grid[:, ny // 2, fin_x]
    zt = int(np.nonzero(col != materials.AIR)[0].max())
    stack = [int(col[z]) for z in range(zt, zt - 6, -1)]
    assert _id("tin") in stack and _id("hafnia") in stack
    assert stack.index(_id("tin")) < stack.index(_id("hafnia"))
    # TiN がシリコンに直接接しない（high-k が必ず介在 = ゲートリーク防止）
    tin_dilated = np.zeros_like(tin)
    tin_dilated[1:] |= tin[:-1]
    tin_dilated[:-1] |= tin[1:]
    assert not (tin_dilated & si).any()


def test_w_contact_plug_structure():
    """W プラグ: TiN ライナー付きで ILD を貫通し拡散層に着底する。"""
    r = presets.build("W コンタクトプラグ")
    w = r.simulate()
    wt = w.grid == _id("tungsten")
    tin = w.grid == _id("tin")
    assert wt.any() and tin.any()
    nz, ny, nx = w.grid.shape
    cx, cy = nx // 2, ny // 2
    col = w.grid[:, cy, cx]
    # 中心列: プラグ（W）が存在し、その直下はバリア（TiN）→拡散層（doped_n）
    zw = np.nonzero(col == _id("tungsten"))[0]
    assert zw.size > 0
    below = int(col[zw.min() - 1])
    assert below == _id("tin")
    z = zw.min() - 1
    while int(col[z]) == _id("tin"):
        z -= 1
    assert int(col[z]) == _id("doped_n")
    # プラグ上面は ILD 酸化膜上面と面一（CMP 分離済 = 表面はほぼ平坦）
    z_top = w.top_surface_z()
    assert int(z_top.max()) - int(z_top.min()) <= 1
