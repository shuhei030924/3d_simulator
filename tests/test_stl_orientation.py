"""STL メッシュの向き（巻き順・法線・閉曲面）の回帰テスト。

voxel surface メッシュが STL 標準に準拠していること:
- 三角形の頂点巻き順（右手系）が外向き法線に一致する
- 符号付き体積が正（外向き）でボクセル体積に一致する
- 各無向エッジが正確に 2 三角形に共有される（watertight / 多様体）
"""
import numpy as np

from semisim import export, materials
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo, Strip
from semisim.recipe import Recipe


def _parse(s):
    tris, normals, cur, nrm = [], [], [], None
    for line in s.splitlines():
        t = line.strip().split()
        if not t:
            continue
        if t[0] == "facet":
            nrm = np.array([float(t[2]), float(t[3]), float(t[4])])
        elif t[0] == "vertex":
            cur.append([float(t[1]), float(t[2]), float(t[3])])
            if len(cur) == 3:
                tris.append(np.array(cur))
                normals.append(nrm)
                cur = []
    return tris, normals


def _signed_volume(tris):
    return sum(np.dot(a, np.cross(b, c)) / 6.0 for a, b, c in tris)


def _box_wafer():
    cfg = WaferConfig(nx=8, ny=6, nz=12, pitch_um=0.1, substrate_um=0.6)
    return Recipe(config=cfg).simulate()


def test_winding_matches_stored_normal():
    """全三角形で巻き順（右手系）法線が格納法線と一致する。"""
    s = export.stl_string(_box_wafer())
    tris, normals = _parse(s)
    assert tris
    for (a, b, c), n in zip(tris, normals):
        wn = np.cross(b - a, c - a)
        ln = np.linalg.norm(wn)
        if ln < 1e-12:
            continue
        wn = wn / ln
        assert np.dot(wn, n) > 0.99, "巻き順法線が格納法線と逆向き"


def test_signed_volume_positive_and_matches_voxels():
    """符号付き体積が正で、ボクセル体積に一致する（外向き向き）。"""
    w = _box_wafer()
    s = export.stl_string(w)
    tris, _ = _parse(s)
    p = w.config.pitch_um
    vox_vol = int((w.grid != materials.AIR).sum()) * p ** 3
    assert _signed_volume(tris) == np.float64(vox_vol) or abs(
        _signed_volume(tris) - vox_vol) < 1e-9
    assert _signed_volume(tris) > 0


def test_watertight_manifold():
    """各無向エッジが正確に 2 三角形に共有される（閉多様体）。"""
    from collections import defaultdict
    r = Recipe(config=WaferConfig(nx=12, ny=8, nz=20, pitch_um=0.1, substrate_um=0.6))
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Photo(mask=Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0,
                                                  "x1": 0.7, "y1": 1.0})]),
                thickness_um=0.8, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5))
    r.add(Strip())
    s = export.stl_string(r.simulate())
    tris, _ = _parse(s)
    edges = defaultdict(int)
    for a, b, c in tris:
        k = [tuple(np.round(v, 6)) for v in (a, b, c)]
        for u, v in ((k[0], k[1]), (k[1], k[2]), (k[2], k[0])):
            edges[tuple(sorted((u, v)))] += 1
    counts = set(edges.values())
    assert counts == {2}, f"非多様体エッジ: 共有数 {counts - {2}}"
