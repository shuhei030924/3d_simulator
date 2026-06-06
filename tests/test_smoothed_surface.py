"""スムーズ表示（visualize.smoothed_surface）の断面平坦性テスト。

pyvista 依存のため未導入環境ではスキップ。クリップした断面（切断面）が
平滑化後も平坦に保たれること、平滑化自体は外形を丸めることを検証する。
"""
import numpy as np
import pytest

pytest.importorskip("pyvista")

from semisim import processes as P  # noqa: E402
from semisim import visualize as V  # noqa: E402
from semisim.grid import WaferConfig  # noqa: E402
from semisim.masks import Mask, Shape  # noqa: E402
from semisim.recipe import Recipe  # noqa: E402


def _mesh():
    cfg = WaferConfig(nx=40, ny=24, nz=40, pitch_um=0.05, substrate_um=0.5)
    r = Recipe(config=cfg)
    r.add(P.CVD(material="oxide", thickness_um=0.3))
    r.add(P.CVD(material="poly", thickness_um=0.3))
    r.add(P.Photo(mask=Mask(shapes=[Shape("rect", {"x0": 0.35, "y0": 0,
                                                    "x1": 0.65, "y1": 1})]),
                  thickness_um=0.6, polarity="negative"))
    r.add(P.DryEtch(targets=["poly"], depth_um=0.3))
    r.add(P.Strip())
    r.add(P.CVD(material="metal_al", thickness_um=0.1))
    return V.solid_unstructured(r.simulate())


def test_cut_plane_stays_flat_with_plane():
    """plane を渡すと、クリップ断面の切断面が平滑化後も平坦（厚み方向 std≈0）。"""
    mesh = _mesh()
    origin = np.array([0.0, 0.6, 0.0])
    normal = np.array([0.0, 1.0, 0.0])
    clip = mesh.clip(normal=tuple(normal), origin=tuple(origin))
    out = V.smoothed_surface(clip, plane=(origin, normal), plane_tol=0.025)
    # 切断面 y=0.6 上の頂点の y ばらつきがほぼ 0
    on = np.abs(out.points[:, 1] - 0.6) < 0.025
    assert on.sum() > 0
    assert out.points[on][:, 1].std() < 1e-6


def test_without_plane_cut_face_ripples():
    """plane を渡さないと（従来挙動）切断面は波打つ（回帰の対照）。"""
    mesh = _mesh()
    origin = (0.0, 0.6, 0.0)
    clip = mesh.clip(normal=(0, 1, 0), origin=origin)
    out = V.smoothed_surface(clip, plane=None)
    on = np.abs(out.points[:, 1] - 0.6) < 0.05
    assert out.points[on][:, 1].std() > 1e-4  # 平坦でない


def test_smoothing_rounds_outer_shape():
    """平滑化で外形は丸まる（頂点が動く）。"""
    mesh = _mesh()
    surf = mesh.extract_surface()
    out = V.smoothed_surface(mesh)
    moved = np.linalg.norm(out.points - surf.points, axis=1)
    assert moved.mean() > 1e-4


def test_smoothed_surface_no_plane_runs():
    """plane なしでも例外なくサーフェスを返す。"""
    out = V.smoothed_surface(_mesh())
    assert out.n_points > 0
