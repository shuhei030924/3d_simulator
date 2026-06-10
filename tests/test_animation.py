"""工程アニメーション（semisim/animation.py）と CLI --gif のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

from semisim import animation, materials
from semisim.cli import main as cli_main
from semisim.grid import WaferConfig
from semisim.masks import Mask, Shape
from semisim.processes import CVD, DryEtch, Photo
from semisim.recipe import Recipe


def _recipe() -> Recipe:
    cfg = WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=1.5)
    mask = Mask(shapes=[Shape("rect", {"x0": 0.3, "y0": 0.0, "x1": 0.7, "y1": 1.0})])
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.5))
    r.add(Photo(mask=mask, thickness_um=0.5, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5))
    return r


def test_step_slices_progression():
    """先頭は初期状態、以降は 1 工程ずつ進み、断面が変化していく。"""
    r = _recipe()
    slices = animation.step_slices(r)
    assert len(slices) == len(r.steps) + 1
    assert slices[0][0] == "初期状態"
    assert "工程 1/3" in slices[1][0]
    assert "工程 3/3" in slices[3][0]
    # 初期状態は基板のみ、CVD 後は酸化膜が現れる
    oxide = materials.get("oxide").id
    assert oxide not in slices[0][1]
    assert oxide in slices[1][1]
    # 各工程で断面が実際に変化する
    for a, b in zip(slices, slices[1:]):
        assert not np.array_equal(a[1], b[1])


def test_render_frame_consistent_shape():
    """全フレームが同寸の RGB 配列になる（GIF 結合の前提）。"""
    r = _recipe()
    shapes = set()
    for title, plane, w, h in animation.step_slices(r):
        rgb = animation.render_frame(title, plane, w, h)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3
        shapes.add(rgb.shape)
    assert len(shapes) == 1


def test_save_gif(tmp_path):
    """GIF が工程数+1 フレームのアニメーションとして書き出される。"""
    from PIL import Image

    r = _recipe()
    out = tmp_path / "anim.gif"
    calls = []
    n = animation.save_gif(r, str(out), progress=lambda d, t: calls.append((d, t)))
    assert n == len(r.steps) + 1
    assert out.exists()
    with Image.open(out) as im:
        assert im.is_animated
        assert im.n_frames == n
    # 進捗コールバックが全フレーム分呼ばれる
    assert calls[-1] == (n, n)


def test_save_gif_invalid_fps(tmp_path):
    with pytest.raises(ValueError):
        animation.save_gif(_recipe(), str(tmp_path / "x.gif"), fps=0.0)


def test_cli_gif(tmp_path, capsys):
    """CLI --gif で GIF が出力される。"""
    recipe_path = tmp_path / "r.json"
    _recipe().save(str(recipe_path))
    out = tmp_path / "flow.gif"
    rc = cli_main([str(recipe_path), "--gif", str(out), "--gif-fps", "2"])
    assert rc == 0
    assert out.exists() and out.stat().st_size > 0
    err = capsys.readouterr().err
    assert "GIF アニメーションを保存しました" in err
