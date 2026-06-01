"""可視化ヘルパの材料非表示(hidden_ids)機能のテスト。"""
from __future__ import annotations

import numpy as np
import pytest

# visualize は pyvista 依存。未導入の headless CI ではモジュールごとスキップ
pytest.importorskip("pyvista")

from semisim import materials, visualize  # noqa: E402
from semisim.grid import Wafer, WaferConfig  # noqa: E402
from semisim.processes import CVD  # noqa: E402


def _wafer():
    cfg = WaferConfig(nx=30, ny=30, nz=40, pitch_um=0.1, substrate_um=2.0)
    w = Wafer(cfg)
    CVD(material="oxide", thickness_um=0.5).apply(w)
    CVD(material="poly", thickness_um=0.5).apply(w)
    return w


def test_slice_2d_hides_material():
    w = _wafer()
    poly = materials.BY_NAME["poly"].id
    plane, _, _ = visualize.slice_2d(w, "Y", 15)
    assert (plane == poly).any()  # 通常はポリが見える
    plane_h, _, _ = visualize.slice_2d(w, "Y", 15, hidden_ids=[poly])
    assert not (plane_h == poly).any()  # 非表示にすると消える
    # 非表示にした分は空気になっている
    assert (plane_h == materials.AIR).sum() > (plane == materials.AIR).sum()


def test_slice_2d_hidden_none_unchanged():
    w = _wafer()
    a, _, _ = visualize.slice_2d(w, "Y", 15)
    b, _, _ = visualize.slice_2d(w, "Y", 15, hidden_ids=None)
    assert np.array_equal(a, b)


def test_solid_unstructured_hidden_reduces_cells():
    w = _wafer()
    poly = materials.BY_NAME["poly"].id
    full = visualize.solid_unstructured(w)
    hidden = visualize.solid_unstructured(w, hidden_ids=[poly])
    assert hidden.n_cells < full.n_cells


def test_cli_hide_png_export(tmp_path):
    from semisim.cli import main as cli_main

    out = tmp_path / "cs.png"
    code = cli_main([
        "--preset",
        "MOSFET フロー",
        "--png",
        str(out),
        "--hide",
        "oxide",
    ])
    assert code == 0
    assert out.exists() and out.stat().st_size > 0


def test_cli_hide_bad_material(tmp_path):
    from semisim.cli import main as cli_main

    out = tmp_path / "cs.png"
    # 未知材料名はエラーで終了コード非0
    try:
        code = cli_main([
            "--preset",
            "MOSFET フロー",
            "--png",
            str(out),
            "--hide",
            "not_a_material",
        ])
    except (ValueError, SystemExit):
        return  # 例外で弾かれるのも許容
    assert code != 0

