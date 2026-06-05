"""非導体（無ドープ Si 基板）を対向電極にした容量計算の回帰テスト。

指定した電極材料が導体分類（ρ>0）でない場合でも、電極＝導体として扱い
有限の容量を返すこと（従来は parasitic_ff=0・field=NaN になっていた）。
"""
import numpy as np
import pytest

from semisim import metrology as M
from semisim.grid import WaferConfig
from semisim.processes import CVD
from semisim.recipe import Recipe


def _metal_over_si():
    """Cu(0.3) / oxide(0.3) / 無ドープ Si 基板。"""
    cfg = WaferConfig(nx=60, ny=20, nz=40, pitch_um=0.05, substrate_um=0.5)
    r = Recipe(config=cfg)
    r.add(CVD(material="oxide", thickness_um=0.3))
    r.add(CVD(material="metal_cu", thickness_um=0.3))
    return r.simulate()


def test_parasitic_ff_with_silicon_electrode_nonzero():
    """基板 Si を電極にしても対基板容量が有限・正で得られる。"""
    w = _metal_over_si()
    c = M.parasitic_capacitance_ff(w, "metal_cu", "silicon")
    assert np.isfinite(c) and c > 0


def test_field_solver_with_silicon_not_nan():
    """場ソルバが Si 電極で NaN を返さない（特異行列回避）。"""
    w = _metal_over_si()
    c = M.parasitic_capacitance_field_ff(w, "metal_cu", "silicon", axis="y")
    assert np.isfinite(c) and c > 0


def test_parallel_plate_and_field_agree_for_si():
    """平行平板近似と場ソルバが Si 電極でも一致する。"""
    w = _metal_over_si()
    cp = M.parasitic_capacitance_ff(w, "metal_cu", "silicon")
    cf = M.parasitic_capacitance_field_ff(w, "metal_cu", "silicon", axis="y")
    assert cf == pytest.approx(cp, rel=0.05)


def test_capacitance_matrix_with_silicon_finite():
    """容量行列に Si 電極を含めても全要素が有限。"""
    w = _metal_over_si()
    cm = M.capacitance_matrix_ff(w, ["metal_cu", "silicon"])
    assert np.all(np.isfinite(np.array(cm["matrix_ff"])))
    coup = list(cm["coupling_ff"].values())[0]
    assert np.isfinite(coup) and coup > 0


def test_rc_delay_over_substrate_nonzero():
    """基板を戻り電極にした RC 遅延が 0 でなく、Elmore≈1/2·RC。"""
    w = _metal_over_si()
    rc = M.rc_delay_ps(w, "metal_cu", "silicon", axis="x")
    el = M.elmore_delay_ps(w, "metal_cu", "silicon", axis="x")
    assert np.isfinite(rc) and rc > 0
    assert el["capacitance_ff"] > 0
    # 一様線で Elmore ≈ 1/2 lumped RC
    assert el["elmore_delay_ps"] == pytest.approx(0.5 * el["lumped_rc_ps"], rel=0.1)


def test_metal_metal_unchanged():
    """金属同士の容量は本修正で変化しない（回帰防止）。"""
    cfg = WaferConfig(nx=60, ny=20, nz=40, pitch_um=0.05, substrate_um=0.5)
    r = Recipe(config=cfg)
    r.add(CVD(material="metal_cu", thickness_um=0.3))
    r.add(CVD(material="oxide", thickness_um=0.3))
    r.add(CVD(material="metal_al", thickness_um=0.3))
    w = r.simulate()
    cp = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")
    cf = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu", axis="y")
    assert cp == pytest.approx(cf, rel=0.02)
    assert cp > 0
