"""ALD・CMP研磨ストップ・TSVプリセットのテスト。"""
from __future__ import annotations

import pytest

from semisim import materials, presets
from semisim.grid import WaferConfig
from semisim.processes import ALD, CMP, CVD
from semisim.recipe import Recipe


# --- ALD -------------------------------------------------------------------
def test_ald_deposits_conformal(wafer):
    before = int((wafer.grid == materials.get("hafnia").id).sum())
    ALD(material="hafnia", cycles=50, growth_per_cycle_nm=2.0).apply(wafer)
    after = int((wafer.grid == materials.get("hafnia").id).sum())
    assert after > before


def test_ald_thickness_from_cycles():
    a = ALD(material="hafnia", cycles=100, growth_per_cycle_nm=1.5)
    assert a.thickness_um == pytest.approx(0.15)


def test_ald_invalid_cycles_raises(wafer):
    with pytest.raises(ValueError):
        ALD(material="hafnia", cycles=0, growth_per_cycle_nm=1.0).apply(wafer)


def test_ald_roundtrip():
    a = ALD(material="tan", cycles=80, growth_per_cycle_nm=1.2)
    restored = ALD._from_params(a.params_dict())
    assert restored.material == "tan"
    assert restored.cycles == 80
    assert restored.growth_per_cycle_nm == 1.2


# --- CMP 研磨ストップ ------------------------------------------------------
def test_cmp_stop_material_preserves_stop_layer():
    cfg = WaferConfig(nx=40, ny=40, nz=60, pitch_um=0.1, substrate_um=2.0)
    r = Recipe(config=cfg)
    r.add(CVD(material="nitride", thickness_um=0.5))  # ストップ層
    r.add(CVD(material="oxide", thickness_um=1.0))    # 上に厚い酸化膜
    r.add(CMP(remove_um=5.0, stop_material="nitride"))
    w = r.simulate()
    # ストップ層（窒化膜）は研磨後も残る
    assert (w.grid == materials.get("nitride").id).any()


def test_cmp_stop_roundtrip():
    c = CMP(remove_um=0.7, stop_material="nitride")
    restored = CMP._from_params(c.params_dict())
    assert restored.remove_um == 0.7
    assert restored.stop_material == "nitride"


# --- TSV プリセット --------------------------------------------------------
def test_tsv_preset_builds_and_simulates():
    r = presets.build("TSV 貫通ビア")
    w = r.simulate()
    # 充填銅とバリアが存在する
    assert (w.grid == materials.get("metal_cu").id).any()
    assert (w.grid == materials.get("tan").id).any()


def test_tsv_in_available():
    assert "TSV 貫通ビア" in presets.available()
