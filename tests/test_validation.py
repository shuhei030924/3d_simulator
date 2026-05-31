"""入力バリデーション（負/ゼロパラメータで ValueError）のテスト。"""
from __future__ import annotations

import pytest

from semisim.masks import Mask
from semisim.processes import (
    CMP,
    CVD,
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    Diffusion,
    DryEtch,
    Epitaxy,
    Implant,
    Oxidation,
    Photo,
    WetEtch,
)


@pytest.mark.parametrize(
    "proc",
    [
        CVD(material="oxide", thickness_um=0.0),
        CVD(material="oxide", thickness_um=-1.0),
        PVD(material="metal_al", thickness_um=-0.5),
        DryEtch(targets=["oxide"], depth_um=0.0),
        WetEtch(targets=["oxide"], depth_um=-0.3),
        Diffusion(dopant="doped_n", depth_um=0.0),
        Oxidation(thickness_um=-0.1),
        CMP(remove_um=0.0),
        Photo(mask=Mask(), thickness_um=0.0),
        Implant(range_um=0.0),
        Anneal(depth_um=-0.2),
        Epitaxy(thickness_um=0.0),
        AnisoWetEtch(depth_um=0.0),
        DRIE(depth_um=-1.0),
    ],
)
def test_invalid_params_raise(proc, wafer):
    with pytest.raises(ValueError):
        proc.apply(wafer)
