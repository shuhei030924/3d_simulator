"""歪み Si のピエゾ抵抗移動度変化のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

_PI = {"electron": -31.6e-11, "hole": 71.8e-11}


def test_zero_stress_no_change():
    """無応力では移動度係数 1.0。"""
    r = M.strained_mobility(0.0, carrier="electron")
    assert r["mobility_factor"] == pytest.approx(1.0, rel=1e-12)
    assert r["delta_mu_over_mu"] == pytest.approx(0.0, abs=1e-15)


def test_delta_mu_formula():
    """Δμ/μ = −π_l·σ。"""
    sigma_mpa = 800.0
    r = M.strained_mobility(sigma_mpa, carrier="electron")
    expected = -_PI["electron"] * sigma_mpa * 1e6
    assert r["delta_mu_over_mu"] == pytest.approx(expected, rel=1e-9)


def test_electron_tensile_enhances():
    """電子は引張応力（σ>0）で移動度向上、圧縮で低下。"""
    tens = M.strained_mobility(1000.0, carrier="electron")["mobility_factor"]
    comp = M.strained_mobility(-1000.0, carrier="electron")["mobility_factor"]
    assert tens > 1.0 > comp


def test_hole_compressive_enhances():
    """正孔は圧縮応力（σ<0）で移動度向上、引張で低下。"""
    comp = M.strained_mobility(-1000.0, carrier="hole")["mobility_factor"]
    tens = M.strained_mobility(1000.0, carrier="hole")["mobility_factor"]
    assert comp > 1.0 > tens


def test_factor_scales_mobility():
    """実効移動度 = base × factor。"""
    r = M.strained_mobility(1000.0, carrier="electron", base_mobility_cm2_vs=1400.0)
    assert r["mobility_cm2_vs"] == pytest.approx(1400.0 * r["mobility_factor"], rel=1e-9)


def test_factor_clipped_nonnegative():
    """極端な応力でも移動度係数は非負にクリップ。"""
    r = M.strained_mobility(1e5, carrier="hole")  # 巨大引張で hole は大幅低下
    assert r["mobility_factor"] >= 0.0


def test_invalid_carrier_raises():
    """未知のキャリア種別はエラー。"""
    with pytest.raises(ValueError):
        M.strained_mobility(500.0, carrier="photon")


def test_channel_strain_reads_material_stress():
    """channel_strain_mobility はチャネル材料の stress_mpa を使う。"""
    w = Wafer(WaferConfig(nx=6, ny=6, nz=12, pitch_um=0.1, substrate_um=0.0))
    w.grid[:6, :, :] = materials.get("silicon").id
    # nitride（stress +1000MPa 引張）をチャネルにすると電子移動度が向上
    direct = M.strained_mobility(materials.get("nitride").stress_mpa,
                                 carrier="electron")
    via = M.channel_strain_mobility(w, channel="nitride", carrier="electron")
    assert via["mobility_factor"] == pytest.approx(direct["mobility_factor"], rel=1e-12)
    assert via["mobility_factor"] > 1.0  # 引張ライナで電子向上
