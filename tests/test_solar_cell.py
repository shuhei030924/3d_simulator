"""太陽電池 I-V（Voc/Isc/FF/効率）のテスト。"""
import math

import pytest

from semisim import metrology as M

_K = 8.617333e-5


def test_isc_equals_photocurrent():
    """短絡電流 Isc=IL。"""
    r = M.solar_cell_iv(photocurrent_a=35e-3)
    assert r["isc_a"] == pytest.approx(35e-3, rel=1e-12)


def test_voc_formula():
    """Voc=n·Vt·ln(IL/Is(T)+1)（25℃ では Is(T)=i_sat_a）。"""
    r = M.solar_cell_iv(photocurrent_a=35e-3, i_sat_a=1e-12, temperature_c=25.0)
    vt = _K * 298.15
    expected = vt * math.log(35e-3 / 1e-12 + 1.0)
    assert r["voc_v"] == pytest.approx(expected, rel=1e-6)


def test_current_zero_at_voc():
    """V=Voc で I≈0。"""
    r = M.solar_cell_iv()
    assert r["i"][-1] == pytest.approx(0.0, abs=1e-9)


def test_fill_factor_in_physical_range():
    """曲線因子 FF は 0.7〜0.85 の現実的範囲。"""
    ff = M.solar_cell_iv(ideality=1.0)["fill_factor"]
    assert 0.7 < ff < 0.85


def test_mpp_within_bounds():
    """最大電力点は 0<Vmp<Voc・0<Imp<Isc。"""
    r = M.solar_cell_iv()
    assert 0.0 < r["v_mp_v"] < r["voc_v"]
    assert 0.0 < r["i_mp_a"] < r["isc_a"]
    assert r["p_max_w"] == pytest.approx(r["v_mp_v"] * r["i_mp_a"], rel=1e-9)


def test_efficiency_definition():
    """効率 η=Pmax/Pin。"""
    r = M.solar_cell_iv(input_power_w=100e-3)
    assert r["efficiency"] == pytest.approx(r["p_max_w"] / 100e-3, rel=1e-9)


def test_voc_decreases_with_temperature():
    """Voc は温度上昇で低下する（Si で約 −2mV/℃）。"""
    v25 = M.solar_cell_iv(temperature_c=25.0, i_sat_a=1e-12)["voc_v"]
    v75 = M.solar_cell_iv(temperature_c=75.0, i_sat_a=1e-12)["voc_v"]
    assert v75 < v25
    dvdt = (v75 - v25) / 50.0 * 1000.0  # mV/℃
    assert -3.0 < dvdt < -1.0


def test_saturation_current_reference_at_25c():
    """Is(25℃)=i_sat_a（基準温度で一致）。"""
    r = M.solar_cell_iv(temperature_c=25.0, i_sat_a=2e-12)
    assert r["i_sat_t_a"] == pytest.approx(2e-12, rel=1e-9)


def test_isc_linear_voc_log_with_illumination():
    """照度↑で Isc は線形、Voc は対数的に増える。"""
    lo = M.solar_cell_iv(photocurrent_a=10e-3, i_sat_a=1e-12)
    hi = M.solar_cell_iv(photocurrent_a=70e-3, i_sat_a=1e-12)
    assert hi["isc_a"] / lo["isc_a"] == pytest.approx(7.0, rel=1e-9)
    # Voc 差は n·Vt·ln(7)
    assert hi["voc_v"] - lo["voc_v"] == pytest.approx(
        _K * 298.15 * math.log(7.0), rel=1e-3)


def test_no_illumination_zero_output():
    """IL=0 では全出力ゼロ。"""
    r = M.solar_cell_iv(photocurrent_a=0.0)
    assert r["voc_v"] == 0.0 and r["p_max_w"] == 0.0 and r["efficiency"] == 0.0


def test_invalid_inputs_raise():
    """非正 Is・ideality・入射電力はエラー。"""
    with pytest.raises(ValueError):
        M.solar_cell_iv(i_sat_a=0.0)
    with pytest.raises(ValueError):
        M.solar_cell_iv(input_power_w=0.0)
