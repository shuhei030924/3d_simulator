"""フォトダイオード応答度 R=η·λ/1.24 と遮断波長のテスト。"""
import pytest

from semisim import metrology as M

_HC = 1.239841984


def test_responsivity_formula():
    """R=η·λ/(hc/q) に一致。"""
    r = M.photodiode_responsivity(0.55, quantum_efficiency=0.8)
    assert r["responsivity_a_w"] == pytest.approx(0.8 * 0.55 / _HC, rel=1e-9)


def test_silicon_cutoff_wavelength():
    """Si（Eg=1.12eV）の遮断波長 λ_c≈1.107µm。"""
    r = M.photodiode_responsivity(0.55)
    assert r["cutoff_wavelength_um"] == pytest.approx(1.107, abs=0.005)


def test_responsivity_increases_with_wavelength():
    """遮断波長以下では波長が長いほど応答度が大きい。"""
    short = M.photodiode_responsivity(0.45, quantum_efficiency=0.8)["responsivity_a_w"]
    long = M.photodiode_responsivity(0.95, quantum_efficiency=0.8)["responsivity_a_w"]
    assert long > short > 0.0


def test_realistic_si_responsivity_at_850nm():
    """850nm・η=0.8 で R≈0.55 A/W（Si PD の代表値）。"""
    r = M.photodiode_responsivity(0.85, quantum_efficiency=0.8)["responsivity_a_w"]
    assert r == pytest.approx(0.55, abs=0.03)


def test_below_bandgap_zero_response():
    """遮断波長より長い波長（光子エネルギー<Eg）では R=0。"""
    r = M.photodiode_responsivity(1.5)
    assert r["responsivity_a_w"] == 0.0
    assert r["below_bandgap"] is True


def test_photon_energy():
    """光子エネルギー Eph=hc/λ を返す（λ=1µm→1.24eV）。"""
    r = M.photodiode_responsivity(1.0)
    assert r["photon_energy_ev"] == pytest.approx(_HC, rel=1e-9)


def test_larger_bandgap_shorter_cutoff():
    """バンドギャップが大きいほど遮断波長は短い。"""
    si = M.photodiode_responsivity(0.5, eg_ev=1.12)["cutoff_wavelength_um"]
    wide = M.photodiode_responsivity(0.5, eg_ev=2.0)["cutoff_wavelength_um"]
    assert wide < si


def test_quantum_efficiency_scales_responsivity():
    """応答度は量子効率に比例。"""
    r_half = M.photodiode_responsivity(0.6, quantum_efficiency=0.4)["responsivity_a_w"]
    r_full = M.photodiode_responsivity(0.6, quantum_efficiency=0.8)["responsivity_a_w"]
    assert r_full == pytest.approx(2.0 * r_half, rel=1e-9)


def test_invalid_inputs_raise():
    """非正波長・範囲外 η・非正 Eg はエラー。"""
    with pytest.raises(ValueError):
        M.photodiode_responsivity(0.0)
    with pytest.raises(ValueError):
        M.photodiode_responsivity(0.5, quantum_efficiency=1.5)
    with pytest.raises(ValueError):
        M.photodiode_responsivity(0.5, eg_ev=0.0)
