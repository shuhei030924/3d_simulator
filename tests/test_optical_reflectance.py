"""薄膜光学反射率（TMM 垂直入射）のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig

LAM = 0.633  # µm


def _bare_si(pitch=0.001, nz=40, sub=20) -> Wafer:
    w = Wafer(WaferConfig(nx=6, ny=6, nz=nz, pitch_um=pitch, substrate_um=0.0))
    w.grid[:, :, :] = materials.AIR
    w.grid[:sub, :, :] = materials.get("silicon").id
    return w


def _film_on_si(film: str, d_um: float, pitch=0.001) -> Wafer:
    nvox = int(round(d_um / pitch))
    w = Wafer(WaferConfig(nx=6, ny=6, nz=600, pitch_um=pitch, substrate_um=0.0))
    w.grid[:, :, :] = materials.AIR
    w.grid[:300, :, :] = materials.get("silicon").id
    w.grid[300:300 + nvox, :, :] = materials.get(film).id
    return w


def test_bare_substrate_fresnel():
    """裸基板は Fresnel R=((n0−ns)/(n0+ns))²（+ k 項）。"""
    w = _bare_si()
    r = M.optical_reflectance(w, wavelength_um=LAM)
    n, k = 3.88, 0.02
    fresnel = ((1 - n) ** 2 + k ** 2) / ((1 + n) ** 2 + k ** 2)
    assert r["reflectance"] == pytest.approx(fresnel, rel=1e-6)
    assert r["n_layers"] == 0


def test_quarter_wave_arc_minimizes_reflectance():
    """λ/4 反射防止膜（n≈√(n0·ns)）で反射率がほぼ 0。"""
    d_qw = LAM / (4 * 2.0)  # 窒化膜 n=2.0
    w = _film_on_si("nitride", d_qw)
    r = M.optical_reflectance(w, wavelength_um=LAM)
    assert r["reflectance"] < 0.01
    assert r["n_layers"] == 1


def test_arc_beats_bare_substrate():
    """ARC 付きは裸基板より低反射。"""
    bare = M.optical_reflectance(_bare_si(), wavelength_um=LAM)["reflectance"]
    arc = M.optical_reflectance(_film_on_si("nitride", LAM / (4 * 2.0)),
                                wavelength_um=LAM)["reflectance"]
    assert arc < bare


def test_half_wave_film_is_invisible():
    """λ/2 膜は光学的に不可視（基板の反射率に戻る）。"""
    bare = M.optical_reflectance(_bare_si(), wavelength_um=LAM)["reflectance"]
    hw = M.optical_reflectance(_film_on_si("nitride", LAM / (2 * 2.0)),
                               wavelength_um=LAM)["reflectance"]
    assert hw == pytest.approx(bare, rel=1e-3)


def test_reflectance_in_valid_range():
    """反射率は 0〜1 の範囲。"""
    for d in (0.02, 0.05, 0.08, 0.12):
        r = M.optical_reflectance(_film_on_si("oxide", d), wavelength_um=LAM)
        assert 0.0 <= r["reflectance"] <= 1.0


def test_negative_wavelength_raises():
    """波長が非正ならエラー。"""
    with pytest.raises(ValueError):
        M.optical_reflectance(_bare_si(), wavelength_um=-0.5)


def test_no_solid_zero_reflectance():
    """固体が無ければ R=0。"""
    w = Wafer(WaferConfig(nx=5, ny=5, nz=10, pitch_um=0.001, substrate_um=0.0))
    w.grid[:, :, :] = materials.AIR
    r = M.optical_reflectance(w, wavelength_um=LAM)
    assert r["reflectance"] == 0.0
    assert r["n_layers"] == 0
