"""CTE 不整合熱応力 σ=E/(1−ν)·Δα·ΔT のテスト。"""
import pytest

from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _stack() -> Wafer:
    """Si 基板 + SiO2 + Al + Cu。"""
    w = Wafer(WaferConfig(nx=10, ny=10, nz=20, pitch_um=0.1, substrate_um=0.0))
    g = w.grid
    g[:10, :, :] = materials.get("silicon").id
    g[10:12, :, :] = materials.get("oxide").id
    g[12:14, :, :] = materials.get("metal_al").id
    g[14:16, :, :] = materials.get("metal_cu").id
    return w


def test_stress_formula_biaxial():
    """σ = E/(1−ν)·(α_ref−α_film)·ΔT（Al）。"""
    w = _stack()
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0, poisson=0.25)
    al = materials.get("metal_al")
    si = materials.get("silicon")
    expected = (al.youngs_modulus_gpa * 1e3) / (1 - 0.25) * (
        (si.cte_ppm_k - al.cte_ppm_k) * 1e-6) * (-400.0)
    assert r["stress_by_material"]["metal_al"] == pytest.approx(expected, rel=1e-9)


def test_cooling_high_cte_is_tensile():
    """冷却（ΔT<0）で高 CTE 膜（Al, Cu）は引張（正）。"""
    w = _stack()
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    assert r["stress_by_material"]["metal_al"] > 0
    assert r["stress_by_material"]["metal_cu"] > 0


def test_low_cte_film_opposite_sign():
    """低 CTE 膜（SiO2 < Si）は逆符号（冷却で圧縮）。"""
    w = _stack()
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    assert r["stress_by_material"]["oxide"] < 0


def test_heating_flips_sign():
    """加熱（ΔT>0）で符号が反転（Al は圧縮）。"""
    w = _stack()
    cool = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    heat = M.thermal_mismatch_stress(w, delta_t_k=+400.0)
    assert heat["stress_by_material"]["metal_al"] == pytest.approx(
        -cool["stress_by_material"]["metal_al"], rel=1e-9)


def test_stress_scales_with_delta_t():
    """熱応力は ΔT に比例。"""
    w = _stack()
    s1 = M.thermal_mismatch_stress(w, delta_t_k=-100.0)["stress_by_material"]["metal_al"]
    s3 = M.thermal_mismatch_stress(w, delta_t_k=-300.0)["stress_by_material"]["metal_al"]
    assert s3 == pytest.approx(3.0 * s1, rel=1e-9)


def test_reference_material_excluded():
    """基準材料（Si）は応力辞書に含まれない。"""
    w = _stack()
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    assert "silicon" not in r["stress_by_material"]


def test_worst_is_max_abs():
    """max_abs_material は |σ| 最大の材料。"""
    w = _stack()
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    worst = max(r["stress_by_material"],
                key=lambda k: abs(r["stress_by_material"][k]))
    assert r["max_abs_material"] == worst


def test_no_films_empty():
    """基板のみ（膜無し）では空。"""
    w = Wafer(WaferConfig(nx=5, ny=5, nz=10, pitch_um=0.1, substrate_um=0.0))
    w.grid[:5, :, :] = materials.get("silicon").id
    r = M.thermal_mismatch_stress(w, delta_t_k=-400.0)
    assert r["stress_by_material"] == {}
    assert r["max_abs_stress_mpa"] == 0.0
