"""リソ空間像モデル・プロセスウィンドウのテスト。"""
import numpy as np
import pytest

from semisim import litho as L


def test_nominal_cd_low_bias():
    """ベストフォーカス・公称露光量で印刷 CD はマスク開口にほぼ一致（バイアス小）。"""
    pitch = 0.005
    for target in (0.08, 0.12, 0.20):
        m = L.isolated_space_mask(target, pitch)
        cd = L.printed_cd_um(m, pitch)
        assert cd == pytest.approx(target, abs=0.003)


def test_cd_increases_with_dose():
    """ポジ型では露光量を上げると開口 CD が広がる（単調）。"""
    pitch = 0.005
    m = L.isolated_space_mask(0.10, pitch)
    cds = [L.printed_cd_um(m, pitch, dose=d) for d in (0.85, 1.0, 1.15, 1.3)]
    assert all(b > a for a, b in zip(cds, cds[1:]))


def test_defocus_degrades_cd():
    """焦点はずれが大きいほどコントラストが落ち、最終的に解像しなくなる。"""
    pitch = 0.005
    m = L.isolated_space_mask(0.10, pitch)
    cd0 = L.printed_cd_um(m, pitch, focus_um=0.0)
    cd_mid = L.printed_cd_um(m, pitch, focus_um=0.1)
    cd_far = L.printed_cd_um(m, pitch, focus_um=0.4)
    assert cd0 > cd_mid >= cd_far
    assert cd_far == 0.0  # 大きな焦点はずれで解像不能


def test_bossung_symmetric_in_focus():
    """Bossung は焦点に対してほぼ対称（±focus で同等 CD）。"""
    pitch = 0.005
    m = L.isolated_space_mask(0.12, pitch)
    cd_plus = L.printed_cd_um(m, pitch, focus_um=0.08)
    cd_minus = L.printed_cd_um(m, pitch, focus_um=-0.08)
    assert cd_plus == pytest.approx(cd_minus, abs=1e-6)


def test_epe_sign_and_zero():
    """EPE は公称で≈0、露光過多で正（広がり）、露光不足で負（細り）。"""
    pitch = 0.005
    assert L.edge_placement_error_um(0.10, pitch, dose=1.0) == pytest.approx(0.0, abs=0.002)
    assert L.edge_placement_error_um(0.10, pitch, dose=1.15) > 0
    assert L.edge_placement_error_um(0.10, pitch, dose=0.85) < 0


def test_meef_rises_near_resolution():
    """MEEF は大きなフィーチャで≈1、解像限界付近で増大する。"""
    pitch = 0.005
    meef_large = L.meef(0.30, pitch)
    meef_small = L.meef(0.05, pitch)
    assert meef_large == pytest.approx(1.0, abs=0.1)
    assert meef_small > 1.5


def test_process_window_nonempty_and_shrinks_with_tol():
    """プロセスウィンドウは規格内点を持ち、許容を厳しくすると縮む。"""
    pitch = 0.005
    m = L.isolated_space_mask(0.10, pitch)
    doses = np.linspace(0.85, 1.15, 16)
    foc = np.linspace(-0.3, 0.3, 16)
    cd = L.bossung(m, pitch, doses, foc)
    pw10 = L.process_window(cd, doses, foc, target_cd_um=0.10, tol_pct=10)
    pw5 = L.process_window(cd, doses, foc, target_cd_um=0.10, tol_pct=5)
    assert pw10["area_frac"] > 0
    assert pw10["dof_um"] > 0
    assert pw10["exposure_latitude_pct"] > 0
    assert pw5["area_frac"] <= pw10["area_frac"]


def test_bossung_shape():
    """Bossung 行列の形状が (focus, dose)。"""
    pitch = 0.005
    m = L.isolated_space_mask(0.10, pitch)
    doses = np.linspace(0.9, 1.1, 5)
    foc = np.linspace(-0.2, 0.2, 7)
    cd = L.bossung(m, pitch, doses, foc)
    assert cd.shape == (7, 5)
