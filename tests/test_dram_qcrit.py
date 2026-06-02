"""DRAM リテンション・ソフトエラー臨界電荷・HCI オーバーフロー修正のテスト。"""
import warnings

import pytest

from semisim import metrology as M


def test_dram_retention_decreases_with_temperature():
    """高温ほどリークが増えリテンション時間が短くなる。"""
    cool = M.dram_retention_time_s(20.0, 0.5, 25.0)
    hot = M.dram_retention_time_s(20.0, 0.5, 85.0)
    assert hot["retention_s"] < cool["retention_s"]
    assert hot["leakage_a"] > cool["leakage_a"]


def test_dram_retention_proportional_to_cap():
    """リテンション時間は蓄積容量に比例する。"""
    r1 = M.dram_retention_time_s(10.0, 0.5, 85.0)
    r4 = M.dram_retention_time_s(40.0, 0.5, 85.0)
    assert r4["retention_s"] == pytest.approx(4 * r1["retention_s"], rel=1e-9)


def test_dram_retention_realistic_at_85c():
    """85°C のリテンションが現実的な範囲（ms〜数 s オーダー）。"""
    r = M.dram_retention_time_s(20.0, 0.5, 85.0)
    assert 1e-3 < r["retention_s"] < 10.0


def test_dram_retention_invalid():
    with pytest.raises(ValueError):
        M.dram_retention_time_s(20.0, 0.5, 85.0, sense_margin_v=0.0)


def test_critical_charge_cv():
    """臨界電荷 Q_crit = C·V。"""
    assert M.critical_charge_fc(20.0, 1.2) == pytest.approx(24.0)
    assert M.critical_charge_fc(40.0, 1.0) == pytest.approx(40.0)


def test_critical_charge_negative_raises():
    with pytest.raises(ValueError):
        M.critical_charge_fc(-1.0)


def test_hci_no_overflow_warning_at_tiny_vds():
    """極小 Vds でも HCI 寿命は警告なしで inf を返す（オーバーフロー回避）。"""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # RuntimeWarning を例外化
        val = M.hci_lifetime(1e-6)
    assert val == float("inf")
    # 通常値は不変
    assert M.hci_lifetime(1.5) == pytest.approx(1.0e-6 * __import__("numpy").exp(30 / 1.5), rel=1e-9)
