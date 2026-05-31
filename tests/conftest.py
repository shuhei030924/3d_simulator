"""pytest 共通設定。GUI 非依存でエンジンを検証する。"""
from __future__ import annotations

import os
import sys

import pytest

# 親ディレクトリ（semisim パッケージのある場所）を import パスに追加
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ヘッドレス（CI/テスト）であることを明示
os.environ.setdefault("SEMISIM_HEADLESS", "1")

from semisim.grid import Wafer, WaferConfig  # noqa: E402


@pytest.fixture
def small_cfg() -> WaferConfig:
    """小さめのウェハ設定（高速テスト用）。"""
    return WaferConfig(nx=40, ny=40, nz=50, pitch_um=0.1, substrate_um=2.0)


@pytest.fixture
def wafer(small_cfg) -> Wafer:
    return Wafer(small_cfg)
