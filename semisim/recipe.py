"""レシピ（工程列）の管理とシミュレーション、保存/読込。"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field

import numpy as np

from .grid import Wafer, WaferConfig
from .processes import Process

FORMAT_VERSION = 1


# スナップショットを保持する最大数（メモリ肥大を防ぐ上限）。
# 大きなグリッド × 多数の工程ではキャッシュが数百 MB になり得るため、
# 直近の工程ほど再利用されやすい性質を利用して古いものから解放する。
DEFAULT_MAX_SNAPSHOTS = 16


@dataclass
class Recipe:
    """ウェハ設定と工程列を保持し、リプレイで断面を再現する。"""

    config: WaferConfig = field(default_factory=WaferConfig)
    steps: list[Process] = field(default_factory=list)
    max_snapshots: int = DEFAULT_MAX_SNAPSHOTS
    # 各ステップ適用後のグリッドのスナップショット（増分計算用キャッシュ）。
    # index i = steps[0..i] を適用済みのグリッド。古いものは None で解放され得る。
    _snapshots: list[np.ndarray | None] = field(
        default_factory=list, init=False, repr=False, compare=False
    )

    # -- キャッシュ --------------------------------------------------------
    def invalidate(self, from_index: int = 0) -> None:
        """from_index 以降のキャッシュを破棄する。"""
        if from_index <= 0:
            self._snapshots = []
        else:
            del self._snapshots[from_index:]

    def _enforce_snapshot_limit(self) -> None:
        """非 None スナップショット数が上限を超えたら古いものから解放する。"""
        if self.max_snapshots <= 0:
            return
        live = [i for i, s in enumerate(self._snapshots) if s is not None]
        # 直近（大きい index）を優先的に残し、古いものを None にする
        for i in live[: max(0, len(live) - self.max_snapshots)]:
            self._snapshots[i] = None

    # -- 編集 --------------------------------------------------------------
    def add(self, proc: Process, index: int | None = None) -> None:
        if index is None:
            self.steps.append(proc)
            self.invalidate(len(self.steps) - 1)
        else:
            self.steps.insert(index, proc)
            self.invalidate(index)

    def replace(self, index: int, proc: Process) -> None:
        """index の工程を差し替え、それ以降のキャッシュを破棄する。"""
        self.steps[index] = proc
        self.invalidate(index)

    def duplicate(self, index: int) -> int:
        """index の工程を複製して直後に挿入。新インデックスを返す。"""
        clone = copy.deepcopy(self.steps[index])
        self.steps.insert(index + 1, clone)
        self.invalidate(index + 1)
        return index + 1

    def remove(self, index: int) -> None:
        del self.steps[index]
        self.invalidate(index)

    def move(self, index: int, delta: int) -> int:
        """工程を delta だけ移動。新しいインデックスを返す。"""
        new_index = max(0, min(len(self.steps) - 1, index + delta))
        if new_index == index:
            return index
        proc = self.steps.pop(index)
        self.steps.insert(new_index, proc)
        self.invalidate(min(index, new_index))
        return new_index

    # -- シミュレーション --------------------------------------------------
    def simulate(self, up_to: int | None = None) -> Wafer:
        """初期状態から工程を順に適用したウェハを返す。

        スナップショットキャッシュを使い、未計算のステップのみを適用する。
        up_to が指定された場合、その本数だけ適用する（プレビュー用）。
        """
        n = len(self.steps) if up_to is None else min(up_to, len(self.steps))
        n = max(0, n)
        wafer = Wafer(self.config)

        # n-1 以下で生きているキャッシュのうち最も深いものから再開する
        start = 0
        for j in range(min(n, len(self._snapshots)) - 1, -1, -1):
            snap = self._snapshots[j]
            if snap is not None:
                wafer.grid = snap.copy()
                start = j + 1
                break

        for i in range(start, n):
            self.steps[i].apply(wafer)
            snap = wafer.grid.copy()
            if i < len(self._snapshots):
                self._snapshots[i] = snap
            else:
                self._snapshots.append(snap)
        if n > start:
            self._enforce_snapshot_limit()
        return wafer

    # -- 保存/読込 ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "config": self.config.to_dict(),
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> Recipe:
        if not isinstance(d, dict):
            raise ValueError("レシピデータが不正です（辞書ではありません）。")
        version = d.get("format_version", FORMAT_VERSION)
        if not isinstance(version, int) or version > FORMAT_VERSION:
            raise ValueError(
                f"未対応のレシピ形式バージョンです: {version}"
                f"（対応は {FORMAT_VERSION} 以下）。"
            )
        config = WaferConfig.from_dict(d.get("config", {}))
        steps = [Process.from_dict(s) for s in d.get("steps", [])]
        return cls(config=config, steps=steps)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str) -> Recipe:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"レシピファイルが見つかりません: {path}") from exc
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"レシピファイルが壊れています（JSON 解析エラー）: {path}\n{exc}"
            ) from exc
        return cls.from_dict(data)
