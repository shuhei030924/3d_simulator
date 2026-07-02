"""アプリ設定の永続化（GUI 非依存）。

最後に使ったフォルダ・最近開いたレシピ・既定ウェハ設定などを JSON で保存し、
次回起動時に復元する。GUI から切り離してあるためヘッドレスでテスト可能。

保存先は既定で `~/.semisim/settings.json`。環境変数 `SEMISIM_CONFIG_DIR` で
上書きできる（テストや可搬運用向け）。
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SETTINGS_VERSION = 1
DEFAULT_MAX_RECENT = 10


def config_dir() -> Path:
    """設定ファイルを置くディレクトリ。環境変数で上書き可能。"""
    override = os.environ.get("SEMISIM_CONFIG_DIR")
    if override:
        return Path(override)
    return Path.home() / ".semisim"


def default_path() -> Path:
    """既定の設定ファイルパス。"""
    return config_dir() / "settings.json"


@dataclass
class AppSettings:
    """ユーザのアプリ設定。

    last_dir: ファイルダイアログの初期フォルダ。
    recent_recipes: 最近開いた/保存したレシピのパス（新しい順、上限 max_recent）。
    default_config: 新規ウェハの既定設定（WaferConfig.to_dict 相当）。
    show_resist: レジスト表示の既定 ON/OFF。
    window_geometry: ウィンドウ位置/サイズの base64 文字列（GUI が任意で使用）。
    """

    last_dir: str = ""
    recent_recipes: list[str] = field(default_factory=list)
    default_config: dict = field(default_factory=dict)
    show_resist: bool = True
    window_geometry: str = ""
    max_recent: int = DEFAULT_MAX_RECENT
    ui_theme: str = "light"  # GUI のテーマ（"light" / "dark"）
    # ビュー操作の既定（前回終了時の状態を復元する）
    clip_mode: str = "Y"  # 断面モード none/X/Y/Z/angle/free
    rotate_style: int = 0  # 0=ターンテーブル / 1=自由（トラックボール）
    smooth_play: bool = True  # 再生のなめらか補間
    smooth_2d: bool = True  # 2D 断面のなめらか表示（境界±半ボクセルの超解像）
    smooth_3d: bool = True  # 3D 表示のサーフェス平滑化（Taubin）

    # -- 編集ヘルパ --------------------------------------------------------
    def add_recent(self, path: str) -> None:
        """レシピパスを最近リストの先頭へ追加（重複排除・上限適用）。"""
        if not path:
            return
        norm = os.path.abspath(path)
        # 既存の同一パスを除去してから先頭へ
        self.recent_recipes = [p for p in self.recent_recipes if os.path.abspath(p) != norm]
        self.recent_recipes.insert(0, norm)
        if self.max_recent > 0:
            del self.recent_recipes[self.max_recent:]
        # 親フォルダを最後に使ったフォルダとして記録
        self.last_dir = os.path.dirname(norm)

    def prune_missing(self) -> None:
        """存在しなくなった最近レシピをリストから除く。"""
        self.recent_recipes = [p for p in self.recent_recipes if os.path.exists(p)]

    # -- シリアライズ ------------------------------------------------------
    def to_dict(self) -> dict:
        d = asdict(self)
        d["settings_version"] = SETTINGS_VERSION
        return d

    @classmethod
    def from_dict(cls, d: dict) -> AppSettings:
        if not isinstance(d, dict):
            raise ValueError("設定データが不正です（辞書ではありません）。")

        # 型が壊れたフィールドはアプリ起動不能にせず既定値へフォールバックする
        # （手編集やバージョン差で型が崩れた設定でも安全に読み込む）。
        def _int(value, default: int) -> int:
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        recent = d.get("recent_recipes", [])
        recent = [str(x) for x in recent] if isinstance(recent, list) else []
        cfg = d.get("default_config", {})
        cfg = dict(cfg) if isinstance(cfg, dict) else {}
        return cls(
            last_dir=str(d.get("last_dir", "")),
            recent_recipes=recent,
            default_config=cfg,
            show_resist=bool(d.get("show_resist", True)),
            window_geometry=str(d.get("window_geometry", "")),
            max_recent=_int(d.get("max_recent", DEFAULT_MAX_RECENT), DEFAULT_MAX_RECENT),
            ui_theme=("dark" if str(d.get("ui_theme", "light")) == "dark" else "light"),
            clip_mode=(
                str(d.get("clip_mode", "Y"))
                if str(d.get("clip_mode", "Y")) in ("none", "X", "Y", "Z", "angle", "free")
                else "Y"
            ),
            rotate_style=1 if _int(d.get("rotate_style", 0), 0) == 1 else 0,
            smooth_play=bool(d.get("smooth_play", True)),
            smooth_2d=bool(d.get("smooth_2d", True)),
            smooth_3d=bool(d.get("smooth_3d", True)),
        )

    # -- 永続化 ------------------------------------------------------------
    def save(self, path: str | os.PathLike | None = None) -> Path:
        """設定を JSON に保存する。フォルダが無ければ作成。"""
        target = Path(path) if path is not None else default_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        return target

    @classmethod
    def load(cls, path: str | os.PathLike | None = None) -> AppSettings:
        """設定を読み込む。ファイルが無い/壊れている場合は既定値を返す。"""
        target = Path(path) if path is not None else default_path()
        if not target.exists():
            return cls()
        try:
            with open(target, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError):
            # 壊れた設定でアプリが起動不能にならないよう既定値にフォールバック
            return cls()
        try:
            return cls.from_dict(data)
        except (ValueError, TypeError):
            # JSON は妥当でも構造/型が不正（dict でない等）な場合の安全網
            return cls()
