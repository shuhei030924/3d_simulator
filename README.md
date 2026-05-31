# 半導体プロセス 3D シミュレータ

ボクセルベースで半導体プロセス（フォトリソ・成膜・エッチング・拡散・酸化・CMP など）を
逐次適用し、3D / 2D 断面で確認できる Python 製シミュレータです。

## 特長

- **充填ボリューム表現**: 断面が空洞にならず、常に中身が詰まって表示されます。
- **自由な断面**: X / Y / Z / 角度指定 / マウス操作の任意断面。2D 断面ビューでは
  クリック 2 点で膜厚・CD を実寸（µm）測定できます。
- **工程を自由に追加**: PHOTO / CVD / PVD / DRY / WET / DIFFUSION / OXIDE / CMP / STRIP。
- **任意角度パターン**: 回転矩形・帯・周期ライン（グレーティング）。
- **アンドゥ / リドゥ**、レシピの JSON 保存 / 読込、スナップショットキャッシュによる
  高速プレビュー。

## セットアップ

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 実行

```powershell
python main.py
```

## テスト

```powershell
python test_engine.py
python test_cache.py
python test_physics.py
python test_slice.py
python test_stack.py
python test_angles.py
```

## 構成

| モジュール | 役割 |
| --- | --- |
| `semisim/materials.py` | 材料定義（ID・色・属性） |
| `semisim/grid.py` | ボクセルグリッド（Wafer） |
| `semisim/masks.py` | フォトマスク図形（分数座標 0..1） |
| `semisim/processes.py` | 各プロセス工程のロジック |
| `semisim/recipe.py` | レシピ管理・シミュレーション・保存/読込 |
| `semisim/visualize.py` | PyVista / matplotlib 可視化ヘルパ |
| `semisim/gui.py` | PyQt5 + PyVista GUI 本体 |
