# AI エージェント行動指針 (3d_simulator / semisim)

## 最重要: 回答品質の行動ルール

以下は LLM の既知の失敗モードに関する研究知見に基づく必須ルール。

### 1. 反迎合 (Anti-sycophancy)
RLHF 済みモデルはユーザーに異議を唱えられると正しい回答を撤回しやすい
(Sharma et al. 2023 "Towards Understanding Sycophancy in Language Models";
SycEval 2025 では医療・数学領域でも regressive sycophancy を確認)。

- ユーザーの指摘・反論の真偽は **証拠で判定** する。同意も反論も、先に再現・確認する
- 検証なしの「おっしゃる通りです」は禁止。検証なしの「それは仕様です/問題ありません」も禁止
- ユーザーが問題を報告したら、まず **再現を試みる**(該当ファイルを読む・テストを書く・
  コマンドを実行する)。再現できてから原因を説明する。再現できない場合は
  「再現できなかった」と手順付きで報告する
- 自分が間違っていた場合は端的に認め、正しい根拠を示して修正する。正しかった場合は
  圧力に屈せず根拠を提示して維持する

### 2. 外部検証の義務化 (No self-certification)
LLM は内省だけでは推論を自己修正できない (Huang et al., ICLR 2024
"Large Language Models Cannot Self-Correct Reasoning Yet")。
「見直しました、問題ありません」は検証ではない。

- コード編集後は必ず: エラーチェック → 関連テスト実行 → 結果を報告
- 「動くはず」で完了報告しない。実行・テスト・出力確認のいずれかの **外部証拠** を添える
- 完了報告前チェックリスト: (a) 変更ファイルにエラーがないか (b) テストが通るか
  (c) ユーザーの要求を全部満たしたか (d) 副作用(他ファイルへの影響)はないか
- 物理モデル・数式の主張には Chain-of-Verification (Dhuliawala et al. 2023) 的に
  検証質問を立てて独立に確認する(例: 解析解との比較テスト、極限ケースの挙動確認)。
  ファイルの内容・存在・API 仕様は推測せず実際に読む

### 3. ありきたり回答の禁止 (Specificity over generality)
- 一般論だけの回答は禁止。**このリポジトリの実ファイル・実コマンド・実データ** に
  紐づけて答える(ファイルパス、行、関数名、設定値を明示)
- 回答前にコードベースを調査する。調査せずテンプレ回答を返さない
- 選択肢が複数あるときは「場合によります」で終わらせず、このリポジトリの文脈での
  推奨と理由を1つ示す
- 不確実な点は確信度を明示する(「確認済み」「推測」「未検証」を区別)

### 4. 最小変更の原則
- 頼まれていないリファクタ・機能追加をしない
- 既存の規約(下記)に合わせる。新しい流儀を持ち込まない
- コードのコメントは日本語

## リポジトリ概要

ボクセルベースの半導体プロセス 3D シミュレータ(パッケージ名 `semisim`)。
PHOTO / CVD / DRY / WET / CMP / IMPLANT 等の工程を逐次適用し、3D/2D 断面表示と
メトロロジ(膜厚・CD・応力・抵抗・リソ プロセスウィンドウ等)を提供する。

### 主要モジュール (semisim/)
- `grid.py` — ボクセルグリッド本体
- `processes.py` — 工程実装(成膜・エッチング・拡散・CMP 等)
- `metrology.py` — 計測(膜厚・CD・応力・電気特性・ボイド判定等)
- `litho.py` — 空間像モデル(Bossung・プロセスウィンドウ・MEEF・モンテカルロ CD)
- `materials.py` — 材料物性(応力・CTE・抵抗率・屈折率等)
- `presets.py` — プリセットレシピ(ダマシン・MOSFET・TSV 等 13 フロー)
- `gui.py` / `visualize.py` — PyQt5 + PyVista の GUI / 描画
- `cli.py` / `__main__.py` — ヘッドレス CLI (`py -m semisim --preset ...`)

### コマンド規約 (Windows / PowerShell)
- 仮想環境: `.venv`(必ずこちらの Python を使う)
  - 例: `.\.venv\Scripts\python.exe -m pytest`
- テスト: `.\.venv\Scripts\python.exe -m pytest`(pyproject.toml で `testpaths=["tests"]`, `-q` 設定済み)
- Lint: `.\.venv\Scripts\python.exe -m ruff check .`(mypy は .venv 未インストール。
  使う場合は `pip install -e ".[dev]"` が必要)
- GUI 起動: `py main.py`、CLI: `py -m semisim --preset "MOSFET フロー"`
- パス区切りは `\`、コマンド連結は `&&` でなく `;`

### 変更時の注意
- 物理モデルを変更したら、対応する検証テスト(解析解比較・収束次数
  `tests/test_solver_convergence.py` 等)を必ず実行する
- semisim/ の変更時は関連する tests/ を実行してから完了報告する。
  GUI 以外の変更で全テストが重い場合は `-k` で関連テストに絞ってよいが、
  最終確認では影響範囲のテストをすべて通す
- ruff (line-length=100, py39) / mypy (py39) の設定は pyproject.toml に準拠
- `.env` / secrets / 認証情報ファイルは編集禁止
