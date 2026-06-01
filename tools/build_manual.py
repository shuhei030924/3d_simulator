"""HTML 使い方説明書ジェネレータ（ヘッドレス）。

各プロセス操作のデモレシピを実行し、2D 断面のスクリーンショット(PNG)を
docs/manual/img/ に生成、説明・パラメータ・コード例込みの
docs/manual/index.html を出力する。GUI/PyVista 不要。

使い方:
    py tools/build_manual.py
"""
from __future__ import annotations

import html
import os
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス描画
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 画像内の日本語タイトルが豆腐にならないよう日本語フォントを優先設定する。
matplotlib.rcParams["font.family"] = [
    "Yu Gothic", "Meiryo", "MS Gothic", "Hiragino Sans", "sans-serif",
]
matplotlib.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semisim import materials, metrology, visualize  # noqa: E402
from semisim.grid import Wafer, WaferConfig  # noqa: E402
from semisim.masks import Mask, Shape  # noqa: E402
from semisim.processes import (  # noqa: E402
    ALD,
    CMP,
    CVD,
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    DryEtch,
    Epitaxy,
    Fill,
    Implant,
    LiftOff,
    Oxidation,
    Photo,
    Spacer,
    Strip,
    WetEtch,
)
from semisim.recipe import Recipe  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "docs", "manual")
IMG_DIR = os.path.join(OUT_DIR, "img")


def cfg(nx=200, ny=20, nz=140, pitch=0.04, sub=2.0) -> WaferConfig:
    return WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=pitch, substrate_um=sub)


def _center_mask(w=0.4) -> Mask:
    a = (1.0 - w) / 2.0
    return Mask(shapes=[Shape("rect", {"x0": a, "y0": 0.0, "x1": a + w, "y1": 1.0})])


def _stripe_mask(period=0.4, width=0.2) -> Mask:
    return Mask(shapes=[Shape("grating", {"angle": 90.0, "period": period, "width": width})])


def render(wafer: Wafer, title: str, fname: str, ylim=None) -> None:
    """中央 Y 断面（XZ 面）を PNG 保存する。"""
    plane, width_um, height_um = visualize.slice_2d(wafer, "Y", wafer.config.ny // 2)
    cmap, norm = visualize.material_listed_cmap()
    fig, ax = plt.subplots(figsize=(7, 4), dpi=110)
    ax.imshow(plane, origin="lower", cmap=cmap, norm=norm,
              extent=[0, width_um, 0, height_um], interpolation="nearest", aspect="auto")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")
    if ylim:
        ax.set_ylim(*ylim)
    present = sorted(set(int(v) for v in np.unique(plane)) - {materials.AIR})
    handles = []
    for mid in present:
        m = materials.BY_ID.get(mid)
        if m is not None:
            handles.append(mpatches.Patch(color=m.color, label=m.name))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, fname))
    plt.close(fig)


def _stepped(material="oxide", thick=0.6, mask_w=0.4, etch=0.5) -> Recipe:
    """段差（トレンチ）を持つ下地を作る共通レシピ。成膜デモの土台に使う。"""
    r = Recipe(config=cfg())
    r.add(CVD(material=material, thickness_um=thick))
    r.add(Photo(mask=_center_mask(mask_w), thickness_um=0.8, polarity="positive"))
    r.add(DryEtch(targets=[material], depth_um=etch))
    r.add(Strip(material="photoresist"))
    return r


# === 各デモの定義 ==========================================================
# (キー, 章タイトル, カテゴリ, 説明HTML, パラメータ表[(名前,説明)], コード例, ビルダー, ylim)
def demos() -> list:
    items = []

    # --- 下地・基板 ---
    def d_substrate():
        return Recipe(config=cfg()).simulate()
    items.append((
        "substrate", "ウェハ基板", "基礎",
        "シミュレーションは 3D ボクセル格子（[z, y, x]）で表現します。最下部に"
        "シリコン基板があり、その上に各プロセスで材料を積み上げ／削っていきます。"
        "下の図は中央 Y 断面（XZ 面）で、横が x、縦が z（下が基板）です。",
        [("nx, ny, nz", "格子分割数"), ("pitch_um", "1 ボクセルの一辺[µm]"),
         ("substrate_um", "初期シリコン基板厚[µm]")],
        "from semisim.grid import WaferConfig\n"
        "from semisim.recipe import Recipe\n"
        "cfg = WaferConfig(nx=200, ny=20, nz=140, pitch_um=0.04, substrate_um=2.0)\n"
        "wafer = Recipe(config=cfg).simulate()",
        d_substrate, None,
    ))

    # --- フォトリソ ---
    def d_photo():
        r = Recipe(config=cfg())
        r.add(Photo(mask=_stripe_mask(0.6, 0.3), thickness_um=0.8, polarity="positive"))
        return r.simulate()
    items.append((
        "photo", "フォトリソグラフィ (Photo)", "リソグラフィ",
        "フォトレジストを塗布し、マスクパターンで露光・現像してレジストを"
        "パターニングします。positive では開口部（露光部）のレジストが除去され、"
        "negative では逆になります。後続のエッチ／注入のマスクになります。",
        [("mask", "マスク形状(rect/grating など)"), ("thickness_um", "レジスト膜厚[µm]"),
         ("polarity", "positive / negative")],
        "from semisim.processes import Photo\n"
        "from semisim.masks import Mask, Shape\n"
        "mask = Mask(shapes=[Shape('grating', {'angle':90,'period':0.6,'width':0.3})])\n"
        "r.add(Photo(mask=mask, thickness_um=0.8, polarity='positive'))",
        d_photo, None,
    ))

    # --- CVD 成膜 ---
    def d_cvd():
        r = _stepped()
        r.add(CVD(material="nitride", thickness_um=0.2))
        return r.simulate()
    items.append((
        "cvd", "CVD 成膜 (CVD)", "成膜",
        "化学気相成長。表面から等方的に材料を堆積します。段差があっても"
        "ほぼ一様な膜厚で覆い（コンフォーマル）、トレンチ内壁にも回り込みます。"
        "下図はトレンチ付き酸化膜の上に窒化膜を CVD した断面です。",
        [("material", "成膜材料"), ("thickness_um", "膜厚[µm]"),
         ("conformality", "段差被覆性(1=完全等方)")],
        "from semisim.processes import CVD\n"
        "r.add(CVD(material='nitride', thickness_um=0.2))",
        d_cvd, None,
    ))

    # --- ALD 成膜 ---
    def d_ald():
        r = _stepped(etch=0.5, mask_w=0.25)
        r.add(ALD(material="hafnia", cycles=150, growth_per_cycle_nm=1.0))
        return r.simulate()
    items.append((
        "ald", "ALD 成膜 (ALD)", "成膜",
        "原子層堆積。1 サイクルごとに原子層 1 層を成長させ、nm 精度で膜厚を"
        "制御します。CVD より優れたコンフォーマル性で、高アスペクト比の溝でも"
        "底まで均一に被覆します。High-k やバリア膜に用います。",
        [("material", "成膜材料(hafnia 等)"), ("cycles", "サイクル数"),
         ("growth_per_cycle_nm", "1 サイクル成長量[nm]")],
        "from semisim.processes import ALD\n"
        "r.add(ALD(material='hafnia', cycles=150, growth_per_cycle_nm=1.0))",
        d_ald, None,
    ))

    # --- PVD 成膜 ---
    def d_pvd():
        r = _stepped(etch=0.6, mask_w=0.3)
        r.add(PVD(material="metal_al", thickness_um=0.3, step_coverage=0.4))
        return r.simulate()
    items.append((
        "pvd", "PVD 成膜 (PVD)", "成膜",
        "スパッタ／蒸着による物理気相成長。指向性が強く、段差被覆性が悪いと"
        "上端にオーバーハング（庇）ができ、トレンチ側壁・底は薄くなります。"
        "step_coverage を下げるほど側壁被覆が悪化します。",
        [("material", "成膜材料"), ("thickness_um", "平坦部膜厚[µm]"),
         ("step_coverage", "側壁/底の被覆率(0〜1)")],
        "from semisim.processes import PVD\n"
        "r.add(PVD(material='metal_al', thickness_um=0.3, step_coverage=0.4))",
        d_pvd, None,
    ))

    # --- スペーサ ---
    def d_spacer():
        r = Recipe(config=cfg())
        r.add(CVD(material="metal_al", thickness_um=0.5))
        r.add(Photo(mask=_center_mask(0.2), thickness_um=0.8, polarity="negative"))
        r.add(DryEtch(targets=["metal_al"], depth_um=0.6))
        r.add(Strip(material="photoresist"))
        r.add(Spacer(material="nitride", thickness_um=0.08))
        return r.simulate()
    items.append((
        "spacer", "サイドウォールスペーサ (Spacer)", "成膜",
        "コンフォーマル成膜＋異方性エッチバックの組合せ。水平面の膜を除去し、"
        "段差（ゲート等）の垂直側壁にだけ材料を残します。LDD やゲートスペーサを"
        "自己整合的に作る代表プロセスです。",
        [("material", "スペーサ材料"), ("thickness_um", "スペーサ幅[µm]"),
         ("overetch_um", "オーバーエッチ量[µm]")],
        "from semisim.processes import Spacer\n"
        "r.add(Spacer(material='nitride', thickness_um=0.08))",
        d_spacer, None,
    ))

    # --- ドライエッチ ---
    def d_dry():
        r = Recipe(config=cfg())
        r.add(CVD(material="oxide", thickness_um=0.8))
        r.add(Photo(mask=_stripe_mask(0.6, 0.3), thickness_um=0.8, polarity="positive"))
        r.add(DryEtch(targets=["oxide"], depth_um=0.6))
        r.add(Strip(material="photoresist"))
        return r.simulate()
    items.append((
        "dryetch", "ドライエッチング (DryEtch)", "エッチング",
        "プラズマによる異方性エッチング。ほぼ垂直な側壁の溝／ホールを形成します。"
        "レジスト開口部の直下のみが削れ、横方向にはほとんど広がりません。",
        [("targets", "エッチ対象材料リスト"), ("depth_um", "エッチ深さ[µm]"),
         ("lateral_bias_um", "横方向バイアス"), ("arde_lag_um", "ARDE/RIE ラグ")],
        "from semisim.processes import DryEtch\n"
        "r.add(DryEtch(targets=['oxide'], depth_um=0.6))",
        d_dry, None,
    ))

    # --- ウェットエッチ ---
    def d_wet():
        r = Recipe(config=cfg())
        r.add(CVD(material="oxide", thickness_um=0.8))
        r.add(Photo(mask=_stripe_mask(0.7, 0.3), thickness_um=0.8, polarity="positive"))
        r.add(WetEtch(targets=["oxide"], depth_um=0.5, lateral_ratio=1.0))
        return r.simulate()
    items.append((
        "wetetch", "ウェットエッチング (WetEtch)", "エッチング",
        "薬液による等方性エッチング。縦と同程度に横方向へも削れるため、マスク"
        "（レジスト）の下に回り込むアンダーカットが生じます。lateral_ratio で"
        "等方〜異方を連続調整できます。レジストを残したまま示しています。",
        [("targets", "エッチ対象材料"), ("depth_um", "エッチ深さ[µm]"),
         ("lateral_ratio", "横/縦エッチ比(1=完全等方)")],
        "from semisim.processes import WetEtch\n"
        "r.add(WetEtch(targets=['oxide'], depth_um=0.5, lateral_ratio=1.0))",
        d_wet, None,
    ))

    # --- 異方性ウェット（KOH） ---
    def d_koh():
        r = Recipe(config=cfg(nz=160, sub=3.0))
        r.add(CVD(material="nitride", thickness_um=0.3))
        r.add(Photo(mask=_center_mask(0.5), thickness_um=0.8, polarity="positive"))
        r.add(DryEtch(targets=["nitride"], depth_um=0.4))
        r.add(Strip(material="photoresist"))
        r.add(AnisoWetEtch(target="silicon", depth_um=1.6, sidewall_angle_deg=54.7))
        return r.simulate()
    items.append((
        "anisowet", "結晶異方性ウェット (AnisoWetEtch)", "エッチング",
        "KOH/TMAH による結晶面依存エッチング。Si(100) では (111) 面が現れ、"
        "54.7° のテーパを持つ V 字溝／逆ピラミッドを形成します。MEMS の"
        "マイクロ構造作製に使います。",
        [("target", "エッチ対象(silicon)"), ("depth_um", "エッチ深さ[µm]"),
         ("sidewall_angle_deg", "側壁角(54.7°=Si(111))")],
        "from semisim.processes import AnisoWetEtch\n"
        "r.add(AnisoWetEtch(target='silicon', depth_um=1.6, sidewall_angle_deg=54.7))",
        d_koh, None,
    ))

    # --- DRIE ---
    def d_drie():
        r = Recipe(config=cfg(nz=180, sub=4.0))
        r.add(Photo(mask=_stripe_mask(0.6, 0.3), thickness_um=1.0, polarity="positive"))
        r.add(DRIE(target="silicon", depth_um=3.0, scallop_um=0.12, scallop_pitch_um=0.35))
        r.add(Strip(material="photoresist"))
        return r.simulate()
    items.append((
        "drie", "深掘り RIE / Bosch (DRIE)", "エッチング",
        "Bosch プロセスによる深掘りエッチ。エッチと側壁保護を交互に繰り返すため、"
        "側壁に特徴的なスキャロップ（扇状の凹凸）が残ります。高アスペクト比の"
        "TSV やトレンチに用います。",
        [("target", "エッチ対象"), ("depth_um", "深さ[µm]"),
         ("scallop_um", "スキャロップ深さ"), ("scallop_pitch_um", "スキャロップ周期")],
        "from semisim.processes import DRIE\n"
        "r.add(DRIE(target='silicon', depth_um=3.0, scallop_um=0.12, scallop_pitch_um=0.35))",
        d_drie, None,
    ))

    # --- 熱酸化 ---
    def d_oxide():
        r = Recipe(config=cfg())
        r.add(CVD(material="nitride", thickness_um=0.2))
        r.add(Photo(mask=_center_mask(0.4), thickness_um=0.8, polarity="negative"))
        r.add(DryEtch(targets=["nitride"], depth_um=0.25))
        r.add(Strip(material="photoresist"))
        r.add(Oxidation(thickness_um=0.4, beak_fraction=0.8))
        return r.simulate()
    items.append((
        "oxidation", "熱酸化 (Oxidation)", "ドーピング・熱処理",
        "露出シリコンを熱酸化し SiO2 に変えます。SiO2 は元の Si より体積膨張する"
        "ため、約 45% の Si を消費しつつ上方へ成長します。窒化膜マスク端では"
        "酸化膜が横に侵入する LOCOS の『バーズビーク』を再現します。",
        [("thickness_um", "酸化膜厚[µm]"), ("consume_fraction", "Si 消費比(≈0.45)"),
         ("beak_fraction", "バーズビーク横侵入比"), ("time_min/temperature_c", "Deal-Grove モード")],
        "from semisim.processes import Oxidation\n"
        "r.add(Oxidation(thickness_um=0.4, beak_fraction=0.8))",
        d_oxide, None,
    ))

    # --- イオン注入 ---
    def d_implant():
        r = Recipe(config=cfg())
        r.add(Photo(mask=_center_mask(0.4), thickness_um=1.0, polarity="negative"))
        r.add(Implant(dopant="doped_n", range_um=0.6, straggle_um=0.18))
        r.add(Strip(material="photoresist"))
        return r.simulate()
    items.append((
        "implant", "イオン注入 (Implant)", "ドーピング・熱処理",
        "ドーパントイオンを加速して打ち込みます。飛程 Rp を中心にガウス分布"
        "（ストラグル）で濃度がピークを持ち、レジスト開口部だけがドープされます。"
        "下図は n 型注入層（doped_n）の埋め込み分布です。",
        [("dopant", "ドーパント(doped_n/doped_p)"), ("range_um", "飛程 Rp[µm]"),
         ("straggle_um", "縦方向ストラグル ΔRp[µm]"), ("tilt_deg", "注入チルト角")],
        "from semisim.processes import Implant\n"
        "r.add(Implant(dopant='doped_n', range_um=0.6, straggle_um=0.18))",
        d_implant, None,
    ))

    # --- 拡散／アニール ---
    def d_diffusion():
        r = Recipe(config=cfg())
        r.add(Photo(mask=_center_mask(0.35), thickness_um=1.0, polarity="negative"))
        r.add(Implant(dopant="doped_p", range_um=0.3, straggle_um=0.1))
        r.add(Strip(material="photoresist"))
        r.add(Anneal(depth_um=0.5, time_min=60, temperature_c=1050))
        return r.simulate()
    items.append((
        "anneal", "アニール / 拡散 (Anneal)", "ドーピング・熱処理",
        "高温熱処理でドーパントを活性化し、濃度勾配に従って拡散させます（ドライブイン）。"
        "注入直後の急峻な分布が、時間 √(D·t) に比例して縦横へ広がります。",
        [("depth_um", "基準ドライブイン深さ[µm]"), ("time_min", "アニール時間[分]"),
         ("temperature_c", "温度[℃]")],
        "from semisim.processes import Anneal\n"
        "r.add(Anneal(depth_um=0.5, time_min=60, temperature_c=1050))",
        d_diffusion, None,
    ))

    # --- エピタキシ ---
    def d_epi():
        r = _stepped(material="oxide", thick=0.4, mask_w=0.4, etch=0.45)
        r.add(Epitaxy(material="epi_si", thickness_um=0.6))
        return r.simulate()
    items.append((
        "epitaxy", "エピタキシャル成長 (Epitaxy)", "成膜",
        "単結晶を下地の結晶情報を引き継いで成長させます。酸化膜上には成長せず、"
        "露出シリコン上にのみ選択的に成長する選択エピを再現します（SiGe ソース"
        "ドレイン等）。",
        [("material", "エピ材料(epi_si 等)"), ("thickness_um", "成長厚[µm]"),
         ("selective", "選択成長フラグ")],
        "from semisim.processes import Epitaxy\n"
        "r.add(Epitaxy(material='epi_si', thickness_um=0.6))",
        d_epi, None,
    ))

    # --- ダマシン Cu + CMP ディッシング（目玉） ---
    def d_damascene():
        r = Recipe(config=cfg(nx=240, nz=160, sub=2.0))
        r.add(CVD(material="oxide", thickness_um=1.0))
        mask = Mask(shapes=[Shape("rect", {"x0": 0.15, "y0": 0.0, "x1": 0.45, "y1": 1.0}),
                            Shape("rect", {"x0": 0.6, "y0": 0.0, "x1": 0.85, "y1": 1.0})])
        r.add(Photo(mask=mask, thickness_um=1.0, polarity="positive"))
        r.add(DryEtch(targets=["oxide"], depth_um=0.8))
        r.add(Strip(material="photoresist"))
        r.add(PVD(material="tin", thickness_um=0.05))
        r.add(Fill(material="metal_cu", overfill_um=0.4))
        r.add(CMP(remove_um=0.5, stop_material="oxide",
                  soft_material="metal_cu", dishing_um=0.25, dishing_width_um=0.3))
        return r.simulate()
    items.append((
        "damascene", "ダマシン配線 + CMP ディッシング (Fill / CMP)", "平坦化・配線",
        "絶縁膜にトレンチを掘り、バリア(TiN)→Cu 充填→CMP 平坦化する銅配線形成です。"
        "CMP では軟らかい Cu が過研磨で<strong>皿状に凹むディッシング</strong>が起き、"
        "<strong>中央ほど深く・縁（絶縁膜際）ほど浅い凹面</strong>になり、"
        "<strong>配線幅が広いほど深く凹みます</strong>（左の太線が右の細線より深い）。"
        "実機の幅依存ディッシングを距離変換ベースの飽和プロファイルで再現しています。",
        [("Fill.material", "充填材料(metal_cu)"), ("Fill.overfill_um", "オーバーフィル量"),
         ("CMP.dishing_um", "最大ディッシング深さ[µm]"),
         ("CMP.dishing_width_um", "幅依存の特性幅[µm]"),
         ("CMP.erosion_um", "パターン密度エロージョン")],
        "from semisim.processes import Fill, CMP\n"
        "r.add(Fill(material='metal_cu', overfill_um=0.4))\n"
        "r.add(CMP(remove_um=0.5, stop_material='oxide',\n"
        "          soft_material='metal_cu', dishing_um=0.25, dishing_width_um=0.3))",
        d_damascene, (1.7, 3.1),
    ))

    # --- リフトオフ ---
    def d_liftoff():
        r = Recipe(config=cfg())
        r.add(Photo(mask=_stripe_mask(0.6, 0.3), thickness_um=0.7, polarity="negative"))
        r.add(PVD(material="metal_al", thickness_um=0.3, step_coverage=0.6))
        r.add(LiftOff())
        return r.simulate()
    items.append((
        "liftoff", "リフトオフ (LiftOff)", "平坦化・配線",
        "レジストパターン上に金属を成膜し、レジストを溶解除去すると、レジスト上の"
        "金属も一緒に剥がれ、開口部の金属だけが残ります。エッチしにくい金属の"
        "微細パターン形成に使います。",
        [("(パラメータなし)", "直前の PVD 膜とレジストから自動処理")],
        "from semisim.processes import LiftOff\n"
        "r.add(LiftOff())",
        d_liftoff, None,
    ))

    return items


def defect_section() -> tuple[str, str]:
    """不良モード検証デモ（ボイド／ピンホール）画像 + 説明 HTML を返す。"""
    # ボイド: 狭いトレンチを段差被覆の悪い PVD で塞いで埋め込みボイドを作る
    r = Recipe(config=cfg(nx=200, nz=160))
    r.add(CVD(material="oxide", thickness_um=1.0))
    r.add(Photo(mask=_stripe_mask(0.5, 0.2), thickness_um=0.8, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.8))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="metal_al", thickness_um=0.4, step_coverage=0.3, overhang=1.5))
    w = r.simulate()
    render(w, "void: 段差被覆不良で塞がれた埋め込みボイド", "defect_void.png", ylim=(1.7, 3.2))
    vm = metrology.void_metrics(w)
    rep = metrology.defect_report(w)
    has_void = "検出" if vm["count"] > 0 else "なし"
    body = (
        "<p>本シミュレータは主要な半導体不良モードを計測 (metrology) で検証できます。"
        "下図は段差被覆の悪い PVD でトレンチ上端が先に塞がり、内部に空洞が残った"
        f"<strong>埋め込みボイド</strong>の例です（void_metrics 個数={vm['count']} → {has_void}）。</p>"
        "<table class='param'><tr><th>不良モード</th><th>計測関数</th></tr>"
        "<tr><td>ボイド（充填不良）</td><td><code>void_metrics</code></td></tr>"
        "<tr><td>エッチ残渣・ストリンガー</td><td><code>etch_residue_metrics</code></td></tr>"
        "<tr><td>アンダーカット</td><td><code>undercut_um</code></td></tr>"
        "<tr><td>ピンホール（貫通欠陥）</td><td><code>pinhole_metrics</code></td></tr>"
        "<tr><td>オープン（断線）</td><td><code>electrical_continuity</code> / <code>line_resistance_ohm</code></td></tr>"
        "<tr><td>ショート</td><td><code>min_spacing_um</code></td></tr>"
        "<tr><td>ディッシング・エロージョン</td><td><code>dishing_depth_um</code></td></tr>"
        "<tr><td>ウェハ反り（膜応力）</td><td><code>wafer_bow_um</code></td></tr>"
        "<tr><td>一括検査</td><td><code>defect_report</code></td></tr>"
        "</table>"
        "<p><code>metrology.defect_report(wafer)</code> でこれらを横断検査した機械可読な"
        "辞書が得られ、CLI の <code>--json-report</code> 出力にも <code>defects</code> として"
        f"含まれます。本例の検出材料数: {len(rep['per_material'])}。</p>"
        "<pre><code>from semisim import metrology\n"
        "rep = metrology.defect_report(wafer)\n"
        "print(rep['voids'], rep['per_material'])</code></pre>"
        "<img src='img/defect_void.png' alt='void'>"
    )
    return "不良モード検証 (metrology)", body


PAGE_TMPL = """<!DOCTYPE html>
<html lang="ja">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>semisim 半導体プロセス 3D シミュレータ 使い方説明書</title>
<style>
 :root {{ --fg:#1c2530; --muted:#5a6b7b; --line:#dce3ea; --accent:#2d6cdf; --bg:#f6f8fb; }}
 * {{ box-sizing:border-box; }}
 body {{ font-family:"Segoe UI","Hiragino Kaku Gothic ProN","Meiryo",sans-serif;
   margin:0; color:var(--fg); background:var(--bg); line-height:1.7; }}
 header {{ background:linear-gradient(120deg,#1f3b6e,#2d6cdf); color:#fff; padding:32px 24px; }}
 header h1 {{ margin:0 0 6px; font-size:26px; }}
 header p {{ margin:0; opacity:.9; }}
 .layout {{ display:flex; max-width:1180px; margin:0 auto; }}
 nav {{ width:240px; flex:0 0 240px; padding:24px 16px; position:sticky; top:0;
   align-self:flex-start; max-height:100vh; overflow:auto; }}
 nav h2 {{ font-size:13px; color:var(--muted); text-transform:uppercase; letter-spacing:.05em; }}
 nav a {{ display:block; color:var(--fg); text-decoration:none; padding:4px 8px;
   border-radius:6px; font-size:14px; }}
 nav a:hover {{ background:#e7eef9; color:var(--accent); }}
 nav .cat {{ margin-top:14px; font-weight:700; font-size:13px; color:var(--muted); }}
 main {{ flex:1; padding:24px; min-width:0; }}
 section {{ background:#fff; border:1px solid var(--line); border-radius:12px;
   padding:22px 24px; margin-bottom:22px; }}
 section h2 {{ margin-top:0; font-size:20px; border-bottom:2px solid var(--accent);
   padding-bottom:8px; display:inline-block; }}
 .cat-badge {{ display:inline-block; background:#e7eef9; color:var(--accent);
   font-size:12px; padding:2px 10px; border-radius:999px; margin-left:8px; vertical-align:middle; }}
 img {{ max-width:100%; border:1px solid var(--line); border-radius:8px; margin-top:12px; background:#fff; }}
 table.param {{ border-collapse:collapse; margin:14px 0; width:100%; }}
 table.param th, table.param td {{ border:1px solid var(--line); padding:6px 10px;
   text-align:left; font-size:14px; }}
 table.param th {{ background:#f0f4f9; }}
 pre {{ background:#0f1b2d; color:#dbe7f5; padding:14px 16px; border-radius:8px;
   overflow:auto; font-size:13px; }}
 code {{ font-family:"Cascadia Code","Consolas",monospace; }}
 p code, td code {{ background:#eef2f7; color:#244; padding:1px 5px; border-radius:4px; }}
 .intro code {{ background:#eef2f7; }}
 footer {{ text-align:center; color:var(--muted); padding:24px; font-size:13px; }}
</style>
</head>
<body>
<header>
 <h1>semisim — 半導体プロセス 3D シミュレータ 使い方説明書</h1>
 <p>各プロセス操作の断面スクリーンショットと説明・パラメータ・コード例</p>
</header>
<div class="layout">
<nav>
 <h2>目次</h2>
 <a href="#intro">はじめに / 起動方法</a>
 {nav}
 <a href="#defect">不良モード検証</a>
 <a href="#output">出力・レポート</a>
</nav>
<main>
 <section id="intro" class="intro">
  <h2>はじめに / 起動方法</h2>
  <p><strong>semisim</strong> は半導体の前工程（成膜・リソグラフィ・エッチング・
  ドーピング・平坦化など）を 3D ボクセル格子上で再現し、断面や 3D 形状を可視化、
  さらに各種メトロロジで不良モードを定量検証できるシミュレータです。</p>
  <table class="param">
   <tr><th>用途</th><th>コマンド</th></tr>
   <tr><td>GUI 起動（対話的にレシピ作成・3D 表示）</td><td><code>py main.py</code></td></tr>
   <tr><td>CLI でレシピ実行＋断面 PNG 出力</td><td><code>py -m semisim.cli recipe.json --slice out.png</code></td></tr>
   <tr><td>JSON メトロロジ／不良レポート出力</td><td><code>py -m semisim.cli recipe.json --json-report report.json</code></td></tr>
   <tr><td>Python API でレシピ構築</td><td><code>Recipe(config=...).add(Process(...)).simulate()</code></td></tr>
  </table>
  <p>以下、代表的なプロセス操作ごとに、実際の出力断面（中央 Y 断面 = XZ 面、"
  横が x・縦が z で下が基板）と説明を示します。</p>
 </section>
 {sections}
 <section id="defect">
  <h2>{defect_title}<span class="cat-badge">検証</span></h2>
  {defect_body}
 </section>
 <section id="output">
  <h2>出力・レポート<span class="cat-badge">出力</span></h2>
  <p>シミュレーション結果は以下の形式で出力できます。</p>
  <table class="param">
   <tr><th>出力</th><th>内容</th></tr>
   <tr><td>断面 PNG (<code>--slice</code>)</td><td>任意の XZ/YZ/XY 断面画像</td></tr>
   <tr><td>3D STL (<code>--stl</code>)</td><td>材料ごとのソリッドメッシュ（CAD/3D 表示用）</td></tr>
   <tr><td>テキストレポート (<code>--report</code>)</td><td>膜厚・段差・電気特性・DRC など人間可読サマリ</td></tr>
   <tr><td>JSON レポート (<code>--json-report</code>)</td><td>メトロロジ + <code>electrical</code> + <code>defects</code> の機械可読辞書</td></tr>
  </table>
  <p>GUI（<code>py main.py</code>）では、レシピを 1 ステップずつ追加しながら 3D"
  形状をリアルタイムに確認でき、任意断面のスライス表示やメトロロジ値の参照も可能です。</p>
 </section>
 <footer>semisim usage manual — 自動生成 (tools/build_manual.py)</footer>
</main>
</div>
</body>
</html>
"""


def main() -> None:
    print(f"出力先: {OUT_DIR}")
    os.makedirs(IMG_DIR, exist_ok=True)
    items = demos()
    nav_parts = []
    section_parts = []
    last_cat = None
    for key, title, cat, desc, params, code, builder, ylim in items:
        wafer = builder()
        render(wafer, title, f"{key}.png", ylim=ylim)
        print(f"  生成: img/{key}.png")
        if cat != last_cat:
            nav_parts.append(f'<div class="cat">{html.escape(cat)}</div>')
            last_cat = cat
        nav_parts.append(f'<a href="#{key}">{html.escape(title)}</a>')
        rows = "".join(
            f"<tr><td><code>{html.escape(n)}</code></td><td>{html.escape(d)}</td></tr>"
            for n, d in params
        )
        section_parts.append(f"""
 <section id="{key}">
  <h2>{html.escape(title)}<span class="cat-badge">{html.escape(cat)}</span></h2>
  <p>{desc}</p>
  <table class="param"><tr><th>主なパラメータ</th><th>説明</th></tr>{rows}</table>
  <pre><code>{html.escape(code)}</code></pre>
  <img src="img/{key}.png" alt="{html.escape(title)}">
 </section>""")

    defect_title, defect_body = defect_section()
    print("  生成: img/defect_void.png")

    page = PAGE_TMPL.format(
        nav="\n ".join(nav_parts),
        sections="\n".join(section_parts),
        defect_title=html.escape(defect_title),
        defect_body=defect_body,
    )
    out_html = os.path.join(OUT_DIR, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"HTML 生成完了: {os.path.relpath(out_html)}")


if __name__ == "__main__":
    main()
