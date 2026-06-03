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
    AtomicLayerEtch,
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


def cfg(nx=1600, ny=16, nz=1120, pitch=0.005, sub=2.0) -> WaferConfig:
    # pitch=0.005µm の超微細ボクセルで断面の曲線・斜面・スキャロップを
    # 滑らかに描く（角つきを極力抑える）。断面は XZ 面なので ny は小さく保つ。
    return WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=pitch, substrate_um=sub)


def _center_mask(w=0.4) -> Mask:
    a = (1.0 - w) / 2.0
    return Mask(shapes=[Shape("rect", {"x0": a, "y0": 0.0, "x1": a + w, "y1": 1.0})])


def _stripe_mask(period=0.4, width=0.2) -> Mask:
    return Mask(shapes=[Shape("grating", {"angle": 90.0, "period": period, "width": width})])


def _mandrel_mask(centers, w=0.10) -> Mask:
    """正規化 x 中心位置のリストに、幅 w の縦ライン（マンドレル）を置く。"""
    half = w / 2.0
    return Mask(shapes=[
        Shape("rect", {"x0": c - half, "y0": 0.0, "x1": c + half, "y1": 1.0})
        for c in centers
    ])


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

    # --- SADP（ピッチダブリング）---
    def d_sadp():
        r = Recipe(config=cfg())
        # 被パターン層（ポリシリコン）
        r.add(CVD(material="poly", thickness_um=0.30))
        # マンドレル（犠牲レジスト）をリソ解像度の緩いピッチで 3 本形成
        r.add(Photo(mask=_mandrel_mask([0.28, 0.5, 0.72], 0.10),
                    thickness_um=0.30, polarity="positive"))
        # コンフォーマル成膜＋エッチバックでマンドレル側壁にスペーサを残す
        r.add(Spacer(material="oxide", thickness_um=0.08, overetch_um=0.02))
        # マンドレル引き抜き → 自立スペーサ（1 マンドレルあたり 2 本＝ピッチ半減）
        r.add(Strip(material="photoresist"))
        # スペーサをハードマスクに被パターン層へ転写
        r.add(DryEtch(targets=["poly"], depth_um=0.32))
        return r.simulate()
    items.append((
        "sadp", "ピッチダブリング (SADP / 自己整合ダブルパターニング)", "微細化",
        "リソ解像限界より細かい配線ピッチを作る先端微細化プロセスです。"
        "①被パターン層の上に<strong>マンドレル（犠牲芯材）</strong>を緩いピッチで"
        "パターニング → ②コンフォーマル成膜＋異方性エッチバックで"
        "<strong>側壁スペーサ</strong>を残す → ③<strong>マンドレルを引き抜く</strong>と"
        "1 本のマンドレルから 2 本のスペーサが自立し<strong>ピッチが半分（線密度2倍）</strong>に"
        "なります → ④スペーサをハードマスクに下地へ転写。本例は 3 本のマンドレルから"
        "6 本のフィンを形成しています（2×）。スペーサ幅がそのまま最終線幅になるため、"
        "露光波長に依存しないサブリソ寸法が得られます。",
        [("マンドレル", "Photo で緩ピッチに形成"),
         ("Spacer.thickness_um", "スペーサ幅＝最終線幅[µm]"),
         ("Strip", "マンドレル引き抜き"),
         ("DryEtch.targets", "転写する下地層")],
        "from semisim.processes import CVD, Photo, Spacer, Strip, DryEtch\n"
        "r.add(CVD(material='poly', thickness_um=0.30))            # 被パターン層\n"
        "r.add(Photo(mask=mandrels, thickness_um=0.30))           # マンドレル\n"
        "r.add(Spacer(material='oxide', thickness_um=0.08))       # 側壁スペーサ\n"
        "r.add(Strip(material='photoresist'))                     # マンドレル引き抜き\n"
        "r.add(DryEtch(targets=['poly'], depth_um=0.32))          # 下地へ転写",
        d_sadp, (1.8, 2.8),
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

    # --- ALE（原子層エッチ） ---
    def d_ale():
        r = Recipe(config=cfg())
        r.add(CVD(material="oxide", thickness_um=0.8))
        r.add(Photo(mask=_stripe_mask(0.7, 0.3), thickness_um=0.8, polarity="positive"))
        r.add(AtomicLayerEtch(targets=["oxide"], cycles=300, etch_per_cycle_nm=1.5,
                              anisotropy=0.3))
        return r.simulate()
    items.append((
        "ale", "ALE 原子層エッチ (ALE)", "エッチング",
        "原子層エッチ。1 サイクルで原子層レベルの極薄量だけ自己制限的に除去し、"
        "cycles×etch_per_cycle_nm で nm 精度に深さを制御します。対象以外の材料で"
        "完全停止（高選択比）し、anisotropy で等方コンフォーマル〜純垂直を切替えます。"
        "ALD（成膜）の対となる先端ノード向けの高精度エッチです。",
        [("targets", "エッチ対象材料"), ("cycles", "サイクル数"),
         ("etch_per_cycle_nm", "1 サイクル除去量[nm]"), ("anisotropy", "0=等方/1=垂直")],
        "from semisim.processes import AtomicLayerEtch\n"
        "r.add(AtomicLayerEtch(targets=['oxide'], cycles=300,\n"
        "                      etch_per_cycle_nm=1.5, anisotropy=0.3))",
        d_ale, None,
    ))

    # --- 異方性ウェット（KOH） ---
    def d_koh():
        r = Recipe(config=cfg(nz=1280, sub=3.0))
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
        r = Recipe(config=cfg(nz=1440, sub=4.0))
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
        r = Recipe(config=cfg(nx=1920, nz=1280, sub=2.0))
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
    # ボイド: 高アスペクト比トレンチを等角性の悪い PVD で埋め、口元が先に
    # 塞がって内部にキーホール（ティアドロップ）状の空洞が残る様子を再現する。
    # 細かいボクセル(pitch=0.005µm)で側壁被覆と口元の絞り込みを滑らかに描く。
    pitch = 0.005
    width, depth = 0.5, 1.5
    span = 1200 * pitch
    frac = width / span
    r = Recipe(config=WaferConfig(nx=1200, ny=12, nz=960, pitch_um=pitch, substrate_um=2.0))
    r.add(CVD(material="oxide", thickness_um=depth + 0.3))
    r.add(Photo(
        mask=Mask(shapes=[Shape("rect", {"x0": (1 - frac) / 2, "y0": 0.0,
                                         "x1": (1 + frac) / 2, "y1": 1.0})]),
        thickness_um=0.6, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=depth))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="metal_cu", thickness_um=0.7, step_coverage=0.72, overhang=0.5))
    w = r.simulate()
    render(w, "キーホールボイド: 段差被覆不良で口元が先に塞がった埋め込み空洞",
           "defect_void.png", ylim=(1.9, 3.7))
    vm = metrology.void_metrics(w)
    rep = metrology.defect_report(w)
    has_void = "検出" if vm["count"] > 0 else "なし"
    body = (
        "<p>本シミュレータは主要な半導体不良モードを計測 (metrology) で検証できます。"
        "下図は段差被覆（ステップカバレッジ）の悪い PVD で高アスペクト比トレンチを"
        "埋めた例です。側壁より口元の堆積が速いため上端が先に閉じ、内部に上方ほど"
        "細る<strong>キーホール（ティアドロップ）状の埋め込みボイド</strong>が"
        f"残ります（void_metrics 個数={vm['count']} → {has_void}）。"
        "口元が塞がった瞬間に内部の空気は最上面との連結が切れ、以降フラックスが"
        "届かず空洞として凍結する——という実プロセスの封止機構をそのまま再現して"
        "います。</p>"
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


def verification_section() -> str:
    """電気・熱・信頼性の検証メトロロジの一覧 HTML を返す（数値例つき）。"""
    # 2 本の並走配線（Cu/W）+ Al 基準面を酸化膜中に作り、容量等を実測する。
    cfg2 = WaferConfig(nx=80, ny=40, nz=50, pitch_um=0.05, substrate_um=0.0)
    w = Recipe(config=cfg2).simulate()
    g = w.grid
    g[:] = materials.BY_NAME["oxide"].id
    g[5:10, :, :] = materials.BY_NAME["metal_al"].id
    g[25:33, 8:18, :] = materials.BY_NAME["metal_cu"].id
    g[25:33, 22:32, :] = materials.BY_NAME["tungsten"].id
    c_ll = metrology.parasitic_capacitance_ff(w, "metal_cu", "tungsten")
    c_field = metrology.parasitic_capacitance_field_ff(w, "metal_cu", "tungsten")
    em = metrology.electromigration_risk(w, "metal_cu", 1.0, "x")
    mttf = metrology.electromigration_mttf(em["j_max_a_cm2"], 110)

    rows = [
        ("寄生容量（平行平板）", "parasitic_capacitance_ff",
         f"{c_ll:.3f} fF（配線間カップリング, ε0·εr·A/d を厳密積算）"),
        ("寄生容量（静電界ソルバ）", "parasitic_capacitance_field_ff",
         f"{c_field:.3f} fF（∇·(εr∇φ)=0 を疎行列で解きフリンジ込み）"),
        ("RC 遅延", "rc_delay_ps", "τ=R·C による配線遅延"),
        ("分布 RC Elmore 遅延", "elmore_delay_ps", "τ=Σ(ΣR)·C。一様線で ½RC（分布効果）"),
        ("伝送線路パラメータ", "transmission_line_params",
         "Z0・インダクタンス・信号速度 v=c/√εr・伝搬遅延（高速配線 RLC）"),
        ("配線総合レポート", "interconnect_report",
         "R/C/L/Z0/遅延/電流密度/EM/IRドロップ/断線を1コールで集計"),
        ("ロジックゲート遅延", "gate_switching_delay_ps", "CV/I モデル τ=C·Vdd/I_drive"),
        ("MOS ゲート容量 / EOT", "mos_gate_capacitance",
         "Cox=ε0/Σ(tᵢ/εrᵢ), EOT。high-k で EOT 薄化"),
        ("MOS C-V 特性 / Vth", "mos_cv_curve / threshold_voltage_v",
         "空乏近似の高周波 C-V（蓄積→空乏→反転）としきい値電圧・空乏層幅"),
        ("短チャネル Vth / DIBL", "short_channel_vth_v",
         "Vth ロールオフ（SCE）+ ドレイン誘起障壁低下（DIBL）"),
        ("接合リーク電流", "junction_leakage_a",
         "逆リーク∝ni(T)²·面積。温度で指数加速（待機電力）"),
        ("DRAM リテンション時間", "dram_retention_time_s",
         "t_ret=C·ΔV/I_leak。容量＋接合リークでリフレッシュ周期を評価"),
        ("ソフトエラー臨界電荷", "critical_charge_fc",
         "Q_crit=C·V。SEU（ビット反転）耐性"),
        ("ダイオード I-V", "diode_current / diode_iv_curve",
         "Shockley 式。順方向指数・逆方向 −Is 飽和・直列抵抗で高電流飽和"),
        ("MOS 小信号特性", "mos_small_signal",
         "gm=∂Id/∂Vg・gds=∂Id/∂Vd・真性利得 Av=gm/gds（飽和域で高利得）"),
        ("MOS 遮断周波数 fT", "mos_cutoff_frequency",
         "fT=gm/(2π·Cgg)・トランジット時間 τ=Cgg/gm（RF/アナログ FoM）"),
        ("MOS 伝達特性 / SS", "mos_transfer_characteristics",
         "Id-Vg から SS=min(ΔVg/Δlog Id)≈n·60mV/dec・Ion・Ioff・Ion/Ioff 比"),
        ("MOS チャネル熱雑音", "mos_thermal_noise",
         "S_id=4kT·γ·gm・入力換算電圧雑音 √(4kTγ/gm)（gm↑で低雑音）"),
        ("MOS マッチング（Pelgrom）", "mos_mismatch",
         "σ(ΔVth)=A_VT/√(W·L)・σ(Δβ/β)=A_β/√(W·L)（大面積ほど高マッチング）"),
        ("pn 接合 空乏層容量", "junction_capacitance / junction_cv_curve",
         "ビルトイン電位・空乏層幅・接合容量。1/Cj²-V 直線（C-V プロファイリング）"),
        ("MOS I-V 特性", "mos_drain_current / mos_iv_curve",
         "Idsat∝(Vg−Vth)²・三極管/飽和・サブスレショルド傾斜 SS≈n·60mV/dec"),
        ("配線抵抗 / シート抵抗", "line_resistance_ohm / sheet_resistance_ohm_sq",
         "断面積を考慮した直列抵抗・薄膜評価"),
        ("Maxwell 容量行列", "capacitance_matrix_ff",
         "複数導体の自己容量・全結合容量を抽出（標準的 RC 抽出, 遮蔽も再現）"),
        ("対全導体総容量", "total_net_capacitance_ff",
         "対象を1V・他全導体を接地で解くドライバ実効負荷容量"),
        ("電源 IR ドロップ", "ir_drop_v",
         "ΔV=I·R による電源配線の電圧降下（パワーインテグリティ）"),
        ("温度依存抵抗（TCR）", "resistance_at_temperature",
         "R(T)=R₀(1+TCR·ΔT)。金属は高温で抵抗増（自己発熱と正帰還）"),
        ("電流密度 / EM リスク", "current_density_stats / electromigration_risk",
         f"J_max={em['j_max_a_cm2']:.2e} A/cm², 余裕={em['margin']:.2f}"),
        ("電流密度プロファイル", "current_density_profile",
         "配線に沿った J(x)。ネッキング=EM ホットスポット位置を可視化"),
        ("TLM 接触抵抗抽出", "tlm_extract",
         "R vs コンタクト間隔の直線回帰で Rc・シート抵抗・伝送長 Lt を抽出"),
        ("EM 寿命（Black）", "electromigration_mttf / em_lifetime_wafer",
         f"MTTF={mttf:.3e}（相対, MTTF=A·J⁻ⁿ·exp(Ea/kT)）"),
        ("EM Blech 不死条件", "blech_immortal",
         "j·L < (jL)_crit で EM 免疫（短配線は故障しない）"),
        ("自己発熱結合 EM 寿命", "em_lifetime_self_heated",
         "ジュール発熱の ΔT を接合温度に加え Black 式で評価（正帰還）"),
        ("HCI 寿命", "hci_lifetime",
         "ホットキャリア注入 TTF=A·exp(B/Vds)。ドレイン電圧律速の劣化"),
        ("NBTI しきい値劣化", "nbti_vth_shift",
         "|ΔVth|∝exp(γ|V|)·exp(−Ea/kT)·tⁿ。電圧/温度/時間で pMOS 経時劣化"),
        ("絶縁破壊 / TDDB 寿命", "dielectric_breakdown / tddb_lifetime",
         "E=V/g vs 破壊電界, E モデル寿命"),
        ("アンテナ比", "antenna_ratio", "プラズマ帯電損傷 DRC（収集面積/ゲート面積）"),
        ("縦方向熱抵抗", "thermal_resistance_k_w / thermal_resistance_map",
         "R=ΣΔz/(k·A) を列毎に算出し並列合成"),
        ("自己発熱 ΔT", "temperature_rise_k / joule_self_heating_k",
         "ΔT=P·Rth, ジュール発熱 P=I²R"),
        ("熱拡散ソルバ（温度分布）", "temperature_field_2d / peak_temperature_rise_k",
         "∇·(k∇T)=−q で横方向ヒートスプレッディングを解く"),
        ("完全 3D 熱拡散ソルバ", "temperature_field_3d / peak_temperature_rise_3d",
         "局所発熱を x・y・z の 3 方向に等方拡散（断面ソルバの上位）"),
        ("局所応力集中", "stress_concentration_map / max_stress_concentration",
         "界面 Δσ×幾何係数でクラック/剥離リスク箇所を検出"),
        ("ウェハ反り（膜応力）", "wafer_bow_um", "Stoney 則による等価反り"),
        ("歩留り推定", "yield_estimate", "Poisson/Murphy/Seeds モデル"),
        ("クリティカルエリア解析", "critical_area_short_um2 / caa_short_yield",
         "欠陥径ごとの臨界面積を距離変換で算出しサイズ分布で積分→ショート歩留り"),
        ("平坦化 DOF バジェット", "planarization_dof_check",
         "表面トポグラフィをリソ焦点深度と比較し焦点外れ面積率を判定"),
        ("ソルバ収束検証", "estimate_convergence_order",
         "容量・熱拡散ソルバが格子細分で解析解へ約1次収束することを定量検証"),
        ("リソ プロセスウィンドウ", "litho.bossung / process_window / meef / EPE",
         "空間像モデルで CD・DOF・露光裕度・MEEF・EPE を評価"),
        ("リソ CD 統計ばらつき", "litho.monte_carlo_cd",
         "露光/焦点変動のモンテカルロで CDU(3σ)・規格内歩留りを統計評価"),
    ]
    table = "".join(
        f"<tr><td>{html.escape(name)}</td><td><code>{html.escape(fn)}</code></td>"
        f"<td>{html.escape(note)}</td></tr>"
        for name, fn, note in rows
    )
    return (
        "<p>形状だけでなく、できあがったデバイス構造の<strong>電気・熱・機械・信頼性</strong>"
        "特性を計測 (metrology) と専用ソルバ（<code>semisim/litho.py</code> 含む）で検証できます。"
        "代表的な検証関数と、酸化膜中の並走 Cu/W 配線＋Al 基準面の構造に対する実測値の例を"
        "示します。</p>"
        "<table class='param'><tr><th>検証項目</th><th>関数</th><th>内容・数値例</th></tr>"
        f"{table}</table>"
        "<pre><code>from semisim import metrology, litho\n"
        "# 寄生容量と RC 遅延\n"
        "c = metrology.parasitic_capacitance_ff(wafer, 'metal_cu', 'tungsten')\n"
        "# 静電界ソルバ（フリンジ込み）\n"
        "c2 = metrology.parasitic_capacitance_field_ff(wafer, 'metal_cu', 'tungsten')\n"
        "# EM 寿命（最小断面の電流密度から）\n"
        "life = metrology.em_lifetime_wafer(wafer, 'metal_cu', current_ma=1.0, temperature_c=110)\n"
        "# 熱拡散による温度分布\n"
        "T = metrology.temperature_field_2d(wafer, source_mask, total_power_w=0.1)\n"
        "# リソ プロセスウィンドウ\n"
        "cd = litho.printed_cd_um(litho.isolated_space_mask(0.1, 0.005), 0.005)</code></pre>"
    )


def gui_section() -> str:
    """GUI 操作画面・設定ダイアログのスクリーンショット説明 HTML を返す。

    画像は tools/capture_gui.py が別途生成した docs/manual/img/gui_*.png を参照する
    （Qt と matplotlib のバックエンド競合を避けるため生成を分離している）。
    """
    return """
  <p><code>py main.py</code> で GUI を起動します。左にプロセスレシピと層構成、
  右に <strong>3D ビュー</strong>と<strong>2D 断面</strong>のタブが並ぶ操作画面です。
  「工程を追加」から工程を選ぶと、その工程専用のパラメータ入力ダイアログが開き、
  追加・編集するたびに 3D 形状と断面がその場で更新されます。</p>
  <h3>操作画面（メインウィンドウ）</h3>
  <img src="img/gui_main.png" alt="GUI メインウィンドウ操作画面">
  <table class="param">
   <tr><th>領域</th><th>役割</th></tr>
   <tr><td>プロセスレシピ（左上）</td><td>工程を上から順に実行。ドラッグ/▲▼ で並べ替え、編集・複製・削除が可能。</td></tr>
   <tr><td>工程を追加 / 編集 / 複製 / 削除</td><td>工程の追加と編集。各工程はパラメータ設定ダイアログで入力する。</td></tr>
   <tr><td>新規 / 保存 / 読込 / STL 出力</td><td>レシピ JSON の入出力と、3D 形状の STL エクスポート。</td></tr>
   <tr><td>プリセット / 最近のレシピ / 計測レポート</td><td>サンプルレシピの呼び出しと、メトロロジ検査レポートの表示。</td></tr>
   <tr><td>層構成（左下）</td><td>中央列の材料スタックと各層の膜厚・合計厚を一覧表示。</td></tr>
   <tr><td>3D ビュー / 2D 断面タブ（右）</td><td>立体形状の回転表示と、任意の XZ/YZ 断面表示。断面上 2 点クリックで距離計測。</td></tr>
  </table>
  <h3>ウェハ設定ダイアログ</h3>
  <p>「新規」からウェハ（ボクセル格子）の解像度と基板厚を設定します。X/Y/Z の
  ボクセル数とボクセル一辺の寸法（pitch）で実寸とメッシュ精度が決まります。</p>
  <img src="img/gui_dialog_wafer.png" alt="ウェハ設定ダイアログ">
  <h3>工程パラメータ設定ダイアログ（工程タイプごとに切替）</h3>
  <p>「工程を追加」で工程タイプを選ぶと、その工程に必要な入力欄だけが現れます。
  代表的な工程の設定画面を示します。</p>
  <div>
   <p><strong>フォトリソ (PHOTO)</strong>：レジスト厚・極性（ポジ/ネガ）・角丸め σ を
   設定し、マスク図形エディタで矩形・円・帯・周期ラインを配置します。図形が無ければ
   全面が対象になります。</p>
   <img src="img/gui_dialog_photo.png" alt="フォトリソ設定ダイアログ">
   <p><strong>CVD 成膜</strong>：材料と膜厚、ローディング効果や表面ラフネスを設定。
   等角性の高いコンフォーマル成膜を行います。</p>
   <img src="img/gui_dialog_cvd.png" alt="CVD 設定ダイアログ">
   <p><strong>PVD 成膜</strong>：材料・膜厚に加え、<code>段差被覆率</code>（低いほど窪み底に
   付きにくい）、<code>オーバーハング(庇)</code>、<code>斜め蒸着 入射角</code> を設定。
   被覆率を下げるとキーホールボイドが発生します。</p>
   <img src="img/gui_dialog_pvd.png" alt="PVD 設定ダイアログ">
   <p><strong>異方性ウェット (KOH)</strong>：結晶面依存エッチの対象・深さ・側壁角
   （Si(111) の 54.7°）を設定し、台形断面の異方性エッチを再現します。</p>
   <img src="img/gui_dialog_koh.png" alt="KOH 設定ダイアログ">
   <p><strong>イオン注入 (IMPLANT)</strong>：ドーパント種・飛程 Rp・ストラグル ΔRp・
   チルト角を設定し、ガウス分布の埋め込みドープ層を形成します。</p>
   <img src="img/gui_dialog_implant.png" alt="イオン注入 設定ダイアログ">
   <p><strong>CMP 平坦化</strong>：研磨対象・除去量を設定し、上面を平坦化します。
   ダマシン配線のディッシング/エロージョンも評価できます。</p>
   <img src="img/gui_dialog_cmp.png" alt="CMP 設定ダイアログ">
  </div>
"""


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
 <a href="#gui">GUI 操作・設定画面</a>
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
 <section id="gui">
  <h2>GUI 操作・設定画面<span class="cat-badge">操作</span></h2>
  {gui}
 </section>
 {sections}
 <section id="defect">
  <h2>{defect_title}<span class="cat-badge">検証</span></h2>
  {defect_body}
 </section>
 <section id="verification">
  <h2>電気・熱・信頼性の検証 (metrology)<span class="cat-badge">検証</span></h2>
  {verification_body}
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
        gui=gui_section(),
        defect_title=html.escape(defect_title),
        defect_body=defect_body,
        verification_body=verification_section(),
    )
    out_html = os.path.join(OUT_DIR, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"HTML 生成完了: {os.path.relpath(out_html)}")


if __name__ == "__main__":
    main()
