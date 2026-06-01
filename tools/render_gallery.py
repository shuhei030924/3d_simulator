"""断面ギャラリー生成ツール（ヘッドレス）。

各種プロセスを実行したウェハの 2D 断面を matplotlib(Agg) で PNG 出力する。
GUI/PyVista 不要。視覚的な動作確認や README 用の図に使う。

使い方:
    py tools/render_gallery.py
出力先:
    docs/gallery/*.png
"""
from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")  # ヘッドレス描画
import matplotlib.patches as mpatches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from semisim import materials, visualize  # noqa: E402
from semisim.grid import WaferConfig  # noqa: E402
from semisim.masks import Mask, Shape  # noqa: E402
from semisim.processes import (  # noqa: E402
    CMP,
    CVD,
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    AtomicLayerEtch,
    Diffusion,
    DryEtch,
    Epitaxy,
    Fill,
    Implant,
    LiftOff,
    Photo,
    Strip,
)
from semisim.recipe import Recipe

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "gallery")


def _center_mask(w=0.4) -> Mask:
    a = (1.0 - w) / 2.0
    b = a + w
    return Mask(shapes=[Shape("rect", {"x0": a, "y0": a, "x1": b, "y1": b})])


def _stripe_mask(period=0.3, width=0.15) -> Mask:
    # angle=90 でラインが y 方向に伸び、x 方向に周期変化する
    # （中央 Y 断面でライン&スペースが見える）
    return Mask(shapes=[Shape("grating", {"angle": 90.0, "period": period, "width": width})])


def cfg() -> WaferConfig:
    # 高解像度（0.05µm/vox）で斜め側壁の階段状ジャギーを抑える。
    # 高解像度（0.025µm/格子）で階段状（カクカク）を細かくし、滑らかな断面にする
    return WaferConfig(nx=320, ny=320, nz=360, pitch_um=0.025, substrate_um=3.0)


def recipe_implant() -> Recipe:
    r = Recipe(config=cfg())
    r.add(Photo(mask=_center_mask(0.4), thickness_um=1.0, polarity="negative"))
    r.add(Implant(dopant="doped_n", range_um=0.8, straggle_um=0.2))
    r.add(Strip(material="photoresist"))
    return r


def recipe_anneal() -> Recipe:
    r = Recipe(config=cfg())
    r.add(Photo(mask=_center_mask(0.3), thickness_um=1.0, polarity="positive"))
    r.add(Diffusion(dopant="doped_p", depth_um=0.6))
    r.add(Strip(material="photoresist"))
    r.add(Anneal(depth_um=0.6))
    return r


def recipe_epitaxy() -> Recipe:
    r = Recipe(config=cfg())
    r.add(CVD(material="oxide", thickness_um=0.4))
    r.add(Photo(mask=_center_mask(0.4), thickness_um=1.0, polarity="positive"))
    r.add(DryEtch(targets=["oxide"], depth_um=0.5))
    r.add(Strip(material="photoresist"))
    r.add(Epitaxy(material="epi_si", thickness_um=0.8))
    return r


def recipe_koh() -> Recipe:
    r = Recipe(config=cfg())
    r.add(CVD(material="nitride", thickness_um=0.3))
    r.add(Photo(mask=_center_mask(0.5), thickness_um=1.0, polarity="positive"))
    r.add(DryEtch(targets=["nitride"], depth_um=0.4))
    r.add(Strip(material="photoresist"))
    r.add(AnisoWetEtch(target="silicon", depth_um=2.0, sidewall_angle_deg=54.7))
    return r


def recipe_drie() -> Recipe:
    r = Recipe(config=cfg())
    r.add(Photo(mask=_stripe_mask(0.3, 0.15), thickness_um=1.2, polarity="positive"))
    r.add(DRIE(target="silicon", depth_um=3.0, scallop_um=0.15, scallop_pitch_um=0.4))
    r.add(Strip(material="photoresist"))
    return r


def recipe_damascene() -> Recipe:
    r = Recipe(config=cfg())
    r.add(CVD(material="low_k", thickness_um=1.2))
    r.add(Photo(mask=_stripe_mask(0.35, 0.18), thickness_um=1.2, polarity="positive"))
    r.add(DryEtch(targets=["low_k"], depth_um=1.0))
    r.add(Strip(material="photoresist"))
    r.add(PVD(material="tin", thickness_um=0.1))
    r.add(Fill(material="metal_cu", overfill_um=0.3))
    r.add(CMP(remove_um=0.5))
    return r


def recipe_liftoff() -> Recipe:
    r = Recipe(config=cfg())
    r.add(Photo(mask=_stripe_mask(0.35, 0.18), thickness_um=0.8, polarity="negative"))
    r.add(PVD(material="metal_al", thickness_um=0.4, step_coverage=0.7))
    r.add(LiftOff())
    return r


def recipe_mosfet() -> Recipe:
    """簡易 MOSFET 風フロー（複合）。"""
    r = Recipe(config=cfg())
    r.add(CVD(material="oxide", thickness_um=0.2))        # ゲート酸化
    r.add(CVD(material="metal_al", thickness_um=0.4))     # ゲート金属
    r.add(Photo(mask=_center_mask(0.3), thickness_um=1.0, polarity="negative"))
    r.add(DryEtch(targets=["metal_al", "oxide"], depth_um=0.7))
    r.add(Strip(material="photoresist"))
    r.add(Implant(dopant="doped_n", range_um=0.4, straggle_um=0.12))  # ソース/ドレイン
    r.add(Anneal(depth_um=0.3))
    return r


def recipe_ale() -> Recipe:
    """ALE による nm 精度の自己制限・高選択リセス（等方成分でマスク下を後退）。"""
    r = Recipe(config=cfg())
    r.add(CVD(material="oxide", thickness_um=1.5))
    r.add(Photo(mask=_stripe_mask(0.4, 0.2), thickness_um=1.0, polarity="positive"))
    # cycles×epc=0.8µm を精密除去。anisotropy=0.3 で側壁に等方アンダーカット。
    r.add(AtomicLayerEtch(targets=["oxide"], cycles=400, etch_per_cycle_nm=2.0,
                          anisotropy=0.3))
    r.add(Strip(material="photoresist"))
    return r


GALLERY = {
    "implant_buried_layer": recipe_implant,
    "ale_recess": recipe_ale,
    "anneal_drivein": recipe_anneal,
    "epitaxy_selective": recipe_epitaxy,
    "koh_vgroove": recipe_koh,
    "drie_scallop": recipe_drie,
    "damascene_cu": recipe_damascene,
    "liftoff_lines": recipe_liftoff,
    "mosfet_flow": recipe_mosfet,
}


def render(name: str, recipe: Recipe) -> str:
    wafer = recipe.simulate()
    # 中央 Y 断面（XZ 面）
    plane, width_um, height_um = visualize.slice_2d(wafer, "Y", wafer.config.ny // 2)
    cmap, norm = visualize.material_listed_cmap()

    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    # ボクセルデータをそのまま忠実に描画する（補間しない）。
    # 角は正しく直角を保ち、斜め側壁は格子解像度（0.05µm）の細かい階段になる。
    # ※ガウス平滑化は矩形トレンチの角まで丸めて物理的に誤った形状になるため使わない。
    ax.imshow(
        plane,
        origin="lower",
        cmap=cmap,
        norm=norm,
        extent=[0, width_um, 0, height_um],
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("z (µm)")

    # 凡例（断面に現れる材料のみ）
    present = sorted(set(int(v) for v in np.unique(plane)) - {materials.AIR})
    handles = []
    for mid in present:
        m = materials.BY_ID.get(mid)
        if m is None:
            continue
        handles.append(mpatches.Patch(color=m.color, label=m.name))
    if handles:
        ax.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.85)

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"{name}.png")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> None:
    print(f"出力先: {OUT_DIR}")
    for name, builder in GALLERY.items():
        path = render(name, builder())
        print(f"  生成: {os.path.relpath(path)}")
    print("ギャラリー生成完了")


if __name__ == "__main__":
    main()
