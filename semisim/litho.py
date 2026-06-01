"""リソグラフィの空間像（aerial image）モデルとプロセスウィンドウ解析。

ボクセル形状とは独立に、1 次元のマスク透過プロファイルから光学的な空間像を
求め、レジストしきい値モデルで印刷される寸法（CD）を計算する純粋関数群。
焦点・露光量に対する CD の応答（Bossung）、プロセスウィンドウ（DOF/露光裕度）、
エッジ配置誤差（EPE）、マスク誤差増幅係数（MEEF）といった検証ができる。

物理モデル（ポジ型レジスト・孤立スペースを想定）:
  - 空間像 I(x) = マスク透過 ⊛ ガウス PSF（光学解像度有限のボケ）
  - 焦点はずれで PSF が広がる: σ(f) = √(σ0² + (k·f)²)
  - 露光量 D を掛けた E(x)=D·I(x) がしきい値 threshold 以上でレジストが現像除去
  - 印刷 CD = 現像除去された連続領域の幅
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

# 既定の光学パラメータ（代表値）。利用側で上書き可能。
# threshold=0.5 は孤立エッジで空間像が 0.5 を横切る位置=マスクエッジに一致するため、
# 解像可能なフィーチャではベストフォーカス・公称露光量で CD バイアスがほぼ 0 になる。
DEFAULT_SIGMA0_UM = 0.03   # ベストフォーカスでの PSF σ（解像度）
DEFAULT_FOCUS_COEF = 0.5    # 焦点はずれ係数 k（σ 増分 = k·|focus|）
DEFAULT_THRESHOLD = 0.5     # レジスト現像しきい値（規格化空間像×dose）


def _sigma_px(sigma0_um: float, focus_um: float, focus_coef: float, pitch_um: float) -> float:
    """焦点はずれを含む実効 PSF σ をピクセル単位で返す。"""
    s_um = float(np.hypot(sigma0_um, focus_coef * focus_um))
    return max(s_um / pitch_um, 1e-6)


def isolated_space_mask(width_um: float, pitch_um: float, pad_um: float = 0.6) -> np.ndarray:
    """中央に幅 width_um の開口（透過=1）を持つ 1D マスクを生成する。

    周囲は遮光（0）。pad_um は両側の余白。CD/EPE/MEEF 計算の入力に使う。
    """
    w = max(1, int(round(width_um / pitch_um)))
    pad = max(1, int(round(pad_um / pitch_um)))
    mask = np.zeros(2 * pad + w, dtype=float)
    mask[pad:pad + w] = 1.0
    return mask


def aerial_image(mask: np.ndarray, sigma_px: float) -> np.ndarray:
    """マスク透過プロファイルの空間像（規格化光強度）を返す。"""
    return gaussian_filter1d(np.asarray(mask, dtype=float), sigma_px, mode="nearest")


def printed_cd_um(
    mask: np.ndarray,
    pitch_um: float,
    dose: float = 1.0,
    focus_um: float = 0.0,
    *,
    sigma0_um: float = DEFAULT_SIGMA0_UM,
    focus_coef: float = DEFAULT_FOCUS_COEF,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """マスクから印刷される中央フィーチャの CD（µm）を返す。

    空間像に露光量 dose を掛けた値が threshold を横切る位置をサブピクセル線形
    補間で求め、中央フィーチャの左右エッジ間隔を CD とする（ポジ型・孤立スペース）。
    中央が露光不足（threshold 未満）で開かない場合は 0。dose を上げる/threshold を
    下げると CD は広がる。焦点はずれ（|focus|↑）でコントラストが落ち CD がずれる。
    サブピクセル補間により CD はピッチに量子化されず連続的に変化する。
    """
    val = aerial_image(mask, _sigma_px(sigma0_um, focus_um, focus_coef, pitch_um)) * dose
    n = len(val)
    c = n // 2
    if val[c] < threshold:
        return 0.0
    # 中央から左右へ threshold 以上の連続範囲を求める
    left = c
    while left > 0 and val[left - 1] >= threshold:
        left -= 1
    right = c
    while right < n - 1 and val[right + 1] >= threshold:
        right += 1
    # サブピクセルのエッジ位置（threshold クロッシングを線形補間）
    if left > 0:
        left_edge = left - (val[left] - threshold) / (val[left] - val[left - 1])
    else:
        left_edge = 0.0
    if right < n - 1:
        right_edge = right + (val[right] - threshold) / (val[right] - val[right + 1])
    else:
        right_edge = float(n - 1)
    return float((right_edge - left_edge) * pitch_um)


def bossung(
    mask: np.ndarray,
    pitch_um: float,
    doses: np.ndarray,
    focuses_um: np.ndarray,
    *,
    sigma0_um: float = DEFAULT_SIGMA0_UM,
    focus_coef: float = DEFAULT_FOCUS_COEF,
    threshold: float = DEFAULT_THRESHOLD,
) -> np.ndarray:
    """焦点×露光量の格子上で CD を計算した行列を返す（Bossung 曲線群）。

    返り値 shape=(len(focuses), len(doses))、要素は CD（µm）。
    """
    out = np.zeros((len(focuses_um), len(doses)), dtype=float)
    for i, f in enumerate(focuses_um):
        for j, d in enumerate(doses):
            out[i, j] = printed_cd_um(
                mask, pitch_um, dose=float(d), focus_um=float(f),
                sigma0_um=sigma0_um, focus_coef=focus_coef, threshold=threshold)
    return out


def process_window(
    cd_matrix: np.ndarray,
    doses: np.ndarray,
    focuses_um: np.ndarray,
    target_cd_um: float,
    tol_pct: float = 10.0,
) -> dict:
    """Bossung 行列から CD 規格内（target±tol%）のプロセスウィンドウを評価する。

    返り値:
      - in_spec: bool 行列（各 focus×dose 点が規格内か）
      - dof_um: 規格内になる dose が 1 つでも存在する焦点範囲（被写界深度 DOF）
      - exposure_latitude_pct: ベストフォーカス付近で規格内になる dose 範囲（%）
      - area_frac: 全格子点に対する規格内点の割合
    """
    lo = target_cd_um * (1 - tol_pct / 100.0)
    hi = target_cd_um * (1 + tol_pct / 100.0)
    in_spec = (cd_matrix >= lo) & (cd_matrix <= hi)
    # DOF: 規格内の dose が存在する焦点の範囲
    foc_ok = in_spec.any(axis=1)
    if foc_ok.any():
        fvals = np.asarray(focuses_um)[foc_ok]
        dof = float(fvals.max() - fvals.min())
    else:
        dof = 0.0
    # 露光裕度: ベストフォーカス（|focus| 最小）の行で規格内 dose の幅
    best = int(np.argmin(np.abs(np.asarray(focuses_um))))
    drow = in_spec[best]
    if drow.any():
        dvals = np.asarray(doses)[drow]
        el = float((dvals.max() - dvals.min()) / np.median(doses) * 100.0)
    else:
        el = 0.0
    return {
        "in_spec": in_spec,
        "dof_um": dof,
        "exposure_latitude_pct": el,
        "area_frac": float(in_spec.mean()),
    }


def meef(
    target_width_um: float,
    pitch_um: float,
    delta_um: float = 0.01,
    *,
    dose: float = 1.0,
    focus_um: float = 0.0,
    sigma0_um: float = DEFAULT_SIGMA0_UM,
    focus_coef: float = DEFAULT_FOCUS_COEF,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """マスク誤差増幅係数 MEEF = dCD_wafer / dCD_mask を返す。

    マスク開口を ±delta_um 変化させたときのウェハ CD 変化率（中心差分）。
    1.0 で線形、>1 でマスク誤差が拡大して転写される（解像限界付近で増大）。
    """
    kw = dict(sigma0_um=sigma0_um, focus_coef=focus_coef, threshold=threshold)
    cd_plus = printed_cd_um(
        isolated_space_mask(target_width_um + delta_um, pitch_um), pitch_um,
        dose=dose, focus_um=focus_um, **kw)
    cd_minus = printed_cd_um(
        isolated_space_mask(target_width_um - delta_um, pitch_um), pitch_um,
        dose=dose, focus_um=focus_um, **kw)
    return float((cd_plus - cd_minus) / (2 * delta_um))


def edge_placement_error_um(
    target_width_um: float,
    pitch_um: float,
    *,
    dose: float = 1.0,
    focus_um: float = 0.0,
    sigma0_um: float = DEFAULT_SIGMA0_UM,
    focus_coef: float = DEFAULT_FOCUS_COEF,
    threshold: float = DEFAULT_THRESHOLD,
) -> float:
    """エッジ配置誤差 EPE（µm, 片側）= (印刷 CD − 目標 CD)/2 を返す。

    正で開口が目標より広い（エッジが外側）、負で狭い。露光量不足や焦点はずれで
    EPE が大きくなる。OPC 補正量の目安に使う。
    """
    cd = printed_cd_um(
        isolated_space_mask(target_width_um, pitch_um), pitch_um,
        dose=dose, focus_um=focus_um,
        sigma0_um=sigma0_um, focus_coef=focus_coef, threshold=threshold)
    return float((cd - target_width_um) / 2.0)
