"""PyQt5 + PyVista による GUI 本体。

レシピ編集（工程の追加/編集/削除/並べ替え）、3D 表示、任意方向の断面、
保存/読込を提供する。
"""
from __future__ import annotations

import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from pyvistaqt import QtInteractor

from . import export, materials, metrology, presets, processes, visualize
from .grid import WaferConfig
from .masks import Mask, Shape
from .processes import (
    ALD,
    CMP,
    CVD,
    DRIE,
    PVD,
    AnisoWetEtch,
    Anneal,
    AtomicLayerEtch,
    Backgrind,
    Diffusion,
    DryEtch,
    Epitaxy,
    Fill,
    Implant,
    LiftOff,
    Oxidation,
    Photo,
    PlasmaClean,
    Process,
    RapidThermalAnneal,
    Reflow,
    Silicidation,
    Spacer,
    SpinCoat,
    SputterEtch,
    Strip,
    WetEtch,
)
from .recipe import Recipe
from .settings import AppSettings


# =============================================================================
# マスク編集ウィジェット（PHOTO 用）
# =============================================================================
class MaskEditor(QtWidgets.QGroupBox):
    """図形（矩形/円）の集合を編集する簡易エディタ。座標は 0..1。"""

    def __init__(self, mask: Mask, parent=None):
        super().__init__("マスク図形 (座標は 0..1)", parent)
        self.mask = Mask(shapes=list(mask.shapes), invert=mask.invert)
        lay = QtWidgets.QVBoxLayout(self)

        self.invert_cb = QtWidgets.QCheckBox("反転 (選択領域を入れ替える)")
        self.invert_cb.setChecked(self.mask.invert)
        lay.addWidget(self.invert_cb)

        # 図形リスト＋プレビューを横並びに
        mid = QtWidgets.QHBoxLayout()
        self.list = QtWidgets.QListWidget()
        self.list.setMaximumHeight(120)
        mid.addWidget(self.list, 1)
        self.preview = QtWidgets.QLabel()
        self.preview.setFixedSize(76, 76)
        self.preview.setToolTip("マスクのプレビュー（青=選択領域）")
        self.preview.setStyleSheet("border:1px solid #d4dce6; border-radius:6px;")
        mid.addWidget(self.preview, 0)
        lay.addLayout(mid)
        self.invert_cb.stateChanged.connect(self._refresh_list)
        self._refresh_list()

        btns = QtWidgets.QHBoxLayout()
        b_rect = QtWidgets.QPushButton("矩形")
        b_circ = QtWidgets.QPushButton("円")
        b_stripe = QtWidgets.QPushButton("帯")
        b_grating = QtWidgets.QPushButton("周期ライン")
        b_dup = QtWidgets.QPushButton("複製")
        b_dup.setToolTip("選択中の図形を複製する")
        b_del = QtWidgets.QPushButton("削除")
        b_rect.clicked.connect(self._add_rect)
        b_circ.clicked.connect(self._add_circle)
        b_stripe.clicked.connect(self._add_stripe)
        b_grating.clicked.connect(self._add_grating)
        b_dup.clicked.connect(self._duplicate)
        b_del.clicked.connect(self._delete)
        for b in (b_rect, b_circ, b_stripe, b_grating, b_dup, b_del):
            btns.addWidget(b)
        lay.addLayout(btns)

        hint = QtWidgets.QLabel("図形が無い場合は全面が対象になります。")
        hint.setStyleSheet("color: gray;")
        lay.addWidget(hint)

    def _refresh_list(self):
        self.list.clear()
        if not self.mask.shapes:
            # 空状態: 全面が対象になることをプレースホルダで明示
            placeholder = QtWidgets.QListWidgetItem("（図形なし — 全面が対象）")
            placeholder.setFlags(QtCore.Qt.NoItemFlags)
            placeholder.setForeground(QtGui.QColor("#9aa7b4"))
            self.list.addItem(placeholder)
        for s in self.mask.shapes:
            self.list.addItem(s.label())
        self._refresh_preview()

    def _refresh_preview(self):
        """現在のマスク（反転状態込み）の即時プレビューを更新する。"""
        if not hasattr(self, "preview"):
            return
        self.mask.invert = self.invert_cb.isChecked()
        size = 72
        rgb = self.mask.preview_rgb(size)
        rgb = np.ascontiguousarray(rgb)
        img = QtGui.QImage(rgb.data, size, size, 3 * size,
                           QtGui.QImage.Format_RGB888)
        self.preview.setPixmap(QtGui.QPixmap.fromImage(img.copy()))

    def _add_rect(self):
        dlg = _RectDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.mask.shapes.append(Shape("rect", dlg.values()))
            self._refresh_list()

    def _add_circle(self):
        dlg = _CircleDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.mask.shapes.append(Shape("circle", dlg.values()))
            self._refresh_list()

    def _add_stripe(self):
        dlg = _StripeDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.mask.shapes.append(Shape("stripe", dlg.values()))
            self._refresh_list()

    def _add_grating(self):
        dlg = _GratingDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self.mask.shapes.append(Shape("grating", dlg.values()))
            self._refresh_list()

    def _duplicate(self):
        """選択中の図形を複製し、直後に挿入して選択する。"""
        row = self.list.currentRow()
        if 0 <= row < len(self.mask.shapes):
            s = self.mask.shapes[row]
            self.mask.shapes.insert(row + 1, Shape(s.kind, dict(s.params)))
            self._refresh_list()
            self.list.setCurrentRow(row + 1)

    def _delete(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.mask.shapes):
            del self.mask.shapes[row]
            self._refresh_list()

    def get_mask(self) -> Mask:
        self.mask.invert = self.invert_cb.isChecked()
        return Mask(shapes=list(self.mask.shapes), invert=self.mask.invert)


def _range_hint(mn: float, mx: float, dec: int = 2) -> str:
    """入力可能範囲のヒント文字列（ツールチップ用）。"""
    return f"入力範囲: {mn:.{dec}f} 〜 {mx:.{dec}f}"


def _spin(value: float, mn=0.0, mx=1.0, step=0.05, dec=2) -> QtWidgets.QDoubleSpinBox:
    s = QtWidgets.QDoubleSpinBox()
    s.setRange(mn, mx)
    s.setSingleStep(step)
    s.setDecimals(dec)
    s.setValue(value)
    # 有効範囲をツールチップで明示（操作のガイド）
    s.setToolTip(_range_hint(mn, mx, dec))
    return s


def _parse_selectivity(text: str) -> dict[str, float]:
    """'material:rate,material:rate' 形式を辞書に変換する（不正な項目は無視）。"""
    result: dict[str, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part or ":" not in part:
            continue
        name, _, rate = part.partition(":")
        name = name.strip()
        try:
            result[name] = float(rate.strip())
        except ValueError:
            continue
    return result


class _RectDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("矩形を追加")
        form = QtWidgets.QFormLayout(self)
        self.x0 = _spin(0.35)
        self.y0 = _spin(0.35)
        self.x1 = _spin(0.65)
        self.y1 = _spin(0.65)
        self.angle = _spin(0.0, -180.0, 180.0, 5.0, 1)
        form.addRow("x0", self.x0)
        form.addRow("y0", self.y0)
        form.addRow("x1", self.x1)
        form.addRow("y1", self.y1)
        form.addRow("回転角 (度)", self.angle)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {
            "x0": min(self.x0.value(), self.x1.value()),
            "y0": min(self.y0.value(), self.y1.value()),
            "x1": max(self.x0.value(), self.x1.value()),
            "y1": max(self.y0.value(), self.y1.value()),
            "angle": self.angle.value(),
        }


class _StripeDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("帯（ライン/トレンチ）を追加")
        form = QtWidgets.QFormLayout(self)
        self.cx = _spin(0.5)
        self.cy = _spin(0.5)
        self.angle = _spin(0.0, -180.0, 180.0, 5.0, 1)
        self.width = _spin(0.2, 0.01, 1.0, 0.02, 3)
        form.addRow("中心 x", self.cx)
        form.addRow("中心 y", self.cy)
        form.addRow("方向 (度)", self.angle)
        form.addRow("帯幅", self.width)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {
            "cx": self.cx.value(),
            "cy": self.cy.value(),
            "angle": self.angle.value(),
            "width": self.width.value(),
        }


class _GratingDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("周期ライン&スペースを追加")
        form = QtWidgets.QFormLayout(self)
        self.angle = _spin(0.0, -180.0, 180.0, 5.0, 1)
        self.period = _spin(0.2, 0.02, 1.0, 0.02, 3)
        self.width = _spin(0.1, 0.01, 1.0, 0.01, 3)
        form.addRow("ライン方向 (度)", self.angle)
        form.addRow("周期", self.period)
        form.addRow("ライン幅", self.width)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {
            "angle": self.angle.value(),
            "period": self.period.value(),
            "width": min(self.width.value(), self.period.value()),
        }


class _CircleDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("円を追加")
        form = QtWidgets.QFormLayout(self)
        self.cx = _spin(0.5)
        self.cy = _spin(0.5)
        self.r = _spin(0.25)
        form.addRow("中心 x", self.cx)
        form.addRow("中心 y", self.cy)
        form.addRow("半径 r", self.r)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def values(self) -> dict:
        return {"cx": self.cx.value(), "cy": self.cy.value(), "r": self.r.value()}


# =============================================================================
# 工程の追加/編集ダイアログ
# =============================================================================
class ProcessDialog(QtWidgets.QDialog):
    """工程タイプに応じてパラメータ入力欄を切り替えるダイアログ。"""

    def __init__(self, proc_type: str, existing: Process | None = None, parent=None):
        super().__init__(parent)
        self.proc_type = proc_type
        self.existing = existing
        label = dict(processes.available_types()).get(proc_type, proc_type)
        self.setWindowTitle(f"{label} の設定")
        self.setMinimumWidth(380)

        self.form = QtWidgets.QFormLayout(self)
        self.mask_editor: MaskEditor | None = None

        # 工程の 1 行説明をダイアログ上部にバナー表示（操作のヒント）
        help_text = processes.process_help(proc_type)
        if help_text:
            banner = QtWidgets.QLabel(help_text)
            banner.setWordWrap(True)
            banner.setObjectName("helpBanner")
            banner.setStyleSheet(
                "#helpBanner { background:rgba(45,108,223,0.10);"
                " border:1px solid rgba(45,108,223,0.28); border-radius:7px;"
                " padding:8px 10px; }"
            )
            self.form.addRow(banner)

        self._build_fields()

        # 現在の入力値から生成される工程サマリのライブプレビュー
        self.preview_lbl = QtWidgets.QLabel()
        self.preview_lbl.setObjectName("procPreview")
        self.preview_lbl.setWordWrap(True)
        self.preview_lbl.setStyleSheet(
            "#procPreview { color:#2d6cdf; font-weight:600; padding:4px 2px; }"
        )
        self.form.addRow("プレビュー", self.preview_lbl)
        self._wire_preview_signals()
        self._update_preview()

        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        self.form.addRow(bb)

    def _wire_preview_signals(self):
        """全入力ウィジェットの変更をプレビュー更新に接続する。"""
        for w in self.findChildren(QtWidgets.QDoubleSpinBox):
            w.valueChanged.connect(self._update_preview)
        for w in self.findChildren(QtWidgets.QSpinBox):
            w.valueChanged.connect(self._update_preview)
        for w in self.findChildren(QtWidgets.QComboBox):
            w.currentIndexChanged.connect(self._update_preview)
        for w in self.findChildren(QtWidgets.QLineEdit):
            w.textChanged.connect(self._update_preview)
        for w in self.findChildren(QtWidgets.QCheckBox):
            w.stateChanged.connect(self._update_preview)
        for w in self.findChildren(QtWidgets.QListWidget):
            w.itemSelectionChanged.connect(self._update_preview)

    def _update_preview(self, *args):
        """現在の設定から工程サマリを生成してプレビューに表示する。"""
        try:
            text = self.build_process().summary()
        except Exception:  # noqa: BLE001 - 編集途中の不正値はプレビュー対象外
            text = "—"
        self.preview_lbl.setText(text)

    # -- フィールド構築 ----------------------------------------------------
    def _build_fields(self):
        t = self.proc_type
        e = self.existing
        if t == "PHOTO":
            self.thick = _spin(getattr(e, "thickness_um", 1.0), 0.05, 20.0, 0.1)
            self.form.addRow("レジスト厚 (µm)", self.thick)
            self.polarity = QtWidgets.QComboBox()
            self.polarity.addItems(["positive", "negative"])
            if e is not None:
                self.polarity.setCurrentText(e.polarity)
            self.form.addRow("極性", self.polarity)
            self.edge_blur = _spin(getattr(e, "edge_blur_sigma_um", 0.0), 0.0, 5.0, 0.05, 3)
            self.form.addRow("角丸めσ (µm)", self.edge_blur)
            mask = e.mask if e is not None else Mask()
            self.mask_editor = MaskEditor(mask)
            self.form.addRow(self.mask_editor)

        elif t in ("CVD", "PVD"):
            self.material = QtWidgets.QComboBox()
            for m in materials.deposit_materials():
                self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("材料", self.material)
            self.thick = _spin(getattr(e, "thickness_um", 0.5), 0.05, 20.0, 0.1)
            self.form.addRow("膜厚 (µm)", self.thick)
            if t == "CVD":
                self.loading = _spin(
                    getattr(e, "loading", 0.0), 0.0, 1.0, 0.05, 2
                )
                self.form.addRow("負荷効果", self.loading)
                self.roughness = _spin(
                    getattr(e, "roughness_um", 0.0), 0.0, 2.0, 0.02, 3
                )
                self.form.addRow("表面粗さ RMS (µm)", self.roughness)
            if t == "PVD":
                self.coverage = _spin(
                    getattr(e, "step_coverage", 1.0) * 100.0, 0.0, 100.0, 5.0, 0
                )
                self.coverage.setSuffix(" %")
                self.form.addRow("段差被覆率", self.coverage)
                self.overhang = _spin(getattr(e, "overhang", 0.0), 0.0, 3.0, 0.1, 1)
                self.form.addRow("オーバーハング (庇)", self.overhang)
                self.tilt = _spin(getattr(e, "tilt_deg", 0.0), 0.0, 89.0, 5.0, 0)
                self.tilt.setSuffix(" °")
                self.form.addRow("斜め蒸着 入射角", self.tilt)
                note = QtWidgets.QLabel("低いほど窪み底に付きにくくなります（100%=一様）。")
                note.setStyleSheet("color: gray;")
                self.form.addRow(note)

        elif t == "ALD":
            self.material = QtWidgets.QComboBox()
            for m in materials.deposit_materials():
                self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("材料", self.material)
            self.cycles = QtWidgets.QSpinBox()
            self.cycles.setRange(1, 5000)
            self.cycles.setValue(getattr(e, "cycles", 100))
            self.form.addRow("サイクル数", self.cycles)
            self.gpc = _spin(
                getattr(e, "growth_per_cycle_nm", 1.0), 0.1, 5.0, 0.1, 2
            )
            self.gpc.setSuffix(" nm")
            self.form.addRow("1サイクル成長量", self.gpc)
            self.ar_coverage = _spin(getattr(e, "ar_coverage", 1.0), 0.0, 1.0, 0.05, 2)
            self.form.addRow("高AR底被覆率", self.ar_coverage)
            self.ar_threshold = _spin(getattr(e, "ar_threshold", 10.0), 1.0, 100.0, 1.0, 1)
            self.form.addRow("被覆低下 AR", self.ar_threshold)
            note = QtWidgets.QLabel("nm 精度の超コンフォーマル膜（High-k/バリア）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "ALE":
            self.targets = QtWidgets.QListWidget()
            self.targets.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            self.targets.setMaximumHeight(150)
            sel = set(getattr(e, "targets", []) or [])
            for m in (m for m in materials.all_materials() if m.etchable):
                it = QtWidgets.QListWidgetItem(m.label)
                it.setData(QtCore.Qt.UserRole, m.name)
                self.targets.addItem(it)
                if m.name in sel:
                    it.setSelected(True)
            self.form.addRow("対象材料\n(未選択=露出材料)", self.targets)
            self.cycles = QtWidgets.QSpinBox()
            self.cycles.setRange(1, 5000)
            self.cycles.setValue(getattr(e, "cycles", 30))
            self.form.addRow("サイクル数", self.cycles)
            self.epc = _spin(getattr(e, "etch_per_cycle_nm", 1.0), 0.1, 5.0, 0.1, 2)
            self.epc.setSuffix(" nm")
            self.form.addRow("1サイクル除去量", self.epc)
            self.anisotropy = _spin(getattr(e, "anisotropy", 0.0), 0.0, 1.0, 0.1, 2)
            self.form.addRow("指向性 (0=等方/1=垂直)", self.anisotropy)
            note = QtWidgets.QLabel("nm 精度・高選択の自己制限エッチ（ALD の対）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t in ("DRY", "WET"):
            self.targets = QtWidgets.QListWidget()
            self.targets.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
            self.targets.setMaximumHeight(150)
            etchables = [m for m in materials.all_materials() if m.etchable]
            sel = set(getattr(e, "targets", []) or [])
            for m in etchables:
                it = QtWidgets.QListWidgetItem(m.label)
                it.setData(QtCore.Qt.UserRole, m.name)
                self.targets.addItem(it)
                if m.name in sel:
                    it.setSelected(True)
            self.form.addRow("対象材料\n(未選択=露出材料)", self.targets)
            self.depth = _spin(getattr(e, "depth_um", 0.5), 0.05, 20.0, 0.1)
            self.form.addRow("エッチ量 (µm)", self.depth)
            if t == "DRY":
                self.overetch = _spin(
                    getattr(e, "overetch_pct", 0.0), 0.0, 100.0, 5.0, 0
                )
                self.overetch.setSuffix(" %")
                self.form.addRow("オーバーエッチ", self.overetch)
                self.lateral = _spin(
                    getattr(e, "lateral_um", 0.0), 0.0, 5.0, 0.05, 3
                )
                self.form.addRow("横方向バイアス (µm)", self.lateral)
                self.selectivity = QtWidgets.QLineEdit()
                sel = getattr(e, "selectivity", {}) or {}
                self.selectivity.setText(
                    ",".join(f"{k}:{v}" for k, v in sel.items())
                )
                self.selectivity.setPlaceholderText("例: oxide:0.33,nitride:0.1")
                self.form.addRow("エッチ選択比", self.selectivity)
                self.mask_erosion = _spin(
                    getattr(e, "mask_erosion", 0.0), 0.0, 2.0, 0.05, 2
                )
                self.form.addRow("マスク消耗比", self.mask_erosion)
                self.taper = _spin(
                    getattr(e, "taper_deg", 0.0), 0.0, 89.0, 1.0, 1
                )
                self.taper.setSuffix(" °")
                self.form.addRow("側壁テーパ角", self.taper)
                self.notch = _spin(getattr(e, "notch_um", 0.0), 0.0, 2.0, 0.02, 3)
                self.form.addRow("RIE ノッチ (µm)", self.notch)
                self.arde = _spin(
                    getattr(e, "arde_lag_um", 0.0), 0.0, 5.0, 0.05, 3
                )
                self.form.addRow("ARDE ラグ (µm)", self.arde)
            else:  # WET
                self.lateral_ratio = _spin(
                    getattr(e, "lateral_ratio", 1.0), 0.0, 1.0, 0.05, 2
                )
                self.form.addRow("横方向比 (1=等方)", self.lateral_ratio)

        elif t == "DIFFUSION":
            self.dopant = QtWidgets.QComboBox()
            for name in ("doped_n", "doped_p"):
                self.dopant.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.dopant.findData(e.dopant)
                if idx >= 0:
                    self.dopant.setCurrentIndex(idx)
            self.form.addRow("ドーパント", self.dopant)
            self.depth = _spin(getattr(e, "depth_um", 0.6), 0.05, 20.0, 0.1)
            self.form.addRow("拡散深さ (µm)", self.depth)

        elif t == "STRIP":
            self.material = QtWidgets.QComboBox()
            skip = {"air", "silicon"}
            for m in materials.all_materials():
                if m.name not in skip:
                    self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("除去する材料", self.material)

        elif t == "CMP":
            self.remove = _spin(getattr(e, "remove_um", 0.5), 0.05, 20.0, 0.1)
            self.form.addRow("研磨量 (µm)", self.remove)
            self.stop_material = QtWidgets.QComboBox()
            self.stop_material.addItem("（なし）", "")
            for m in materials.all_materials():
                if m.name not in {"air", "silicon"}:
                    self.stop_material.addItem(m.label, m.name)
            if e is not None:
                idx = self.stop_material.findData(getattr(e, "stop_material", ""))
                if idx >= 0:
                    self.stop_material.setCurrentIndex(idx)
            self.form.addRow("研磨ストップ層", self.stop_material)
            self.soft_material = QtWidgets.QComboBox()
            self.soft_material.addItem("（なし）", "")
            for m in materials.all_materials():
                if m.name not in {"air", "silicon"}:
                    self.soft_material.addItem(m.label, m.name)
            if e is not None:
                idx = self.soft_material.findData(getattr(e, "soft_material", ""))
                if idx >= 0:
                    self.soft_material.setCurrentIndex(idx)
            self.form.addRow("軟材料(ディッシング)", self.soft_material)
            self.dishing = _spin(getattr(e, "dishing_um", 0.0), 0.0, 5.0, 0.05, 3)
            self.form.addRow("ディッシング量 (µm)", self.dishing)
            self.erosion = _spin(getattr(e, "erosion_um", 0.0), 0.0, 5.0, 0.05, 3)
            self.form.addRow("エロージョン量 (µm)", self.erosion)
            self.density_radius = _spin(
                getattr(e, "density_radius_um", 1.0), 0.1, 20.0, 0.1, 2
            )
            self.form.addRow("密度平均半径 (µm)", self.density_radius)
            note = QtWidgets.QLabel("最も高い点から指定量削り、上面を水平にします。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "BACKGRIND":
            self.thin = _spin(getattr(e, "thin_um", 1.0), 0.1, 100.0, 0.5, 2)
            self.form.addRow("研削量 (µm)", self.thin)
            note = QtWidgets.QLabel("裏面（底）から基板を研削しウェハを薄化します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "OXIDE":
            self.thick = _spin(getattr(e, "thickness_um", 0.3), 0.05, 20.0, 0.1)
            self.form.addRow("酸化膜厚 (µm)", self.thick)
            self.consume = _spin(
                getattr(e, "consume_fraction", 0.45), 0.0, 0.95, 0.05, 2
            )
            self.form.addRow("Si 消費比", self.consume)
            self.beak = _spin(getattr(e, "beak_fraction", 0.0), 0.0, 3.0, 0.1, 2)
            self.form.addRow("バーズビーク比", self.beak)
            # Deal-Grove モード（time_min>0 で膜厚を物理計算）
            self.ox_time = _spin(getattr(e, "time_min", 0.0), 0.0, 6000.0, 5.0, 1)
            self.form.addRow("酸化時間 (分, 0=厚さ指定)", self.ox_time)
            self.ox_temp = _spin(
                getattr(e, "temperature_c", 1000.0), 600.0, 1300.0, 10.0, 0
            )
            self.form.addRow("酸化温度 (℃)", self.ox_temp)
            self.ox_ambient = QtWidgets.QComboBox()
            self.ox_ambient.addItems(["dry", "wet"])
            self.ox_ambient.setCurrentText(getattr(e, "ambient", "dry"))
            self.form.addRow("雰囲気", self.ox_ambient)
            note = QtWidgets.QLabel(
                "露出シリコンを熱酸化します（Si を一部消費）。"
                "時間>0 で Deal-Grove により膜厚を自動計算。"
            )
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "SALICIDE":
            self.thick = _spin(getattr(e, "thickness_um", 0.05), 0.01, 2.0, 0.01, 3)
            self.form.addRow("シリサイド厚 (µm)", self.thick)
            self.react_poly = QtWidgets.QCheckBox("ゲートポリも反応させる")
            self.react_poly.setChecked(bool(getattr(e, "react_poly", True)))
            self.form.addRow(self.react_poly)
            note = QtWidgets.QLabel("露出 Si／ポリ上のみシリサイド化（自己整合）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "SPACER":
            self.material = QtWidgets.QComboBox()
            for name in ("nitride", "oxide"):
                self.material.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("材料", self.material)
            self.thick = _spin(getattr(e, "thickness_um", 0.05), 0.01, 2.0, 0.01, 3)
            self.form.addRow("膜厚 (µm)", self.thick)
            self.overetch = _spin(getattr(e, "overetch_um", 0.0), 0.0, 1.0, 0.01, 3)
            self.form.addRow("オーバーエッチ (µm)", self.overetch)
            note = QtWidgets.QLabel("コンフォーマル成膜＋異方性エッチバックで側壁に残します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "EPI":
            self.material = QtWidgets.QComboBox()
            for name in ("epi_si", "silicon"):
                self.material.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("材料", self.material)
            self.thick = _spin(getattr(e, "thickness_um", 0.5), 0.05, 20.0, 0.1)
            self.form.addRow("エピ厚 (µm)", self.thick)
            self.facet = _spin(getattr(e, "facet_angle_deg", 0.0), 0.0, 89.0, 1.0, 1)
            self.form.addRow("ファセット角 (°,0=無)", self.facet)
            note = QtWidgets.QLabel("露出シリコン上のみに選択成長します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "IMPLANT":
            self.dopant = QtWidgets.QComboBox()
            for name in ("doped_n", "doped_p"):
                self.dopant.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.dopant.findData(e.dopant)
                if idx >= 0:
                    self.dopant.setCurrentIndex(idx)
            self.form.addRow("ドーパント", self.dopant)
            self.range_um = _spin(getattr(e, "range_um", 0.4), 0.05, 20.0, 0.1)
            self.form.addRow("投影飛程 Rp (µm)", self.range_um)
            self.straggle = _spin(getattr(e, "straggle_um", 0.1), 0.01, 5.0, 0.05, 3)
            self.form.addRow("飛程ばらつき σ (µm)", self.straggle)
            self.channeling = _spin(
                getattr(e, "channeling_fraction", 0.0), 0.0, 1.0, 0.05, 2
            )
            self.form.addRow("チャネリング裾比", self.channeling)
            self.tail_decay = _spin(
                getattr(e, "tail_decay_um", 0.0), 0.0, 10.0, 0.05, 3
            )
            self.form.addRow("裾減衰長 (µm, 0=Rp×0.5)", self.tail_decay)
            self.tilt = _spin(getattr(e, "tilt_deg", 0.0), 0.0, 60.0, 1.0, 1)
            self.tilt.setSuffix(" °")
            self.form.addRow("チルト角", self.tilt)
            note = QtWidgets.QLabel("Rp を中心に埋込ドープ（レジストで遮蔽）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "ANNEAL":
            self.depth = _spin(getattr(e, "depth_um", 0.3), 0.05, 20.0, 0.1)
            self.form.addRow("ドライブイン量 (µm)", self.depth)
            self.an_time = _spin(getattr(e, "time_min", 0.0), 0.0, 6000.0, 5.0, 1)
            self.form.addRow("アニール時間 (分, 0=量指定)", self.an_time)
            self.an_temp = _spin(
                getattr(e, "temperature_c", 1000.0), 600.0, 1300.0, 10.0, 0
            )
            self.form.addRow("アニール温度 (℃)", self.an_temp)
            note = QtWidgets.QLabel(
                "既存の拡散層を等方的に再分布させます。"
                "時間>0 で拡散長 L=√(Dt) から深さを自動計算。"
            )
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "RTP":
            self.depth = _spin(getattr(e, "depth_um", 0.15), 0.02, 10.0, 0.05, 3)
            self.form.addRow("ドライブイン量 (µm)", self.depth)
            self.lateral_factor = _spin(
                getattr(e, "lateral_factor", 0.3), 0.0, 1.0, 0.05, 2
            )
            self.form.addRow("横拡散比", self.lateral_factor)
            note = QtWidgets.QLabel("急速熱処理。横方向拡散を抑えて浅く活性化します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "KOH":
            self.targets_combo = QtWidgets.QComboBox()
            for name in ("silicon",):
                self.targets_combo.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.targets_combo.findData(e.target)
                if idx >= 0:
                    self.targets_combo.setCurrentIndex(idx)
            self.form.addRow("対象材料", self.targets_combo)
            self.depth = _spin(getattr(e, "depth_um", 1.0), 0.05, 30.0, 0.1)
            self.form.addRow("エッチ深さ (µm)", self.depth)
            self.angle = _spin(getattr(e, "sidewall_angle_deg", 54.7), 5.0, 89.0, 1.0, 1)
            self.form.addRow("側壁角度 (°)", self.angle)
            note = QtWidgets.QLabel("結晶異方性で斜め側壁（V 溝/台形）を形成。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "DRIE":
            self.targets_combo = QtWidgets.QComboBox()
            for name in ("silicon",):
                self.targets_combo.addItem(materials.get(name).label, name)
            if e is not None:
                idx = self.targets_combo.findData(e.target)
                if idx >= 0:
                    self.targets_combo.setCurrentIndex(idx)
            self.form.addRow("対象材料", self.targets_combo)
            self.depth = _spin(getattr(e, "depth_um", 2.0), 0.05, 50.0, 0.1)
            self.form.addRow("エッチ深さ (µm)", self.depth)
            self.scallop = _spin(getattr(e, "scallop_um", 0.0), 0.0, 5.0, 0.05, 3)
            self.form.addRow("スキャロップ深さ (µm)", self.scallop)
            self.scallop_pitch = _spin(
                getattr(e, "scallop_pitch_um", 0.5), 0.05, 5.0, 0.05, 3
            )
            self.form.addRow("スキャロップ周期 (µm)", self.scallop_pitch)
            self.lag = _spin(getattr(e, "lag", 0.0), 0.0, 1.0, 0.05, 2)
            self.form.addRow("RIE ラグ", self.lag)
            self.redeposit = _spin(getattr(e, "redeposit_um", 0.0), 0.0, 5.0, 0.05, 3)
            self.form.addRow("再付着厚 (µm)", self.redeposit)
            self.microtrench = _spin(
                getattr(e, "microtrench_um", 0.0), 0.0, 5.0, 0.05, 3
            )
            self.form.addRow("μトレンチ深さ (µm)", self.microtrench)
            note = QtWidgets.QLabel("高アスペクト比の垂直深掘り（Bosch）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "FILL":
            self.material = QtWidgets.QComboBox()
            for m in materials.deposit_materials():
                self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("充填材料", self.material)
            self.overfill = _spin(getattr(e, "overfill_um", 0.1), 0.0, 10.0, 0.05, 3)
            self.form.addRow("オーバーフィル (µm)", self.overfill)
            self.void_ar = _spin(getattr(e, "void_ar", 0.0), 0.0, 50.0, 0.5, 1)
            self.form.addRow("ボイド AR しきい値", self.void_ar)
            note = QtWidgets.QLabel("開口/トレンチをボトムアップ充填（ダマシン）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "SPINON":
            self.material = QtWidgets.QComboBox()
            for m in materials.deposit_materials():
                self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.material)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("塗布材料", self.material)
            self.cap = _spin(getattr(e, "cap_um", 0.3), 0.0, 10.0, 0.05, 3)
            self.form.addRow("キャップ厚 (µm)", self.cap)
            self.planarization = _spin(
                getattr(e, "planarization", 1.0), 0.0, 1.0, 0.05, 2
            )
            self.form.addRow("平坦化度 (1=完全)", self.planarization)
            note = QtWidgets.QLabel("スピンオンで全面を覆い上面を平坦化（SOG/SOD）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "LIFTOFF":
            note = QtWidgets.QLabel("レジストとその上の膜を一括除去します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "SPUTTER":
            self.depth = _spin(getattr(e, "depth_um", 0.3), 0.05, 20.0, 0.1)
            self.form.addRow("エッチ量 (µm)", self.depth)
            self.isotropic = _spin(
                getattr(e, "isotropic", 0.0) * 100.0, 0.0, 100.0, 5.0, 0
            )
            self.isotropic.setSuffix(" %")
            self.form.addRow("等方成分", self.isotropic)
            self.faceting = _spin(
                getattr(e, "faceting", 0.0) * 100.0, 0.0, 100.0, 5.0, 0
            )
            self.faceting.setSuffix(" %")
            self.form.addRow("ファセット (面取り)", self.faceting)
            note = QtWidgets.QLabel("材料を問わず物理的に削ります（イオンミリング）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "CLEAN":
            self.material = QtWidgets.QComboBox()
            skip = {"air", "silicon"}
            for m in materials.all_materials():
                if m.name not in skip:
                    self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.target)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("対象材料", self.material)
            self.thick = _spin(getattr(e, "thickness_um", 0.05), 0.01, 2.0, 0.01, 3)
            self.form.addRow("除去量 (µm)", self.thick)
            note = QtWidgets.QLabel("露出表面を薄く等方除去（デスカム/残渣）。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

        elif t == "REFLOW":
            self.material = QtWidgets.QComboBox()
            skip = {"air", "silicon"}
            for m in materials.all_materials():
                if m.name not in skip:
                    self.material.addItem(m.label, m.name)
            if e is not None:
                idx = self.material.findData(e.target)
                if idx >= 0:
                    self.material.setCurrentIndex(idx)
            self.form.addRow("対象材料", self.material)
            self.radius = _spin(getattr(e, "radius_um", 0.2), 0.05, 5.0, 0.05, 3)
            self.form.addRow("平滑化半径 (µm)", self.radius)
            note = QtWidgets.QLabel("熱リフローで角を丸めて平滑化します。")
            note.setStyleSheet("color: gray;")
            self.form.addRow(note)

    # -- 結果取得 ----------------------------------------------------------
    def build_process(self) -> Process:
        t = self.proc_type
        if t == "PHOTO":
            return Photo(
                mask=self.mask_editor.get_mask(),
                thickness_um=self.thick.value(),
                polarity=self.polarity.currentText(),
                edge_blur_sigma_um=self.edge_blur.value(),
            )
        if t == "CVD":
            return CVD(
                material=self.material.currentData(),
                thickness_um=self.thick.value(),
                loading=self.loading.value(),
                roughness_um=self.roughness.value(),
            )
        if t == "ALD":
            return ALD(
                material=self.material.currentData(),
                cycles=self.cycles.value(),
                growth_per_cycle_nm=self.gpc.value(),
                ar_coverage=self.ar_coverage.value(),
                ar_threshold=self.ar_threshold.value(),
            )
        if t == "PVD":
            return PVD(
                material=self.material.currentData(),
                thickness_um=self.thick.value(),
                step_coverage=self.coverage.value() / 100.0,
                overhang=self.overhang.value(),
                tilt_deg=self.tilt.value(),
            )
        if t == "ALE":
            tgts = [it.data(QtCore.Qt.UserRole) for it in self.targets.selectedItems()]
            return AtomicLayerEtch(
                targets=tgts,
                cycles=self.cycles.value(),
                etch_per_cycle_nm=self.epc.value(),
                anisotropy=self.anisotropy.value(),
            )
        if t in ("DRY", "WET"):
            tgts = [
                it.data(QtCore.Qt.UserRole)
                for it in self.targets.selectedItems()
            ]
            if t == "DRY":
                return DryEtch(
                    targets=tgts,
                    depth_um=self.depth.value(),
                    overetch_pct=self.overetch.value(),
                    lateral_um=self.lateral.value(),
                    selectivity=_parse_selectivity(self.selectivity.text()),
                    mask_erosion=self.mask_erosion.value(),
                    taper_deg=self.taper.value(),
                    notch_um=self.notch.value(),
                    arde_lag_um=self.arde.value(),
                )
            return WetEtch(targets=tgts, depth_um=self.depth.value(),
                           lateral_ratio=self.lateral_ratio.value())
        if t == "DIFFUSION":
            return Diffusion(dopant=self.dopant.currentData(), depth_um=self.depth.value())
        if t == "STRIP":
            return Strip(material=self.material.currentData())
        if t == "CMP":
            return CMP(
                remove_um=self.remove.value(),
                stop_material=self.stop_material.currentData(),
                soft_material=self.soft_material.currentData(),
                dishing_um=self.dishing.value(),
                erosion_um=self.erosion.value(),
                density_radius_um=self.density_radius.value(),
            )
        if t == "BACKGRIND":
            return Backgrind(thin_um=self.thin.value())
        if t == "OXIDE":
            return Oxidation(
                thickness_um=self.thick.value(),
                consume_fraction=self.consume.value(),
                beak_fraction=self.beak.value(),
                time_min=self.ox_time.value(),
                temperature_c=self.ox_temp.value(),
                ambient=self.ox_ambient.currentText(),
            )
        if t == "SALICIDE":
            return Silicidation(
                thickness_um=self.thick.value(),
                react_poly=self.react_poly.isChecked(),
            )
        if t == "SPACER":
            return Spacer(
                material=self.material.currentData(),
                thickness_um=self.thick.value(),
                overetch_um=self.overetch.value(),
            )
        if t == "EPI":
            return Epitaxy(
                material=self.material.currentData(),
                thickness_um=self.thick.value(),
                facet_angle_deg=self.facet.value(),
            )
        if t == "IMPLANT":
            return Implant(
                dopant=self.dopant.currentData(),
                range_um=self.range_um.value(),
                straggle_um=self.straggle.value(),
                channeling_fraction=self.channeling.value(),
                tail_decay_um=self.tail_decay.value(),
                tilt_deg=self.tilt.value(),
            )
        if t == "ANNEAL":
            return Anneal(
                depth_um=self.depth.value(),
                time_min=self.an_time.value(),
                temperature_c=self.an_temp.value(),
            )
        if t == "RTP":
            return RapidThermalAnneal(
                depth_um=self.depth.value(),
                lateral_factor=self.lateral_factor.value(),
            )
        if t == "KOH":
            return AnisoWetEtch(
                target=self.targets_combo.currentData(),
                depth_um=self.depth.value(),
                sidewall_angle_deg=self.angle.value(),
            )
        if t == "DRIE":
            return DRIE(
                target=self.targets_combo.currentData(),
                depth_um=self.depth.value(),
                scallop_um=self.scallop.value(),
                scallop_pitch_um=self.scallop_pitch.value(),
                lag=self.lag.value(),
                redeposit_um=self.redeposit.value(),
                microtrench_um=self.microtrench.value(),
            )
        if t == "FILL":
            return Fill(
                material=self.material.currentData(),
                overfill_um=self.overfill.value(),
                void_ar=self.void_ar.value(),
            )
        if t == "SPINON":
            return SpinCoat(
                material=self.material.currentData(),
                cap_um=self.cap.value(),
                planarization=self.planarization.value(),
            )
        if t == "LIFTOFF":
            return LiftOff()
        if t == "SPUTTER":
            return SputterEtch(
                depth_um=self.depth.value(),
                isotropic=self.isotropic.value() / 100.0,
                faceting=self.faceting.value() / 100.0,
            )
        if t == "CLEAN":
            return PlasmaClean(
                target=self.material.currentData(),
                thickness_um=self.thick.value(),
            )
        if t == "REFLOW":
            return Reflow(
                target=self.material.currentData(),
                radius_um=self.radius.value(),
            )
        raise ValueError(t)


# =============================================================================
# 新規ウェハ設定ダイアログ
# =============================================================================
def _grid_size_hint(nx: int, ny: int, nz: int) -> str:
    """格子点数と基本グリッドの概算メモリ（uint8）のヒント文字列。"""
    cells = int(nx) * int(ny) * int(nz)
    mb = cells / (1024.0 * 1024.0)  # uint8 = 1 byte/cell
    return f"格子点数: {cells:,}　基本グリッド ≈ {mb:.1f} MB"


class WaferDialog(QtWidgets.QDialog):
    def __init__(self, config: WaferConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ウェハ設定（新規）")
        form = QtWidgets.QFormLayout(self)
        self.nx = QtWidgets.QSpinBox()
        self.nx.setRange(20, 300)
        self.nx.setValue(config.nx)
        self.ny = QtWidgets.QSpinBox()
        self.ny.setRange(20, 300)
        self.ny.setValue(config.ny)
        self.nz = QtWidgets.QSpinBox()
        self.nz.setRange(20, 300)
        self.nz.setValue(config.nz)
        self.pitch = _spin(config.pitch_um, 0.01, 2.0, 0.01, 3)
        self.sub = _spin(config.substrate_um, 0.2, 20.0, 0.1, 2)
        form.addRow("X ボクセル数", self.nx)
        form.addRow("Y ボクセル数", self.ny)
        form.addRow("Z ボクセル数", self.nz)
        form.addRow("ピッチ (µm/vox)", self.pitch)
        form.addRow("基板厚 (µm)", self.sub)
        # 格子点数・概算メモリのライブリードアウト（重さの目安）
        self.size_lbl = QtWidgets.QLabel()
        self.size_lbl.setStyleSheet("color: #2d6cdf; font-weight: 600;")
        form.addRow("規模", self.size_lbl)
        for sp in (self.nx, self.ny, self.nz):
            sp.valueChanged.connect(self._update_size_hint)
        self._update_size_hint()
        note = QtWidgets.QLabel("解像度を上げると拡大時のジャギーが減りますが\n計算は重くなります。")
        note.setStyleSheet("color: gray;")
        form.addRow(note)
        bb = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        form.addRow(bb)

    def _update_size_hint(self, *args):
        self.size_lbl.setText(
            _grid_size_hint(self.nx.value(), self.ny.value(), self.nz.value())
        )

    def get_config(self) -> WaferConfig:
        return WaferConfig(
            nx=self.nx.value(), ny=self.ny.value(), nz=self.nz.value(),
            pitch_um=self.pitch.value(), substrate_um=self.sub.value(),
        )


# =============================================================================
# 2D 断面ビュー（matplotlib 埋め込み + クリック距離測定）
# =============================================================================
class CrossSection2D(QtWidgets.QWidget):
    """ウェハの 2D 断面を表示し、2 点クリックで距離を測定する。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.wafer = None
        self.axis = "Y"
        self.index = 0
        self.include_resist = True
        self._measure_pts: list[tuple[float, float]] = []
        # 直近に描画した断面（材料 ID 2D 配列）とその実寸。カーソル下の材料判定に使う。
        self._plane = None
        self._plane_wh = (0.0, 0.0)
        self._show_grid = False  # 寸法読み取り用グリッド線の表示

        lay = QtWidgets.QVBoxLayout(self)
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("断面軸:"))
        self.axis_combo = QtWidgets.QComboBox()
        self.axis_combo.addItems(["X (YZ面)", "Y (XZ面)", "Z (XY面)"])
        self.axis_combo.setCurrentIndex(1)
        self.axis_combo.currentIndexChanged.connect(self._on_axis)
        bar.addWidget(self.axis_combo)
        bar.addWidget(QtWidgets.QLabel("位置:"))
        self.idx_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.idx_slider.setRange(0, 100)
        self.idx_slider.valueChanged.connect(self._on_index)
        bar.addWidget(self.idx_slider, 1)
        self.pos_lbl = QtWidgets.QLabel("")
        self.pos_lbl.setMinimumWidth(90)
        bar.addWidget(self.pos_lbl)
        self.measure_btn = QtWidgets.QPushButton("測定: OFF")
        self.measure_btn.setCheckable(True)
        self.measure_btn.toggled.connect(self._on_measure_toggle)
        bar.addWidget(self.measure_btn)
        clr = QtWidgets.QPushButton("測定クリア")
        clr.clicked.connect(self._clear_measure)
        bar.addWidget(clr)
        self.grid_cb = QtWidgets.QCheckBox("グリッド")
        self.grid_cb.setToolTip("寸法読み取り用の補助グリッド線を表示")
        self.grid_cb.toggled.connect(self._on_grid_toggle)
        bar.addWidget(self.grid_cb)
        self.save_btn = QtWidgets.QPushButton("画像保存")
        self.save_btn.setToolTip("現在の断面を PNG 画像として保存")
        self.save_btn.clicked.connect(self.save_image)
        bar.addWidget(self.save_btn)
        lay.addLayout(bar)

        self.fig = Figure(figsize=(5, 4), tight_layout=True)
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self.canvas.mpl_connect("button_press_event", self._on_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        lay.addWidget(self.canvas, 1)

        info_row = QtWidgets.QHBoxLayout()
        self.info_lbl = QtWidgets.QLabel("断面上の2点をクリックすると距離を表示します。")
        self.info_lbl.setStyleSheet("color: #225;")
        info_row.addWidget(self.info_lbl, 1)
        # カーソル下の材料名リードアウト（層の識別用）
        self.mat_lbl = QtWidgets.QLabel("")
        self.mat_lbl.setStyleSheet("color: #225; font-weight: 600;")
        info_row.addWidget(self.mat_lbl)
        lay.addLayout(info_row)

    # -- 設定 --------------------------------------------------------------
    def set_wafer(self, wafer, include_resist: bool):
        self.wafer = wafer
        self.include_resist = include_resist
        self._update_slider_range()
        self.redraw()

    def _update_slider_range(self):
        if self.wafer is None:
            return
        nz, ny, nx = self.wafer.grid.shape
        n = {"X": nx, "Y": ny, "Z": nz}[self.axis]
        self.idx_slider.blockSignals(True)
        self.idx_slider.setRange(0, n - 1)
        if self.index >= n:
            self.index = n // 2
        # 初回は中央
        if self.idx_slider.value() == 0 and self.index == 0:
            self.index = n // 2
        self.idx_slider.setValue(self.index)
        self.idx_slider.blockSignals(False)

    # -- イベント ----------------------------------------------------------
    def _on_axis(self, idx):
        self.axis = ["X", "Y", "Z"][idx]
        self.index = 0
        self._clear_measure()
        self._update_slider_range()
        self.redraw()

    def _on_index(self, value):
        self.index = value
        self.redraw()

    def _on_measure_toggle(self, on):
        self.measure_btn.setText("測定: ON" if on else "測定: OFF")
        if not on:
            self._clear_measure()

    def _clear_measure(self):
        self._measure_pts = []
        self.info_lbl.setText("断面上の2点をクリックすると距離を表示します。")
        self.redraw()

    def _on_grid_toggle(self, on):
        self._show_grid = bool(on)
        self.redraw()

    def _on_click(self, event):
        if not self.measure_btn.isChecked():
            return
        if event.inaxes != self.ax or event.xdata is None:
            return
        self._measure_pts.append((event.xdata, event.ydata))
        if len(self._measure_pts) > 2:
            self._measure_pts = self._measure_pts[-2:]
        if len(self._measure_pts) == 2:
            (x0, y0), (x1, y1) = self._measure_pts
            dx, dy = x1 - x0, y1 - y0
            dist = float(np.hypot(dx, dy))
            self.info_lbl.setText(
                f"距離 = {dist:.3f} µm  (Δ横={dx:+.3f}, Δ縦={dy:+.3f} µm)"
            )
        else:
            self.info_lbl.setText("2点目をクリックしてください。")
        self.redraw()

    def _material_at(self, xdata, ydata) -> str | None:
        """断面上の実寸座標 (µm) にある材料の表示名を返す（範囲外は None）。"""
        if self._plane is None or xdata is None or ydata is None:
            return None
        w_um, h_um = self._plane_wh
        if not (0.0 <= xdata < w_um and 0.0 <= ydata < h_um):
            return None
        p = self.wafer.config.pitch_um
        rows, cols = self._plane.shape
        col = min(cols - 1, max(0, int(xdata / p)))
        row = min(rows - 1, max(0, int(ydata / p)))
        mid = int(self._plane[row, col])
        return materials.get(mid).label

    def _on_motion(self, event):
        """カーソル下の材料名をリードアウトに表示する。"""
        if event.inaxes != self.ax:
            self.mat_lbl.setText("")
            return
        label = self._material_at(event.xdata, event.ydata)
        self.mat_lbl.setText(f"材料: {label}" if label else "")

    # -- 描画 --------------------------------------------------------------
    def redraw(self):
        self.ax.clear()
        if self.wafer is None:
            self.canvas.draw_idle()
            return
        plane, w_um, h_um = visualize.slice_2d(
            self.wafer, self.axis, self.index, self.include_resist
        )
        self._plane = plane
        self._plane_wh = (w_um, h_um)
        cmap, norm = visualize.material_listed_cmap()
        # Z 断面は縦軸が Y、それ以外は縦軸が Z（下=基板）。origin=lower で下が原点。
        self.ax.imshow(
            plane, origin="lower", cmap=cmap, norm=norm,
            extent=[0, w_um, 0, h_um], interpolation="nearest", aspect="equal",
        )
        labels = {
            "X": ("Y (µm)", "Z (µm)"),
            "Y": ("X (µm)", "Z (µm)"),
            "Z": ("X (µm)", "Y (µm)"),
        }[self.axis]
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        p = self.wafer.config.pitch_um
        self.pos_lbl.setText(f"{self.axis} = {self.index * p:.2f} µm")

        # 寸法読み取り用の補助グリッド線
        if self._show_grid:
            self.ax.grid(True, color="#888", lw=0.4, alpha=0.5)

        # 測定マーカー
        if self._measure_pts:
            xs = [pt[0] for pt in self._measure_pts]
            ys = [pt[1] for pt in self._measure_pts]
            self.ax.plot(xs, ys, "o-", color="red", lw=1.5, ms=6)
        self.canvas.draw_idle()

    def save_image(self, path: str | None = None) -> str | None:
        """現在の断面を PNG 画像として保存する。

        path 省略時はファイルダイアログで保存先を尋ねる。測定線も含めた
        現在の表示をそのまま書き出す。保存したパス（未保存なら None）を返す。
        """
        if self.wafer is None:
            self.info_lbl.setText("保存する断面がありません。")
            return None
        if path is None:
            default = f"断面_{self.axis}_{self.index}.png"
            path, _ = QtWidgets.QFileDialog.getSaveFileName(
                self, "断面画像を保存", default, "PNG 画像 (*.png)"
            )
            if not path:
                return None
        self.fig.savefig(path, dpi=150)
        self.info_lbl.setText(f"画像を保存しました: {os.path.basename(path)}")
        return path


# =============================================================================
# メインウィンドウ
# =============================================================================
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("半導体プロセス 3D シミュレータ")
        self.resize(1280, 800)

        # アプリ設定（最後のフォルダ・最近のレシピ・既定ウェハ設定）を復元
        self.settings = AppSettings.load()
        self.settings.prune_missing()
        if self.settings.default_config:
            try:
                base_cfg = WaferConfig.from_dict(self.settings.default_config)
            except Exception:  # noqa: BLE001
                base_cfg = WaferConfig()
        else:
            base_cfg = WaferConfig()

        self.recipe = Recipe(config=base_cfg)
        self.solid_mesh = None  # キャッシュした固体メッシュ
        self.clip_mode = "Z"     # none / X / Y / Z / angle / free
        self.clip_invert = False
        self.clip_frac = 50      # 0..100
        self.azimuth = 0         # 方位角(度)
        self.elevation = 0       # 仰角(度)
        self.show_resist = self.settings.show_resist
        self.smooth = False

        # アンドゥ/リドゥ履歴（レシピ全体の dict スナップショット）
        self._undo_stack: list[dict] = []
        self._redo_stack: list[dict] = []

        self._build_ui()
        self._install_menubar()
        self._install_shortcuts()
        self._restore_geometry()
        self._load_sample_recipe()
        self.rebuild_and_render()

    # -- メニューバー（表示テーマ等） --------------------------------------
    def _install_menubar(self):
        """表示メニュー（ライト/ダークテーマ切替）を設置する。"""
        view = self.menuBar().addMenu("表示")
        self.dark_action = view.addAction("ダークテーマ")
        self.dark_action.setCheckable(True)
        self.dark_action.setChecked(self.settings.ui_theme == "dark")
        self.dark_action.setShortcut("Ctrl+D")
        self.dark_action.triggered.connect(self.toggle_theme)

    def toggle_theme(self):
        """ライト/ダークテーマを切り替え、即時適用して設定に保存する。"""
        from .gui_style import stylesheet
        self.settings.ui_theme = "dark" if self.settings.ui_theme != "dark" else "light"
        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.setStyleSheet(stylesheet(self.settings.ui_theme))
        if hasattr(self, "dark_action"):
            self.dark_action.setChecked(self.settings.ui_theme == "dark")
        try:
            self.settings.save()
        except Exception:  # noqa: BLE001
            pass

    # -- アンドゥ/リドゥ ----------------------------------------------------
    def _install_shortcuts(self):
        QtWidgets.QShortcut(QtGui.QKeySequence.Undo, self, self.undo)
        QtWidgets.QShortcut(QtGui.QKeySequence.Redo, self, self.redo)
        # Ctrl+Y も明示的にリドゥへ
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+Y"), self, self.redo)

    def _push_undo(self):
        """変更前のレシピ状態を履歴に積む（リドゥ履歴はクリア）。"""
        self._undo_stack.append(self.recipe.to_dict())
        if len(self._undo_stack) > 100:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._update_undo_buttons()

    def _restore(self, state: dict):
        self.recipe = Recipe.from_dict(state)
        self._refresh_list()
        self.rebuild_and_render()

    def undo(self):
        if not self._undo_stack:
            return
        self._redo_stack.append(self.recipe.to_dict())
        self._restore(self._undo_stack.pop())
        self._update_undo_buttons()
        self.status.showMessage("元に戻しました", 3000)

    def redo(self):
        if not self._redo_stack:
            return
        self._undo_stack.append(self.recipe.to_dict())
        self._restore(self._redo_stack.pop())
        self._update_undo_buttons()
        self.status.showMessage("やり直しました", 3000)

    def _update_undo_buttons(self):
        if hasattr(self, "undo_btn"):
            self.undo_btn.setEnabled(bool(self._undo_stack))
            self.redo_btn.setEnabled(bool(self._redo_stack))

    # -- 終了時の設定保存 --------------------------------------------------
    def _restore_geometry(self):
        """保存済みウィンドウジオメトリを復元する。"""
        geo = self.settings.window_geometry
        if not geo:
            return
        try:
            ba = QtCore.QByteArray.fromBase64(geo.encode("ascii"))
            self.restoreGeometry(ba)
        except Exception:  # noqa: BLE001
            pass

    def closeEvent(self, event):
        """終了時に現在のウェハ設定・レジスト表示状態・ウィンドウ位置を保存する。"""
        try:
            self.settings.default_config = self.recipe.config.to_dict()
            self.settings.show_resist = self.show_resist
            self.settings.window_geometry = bytes(
                self.saveGeometry().toBase64()
            ).decode("ascii")
            self.settings.save()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    # -- UI 構築 -----------------------------------------------------------
    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)

        # 左パネル（レシピ）
        left = QtWidgets.QVBoxLayout()
        root.addLayout(left, 0)

        left.addWidget(QtWidgets.QLabel("<b>プロセスレシピ</b>"))
        self.list = QtWidgets.QListWidget()
        self.list.setMinimumWidth(300)
        self.list.currentRowChanged.connect(lambda *_: self._on_select())
        self.list.itemDoubleClicked.connect(lambda *_: self.edit_step())
        self.list.keyPressEvent = self._list_key_press
        left.addWidget(self.list, 1)

        # 追加メニュー（カテゴリ別サブメニューで選びやすく）
        add_btn = QtWidgets.QToolButton()
        add_btn.setText("工程を追加 ▾")
        add_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        menu = QtWidgets.QMenu(add_btn)
        for category, types in processes.categorized_types():
            submenu = menu.addMenu(category)
            submenu.setToolTipsVisible(True)
            for t, label in types:
                act = submenu.addAction(f"{t} — {label}")
                act.setToolTip(processes.process_help(t))
                act.triggered.connect(lambda _=False, tt=t: self.add_step(tt))
        add_btn.setMenu(menu)
        left.addWidget(add_btn)

        row = QtWidgets.QHBoxLayout()
        for text, fn in [
            ("編集", self.edit_step),
            ("複製", self.duplicate_step),
            ("削除", self.delete_step),
            ("▲", lambda: self.move_step(-1)),
            ("▼", lambda: self.move_step(1)),
        ]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            row.addWidget(b)
        left.addLayout(row)

        frow = QtWidgets.QHBoxLayout()
        for text, fn in [
            ("新規", self.new_recipe),
            ("保存", self.save_recipe),
            ("読込", self.load_recipe),
            ("STL出力", self.export_stl),
        ]:
            b = QtWidgets.QPushButton(text)
            b.clicked.connect(fn)
            frow.addWidget(b)
        left.addLayout(frow)

        # プリセット読込・最近のレシピ・計測レポート
        prow = QtWidgets.QHBoxLayout()
        preset_btn = QtWidgets.QToolButton()
        preset_btn.setText("プリセット ▾")
        preset_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        pmenu = QtWidgets.QMenu(preset_btn)
        for name in presets.available():
            act = pmenu.addAction(name)
            act.triggered.connect(lambda _=False, nm=name: self.load_preset(nm))
        preset_btn.setMenu(pmenu)
        prow.addWidget(preset_btn)

        self.recent_btn = QtWidgets.QToolButton()
        self.recent_btn.setText("最近のレシピ ▾")
        self.recent_btn.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        prow.addWidget(self.recent_btn)
        self._rebuild_recent_menu()

        report_btn = QtWidgets.QPushButton("計測レポート")
        report_btn.clicked.connect(self.show_report)
        prow.addWidget(report_btn)
        left.addLayout(prow)

        # アンドゥ/リドゥ
        urow = QtWidgets.QHBoxLayout()
        self.undo_btn = QtWidgets.QPushButton("↶ 元に戻す")
        self.undo_btn.setToolTip("Ctrl+Z")
        self.undo_btn.clicked.connect(self.undo)
        self.undo_btn.setEnabled(False)
        self.redo_btn = QtWidgets.QPushButton("↷ やり直す")
        self.redo_btn.setToolTip("Ctrl+Y")
        self.redo_btn.clicked.connect(self.redo)
        self.redo_btn.setEnabled(False)
        urow.addWidget(self.undo_btn)
        urow.addWidget(self.redo_btn)
        left.addLayout(urow)

        # プレビュー段数
        self.preview_cb = QtWidgets.QCheckBox("選択工程までを表示")
        self.preview_cb.stateChanged.connect(lambda *_: self.rebuild_and_render())
        left.addWidget(self.preview_cb)

        # 中央列の層構成リードアウト
        left.addWidget(QtWidgets.QLabel("<b>層構成（中央列, 上→下）</b>"))
        self.stack_view = QtWidgets.QTextEdit()
        self.stack_view.setReadOnly(True)
        self.stack_view.setMaximumHeight(150)
        self.stack_view.setStyleSheet("font-family: monospace;")
        left.addWidget(self.stack_view)

        # 右パネル（3D + 断面コントロール）
        right = QtWidgets.QVBoxLayout()
        root.addLayout(right, 1)

        # 断面コントロールバー
        ctrl = QtWidgets.QHBoxLayout()
        ctrl.addWidget(QtWidgets.QLabel("断面:"))
        self.clip_combo = QtWidgets.QComboBox()
        self.clip_combo.addItems(["なし", "X断面", "Y断面", "Z断面", "角度指定", "自由(マウス操作)"])
        self.clip_combo.setCurrentIndex(3)
        self.clip_combo.currentIndexChanged.connect(self._on_clip_mode)
        ctrl.addWidget(self.clip_combo)

        self.invert_cb = QtWidgets.QCheckBox("反対側")
        self.invert_cb.stateChanged.connect(self._on_invert)
        ctrl.addWidget(self.invert_cb)

        # 角度指定用（方位角・仰角）
        self.azim_label = QtWidgets.QLabel("方位")
        ctrl.addWidget(self.azim_label)
        self.azim_spin = QtWidgets.QSpinBox()
        self.azim_spin.setRange(0, 359)
        self.azim_spin.setSuffix("°")
        self.azim_spin.setWrapping(True)
        self.azim_spin.valueChanged.connect(self._on_angle)
        ctrl.addWidget(self.azim_spin)
        self.elev_label = QtWidgets.QLabel("仰角")
        ctrl.addWidget(self.elev_label)
        self.elev_spin = QtWidgets.QSpinBox()
        self.elev_spin.setRange(-90, 90)
        self.elev_spin.setSuffix("°")
        self.elev_spin.valueChanged.connect(self._on_angle)
        ctrl.addWidget(self.elev_spin)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(50)
        self.slider.valueChanged.connect(self._on_slider)
        ctrl.addWidget(self.slider, 1)

        self.pos_label = QtWidgets.QLabel("")
        self.pos_label.setMinimumWidth(96)
        self.pos_label.setStyleSheet("color: #225;")
        ctrl.addWidget(self.pos_label)

        self.resist_cb = QtWidgets.QCheckBox("レジスト表示")
        self.resist_cb.setChecked(True)
        self.resist_cb.stateChanged.connect(self._on_resist)
        ctrl.addWidget(self.resist_cb)

        self.smooth_cb = QtWidgets.QCheckBox("スムーズ")
        self.smooth_cb.setToolTip("外形を平滑化して表示（断面の精度より見た目優先）")
        self.smooth_cb.stateChanged.connect(self._on_smooth)
        ctrl.addWidget(self.smooth_cb)

        # 視点プリセット & 画像保存
        self.view_combo = QtWidgets.QComboBox()
        self.view_combo.addItems(["等角", "上(XY)", "前(XZ)", "横(YZ)"])
        self.view_combo.currentIndexChanged.connect(self._on_view_preset)
        ctrl.addWidget(self.view_combo)
        shot_btn = QtWidgets.QPushButton("画像保存")
        shot_btn.clicked.connect(self.save_screenshot)
        ctrl.addWidget(shot_btn)
        stl_btn = QtWidgets.QPushButton("STL書出")
        stl_btn.clicked.connect(self.export_stl)
        ctrl.addWidget(stl_btn)

        # 3D タブと 2D タブを切り替えるタブウィジェット
        self.tabs = QtWidgets.QTabWidget()
        right.addWidget(self.tabs, 1)

        # --- 3D タブ ---
        tab3d = QtWidgets.QWidget()
        v3 = QtWidgets.QVBoxLayout(tab3d)
        v3.setContentsMargins(0, 0, 0, 0)
        v3.addLayout(ctrl)
        self.plotter = QtInteractor(tab3d)
        v3.addWidget(self.plotter.interactor, 1)
        try:
            self.plotter.enable_anti_aliasing("msaa", multi_samples=8)
        except Exception:
            try:
                self.plotter.enable_anti_aliasing()
            except Exception:
                pass
        self.plotter.set_background("white")
        self.tabs.addTab(tab3d, "3D ビュー")

        # --- 2D 断面タブ ---
        self.view2d = CrossSection2D()
        self.tabs.addTab(self.view2d, "2D 断面")
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # 凡例
        self._build_legend(right)

        self.status = self.statusBar()
        # 初期は Z 断面なので角度コントロールは隠す
        for w in (self.azim_label, self.azim_spin, self.elev_label, self.elev_spin):
            w.setVisible(False)

    def _build_legend(self, layout):
        box = QtWidgets.QHBoxLayout()
        box.addWidget(QtWidgets.QLabel("凡例:"))
        self._legend_entries = {}  # material id -> (swatch, label) ウィジェット
        for m in materials.all_materials():
            if m.name == "air":
                continue
            sw = QtWidgets.QLabel("  ")
            r, g, b = [int(c * 255) for c in m.color]
            sw.setStyleSheet(
                f"background-color: rgb({r},{g},{b}); border: 1px solid #888;"
            )
            sw.setFixedSize(16, 16)
            lbl = QtWidgets.QLabel(m.label)
            box.addWidget(sw)
            box.addWidget(lbl)
            self._legend_entries[m.id] = (sw, lbl)
        box.addStretch(1)
        w = QtWidgets.QWidget()
        w.setLayout(box)
        layout.addWidget(w)

    def _update_legend(self):
        """現在のウェハに存在する材料だけを凡例に表示する。"""
        if not hasattr(self, "_legend_entries"):
            return
        present = set()
        if getattr(self, "wafer", None) is not None:
            present = set(np.unique(self.wafer.grid).tolist())
        for mid, (sw, lbl) in self._legend_entries.items():
            visible = mid in present
            sw.setVisible(visible)
            lbl.setVisible(visible)

    # -- レシピ操作 --------------------------------------------------------
    def _refresh_list(self):
        cur = self.list.currentRow()
        self.list.clear()
        for i, s in enumerate(self.recipe.steps):
            item = QtWidgets.QListWidgetItem(f"{i + 1:>2}. {s.summary()}")
            # 工程カテゴリの識別色を小さなドットアイコンで表示（視認性向上）
            pm = QtGui.QPixmap(11, 11)
            pm.fill(QtGui.QColor(processes.process_color(s.type)))
            item.setIcon(QtGui.QIcon(pm))
            cat = processes.process_category(s.type)
            item.setToolTip(f"{cat}：{processes.process_help(s.type)}")
            self.list.addItem(item)
        if 0 <= cur < self.list.count():
            self.list.setCurrentRow(cur)

    def _selected_type_label(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.recipe.steps):
            return self.recipe.steps[row].type
        return None

    def add_step(self, proc_type: str):
        dlg = ProcessDialog(proc_type, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            proc = dlg.build_process()
            self._push_undo()
            row = self.list.currentRow()
            index = row + 1 if row >= 0 else None
            self.recipe.add(proc, index)
            self._refresh_list()
            new_row = index if index is not None else len(self.recipe.steps) - 1
            self.list.setCurrentRow(new_row)
            self.rebuild_and_render()

    def edit_step(self):
        row = self.list.currentRow()
        if not (0 <= row < len(self.recipe.steps)):
            return
        proc = self.recipe.steps[row]
        dlg = ProcessDialog(proc.type, existing=proc, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._push_undo()
            self.recipe.replace(row, dlg.build_process())
            self._refresh_list()
            self.list.setCurrentRow(row)
            self.rebuild_and_render()

    def duplicate_step(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.recipe.steps):
            self._push_undo()
            new_row = self.recipe.duplicate(row)
            self._refresh_list()
            self.list.setCurrentRow(new_row)
            self.rebuild_and_render()

    def delete_step(self):
        row = self.list.currentRow()
        if 0 <= row < len(self.recipe.steps):
            self._push_undo()
            self.recipe.remove(row)
            self._refresh_list()
            self.rebuild_and_render()

    def move_step(self, delta: int):
        row = self.list.currentRow()
        if 0 <= row < len(self.recipe.steps):
            new_row = max(0, min(len(self.recipe.steps) - 1, row + delta))
            if new_row == row:
                return
            self._push_undo()
            new_row = self.recipe.move(row, delta)
            self._refresh_list()
            self.list.setCurrentRow(new_row)
            self.rebuild_and_render()

    def _on_select(self):
        if self.preview_cb.isChecked():
            self.rebuild_and_render()

    def _list_key_press(self, event):
        """レシピ一覧でのキー操作（Delete=削除）。"""
        if event.key() in (QtCore.Qt.Key_Delete, QtCore.Qt.Key_Backspace):
            self.delete_step()
            event.accept()
            return
        QtWidgets.QListWidget.keyPressEvent(self.list, event)

    def new_recipe(self):
        dlg = WaferDialog(self.recipe.config, parent=self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            self._push_undo()
            self.recipe = Recipe(config=dlg.get_config())
            self._refresh_list()
            self.rebuild_and_render()

    def save_recipe(self):
        start = self.settings.last_dir or "recipe.json"
        if self.settings.last_dir:
            start = os.path.join(self.settings.last_dir, "recipe.json")
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "レシピを保存", start, "JSON (*.json)"
        )
        if path:
            self.recipe.save(path)
            self._remember_recipe(path)
            self.status.showMessage(f"保存しました: {path}", 5000)

    def load_recipe(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "レシピを読込", self.settings.last_dir, "JSON (*.json)"
        )
        if path:
            self._load_recipe_path(path)

    def _load_recipe_path(self, path: str):
        """指定パスのレシピを読み込んで反映する（最近メニュー/読込で共用）。"""
        try:
            loaded = Recipe.load(path)
        except Exception as ex:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "読込エラー", str(ex))
            return
        self._push_undo()
        self.recipe = loaded
        self._refresh_list()
        self.rebuild_and_render()
        self._remember_recipe(path)
        self.status.showMessage(f"読込みました: {path}", 5000)

    def load_preset(self, name: str):
        """組み込みプリセットを読み込む。"""
        try:
            recipe = presets.build(name)
        except Exception as ex:  # noqa: BLE001
            QtWidgets.QMessageBox.critical(self, "プリセット", str(ex))
            return
        self._push_undo()
        self.recipe = recipe
        self._refresh_list()
        self.rebuild_and_render()
        self.status.showMessage(f"プリセットを読込みました: {name}", 5000)

    def show_report(self):
        """現在のウェハの計測レポートをダイアログ表示し、任意で保存できる。"""
        if getattr(self, "wafer", None) is None:
            QtWidgets.QMessageBox.information(self, "計測レポート", "対象がありません。")
            return
        text = metrology.report(self.wafer)
        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("計測レポート")
        dlg.resize(560, 480)
        v = QtWidgets.QVBoxLayout(dlg)
        edit = QtWidgets.QTextEdit()
        edit.setReadOnly(True)
        edit.setStyleSheet("font-family: monospace;")
        edit.setText(text)
        v.addWidget(edit)
        btns = QtWidgets.QHBoxLayout()
        save_b = QtWidgets.QPushButton("テキスト保存")
        save_b.clicked.connect(lambda: self._save_report_text(text))
        close_b = QtWidgets.QPushButton("閉じる")
        close_b.clicked.connect(dlg.accept)
        btns.addStretch(1)
        btns.addWidget(save_b)
        btns.addWidget(close_b)
        v.addLayout(btns)
        dlg.exec_()

    def _save_report_text(self, text: str):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "レポートを保存", self.settings.last_dir or "report.txt", "Text (*.txt)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as f:
                    f.write(text)
                self.status.showMessage(f"レポートを保存しました: {path}", 5000)
            except OSError as ex:
                QtWidgets.QMessageBox.warning(self, "保存エラー", str(ex))

    def _remember_recipe(self, path: str):
        """最近のレシピに追加して設定を保存し、メニューを更新する。"""
        self.settings.add_recent(path)
        self.settings.show_resist = self.show_resist
        try:
            self.settings.save()
        except OSError:
            pass
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        """最近のレシピメニューを再構築する。"""
        if not hasattr(self, "recent_btn"):
            return
        menu = QtWidgets.QMenu(self.recent_btn)
        if not self.settings.recent_recipes:
            act = menu.addAction("(履歴なし)")
            act.setEnabled(False)
        else:
            for p in self.settings.recent_recipes:
                act = menu.addAction(os.path.basename(p))
                act.setToolTip(p)
                act.triggered.connect(lambda _=False, pp=p: self._load_recipe_path(pp))
        self.recent_btn.setMenu(menu)

    # -- 断面コントロール --------------------------------------------------
    def _on_clip_mode(self, idx: int):
        self.clip_mode = ["none", "X", "Y", "Z", "angle", "free"][idx]
        has_offset = self.clip_mode in ("X", "Y", "Z", "angle")
        is_angle = self.clip_mode == "angle"
        self.slider.setEnabled(has_offset)
        self.invert_cb.setEnabled(has_offset)
        for w in (self.azim_label, self.azim_spin, self.elev_label, self.elev_spin):
            w.setVisible(is_angle)
        self.render()

    def _on_invert(self, state):
        self.clip_invert = bool(state)
        self.render()

    def _on_slider(self, value):
        self.clip_frac = value
        if self.clip_mode in ("X", "Y", "Z", "angle"):
            self.render()

    def _on_angle(self, *_):
        self.azimuth = self.azim_spin.value()
        self.elevation = self.elev_spin.value()
        if self.clip_mode == "angle":
            self.render()

    def _on_resist(self, state):
        self.show_resist = bool(state)
        # 再シミュレーションは不要。キャッシュ済みウェハからメッシュのみ作り直す。
        self._rebuild_mesh()
        self.render()

    def _on_smooth(self, state):
        self.smooth = bool(state)
        self.render()

    def _on_view_preset(self, idx):
        views = [
            self.plotter.view_isometric,
            self.plotter.view_xy,
            self.plotter.view_xz,
            self.plotter.view_yz,
        ]
        if 0 <= idx < len(views):
            try:
                views[idx]()
                self.plotter.render()
            except Exception:
                pass

    def _on_tab_changed(self, idx):
        """2D タブに切り替わったら最新ウェハで描画する。"""
        if self.tabs.tabText(idx).startswith("2D") and hasattr(self, "view2d"):
            self.view2d.set_wafer(getattr(self, "wafer", None), self.show_resist)

    def save_screenshot(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "画像を保存", "snapshot.png", "PNG (*.png)"
        )
        if path:
            try:
                self.plotter.screenshot(path)
                self.status.showMessage(f"画像を保存しました: {path}", 5000)
            except Exception as ex:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "保存エラー", str(ex))

    def export_stl(self):
        """現在のウェハを STL メッシュとして書き出す。"""
        wafer = getattr(self, "wafer", None)
        if wafer is None:
            QtWidgets.QMessageBox.information(self, "STL書出", "先にシミュレーションを実行してください。")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "STL を保存", "wafer.stl", "STL (*.stl)"
        )
        if path:
            try:
                n = export.to_stl(wafer, path)
                self.status.showMessage(f"STL を保存しました（{n} 面）: {path}", 5000)
            except Exception as ex:  # noqa: BLE001
                QtWidgets.QMessageBox.warning(self, "保存エラー", str(ex))

    # -- シミュレーション & 描画 ------------------------------------------
    def rebuild_and_render(self):
        """レシピを再計算してメッシュを作り直し、描画する。"""
        up_to = None
        if self.preview_cb.isChecked():
            row = self.list.currentRow()
            if row >= 0:
                up_to = row + 1
        self.status.showMessage("計算中...")
        QtWidgets.QApplication.processEvents()
        self.wafer = self.recipe.simulate(up_to=up_to)
        self._rebuild_mesh()
        self.render()
        n_cells = self.solid_mesh.n_cells if self.solid_mesh is not None else 0
        self.status.showMessage(f"工程数 {len(self.recipe.steps)} / 固体セル {n_cells:,}", 8000)

    def _rebuild_mesh(self):
        """キャッシュ済みウェハから表示メッシュを作り直す（再計算なし）。"""
        if getattr(self, "wafer", None) is None:
            self.solid_mesh = None
            return
        self.solid_mesh = visualize.solid_unstructured(
            self.wafer, include_resist=self.show_resist
        )
        self._update_legend()
        self._update_stack_view()
        if hasattr(self, "view2d"):
            self.view2d.set_wafer(self.wafer, self.show_resist)

    def _update_stack_view(self):
        """中央列の層構成を上→下の順に表示する。"""
        if not hasattr(self, "stack_view"):
            return
        w = getattr(self, "wafer", None)
        if w is None:
            self.stack_view.setPlainText("")
            return
        cx = w.config.nx // 2
        cy = w.config.ny // 2
        stack = w.column_stack(cx, cy)  # 下→上
        if not stack:
            self.stack_view.setHtml("<i>（固体なし）</i>")
            return
        rows = []
        total = 0.0
        for mid, thick in reversed(stack):  # 上→下で表示
            m = materials.BY_ID.get(mid)
            label = m.label if m is not None else f"ID{mid}"
            color = m.color if m is not None else (0.5, 0.5, 0.5)
            r, g, b = [int(c * 255) for c in color]
            rows.append(
                f'<tr><td style="background-color:rgb({r},{g},{b});">'
                f'&nbsp;&nbsp;</td><td>&nbsp;{label}</td>'
                f'<td align="right">{thick:.2f} µm</td></tr>'
            )
            total += thick
        html = (
            "<table cellspacing='0' cellpadding='2'>"
            + "".join(rows)
            + f"</table><div>合計 {total:.2f} µm</div>"
        )
        self.stack_view.setHtml(html)

    def render(self):
        if self.solid_mesh is None:
            return
        cam = self.plotter.camera_position
        self.plotter.clear()
        cmap, clim = visualize.material_colormap()
        mesh = self.solid_mesh

        if mesh.n_cells == 0:
            self.plotter.add_text("（固体がありません）", color="gray")
            self.plotter.render()
            return

        common = dict(
            scalars="material",
            cmap=cmap,
            clim=clim,
            show_scalar_bar=False,
            smooth_shading=self.smooth,
            interpolate_before_map=False,
            show_edges=False,
        )

        if self.clip_mode in ("X", "Y", "Z", "angle"):
            b = mesh.bounds  # xmin,xmax,ymin,ymax,zmin,zmax
            cx = 0.5 * (b[0] + b[1])
            cy = 0.5 * (b[2] + b[3])
            cz = 0.5 * (b[4] + b[5])
            frac = self.clip_frac / 100.0
            if self.clip_mode == "X":
                normal = (1, 0, 0)
                origin = (b[0] + frac * (b[1] - b[0]), cy, cz)
            elif self.clip_mode == "Y":
                normal = (0, 1, 0)
                origin = (cx, b[2] + frac * (b[3] - b[2]), cz)
            elif self.clip_mode == "Z":
                normal = (0, 0, 1)
                origin = (cx, cy, b[4] + frac * (b[5] - b[4]))
            else:  # angle: 方位角・仰角から法線ベクトルを生成
                az = np.deg2rad(self.azimuth)
                el = np.deg2rad(self.elevation)
                normal = (
                    np.cos(el) * np.cos(az),
                    np.cos(el) * np.sin(az),
                    np.sin(el),
                )
                # 中心からスライダー量だけ法線方向にオフセット
                half = 0.5 * np.sqrt(
                    (b[1] - b[0]) ** 2 + (b[3] - b[2]) ** 2 + (b[5] - b[4]) ** 2
                )
                off = (frac - 0.5) * 2.0 * half * 0.5
                origin = (
                    cx + normal[0] * off,
                    cy + normal[1] * off,
                    cz + normal[2] * off,
                )
            try:
                clipped = mesh.clip(normal=normal, origin=origin, invert=self.clip_invert)
            except Exception:
                clipped = mesh
            if clipped.n_cells == 0:
                clipped = mesh
            self.plotter.add_mesh(
                self._maybe_smooth(clipped, plane=(origin, normal)), **common
            )
            self._update_pos_label(origin)
        elif self.clip_mode == "free":
            # 自由クリップは体積メッシュを対話的に切り、断面の中身を常に詰めて
            # 見せる（設計上の不変条件）。表面抽出は中空シェル化して断面が空洞に
            # 見えるため、幾何平滑化はかけない（見た目の平滑化は smooth_shading）。
            self.plotter.add_mesh_clip_plane(mesh, **common)
            self.pos_label.setText("")
        else:
            self.plotter.add_mesh(self._maybe_smooth(mesh), **common)
            self.pos_label.setText("")

        self.plotter.show_bounds(
            grid="back", location="outer", color="gray",
            xtitle="X (µm)", ytitle="Y (µm)", ztitle="Z (µm)",
        )
        if cam is not None and self._has_rendered():
            self.plotter.camera_position = cam
        else:
            self.plotter.view_isometric()
        self._rendered = True
        self.plotter.render()

    def _has_rendered(self) -> bool:
        return getattr(self, "_rendered", False)

    def _update_pos_label(self, origin):
        """断面の位置を µm で表示する。"""
        if self.clip_mode == "X":
            self.pos_label.setText(f"X = {origin[0]:.2f} µm")
        elif self.clip_mode == "Y":
            self.pos_label.setText(f"Y = {origin[1]:.2f} µm")
        elif self.clip_mode == "Z":
            self.pos_label.setText(f"Z = {origin[2]:.2f} µm")
        elif self.clip_mode == "angle":
            self.pos_label.setText(
                f"方位{self.azimuth}° 仰角{self.elevation}°"
            )
        else:
            self.pos_label.setText("")

    def _maybe_smooth(self, mesh, plane=None):
        """スムーズ表示が有効ならサーフェスを抽出して平滑化する。

        plane=(origin, normal) を渡すと断面のクリップ面を平坦に保つ（平滑化で
        切断面が波打つのを防ぐ）。
        """
        if not self.smooth:
            return mesh
        try:
            tol = 0.5 * (self.wafer.config.pitch_um
                         if getattr(self, "wafer", None) is not None else 0.05)
            return visualize.smoothed_surface(mesh, plane=plane, plane_tol=tol)
        except Exception:
            return mesh

    # -- サンプル ----------------------------------------------------------
    def _load_sample_recipe(self):
        r = self.recipe
        r.add(CVD(material="oxide", thickness_um=0.4))
        r.add(CVD(material="nitride", thickness_um=0.3))
        m = Mask(shapes=[Shape("rect", {"x0": 0.30, "y0": 0.30, "x1": 0.70, "y1": 0.70})])
        r.add(Photo(mask=m, thickness_um=1.2, polarity="positive"))
        r.add(DryEtch(targets=["nitride", "oxide"], depth_um=0.7))
        r.add(Strip(material="photoresist"))
        r.add(Diffusion(dopant="doped_n", depth_um=0.6))
        r.add(PVD(material="metal_al", thickness_um=0.5))
        self._refresh_list()


def run():
    import sys

    from .gui_style import stylesheet

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    # 起動時に保存テーマ（既定ライト）の QSS を適用してフラットな外観にする
    theme = AppSettings.load().ui_theme
    app.setStyleSheet(stylesheet(theme))
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
