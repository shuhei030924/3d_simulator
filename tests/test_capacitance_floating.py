"""静電界ソルバの浮遊導体（等電位ノード）扱いのテスト。

浮遊導体は接地シールド（完全遮蔽=容量0）と異なり、等電位で直列に電界を
通すため A-B 容量を大きく変えない——この物理を検証する。
"""
from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _stack(with_float: bool, pitch: float = 0.05) -> Wafer:
    cfg = WaferConfig(nx=30, ny=8, nz=40, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[8:11, :, :] = materials.get("metal_al").id   # A 下
    g[28:31, :, :] = materials.get("metal_cu").id   # B 上
    if with_float:
        g[18:20, :, :] = materials.get("tungsten").id  # 中央 浮遊 W
    return w


def test_floating_plate_does_not_shield():
    """中央の浮遊板は遮蔽せず、A-B 容量を大きく変えない（直列通過）。"""
    c_direct = M.parasitic_capacitance_field_ff(_stack(False), "metal_al", "metal_cu")
    c_float = M.parasitic_capacitance_field_ff(_stack(True), "metal_al", "metal_cu")
    assert c_direct > 0 and c_float > 0
    # 中央浮遊板の直列容量は概ね直接容量に等しい（接地遮蔽の 0 とは明確に異なる）
    assert 0.85 < c_float / c_direct < 1.3


def test_floating_differs_from_grounded_shield_model():
    """同じ中間導体でも、浮遊（field）は非ゼロ、接地遮蔽モデル（line-scan）は 0。"""
    w = _stack(True)
    c_float = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    c_shield = M.parasitic_capacitance_ff(w, "metal_al", "metal_cu")  # 第3導体で遮断
    assert c_float > 0.0
    assert c_shield == 0.0


def test_floating_solver_mesh_convergent():
    """浮遊導体を含んでもメッシュ細分で容量が収束する（差が縮小）。"""
    def cap(pitch):
        nx, ny, nz = round(1.5 / pitch), round(0.5 / pitch), round(2.0 / pitch)
        w = Wafer(WaferConfig(nx=nx, ny=ny, nz=nz, pitch_um=pitch, substrate_um=0.0))
        g = w.grid
        g[:] = materials.get("oxide").id
        z0, pt = nz // 4, round(0.15 / pitch)
        g[z0:z0 + pt, :, :] = materials.get("metal_al").id
        g[nz - z0 - pt:nz - z0, :, :] = materials.get("metal_cu").id
        zc = nz // 2
        g[zc:zc + pt, :, :] = materials.get("tungsten").id
        return M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    v = [cap(p) for p in (0.1, 0.05, 0.025)]
    assert abs(v[2] - v[1]) < abs(v[1] - v[0])  # 収束（差が縮小）


def test_no_floating_unchanged():
    """浮遊導体が無い場合は従来どおり解析的な平行平板に一致。"""
    eps0_ff = 8.854e-3
    n, pitch = 24, 0.1
    cfg = WaferConfig(nx=n, ny=n, nz=n, pitch_um=pitch, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:] = materials.get("oxide").id
    g[6:9, :, :] = materials.get("metal_al").id
    g[15:18, :, :] = materials.get("metal_cu").id  # gap 6 vox
    c = M.parasitic_capacitance_field_ff(w, "metal_al", "metal_cu")
    analytic = eps0_ff * 3.9 * (n * pitch) ** 2 / ((6 + 1) * pitch)
    assert abs(c - analytic) / analytic < 0.05
