"""ALE（原子層エッチ）プロセスのテスト。"""
from semisim import materials
from semisim import processes as P
from semisim.grid import Wafer, WaferConfig
from semisim.processes import Process


def _stack() -> Wafer:
    """下から nitride(ストップ) / oxide(対象) の積層ウェハ。"""
    cfg = WaferConfig(nx=40, ny=40, nz=40, pitch_um=0.01, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:10, :, :] = materials.get("nitride").id
    g[10:30, :, :] = materials.get("oxide").id
    return w


def test_ale_precise_depth():
    """除去量は cycles×etch_per_cycle_nm で厳密（自己制限）。"""
    w = _stack()
    top0 = int(w.top_surface_z().max())
    ale = P.AtomicLayerEtch(targets=["oxide"], cycles=20, etch_per_cycle_nm=1.0)
    assert ale.depth_um == 0.020
    ale.apply(w)
    removed = top0 - int(w.top_surface_z().max())
    assert removed == w.um_to_vox(ale.depth_um) == 2


def test_ale_selective_stops_on_other_material():
    """対象（oxide）のみ除去し、下の nitride は完全に残る（高選択比）。"""
    w = _stack()
    ni_before = int((w.grid == materials.get("nitride").id).sum())
    P.AtomicLayerEtch(targets=["oxide"], cycles=100, etch_per_cycle_nm=1.0).apply(w)
    ni_after = int((w.grid == materials.get("nitride").id).sum())
    assert ni_after == ni_before  # ストップ層は無傷


def test_ale_nontarget_unchanged():
    """対象に指定しない材料は削れない。"""
    w = _stack()
    ox_before = int((w.grid == materials.get("oxide").id).sum())
    P.AtomicLayerEtch(targets=["poly"], cycles=20).apply(w)
    assert int((w.grid == materials.get("oxide").id).sum()) == ox_before


def test_ale_isotropic_undercuts_more_than_directional():
    """等方 ALE はマスク下に横アンダーカット、指向性 ALE は垂直のみ。"""
    cfg = WaferConfig(nx=60, ny=10, nz=40, pitch_um=0.01, substrate_um=0.0)

    def run(aniso):
        w = Wafer(cfg)
        g = w.grid
        g[:30, :, :] = materials.get("oxide").id
        g[30:36, :, 25:35] = materials.get("photoresist").id  # 中央にマスク
        P.AtomicLayerEtch(targets=["oxide"], cycles=80, etch_per_cycle_nm=1.0,
                          anisotropy=aniso).apply(w)
        # マスク直下（x=30, z=29）の oxide が残っているか
        return g[29, 5, 30] == materials.get("oxide").id

    iso_intact = run(0.0)     # 等方: マスク下が抉られる → 残らない
    dir_intact = run(1.0)     # 指向性: マスク下は保護 → 残る
    assert dir_intact and not iso_intact


def test_ale_recipe_roundtrip():
    """to_dict/from_dict でパラメータが保存・復元される。"""
    ale = P.AtomicLayerEtch(targets=["oxide", "poly"], cycles=15,
                            etch_per_cycle_nm=0.8, anisotropy=0.3)
    d = ale.to_dict()
    assert d["type"] == "ALE"
    restored = Process.from_dict(d)
    assert restored.targets == ["oxide", "poly"]
    assert restored.cycles == 15
    assert restored.etch_per_cycle_nm == 0.8
    assert restored.anisotropy == 0.3


def test_ale_in_available_types():
    """ALE が工程メニュー一覧に含まれる。"""
    types = dict(P.available_types())
    assert "ALE" in types
