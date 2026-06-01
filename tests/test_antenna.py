"""アンテナ比（プラズマ帯電損傷 DRC）のテスト。"""
from semisim import materials
from semisim import metrology as M
from semisim.grid import Wafer, WaferConfig


def _build(metal_half_width: int) -> Wafer:
    """小さいゲート酸化膜の上に幅可変の金属アンテナ電極を載せる。"""
    cfg = WaferConfig(nx=80, ny=80, nz=30, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    g = w.grid
    g[:5, :, :] = materials.get("silicon").id
    g[5:7, 38:42, 38:42] = materials.get("oxide").id  # ゲート酸化膜 4x4
    a, b = 40 - metal_half_width, 40 + metal_half_width
    g[7:10, a:b, a:b] = materials.get("metal_al").id  # 金属アンテナ
    return w


def test_antenna_ratio_increases_with_metal_area():
    """金属アンテナが大きいほどアンテナ比が増える。"""
    r_small = M.antenna_ratio(_build(2), "metal_al", "oxide")["ratio"]
    r_mid = M.antenna_ratio(_build(10), "metal_al", "oxide")["ratio"]
    r_big = M.antenna_ratio(_build(30), "metal_al", "oxide")["ratio"]
    assert 0 < r_small < r_mid < r_big


def test_antenna_ratio_fail_above_limit():
    """大きいアンテナで限界超過 fail、小さいアンテナは合格。"""
    assert not M.antenna_ratio(_build(2), "metal_al", "oxide")["fail"]
    big = M.antenna_ratio(_build(30), "metal_al", "oxide", ratio_limit=400.0)
    assert big["fail"] and big["ratio"] > 400.0


def test_antenna_ratio_no_gate_contact_zero():
    """ゲート酸化膜に接続が無ければ ratio=0。"""
    cfg = WaferConfig(nx=40, ny=40, nz=20, pitch_um=0.05, substrate_um=0.0)
    w = Wafer(cfg)
    w.grid[:5, :, :] = materials.get("silicon").id
    w.grid[10:13, 10:20, 10:20] = materials.get("metal_al").id  # ゲートと非接触
    res = M.antenna_ratio(w, "metal_al", "oxide")
    assert res["ratio"] == 0.0 and not res["fail"]


def test_antenna_ratio_custom_limit():
    """ratio_limit を厳しくすると fail しやすくなる。"""
    w = _build(10)
    assert not M.antenna_ratio(w, "metal_al", "oxide", ratio_limit=1000.0)["fail"]
    assert M.antenna_ratio(w, "metal_al", "oxide", ratio_limit=10.0)["fail"]
