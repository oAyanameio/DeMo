"""ethucy_benchmark_preprocess 导入路径回归测试（不生成真实数据集）。

验证 9/3 仓库重组后（build_ethucy_manifest.py 移入 scripts/数据集构建/），
preprocess() 在需要自动创建 source_manifest 时能定位到 manifest 构建模块。
"""

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def test_manifest_module_locatable_from_any_cwd():
    """从任意工作目录执行时，模块搜索路径应能找到 build_ethucy_manifest.py。"""
    src_file = REPO / "src" / "datamodule" / "ethucy_benchmark_preprocess.py"
    mod_dir = os.path.dirname(os.path.abspath(src_file))
    cand = [
        os.path.join(mod_dir, "..", "..", "scripts", "数据集构建"),
        os.path.join(mod_dir, "..", "scripts"),
    ]
    found = [d for d in cand if os.path.exists(os.path.join(d, "build_ethucy_manifest.py"))]
    assert found, f"build_ethucy_manifest.py 不在候选路径中: {cand}"


def test_manifest_module_importable_after_repo_reorg():
    """重组后的脚本目录在 sys.path 中时，模块可正常导入且暴露 build_manifest。"""
    script_dir = str(REPO / "scripts" / "数据集构建")
    assert os.path.exists(os.path.join(script_dir, "build_ethucy_manifest.py"))
    sys.path.insert(0, script_dir)
    try:
        import build_ethucy_manifest  # noqa: F401
        assert callable(getattr(build_ethucy_manifest, "build_manifest", None))
    finally:
        sys.path.remove(script_dir)


def test_preprocess_reads_existing_manifest(tmp_path):
    """已存在的 source_manifest 直接读取（不走自动构建分支），空 folds 收敛安全。

    构造与 build_ethucy_manifest.build_manifest 真实返回一致的键格式
    （fold 名不带 fold_ 前缀），验证 preprocess 主循环与其格式匹配。
    """
    import json

    fake_root = tmp_path / "raw"
    out_root = tmp_path / "out"

    folds = {f: {"train": [], "val": [], "test": []}
             for f in ("ETH", "HOTEL", "UNIV", "ZARA1", "ZARA2")}
    sm_path = out_root / "source_manifest.json"
    out_root.mkdir(parents=True)
    sm_path.write_text(json.dumps({
        "version": "ethucy_benchmark_v1_source",
        "raw_root": str(fake_root),
        "folds": folds,
    }))

    import src.datamodule.ethucy_benchmark_preprocess as pp
    manifest = pp.preprocess(str(fake_root), str(out_root), source_manifest=str(sm_path))

    assert manifest["version"] == "ethucy_benchmark_v1"
    assert manifest["dt"] == 0.4
    assert set(manifest["folds"].keys()) == set(folds.keys())
