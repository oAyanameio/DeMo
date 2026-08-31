import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
from build_ethucy_manifest import build_manifest, scene_of  # noqa: E402


@pytest.fixture
def raw_tree(tmp_path):
    """Create a minimal valid raw tree for one fold (ETH held out)."""
    def write(d, name, content="0.0\t1.0\t1.0\t1.0\n0.0\t2.0\t1.1\t1.0\n"):
        p = tmp_path / d
        p.mkdir(parents=True, exist_ok=True)
        (p / name).write_text(content)
    write("eth/train", "biwi_hotel_train.txt")
    write("eth/val", "biwi_hotel_val.txt")
    write("eth/test", "biwi_eth.txt")
    return tmp_path


def test_full_fold_manifest(raw_tree):
    m = build_manifest(str(raw_tree), ["ETH"])
    assert set(m["ETH"].keys()) == {"train", "val", "test"}
    test_items = m["ETH"]["test"]
    assert len(test_items) == 1
    assert test_items[0]["scene_id"] == "ETH"
    assert test_items[0]["num_source_rows"] == 2
    assert test_items[0]["num_source_frames"] == 1
    assert len(test_items[0]["sha256"]) == 64
    # train file biwi_hotel maps to HOTEL scene
    assert m["ETH"]["train"][0]["scene_id"] == "HOTEL"


def test_cross_split_duplicate_fails(raw_tree):
    # copy train file into val under same base name
    src = raw_tree / "eth" / "train" / "biwi_hotel_train.txt"
    (raw_tree / "eth" / "val" / "biwi_hotel_train.txt").write_text(src.read_text())
    with pytest.raises(ValueError, match="multiple splits"):
        build_manifest(str(raw_tree), ["ETH"])


def test_unknown_scene_file_fails(raw_tree):
    (raw_tree / "eth" / "train" / "mystery_scene_train.txt").write_text("0.0\t1.0\t0\t0\n")
    with pytest.raises(ValueError, match="Unknown scene"):
        build_manifest(str(raw_tree), ["ETH"])


def test_missing_test_dir_fails(raw_tree):
    import shutil
    shutil.rmtree(raw_tree / "eth" / "test")
    with pytest.raises(FileNotFoundError):
        build_manifest(str(raw_tree), ["ETH"])


def test_scene_mapping_explicit():
    assert scene_of("biwi_eth.txt") == "ETH"
    assert scene_of("crowds_zara03_train.txt") == "ZARA2"
    assert scene_of("uni_examples_val.txt") == "UNIV"
