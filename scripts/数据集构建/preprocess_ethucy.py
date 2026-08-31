"""ETH/UCY 数据预处理脚本。

将原始 ETH/UCY .txt 文件转换为按场景组织的 .pt 样本文件。

Usage:
    python scripts/数据集构建/preprocess_ethucy.py \
        --data_root /path/to/ethucy \
        --output_root data/ETHUCY_processed \
        --frame_stride 10 \
        --obs_len 8 \
        --pred_len 12
"""

from argparse import ArgumentParser
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.datamodule.ethucy_extractor import EthUcyExtractor


def preprocess(
    data_root: str,
    output_root: str = "data/ETHUCY_processed",
    frame_stride: int = 10,
    obs_len: int = 8,
    pred_len: int = 12,
) -> None:
    _data_root = Path(data_root)
    _output_root = Path(output_root)
    _output_root.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {_data_root}")
    print(f"Output root: {_output_root}")
    print(f"Frame stride: {frame_stride}, obs_len: {obs_len}, pred_len: {pred_len}")

    extractor = EthUcyExtractor(
        save_path=_output_root,
        frame_stride=frame_stride,
        obs_len=obs_len,
        pred_len=pred_len,
    )

    files = EthUcyExtractor.glob_files(_data_root)
    print(f"Found {len(files)} .txt files")

    for f in files:
        extractor.save(f)

    print("Done!")


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Preprocess ETH/UCY raw data into .pt samples"
    )
    parser.add_argument(
        "--data_root", "-d", type=str, required=True,
        help="Path to root directory containing ETH/UCY .txt files"
    )
    parser.add_argument(
        "--output_root", "-o", type=str, default="data/ETHUCY_processed",
        help="Output directory for processed .pt files"
    )
    parser.add_argument(
        "--frame_stride", type=int, default=10,
        help="Downsampling stride (default: 10, ~2.5 Hz)"
    )
    parser.add_argument(
        "--obs_len", type=int, default=8,
        help="Number of historical frames (default: 8)"
    )
    parser.add_argument(
        "--pred_len", type=int, default=12,
        help="Number of future frames (default: 12)"
    )

    args = parser.parse_args()
    preprocess(
        data_root=args.data_root,
        output_root=args.output_root,
        frame_stride=args.frame_stride,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
    )