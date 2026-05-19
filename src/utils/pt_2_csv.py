from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_pt_file(pt_path: Path) -> Any:
    """加载 .pt 文件，优先使用安全模式，必要时回退到兼容模式。"""
    try:
        import torch  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("pt_2_csv 需要先安装 PyTorch（torch）。") from e

    try:
        return torch.load(str(pt_path), map_location="cpu")
    except Exception as e:
        maybe_weights_only_issue = (
            "weights_only" in str(e)
            or "WeightsUnpickler" in str(e)
            or "Unsupported operand" in str(e)
        )
        if maybe_weights_only_issue:
            return torch.load(str(pt_path), map_location="cpu", weights_only=False)
        raise


def _to_serializable(value: Any) -> Any:
    """将 Tensor/ndarray/Path 等对象递归转换为可写入 CSV 的基础结构。"""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _to_serializable(v) for k, v in value.items()}

    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]

    if isinstance(value, set):
        return [_to_serializable(v) for v in sorted(value, key=repr)]

    detach = getattr(value, "detach", None)
    cpu = getattr(value, "cpu", None)
    if callable(detach) and callable(cpu):
        try:
            value = value.detach().cpu()
        except Exception:
            pass

    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            return _to_serializable(tolist())
        except Exception:
            pass

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _to_serializable(item())
        except Exception:
            pass

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")

    return str(value)


def pt_to_csv(
    pt_path: str | Path,
    csv_path: str | Path | None = None,
    *,
    column_name: str = "simulation_data",
) -> Path:
    """
    将 .pt 文件转换为 .csv 文件。

    当前导出格式与 outputs/simulation_data.csv 保持一致：
    - 只有一列表头，默认列名为 simulation_data
    - 第二行开始每一行是一个 JSON 字符串

    对于 q_sumo_ryl.pt 这类二维 Tensor，会把整个二维数组写成
    一条 JSON 文本，和 simulation_data.csv 中每次追加的内容一致。
    """
    src = Path(pt_path)
    if not src.exists():
        raise FileNotFoundError(f"pt 文件不存在: {src}")

    out = Path(csv_path) if csv_path is not None else src.with_suffix(".csv")
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    data = _to_serializable(_load_pt_file(src))

    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([column_name])
        writer.writerow([json.dumps(data, ensure_ascii=False)])

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="将 .pt 文件按 simulation_data.csv 的格式转换为 .csv")
    parser.add_argument(
        "pt_path",
        nargs="?",
        default="../../outputs/q_sumo_ryl.pt",
        help="输入的 .pt 文件路径，默认 ../../outputs/q_sumo_ryl.pt",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="csv_path",
        default="../../outputs/q_sumo_ryl.csv",
        help="输出的 .csv 文件路径，默认 ../../outputs/q_sumo_ryl.csv",
    )
    parser.add_argument(
        "--column-name",
        default="simulation_data",
        help="CSV 表头名称，默认 simulation_data",
    )
    args = parser.parse_args()

    out_path = pt_to_csv(args.pt_path, args.csv_path, column_name=args.column_name)
    print(f"CSV 文件已生成: {out_path}")


if __name__ == "__main__":
    main()
