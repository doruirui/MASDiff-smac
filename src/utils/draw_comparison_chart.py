from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _load_last_json_cell(csv_path: str | Path) -> Any:
    """Load the last non-header JSON cell from a single-column CSV file."""
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))

    if len(rows) < 2:
        raise ValueError(f"No data rows found in CSV: {path}")

    last_row = rows[-1]
    if not last_row:
        raise ValueError(f"Last row is empty in CSV: {path}")

    return json.loads(last_row[0])


def _extract_last_vector(data: Any, *, source_name: str) -> list[float]:
    """
    Extract a 1D vector for plotting.

    Rules:
    - if data is 2D, use the last row;
    - if data is already 1D, use it directly.
    """
    if not isinstance(data, list) or not data:
        raise ValueError(f"Invalid data in {source_name}: expected a non-empty list")

    if isinstance(data[0], list):
        last_row = data[-1]
        if not isinstance(last_row, list) or not last_row:
            raise ValueError(f"Invalid 2D data in {source_name}: last row is empty")
        return [float(x) for x in last_row]

    return [float(x) for x in data]


def draw_comparison_chart(
    q_csv_path: str | Path,
    simulation_csv_path: str | Path,
    output_path: str | Path | None = None,
    *,
    show: bool = False,
) -> Path:
    """
    Draw a comparison chart between:
    - the last data item in q_sumo_ryl.csv
    - the last data item in simulation_data.csv

    If each item is a 2D array, this function compares their last rows.
    """
    q_data = _load_last_json_cell(q_csv_path)
    simulation_data = _load_last_json_cell(simulation_csv_path)

    q_vector = _extract_last_vector(q_data, source_name="q_sumo_ryl.csv")
    sim_vector = _extract_last_vector(simulation_data, source_name="sumo_ryl_only_astar_best_rho_history_best_simulation_data.csv")

    if len(q_vector) != len(sim_vector):
        raise ValueError(
            f"Vector length mismatch: q={len(q_vector)}, simulation={len(sim_vector)}"
        )

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as e:  # pragma: no cover
        raise ImportError("matplotlib is required for draw_comparison_chart.py") from e

    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    indices = list(range(len(q_vector)))
    diff_vector = [sim - q for q, sim in zip(q_vector, sim_vector)]
    mse = sum((sim - q) ** 2 for q, sim in zip(q_vector, sim_vector)) / len(q_vector)
    mae = sum(abs(sim - q) for q, sim in zip(q_vector, sim_vector)) / len(q_vector)

    out = Path(output_path) if output_path is not None else Path("outputs/q_vs_simulation_last.png")
    if out.parent != Path("."):
        out.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)

    axes[0].plot(indices, q_vector, label="Q last row", linewidth=1.8, color="#1f77b4")
    axes[0].plot(indices, sim_vector, label="Simulation last row", linewidth=1.8, color="#ff7f0e")
    axes[0].set_ylabel("Value")
    axes[0].set_title(f"Q vs Simulation (last data) | MSE={mse:.4f}, MAE={mae:.4f}")
    axes[0].grid(True, linestyle="--", alpha=0.35)
    axes[0].legend()

    axes[1].bar(indices, diff_vector, color="#2ca02c", alpha=0.85, width=0.8)
    axes[1].axhline(0.0, color="black", linewidth=1.0)
    axes[1].set_xlabel("Road index")
    axes[1].set_ylabel("Simulation - Q")
    axes[1].set_title("Difference")
    axes[1].grid(True, axis="y", linestyle="--", alpha=0.35)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Draw a comparison chart for the last data in q_sumo_ryl.csv and simulation_data.csv"
    )
    parser.add_argument(
        "--q-csv",
        default="outputs/Q_nov.csv",
        help="Path to q_sumo_ryl.csv",
    )
    parser.add_argument(
        "--simulation-csv",
        default="outputs/sumo_ryl_only_astar_nov_best_rho_history_best_simulation_data.csv",
        help="Path to simulation_data.csv",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="outputs/q_vs_simulation_last.png",
        help="Output image path",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the chart after saving it",
    )
    args = parser.parse_args()

    out_path = draw_comparison_chart(
        q_csv_path=args.q_csv,
        simulation_csv_path=args.simulation_csv,
        output_path=args.output,
        show=args.show,
    )
    print(f"Chart saved to: {out_path}")


if __name__ == "__main__":
    main()
