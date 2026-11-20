import argparse
import json
from pathlib import Path

# 在导入 pyplot 前设置 Agg 后端
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 从已有文件中复用 load_metrics 函数
from plot_metrics import load_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare validation curves across multiple training runs."
    )
    parser.add_argument(
        "--run-dirs",
        nargs="+",
        required=True,
        type=str,
        help="List of directories containing the run results.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Directory to save the comparison plots.",
    )
    return parser.parse_args()


def plot_validation_comparison(run_dirs: list[str | Path], output_dir: str | Path) -> None:
    runs = []

    # 1. 读取并汇总所有运行数据
    for run_dir in run_dirs:
        run_path = Path(run_dir)
        config_path = run_path / "config.json"
        metrics_path = run_path / "metrics.jsonl"

        with open(config_path, "r", encoding="utf-8") as file:
            config = json.load(file)

        max_learning_rate = float(config["max_learning_rate"])
        label = f"max_lr={max_learning_rate}"

        records = load_metrics(metrics_path)
        validation_records = [record for record in records if record["split"] == "validation"]

        runs.append({
            "label": label,
            "records": validation_records
        })

    # 2. 创建输出目录
    output_dir_path = Path(output_dir)
    # parents=True 自动创建父目录，exist_ok=False 确保目录唯一
    output_dir_path.mkdir(parents=True, exist_ok=False)

    # 3. 绘制两张对照图
    plot_specs = [
        ("iteration", "Completed updates", "validation_loss_by_iteration.png"),
        ("elapsed_seconds", "Elapsed time (s)", "validation_loss_by_time.png"),
    ]

    for x_field, x_label, filename in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))

        for run in runs:
            # 提取横坐标和纵坐标
            x_values = [record[x_field] for record in run["records"]]
            loss_values = [record["loss"] for record in run["records"]]

            # 绘制折线，并用 'o' 标记观测点
            ax.plot(x_values, loss_values, label=run["label"], marker="o")

        ax.set_xlabel(x_label)
        ax.set_ylabel("Cross-entropy loss")
        ax.legend()
        fig.tight_layout()

        output_path = output_dir_path / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    plot_validation_comparison(args.run_dirs, args.output_dir)


if __name__ == "__main__":
    main()
