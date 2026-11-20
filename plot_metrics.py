import argparse
import json
from pathlib import Path
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics-path", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    return args


def load_metrics(path: str | Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            records.append(json.loads(line))
    return records


def plot_loss_curves(records, output_dir) -> None:
    train_records = []
    validation_records = []

    for record in records:
        if record["split"] == "train":
            train_records.append(record)
        else:
            validation_records.append(record)

    train_y = [r["loss"] for r in train_records]
    validation_y = [r["loss"] for r in validation_records]


    plot_specs = [("iteration", "Completed updates", "loss_by_iteration.png"),
                  ("elapsed_seconds", "Elapsed time (s)", "loss_by_time.png")]
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    for x_field, x_label, filename in plot_specs:

        train_x = [r[x_field] for r in train_records]
        validation_x = [r[x_field] for r in validation_records]

        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(train_x, train_y, label="Train")
        ax.plot(validation_x, validation_y, label="Validation", marker="o")
        ax.set_xlabel(x_label)
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title("Training and validation loss")
        ax.legend()

        fig.tight_layout()
        output_path = Path(output_dir) / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)


def main():
    args = parse_args()
    records = load_metrics(args.metrics_path)
    plot_loss_curves(records, args.output_dir)
if __name__ == "__main__":
    main()
