"""Plot run history without coupling visualisation to the training loop."""

from pathlib import Path

import pandas as pd


def plot_history(history_csv: Path, output_path: Path) -> None:
    """Plot total/data/physics loss against epoch for one saved run."""
    import matplotlib.pyplot as plt

    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(7, 4))
    for column in ("total_loss", "data_loss", "physics_loss"):
        if column in history:
            ax.semilogy(history["epoch"], history[column], label=column)
    ax.set(xlabel="Epoch", ylabel="Loss")
    ax.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
