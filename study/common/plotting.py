from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_r2_barplot(df: pd.DataFrame, output_path: str | Path, value_col: str = "r2_test", label_col: str = "property") -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_df = df.sort_values(value_col, ascending=False)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(plot_df))))
    ax.barh(plot_df[label_col], plot_df[value_col])
    ax.invert_yaxis()
    ax.set_xlabel(value_col)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
