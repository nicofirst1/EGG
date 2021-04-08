from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def plot_confusion_matrix(
    df: pd.DataFrame, title, save_dir: Path, use_scaler=True, show=True
):
    fig, ax = plt.subplots()
    plt.yticks(np.arange(0.9, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)

    if use_scaler:
        x = df.values  # returns a numpy array
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled)

    plt.pcolor(df)
    fig.suptitle(title, fontsize=15)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5)
    # fig.autofmt_xdate()
    plt.savefig(save_dir.joinpath(f"{title}.jpg"))

    if show:
        plt.show()

    plt.close()


def plot_multi_scatter(plot_list, save_dir: Path, max_plot_figure=4, show=True):
    figures = len(plot_list) // max_plot_figure
    square_len = max_plot_figure // 2
    idx = 0
    for _ in range(figures):
        fig, axs = plt.subplots(square_len, square_len)

        for jdx in range(square_len):
            for kdx in range(square_len):
                to_plot = plot_list[idx]
                n1, n2, p1, p2, corr = to_plot

                axs[jdx, kdx].scatter(
                    p1,
                    p2,
                    s=10,
                )
                axs[jdx, kdx].set_xlabel(n1)
                axs[jdx, kdx].set_ylabel(n2)
                axs[jdx, kdx].set_title(f"Corr {corr:.3f}")
                idx += 1

        fig.tight_layout()
        fig.savefig(save_dir.joinpath(f"fig{idx}.jpg"))
        if show:
            plt.show()
        plt.close()
