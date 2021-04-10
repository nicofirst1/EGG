import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="ticks", color_codes=True)
from sklearn import preprocessing


def sort_dataframe(df, row_wise):
    if not row_wise:
        s = df.sum()
        df = df[s.sort_values(ascending=False).index]

    else:
        s = df.sum(axis=1)
        df = df.loc[s.sort_values(ascending=False).index]
    return df


def plot_confusion_matrix(
        df: pd.DataFrame, title, save_dir: Path, use_scaler=True, show=True
) -> object:
    fig, ax = plt.subplots()
    plt.yticks(np.arange(0.9, len(df.index), 1), df.index)
    plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
    x = df.values

    if use_scaler:
        min_max_scaler = preprocessing.MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        df = pd.DataFrame(x_scaled, index=df.index, columns=df.columns)

    annot = x if df.shape[0] * df.shape[1] < 100 else False
    sns.heatmap(df, annot=annot, fmt=".2f")  # , xticklabels=True, yticklabels=True)

    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(bottom=0.2)

    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5)

    title = title.replace(" ", "_")
    plt.savefig(save_dir.joinpath(f"{title}.jpg"))

    if show:
        plt.show()

    plt.close()


def plot_multi_scatter(plot_list, save_dir: Path, max_plot_figure=4, show=True):
    figures = math.ceil(len(plot_list) / max_plot_figure)
    square_len = max_plot_figure // 2
    idx = 0
    for _ in range(figures):
        fig, axs = plt.subplots(square_len, square_len)

        for jdx in range(square_len):
            for kdx in range(square_len):

                if idx >= len(plot_list): continue

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


def plot_histogram(
        data: pd.Series, title, save_dir: Path, show=True
):
    fig, ax = plt.subplots()

    data = pd.DataFrame(dict(classes=data.index, value=data.values))

    sns.barplot(x="classes", y="value", data=data)

    fig.suptitle(title, fontsize=15)
    fig.subplots_adjust(bottom=0.2)
    ax.tick_params(axis="x", labelrotation=90, labelsize=5)
    ax.tick_params(axis="y", labelrotation=0, labelsize=5)

    title = title.replace(" ", "_")
    plt.savefig(save_dir.joinpath(f"{title}.jpg"))

    if show:
        plt.show()

    plt.close()
