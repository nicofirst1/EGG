import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from rich.progress import track

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


def plot_multi_bar(df1, df2, models, intensity, save_dir, superclass=False):
    def filter_patches(patches):
        """
        Get the patch with the max height
        """
        for idx in range(len(patches)):
            p = patches[idx]
            x = p[0]
            y = p[1]

            if abs(x._height) > abs(y._height):
                patches[idx] = x
            else:
                patches[idx] = y
        return patches

    models = list(models)

    for col_id in track(sorted(df1.columns), description="Plotting comparison..."):
        # get data
        col1 = df1[col_id]
        col2 = df2[col_id]
        col3 = intensity[col_id]

        bool_col = col3 > -1
        col1 = col1[bool_col]
        col2 = col2[bool_col]
        col3 = col3[bool_col]

        # define df
        df = pd.DataFrame([col1, col2], index=models)
        df['model'] = models
        df = pd.melt(df, id_vars="model", var_name="col", value_name="correlation")

        # plot
        g = sns.catplot(x='col', y='correlation', hue='model', data=df, kind='bar', aspect=1.5, )
        g.set_xticklabels(rotation=90)

        title = "[Superclass] " if superclass else "[Class] "
        title += col_id

        plt.suptitle(title)
        plt.gcf().subplots_adjust(bottom=0.3)

        # add text
        ax = g.facet_axis(0, 0)
        patches1 = ax.patches[:len(col1)]
        patches2 = ax.patches[len(col1):]
        patches = filter_patches(list(zip(patches1, patches2)))

        idx = 0
        for p in patches:
            x = p.get_x() - 0.01
            y = p.get_height() + 0.02 if p.get_height() > 0 else p.get_height() - 0.07
            text = f"{col3.iloc[idx] * 100:.3f}%"
            ax.text(x, y, text, color='black', rotation='horizontal', size=7)
            idx += 1

        # save
        plt.savefig(save_dir.joinpath(f"{col_id}.jpg"))
        # plt.show()
        plt.close()


def plot_multi_bar4(df_class, df_superclass, models, save_dir):
    models = list(models) * 2

    models[0] = f"class {models[0]}"
    models[1] = f"class {models[1]}"

    models[2] = f"superclass {models[2]}"
    models[3] = f"superclass {models[3]}"

    df1_class, df2_class = df_class
    df1_superclass, df2_superclass = df_superclass

    for col_id in track(sorted(df1_superclass.columns), description="Plotting comparison4..."):
        # get data
        try:
            cc1 = df1_class[col_id]
            cc2 = df2_class[col_id]
            csc1 = df1_superclass[col_id]
            csc2 = df2_superclass[col_id]
        except KeyError:
            continue
        # define df
        df = pd.DataFrame([cc1, cc2, csc1, csc2], index=models)
        df = df[df <= 1].dropna(axis=1)
        df['model'] = models
        df = pd.melt(df, id_vars="model", var_name="metrics", value_name="correlation")

        # plot
        g = sns.catplot(x='metrics', y='correlation', hue='model', data=df, kind='bar', aspect=1.5, )
        g.set_xticklabels(rotation=90)

        title = col_id

        plt.suptitle(title)
        plt.gcf().subplots_adjust(bottom=0.35)

        # save
        plt.savefig(save_dir.joinpath(f"{col_id}.jpg"))
        # plt.show()
        plt.close()
