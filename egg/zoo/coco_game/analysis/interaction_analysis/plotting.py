import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing


def plot_confusion_matrix(df: pd.DataFrame, title, use_scaler=True):
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
    plt.show()
