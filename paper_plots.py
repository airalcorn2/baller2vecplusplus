import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.stats import pearsonr

font = {"size": 22}

matplotlib.rc("font", **font)


def plot_nll_over_time():
    # From: https://stackoverflow.com/a/59421062/1316276.
    # Numbers of pairs of bars you want.
    N = 20

    # Data on x-axis.

    # Specify the values of blue bars (height).
    blue_bar = [
        1.86897819,
        0.58918497,
        0.49199169,
        0.47602542,
        0.45367828,
        0.43433007,
        0.4377114,
        0.43828331,
        0.41888807,
        0.42203086,
        0.42575513,
        0.41059803,
        0.42874022,
        0.41963211,
        0.42825773,
        0.42731653,
        0.44892668,
        0.44514571,
        0.43499798,
        0.46455422,
    ]
    # Specify the values of orange bars (height).
    orange_bar = [
        1.56743299,
        0.54410327,
        0.46949427,
        0.45175594,
        0.42762741,
        0.41282431,
        0.41227819,
        0.41077594,
        0.38996656,
        0.39531277,
        0.40007369,
        0.37965269,
        0.39489294,
        0.3802205,
        0.39562016,
        0.39000066,
        0.40920398,
        0.39750678,
        0.39637249,
        0.41968062,
    ]

    # Position of bars on x-axis.
    ind = np.arange(N)

    # Figure size.
    plt.figure(figsize=(10, 5))

    # Width of a bar.
    width = 0.3

    # Plotting.
    plt.bar(ind, blue_bar, width, label="baller2vec")
    plt.bar(ind + width, orange_bar, width, label="baller2vec++")

    plt.xlabel("Time step")
    plt.ylabel("Average NLL")

    # xticks().
    # First argument: a list of positions at which ticks should be placed.
    # Second argument: a list of labels to place at the given locations.
    plt.xticks(ind + width / 2, list(range(1, N + 1)))

    # Finding the best position for legends and putting it there.
    plt.legend(loc="best", prop={"family": "monospace"})
    plt.show()


def plot_permutations():
    df = pd.read_csv("test.csv")
    print(pearsonr(df["anchor"], df["shuffled"]))
    plt.scatter(df["anchor"], df["shuffled"])
    plt.xlabel("Unshuffled average NLL")
    plt.ylabel("Shuffled average NLL")
    plt.show()
