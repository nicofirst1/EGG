import numpy as np
from matplotlib import pyplot


def inrange(val, mid, epsilon=0.01):
    return val - epsilon <= mid <= val + epsilon


if __name__ == '__main__':

    file = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/pred.txt"
    with open(file, "r") as f:
        lines = f.readlines()

    epocs = dict(
        train=[],
        test=[]
    )
    new_lines = []
    for l in lines:

        try:
            new_lines.append([float(e) for e in l.split()])
        except ValueError:

            l = eval(l)
            epocs[l['mode']].append(new_lines)
            new_lines = []

        except NameError:
            continue

    counter = 0
    for idx in range(len(epocs['test'])):
        train = epocs['train'][idx]
        test = epocs['test'][idx]

        train_plot = [np.mean(e) for e in train]
        test_plot = [np.mean(e) for e in test]

        train_len = len(train) + counter
        test_len = train_len + len(test)

        pyplot.scatter(range(counter, train_len), train_plot, s=3, c="b")
        pyplot.scatter(range(train_len, test_len), test_plot, s=3, c="r")
        pyplot.vlines(test_len + 1, min(train_plot + test_plot), max(train_plot + test_plot))
        pyplot.hlines(np.mean(train_plot), counter, test_len, colors="orange", linestyles="dashed")
        pyplot.hlines(np.mean(test_plot), counter, test_len, colors="r", linestyles="dashed")
        counter = test_len + 1
        # pyplot.show()

        a = 1

    pyplot.ylabel("Accuracy")
    pyplot.xlabel("Batches")
    pyplot.title("Distractors=2")
    pyplot.show()
