import numpy as np
from matplotlib import pyplot


def inrange(val, mid, epsilon=0.01):
    return val - epsilon <= mid <= val + epsilon


def normal_plot(epocs):
    counter = 0

    for idx in range(len(epocs['test'])):
        train = epocs['train'][idx]
        test = epocs['test'][idx]

        train_len = len(train) + counter
        test_len = train_len + len(test)

        pyplot.scatter(range(counter, train_len), train, s=3, c="b")
        pyplot.scatter(range(train_len, test_len), test, s=3, c="r")
        pyplot.vlines(test_len + 1, min(train + test), max(train + test))
        pyplot.hlines(np.mean(train), counter, test_len, colors="orange", linestyles="dashed")
        pyplot.hlines(np.mean(test), counter, test_len, colors="r", linestyles="dashed")
        counter = test_len + 1
        # pyplot.show()


def condensed_plot(epocs):
    train = [np.mean(e) for e in epocs['train']]
    test = [np.mean(e) for e in epocs['test']]

    train_range = range(len(train))
    test_range = range(len(test))
    pyplot.plot(train_range, train, c="b")
    pyplot.plot(test_range, test, c="r")

    pyplot.scatter(train_range, train, s=10, c="b")
    pyplot.scatter(test_range, test, s=10, c="r")

    # pyplot.show()


def epochs_extractor(file):
    with open(file, "r") as f:
        lines = f.readlines()

    epocs = dict(
        train=[],
        test=[]
    )
    new_lines = []
    for l in lines:

        try:
            new_lines.append(float(l))
        except ValueError:

            l = eval(l)
            mean = np.mean(new_lines)

            if not inrange(mean, l['accuracy']): print(f"Logger accuracy {l['accuracy']} vs mean {mean}")
            epocs[l['mode']].append(new_lines)
            new_lines = []

        except NameError:
            continue

    return epocs


def condensed_epochs_extractor(file):
    with open(file, "r") as f:
        lines = f.readlines()

    epocs = dict(
        train=[],
        test=[]
    )
    for l in lines:

        l = eval(l)

        if isinstance(l, dict):
            epocs[l['mode']].append(l['accuracy'])

    return epocs


if __name__ == '__main__':
    file = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/std_out.txt"

    epocs = condensed_epochs_extractor(file)

    condensed_plot(epocs)

    pyplot.ylabel("Accuracy")
    pyplot.xlabel("Batches")
    pyplot.title("Train computed after learning")
    pyplot.show()
