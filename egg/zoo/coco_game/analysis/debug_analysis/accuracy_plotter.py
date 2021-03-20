import numpy as np
from matplotlib import pyplot


def inrange(val, mid, epsilon=0.01):
    return val - epsilon <= mid <= val + epsilon


if __name__ == '__main__':

    file = "/home/dizzi/Desktop/EGG/egg/zoo/coco_game/std_out.txt"
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

            assert inrange(mean, l['accuracy'])
            epocs[l['mode']].append(new_lines)
            new_lines = []

        except NameError:
            continue

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

        a = 1

    pyplot.ylabel("Accuracy")
    pyplot.xlabel("Batches")
    pyplot.title("Train as Validation Data")
    pyplot.show()
