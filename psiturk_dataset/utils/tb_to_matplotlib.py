import argparse
import numpy as np
# from tensorflow.python.summary.event_accumulator import EventAccumulator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage.filters import gaussian_filter1d


import matplotlib as mpl
import matplotlib.pyplot as plt

def plot_tensorflow_log(path, path_2):

    # Loading too much data is slow...
    tf_size_guidance = {
        'compressedHistograms': 10,
        'images': 0,
        'scalars': 100,
        'histograms': 1
    }

    event_acc = EventAccumulator(path, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    print(event_acc.Tags())

    training_accuracies =   event_acc.Scalars('eval_metrics')

    event_acc = EventAccumulator(path_2, tf_size_guidance)
    event_acc.Reload()
    validation_accuracies = event_acc.Scalars('eval_metrics')

    steps = 10
    x = np.arange(steps) + 1
    y = np.zeros([steps, 2])

    for i in range(steps):
        y[i, 0] = training_accuracies[i][2] # value
        y[i, 1] = validation_accuracies[i][2]
    y[:, 0] = gaussian_filter1d(y[:, 0], sigma=2)
    y[:, 1] = gaussian_filter1d(y[:, 0], sigma=2)
    print(x)
    print(y[:, 0])
    # poly = np.polyfit(x,y[:, 0],5)
    # poly_y = np.poly1d(poly)(x)
    # y[:, 0] = poly_y

    # poly = np.polyfit(x,y[:, 1],5)
    # poly_y = np.poly1d(poly)(x)
    # y[:, 1] = poly_y

    plt.plot(x, y[:,0], label='Validation success')
    plt.plot(x, y[:,1], label='Validation spl')

    plt.xlabel("Steps (experience) in millions")
    plt.ylabel("Performance (higher is better)")
    plt.title("Training Progress")
    plt.legend(bbox_to_anchor=(1.04, 0), loc="lower left", borderaxespad=0)
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log", type=str, default="replays/demo_1.json.gz"
    )
    parser.add_argument(
        "--log2", type=str, default="replays/demo_1.json.gz"
    )
    args = parser.parse_args()
    plot_tensorflow_log(args.log, args.log2)