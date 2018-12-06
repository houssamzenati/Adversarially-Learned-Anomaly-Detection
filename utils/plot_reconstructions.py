import numpy as np
from utils.evaluations import save_results_csv
import time
import os
import matplotlib.pyplot as plt


def plot_hist_dis_reconstructions():
    walk_dir = 'scores'
    label = []
    for root, _, files in os.walk(walk_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            try:
                array = np.vstack(
                    [array, np.genfromtxt(file_path, delimiter=',')])
            except NameError:
                array = np.genfromtxt(file_path, delimiter=',')

            label += [file_path.split('/')[-3] + '-' + file_path.split('/')[-2]]

    hrange = (np.min(array), np.max(array))

    for i in range(array.shape[0]):
        plt.figure()
        plt.hist(array[i, :], 50,
                 label=label[i], density=True, range=hrange)
        plt.title(label[i])
        plt.savefig(walk_dir + '/' + label[i] + '.png')
        plt.close()

if __name__ == "__main__":
    plot_hist_dis_reconstructions()