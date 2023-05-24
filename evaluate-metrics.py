# coding=utf-8
import argparse
from pathlib import Path

from skimage.morphology import remove_small_objects
from skimage.filters import threshold_otsu
from scipy.ndimage import binary_fill_holes

import matplotlib.pyplot as plt


import h5py
import numpy as np
from runstats import Statistics

from mridc.collections.common.parts import utils
from mridc.collections.common.metrics.reconstruction_metrics import mse, nmse, psnr, ssim

METRIC_FUNCS = {"MSE": mse, "NMSE": nmse, "PSNR": psnr, "SSIM": ssim}


class Metrics:
    """Maintains running statistics for a given collection of metrics."""

    def __init__(self, metric_funcs):
        """
        Args:
            metric_funcs (dict): A dict where the keys are metric names and the
                values are Python functions for evaluating that metric.
        """
        self.metrics_scores = {metric: Statistics() for metric in metric_funcs}

    def push(self, target, recons):
        """
        Pushes a new batch of metrics to the running statistics.
        Args:
            target: target image
            recons: reconstructed image
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        for metric, func in METRIC_FUNCS.items():
            self.metrics_scores[metric].push(func(target, recons))

    def means(self):
        """
        Mean of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.mean() for metric, stat in self.metrics_scores.items()}

    def stddevs(self):
        """
        Standard deviation of the means of each metric.
        Returns:
            dict: A dict where the keys are metric names and the values are
        """
        return {metric: stat.stddev() for metric, stat in self.metrics_scores.items()}

    def __repr__(self):
        """
        Representation of the metrics.
        Returns:
            str: A string representation of the metrics.
        """
        means = self.means()
        stddevs = self.stddevs()
        metric_names = sorted(list(means))

        res = " ".join(
            f"{name} = {means[name]:.4g} +/- {2 * stddevs[name]:.4g}" for name in metric_names) + "\n"

        return res


def compute_brain_mask(volume, threshold=None, min_size=100):
    # Compute threshold value using Otsu's algorithm if not provided
    if threshold is None:
        threshold = threshold_otsu(volume)
        print("Calculated Otsu threshold:", threshold)

    # Apply threshold to the volume
    mask = np.zeros_like(volume, dtype=np.int64)
    mask[volume > threshold] = 1

    # Fill any holes inside the brain in all dimensions
    for i in range(mask.shape[2]):
        mask[:, :, i] = binary_fill_holes(mask[:, :, i])
    for i in range(mask.shape[1]):
        mask[:, i, :] = binary_fill_holes(mask[:, i, :])
    for i in range(mask.shape[0]):
        mask[i, :, :] = binary_fill_holes(mask[i, :, :])

    # Remove any small disconnected regions
    mask = mask > 0
    # mask = remove_small_objects(label(mask), min_size=min_size)
    mask = remove_small_objects(mask, min_size=min_size)
    mask = np.where(mask == True, 1.0, 0.0)

    return mask


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title("use scroll wheel to navigate images")

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(np.abs(self.X[:, :, self.ind]))
        self.update()

    def onscroll(self, event):
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(np.abs(self.X[:, :, self.ind]))
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


def plot_volume(volume_data):
    # Check if values are complex
    if np.iscomplexobj(volume_data):

        # Take absolute value of volume data
        volume_data = np.abs(volume_data)

    # Check dimensions of volume data
    if len(volume_data.shape) != 3:

        # Remove dimension with length 1
        volume_data = np.squeeze(volume_data)

    fig, ax = plt.subplots(1, 1)
    plt.gray()

    tracker = IndexTracker(ax, volume_data)
    fig.canvas.mpl_connect("scroll_event", tracker.onscroll)

    plt.show()


def main(args):
    # if json file
    if args.targets_dir.endswith(".json"):
        import json

        with open(args.targets_dir, "r") as f:
            targets = json.load(f)
        targets = [Path(target) for target in targets]
    else:
        targets = list(Path(args.targets_dir).iterdir())

    scores = Metrics(METRIC_FUNCS)
    ssims = []
    for target in targets:
        reconstruction = h5py.File(Path(args.reconstructions_dir) / str(target).split("/")[-1], "r")["reconstruction"][
            ()
        ].squeeze()
        if "reconstruction_sense" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")[
                "reconstruction_sense"][()].squeeze()
        elif "reconstruction_rss" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")["reconstruction_rss"][()].squeeze()
        elif "reconstruction" in h5py.File(target, "r").keys():
            target = h5py.File(target, "r")["reconstruction"][()].squeeze()
        else:
            target = h5py.File(target, "r")["target"][()].squeeze()

        if args.mask:
            # Compute brain mask
            mask = compute_brain_mask(target, threshold=args.threshold)

            # Replace values outside the mask with NaN
            target = np.where(mask == 0, np.nan, target)

        if args.eval_per_slice:
            for sl in range(target.shape[0]):

                # Normalise slice
                target[sl] = target[sl] / np.max(np.abs(target[sl]))
                reconstruction[sl] = reconstruction[sl] / \
                    np.max(np.abs(reconstruction[sl]))

                # Take absolute value
                target[sl] = np.abs(target[sl])
                reconstruction[sl] = np.abs(reconstruction[sl])

                # Clip values to [0, 1]
                target[sl] = np.clip(target[sl], 0, 1)
                reconstruction[sl] = np.clip(reconstruction[sl], 0, 1)

                # Calculate metrics
                scores.push(target[sl], reconstruction[sl])

        else:
            # Normalise
            target = target / np.max(np.abs(target))
            reconstruction = reconstruction / np.max(np.abs(reconstruction))

            # Take absolute value
            target = np.abs(target)
            reconstruction = np.abs(reconstruction)

            # Clip values to [0, 1]
            target = np.clip(target, 0, 1)
            reconstruction = np.clip(reconstruction, 0, 1)

            # Calculate metrics
            scores.push(target, reconstruction)

    # Print the scores
    print(scores.__repr__()[:-1])


if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser()

    # Add arguments to the parser
    parser.add_argument("-tp", "--targets-path", type=str,
                        required=True, help="Path to targets directory")
    parser.add_argument("-rp", "--reconstructions-path", type=str,
                        required=True, help="Path to reconstructions directory")
    parser.add_argument("-eps", "--eval-per-slice", action="store_true",
                        default=False, help="Evaluate metrics per slice")
    parser.add_argument("-m", "--mask", type="store_true", default=False,
                        help="Only calculate metrics for masked region")
    parser.add_argument("-t", "--threshold", type=float, default=0.1,
                        help="Threshold value for segmentation and mask creation. If not provided, the threshold will be calculated using the Otsu method. Default: 0.1")

    # Parse the arguments
    args = parser.parse_args()

    # Print the arguments
    print("Running evaluation script with the following arguments:")
    for arg in vars(args):
        print(f"\t{arg}: {getattr(args, arg)}")

    # Run the main function
    main(args)
