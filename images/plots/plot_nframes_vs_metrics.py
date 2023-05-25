"""Plot the each metric of the model as a function of the clip length."""

import argparse
import logging
import os

import matplotlib.pyplot as plt
import numpy as np


def nearest_neighbour(y: np.ndarray, window_size: int) -> np.ndarray:
    """Apply a nearest neighbour filter to a function defined by f(x) = y.

    Args:
        y (np.ndarray): The function's output values.
        window_size (int): The size of the window to use for the filter.

    Returns:
        np.ndarray: The nearest neighbour smoothed output values.
    """
    smoothed_y = []

    for i in range(len(y)):
        # Get the values in the window.
        window = y[max(0, i - window_size // 2) : i + (window_size + 1) // 2]

        # Get the mean of the window.
        smoothed_y.append(np.mean(window))

    return np.array(smoothed_y)


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply a Gaussian filter to a function defined by f(x) = y.

    Args:
        x (np.ndarray): The function's input values.
        y (np.ndarray): The function's output values.
        sigma (float): The standard deviation of the Gaussian filter.

    Returns:
        np.ndarray: The Gaussian kernel smoothed output values.
    """
    smoothed_y = []

    for x_i in x:
        # Create the Gaussian kernel. These are the weights that will be
        # applied to the output values. Source:
        # https://en.wikipedia.org/wiki/Kernel_smoother
        kernel = np.exp(-((x_i - x) ** 2) / (2 * sigma**2))

        # Normalize the kernel.
        kernel /= np.sum(kernel)

        # Apply the kernel to the data.
        smoothed_y.append(np.sum(y * kernel))

    return np.array(smoothed_y)


def main(args: argparse.Namespace) -> None:
    """The entry point of the program.

    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # First read the data file. It contains a dict with the following format:
    # {
    #     "Bleu_1": [(nframes_clip_1, Bleu_1_clip_1), ...],
    #     "Bleu_2": [(nframes_clip_1, Bleu_2_clip_1), ...],
    #     "Bleu_3": [(nframes_clip_1, Bleu_3_clip_1), ...],
    #     "Bleu_4": [(nframes_clip_1, Bleu_4_clip_1), ...],
    #     "METEOR": [(nframes_clip_1, METEOR_clip_1), ...],
    #     "Rouge": [(nframes_clip_1, Rouge_clip_1), ...],
    # }
    with open(args.path_metric_data, "r") as f:
        lines = f.read()
    metric_results = eval(lines)

    # Get the base name of the file (i.e. remove the extension name).
    base_name = os.path.splitext(os.path.basename(args.path_metric_data))[0]

    # Plot each metric as a function of the clip length.
    for i, (metric, results) in enumerate(metric_results.items()):
        # Sort the results by the clip length.
        nframes, metric_values = zip(*sorted(results))
        nframes = np.array(nframes)
        metric_values = np.array(metric_values)
        max_nframes = nframes.max()

        if i == 0:
            # Show the frequency of the clip lengths as a histogram.
            plt.hist(nframes, bins=20, histtype="bar", linewidth=2)
            plt.title(f"Clip Length Frequency - {base_name}")
            plt.xlabel("Clip Length (frames)")
            plt.ylabel("Frequency")
            plt.xlim(0, max_nframes)
            plt.ylim(bottom=0)
            plt.savefig(f"images/plots/{base_name}_nframes_frequency.png")
            plt.clf()

        if args.smoothing_type == "nearest_neighbour":
            # Plot the metric as a running average produced by a nearest
            # neighbour filter.
            metric_values = nearest_neighbour(metric_values, args.window_size)
        elif args.smoothing_type == "gaussian_kernel":
            # Plot the metric as a running average produced by a Gaussian
            # kernel filter.
            metric_values = gaussian_kernel(nframes, metric_values, args.sigma)

        # Plot the metric as a function of the video clip length.
        plt.plot(nframes, metric_values, linewidth=2, label=metric)

    # Save the plot.
    plt.title(
        f"Metric vs. Clip Length - {base_name}, "
        + (
            f"window size = {args.window_size}"
            if args.smoothing_type == "nearest_neighbour"
            else f"sigma = {args.sigma}"
        )
    )
    plt.xlabel("Clip Length (frames)")
    plt.ylabel("Metric Score")
    plt.xlim(0, max_nframes)
    plt.ylim(bottom=0)
    plt.legend(loc="upper right")
    plt.savefig(
        f"images/plots/{base_name}_nframes_vs_metric_"
        + (
            f"window_size_{args.window_size}"
            if args.smoothing_type == "nearest_neighbour"
            else f"sigma_{args.sigma}"
        )
        + ".png"
    )
    plt.clf()


if __name__ == "__main__":
    # Set up logging.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    # Create the argument parser.
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Define command line arguments.
    parser.add_argument(
        "--path_metric_data",
        type=str,
        default="images/plots/data/metrics_mem_all.txt",
        help="Path to the file containing the metric data.",
    )
    parser.add_argument(
        "--smoothing_type",
        type=str,
        choices=["nearest_neighbour", "gaussian_kernel"],
        default="nearest_neighbour",
        help="The type of smoothing to apply to the metric data.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=40,
        help="The window size for the nearest neighbour filter.",
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="The standard deviation for the Gaussian kernel filter.",
    )

    # Parse the arguments.
    args = parser.parse_args()
    main(args)
