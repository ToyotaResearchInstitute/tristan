import os
import pathlib
from typing import Optional

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

MISSES_STAT_NAME = "misses"
MISSES_PARTIAL = "misses_partial"
MISSES_FULL_LIST = [MISSES_STAT_NAME, MISSES_STAT_NAME + "_x", MISSES_STAT_NAME + "_y"]
MISSES_PARTIAL_LIST = [MISSES_PARTIAL, MISSES_PARTIAL + "_x", MISSES_PARTIAL + "_y"]
MISSES_LIST = MISSES_FULL_LIST + MISSES_PARTIAL_LIST


def save_temporal_plot(x, y, y_std, title, filename, xlabel: Optional[str] = None, ylabel: Optional[str] = None):
    fig = plt.figure()
    fig.add_axes([0.05, 0.2, 0.75, 0.75])  # [left, bottom, width, height]
    plt.plot(x, y)
    for i in range(len(y)):
        plt.plot([x[i], x[i]], [y[i] + y_std[i], y[i] - y_std[i]], "o-")
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    image_format = pathlib.Path(filename).suffix.lstrip(".")
    fig.savefig(filename, format=image_format, dpi=600)
    plt.close()


def save_bar_plot(
    labels: list, values: list, title: str, filename: str, xlabel: Optional[str] = None, ylabel: Optional[str] = None
):
    """Basic helper for creating and saving bar plots."""
    fig = plt.figure()
    plt.bar(np.arange(len(values)), values, tick_label=labels)
    if title is not None:
        plt.title(title)
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)
    image_format = pathlib.Path(filename).suffix.lstrip(".")
    fig.savefig(filename, format=image_format, dpi=600)
    plt.close()


def plot_label_plots(stats, filename):
    """Plot the confusion matrices of semantic labels.

    Parameters
    ----------
    stats: dict, statistics to be plotted.
    filename: str, filename to save in, should be compatible with savefig.

    """
    num_label_types = len(stats["aggregate_statistics"]["labels"])
    if num_label_types == 0:
        return None
    fig, axes = plt.subplots(1, num_label_types, figsize=(3 * num_label_types, 3), sharey="row")
    for i, key in enumerate(stats["aggregate_statistics"]["labels"]):
        label_name = str(key)
        predicts = stats["aggregate_statistics"]["labels"][key]["predictions"]
        targets = stats["aggregate_statistics"]["labels"][key]["targets"]
        accuracy = np.round(accuracy_score(targets, predicts), 4)
        values = ["Yes", "No"]
        conf_mat = confusion_matrix(targets, predicts, labels=[True, False], normalize="true")
        disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=values)
        disp.plot(ax=axes[i])
        disp.ax_.set_title("Acc = {} ({})".format(accuracy, label_name), fontsize=6)
        disp.im_.colorbar.remove()
        disp.ax_.set_xlabel("Predicted label", fontsize=6)
        if i == 0:
            disp.ax_.set_ylabel("True label", fontsize=6)
        else:
            disp.ax_.set_ylabel("")
        disp.ax_.set_xticklabels(values, fontsize=6)
        disp.ax_.set_yticklabels(values, fontsize=6)

    image_format = pathlib.Path(filename).suffix.lstrip(".")
    fig.savefig(filename, format=image_format, dpi=600)
    plt.close()


def plot_statistics_plots(stats: dict, output_folder_name: str, img_ext):
    plot_stats = stats["plot_stats"]
    plot_labels = stats["plot_labels"]
    prefix = stats["prefix"]
    report_unique_id = stats["report_unique_id"]
    valid_point_list = stats["valid_point_list"]
    err_horizons_timepoints = stats["err_horizons_timepoints"]
    for key in plot_labels:
        label = plot_labels[key]
        xs = np.array(list(plot_stats[key].keys()))
        ys = np.array([x["mean"] for x in list(plot_stats[key].values())])
        ys_std = np.array([x["std"] for x in list(plot_stats[key].values())])
        try:
            filename = os.path.join(output_folder_name, f"{prefix}_{report_unique_id}_{label}.{img_ext}")
            print(filename)
            save_temporal_plot(xs, ys, ys_std, None, filename, xlabel="Horizon [s]", ylabel="[m]")
        except:
            import IPython

            IPython.embed(header="Failed to save plot.")

    filename = os.path.join(output_folder_name, f"{prefix}_{report_unique_id}_VALIDPOINTS.{img_ext}")
    print(filename)
    save_bar_plot(
        err_horizons_timepoints,
        valid_point_list,
        "Valid Points",
        filename,
        xlabel="Horizon [s]",
        ylabel="Num. of Trajectories",
    )

    for axis in ["", "_x", "_y"]:
        for misses_type in [MISSES_STAT_NAME, MISSES_PARTIAL]:
            for cur_threshold_idx in range(len(stats["miss_thresholds" + axis])):
                cur_threshold = stats["miss_thresholds" + axis][cur_threshold_idx]
                filename = os.path.join(
                    output_folder_name,
                    f"{prefix}_{report_unique_id}_{misses_type + axis}_rate_{cur_threshold}.{img_ext}",
                )
                try:
                    missed_point_ratios = [
                        stats["aggregate_statistics"][misses_type][k][cur_threshold_idx]["sum"]
                        / (stats["aggregate_statistics"][misses_type][k][cur_threshold_idx]["count"] + 1e-8)
                        for k in stats["aggregate_statistics"][misses_type]
                    ]
                except:
                    missed_point_ratios = [
                        stats["aggregate_statistics"][misses_type][str(k)][str(cur_threshold)]["sum"]
                        / (stats["aggregate_statistics"][misses_type][str(k)][str(cur_threshold)]["count"] + 1e-8)
                        for k in stats["aggregate_statistics"][misses_type]
                    ]

                print(filename)
                if axis == "":
                    plot_title_addition = ""
                else:
                    plot_title_addition = " along the " + axis + "-axis"
                save_bar_plot(
                    stats["err_horizons_timepoints" + axis],
                    missed_point_ratios,
                    f"Fraction of Missed Points@{cur_threshold}m" + plot_title_addition,
                    filename,
                    xlabel="Horizon [s]",
                    ylabel="Fraction of points",
                )

    filename = os.path.join(output_folder_name, f"{prefix}_{report_unique_id}_LABELS.{img_ext}")
    print(filename)
    plot_label_plots(stats, filename)
