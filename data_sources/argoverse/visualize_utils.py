import os
import uuid

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import LineString

from loaders.ado_key_names import END_TIMESTAMP, START_TIMESTAMP, TOKEN


def visualize_stats_hist(data, name, output_dir):
    """
    Visualize map stats, including number of lanes and number of points in each lane.
    This shall be merged/updated for other datasets.

    Parameters
    ----------
    data: numpy.ndarray or list
        Data to visualize.
    name: str
        Data name.
    output_dir: str
        Directory to save the plot.
    """
    plt.clf()
    fig, axes = plt.subplots()
    fig.set_size_inches(9, 7)
    axes.hist(data)
    axes.set_title("{} hist, max: {}".format(name, np.max(data)))
    plt.savefig(os.path.join(output_dir, "{}.png".format(name)))
    plt.close(fig)


def visualize_centerline(centerline: LineString, index=None) -> None:
    """Visualize the computed centerline.

    Args:
        centerline: Sequence of coordinates forming the centerline
    """
    line_coords = list(zip(*centerline))
    lineX = line_coords[0]
    lineY = line_coords[1]
    plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)

    if index is not None:
        plt.text(lineX[0], lineY[0], "s" + index)
    else:
        plt.text(lineX[0], lineY[0], "s")
    plt.text(lineX[-1], lineY[-1], "e")

    plt.axis("equal")


def visualize_traj_and_map(agent_traj, lanes, agent_id, output_dir, mode_data=None, smoothed_agent_traj=None):
    """
    Visualize agent trajectory and map.
    Parameters
    ----------
    data: trajectory data.
    lanes: map data.
    agent_id: agent id of trajectory data.
    output_dir: directory to save plot.
    lane_sequence: index of closes lane along trajectory.
    """
    figure_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(14, 12)
    plt.plot(
        agent_traj[:20, 0],
        agent_traj[:20, 1],
        "-",
        color="red",
        alpha=1,
        linewidth=1,
        zorder=15,
    )
    plt.plot(
        agent_traj[19:, 0],
        agent_traj[19:, 1],
        "-",
        color="blue",
        alpha=1,
        linewidth=1,
        zorder=15,
    )
    plt.text(agent_traj[0, 0], agent_traj[0, 1], "s")
    plt.text(agent_traj[-1, 0], agent_traj[-1, 1], "e")

    # Plot smoothed trajectory.
    if smoothed_agent_traj is not None:
        plt.plot(
            smoothed_agent_traj[:20, 0],
            smoothed_agent_traj[:20, 1],
            "-",
            color="magenta",
            alpha=0.5,
            linewidth=0.6,
            zorder=15,
        )
        plt.plot(
            smoothed_agent_traj[19:, 0],
            smoothed_agent_traj[19:, 1],
            "-",
            color="cyan",
            alpha=0.5,
            linewidth=0.6,
            zorder=15,
        )

    for i, ln in enumerate(lanes):
        visualize_centerline(ln["center_line"], str(i))

    # Compute lane stats.
    lane_point_sizes = [len(lane["center_line"]) for lane in lanes]
    title = "File id - {}".format(agent_id)
    title += "\nLane size: {}, point size: {}".format(len(lanes), np.sum(lane_point_sizes))

    # Add mode to title if available.
    if mode_data:
        for mode_data_key in mode_data:
            title += "\n{} past {},\n future{}".format(
                mode_data_key, mode_data[mode_data_key][:20], mode_data[mode_data_key][20:]
            )
    plt.title(title)
    fig.savefig(figure_dir + "/{}.png".format(agent_id))
    plt.close(fig)
