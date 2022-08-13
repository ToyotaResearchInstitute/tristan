import copy
import io
import json
from typing import Iterable, List, Optional, Tuple

import cv2
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import hsv_to_rgb, rgb_to_hsv
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, confusion_matrix, roc_curve
from torchvision.utils import make_grid

try:
    from intent.multiagents.additional_callback import AdditionalTrainerCallback
except ImportError:
    from typing import Any as AdditionalTrainerCallback

from intent.multiagents.latent_factors import load_latent_factor_info
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_map_handlers import MapPointType
from triceps.protobuf.prediction_dataset_semantic_handler import (
    SEMANTIC_HANDLER_FINISH_IDX,
    SEMANTIC_HANDLER_START_IDX,
    SEMANTIC_HANDLER_TYPE_IDX,
    SEMANTIC_HANDLER_VALID_IDX,
    SEMANTIC_HANDLER_VALUE_IDX,
)
from util.prediction_metrics import create_expected_traversal_rectangles

REL_PED_PREDICTED_TRAJ_COLOR = (0, 1, 0)  # green
REL_PED_TRAJ_COLOR = (0, 0.5, 0)  # dark green


def wrap_text(input_text, length_line=110):
    output_text = ""
    num_line = (len(input_text) // length_line) + 1
    for i in range(num_line):
        if i == num_line - 1:
            output_text = output_text + input_text[i * length_line :]
        else:
            output_text = output_text + input_text[i * length_line : (i + 1) * length_line] + "\n"
    return output_text


def get_img_from_fig(fig, image_format: str, dpi=300):
    """
    Return a cv2 image object from a numpy array of pyplot figure
    """
    buf = io.BytesIO()
    fig.savefig(buf, format=image_format, dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def get_cpu_detach(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().detach()
    elif isinstance(v, dict):
        return {k: get_cpu_detach(v) for k, v in v.items()}
    elif isinstance(v, list):
        return [get_cpu_detach(v) for v in v]
    return v


def visualize_histogram(
    values: np.array, title: str, image_format: str, xlabel: Optional[str] = None, ylabel: Optional[str] = "# of points"
) -> np.ndarray:
    """Visualizes a (log) histogram of given values in tensorboard.

    Log histogram is used if the smallest value in the data is larger than 0. Otherwise, a regular histogram is
    computed.

    Parameters
    ----------
    values : numpy.array
        Values over which the histogram is computed.
    dataset_type : str
        The type of the dataset for which this visualization is made.
    title : str
        Title of the histogram
    image_format: str
        The image format that histogram should be rendered as
    xlabel : str, optional
        x axis label (default: none)
    ylabel : str, optional
        y axis label (default: '# of points')

    Returns
    -------
    numpy.ndarray
        The image of the histogram.
    """
    fig = plt.figure()
    if values.min() > 0.0:
        # This results in 10 bins which is pyplot default
        logbins = np.geomspace(values.min(), values.max(), 11)
        plt.hist(values, bins=logbins)
        plt.title(title + " (log bins)")
        plt.xscale("log")
    else:
        plt.hist(values)
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
        plt.ylabel(ylabel)

    img = get_img_from_fig(fig, image_format)
    plt.close(fig)
    return img


def get_cpu_detach(v):
    if isinstance(v, torch.Tensor):
        return v.cpu().detach()
    elif isinstance(v, dict):
        return {k: get_cpu_detach(v) for k, v in v.items()}
    elif isinstance(v, list):
        return [get_cpu_detach(v) for v in v]
    return v


def visualize_samples(
    batch_itm_idx,
    batch_size,
    batch,
    scale,
    offset_x,
    offset_y,
    g_stats,
    cost,
    predicted,
    predicted_trajectories_scene,
    is_future_valid,
    param,
    summary_writer,
    iter,
    tag_prefix,
    num_past_timepoints,
    label_weights,
    semantic_labels,
    map_coordinates,
    map_validity,
    map_others,
    visualization_callbacks,
):
    """Visualize samples, loop over agents. Can be called from multiprocess worker

    Parameters:
        see _visualize_sample()
    """
    with torch.no_grad():
        for b in range(batch_size):
            visualization_idx = batch_itm_idx * batch_size + b
            if visualization_idx < param["num_visualization_images"]:
                solution = {
                    "predicted_trajectories": predicted_trajectories_scene[b, :, :, :, :],
                    "is_future_valid": is_future_valid[b, :, :],
                }
                for cb in visualization_callbacks:
                    cb.update_solution(solution, predicted, b)

                predicted_semantics = None if not param["use_semantics"] else g_stats["predicted_semantics"]
                _visualize_sample(
                    batch=batch,
                    batch_idx=b,
                    solution=solution,
                    scale=scale,
                    means=(offset_x[b, :], offset_y[b, :]),
                    cost=cost,
                    param=param,
                    summary_writer=summary_writer,
                    iter=iter,
                    visual_idx=visualization_idx,
                    tag_prefix=tag_prefix,
                    num_past_timepoints=num_past_timepoints,
                    label_weights=label_weights,
                    semantic_labels=semantic_labels,
                    predicted_semantics=predicted_semantics,
                    map_coordinates=map_coordinates,
                    map_validity=map_validity,
                    map_others=map_others,
                    visualization_callbacks=visualization_callbacks,
                )


def visualize_sample_process(
    batch_itm_idx,
    batch_size,
    batch_itm,
    g_stats,
    predicted,
    predicted_trajectories_scene,
    is_future_valid,
    scale,
    offset_x,
    offset_y,
    cost,
    param,
    summary_writer,
    iter,
    tag_prefix,
    num_past_timepoints,
    label_weights,
    semantic_labels,
    predicted_semantics,
    map_coordinates=None,
    map_validity=None,
    map_others=None,
    visualization_callbacks: Optional[Iterable[AdditionalTrainerCallback]] = None,
):
    """This function is used to do visualization using multiprocessing workers.

    Parameters:
    * Fot the first n parameters, see _visualize_sample()
    """
    if visualization_callbacks is None:
        visualization_callbacks = ()
    # Replace params in callback with the pickable params
    visualization_callbacks_new = []
    for cb in visualization_callbacks:
        cb = copy.copy(cb)
        if hasattr(cb, "params"):
            cb.params = param
        visualization_callbacks_new.append(cb)

    visualize_samples(
        batch_itm_idx=batch_itm_idx,
        batch_size=batch_size,
        batch=batch_itm,
        scale=scale,
        g_stats=g_stats,
        cost=cost,
        offset_x=offset_x,
        offset_y=offset_y,
        predicted=predicted,
        predicted_trajectories_scene=predicted_trajectories_scene,
        is_future_valid=is_future_valid,
        param=param,
        summary_writer=summary_writer,
        iter=iter,
        tag_prefix=tag_prefix,
        num_past_timepoints=num_past_timepoints,
        label_weights=label_weights,
        semantic_labels=semantic_labels,
        # predicted_semantics=predicted_semantics,
        map_coordinates=map_coordinates,
        map_validity=map_validity,
        map_others=map_others,
        visualization_callbacks=visualization_callbacks_new,
    )


def _visualize_sample(
    batch,
    batch_idx,
    solution,
    scale,
    means,
    cost,
    param,
    summary_writer,
    iter,
    visual_idx,
    tag_prefix,
    num_past_timepoints,
    label_weights,
    semantic_labels,
    predicted_semantics,
    map_coordinates=None,
    map_validity=None,
    map_others=None,
    visualization_callbacks=(),
):
    """Visualize a sample, including BEV (trajectories), images, agent images, maps, semantics, etc.

    Parameters
    ----------
    batch : dict
        A batch of data (or selected items, e.g., worst cases)
    batch_idx : int
        Index of the item in the batch
    solution : dict
        Prediction solution
    scale : float
        Position scale for global normalization
    means : tuple of (offset_x, offset_y)
        Position offset for global normalization
    cost : torch.Tensor
        Cost tensor of shape (batch_size)
    param : dict
        Trainer parameters
    summary_writer : tensorboardX.writer.SummaryWriter
        Tensorboard writer for trainer
    iter : int
        Global epoch
    visual_idx : int
        Index of the item for visualization
    tag_prefix : str
        Prefix of the tag in tensorboard (e.g., 'vis')
    num_past_timepoints : int
        Number of past timepoints
    skip_visualization : bool
        Whether skip visualization
    label_weights : torch.Tensor (or None if not using semantic labels)
        Semantic label weights tensor of shape (semantic_num)
    semantic_labels : torch.Tensor (or None if not using semantic labels)
        Target semantic labels tensor of shape (batch_size, semantic_num, 5)
    predicted_semantics : torch.Tensor (or None if not using semantic labels)
        Predicted semantic labels tensor of shape (batch_size, semantic_num, 5)
    map_coordinates: torch.Tensor (or None if disabling map inputs)
        If map input type is point, map coordinates tensor of shape (batch_size, element_num, points_per_element, 2)
    map_validity : torch.Tensor (or None if disabling map inputs)
        Validity tensor (for map coordinates) of shape (batch_size, element_num, points_per_element)
    map_others : torch.Tensor (or None if disabling map inputs)
        Point type and tangent information for each point of shape
        (batch_size, max_element_num, max_point_num, 3). The elements in the last 3 dimensions are
        structured as follows (point type, sin(theta), cos(theta))), where point type is corresponding
        to the integer in MapPointType.
    visualization_callbacks : list
        List of visualization callbacks.
    """
    # Visualize BEV (trajectories) w/ or w/o map
    if ProtobufPredictionDataset.DATASET_KEY_MAP in batch and param["map_input_type"] == "point":
        img, _ = visualize_prediction(
            batch,
            batch_idx,
            solution=solution,
            scale=scale,
            means=means,
            cost=cost,
            param=param,
            map_coordinates=map_coordinates / scale,
            map_validity=map_validity,
            map_others=map_others,
            visualization_callbacks=visualization_callbacks,
        )
    else:
        img, _ = visualize_prediction(
            batch,
            batch_idx,
            solution,
            scale=scale,
            means=means,
            cost=cost,
            param=param,
            visualization_callbacks=visualization_callbacks,
        )

    summary_writer.add_image("{}/{}/BEV".format(tag_prefix, visual_idx), img.transpose(2, 0, 1), global_step=iter)

    # Visualize global images
    if (
        summary_writer
        and ProtobufPredictionDataset.DATASET_KEY_IMAGES in batch
        and not (batch[ProtobufPredictionDataset.DATASET_KEY_IMAGES][batch_idx] == 0).all()
    ):
        # shape: (max_num_imgs, 3, height, width)
        cur_images = batch[ProtobufPredictionDataset.DATASET_KEY_IMAGES][batch_idx]

        # shape (max_num_imgs)
        active_imgs_mask = cur_images.sum(dim=(1, 2, 3)) != 0.0

        # shape (num_actual_images, 3 or 6, height, width)
        actual_images = cur_images[active_imgs_mask, :3, :, :]

        image_grid = make_grid(actual_images.cpu().detach(), nrow=actual_images.shape[0])

        summary_writer.add_image(f"{tag_prefix}/{visual_idx}/sensor_images", image_grid, global_step=iter)

        if cur_images.shape[1] >= 6:
            # Visualize additional channels
            actual_masks = cur_images[active_imgs_mask, 3:6, :, :]
            image_grid = make_grid(actual_masks.cpu().detach(), nrow=actual_masks.shape[0])

            summary_writer.add_image(f"{tag_prefix}/{visual_idx}/sensor_masks", image_grid, global_step=iter)

    # Visualize agent images
    # This involves all agent images in the batch without additional identifying information.
    if (
        summary_writer
        and ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES in batch
        and not (batch[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES][batch_idx] == 0).all()
    ):
        # shape: (max_num_imgs, 3, height, width)
        cur_images = batch[ProtobufPredictionDataset.DATASET_KEY_AGENT_IMAGES][batch_idx]

        # shape (max_num_imgs)
        active_imgs_mask = cur_images.sum(dim=(1, 2, 3)) != 0.0

        # shape (num_actual_images, 3 or 6, height, width)
        actual_images = cur_images[active_imgs_mask, :3, :, :]

        image_grid = make_grid(actual_images.cpu().detach(), nrow=actual_images.shape[0])

        summary_writer.add_image(f"{tag_prefix}/{visual_idx}/agent_images", image_grid, global_step=iter)

        if cur_images.shape[1] >= 6:
            # Visualize additional channels
            actual_masks = cur_images[active_imgs_mask, 3:6, :, :]
            image_grid = make_grid(actual_masks.cpu().detach(), nrow=actual_masks.shape[0])

            summary_writer.add_image(f"{tag_prefix}/{visual_idx}/agent_masks", image_grid, global_step=iter)

    # Display annotated semantic labels and predicted semantics
    if param["use_semantics"] and (semantic_labels is not None) and (semantic_labels[batch_idx, :, :] ** 2).sum() > 0:
        label_img = visualize_labels(
            semantic_labels,
            predicted_semantics,
            batch_idx,
            param["latent_factors_file"],
            param["visualization_image_format"],
        )
        summary_writer.add_image(
            "{}/{}/semantic_labels".format(tag_prefix, visual_idx), label_img.transpose(2, 0, 1), global_step=iter
        )

    # Visualize maps
    if ProtobufPredictionDataset.DATASET_KEY_MAP in batch:
        if param["map_input_type"] == "point":
            map_image = visualize_map_coordinates(
                map_coordinates / scale,
                map_validity,
                map_others,
                batch_idx,
                param["visualization_image_format"],
            )
            map_image = map_image.transpose(2, 0, 1)
        else:
            map_image = (
                batch[ProtobufPredictionDataset.DATASET_KEY_MAP][batch_idx, num_past_timepoints, :, :, :].cpu().numpy()
            )

        summary_writer.add_image("{}/{}/map".format(tag_prefix, visual_idx), map_image, global_step=iter)

    # Record text of sample information
    sample_dict = {
        "protobuf_file": batch[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE][batch_idx],
    }
    if batch[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][batch_idx] != "":
        instance_info = json.loads(batch[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][batch_idx])
        sample_dict.update(
            {
                "json_dir": instance_info["json_dir"],
                "source_tlog": instance_info["source_tlog"],
                "timestamp": str(instance_info["timestamp"]),
            }
        )
    sample_info_text = ""
    for key, value in sample_dict.items():
        txt_i = "{}: {}  \n".format(key, value)
        sample_info_text += txt_i

    summary_writer.add_text("{}/{}/instance_info".format(tag_prefix, visual_idx), sample_info_text, global_step=iter)


def visualize_itm(batch_itm, batch_idx):
    """
    Visualize the trajectories in a batch
    """
    positions = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, :2]
    timestamps = batch_itm[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS][batch_idx, :]
    is_valid = batch_itm[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, 2]
    num_agents = positions.shape[0]
    for agent_id in range(num_agents):
        if agent_id == 0:
            agent_color = (0.0, 0.0, 0.0)
        else:
            agent_color = (0.0, 0.0, 1.0)
        valid_pos = positions[agent_id, is_valid[agent_id, :].bool(), :]
        valid_t = timestamps[is_valid[agent_id, :].bool()]
        plt.plot(valid_pos[:, 0].detach().cpu().numpy(), valid_pos[:, 1].detach().cpu().numpy(), color=agent_color)
    plt.axes().set_aspect("equal")
    plt.show()


def visualize_prediction(
    batch,
    batch_idx,
    solution,
    scale,
    means,
    cost,
    param,
    map_coordinates=None,
    map_validity=None,
    map_others=None,
    agent_set=None,
    visualization_callbacks=None,
):
    if visualization_callbacks is None:
        visualization_callbacks = []
    _, fig, additional_saving_stats = visualize_prediction_fig(
        batch,
        batch_idx,
        solution,
        scale,
        means,
        cost,
        param,
        map_coordinates,
        map_validity,
        map_others,
        agent_set,
        visualization_callbacks,
    )
    img = get_img_from_fig(fig, image_format=param["visualization_image_format"])
    plt.close(fig)
    return img, additional_saving_stats


def plot_arrow(ax, xy, xy_next, color="k", alpha=0.6):
    # ax.annotate(
    #     "",
    #     xy=xy_next,
    #     xytext=xy,
    #     arrowprops=dict(arrowstyle="-|>"),
    # )
    ax.arrow(
        xy[0],
        xy[1],
        xy_next[0] - xy[0],
        xy_next[1] - xy[1],
        width=0.04,
        fc=color,
        ec=color,
        alpha=alpha,
        length_includes_head=True,
    )


def visualize_agents(
    ax: Axes,
    param: dict,
    batch,
    batch_idx: int,
    solution: dict,
    agent_set,
    mean_x: torch.Tensor,
    mean_y: torch.Tensor,
    scale,
    min_max_trajectory_position,
    additional_saving_stats,
    visualization_callbacks,
    agent_colors=None,
):
    positions = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, :2].float()

    timestamps = batch[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS][batch_idx, :].float()
    is_valid = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][batch_idx, :, :, 2]
    num_past_points = batch[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS][batch_idx]
    dot_keys = batch[ProtobufPredictionDataset.DATASET_KEY_DOT_KEYS][batch_idx, :]
    is_ego = batch[ProtobufPredictionDataset.DATASET_KEY_IS_EGO_VEHICLE][batch_idx, :]
    is_rel_ped = batch[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT][batch_idx, :]

    num_agents = positions.shape[0]
    if agent_set is None:
        agent_set = list(range(num_agents))
    if agent_colors is None:
        agent_colors = _default_agent_colors(len(agent_set))

    for agent_id in list(agent_set):
        valid_pos = positions[agent_id, is_valid[agent_id, :].bool(), :]
        if len(valid_pos) == 0:
            continue
        valid_t = timestamps[is_valid[agent_id, :].bool()]
        valid_x = valid_pos[:, 0].detach().cpu().numpy()
        valid_y = valid_pos[:, 1].detach().cpu().numpy()

        if is_ego[agent_id] == 1.0:
            agent_label = "Ego Vehicle"
            agent_color = (0.5, 0.0, 0.0)  # set color of ego-vehicle to red
            agent_color_predicted = (1, 0, 0)
            # Plot ego vehicle crossing box
            if "ego_rotation" in solution:
                if param["fixed_ego_orientation"] == True:
                    rotations_scene_ego = torch.eye(2, device=positions.device).unsqueeze(0)
                else:
                    rotations_scene_ego = solution["ego_rotation"].unsqueeze(0)
                ego_pose = valid_pos[num_past_points : num_past_points + 1, :]
                rectangles = create_expected_traversal_rectangles(rotations_scene_ego, ego_pose, 1)
                x, y = rectangles[0].exterior.xy
                ax.plot(np.array(x), np.array(y))
        elif is_rel_ped[agent_id] == 1.0:
            agent_label = "Relevant Pedestrian"
            agent_color = REL_PED_TRAJ_COLOR  # set color of relevant pedestrian to green
            agent_color_predicted = REL_PED_PREDICTED_TRAJ_COLOR
        else:
            agent_label = f"Agent {agent_id}"
            agent_color = agent_colors[agent_id]
            agent_color_predicted = [min(max(c, 0.9), c * 1.3) for c in agent_color]

        # Past traj
        ax.plot(
            valid_x[:num_past_points],
            valid_y[:num_past_points],
            "o-",
            color=agent_color,
            lw=2.0,
            markersize=2,
            markerfacecolor=(*agent_color, 0.5),
        )

        # Plot arrow at the start of past traj
        if valid_x.shape[0] >= 2 and valid_y.shape[0] >= 2:
            plot_arrow(ax, (valid_x[0], valid_y[0]), (valid_x[1], valid_y[1]), color=agent_color)

        ax.plot(
            valid_x[num_past_points - 1 :],
            valid_y[num_past_points - 1 :],
            "o-",
            color=agent_color,
            lw=1.0,
            label=agent_label,
            markersize=2,
            markerfacecolor=(*agent_color, 0.5),
        )
        min_max_trajectory_position[0] = max(min_max_trajectory_position[0], max(valid_x))
        min_max_trajectory_position[1] = max(min_max_trajectory_position[1], max(valid_y))
        min_max_trajectory_position[2] = min(min_max_trajectory_position[2], min(valid_x))
        min_max_trajectory_position[3] = min(min_max_trajectory_position[3], min(valid_y))
        if "tracks" not in additional_saving_stats:
            additional_saving_stats["tracks"] = {}
        if str(agent_id) not in additional_saving_stats["tracks"]:
            additional_saving_stats["tracks"][str(agent_id)] = {}
            additional_saving_stats["tracks"][str(agent_id)]["dot_key"] = str(int(dot_keys[agent_id].cpu().item()))
        if "predicted" not in additional_saving_stats["tracks"][str(agent_id)]:
            additional_saving_stats["tracks"][str(agent_id)]["predicted"] = []
        ax.text(valid_x[0], valid_y[0], str(int(dot_keys[agent_id].cpu().item())), color=agent_color, size=6)

        # Visualize predicted trajectories.
        if "predicted_trajectories" in solution:
            is_future_valid = solution["is_future_valid"]
            for sample_i in range(solution["predicted_trajectories"].shape[3]):
                predicted_sample = solution["predicted_trajectories"][
                    agent_id, is_future_valid[agent_id, :].bool(), :, sample_i
                ]
                if predicted_sample.numel() == 0:
                    continue
                predicted_x = (predicted_sample[:, 0].detach().cpu().numpy() + mean_x[agent_id].cpu().numpy()) / scale
                predicted_y = (predicted_sample[:, 1].detach().cpu().numpy() + mean_y[agent_id].cpu().numpy()) / scale

                prediction = {"trajectory": list(zip(predicted_x.tolist(), predicted_y.tolist()))}
                additional_saving_stats["tracks"][str(agent_id)]["predicted"].append(prediction)
                past_inputs = {
                    "trajectory": list(zip(valid_x[:num_past_points].tolist(), valid_y[:num_past_points].tolist()))
                }
                additional_saving_stats["tracks"][str(agent_id)]["past_inputs"] = past_inputs
                ground_truth = {
                    "trajectory": list(zip(valid_x[num_past_points:].tolist(), valid_y[num_past_points:].tolist()))
                }
                prediction_timestamps = valid_t[num_past_points:].tolist()
                additional_saving_stats["tracks"][str(agent_id)]["ground_truth"] = ground_truth
                additional_saving_stats["tracks"][str(agent_id)]["ground_truth_is_valid"] = is_future_valid.tolist()
                additional_saving_stats["tracks"][str(agent_id)]["prediction_timestamps"] = prediction_timestamps

                # Change agent color if needed.
                for cb in visualization_callbacks:
                    agent_color_predicted = cb.update_visualization_agent_color(
                        agent_color_predicted, param, sample_i, solution
                    )

                ax.plot(predicted_x, predicted_y, color=agent_color_predicted, lw=0.5, alpha=0.5)
                ax.plot(
                    predicted_x[0], predicted_y[0], "o", color=agent_color_predicted, lw=0.5, alpha=0.3, markersize=3
                )
                ax.plot(
                    predicted_x[-1], predicted_y[-1], "o", color=agent_color_predicted, lw=0.5, alpha=0.3, markersize=3
                )
                if len(predicted_x) > 1:
                    plot_arrow(
                        ax,
                        (predicted_x[0], predicted_y[0]),
                        (predicted_x[1], predicted_y[1]),
                        agent_color_predicted,
                        0.6,
                    )
                min_max_trajectory_position[0] = max(min_max_trajectory_position[0], max(predicted_x))
                min_max_trajectory_position[1] = max(min_max_trajectory_position[1], max(predicted_y))
                min_max_trajectory_position[2] = min(min_max_trajectory_position[2], min(predicted_x))
                min_max_trajectory_position[3] = min(min_max_trajectory_position[3], min(predicted_y))

                # Visualize additional agent info if needed.
                for cb in visualization_callbacks:
                    cb.visualize_agent_additional_info(
                        agent_id,
                        is_future_valid,
                        sample_i,
                        solution,
                        ax,
                        predicted_x,
                        predicted_y,
                        agent_color_predicted,
                    )

    return min_max_trajectory_position


def visualize_prediction_fig(
    batch,
    batch_idx,
    solution,
    scale,
    means,
    cost,
    param,
    map_coordinates=None,
    map_validity=None,
    map_others=None,
    agent_set=None,
    visualization_callbacks=None,
):
    """Create a visualization image for an item + a solution

    Parameters
    ----------
    batch : dict
        A batch from the dataloader
    batch_idx : int
        The item index
    solution : dict
        The solution to be visualized.
    scale :  float
        for normalizing the coordinates
    means : tuple
        for normalizing the coordinates
    cost :
        for showing the cost of the solution
    map_coordinates : torch.Tensor
        The positions of the map elements of shape (batch_size, max_element_num, max_point_num, 2)
    map_validity : torch.Tensor
        The positions of the map elements of shape (batch_size, max_element_num, max_point_num)
    map_others : torch.Tensor
        Point type and tangent information for each point of shape
        (batch_size, max_element_num, max_point_num, 3). The elements in the last 3 dimensions are
        structured as follows (point type, sin(theta), cos(theta))), where point type is corresponding
        to the integer in MapPointType.
    visualization_callbacks : list
        List of visualization callbacks.

    Returns
    -------
    numpy.ndarray
        An image with the visualization from pyplot + additional_saving_stats dictionary.
    """
    if not visualization_callbacks:
        visualization_callbacks = []

    # Stores tuples contianing line type and color for map elements.
    MAP_STYLE = {
        MapPointType.UNDEFINED: ("--", "darkorange"),
        MapPointType.CENTER_LANE_LINE: ("--", "gray"),
        MapPointType.RIGHT_LANE_LINE: ("--", "darkgreen"),
        MapPointType.LEFT_LANE_LINE: (":", "blue"),
        MapPointType.CROSSWALK: ("-.", "darkgoldenrod"),
        MapPointType.LANE_BOUNDARY_LINE: (":", "darkblue"),
    }
    additional_saving_stats = {}

    # Verify the area shown is at least +/-5m around the center of the tracks.
    map_view_min_span_size = param["view_min_span_size"]
    # Verify the area shown includes no more than 5 meters beyond the predicted tracks's bounding box
    map_view_margin_around_trajectories = param["map_view_margin_around_trajectories"]

    mean_x, mean_y = means
    # positions[:, :, 0] -= mean_x.unsqueeze(1) / scale
    # positions[:, :, 1] -= mean_y.unsqueeze(1) / scale

    fig = plt.figure()
    ax = fig.add_axes([0.05, 0.2, 0.75, 0.75])  # [left, bottom, width, height]
    num_agents = batch[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[1]
    # Pick the color in the middle of the color map
    agent_colors = _default_agent_colors(num_agents)

    # Visualize map if it exists.
    if map_coordinates is not None:
        assert map_others is not None and map_validity is not None, "Map validity and additional map inputs required."
        map_v = map_validity[batch_idx].detach().cpu().numpy()
        if np.sum(map_v) > 0:

            # This assumes point type is the first element of map_others.
            map_others_point_type_idx = 0
            map_point_type = map_others[batch_idx, :, map_others_point_type_idx].detach().cpu().numpy()[map_v > 0]

            map_xy = map_coordinates[batch_idx].detach().cpu().numpy()
            map_x = map_xy[:, 0][map_v > 0]
            map_y = map_xy[:, 1][map_v > 0]

            # Obtain map id to visualize each element separately. (Assuming map id is the last element in map_others).
            map_others_id_idx = -1
            map_id = map_others[batch_idx, :, map_others_id_idx].detach().cpu().numpy()[map_v > 0]
            map_unique_ids = np.unique(map_id)

            # Visualize each map element separately.
            map_types_plotted = []
            for unique_id in map_unique_ids:
                map_id_mask = map_id == unique_id
                if np.sum(map_id_mask) == 0:
                    continue

                # Assume map type is consistent across the same element.
                map_point_type_i = MapPointType(map_point_type[map_id_mask][0])
                map_types_plotted.append(map_point_type_i)
                ax.plot(
                    map_x[map_id_mask],
                    map_y[map_id_mask],
                    MAP_STYLE[map_point_type_i][0],
                    color=MAP_STYLE[map_point_type_i][1],
                    alpha=1,
                    linewidth=1,
                    zorder=15,
                )
    min_max_trajectory_position = [
        -np.Inf,  # max_trajectory_position_x,
        -np.Inf,  # max_trajectory_position_y,
        np.Inf,  # min_trajectory_position_x,
        np.Inf,  # min_trajectory_position_y,
    ]
    min_max_trajectory_position = visualize_agents(
        ax,
        param,
        batch,
        batch_idx,
        solution,
        agent_set,
        mean_x,
        mean_y,
        scale,
        min_max_trajectory_position,
        additional_saving_stats,
        visualization_callbacks,
        agent_colors,
    )

    (
        max_trajectory_position_x,
        max_trajectory_position_y,
        min_trajectory_position_x,
        min_trajectory_position_y,
    ) = min_max_trajectory_position

    ax.set_aspect("equal")
    x_min, x_max, y_min, y_max = ax.axis()

    x_c = (x_min + x_max) / 2
    y_c = (y_min + y_max) / 2
    x_min = min(x_min, x_c - map_view_min_span_size)
    y_min = min(y_min, y_c - map_view_min_span_size)
    x_max = max(x_max, x_c + map_view_min_span_size)
    y_max = max(y_max, y_c + map_view_min_span_size)
    x_min = max(x_min, min_trajectory_position_x - map_view_margin_around_trajectories)
    y_min = max(y_min, min_trajectory_position_y - map_view_margin_around_trajectories)
    x_max = min(x_max, max_trajectory_position_x + map_view_margin_around_trajectories)
    y_max = min(y_max, max_trajectory_position_y + map_view_margin_around_trajectories)
    try:
        ax.axis((x_min, x_max, y_min, y_max))
    except:
        print("Failed to change axis")

    cost_str = str(cost[batch_idx].detach().cpu().item()) if cost is not None else ""
    if batch[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][batch_idx] != "":
        try:
            data = json.loads(batch[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO][batch_idx])
            fig.text(
                0.01,
                0.01,
                f"[json] {wrap_text(data['json_dir'])}"
                + f"\n[tlog] {wrap_text(data['source_tlog'])}"
                + f"\n[timestamp] {data['timestamp']}, cost: {cost_str}",
                fontsize=6,
            )
        except:
            print(ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO + " information not available.")

    # Overwrite text with callbacks if available.
    if visualization_callbacks:
        num_past_points = batch[ProtobufPredictionDataset.DATASET_KEY_NUM_PAST_POINTS][batch_idx].item()
        for cb in visualization_callbacks:
            text = cb.update_visualization_text(cost_str, solution, batch, batch_idx, num_past_points)
            fig.text(0.01, 0.75, text, fontsize=4)

    # Visualize additional info of the solution if needed.
    if visualization_callbacks:
        for cb in visualization_callbacks:
            cb.visualize_additional_info(solution, fig)

    # ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    ax.legend(fontsize=5)
    return ax, fig, additional_saving_stats


def visualize_labels(semantic_labels, predicted_semantics, batch_idx, latent_factors_file, image_format: str):
    """
    Create a visualization image for GroundTruth & Predicted semantic labels
    :param
    :return:
    """
    assert semantic_labels.shape == predicted_semantics.shape, "inconsistent shape of predicted semantic labels"
    try:
        with open(latent_factors_file) as fp:
            label_definitions = json.load(fp)
    except:
        import IPython

        IPython.embed(header="failed to read latent_factors_file")
    num_label_types = semantic_labels.shape[1]
    fig = plt.figure()
    fig.text(0.15, 0.9, "Label name", fontsize=12)
    fig.text(0.56, 0.9, "Ground truth", fontsize=12)
    fig.text(0.76, 0.9, "Prediction", fontsize=12)
    is_id_matched = False
    for i in range(num_label_types):
        for label_definition in label_definitions:
            if semantic_labels[batch_idx, i, SEMANTIC_HANDLER_TYPE_IDX] == label_definition["id"]:
                label_name = label_definition["name"]
                is_id_matched = True
                break

        if not is_id_matched:
            import IPython

            IPython.embed(header="failed to match label ID")

        if semantic_labels[batch_idx, i, SEMANTIC_HANDLER_VALID_IDX] > 1e-6:
            gt_label = semantic_labels[batch_idx, i, SEMANTIC_HANDLER_VALUE_IDX].detach().cpu().item() > 0
        else:
            gt_label = "Invalid"

        predicted_label = predicted_semantics[batch_idx, i, SEMANTIC_HANDLER_VALUE_IDX].detach().cpu().item() > 0.5
        fig.text(0.01, 0.85 - i * 0.05, str(i + 1), fontsize=12)
        fig.text(0.06, 0.85 - i * 0.05, label_name, fontsize=12)
        fig.text(0.59, 0.85 - i * 0.05, str(gt_label), fontsize=12)
        fig.text(0.8, 0.85 - i * 0.05, str(predicted_label), fontsize=12)

    label_img = get_img_from_fig(fig, image_format=image_format)
    plt.close(fig)
    return label_img


def visualize_map_coordinates(map_coordinates, map_validity, map_others, batch_idx, image_format: str):
    """
    Create a visualization image for map coordinates
    :param map_coordinates:
    :param map_validity
    :param map_others
    :param batch_idx:
    :param image_format: image format to render the visualization images as
    :return:
    """
    fig = plt.figure()
    map_xy = map_coordinates[batch_idx].detach().cpu().numpy()
    map_v = map_validity[batch_idx].detach().cpu().numpy()
    map_x = map_xy[:, 0][map_v > 0]
    map_y = map_xy[:, 1][map_v > 0]

    # Obtain map id to visualize each element separately. (Assuming map id is the last element in map_others).
    map_others_id_idx = -1
    map_id = map_others[batch_idx, :, map_others_id_idx].detach().cpu().numpy()[map_v > 0]
    map_unique_ids = np.unique(map_id)

    # Visualize each map element separately.
    for unique_id in map_unique_ids:
        map_id_mask = map_id == unique_id
        if np.sum(map_id_mask) > 0:
            plt.plot(map_x[map_id_mask], map_y[map_id_mask], "--", color="grey", alpha=1, linewidth=1, zorder=15)

    map_img = get_img_from_fig(fig, image_format=image_format)
    plt.close(fig)
    return map_img


def visualize_label_accuracy(
    all_semantic_labels,
    all_predicted_semantics,
    label_weights,
    latent_factors_file,
    summary_writer,
    dataloader_type,
    cur_iter,
    skip_visualizations,
    image_format: str,
    replace_ids=False,
):
    """
    :param all_semantic_labels:
    :param all_predicted_semantics:
    :param label_weights:
    :param latent_factors_file:
    :param summary_writer:
    :param dataloader_type:
    :param cur_iter: the iteration, to save tensorboard.
    :param skip_visualizations - whether to show image figures such as confusion matrices.
    :param image_format: image format to render the visualization images as
    :return: a dictionary of prediction accuracies for all labels on different time segments
    """
    assert all_semantic_labels.shape == all_predicted_semantics.shape, "inconsistent shape of predicted semantic labels"
    try:
        label_definitions = load_latent_factor_info(latent_factors_file, replace_ids=replace_ids)
    except:
        import IPython

        IPython.embed(header="failed to read latent_factors_file")

    time_segments = [(-0.5, 0.5), (0, 1), (1, 2)]
    num_label_types = all_semantic_labels.shape[1]
    prediction_accuracy = {}
    for t_i in range(len(time_segments) + 1):
        # Note: the last iteration is not for time segment, but for the whole labeled timespan
        if t_i < len(time_segments):
            time_segment = time_segments[t_i]
            time_seg_name = "from_{}_to_{}".format(time_segment[0], time_segment[1])
        else:
            time_seg_name = "all_time"
        prediction_accuracy[time_seg_name] = []
        is_id_matched = False
        if not skip_visualizations:
            fig, axes = plt.subplots(2, num_label_types, figsize=(3 * num_label_types, 4), sharey="row")
        overall_targets = []
        overall_predictions = []
        overall_weights = []
        for i in range(num_label_types):
            semantic_labels = all_semantic_labels[:, i, :]
            # if the label tensor of a sample includes all zeros, it is not annotated (augmented).
            is_annotated = semantic_labels.sum(dim=1).bool()
            for label_definition in label_definitions:
                if (
                    not (semantic_labels[is_annotated].numel() == 0)
                    and semantic_labels[is_annotated][0, SEMANTIC_HANDLER_TYPE_IDX] == label_definition["id"]
                ):
                    label_name = label_definition["name"]
                    is_id_matched = True
                    break
            if not is_id_matched:
                import IPython

                IPython.embed(header="failed to match label ID")
            predicted_semantics = all_predicted_semantics[:, i, :]
            # Select the samples covering the time segment
            if t_i < len(time_segments):
                idx_seg = (semantic_labels[:, SEMANTIC_HANDLER_START_IDX] <= time_segment[0]) & (
                    semantic_labels[:, SEMANTIC_HANDLER_FINISH_IDX] >= time_segment[1]
                )
                semantic_labels = semantic_labels[idx_seg, :]
                predicted_semantics = predicted_semantics[idx_seg, :]
            validity = semantic_labels[:, SEMANTIC_HANDLER_VALID_IDX].bool()
            valid_target_labels = semantic_labels[:, SEMANTIC_HANDLER_VALUE_IDX][validity]
            valid_predicted_labels = predicted_semantics[:, SEMANTIC_HANDLER_VALUE_IDX][validity]
            # Original target values are [-1, 1]
            targets = (valid_target_labels > 0).detach().cpu().numpy()
            # For BCE loss, prediction were sigmoid to [0, 1]
            predicts = (valid_predicted_labels > 0.5).detach().cpu().numpy()
            try:
                example_weights = label_weights[i].repeat(validity.sum())
                # Old version for replace_ids=True
                # example_weights = label_weights[semantic_labels[:, SEMANTIC_HANDLER_TYPE_IDX][validity].int().cpu().detach().numpy()]
            except:
                import IPython

                IPython.embed(header="check label visualization")
            assert len(example_weights) == len(predicts)
            assert len(example_weights) == len(targets)
            overall_targets.append(targets)
            overall_predictions.append(predicts)
            overall_weights.append(example_weights)
            if len(targets) == 0:
                accuracy = "N/A"
            else:
                accuracy = np.round(accuracy_score(targets, predicts), 4)
                # Display "accuracy vs. iteration" curve
                if summary_writer:
                    summary_writer.add_scalar(
                        "{}_label_prediction_accuracy/{}/{}".format(dataloader_type, time_seg_name, label_name),
                        accuracy,
                        global_step=cur_iter,
                    )
            prediction_accuracy[time_seg_name].append(accuracy)

            # Display confusion matrix for each semantic
            if len(targets) > 0 and len(predicts) > 0 and not skip_visualizations:
                # The first row display confusion matrix with percentages; the second row display confusion matrix with counts
                conf_mat = confusion_matrix(targets, predicts, labels=[True, False], normalize="true")
                conf_mat2 = confusion_matrix(targets, predicts, labels=[True, False], normalize=None)
                values = ["Yes", "No"]
                disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=values)
                disp2 = ConfusionMatrixDisplay(confusion_matrix=conf_mat2, display_labels=values)
                disp.plot(ax=axes[0][i])
                disp2.plot(ax=axes[1][i])
                disp.ax_.set_title("Acc = {} ({})".format(accuracy, label_name), fontsize=6)
                disp.im_.colorbar.remove()
                disp2.im_.colorbar.remove()
                disp.ax_.set_xlabel("Predicted label", fontsize=6)
                disp2.ax_.set_xlabel("Predicted label", fontsize=6)
                if i == 0:
                    disp.ax_.set_ylabel("True label", fontsize=6)
                    disp2.ax_.set_ylabel("True label", fontsize=6)
                else:
                    disp.ax_.set_ylabel("")
                    disp2.ax_.set_ylabel("")
                disp.ax_.set_xticklabels(values, fontsize=6)
                disp.ax_.set_yticklabels(values, fontsize=6)
                disp2.ax_.set_xticklabels(values, fontsize=6)
                disp2.ax_.set_yticklabels(values, fontsize=6)
        if len(overall_targets) > 0:
            overall_targets = np.concatenate(overall_targets)
            overall_predictions = np.concatenate(overall_predictions)
            overall_weights = np.concatenate(overall_weights)

            overall_accuracy = np.round(
                accuracy_score(overall_targets, overall_predictions, sample_weight=overall_weights), 4
            )
            if summary_writer:
                summary_writer.add_scalar(
                    "{}_label_prediction_accuracy/{}/overall_weighted".format(
                        dataloader_type, time_seg_name, label_name
                    ),
                    overall_accuracy,
                    global_step=cur_iter,
                )

        if not skip_visualizations:
            plt.subplots_adjust(wspace=1, hspace=0.1)
            fig.colorbar(disp.im_, ax=axes)
            img = get_img_from_fig(fig, image_format=image_format)
            plt.close(fig)
            if summary_writer:
                summary_writer.add_image(
                    "{}_semantic_label_prediction/{}".format(dataloader_type, time_seg_name),
                    img.transpose(2, 0, 1),
                    global_step=cur_iter,
                )

    return prediction_accuracy


def visualize_mode_sequences(mode_actual, mode_predicted, num_past_points, stats, batch_idx, image_format="png"):
    """
    Create a visualization image for mode sequences.
    :param mode_actual: actual mode sequence, with shape [batch_size, num_points].
    :param mode_predicted: predicted mode sequence,
        with shape [batch_size, num_agents, num_future_points, 1, num_samples].
    :param num_past_points: observed length.
    :param stats: dict including additional info.
    :param batch_idx: batch index.
    :param image_format: image format to render the visualization images as
    :return:
    """
    fig = plt.figure()
    mode_actual = mode_actual[batch_idx].detach().cpu().numpy()
    mode_predicted = mode_predicted[batch_idx, 0, :, 0].detach().cpu().numpy()

    # Change mode from [0, 1, 2] to [0, 1, -1] for better visualization.
    mode_actual[mode_actual == 2] = -1.0
    mode_predicted[mode_predicted == 2] = -1.0

    num_total_points = mode_actual.shape[-1]
    num_samples = mode_predicted.shape[-1]

    if "log_prob" in stats:
        log_prob = stats["log_prob"][batch_idx, 0].detach().cpu().numpy()
        sample_weight = np.exp(log_prob)
        sample_lw = (sample_weight**2) * 10
    else:
        sample_lw = np.ones(num_samples)
    # Plot ground truth mode sequence.
    timesteps = np.arange(num_total_points)
    plt.plot(timesteps[:num_past_points], mode_actual[:num_past_points], "o", color="red", lw=0.5)
    plt.plot(timesteps[num_past_points:], mode_actual[num_past_points:], "x", color="red", lw=0.5)

    # Plot predicted samples.
    for i in range(num_samples):
        sample_y_shift = 0.02 * (i - (num_samples // 2))
        mode_predicted_sample = mode_predicted[:, i]
        plt.plot(
            timesteps[num_past_points:],
            mode_predicted_sample + sample_y_shift,
            "--",
            color="red",
            alpha=1,
            linewidth=sample_lw[i],
            zorder=15,
        )
    mode_img = get_img_from_fig(fig, image_format=image_format)
    plt.close(fig)
    return mode_img


def visualize_map_process(
    summary_writer,
    agent_id,
    map_data,
    predictor_normalization_scale,
    agent_position,
    agent_tangent,
    trajectory_data,
    writer_global_params,
):
    map_data = get_cpu_detach(map_data)
    agent_position = get_cpu_detach(agent_position)
    agent_tangent = get_cpu_detach(agent_tangent)
    trajectory_data = get_cpu_detach(trajectory_data)
    visualize_map(
        summary_writer=summary_writer,
        agent_id=agent_id,
        map_data=map_data,
        predictor_normalization_scale=predictor_normalization_scale,
        agent_position=agent_position,
        agent_tangent=agent_tangent,
        trajectory_data=trajectory_data,
        writer_global_params=writer_global_params,
    )


def visualize_map(
    summary_writer,
    agent_id,
    map_data,
    predictor_normalization_scale,
    agent_position,
    agent_tangent,
    trajectory_data,
    writer_global_params,
):
    """Visualize map."""
    fig = plt.figure()
    gt_positions_example = map_data[agent_id, :, :2].view(-1, 2) / predictor_normalization_scale
    emitted_positions_example = agent_position[agent_id, ...].view(-1, 2) / predictor_normalization_scale
    tangent_example = agent_tangent[agent_id, ...].view(-1, 2)

    emitted_positions_example = emitted_positions_example.cpu().detach().numpy()
    gt_positions_example = gt_positions_example.cpu().detach().numpy()
    tangent_example = tangent_example.cpu().detach().numpy()
    plt.plot(
        gt_positions_example[:, 0],
        gt_positions_example[:, 1],
        "r.",
        emitted_positions_example[:, 0],
        emitted_positions_example[:, 1],
        ".",
    )
    plt.quiver(
        emitted_positions_example[:, 0],
        emitted_positions_example[:, 1],
        tangent_example[:, 0],
        tangent_example[:, 1],
        emitted_positions_example[:, 0] * 0.5,
    )
    if trajectory_data is not None:
        trajectory_example = (
            trajectory_data[agent_id, :, :2].view(-1, 2).cpu().detach().numpy() / predictor_normalization_scale
        )
        plt.plot(trajectory_example[:, 0], trajectory_example[:, 1], "k.")
    plt.axis("equal")
    map_view_margin_around_trajectories = 20
    x_min, x_max, y_min, y_max = plt.axis()
    min_trajectory_position_x = np.min(trajectory_example[:, 0])
    max_trajectory_position_x = np.min(trajectory_example[:, 0])
    min_trajectory_position_y = np.max(trajectory_example[:, 1])
    max_trajectory_position_y = np.max(trajectory_example[:, 1])
    x_min = max(x_min, min_trajectory_position_x - map_view_margin_around_trajectories)
    y_min = max(y_min, min_trajectory_position_y - map_view_margin_around_trajectories)
    x_max = min(x_max, max_trajectory_position_x + map_view_margin_around_trajectories)
    y_max = min(y_max, max_trajectory_position_y + map_view_margin_around_trajectories)
    plt.axis((x_min, x_max, y_min, y_max))

    summary_writer.add_figure("map_position_estimation_{}".format(agent_id), fig, global_step=writer_global_params)
    plt.close()


def _default_agent_colors(num_agents: int) -> List[np.ndarray]:
    """Returns list of RBG colors, one for each agent"""
    colormap = cm.get_cmap("viridis", num_agents * 3)
    colors = []
    for i in range(num_agents):
        c = colormap(i + num_agents)[:3]
        hsv = rgb_to_hsv(c)
        hsv[1] /= 2
        c = hsv_to_rgb(hsv)
        colors.append(c)
    return colors


def plot_roc_curve(y_true, y_score, image_format):
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fig = plt.figure()

    display = RocCurveDisplay(fpr=fpr, tpr=tpr)
    display.plot(fig.add_subplot())

    img = get_img_from_fig(fig, image_format=image_format)
    plt.close(fig)

    return img
