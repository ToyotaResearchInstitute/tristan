"""
Apply filters to instances and write out the results to JSON files.

There are also some optional ways to visualize the results of the filters.

This script takes many of the same arguments for data loading as the main
prediction trainer. For example, to run this script on waymo data (in protobuf
file format; the output of convert_waymo_data.py) we can use:
```
python intent/multiagents/create_resampling_data.py \
    --input-dir data/waymo/validation_pb --image-dir '' \
    --past-timesteps 6 --future-timesteps 39 --past-timestep-size 0.2 --future-timestep-size 0.2 \
    --max-agents 8 --vis-interval 10 --scene-image-mode=none --agent-image-mode=none \
    --num-visualization-images 10 \
    --datasets-name-lists none --ignore-ego \
    --visualize true --visualize-failed-instances true \
    --filter-strategy abrupt_driver ttc
```
The `--filter-strategy` option is one of the keys of `FILTER_STRATEGIES` below.
Watch the script output for `Output dir is ...` to see where the output JSON and
any visualizations are written.

Note: only the last two lines of arguments are specific to this script,
the other arguments are prediction dataloader arguments.

See notebooks/intent/filter_result_explorer.ipynb for a notebook to explore the
results.
"""

import copy
import json
import multiprocessing
import os.path
import time
from collections import OrderedDict
from functools import partial, wraps
from pathlib import Path
from typing import Callable, Dict, Optional

import holoviews as hv
import networkx as nx
import numpy as np
import torch
import tqdm

torch.multiprocessing.set_sharing_strategy("file_descriptor")
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy("file_descriptor")

from intent.intent_conditions import (
    TokenEncoder,
    acceleration_filter,
    conservative_yield_filter,
    follow_filter,
    merge_filter,
    turn_filter,
    turn_filter2,
)
from intent.multiagents.pedestrian_trajectory_prediction_util import (
    latent_factors_argument_setter,
    pedestrian_prediction_setter,
    prepare_pedestrian_model_params,
)
from intent.multiagents.training_config import TrainingConfig
from intent.multiagents.training_utils import (
    create_handlers,
    get_dataloader,
    load_datasets,
    prepare_cache,
    set_datasets_item_post_processors,
)
from intent.trajectory_prediction_misc import str2bool, trajectory_pred_argument_setter
from radutils.misc import generate_session_id
from radutils.visualization.trajectory import (
    ViewLimit,
    get_frame_bounds,
    render_visualizations,
    set_axis_limit,
    update_view_limit,
    visualize_point_map,
    visualize_trajectory_group,
    visualize_trajectory_group_heading,
)
from triceps.protobuf.prediction_dataset import PredictionDataset, ProtobufPredictionDataset
from triceps.protobuf.proto_arguments import args_to_dict, parse_arguments

multiprocessing.log_to_stderr()

AGENT_RESULT_PREFIX = "agent_"


def generate_instance_id(item, b):
    protobuf_file = item["protobuf_file"]
    return f"{protobuf_file[b]}"


def compute_item_filter_result_wrapper(
    filter_strategy: Callable[[dict, dict, dict], float], dataset, item: dict, params_in: dict
):
    stats = {}
    item["filter_result"] = filter_strategy(item, stats, params_in)
    item["stats"] = stats
    return item


def visualize_scenario(
    target_agent_trajectories,
    other_agent_trajectories,
    point_map,
    static_img_path: Path,
    animated_img_path: Optional[Path] = None,
    force: bool = False,
):
    """Visualize a scenario
    params:
        agent_trajectories: List[torch.Tensor]
            tensor: [timestamps, 3]

        point_map: torch.Tensor
            [points, 2]

        force: Bool
            Recompute visualizations even if they already exist
    return
        fig: plt.Figure
    """
    if not force and os.path.exists(static_img_path):
        print("Skipping already visualized trajectory")
        return

    total_frames = 12

    if not len(target_agent_trajectories):
        return
    max_time_point = max(len(traj) for traj in target_agent_trajectories)
    if animated_img_path:
        frame_limits = get_frame_bounds(total_frames, max_time_point)
    else:
        frame_limits = [max_time_point]

    # Render
    if point_map is not None:
        static_plots = [
            visualize_point_map(
                point_map=point_map,
                color="royalblue",
                size=4,
                point_params={"alpha": 0.5},
            ),
        ]
    else:
        static_plots = []
    plots_over_time = {}
    for frame_num in frame_limits:
        target = [traj[max(0, min(frame_num - 10, len(traj) - 10)) : frame_num] for traj in target_agent_trajectories]
        others = [traj[max(0, min(frame_num - 10, len(traj) - 10)) : frame_num] for traj in other_agent_trajectories]
        target_plot = visualize_trajectory_group(
            trajectories=target,
            trajectory_params={
                "show_steps": True,
                "color": "green",
                "path_opts": {
                    "linewidth": 1,
                },
                "point_opts": {
                    "alpha": 0.6,
                    "s": 6,
                },
            },
        )

        target_heading_plot = visualize_trajectory_group_heading(
            trajectories=target,
            trajectory_params={
                "color": "green",
                "last_only": True,
                "point_opts": {
                    "s": 50,
                },
                "align_to_front": False,
                "suppress": "stem",
            },
        )
        others_plot = visualize_trajectory_group(
            trajectories=others,
            trajectory_params={
                "show_steps": True,
                "color": "red",
                "path_opts": {
                    "linewidth": 1,
                },
                "point_opts": {
                    "alpha": 0.4,
                    "s": 5,
                },
            },
        )
        plots_over_time[frame_num] = [target_plot, target_heading_plot, others_plot]
    final_plots = plots_over_time[frame_limits[-1]]

    # Calculate trajectory bounds
    limit = ViewLimit()
    for traj in target_agent_trajectories:
        limit = update_view_limit(limit, traj)

    try:
        _, full_scenario_viz = render_visualizations(
            [],
            independent_visualizations=static_plots + final_plots,
            overlay_options=dict(padding=0.1),
            finalizers=[lambda overlay: set_axis_limit(overlay, limit)],
            render_options={"fmt": static_img_path.suffix.lstrip("."), "filename": static_img_path},
        )
    except ZeroDivisionError:
        # TODO (xiongyi.cui) Look into this.
        print("Warning, divided by zero.")
        return None, None

    if animated_img_path:
        _, animated_scenario_viz = render_visualizations(
            [],
            independent_visualizations=[
                *static_plots,
                hv.HoloMap(
                    {time_point: hv.Overlay(plots) for time_point, plots in plots_over_time.items()},
                    kdims="time_point",
                ),
            ],
            overlay_options=dict(padding=0.1),
            finalizers=[lambda overlay: set_axis_limit(overlay, limit)],
            render_options={"fmt": animated_img_path.suffix.lstrip("."), "filename": animated_img_path},
        )
    else:
        animated_scenario_viz = None

    return full_scenario_viz, animated_scenario_viz


def visualize(
    out_dir,
    animated_visualizations,
    instance_id,
    instance_index,
    item,
    targe_agent_trajectories,
    other_agent_trajectories: np.ndarray,
    force=False,
):
    static_img_format = "png"
    if animated_visualizations:
        animated_img_format = "gif"
    else:
        animated_img_format = None

    # Only visualize when it passes the filter
    point_map = item.get("map")
    if point_map is not None:
        point_map = point_map[point_map[:, 2] == 1, :2]

    agents = "_".join([str(a) for a in targe_agent_trajectories.keys()])
    name = f"fig_{instance_id.replace('/', '-')}-{instance_index}-agent_{agents}"
    full_scenario_viz_path = Path(out_dir, f"{name}.{static_img_format}")

    if animated_visualizations:
        animated_scenario_viz_path = Path(out_dir, f"{name}.{animated_img_format}")
    else:
        animated_scenario_viz_path = None

    visualize_scenario(
        targe_agent_trajectories.values(),
        other_agent_trajectories,
        point_map,
        static_img_path=full_scenario_viz_path,
        animated_img_path=animated_scenario_viz_path,
        force=force,
    )
    print(f"animation path {animated_scenario_viz_path}")
    return full_scenario_viz_path


def filter_from_intent_condition(
    filter_method: Callable, filter_params: dict, item: dict, stats: dict, params_in: dict
) -> list:
    """

    Parameters
    ----------
    item: dict
        A batch item with fields positions, dot_keys, protobuf_file, is_relevant_agent, timestamps (see the design doc for all fields)
    stats: dict
        A dictionary for statistics collection
    params_in: dict
        The related parameters

    Returns
    -------
    result: dict
        A dictionary mapping instance_id to a boolean filter result. instance_id is currently the protobuf name.

    """
    should_visualize = params_in["visualize"]
    animated_visualizations = params_in.get("animated_visualizations", False)
    vis_output_dir = params_in["filter_vis_output_dir"]
    instance_id = item["protobuf_file"]
    if vis_output_dir is None:
        vis_output_dir = params_in["logs_dir"]
    vis_failed_output_dir = params_in.get("vis_failed_output_dir", None)
    requires_two_agents = filter_params["requires_two_agents"]
    build_interaction_graph = filter_params.get("build_interaction_graph")
    interaction_graph_min_size = filter_params.get("interaction_graph_min_size")
    if build_interaction_graph:
        interaction_graph = nx.Graph()
    if interaction_graph_min_size is None:
        interaction_graph_min_size = 0

    if "total_seconds" not in stats:
        stats["instance_count"] = 0

    # Remove duplicate agents
    agent_index, agent_valid_index = np.unique(item["agent_index"], True)
    original_id_idx = item["agent_index"][: len(item["raw_agent_ids"])].astype(int)
    positions = item[ProtobufPredictionDataset.DATASET_KEY_POSITIONS]
    positions_for_vis = [p[p[:, -1] != 0] for p in positions]

    timestamps = item[ProtobufPredictionDataset.DATASET_KEY_TIMESTAMPS]
    is_relevant = item[ProtobufPredictionDataset.DATASET_KEY_IS_RELEVANT_AGENT]
    agent_types = item[ProtobufPredictionDataset.DATASET_KEY_AGENT_TYPE]
    try:
        raw_agent_ids = item[ProtobufPredictionDataset.DATASET_KEY_RAW_AGENT_IDS]
        has_raw_agent_ids = True
    except KeyError:
        raw_agent_ids = None
        has_raw_agent_ids = False
    except IndexError as err:
        print("raw id index error")
        # TODO (mark.flanagan.ctr): fix collation issue with raw_agent_ids in converted waymo protobufs
        instance_index = item[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INDEX]
        protobuf_file = item[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE]
        filter_result = {
            "instance_index": instance_index,
            "protobuf_file": protobuf_file,
            "matches": False,
            "occurrences": [],
            "instance_info": {},
        }
        return [
            instance_id,
            # convert to a json string so this result is safe for pytorch's collate method
            json.dumps(filter_result, cls=TokenEncoder),
        ]

    num_agents = positions.shape[0]
    all_agent_indices = list(range(num_agents))
    num_agents = int(is_relevant.sum())

    # TODO, replace this with named based agent type
    vehicle_type_idx = 0

    instance_index = item[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INDEX]
    protobuf_file = item[ProtobufPredictionDataset.DATASET_KEY_PROTOBUF_FILE]
    filter_result = {
        "instance_index": instance_index,
        "protobuf_file": protobuf_file,
        "matches": False,
        "occurrences": [],
    }
    agent_trajectories = OrderedDict()
    filter_result["instance_info"] = json.loads(item[ProtobufPredictionDataset.DATASET_KEY_INSTANCE_INFO])
    for agent_1_i in range(num_agents):

        if not is_relevant[agent_1_i]:
            continue

        # Determine agent indices for filter loop
        if requires_two_agents:
            current_agents = [agent_2_i for agent_2_i in range(num_agents) if agent_2_i != agent_1_i]
        else:
            current_agents = [agent_1_i]  # only runs the filter once per agent

        # TODO(xiongyi.cui) Agent types idx might be wrong here since we are not using --use-waymo-dataset.
        # Keep vehicles only
        if agent_types[agent_1_i][vehicle_type_idx] != 1:
            continue

        position1 = positions[agent_1_i]
        tstamps = timestamps
        valid1 = np.array(position1[:, 2], dtype=bool)

        valid_position1 = position1[valid1, :]
        if not (valid_position1.shape[0] > 3):
            continue

        for agent_2_i in current_agents:
            agent_trajectories = OrderedDict()

            agent_trajectories[agent_1_i] = copy.copy(position1[valid1, :])
            agent_trajectories[agent_1_i][:, 2] = tstamps[valid1]

            if requires_two_agents:
                if agent_types[agent_2_i][vehicle_type_idx] != 1:
                    continue
                position2 = positions[agent_2_i]
                valid2 = np.array(position2[:, 2], dtype=bool)
                valid_position2 = position2[valid2, :]
                if not (valid_position2.shape[0] > 3):
                    continue
                agent_trajectories[agent_2_i] = copy.copy(position2[valid2, :])
                agent_trajectories[agent_2_i][:, 2] = tstamps[valid2]

            # Run the filter on the agent(s), once per pair
            # TODO(nicholas.guyett.ctr) Add actual scene information, if available
            filter_tokens = filter_method(
                agent_trajectories=agent_trajectories, scene_information={}, params=filter_params
            )

            if filter_tokens:
                filter_result["matches"] = True
                occurrences = {}
                filter_result["occurrences"].append(occurrences)
                occurrences[f"info"] = {"tokens": filter_tokens}

                if requires_two_agents:
                    # TODO(justin.lidard): Do we need this if only most recent match will be stored?
                    occurrences["agents"] = [agent_1_i, agent_2_i]
                    if has_raw_agent_ids:
                        occurrences["raw_agent_ids"] = [raw_agent_ids[agent_1_i], raw_agent_ids[agent_2_i]]
                else:
                    occurrences["agents"] = [agent_1_i]
                    if has_raw_agent_ids:
                        occurrences["raw_agent_ids"] = [raw_agent_ids[agent_1_i]]

                if should_visualize:
                    other_agent_indices = copy.copy(all_agent_indices)
                    other_agent_indices.remove(agent_1_i)
                    other_agent_indices.remove(agent_2_i)
                    other_agent_trajectories = [
                        positions_for_vis[i] for i in other_agent_indices if len(positions_for_vis[i])
                    ]
                    full_scenario_viz_path = visualize(
                        vis_output_dir,
                        animated_visualizations,
                        instance_id,
                        instance_index,
                        item,
                        agent_trajectories,
                        other_agent_trajectories,
                        force=params_in["force"],
                    )
                    occurrences["vis_path"] = str(full_scenario_viz_path)

                if build_interaction_graph:
                    # Take the other agent matches from the filter and construct an edge list with the current agent
                    matches = [token["subject"] for token in filter_tokens]
                    source_nodes = [agent_1_i] * len(matches)
                    edge_list = [(agent_1_i, agent_2_i) for agent_1_i, agent_2_i in zip(source_nodes, matches)]
                    interaction_graph.add_edges_from(edge_list)

    # For network interactions, prune away the matches that don't have dense interactions
    if build_interaction_graph:
        if interaction_graph.number_of_nodes() > 0:  # if false, there was never a match, so do nothing
            largest_connected_component = max(nx.connected_components(interaction_graph), key=len)
            nodes_in_connected_component = len(largest_connected_component)
            if nodes_in_connected_component < interaction_graph_min_size:
                filter_result["matches"] = False
            else:
                filter_result["multi_interaction"] = {"num_interacting_agents": nodes_in_connected_component}
                interact_agent_trajectories = {}
                for i, n in enumerate(largest_connected_component):
                    agent_idx = n[i]
                    position2 = positions[agent_idx]
                    valid2 = np.array(position2[:, 2], dtype=bool)
                    valid_position2 = position2[valid2, :]
                    if not (valid_position2.shape[0] > 3):
                        continue
                    interact_agent_trajectories[agent_idx] = copy.copy(position2[valid2, :])
                    interact_agent_trajectories[agent_idx][:, 2] = tstamps[valid2]

                visualize(
                    vis_output_dir,
                    animated_visualizations,
                    instance_id,
                    instance_index,
                    item,
                    interact_agent_trajectories,
                    force=params_in["force"],
                )
    if should_visualize and not filter_result.get("matches"):
        # visualize(vis_failed_output_dir, animated_visualizations, instance_id, instance_index, item, agent_trajectories)

        full_scenario_viz_path = visualize(
            vis_output_dir,
            animated_visualizations,
            instance_id,
            instance_index,
            item,
            agent_trajectories,
            positions_for_vis,
            force=params_in["force"],
        )
        filter_result["vis_path"] = str(full_scenario_viz_path.parent / ("negative-" + full_scenario_viz_path.stem))

    if should_visualize and vis_failed_output_dir and not filter_result.get("matches") and agent_trajectories:
        visualize(vis_failed_output_dir, animated_visualizations, instance_id, instance_index, item, agent_trajectories)

    ret = [
        instance_id,
        # convert to a json string so this result is safe for pytorch's collate method
        json.dumps(filter_result, cls=TokenEncoder),
    ]
    return ret


def build_filter_strategy(intent_condition, filter_params):
    """Helper method that wraps an intent condition and filter params, so that it can be used as a filter strategy"""
    return partial(filter_from_intent_condition, intent_condition, filter_params)


def reduced_token_condition(filter_condition: Callable, *allowed_labels: str):
    @wraps(filter_condition)
    def condition(agent_trajectories, scene_information, params):
        filtered_tokens = [
            token
            for token in filter_condition(agent_trajectories, scene_information, params)
            if token["label"] in allowed_labels
        ]
        return filtered_tokens

    return condition


FILTER_STRATEGIES_KEYS = [
    "yield",
    "merge",
    "close_merge",
    "conservative_yield_r2_check",
    "multi_yield_3",
    "multi_yield_4",
    "multi_yield_5",
    "yield_2",
    "abrupt_driver",
    "left_turn",
    "left_turn2",
    "right_turn",
    "right_turn2",
    "follow",
    "ttc",
    "close_yield",
]
# Lookup dictionary used to map CLI parameters to filter methods
# TODO(guy.rosman): extract into a separate file, as filter strategies should match between e.g. resampling, generation of videos, and other tasks that can be in different scripts.
def create_filter_strategies(params: Dict):
    original_records_map = params.get("original_records_map", None)

    additional_params = {"original_records_map": original_records_map}
    FILTER_STRATEGIES = {
        "merge": build_filter_strategy(
            merge_filter,
            dict(
                {
                    "merge_time_threshold": 200,
                    # "yielding_time_gap": 1,
                    "yielding_prefix": 5,
                    # "yielding_dt": 5,
                    "yielding_dilation_radius": 0.5,
                    "yielding_initial_distance": 5,
                    "requires_two_agents": True,
                },
                **additional_params,
            ),
        ),
    }
    assert list(FILTER_STRATEGIES.keys()).sort() == FILTER_STRATEGIES_KEYS.sort(), (
        str(list(FILTER_STRATEGIES.keys())) + " != " + str(FILTER_STRATEGIES_KEYS)
    )

    return FILTER_STRATEGIES


def write_json(file, d):
    """Atomic write, prevent incomplete write on interrupt."""
    tmp_file = f"{file}.tmp"
    with open(tmp_file, "w") as fp:
        json.dump(d, fp, indent=2)
    os.rename(tmp_file, file)


class JSONWriter:
    def __init__(self):
        self.last_write_time = 0
        self.interval = 300  # write every 5 minutes

    def timed_write_json(self, file, j):
        """Atomic write, prevent incomplete write on interrupt."""
        if time.monotonic() - self.last_write_time < self.interval:
            # Only write longer than the interval
            return
        tmp_file = f"{file}.tmp"
        with open(tmp_file, "w") as fp:
            json.dump(j, fp, indent=2)
        os.rename(tmp_file, file)
        self.last_write_time = time.monotonic()


def read_json(file):
    try:
        with open(file, "r") as fp:
            return json.load(fp)
    except FileNotFoundError:
        return None


def get_dataset_filter_results(params, config, filter_strategy: Callable, output_file):
    """Runs over all the datasets, calculating the filter results for each instance"""

    existing_result = read_json(output_file)

    data_handlers, _ = create_handlers(params)

    datasets, datasets_name_lists = load_datasets(
        params,
        data_transforms=data_handlers,
        dataset_cls=config.dataset_cls,
        prediction_instance_cls=config.prediction_instance_cls,
    )

    set_datasets_item_post_processors(datasets, [partial(compute_item_filter_result_wrapper, filter_strategy)])

    datasets_iter = tqdm.tqdm(datasets.values(), desc="datasets", disable=params["disable_tqdm"])
    if existing_result is None or params["force"]:
        instance_filter_results = {}
    else:
        instance_filter_results = existing_result
    writer = JSONWriter()
    for dataset in datasets_iter:
        if existing_result is not None:
            # Read existing output_file and remove pb files if they have been filtered
            dataset_pbs = {a.rel_filename: a for a in dataset.datasets}
            print(f"in json {len(existing_result)}")
            print(f"in dataset {len(dataset_pbs)}")
            dataset_pbs_set = set(dataset_pbs)
            existing_result_set = set(existing_result)
            unseen = dataset_pbs_set - existing_result_set
            print(f"total unseen {len(unseen)}")
            unseen_in_this_dataset = unseen.intersection(dataset_pbs_set)
            print(f"intersection {len(unseen_in_this_dataset)}")
            new_datasets = [d for name, d in dataset_pbs.items() if name in unseen_in_this_dataset]
            print(f"unseen_in_dataset {len(new_datasets)}")
            if not new_datasets:
                continue
            dataset = PredictionDataset(new_datasets, dataset.params, dataset.dataset_label)

        print(f"\ndataset {dataset.dataset_label} has {len(dataset)} instances")
        print(f"Result writing to {output_file}")
        dataloader = get_dataloader(dataset, params)

        for i, itm in enumerate(tqdm.tqdm(dataloader, desc="filtering")):
            # TODO(guy.rosman): Add more test on the data items.
            data = {k: json.loads(v) for k, v in zip(*itm["filter_result"])}
            instance_filter_results.update(data)
            # Write every 5 minutes.
            writer.timed_write_json(output_file, instance_filter_results)

    return instance_filter_results


def run_filter(params, config, filter_strategy_name, strategy_dict):
    print(f"Running {filter_strategy_name} filter")
    output_dir = params["output_dir"]
    if params["vis_output_dir"] is not None:
        vis_output_dir = Path(params["vis_output_dir"])
    else:
        vis_output_dir = output_dir

    if params["vis_use_filter_prefix"]:
        vis_output_dir = vis_output_dir / ("vis_" + filter_strategy_name)

    vis_output_dir.mkdir(parents=True, exist_ok=True)
    if params["visualize_failed_instances"]:
        vis_failed_output_dir = vis_output_dir / "failed"
        vis_failed_output_dir.mkdir(parents=True, exist_ok=True)
        params["vis_failed_output_dir"] = str(vis_failed_output_dir)
    params["filter_vis_output_dir"] = str(vis_output_dir)

    default_output_file = output_dir / f"{filter_strategy_name}.instance_filter_results.json"
    output_file = params.get("output_file")
    if output_file is None:
        output_file = default_output_file
    else:
        output_file = output_dir / output_file

    instance_filter_results = get_dataset_filter_results(
        params, config, strategy_dict[filter_strategy_name], output_file
    )

    write_json(output_file, instance_filter_results)

    criteria_match_count = sum(filter_results["matches"] for filter_results in instance_filter_results.values())

    print(
        f"{filter_strategy_name} filter: processed {len(instance_filter_results)} instances, {criteria_match_count} fit the criteria"
    )


def main(args):
    params = args_to_dict(args, additional_file_path_params=["latent_factors_file"])
    params.setdefault("past_timestep_starting_offset", None)
    params.setdefault("future_timestep_starting_offset", None)

    params = prepare_pedestrian_model_params(params)

    if params["image_list_key_num"] <= 1:
        # Set it to minimum of 2 do run_validation
        params["image_list_key_num"] = 2

    config = TrainingConfig.from_params(params)
    data_handlers, _ = create_handlers(params)
    params = prepare_cache(params, data_handlers)
    session_id = generate_session_id(None, "filter_result")
    output_dir = Path(params["logs_dir"], "../filter_result", params["log_name_string"])
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir is {output_dir}")
    params["output_dir"] = output_dir

    FILTER_STRATEGIES = create_filter_strategies(params)
    for filter_strategy_name in params["filter_strategy"]:
        run_filter(params, config, filter_strategy_name, FILTER_STRATEGIES)


def resample_filter_argument_setter(parser):
    parser.add_argument(
        "--visualize", type=str2bool, default="false", help="Visualize the filter result, the passed instances."
    )
    parser.add_argument("--run-data-stats", type=str2bool, default="false", help="Compute the statistics of the data.")
    parser.add_argument(
        "--animated-visualizations",
        type=str2bool,
        default="false",
        help="Generated animated visualizations",
    )
    parser.add_argument(
        "--visualize-failed-instances",
        type=str2bool,
        default="false",
        help="Visualize the filter result, the failed instances.",
    )
    parser.add_argument(
        "--vis-use-filter-prefix",
        default="true",
        type=str2bool,
        help="If true, visualizations will be saved under a separate subdirectory per filter",
    )
    parser.add_argument("--vis-output-dir", type=str, default=None, help="The output dir of visualization images.")
    parser.add_argument(
        "--filter-strategy",
        choices=list(FILTER_STRATEGIES_KEYS),
        default=(),
        nargs="*",
        help="The instance filtering strategy to use. Can select one or more; if none specified then all filters will be use.",
    )
    parser.add_argument(
        "--single-process",
        type=str2bool,
        default="false",
        help="Use single process to load data, mostly for debugging purpose",
    )
    parser.add_argument(
        "--force",
        type=str2bool,
        default="false",
        help="Recompute filter, overwrites existing results",
    )
    parser.add_argument("--output-file", default=None, help="File path to serialize results to")

    # Compatibility flags used by private training launch commands. The
    # resampling filter does not consume these directly, but accepting them lets
    # existing launch lines run against the public release parser.
    parser.add_argument("--limit-cuda-memory", type=float, default=None)
    parser.add_argument("--encoder-normalized-other-agents-position", type=str2bool, default=False)
    parser.add_argument("--encoder-use-map", type=str2bool, default=False)
    parser.add_argument("--plot-debug-normalization", type=str2bool, default=False)
    parser.add_argument("--use-transformer-baseline-encoder", type=str2bool, default=False)
    parser.add_argument("--encoder-num-mlp-layers", type=int, default=None)
    parser.add_argument("--decoder-num-mlp-layers", type=int, default=None)
    parser.add_argument("--use-anchor-trajectories", default=None)
    parser.add_argument("--use-simple-metrics", type=str2bool, default=False)
    parser.add_argument("--map-use-vectors", type=str2bool, default=False)
    return parser


def parse_args(args=None, additional_arguments_setter=None):
    if not additional_arguments_setter:
        additional_arguments_setter = []

    return parse_arguments(
        args,
        additional_arguments_setter=[
            trajectory_pred_argument_setter,
            latent_factors_argument_setter,
            pedestrian_prediction_setter,
            resample_filter_argument_setter,
            *additional_arguments_setter,
        ],
    )


if __name__ == "__main__":
    main(parse_args())
