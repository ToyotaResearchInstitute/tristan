# A script to augment prediction protobufs with language tokens.
# Example use:
# python data_sources/augment_protobuf_with_language.py
# --source-protobufs-dir ~/argodataset/sample_pb_train/
# --augmented-protobufs-dir ~/argodataset/augmented_sample_pb_train/
# --smooth --viz

import argparse
import glob
import json
import os
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
import numpy as np
import tqdm
from google.protobuf.timestamp_pb2 import Timestamp

from data_sources.argoverse.trajectory_utils import smooth_trajectory
from data_sources.pedestrians.annotation_functor import AnnotationFunctor
from data_sources.waymo.annotation.html_visualization import MAX_NUM_SENTENCES, find_value_by_key
from intent.intent_conditions import (
    acceleration_filter,
    follow_filter,
    lane_change_filter,
    turn_filter,
    velocity_filter,
    yield_filter,
)
from intent.multiagents.trainer_visualization import wrap_text
from triceps.protobuf.prediction_dataset_semantic_handler import TYPE_LANGUAGE_TOKENS
from triceps.protobuf.prediction_training_pb2 import (
    LocalMap,
    PredictionInstance,
    PredictionSet,
    TimestampedSemanticTarget,
)

FILTER_PARAMS = {
    "turn": {"turn_threshold": 2.0, "window_size": 5, "window_std": 10.0},
    "velocity": {"vel_threshold": (1.0, 0.05)},
    "acceleration": {"acc_threshold": 0.04, "window_size": 5, "window_std": 0.04},
    "lane_change": {
        "lane_threshold": 0.3,
        "skip_threshold": 2,
        "window_size": 5,
        "max_intersection": 3,
        "max_lanes_in_window": 2,
        "intersection_radius": 7.5,  # Standard lane width is 3.7m in the US
    },
    "follow": {"min_overlaps": 10},
    "yield": {
        "yielding_time_gap": 1,
        "yielding_prefix": 5,
        "yielding_dt": 5,
        "yielding_dilation_radius": 0.5,
        "yielding_initial_distance": 5,
    },
}


def visualize_map_traj_tokens(
    agent_trajectories: OrderedDict,
    agent_id: str,
    map_elements: dict,
    tokens_dict: dict,
    relevant_agents: set,
    output_dir: str,
    params: dict,
):
    """Visualize the target agent trajectory along with the lane centers and the extracted tokens.

    Parameters
    ----------
    agent_traj: np.ndarray
        The trajectory of the target agent.
    agent_id: str
        The id of the target agent.
    map_elements: dict
        The dictionary contains the map elements including "lane_centers".
    token_dict: dict
        The dictionary that maps token type to a list of tokens of that type.
    relevant_agents: set
        The relevant agents for the target agent, i.e. has some interactions.
    output_dir: str
        The output directory of the image.
    params: dict
        Other params for augmentation.
    """
    figure_dir = os.path.join(output_dir, "figures")
    if not os.path.exists(figure_dir):
        os.mkdir(figure_dir)
    plt.clf()
    fig = plt.figure()
    fig.set_size_inches(14, 12)

    def plot_agent(agent_traj, agent_id, linespec, color, alpha, lw, zorder, plot_point=False):
        plt.plot(
            agent_traj[:, 0],
            agent_traj[:, 1],
            "-",
            color=color,
            alpha=alpha,
            linewidth=lw,
            zorder=zorder,
        )
        if plot_point:
            plt.plot(
                agent_traj[:, 0],
                agent_traj[:, 1],
                linespec,
                color=color,
                alpha=0.5,
                linewidth=0.5,
                zorder=zorder,
            )
        plt.text(agent_traj[0, 0], agent_traj[0, 1], "{}-s".format(agent_id))
        plt.text(agent_traj[-1, 0], agent_traj[-1, 1], "{}-e".format(agent_id))

    target_agent_id = list(agent_trajectories.keys())[0]
    plot_agent(agent_trajectories[target_agent_id], target_agent_id, "-x", "red", 1, 2.0, 15, plot_point=True)
    other_agent_ids = [agent_i for agent_i in agent_trajectories.keys() if agent_i != target_agent_id]
    for other_agent_id in other_agent_ids:
        if other_agent_id in relevant_agents:
            plot_agent(agent_trajectories[other_agent_id], other_agent_id, "-x", "blue", 1.0, 2.0, 15, plot_point=True)
        else:
            plot_agent(agent_trajectories[other_agent_id], other_agent_id, "-", "green", 0.8, 1.0, 0.8)

    for i, lane in enumerate(map_elements["lane_centers"]):
        line_coords = list(zip(*lane))
        lineX = line_coords[0]
        lineY = line_coords[1]
        plt.plot(lineX, lineY, "--", color="grey", alpha=1, linewidth=1, zorder=0)
        if not params["skip_viz_lane_labels"]:
            plt.text(lineX[0], lineY[0], "Lane{}-s".format(i))
            plt.text(lineX[-1], lineY[-1], "Lane{}-e".format(i))

    # Compute lane stats.
    title = "Agent id - {}".format(agent_id)
    # Add tokens to title if available.
    for name, tokens in tokens_dict.items():
        if params["source_caption_json_dir"] != "":
            title += wrap_text("\n {}".format("\n".join(tokens)), length_line=200)
        elif len(tokens) > 0:
            title += wrap_text("\n {}: {}".format(name, ",".join(tokens)), length_line=100)
    plt.title(title)
    fig.savefig(figure_dir + "/{}.png".format(agent_id))
    plt.close(fig)


class AugmentProtobufLanguage(AnnotationFunctor):
    """Augment the protobufs with language tokens."""

    def __init__(self, params):
        super().__init__(params)
        self.params = params
        os.makedirs(params["augmented_protobufs_dir"], exist_ok=True)
        self.should_augment_caption = params["source_caption_json_dir"] != ""

    def build_agent_trajectories(self, prediction_instance: PredictionInstance):
        """Turn the agent trajectory protobufs into an OrderedDict and skip the invalid points.

        Parameters
        ----------
        precition_instance: PredictionInstance
            The instance of the target scenario.

        Returns
        -------
        agent_trajectories: OrderedDict
            The map from agent_id to a Nx3 float array of positions over time, (x,y,t)
        """
        agent_trajectories = OrderedDict()
        for agent_id, agent_traj_pb in prediction_instance.agent_trajectories.items():
            traj_length = len(agent_traj_pb.trajectory)
            # Prepare trajectories.
            agent_traj = np.zeros((traj_length, 3))
            n_valid = 0
            for i, traj in enumerate(agent_traj_pb.trajectory):
                agent_traj[i][0] = traj.position.x
                agent_traj[i][1] = traj.position.y
                agent_traj[i][2] = traj.timestamp.seconds + traj.timestamp.nanos * 1e-9
                if np.sum(agent_traj[i][:2]) == 0:
                    break
                n_valid += 1
            agent_traj = agent_traj[:n_valid]
            # Do not include the agent if the number of valid points is too few.
            if len(agent_traj) < traj_length / 2:
                continue
            # Do not include the agent if the movement is too little.
            d = np.linalg.norm(agent_traj[-1, :2] - agent_traj[0, :2])
            if d < self.params["min_distance"]:
                continue
            if "smooth" in self.params and self.params["smooth"]:
                agent_traj[:, :2] = smooth_trajectory(agent_traj[:, :2], mode="other")
            agent_trajectories[agent_id] = agent_traj
        return agent_trajectories

    def agent_trajectories_by_id(self, agent_trajectories: OrderedDict, agent_id: str):
        """Reorder the agent_trajectories dictionary to have the target agent to be the first.

        Parameters
        ----------
        agent_trajectories: OrderedDict
            Input agent id to trajectory map.
        agent_id: str
            The target agent id.

        Returns
        -------
        new_trajs: OrderedDict
            The new agent id to trajectory map with the target agent to be the first.
        """
        new_trajs = OrderedDict({agent_id: agent_trajectories[agent_id]})
        agent_keys = [i for i in agent_trajectories.keys() if i != agent_id]
        for agent_key in agent_keys:
            new_trajs[agent_key] = agent_trajectories[agent_key]
        return new_trajs

    def build_map_elements(self, map_info: LocalMap):
        """Build the map element dictionary from a LocalMap.

        Parameters
        ----------
        map_info: LocalMap
            The map information of a given scenario.

        Returns
        -------
        map_elements: str
            The extracted map elements including
            'lane_centers': An list of lane center lines.
        """
        map_elements = defaultdict(list)
        for lane in map_info.lanes:
            if len(lane.center_line) == 0:
                continue
            centerlines = [[center_line.start.x, center_line.start.y] for center_line in lane.center_line]
            centerlines.append([lane.center_line[-1].end.x, lane.center_line[-1].end.y])
            map_elements["lane_centers"].append(np.array(centerlines))
        for feature in map_info.map_features:
            segments = [[segment.start.x, segment.start.y] for segment in feature.segments]
            segments.append([feature.segments[-1].end.x, feature.segments[-1].end.y])
            if feature.type == "TYPE_SPEED_BUMP":
                map_elements["speed_bump"].append(np.array(segments))
            elif feature.type == "TYPE_STOP_SIGN":
                map_elements["stop_sign"].append(np.array(segments))
            elif feature.type == "TYPE_CROSSWALK":
                map_elements["crosswalk"].append(np.array(segments))
            elif feature.type == "TYPE_SURFACE_STREET":
                map_elements["lane_centers"].append(np.array(segments))
        return map_elements

    def filter_tokens(self, agent_trajectories: OrderedDict, map_elements: dict):
        """Compute the tokens for the target agent with filters.

        Parameters
        ----------
        agent_trajectories: OrderedDict
            A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
        map_elements: dict
            The dictionary of map info. Contains "lane_centers".

        Returns
        -------
        output_tokens: list
            The extracted token list as semantic targets.
        """
        target_agent_id = list(agent_trajectories.keys())[0]
        output_tokens = []
        tokens_dict = {}

        # Generate the tokens for all filters.
        # We keep all tokens in semantic target and will have the handler to select what tokens to be used.
        filter_params = FILTER_PARAMS
        relevant_agents = set()
        filters = [
            ("yield", yield_filter, filter_params["yield"]),
            ("follow", follow_filter, filter_params["follow"]),
            ("turn", turn_filter, filter_params["turn"]),
            ("lane change", lane_change_filter, filter_params["lane_change"]),
            ("velocity", velocity_filter, filter_params["velocity"]),
            ("acceleration", acceleration_filter, filter_params["acceleration"]),
        ]
        for name, filter, param in filters:
            scene_information = {"map_elements": map_elements}
            tokens = filter(agent_trajectories, scene_information, param)
            if name not in tokens_dict:
                tokens_dict[name] = []
            for token in tokens:
                if "subject" in token:
                    label = "{} {}".format(token["label"], token["subject"])
                    relevant_agents.add(token["subject"])
                else:
                    label = token["label"]
                tokens_dict[name].append(label)
                # Add tokens to semantic target
                start_time = token["start_time"]
                end_time = token["end_time"]
                output_tokens.append(
                    TimestampedSemanticTarget(
                        timestamp_start=Timestamp(
                            seconds=int(start_time // 1), nanos=int((start_time - start_time // 1) * 1e9)
                        ),
                        timestamp_end=Timestamp(
                            seconds=int(end_time // 1), nanos=int((end_time - end_time // 1) * 1e9)
                        ),
                        is_interval=True,
                        agent_id=target_agent_id,
                        type_id=TYPE_LANGUAGE_TOKENS,
                        value=label,
                    )
                )
        if self.params["viz"] and len(output_tokens) > 0:
            visualize_map_traj_tokens(
                agent_trajectories,
                target_agent_id,
                map_elements,
                tokens_dict,
                relevant_agents,
                self.params["augmented_protobufs_dir"],
                self.params,
            )
        return output_tokens

    def build_caption_map(self):
        """Build a map from the video name (scenario_id + agent_id) to a list of annotated sentences.

        Returns
        -------
        captions: dict
           The dictionary that maps video name to captions.
        """
        captions = {}
        json_files_list = list(glob.glob(os.path.join(self.params["source_caption_json_dir"], "*.json")))
        for json_file in json_files_list:
            with open(json_file, "rb") as fp:
                responses = json.load(fp)
                for response in responses:
                    video_path = find_value_by_key(response, "s3Uri")
                    video_name = os.path.split(video_path)[1].replace(".mp4", "")
                    if video_name in captions:
                        continue
                    labels = find_value_by_key(response, "labels")
                    sentences = []
                    for s in range(MAX_NUM_SENTENCES):
                        s_idx = "s{}".format(s)
                        start_idx = "t{}".format(s)
                        end_idx = "e{}".format(s)
                        if s_idx not in labels or start_idx not in labels or end_idx not in labels:
                            break
                        if not labels[start_idx].isnumeric() or not labels[end_idx].isnumeric():
                            continue
                        sentences.append((labels[start_idx], labels[end_idx], labels[s_idx]))
                    captions[video_name] = sentences
        return captions

    def caption2pb(
        self,
        prediction_instance: PredictionInstance,
        target_agent_id: str,
        agent_trajectories: OrderedDict,
        map_elements: dict,
    ):
        """Look up the caption and convert it to semantic target.

        Parameters
        ----------
        prediction_instance: PredictionInstance
            The current prediction instance.
        target_agent_id: str
            The id of the target agent.
        agent_trajectories: OrderedDict
            A map from agent_id to a Nx3 float array of positions over time, (x,y,t)
        map_elements: dict
            The dictionary of map info. Contains "lane_centers".

        Returns
        -------
        semantic_targets: list
            The sentence list as semantic targets.
        """
        info = json.loads(prediction_instance.prediction_instance_info)
        scenario_id = info["source_tlog"]
        video_name = "{}_{}".format(scenario_id, target_agent_id)
        agent_traj = prediction_instance.agent_trajectories[target_agent_id]
        semantic_targets = []
        relevant_agents = []
        agent_ids = list(prediction_instance.agent_trajectories.keys())
        if video_name in self.captions:
            for start_idx, end_idx, sentence in self.captions[video_name]:
                if int(start_idx) >= len(agent_traj.trajectory) or int(end_idx) >= len(agent_traj.trajectory):
                    continue
                semantic_targets.append(
                    TimestampedSemanticTarget(
                        timestamp_start=agent_traj.trajectory[int(start_idx)].timestamp,
                        timestamp_end=agent_traj.trajectory[int(end_idx)].timestamp,
                        is_interval=True,
                        agent_id=target_agent_id,
                        type_id=TYPE_LANGUAGE_TOKENS,
                        value=sentence,
                    )
                )
                sentence = sentence.replace("agent #", "").replace("agent#", "")
                relevant_agents.extend(
                    [
                        agent_ids[int(token)]
                        for token in sentence.split(" ")
                        if token.isnumeric() and int(token) in agent_ids
                    ]
                )
        if self.params["viz"] and len(semantic_targets) > 0:
            visualize_map_traj_tokens(
                agent_trajectories,
                target_agent_id,
                map_elements,
                {"sentence": [semantic_target.value for semantic_target in semantic_targets]},
                relevant_agents,
                self.params["augmented_protobufs_dir"],
                self.params,
            )
        return semantic_targets

    def process_pb_file(self, pb_file):
        with open(pb_file, "rb") as fp:
            data = PredictionSet()
            data.ParseFromString(fp.read())
            for p_idx, prediction_instance in enumerate(data.prediction_instances):
                agent_trajectories = self.build_agent_trajectories(prediction_instance)
                for agent_i, agent_traj in agent_trajectories.items():
                    if self.should_augment_caption:
                        if self.params["viz"]:
                            traj_by_id = self.agent_trajectories_by_id(agent_trajectories, agent_i)
                            map_elements = self.build_map_elements(prediction_instance.map_information)
                        else:
                            traj_by_id = None
                            map_elements = None
                        token_seq = self.caption2pb(prediction_instance, agent_i, traj_by_id, map_elements)
                    else:
                        token_seq = self.filter_tokens(
                            self.agent_trajectories_by_id(agent_trajectories, agent_i),
                            self.build_map_elements(prediction_instance.map_information),
                        )
                    for token in token_seq:
                        data.prediction_instances[p_idx].semantic_targets.append(token)
        # Serialize the modified protobuf to the new folder.
        _, filename = os.path.split(pb_file)
        modified_file_path = os.path.join(self.params["augmented_protobufs_dir"], filename)
        with open(modified_file_path, "wb") as pbfile:
            pbfile.write(data.SerializeToString())

    def process(self):
        super().process(reharvester_jsn="", annotation_labels="", annotation_video_pathname="")
        if self.should_augment_caption:
            self.captions = self.build_caption_map()
        pb_files_list = list(glob.glob(os.path.join(self.params["source_protobufs_dir"], "*.pb")))
        if self.params["worker_count"] <= 1:
            for pb_file in tqdm.tqdm(pb_files_list, desc="Augmenting files"):
                self.process_pb_file(pb_file)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=self.params["worker_count"]) as executor:
                for pb_file in tqdm.tqdm(pb_files_list, desc="Augmenting files"):
                    future = executor.submit(self.process_pb_file, pb_file)
                    futures.append(future)
                [f.result() for f in futures]

    def finalize(self):
        return None


def add_arguments():
    """Parse the commandline options as dictionary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--source-protobufs-dir", type=str, default="./", help="Folder from which to read the source protobufs."
    )
    parser.add_argument(
        "--source-caption-json-dir",
        type=str,
        default="",
        help="Folder from which to read the source caption. If not set, use filters to augment.",
    )
    parser.add_argument(
        "--augmented-protobufs-dir", type=str, default="./", help="Folder to which to save the modified protobufs."
    )
    parser.add_argument("--worker-count", type=int, default=1, help="Number of workers for running augmentation.")
    parser.add_argument(
        "--smooth", action="store_true", help="Smooth agent trajectory before filtering. This is only for argoverse."
    )
    parser.add_argument(
        "--min-distance", type=float, default=5, help="Minimum distance to consider an agent in augmentation."
    )
    parser.add_argument("--viz", action="store_true", help="Visualize .")
    parser.add_argument("--skip-viz-lane-labels", action="store_true", help="Do not visualize lane labels.")
    args = parser.parse_args()
    return vars(args)


def main():
    params = add_arguments()
    augment_language = AugmentProtobufLanguage(params)
    augment_language.process()


if __name__ == "__main__":
    main()
