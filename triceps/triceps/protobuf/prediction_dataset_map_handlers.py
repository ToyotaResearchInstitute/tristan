import copy
import hashlib
from enum import IntEnum
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import shapely
import shapely.geometry.linestring
import torch

import triceps.protobuf
import util.rasterize
from radutils.torch.torch_utils import apply_2d_coordinate_rotation_transform
from triceps.protobuf import protobuf_training_parameter_names
from triceps.protobuf.prediction_dataset import ProtobufPredictionDataset
from triceps.protobuf.prediction_dataset_auxiliary import InputsHandler
from triceps.protobuf.prediction_dataset_cache import CacheElement


class MapDataIndices(IntEnum):
    """
    Save the indices for different element in map data tensor.
    """

    MAP_IDX_VALIDITY = 2
    MAP_IDX_TYPE = 3
    MAP_IDX_TANGENT = 4
    MAP_IDX_NORMAL = 6
    MAP_IDX_ID = 8


class MapPointType(IntEnum):
    UNDEFINED = 0
    CENTER_LANE_LINE = 1
    RIGHT_LANE_LINE = 2
    LEFT_LANE_LINE = 3
    CROSSWALK = 4
    LANE_BOUNDARY_LINE = 5


def plot_road_elements(element_points, filename):
    """
    Allows plotting of road elements from within the PointMapHandler.

    Parameters
    ----------
    element_points : list
        A list of pairs of coordinates
    filename : str
        A filename to save the figure to.
    """
    plt.figure()
    for i in range(len(element_points)):
        plt.plot(
            [x for x, y in element_points[i]],
            [y for x, y in element_points[i]],
            "-",
            [x for x, y in element_points[i]],
            [y for x, y in element_points[i]],
            "o",
        )
    if filename is not None:
        plt.savefig(filename, dpi=600)
        plt.close()


class RasterMapHandler(InputsHandler):
    def __init__(self, params, map_scale=0.2, halfwidth=30, halfheight=30):
        """

        :param map_scale: Map resolution in meters.
        :param halfwidth: Map "radius" in meters.
        :param halfheight: Map "radius" in meters.
        """
        super().__init__(params)
        self.halfwidth = halfwidth
        self.halfheight = halfheight
        self.map_scale = map_scale
        self.pixel_width = np.int64(np.ceil(self.halfwidth / self.map_scale) * 2 + 1)
        self.pixel_height = np.int64(np.ceil(self.halfheight / self.map_scale) * 2 + 1)

    def get_hash_param_keys(self) -> List[str]:
        return [
            "disable_map_input",
            "map_input_type",
            "map_halfwidth",
            "map_halfheight",
            "map_scale",
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params,
        filename,
        index,
    ):
        ret = {}

        if not result_dict[ProtobufPredictionDataset.DATASET_KEY_MAP_INFORMATION] is not None:
            ret[ProtobufPredictionDataset.DATASET_KEY_MAP] = np.zeros(
                [
                    params[protobuf_training_parameter_names.PARAM_MAX_TIMESTEPS],
                    3,
                    self.halfheight * 2 + 1,
                    self.halfwidth * 2 + 1,
                ],
                dtype=np.float32,
            )
        else:
            position = result_dict[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][0, 0, :2]
            map_information = result_dict["mapInformation"]
            raster = util.rasterize.Rasterize(
                scale=self.map_scale,
                halfwidth=self.halfwidth,
                halfheight=self.halfheight,
                ego_position=position,
                ego_angle=0.0,
                maptype="boundary",
            )
            raster.start_raster()

            for lane in map_information["lanes"]:
                rlb_line = lane.right_lane_boundary
                if rlb_line is not None:
                    ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in rlb_line] + [
                        [rlb_line[-1]["end"]["x"], rlb_line[-1]["end"]["y"]]
                    ]
                    raster.add_lane(ln_coords, value=(1, 0, 0))
            for lane in map_information["lanes"]:
                llb_line = lane.left_lane_boundary
                if llb_line is not None:
                    ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in llb_line] + [
                        [llb_line[-1]["end"]["x"], llb_line[-1]["end"]["y"]]
                    ]
                    raster.add_lane(ln_coords, value=(0, 1, 0))

            for lane in map_information["lanes"]:
                center_line = lane.center_line
                ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in center_line] + [
                    [center_line[-1]["end"]["x"], center_line[-1]["end"]["y"]]
                ]
                raster.add_lane(ln_coords, value=(0, 0, 1))

            for zone in map_information["zones"]:
                zone_line = zone.polygon
                ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in zone_line] + [
                    [zone_line[-1]["end"]["x"], zone_line[-1]["end"]["y"]],
                    [zone_line[0]["start"]["x"], zone_line[0]["start"]["y"]],
                ]
                raster.add_lane(ln_coords, value=(1, 0, 1))

            img = raster.get_raster()
            img2 = np.ones([self.pixel_height, self.pixel_width, img.shape[2]]) * 255
            img2[
                (self.pixel_height - img.shape[0]) // 2 : (self.pixel_height - img.shape[0]) // 2 + img.shape[0],
                (self.pixel_width - img.shape[1]) // 2 : (self.pixel_width - img.shape[1]) // 2 + img.shape[1],
                :,
            ] = img
            img = img2
            img = img.transpose([2, 1, 0])
        ret[ProtobufPredictionDataset.DATASET_KEY_MAP] = np.float32(
            np.tile(img, [params[protobuf_training_parameter_names.PARAM_MAX_TIMESTEPS], 1, 1, 1])
        )
        return ret


def create_polynomial_features(ln_coords, horizon=2, stepsize=0.5, degree=3):
    """Creates polynomial interpolation features, to allow more efficient leveraging of lane information.

    Parameters
    ----------
    ln_coords: list, list of lane coordinates.
    horizon: float, the horizon for which we want the approximation to be good.
    stepsize: float, the step size for sample points.
    degree: int, the degree of the polynomial

    Returns
    -------
    result_polynomial_features: np.array, of size n_points x (2xdegree), to capture polynomial coefficients on x,y.
    """
    # The offsets (arclength along the road) that we want to compute positions for.
    offsets = np.arange(start=0, stop=horizon, step=stepsize)
    num_sample_points = len(offsets)
    num_positions = len(ln_coords)
    vandermonde = np.vander(offsets, degree)
    A = []
    ln_coords_shifted = copy.copy(ln_coords)
    for i in range(num_positions):
        ln_polyline = shapely.geometry.linestring.LineString(ln_coords_shifted)
        sampled_positions = [ln_polyline.interpolate(o).xy for o in offsets]
        sampled_positions = np.array(sampled_positions).squeeze(2)
        A.append(sampled_positions)
        ln_coords_shifted = ln_coords_shifted[1:] + [ln_coords_shifted[-1]]
    A = np.array(A)
    dim = 2
    p_inv_vandermonde = np.linalg.pinv(vandermonde)
    stacked_A = np.concatenate([A[:, :, d] for d in range(dim)], 0).transpose()
    coeffs = np.matmul(p_inv_vandermonde, stacked_A)
    result_polynomial_features = np.concatenate(np.split(coeffs, dim, 1), 0)
    return result_polynomial_features


class PointMapHandler(InputsHandler):
    def __init__(self, params, max_point_num=10, sampling_length=None, sampling_minimum_length=None):
        """

        Parameters
        ----------
        max_point_num : int
            Max number of points in each map element.
        sampling_length : float
            Typical length to resample map elements.
        sampling_minimum_length : float
            Minimum length for resampling map elements.
        """

        super().__init__(params)
        self.max_point_num = max_point_num
        self.sampling_length = sampling_length
        self.sampling_minimum_length = sampling_minimum_length

    def get_hash_param_keys(self) -> List[str]:
        return [
            "disable_map_input",
            "map_input_type",
            "map_points_max",
            "map_sampling_length",
            "map_sampling_minimum_length",
            "map_polyline_feature_degree",  # This not used in this class, but as an agent_input_handlers.
        ]

    def _process_impl(
        self,
        result_dict: dict,
        params: dict,
        filename: str,
        index: int,
    ) -> dict:
        """Given an instance input, returns corresponding map information


        Parameters
        ----------
        result_dict : dict
            result of previous handlers.
        instance :  dict
            a prediction instance.
        params :  dict
            config parameters.
        filename : str
            not used.
        index : int
            not used.

        Returns
        -------
        dict
            updated result_dict with an added key "map" that contains map coordinates
            of the shape  (max_point_num, 9], the last dimension
            contains (x_pos, y_pos, validity, point type, sin(theta), cos(theta), cos(theta), -sin(theta), id).
        """
        # Initialize the results to 0s.
        # Each point has 9 dim: (x, y, validity, point type, sin(theta), cos(theta), cos(theta), -sin(theta), id).
        result = np.zeros([self.max_point_num, 9], dtype=np.float32)
        ret = {}

        # Pool the map elements that are closest to the first agent's most recent observed position.
        agent_timestep = result_dict["num_past_points"] - 1

        positions = []
        num_agents = result_dict[ProtobufPredictionDataset.DATASET_KEY_POSITIONS].shape[0]
        for i in range(num_agents):
            agent_position = result_dict[ProtobufPredictionDataset.DATASET_KEY_POSITIONS][i, agent_timestep, :2]
            positions.append(agent_position)
        positions = np.stack(positions)

        # Compute center of all agent positions to pool close map elements.
        position = np.mean(positions, 0)

        map_information: dict = result_dict[ProtobufPredictionDataset.DATASET_KEY_MAP_INFORMATION]
        element_points = []
        dists_to_agent = []
        point_types = []
        point_ids = []
        all_tangent_vectors = []
        all_normal_vectors = []
        point_id = 0

        def append_map_local_features(ln_coords, all_tangent_vectors, all_normal_vectors):
            """Append local features to  all_tangent_vectors

            Parameters
            ----------
            ln_coords: list, list of coordinates
            all_tangent_vectors: list to be appended to, of local features. Currently: cos, sin of the tangent at that point.

            """
            tangent_vectors = [
                [elem1[0] - elem2[0], elem1[1] - elem2[1]] for elem1, elem2 in zip(ln_coords[1:], ln_coords[:-1])
            ]
            normal_vectors = [
                [elem1[1] - elem2[1], elem2[0] - elem1[0]] for elem1, elem2 in zip(ln_coords[1:], ln_coords[:-1])
            ]
            # poly_features = create_polynomial_features(ln_coords)
            if len(tangent_vectors) > 0:
                # Populate the last tangent/normal value with based on the last 2 positions
                # (similar to the one-before-last)
                tangent_vectors = np.array(tangent_vectors + [tangent_vectors[-1]])
                normal_vectors = np.array(normal_vectors + [normal_vectors[-1]])
            else:
                tangent_vectors = np.array([[0, 0]])
                normal_vectors = np.array([[0, 0]])

            # Compute normalized tangent vectors, as sin and cos.
            try:
                # Add small epsilon to denominator to avoid divide by 0.
                tangent_vectors_normalized = tangent_vectors / (
                    np.linalg.norm(np.array(tangent_vectors), axis=1, keepdims=True) + 1e-20
                )

                normal_vectors_normalized = normal_vectors / (
                    np.linalg.norm(np.array(normal_vectors), axis=1, keepdims=True) + 1e-20
                )
            except:
                normal_vectors_normalized = normal_vectors * 0
                tangent_vectors_normalized = tangent_vectors * 0

            all_tangent_vectors.append(tangent_vectors_normalized)
            all_normal_vectors.append(normal_vectors_normalized)

        # Collect information from map elements and compute distance to target agent.
        def update_map(
            ln_coords,
            point_type,
            point_id,
            element_points,
            point_types,
            all_tangent_vectors,
            all_normal_vectors,
        ):
            if self.sampling_length is not None:
                ln_coords = copy.copy(ln_coords)
                ln_coords_ls = shapely.geometry.linestring.LineString(ln_coords)
                sampling_length = min(self.sampling_length, ln_coords_ls.length / len(ln_coords))
                if self.sampling_minimum_length is not None:
                    sampling_length = max(sampling_length, self.sampling_minimum_length)
                ln_coords_interp = [
                    ln_coords_ls.interpolate(x) for x in np.arange(0, ln_coords_ls.length, sampling_length)
                ]
                ln_coords_interp = [[pt.x, pt.y] for pt in ln_coords_interp]
                ln_coords = ln_coords_interp
            dist_to_agent = np.min(np.sum((np.array(ln_coords) - np.array(position)) ** 2, axis=-1))
            dists_to_agent.append(dist_to_agent)
            element_points.append(np.array(ln_coords))
            point_types.append(np.array([point_type] * len(ln_coords)))
            append_map_local_features(ln_coords, all_tangent_vectors, all_normal_vectors)

            # Assign a unique id to each element for tracking.
            point_ids.append(np.array([point_id] * len(ln_coords)))
            assert len(point_ids[-1]) == len(point_types[-1])
            assert len(point_ids[-1]) == len(all_tangent_vectors[-1])
            assert len(point_ids[-1]) == len(all_normal_vectors[-1])
            assert len(point_ids[-1]) == len(element_points[-1])

        for lane in map_information.get("lanes", []):
            if "leftLaneBoundary" in lane:
                llb_line = lane["leftLaneBoundary"]
                if llb_line is not None and len(llb_line) > 1:
                    ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in llb_line] + [
                        [llb_line[-1]["end"]["x"], llb_line[-1]["end"]["y"]]
                    ]
                    if ln_coords[0] == ln_coords[-1]:
                        # If the points defining the lane boundary are all the same point,
                        # this causes an error with the interpolation when updating the map.
                        # Since the lane definition is poor, just skip adding the lane to the map.
                        continue
                    update_map(
                        ln_coords,
                        MapPointType.LEFT_LANE_LINE,
                        point_id,
                        element_points,
                        point_types,
                        all_tangent_vectors,
                        all_normal_vectors,
                    )
                    point_id += 1

            if "rightLaneBoundary" in lane:
                rlb_line = lane["rightLaneBoundary"]
                if rlb_line is not None and len(rlb_line) > 1:
                    ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in rlb_line] + [
                        [rlb_line[-1]["end"]["x"], rlb_line[-1]["end"]["y"]]
                    ]
                    if ln_coords[0] == ln_coords[-1]:
                        # If the points defining the lane boundary are all the same point,
                        # this causes an error with the interpolation when updating the map.
                        # Since the lane definition is poor, just skip adding the lane to the map.
                        continue
                    update_map(
                        ln_coords,
                        MapPointType.RIGHT_LANE_LINE,
                        point_id,
                        element_points,
                        point_types,
                        all_tangent_vectors,
                        all_normal_vectors,
                    )
                    point_id += 1

            center_line = lane.get("centerLine")
            if center_line is not None and len(center_line) > 1:
                # Filter out missing data in center line.
                new_center_line = []
                for elem in center_line:
                    if "x" in elem["start"] and "x" in elem["end"] and "y" in elem["start"] and "y" in elem["end"]:
                        new_center_line.append(elem)
                center_line = new_center_line

                ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in center_line] + [
                    [center_line[-1]["end"]["x"], center_line[-1]["end"]["y"]]
                ]
                if ln_coords[0] == ln_coords[-1]:
                    # If the points defining the lane center line are all the same point,
                    # this causes an error with the interpolation when updating the map.
                    # Since the lane definition is poor, just skip adding the lane to the map.
                    continue
                update_map(
                    ln_coords,
                    MapPointType.CENTER_LANE_LINE,
                    point_id,
                    element_points,
                    point_types,
                    all_tangent_vectors,
                    all_normal_vectors,
                )
                point_id += 1

            if "laneBoundary" in lane:
                boundary_line = lane["laneBoundary"]
                if boundary_line is not None and len(boundary_line) > 1:
                    ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in boundary_line] + [
                        [boundary_line[-1]["end"]["x"], boundary_line[-1]["end"]["y"]]
                    ]
                    update_map(
                        ln_coords,
                        MapPointType.LANE_BOUNDARY_LINE,
                        point_id,
                        element_points,
                        point_types,
                        all_tangent_vectors,
                        all_normal_vectors,
                    )
                    point_id += 1

        for zone in map_information.get("zones", []):
            zone_line = zone["polygon"]
            if zone_line is not None and len(zone_line) > 1:
                ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in zone_line] + [
                    [zone_line[-1]["end"]["x"], zone_line[-1]["end"]["y"]],
                    [zone_line[0]["start"]["x"], zone_line[0]["start"]["y"]],
                ]
                update_map(
                    ln_coords,
                    MapPointType.CROSSWALK,
                    point_id,
                    element_points,
                    point_types,
                    all_tangent_vectors,
                    all_normal_vectors,
                )
                point_id += 1

        # Add individual map features from map data.
        for map_feature in map_information.get("mapFeatures", []):
            segments = map_feature["segments"]
            if segments is not None and len(segments) > 1:
                ln_coords = [[elem["start"]["x"], elem["start"]["y"]] for elem in segments] + [
                    [segments[-1]["end"]["x"], segments[-1]["end"]["y"]],
                    [segments[0]["start"]["x"], segments[0]["start"]["y"]],
                ]
                update_map(
                    ln_coords,
                    MapPointType.CENTER_LANE_LINE,
                    point_id,
                    element_points,
                    point_types,
                    all_tangent_vectors,
                    all_normal_vectors,
                )
                point_id += 1

        # Pool information from all elements together, if map exists. Otherwise return a 0 tensor.
        assert len(element_points) == len(point_types)
        assert len(element_points) == len(point_ids)
        assert len(element_points) == len(all_tangent_vectors)
        assert len(element_points) == len(all_normal_vectors)

        if len(element_points) > 0:
            element_points = np.vstack(element_points)
            point_types = np.hstack(point_types)
            point_ids = np.hstack(point_ids)
            all_tangent_vectors = np.vstack([np.array([[0, 0]]) if len(x) == 0 else x for x in all_tangent_vectors])
            all_normal_vectors = np.vstack([np.array([[0, 0]]) if len(x) == 0 else x for x in all_normal_vectors])

            # Each point has 9 dimensions: (x, y, validity, type, tan_x, tan_y, normal_x, normal_y, id).
            point_size = np.minimum(self.max_point_num, element_points.shape[0])
            result[:point_size, :2] = element_points[:point_size]
            result[:point_size, MapDataIndices.MAP_IDX_VALIDITY] = np.ones(point_size)
            result[:point_size, MapDataIndices.MAP_IDX_TYPE] = point_types[:point_size]
            result[
                :point_size, MapDataIndices.MAP_IDX_TANGENT : (MapDataIndices.MAP_IDX_TANGENT + 2)
            ] = all_tangent_vectors[:point_size]
            result[
                :point_size, MapDataIndices.MAP_IDX_NORMAL : (MapDataIndices.MAP_IDX_NORMAL + 2)
            ] = all_normal_vectors[:point_size]
            result[:point_size, MapDataIndices.MAP_IDX_ID] = point_ids[:point_size]

        ret[ProtobufPredictionDataset.DATASET_KEY_MAP] = result

        return ret


def normalize_agent_additional_inputs(agent_additional_inputs, transforms, should_normalize=False, params=None):
    """
    Normalize map coordinates to agent local frame.
    Parameters
    ----------
    agent_additional_inputs
        dictionary including additional inputs to the agent, e.g., map inputs, with dim [num_batch, num_max_point, 9],
        where the last dim include
        (x, y, validity, point type, sin(theta), cos(theta), cos(theta), -sin(theta), point id).
    transforms
        local transform matrix, with shape [num_batch, num_agent, 3, 2].
    """
    assert "map" in agent_additional_inputs, "Map not available in agent_additional_inputs"
    normalized_map_data = agent_additional_inputs["map"].clone()

    # Skip if the map data is already normalized.
    # Assume agent-centric normalized map as a 4d tensor [num_batch, num_agent, num_max_point, 9],
    # and unnormalized map as a 3d tensor [num_batch, num_max_point, 9].
    if len(normalized_map_data.shape) > 3:
        return

    num_agent = transforms.shape[1]
    num_batch, num_max_point, _ = tuple(normalized_map_data.shape)
    # Make a copy for each agent.
    # Shape [num_batch, num_agent, num_max_point, 9]
    normalized_map_data = normalized_map_data.unsqueeze(1).repeat(1, num_agent, 1, 1)

    if should_normalize:
        # Shift positions.
        normalized_map_data[..., :2] = normalized_map_data[..., :2] + transforms[:, :, 2].unsqueeze(2)
        # Rotate positions.
        normalized_map_data[..., :2] = torch.matmul(normalized_map_data[..., :2], transforms[:, :, :2])
        # Rotate angles.
        tangent_slice = slice(MapDataIndices.MAP_IDX_TANGENT, (MapDataIndices.MAP_IDX_TANGENT + 2))
        normalized_map_data[..., tangent_slice] = apply_2d_coordinate_rotation_transform(
            transforms[:, :, :2],
            normalized_map_data[..., tangent_slice],
            result_einsum_prefix="bam",  # batch_size x num_map_points
            rotation_einsum_prefix="ba",  # batch_size x num_agents
        )

        normal_slice = slice(MapDataIndices.MAP_IDX_NORMAL, (MapDataIndices.MAP_IDX_NORMAL + 2))
        normalized_map_data[..., normal_slice] = apply_2d_coordinate_rotation_transform(
            transforms[:, :, :2],
            normalized_map_data[..., normal_slice],
            result_einsum_prefix="bam",
            rotation_einsum_prefix="ba",
        )
    # Save locally normalized result
    agent_additional_inputs["map"] = normalized_map_data


def add_polynomial_features(agent_additional_inputs, transforms=None, should_normalize=False, params=None):
    """
    Add polynomial features to map tensor.
    Parameters
    ----------
    agent_additional_inputs
        dictionary including additional inputs to the agent, e.g., map inputs, with dim [num_batch, num_elements,
        num_points_per_element, 6], where the last dim includes (x, y, sin(theta), cos(theta), point type, validity).
    params
        dictionary including parameters.
    Returns
    -------

    """
    assert "map" in agent_additional_inputs, "Map not available in agent_additional_inputs"
    map_data = agent_additional_inputs["map"].clone()
    batch_size, num_agent, num_map_elements, num_point_per_element, _ = map_data.shape
    map_coordinates = map_data[..., :2]
    map_validity = map_data[..., MapDataIndices.MAP_IDX_VALIDITY]
    map_coordinates = map_coordinates.view(-1, num_point_per_element, 2).cpu().detach().numpy()
    map_validity = map_validity.view(-1, num_point_per_element).cpu().detach().numpy()
    polynomial_degree = params["map_polyline_feature_degree"]
    polynomial_features = torch.zeros(map_validity.shape[0], num_point_per_element, polynomial_degree * 2)
    for i in range(map_coordinates.shape[0]):
        validity = map_validity[i]
        if np.sum(validity) > 1:
            ln_coords = map_coordinates[i][validity > 0]
            # Create cache for poly feature.
            sha = hashlib.md5()
            sha.update(ln_coords.tostring())
            sha.update(bytes(polynomial_degree))
            ln_coords_hash = sha.hexdigest()

            cache_id = "map_poly_" + ln_coords_hash + "_" + str(polynomial_degree)
            cache_poly = CacheElement(params["cache_dir"], cache_id, "pkl", disable_cache=params["disable_cache"])
            if cache_poly.is_cached():
                poly_feature = np.array(cache_poly.load())
            else:
                # Compute poly feature if not cached.
                poly_feature = create_polynomial_features(ln_coords.tolist(), degree=polynomial_degree).T
                cache_poly.save(poly_feature.tolist())
            polynomial_features[i, : ln_coords.shape[0]] = torch.from_numpy(poly_feature)
    polynomial_features = polynomial_features.view(batch_size, num_agent, num_map_elements, num_point_per_element, -1)
    polynomial_features = polynomial_features.to(map_data.device)
    agent_additional_inputs["map"] = torch.cat((map_data, polynomial_features), -1)
