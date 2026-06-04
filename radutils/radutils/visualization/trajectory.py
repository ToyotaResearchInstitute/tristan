from functools import reduce
from typing import Callable, Iterable, List, Optional, Tuple

import holoviews as hv
import numpy as np
import torch

hv.extension("matplotlib")


class ViewLimit:
    def __init__(self):
        self.x_min = float("inf")
        self.x_max = float("-inf")
        self.y_min = float("inf")
        self.y_max = float("-inf")


def _xy_array(xy):
    if torch.is_tensor(xy):
        return xy.detach().cpu().numpy()
    return np.asarray(xy)


def update_view_limit(view_limit: ViewLimit, xy):
    xy = _xy_array(xy)
    view_limit.x_min = min(view_limit.x_min, float(np.min(xy[:, 0])))
    view_limit.x_max = max(view_limit.x_max, float(np.max(xy[:, 0])))
    view_limit.y_min = min(view_limit.y_min, float(np.min(xy[:, 1])))
    view_limit.y_max = max(view_limit.y_max, float(np.max(xy[:, 1])))
    return view_limit


def set_axis_limit(element, view_limit: ViewLimit, padding: float = 0.15):
    x_span = max(view_limit.x_max - view_limit.x_min, 1e-6)
    y_span = max(view_limit.y_max - view_limit.y_min, 1e-6)
    x_pad = x_span * padding
    y_pad = y_span * padding
    xlim = (view_limit.x_min - x_pad, view_limit.x_max + x_pad)
    ylim = (view_limit.y_min - y_pad, view_limit.y_max + y_pad)
    if isinstance(element, hv.core.Dimensioned):
        return element.options(xlim=xlim, ylim=ylim)
    element.axis([*xlim, *ylim])
    element.set_aspect("equal")
    return element


def get_frame_bounds(total_frames: int, total_time_points: int) -> List[int]:
    if total_time_points <= 0:
        return []
    return np.linspace(1, total_time_points, num=min(total_frames, total_time_points), dtype=int).tolist()


def visualize_trajectory(
    trajectory,
    label: Optional[str] = None,
    color=None,
    show_steps: bool = False,
    path_params: Optional[dict] = None,
    path_opts: Optional[dict] = None,
    point_params: Optional[dict] = None,
    point_opts: Optional[dict] = None,
):
    trajectory = _xy_array(trajectory)[:, :2]
    path_params = dict(path_params or {})
    path_opts = dict(path_opts or {})
    point_params = dict(point_params or {})
    point_opts = dict(point_opts or {})
    path_opts.setdefault("data_aspect", 1)
    if label:
        path_params["label"] = label
    if color:
        path_opts["color"] = color
        point_opts["color"] = color
    path = hv.Path(trajectory, **path_params).options(show_legend=bool(label), **path_opts)
    if show_steps:
        return path * hv.Points(trajectory, **point_params).options(**point_opts)
    return path


def visualize_trajectory_group(
    trajectories,
    labels: Optional[List[str]] = None,
    colors: Optional[List] = None,
    group: Optional[str] = None,
    label: Optional[str] = None,
    trajectory_params: Optional[dict] = None,
    overlay_params: Optional[dict] = None,
):
    trajectory_params = dict(trajectory_params or {})
    overlay_params = dict(overlay_params or {})
    items = []
    for index, trajectory in enumerate(trajectories):
        params = dict(trajectory_params)
        if labels:
            params["label"] = labels[index]
        if colors:
            params["color"] = colors[index]
        items.append(visualize_trajectory(trajectory, **params))
    return hv.Overlay(items, group, label, **overlay_params)


def visualize_trajectory_group_heading(
    trajectories,
    labels: Optional[List[str]] = None,
    colors: Optional[List] = None,
    group: Optional[str] = None,
    label: Optional[str] = None,
    trajectory_params: Optional[dict] = None,
    overlay_params: Optional[dict] = None,
):
    trajectory_params = dict(trajectory_params or {})
    overlay_params = dict(overlay_params or {})
    items = []
    for index, trajectory in enumerate(trajectories):
        xy = _xy_array(trajectory)[:, :2]
        if len(xy) < 2:
            continue
        params = dict(trajectory_params)
        color = params.pop("color", None)
        if labels:
            params["label"] = labels[index]
        if colors:
            color = colors[index]
        path_opts = dict(params.pop("path_opts", {}))
        point_opts = dict(params.pop("point_opts", {}))
        if color:
            path_opts["color"] = color
            point_opts["color"] = color
        start, end = xy[-2], xy[-1]
        items.append(hv.Path(np.stack([start, end])).options(**path_opts))
        items.append(hv.Points(end[None, :]).options(marker=">", show_legend=False, **point_opts))
    return hv.Overlay(items, group, label, **overlay_params)


def visualize_point_map(
    point_map,
    labels: Optional[List[str]] = None,
    color: str = "b",
    marker: str = "o",
    size: float = 50,
    point_params: Optional[dict] = None,
    overlay_params: Optional[dict] = None,
):
    point_map = _xy_array(point_map)[:, :2]
    point_params = dict(point_params or {})
    overlay_params = dict(overlay_params or {})
    plot = hv.Points(point_map).options(color=color, marker=marker, s=size, **point_params)
    return hv.Overlay([plot], **overlay_params)


def render_visualizations(
    data,
    data_visualizations: Optional[Iterable[Tuple[Callable, dict]]] = None,
    independent_visualizations: Optional[Iterable[hv.core.ViewableElement]] = None,
    matplot_visualizations: Optional[Iterable[Tuple[Callable, dict]]] = None,
    overlay_params: Optional[dict] = None,
    overlay_options: Optional[dict] = None,
    render_options: Optional[dict] = None,
    finalizers: Optional[Iterable[Callable]] = None,
):
    data_visualizations = list(data_visualizations or [])
    independent_visualizations = list(independent_visualizations or [])
    matplot_visualizations = list(matplot_visualizations or [])
    overlay_params = dict(overlay_params or {})
    overlay_options = dict(overlay_options or {})
    render_options = dict(render_options or {})
    overlay_options.setdefault("legend_position", "best")

    previous_backend = hv.Store.current_backend
    hv.Store.set_current_backend("matplotlib")
    overlay = hv.Overlay(
        [*(viz(data, **params) for viz, params in data_visualizations), *independent_visualizations],
        **overlay_params,
    ).options(**overlay_options, backend="matplotlib")
    overlay = overlay.collate().options(**overlay_options)
    if finalizers:
        overlay = reduce(lambda current, finalizer: finalizer(current), finalizers, overlay)

    renderer = hv.renderer("matplotlib")
    plot = renderer.get_plot(overlay)
    fig = plot.state
    for matplot_viz, params in matplot_visualizations:
        matplot_viz(fig, data, **params)

    if "fmt" in render_options:
        render_options.setdefault("dpi", 150)
        filename = render_options.pop("filename", None)
        if filename:
            hv.save(overlay, filename, **render_options)
            result = overlay, None
        else:
            result = overlay, fig
    else:
        result = overlay, fig

    hv.Store.set_current_backend(previous_backend)
    return result
