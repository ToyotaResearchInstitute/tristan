#!/usr/bin/env python
# Copyright 2020 Toyota Research Institute. All rights reserved.
"""
This file allows to rasterize lanes and point positions for CNN ingest of processed perception
Taken from trams repo (i.e, the source is there, for any major fixed, please update both)
"""
import io

import matplotlib
from PIL import Image

matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np


class InvalidLaneError(Exception):
    def __init__(self, msg=""):
        self.msg = msg


class Rasterize:
    def __init__(self, scale, halfwidth, halfheight, ego_position, ego_angle, maptype="traverse"):
        """

        :param scale: The resolution of the raster map. (how many pixels)
        :param halfwidth: The physical half width of the map. (how many meters)
        :param halfheight: The physical half width of the map.
        :param ego_position: Position of the center frame of the map.
        :param ego_angle: Angle of the center frame of the map.
        :param maptype: can be 'traverse' or 'boundaries'
        """
        self.scale = scale
        self.halfwidth = halfwidth
        self.halfheight = halfheight
        self.ego_position = ego_position
        self.ego_angle = ego_angle
        self.maptype = maptype
        alpha = self.ego_angle
        self.rotation_matrix = np.array([[np.cos(alpha), np.sin(alpha)], [-np.sin(alpha), np.cos(alpha)]])
        self.rotation_matrix_T = self.rotation_matrix.T
        self.translation_vector = np.dot(self.rotation_matrix_T, -np.array(self.ego_position))
        self.dpi = 100
        self.lanes = 0

    def start_raster(self):
        # self.fig = plt.figure(frameon=False,facecolor='0.0')
        self.fig = plt.figure(
            frameon=False,
            figsize=[self.halfwidth * 2 / self.scale / 100 / 0.8, self.halfheight * 2 / self.scale / 100 / 0.8],
            dpi=100,
            facecolor="0.0",
        )
        self.ax = self.fig.gca()
        self.ax.set_facecolor((0.0, 0.0, 0.0))
        self.ax.set_ylim(-self.halfheight, self.halfheight)
        self.ax.set_xlim(-self.halfwidth, self.halfwidth)
        # plt.ylim(-self.halfheight,self.halfheight)
        # plt.xlim(-self.halfwidth, self.halfwidth)
        # self.ax = self.fig.add_axes([0, 0, 1, 1])

        # set axis

    def add_lane(self, lane, value=1.0, tl_state_color="r"):
        """

        :param lane: either an object with get_corners to return a list of x,y values, or a list of x,y values
        :param value: color for the resulting image
        :param tl_state_color: for traffic list states
        :return:
        """
        # transform
        if type(lane) is list:
            corners = lane
        else:
            corners = lane.get_corners()
        # polygon=lane.get_polygon()
        self.lanes += 1
        try:
            # adapt to sumo lane features
            transformed = [np.dot(self.rotation_matrix_T, np.array(c)) + self.translation_vector for c in corners]
            x = [c[0] for c in transformed]
            y = [c[1] for c in transformed]
            # x,y=polygon.exterior.xy
            if type(value) == list or type(value) == tuple:
                color3 = value
            else:
                color3 = (value, value, value)
            if self.maptype == "traverse":
                self.ax.fill(x, y, color=color3)
                # plt.fill(x,y,color=(value,value,value))
            elif self.maptype == "boundary":
                self.ax.plot(x, y, color=color3, linewidth=1)
            elif self.maptype == "tl_state":
                self.ax.fill(x, y, color=tl_state_color)
            else:
                raise InvalidRasterError("wrong maptype")
        except:
            pass

    def add_point(self, c, value=1.0):
        # transform
        try:
            transformed = np.dot(self.rotation_matrix_T, np.array(c)) + self.translation_vector
            x = transformed[0]
            y = transformed[1]
            self.ax.plot(x, y, "o", color=(value, value, value), linewidth=1, markersize=2, fillstyle="full")

        except:
            pass

    def get_raster(self):
        buf = io.BytesIO()
        self.ax.set_ylim(-self.halfheight, self.halfheight)
        self.ax.set_xlim(-self.halfwidth, self.halfwidth)
        self.ax.axis("off")
        self.ax.get_xaxis().set_ticks([])
        self.ax.get_yaxis().set_ticks([])
        self.ax.get_xaxis().tick_bottom()
        self.ax.axes.get_yaxis().set_visible(False)
        self.fig.savefig(
            buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=self.fig.get_facecolor(), edgecolor="none"
        )
        buf.seek(0)
        im = Image.open(buf)
        # arr = np.fromstring(buf.getvalue(), dtype=np.uint8)
        arr = np.array(im.getdata()).reshape(im.size[1], im.size[0], 4)
        buf.close()
        plt.close(self.fig)
        result = arr[:, :, :3]
        if (result.shape[0]) < self.halfheight * 2:
            result = np.concatenate([np.expand_dims(result[0, :, :], axis=0), result], axis=0)
        if (result.shape[1]) < self.halfwidth * 2:
            result = np.concatenate([np.expand_dims(result[:, 0, :], axis=1), result], axis=1)
        return result
