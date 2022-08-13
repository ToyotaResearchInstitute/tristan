"""End2End helper"""
import io
import json
import math
import os
import sqlite3
import subprocess
from pprint import pprint
from urllib.request import urlopen

import utm
from scipy.spatial import KDTree

# For OpenStreetMap intersections
close_enough = 500


def get_overpass(easting, northing, zone_number, zone_letter):

    min_lat, min_lon = utm.to_latlon(easting - close_enough, northing - close_enough, zone_number, zone_letter)
    max_lat, max_lon = utm.to_latlon(easting + close_enough, northing + close_enough, zone_number, zone_letter)

    overpass_bbox = "{},{},{},{}".format(min_lat, min_lon, max_lat, max_lon)
    overpass_query = str.join(
        "",
        [
            "[bbox:{}];".format(overpass_bbox),
            "(way[lanes][highway!=footway];way[highway][highway!=footway];);",
            "(._;>;);",
            "out;",
        ],
    )
    overpass_url = "http://overpass-api.de/api/interpreter?data={}".format(overpass_query)
    print("Querying Overpass!", overpass_url)
    with urlopen(overpass_url) as overpass_response:
        osm_xml = overpass_response.read()
        return osm_xml


class IntentOSM:
    """Load OpenStreetMap data"""

    def __init__(self, osm_dir=None):
        if isinstance(osm_dir, str) and os.path.isdir(osm_dir):
            self.osm_dir = osm_dir
        else:
            self.osm_dir = os.path.join(os.getenv("HOME"), "intent", "osm_cache")

        # Make sure this directory exists
        os.makedirs(self.osm_dir, exist_ok=True)

        # Make sure the database exists
        fname_db = os.path.join(self.osm_dir, "index.db")
        conn_db = sqlite3.connect(fname_db)
        c = conn_db.cursor()
        c.execute(
            "CREATE TABLE IF NOT EXISTS centroids (osm_id varchar(16) primary key, easting real, northing real, zone_number integer, zone_letter text)"
        )
        conn_db.commit()
        # conn.close()
        self.conn_db = conn_db

        self.osm_intersections = {}
        self.osm_inter_kd = {}
        self.osm_segments = {}

    def load_all(self):
        c = self.conn_db.cursor()
        c.execute("SELECT * FROM centroids")
        for osm_id, easting, northing, zone_number, zone_letter in c:
            # Open the osm_id
            centroid = (easting, northing, zone_number, zone_letter)
            self.load(osm_id, centroid)

    def load(self, osm_id, centroid):
        if osm_id in self.osm_intersections:
            return
        inter_kd = []
        intersections = {}
        segments = {}
        fname = os.path.join(self.osm_dir, osm_id + ".osm")
        proc = subprocess.Popen(["luajit", "roads.lua", fname], stdout=subprocess.PIPE)
        for line in proc.stdout.readlines():
            obj = json.loads(line)
            ref, props = list(obj.items())[0]
            if ref[0].isupper():
                segments[ref] = props
            else:
                latlon = props["latlon"]
                easting, northing, zone_number, zone_letter = utm.from_latlon(latlon[0], latlon[1])
                inter_kd.append((easting, northing))
                intersections[ref] = props
        self.osm_intersections[osm_id] = intersections
        self.osm_inter_kd[osm_id] = KDTree(inter_kd)
        self.osm_segments[osm_id] = segments

    def latlon_to_cache(self, lat0, lon0):
        easting0, northing0, zone_number0, zone_letter0 = utm.from_latlon(lat0, lon0)
        # Load the points
        c = self.conn_db.cursor()
        query = "SELECT * FROM centroids WHERE zone_number={} AND zone_letter='{}'".format(zone_number0, zone_letter0)
        # print("query", query)
        c.execute(query)
        min_id = None
        min_dist = math.inf
        for osm_id, easting, northing, zone_number, zone_letter in c:
            dist = math.hypot(easting - easting0, northing - northing0)
            # print(osm_id, dist)
            if dist <= (close_enough / 2):
                min_id = osm_id
                min_dist = dist
                break
            elif dist < min_dist:
                min_id = osm_id
                min_dist = dist

        # More than a kilometer away?
        # print(min_id, min_dist)
        if min_dist > close_enough:
            # print("Adding lat/lon", lat0, lon0)
            osm_id0 = "{:d}_{:d}".format(int(lat0 * 1e4), int(lon0 * 1e4))
            # print("osm_id0", osm_id0)
            min_id = osm_id0
            xml0 = get_overpass(easting0, northing0, zone_number0, zone_letter0)
            fname = os.path.join(self.osm_dir, osm_id0 + ".osm")
            f_osm = io.open(fname, "wb+")
            f_osm.write(xml0)
            f_osm.close()
            # save the file
            c.execute(
                "insert or replace into centroids values (?, ?, ?, ?, ?)",
                [osm_id0, easting0, northing0, zone_number0, zone_letter0],
            )
            self.conn_db.commit()
            # TODO: Execute the bounds.lua command

        return min_id, (easting0, northing0, zone_number0, zone_letter0)

    def query_intersection(self, latlon, threshold=20):
        # Check if this coordinate is near an intersection
        if len(latlon) > 2:
            latlon = latlon[:2]

        osm_id, utm_coord = self.latlon_to_cache(*latlon)
        self.load(osm_id, utm_coord)
        dist, intersection_id = self.osm_inter_kd[osm_id].query(utm_coord[0:2])
        # print("Queried KDTree:", dist)
        # If within 20 meters...
        if dist > threshold:
            return False, osm_id
        # TODO: Get some data about the point?
        # ref, props = list(self.osm_intersections[osm_id].items())[intersection_id]
        # print("Close to intersection", ref)
        # pprint(props)
        return True, osm_id

    # Compute distance to intersection
    def dist_to_intersection(self, latlon):
        osm_id, utm_coord = self.latlon_to_cache(latlon[0], latlon[1])
        self.load(osm_id, utm_coord)
        dist, intersection_id = self.osm_inter_kd[osm_id].query(utm_coord[0:2])

        return dist
