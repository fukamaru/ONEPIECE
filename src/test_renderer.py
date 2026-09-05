import os
import json
import numpy as np
import pandas as pd
import datetime as dt
import geopandas as gpd
from shapely.geometry import Point, LineString

from utils.data_utils import load_mobility_data
from map_engine.process import _lonlat_to_tile_custom
from map_engine.tile_system import XYZTileSystem
from map_engine.vector_render import VectorRenderer, Layer, LineStyle
from map_engine.renderer import render_vector_map

FILE_PATH = "/Users/linlifeng/Downloads/mobility_data/DiDi-chengdu-simplified-tiled.geojson"

STYLE = {
    "background": "#f4f3ef",
    "layers": [
        {"type":"fill", "color":"#f0bf5b", "opacity":1.0}, 
        {"type":"line", "color":"#f0bf5b", "width":4.0, "join":"round", "cap":"round"},
        {"type":"circle", "color":"#f0bf5b", "radius": 4.0}
    ]
}

def process_hqbw(filepath):
    raw = pd.read_csv(
        filepath, 
        usecols=[2, 4, 5, 6, 7]
    )
    raw["time"] = pd.to_datetime(raw["time"], format='%Y-%m-%d %H:%M')

    grouped = raw.groupby("segment_id", sort=True)

    result = grouped.agg(
        start_time=("time", "first"),
        end_time=("time", "last"),
        behavior_type=("trans_mode2", "first"),
        last_x=("lon", "last"),
        last_y=("lat", "last")
    ).reset_index()

    coords = grouped.apply(
        lambda g: list(zip(g["lon"], g["lat"])),
    )

    result["geometry"] = [
        Point(x, y) if btype == "STAY" else LineString(coords.loc[bid])
        for bid, btype, x, y in zip(
            result["segment_id"],
            result["behavior_type"],
            result["last_x"],
            result["last_y"]
        )
    ]

    result = result.drop(columns=["last_x", "last_y"])

    result = gpd.GeoDataFrame(result, geometry="geometry", crs="EPSG:4326")

    return result


def main() -> None:
    data, _ = load_mobility_data("/Users/linlifeng/Downloads/mobility_data/HQBW_UID569.geojson", feature_only=True)

    tile_system = XYZTileSystem()
    local_tile = tile_system.query_single_loose(
        [136.97375012, 34.8390457, 136.9986905, 34.87650103],
        loose_factor=1.8
    )
    local_bounds = local_tile.loose_bbox.as_tuple()

    print(
        f"z: {local_tile.z} | "
        f"x: {local_tile.x} | "
        f"y: {local_tile.y} | "
        f"quadkey: {local_tile.quadkey} | "
        f"bounds: {local_tile.loose_bbox.as_tuple()}"
    )

    oneday = data[:8]
    for traj in oneday:
        if traj["geometry"]["type"] == "Point":
            traj["geometry"]["coordinates"] = tuple(
                _lonlat_to_tile_custom(
                    np.array(traj["geometry"]["coordinates"], dtype=np.float64),
                    bboxs=local_bounds
                )
            )
        else:
            traj["geometry"]["coordinates"] = tuple(
                map(
                    tuple, 
                    _lonlat_to_tile_custom(
                        lonlat=traj["geometry"]["coordinates"], 
                        bboxs=local_bounds
                    )
                )
            )

    # print(oneday)

    renderer = VectorRenderer(
        width=512,
        height=512,
        background="white",
        antialias=4
    )

    renderer.add_source("one-day-traj", oneday)
    # Render ICON in the stay point
    # renderer.add_layer(
    #     Layer(
    #         id="start-point", 
    #         type="icon", 
    #         source="one-day-traj",
    #         filter=lambda p: p.get("segment_id") == 1,
    #         paint={
    #             "icon": "/Users/linlifeng/Downloads/star.png",
    #             "size": (16.0, 16.0),
    #             "anchor": "center"
    #         }
    #     )
    # )
    renderer.add_layer(
        Layer(
            id="move",
            type="line",
            source="one-day-traj",
            filter=lambda p: p.get("behavior_type") != "STAY",
            paint=LineStyle(
                color="#2222cc",
                width=4,
                join="miter",
                cap="round"
            )
        )
    )

    # Render ICON in the stay point
    renderer.add_layer(
        Layer(
            id="start-point", 
            type="icon", 
            source="one-day-traj",
            filter=lambda p: p.get("behavior_type") == "STAY",
            paint={
                "icon": "/Users/linlifeng/Downloads/star.png",
                "size": (16.0, 16.0),
                "anchor": "center"
            }
        )
    )

    image = renderer.render()
    image.save("/Users/linlifeng/Downloads/HQBW_UID569_20230101.png", optimize=True)

if __name__ == "__main__":

    main()

    # trajs, _ = load_mobility_data(
    #     filepath=FILE_PATH,
    #     feature_only=True
    # )

    # print(trajs[0])

    # print(f"Start: {dt.datetime.now()}")

    # image = render_vector_map(
    #     features=trajs[1:2],
    #     size=(512, 512),
    #     style=STYLE,
    #     supersample=2
    # )
    # print(f"End: {dt.datetime.now()}")
    # image.show()
