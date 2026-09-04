import os
import json
import numpy as np

from utils.data_utils import load_mobility_data
from map_engine.tile_system import BaseTileSystem, XYZTileSystem
# from utils.map_engine.tiles import MapTile, MapTileBatch, MapTileSystem
from utils.map_engine.process import _iter_geometry_coordinate_arrays, _lonlat_to_tile_custom

FILE_PATH = "/Users/linlifeng/Downloads/mobility_data/DiDi-chengdu-simplified.geojson"

CHENGDU_METADATA = {
    "dataset-name": "DiDi-Chengdu", 
    "dataset-version": "1.1.0", 
    "geographical-range": 
        {
            "min_lon": 103.26794370017268, 
            "min_lat": 30.293550975380697, 
            "max_lon": 104.60664682423808, 
            "max_lat": 31.03470084901879
        }, 
    "tile-system": 
        {
            "type": "custom", 
            "max_zoom": 10, 
            "min_tile_size": 1000.0
        },
}

if __name__ == "__main__":
    # Load Data
    data, header = load_mobility_data(FILE_PATH, feature_only=False)
    print(header)

    tile_system = XYZTileSystem()

    # Build Tile System
    # tile_system = MapTileSystem(
    #     bbox=tuple(CHENGDU_METADATA["geographical-range"].values()),
    #     max_zoom=CHENGDU_METADATA["tile-system"]["max_zoom"],
    #     min_tile_size=CHENGDU_METADATA["tile-system"]["min_tile_size"]
    # )

    # bounds of every trajectory
    bboxs = np.array(
        [
            [
                traj["properties"]["min_lon"], traj["properties"]["min_lat"],
                traj["properties"]["max_lon"], traj["properties"]["max_lat"]
            ]
            for traj in data
        ],
        dtype=np.float64
    )

    # the located tile of every trajectory
    local_tiles = tile_system.query_single_loose(
        bbox=bboxs,
        loose_factor=1.5
    )
    local_tiles = [
        {
            "quadkey": str(t.quadkey),
            "bbox": t.loose_bbox.as_tuple(),
            "z": t.z,
            "x": t.x,
            "y": t.y,
            "coord_system": "geographic"
        }
        for t in local_tiles
    ]

    for traj, tile in zip(data, local_tiles):
        traj["properties"].update(
            {
                "quadkey": tile["quadkey"],
                "bbox": tile["bbox"],
                "z": tile["z"],
                "x": tile["x"],
                "y": tile["y"]
            }
        )

    # print(data[3828])

    tiled_data = data.copy()
    for traj in tiled_data:
        traj["geometry"]["coordinates"] = tuple(
            map(
                tuple, 
                _lonlat_to_tile_custom(
                    lonlat=traj["geometry"]["coordinates"], 
                    bboxs=traj["properties"]["bbox"]
                )
            )
        )
        traj["properties"]["coord_system"] = "tile"

    print(len(tiled_data))
    print(tiled_data[1])

    all_tiles = [
        traj["properties"]["quadkey"]
        for traj in tiled_data
    ]
    print(os.path.commonprefix(all_tiles))

    max_tile_zoom = min(map(len, all_tiles))
    print([i for i, t in enumerate(all_tiles) if len(t) == max_tile_zoom])

    # output = header.copy()
    # output["name"] = "DiDi-chengdu-simplified-tiled"
    # output["features"] = tiled_data

    # with open("/Users/linlifeng/Downloads/mobility_data/DiDi-chengdu-simplified-tiled.geojson", "w") as f:
    #     json.dump(output, f, ensure_ascii=False)
