import os
import math
import numpy

from utils.map_engine.tiles import BBox, MapTile, MapTileBatch, MapTileSystem


if __name__ == "__main__":
    tile_system = MapTileSystem(
        bbox=BBox(
            min_lon=103.26794370017268, 
            min_lat=30.293550975380697, 
            max_lon=104.60664682423808, 
            max_lat=31.03470084901879
        ),
        max_zoom=12,
        min_tile_size=1000.0
    )

    info = tile_system.zoom_level_info()

    for z, n, count, size in zip(
        info["zoom"],
        info["tiles_per_axis"],
        info["tile_count"],
        info["tile_size"]
    ):
        print(
            f"z={z}:"
            f"{n} x {n},"
            f"{count} tiles,"
            f"tile_size={size:.2f} m"
        )

    # test_bbox = BBox(min_lon=136.70537864, min_lat=34.889320791, max_lon=139.81287319, max_lat=36.04896637)
    # test_bbox = BBox(min_lon=136.70537864, min_lat=34.869052, max_lon=136.983356, max_lat=34.889320791)
    test_bbox = BBox(min_lon=103.954648, min_lat=30.573041, max_lon= 104.135394, max_lat=30.639552)
    found_tile = tile_system.query_single(bbox=test_bbox)
    print(
        f"Found Tile ({found_tile.z}/{found_tile.x}/{found_tile.y}): "
        f"QuadKey {found_tile.quadkey}, "
        f"bbox: {found_tile.bbox.min_lat} -> {found_tile.bbox.max_lat}; {found_tile.bbox.min_lon} -> {found_tile.bbox.max_lon}"
    )
