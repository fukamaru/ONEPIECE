import os
import json
import datetime as dt

from utils.data_utils import load_mobility_data
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

if __name__ == "__main__":

    trajs, _ = load_mobility_data(
        filepath=FILE_PATH,
        feature_only=True
    )

    print(trajs[0])

    print(f"Start: {dt.datetime.now()}")

    image = render_vector_map(
        features=trajs[1:2],
        size=(512, 512),
        style=STYLE,
        supersample=2
    )
    print(f"End: {dt.datetime.now()}")
    image.show()
