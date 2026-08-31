"""
                     INPUT
                       │
    ┌──────────────────┼──────────────────┐
    │                  │                  │
ndarray             GeoJSON          GeoPandas
    │                  │                  │
    └────────── GeometryAdapter ──────────┘
                       │
                       ▼
            Source Coordinate
              float64 vertices
                       │
                       ▼
          source_bounds mapping
                       │
                       ▼
          ┌────────────────────┐
          │  TILE COORDINATE   │
          │                    │
          │ EXTENT = 4096      │
          │                    │
          │ [0,4096]²          │
          └────────────────────┘
                       │
          ┌────────────┼─────────────┐
          │            │             │
          ▼            ▼             ▼
       clipping    simplification  quantization
          │            │             │
          └────────────┼─────────────┘
                       │
                       ▼
              GeometryBatch
                vertices:int
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
      tessellation              vertices
           │                       │
           ▼                       ▼
       triangles               markers
           │
           └───────────┬───────────┘
                       ▼
              viewport transform

          pixel = tile / extent * size

                       │
                       ▼
                Rasterization
"""
from __future__ import annotations

import math
import copy
import numpy as np

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union, Literal

from tiles import BBox, MapTile, MapTileSystem


EXTENT = 4096
TILE_PX = 512
UNITS_PER_PX = EXTENT / TILE_PX
WEB_MERCATOR_MAX_LAT = 84.0511287798066

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]
GeoJSONLike = Mapping[str, Any]

# =========================================================================
# Coordination Conversion
# Geographical Coordination <-> Tile Coordination
# =========================================================================

def _as_xy_array(
        lonlat_coords: ArrayLike,
        copy_array: bool = False
) -> np.ndarray:
    arr = (
        np.array(lonlat_coords, dtype=np.float64, copy=True)
        if copy_array
        else np.asarray(lonlat_coords, dtype=np.float64)
    )
    if arr.ndim == 0 or arr.shape[-1] < 2:
        raise ValueError("Wrong shape! Make sure that at least two dimension.")
    return arr[..., :2]


def lonlat_to_web_mercator_normalized(
        lonlat_coords: ArrayLike
) -> np.ndarray:
    arr = _as_xy_array(lonlat_coords)

    lon = arr[..., 0]
    lat = np.clip(
        arr[..., 1],
        - WEB_MERCATOR_MAX_LAT,
        WEB_MERCATOR_MAX_LAT
    )

    x = (lon + 180.0) / 360.0
    lat_rad = np.deg2rad(lat)
    y = (1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi) * 0.5

    return np.stack((x, y), axis=-1)


def web_mercator_normalized_to_lonlat(
        coords: ArrayLike
) -> np.ndarray:
    arr = _as_xy_array(coords)
    x = arr[..., 0]
    y = arr[..., 1]
    
    lon = x * 360.0 - 180.0
    lat = np.rad2deg(
        np.arctan(
            np.sinh(
                np.pi * (1.0 - 2.0 * y)
            )
        )
    )
    
    return np.stack((lon, lat), axis=-1)


@dataclass(frozen=True)
class TileTransform:

    width: int = 256
    height: int = 256
    mode: Literal['xyz', 'bbox'] = 'bbox'
    # for bbox
    bbox: BBox | Tuple[float, float, float, float] | None = None
    # for standard tiles xyz
    x: int | None = None
    y: int | None = None
    z: int | None = None

    @classmethod
    def from_xyz(
        cls,
        z: int,
        x: int,
        y: int,
        width: int = 256,
        height: int = 256
    ) -> TileTransform:
        if z < 0:
            raise ValueError(f"Zoom must be non-negative but {z} received")

        n = 1 << z
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"invalid tile index {z}/{x}/{y} under zoom={z}")

        if width <= 0 or height <= 0:
            raise ValueError(f"both width and height configuration must be positive")

        return cls(width=width, height=height, mode='xyz', z=z, x=x, y=y)

    @classmethod
    def from_lonlat_bbox(
        cls, 
        bbox: BBox | Tuple[float, float, float, float],
        width: int = 256,
        height: int = 256
    ) -> TileTransform:
        
        if width <= 0 or height <= 0:
                raise ValueError(f"both width and height configuration must be positive")
        
        return cls(width=width, height=height, mode='bbox', bbox=bbox)

    def lonlat_to_pixels(
            self,
            lonlat_coords: ArrayLike
    ) -> np.ndarray:

        arr = _as_xy_array(lonlat_coords)

        if self.mode == "xyz":
            assert (
                self.z is not None
                and self.x is not None
                and self.y is not None
            )

            world = lonlat_to_web_mercator_normalized(arr)
            n = float(1 << self.z)

            tile_x = world[..., 0] * n
            tile_y = world[..., 1] * n

            px = (tile_x - self.x) * self.width
            py = (tile_y - self.y) * self.height

        if self.mode == 'bbox':
            assert self.bbox is not None

            min_lon, min_lat, max_lon, max_lat = self.bbox.as_tuple() if isinstance(self.bbox, BBox) else self.bbox
            px = (arr[..., 0] - min_lon) / (max_lon - min_lon) * self.width
            py = (max_lat -  arr[..., 1]) / (max_lat -  min_lat) * self.height

        return np.stack((px, py), axis=-1)

    def tile_bbox_lonlat(
            self
    ) -> Tuple[float, float, float, float]:
        if self.mode == 'bbox':
            assert self.bbox is not None
            return self.bbox if isinstance(self.bbox, Tuple) else self.bbox.as_tuple()

        assert (
            self.z is not None
            and self.x is not None
            and self.y is not None
        )

        n = float(1 << self.z)
        tl = np.array([self.x / n, self.y / n], dtype=np.float64)
        br = np.array([(self.x + 1) / n, (self.y + 1) / n], dtype=np.float64)
        lonlat = web_mercator_normalized_to_lonlat(np.stack((tl, br)))
        min_lon, max_lat = lonlat[0]
        max_lon, min_lat = lonlat[1]

        return float(min_lon), float(min_lat), float(max_lon), float(max_lat)


# =========================================================================
# Processing before sending into render
# =========================================================================

def _point_segment_distance(
        points: np.ndarray,
        o: np.ndarray,
        d: np.ndarray
) -> np.ndarray:
    od = d - o
    denom = float(np.dot(od, od))

    if denom <= np.finfo(np.float64).eps:
        diff = points - o
        return np.einsum("ij,ij->i", diff, diff)

    op = points - o
    t = np.clip(
        (op @ od) / denom,
        0.0, 1.0
    )
    proj = o + t[:, None] * od
    diff = points - proj

    return np.einsum("ij,ij->i", diff, diff)

def douglas_peucker(
        points: ArrayLike,
        tolerance: float
):
    pts = _as_xy_array(points)

    if pts.ndim != 2:
        raise ValueError(f"Douglas-Peucker requires np.ndarray with shape [N,2] but {pts.shape} received")

    n = len(pts)
    if n <= 2 or tolerance <= 0:
        return pts.copy()

    tol_sq = float(tolerance) ** 2
    keep = np.zeros(n, dtype=bool)
    keep[0] = True
    keep[-1] = True

    stack: List[Tuple[int, int]] = [(0, n - 1)]
    while stack:
        start, end = stack.pop()
        if end <= start + 1:
            continue

        interior = pts[start + 1 : end]
        d2 = _point_segment_distance(
            interior, pts[start], pts[end]
        )
        local_idx = int(np.argmax(d2))
        max_d2 = float(d2[local_idx])

        if max_d2 > tol_sq:
            idx = start + 1 + local_idx
            keep[idx] = True
            stack.append((start, end))
            stack.append((idx, end))

    return pts[keep]

def simplify_ring(
        ring: ArrayLike,
        tolerance: float
) -> np.ndarray:
    
    pts = _as_xy_array(ring)

    if pts.ndim != 2:
        raise ValueError(f"ring simplification requires np.ndarray with shape [N,2] but {pts.shape} received")
    if len(pts) < 4 or tolerance <= 0:
        if len(pts) and not np.allclose(pts[0], pts[1]):
            return np.vstack((pts, pts[0]))
        return pts.copy()

    # remove duplicated close points
    core = pts[:-1] if np.allclose(pts[0], pts[1]) else pts
    if len(core) < 3:
        return np.vstack((core, core[0])) if len(core) else core.copy()

    d2 = np.einsum("ij,ij->i", core - core[0], core - core[0])
    split = int(np.argmax(d2))
    if split == 0:
        return np.vstack((core, core[0]))

    part_a = core[:split+1]
    part_b = np.vstack((core[split:], core[:1]))

    simp_a = douglas_peucker(part_a, tolerance)
    simp_b = douglas_peucker(part_b, tolerance)

    merged = np.vstack((simp_a[:-1], simp_b[:-1]))
    if len(merged) < 3:
        merged = core

    return np.vstack((merged, merged[0]))


# =========================================================================
# Geometry Adaptor
# =========================================================================

def _is_geojson_mapping(
        obj: Any
) -> bool:
    return isinstance(obj, Mapping) and isinstance(obj.get("type"), str)


def _geometry_from_feature_like(
        obj: GeoJSONLike
) -> GeoJSONLike | None:
    typ = obj.get("type")

    if typ == "Feature":
        geom = obj.get("geometry")
        if geom is None:
            return None
        if not isinstance(geom, Mapping):
            raise TypeError("Feature.geometry must be mapping or None")
        return geom

    if typ == "FeatureCollection":
        return None

    return obj


def _iter_geometry_coordinate_arrays(
        geometry: GeoJSONLike
) -> Iterable[np.ndarray]:

    typ = geometry.get("type")
    lonlat_coords = geometry.get("coordinates")

    if typ == "Point":
        yield _as_xy_array(lonlat_coords).reshape(1, 2)
    elif typ == "MultiPoint":
        yield _as_xy_array(lonlat_coords).reshape(-1, 2)
    elif typ == "LineString":
        yield _as_xy_array(lonlat_coords).reshape(-1, 2)
    elif typ == "MultiLineString":
        for line in lonlat_coords:
            yield _as_xy_array(line).reshape(-1, 2)
    elif typ == "Polygon":
        for ring in lonlat_coords:
            yield _as_xy_array(ring).reshape(-1, 2)
    elif typ == "MultiPolygon":
        for polygon in lonlat_coords:
            for ring in polygon:
                yield _as_xy_array(ring).reshape()
    elif typ == "GeometryCollection":
        for geom in geometry.get("geometries", []):
            yield from _iter_geometry_coordinate_arrays(geom)
    else:
        raise TypeError(f"Unsupported GeoJSON geometry type: {typ!r}")
