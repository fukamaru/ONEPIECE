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

from .tiles import BBox, MapTile, MapTileSystem


EXTENT = 4096
TILE_PX = 512
UNITS_PER_PX = EXTENT / TILE_PX
WEB_MERCATOR_MAX_LAT = 84.0511287798066

ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]
GeoJSONLike = Mapping[str, Any]

# =========================================================================
# Helper
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


def _broadcast(value, shape, name: str, dtype):
    arr = np.asarray(value, dtype=dtype)
    try:
        return np.broadcast_to(arr, shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {arr.shape} cannot broadcast to coordinate shape {shape}"
        ) from exc


def _broadcast_bounds(bounds, shape) -> np.ndarray:
    arr = np.asarray(bounds, dtype=np.float64)

    if arr.shape[-1:] != (4,):
        raise ValueError(
            f"bounds must have shape (..., 4) but {arr.shape} received"
        )

    target_shape = shape + (4,)

    try:
        return np.broadcast_to(arr, target_shape)
    except ValueError as exc:
        raise ValueError(
            f"bounds with shape {arr.shape} cannot broadcast to {target_shape}"
        ) from exc

# =========================================================================
# Geographical Coordinates <-> Standard XYZ Tile Coordinates
# =========================================================================

def _geographic_to_tile_xyz(
        lonlat: ArrayLike,
        z: Union[int, Sequence[int]],
        x: Union[int, Sequence[int]],
        y: Union[int, Sequence[int]],
        extent: int | float = EXTENT
) -> None:
    lonlat_arr = _as_xy_array(lonlat)
    shape = lonlat_arr.shape[:-1]

    z = _broadcast(z, shape, "z", dtype=np.float64)
    x = _broadcast(x, shape, "x", dtype=np.float64)
    y = _broadcast(y, shape, "y", dtype=np.float64)
    extent = _broadcast(extent, shape, "extent", dtype=np.float64)

    lon = lonlat_arr[..., 0]
    lat = lonlat_arr[..., 1]

    world_size = np.exp2(z)
    # global floating XYZ coordinate at zoom z
    global_x = (lon + 180.0) / 360.0 * world_size
    lat_rad = np.deg2rad(lat)
    global_y = (
        1.0 - np.arcsinh(np.tan(lat_rad)) / np.pi
    ) * 0.5 * world_size

    # global XYZ -> local tile coordinate
    tile_x = (global_x - x) * extent
    tile_y = (global_y - y) * extent

    return np.stack((tile_x, tile_y), axis=-1)


def _geographic_from_tile_xyz(
        tile_xy: ArrayLike,
        z: Union[int, Sequence[int]],
        x: Union[int, Sequence[int]],
        y: Union[int, Sequence[int]],
        extent: int | float = EXTENT
) -> np.ndarray:
    xy_arr = _as_xy_array(tile_xy)
    shape = xy_arr.shape[:-1]

    z = _broadcast(z, shape, "z", dtype=np.float64)
    x = _broadcast(x, shape, "x", dtype=np.float64)
    y = _broadcast(y, shape, "y", dtype=np.float64)
    extent = _broadcast(extent, shape, "extent", dtype=np.float64)

    global_x = x + xy_arr[..., 0] / extent
    global_y = y + xy_arr[..., 1] / extent

    world_size = np.exp2(z)

    # inverse global XYZ x
    lon = global_x / world_size * 360.0 -180.0
    # inverse web-mercator y
    mercator_y = np.pi * (
        1.0 - 2.0 * global_y / world_size
    )
    lat = np.rad2deg(
        np.arctan(
            np.sinh(mercator_y)
        )
    )

    return np.stack((lon, lat), axis=-1)

# =========================================================================
# Geographical Coordinates <-> Custom Tile Coordinates
# =========================================================================

def _lonlat_to_tile_custom(
        lonlat: ArrayLike,
        bboxs: Union[Tuple[float, float, float, float], BBox],
        extent: int | float = EXTENT
) -> np.ndarray:
    '''
    Convert geograpgical coordinates (lon, lat) x N to local tile coordinates in custom tile system.
    '''
    lonlat_arr = _as_xy_array(lonlat)
    # how many coordinates in the batch
    shape = lonlat_arr.shape[:-1]

    bboxs = _broadcast_bounds(bboxs, shape)
    extent = _broadcast(extent, shape, "extent", dtype=np.float64)

    min_x = bboxs[..., 0]
    min_y = bboxs[..., 1]
    max_x = bboxs[..., 2]
    max_y = bboxs[..., 3]

    span_x = max_x - min_x
    span_y = max_y - min_y

    if np.any(span_x <= 0) or np.any(span_y <= 0):
        raise ValueError(
            "invalid bounds: bounds must satisfy max_x > min_x and max_y > min_y"
        )

    tile_x = (
        (lonlat_arr[..., 0] - min_x)
        / span_x * extent
    )
    tile_y = (
        (max_y - lonlat_arr[..., 1])
        / span_y * extent
    )

    return np.stack((tile_x, tile_y), axis=-1)

def _lonlat_from_tile_custom(
        tile_xy: ArrayLike,
        bboxs: Union[Tuple[float, float, float, float], BBox],
        extent: int | float = EXTENT
) -> np.ndarray:
    xy_arr = _as_xy_array(tile_xy)
    shape = xy_arr.shape[:-1]

    bboxs = _broadcast_bounds(bboxs, shape)
    extent = _broadcast(extent, shape, "extent", dtype=np.float64)

    if np.any(extent <= 0):
        raise ValueError(
            f"extent must be positive but {extent} received"
        )

    min_x = bboxs[..., 0]
    min_y = bboxs[..., 1]
    max_x = bboxs[..., 2]
    max_y = bboxs[..., 3]

    span_x = max_x - min_x
    span_y = max_y - min_y

    if np.any(span_x <= 0) or np.any(span_y <= 0):
        raise ValueError(
            "invalid bounds: bounds must satisfy max_x > min_x and max_y > min_y"
        )

    lon = (
        min_x + xy_arr[..., 0] / extent * span_x
    )
    lat = (
        max_y - xy_arr[..., 1] / extent * span_y
    )

    return np.stack((lon, lat), axis=-1)

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


__all__ = [
    "_to_tile_xyz",
    "_from_tile_xyz",

    "_to_tile_custom",
    "_from_tile_custom",

    "douglas_peucker",
    "simplify_tile_geometries"
]
