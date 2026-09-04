"""
A compact CPU vector-map renderer focused on geometry rendering.
It intentionally resembles the conceptual stages of modern vector map engines:
style selection -> geometry tessellation -> triangle rasterization -> blend
"""
from __future__ import annotations

import math
import numpy as np

from PIL import Image
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping, List, Union, Sequence, Literal


ArrayLike = Union[np.ndarray, Sequence[Sequence[float]], Sequence[float]]

# ========================================================================
# Main renderer
# ========================================================================

def render_vector_map(
        features: Sequence[Mapping[str, Any]],
        style: Mapping[str, Any],
        *,
        size: tuple[int, int] = (256, 256),
        extent: int | float = 4096,
        supersample: int = 1
) -> Image.Image:

    width, height = map(int, size)

    raster = _Raster(
        width=width,
        height=height,
        supersample=supersample,
        background=parse_color(style.get("background", "#ffffff"))
    )

    sx = raster.width / float(extent)
    sy = raster.height / float(extent)

    layers = [_prepare_layer(layer) for layer in style.get("layers", [])]

    for layer in layers:
        for feature in features:
            geometry = feature.get("geometry")
            if not geometry:
                continue

            properties = feature.get("properties") or {}
            # if not _matches(properties, layer["filter"]):
            #     continue

            if layer["type"] == "fill":
                _draw_fill(raster=raster, geometry=geometry, layer=layer, sx=sx, sy=sy)

            elif layer["type"] == "line":
                _draw_line(raster=raster, geometry=geometry, layer=layer, sx=sx, sy=sy)

            elif layer["type"] == "circle":
                _draw_circle(raster=raster, geometry=geometry, layer=layer, sx=sx, sy=sy)

    return raster.image()

# ========================================================================
# Tessellation and Triangle Rasterization
# ========================================================================

def tessellate_line(
        coords: ArrayLike,
        width: float,
        *,
        closed: bool = False,
        join: Literal["round", "bevel", "miter"] = "round",
        cap: Literal["round", "butt"] = "round",
        miter_limit: float = 4.0
) -> np.ndarray:
    
    points = _clean_path(coords, closed)
    width = float(width)

    if width < 0:
        return _empty_triangles()

    if join not in ["round", "bevel", "miter"]:
        raise ValueError(
            f"unsupported join {join}, must be 'round', 'bevel' or 'miter'"
        )
    if cap not in ["round", "butt"]:
        raise ValueError(
            f"unsupported cap {cap}, must be 'round', 'butt'"
        )
    if miter_limit <= 0:
        raise ValueError(
            "miter_limit must be positive"
        )

    if len(points) < (3 if closed else 2):
        return _empty_triangles()

    starts = points if closed else points[:-1]
    ends = np.roll(points, -1, axis=0) if closed else points[1:]

    vectors = ends - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid = lengths > 1e-12

    starts, ends = starts[valid], ends[valid]
    vectors, lengths = vectors[valid], lengths[valid]

    if not len(vectors):
        return _empty_triangles()

    tangents = vectors / lengths[:, None]
    normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
    half = width * 0.5

    l0, r0 = starts + normals * half, starts - normals * half
    l1, r1 = ends + normals * half, ends - normals * half

    pieces = [
        np.concatenate(
            (
                np.stack((l0, r0, l1), axis=1),
                np.stack((r0, r1, l1), axis=1),
            ),
            axis=0
        )
    ]

    joins = []
    if closed:
        join_points = points
        prev_idx = np.roll(np.arange(len(tangents)), 1)
        next_idx = np.arange(len(tangents))
    else:
        join_points = points[1:-1]
        prev_idx = np.arange(len(join_points))
        next_idx = prev_idx + 1

    for point, i0, i1 in zip(join_points, prev_idx, next_idx):
        t0, t1 = tangents[i0], tangents[i1]
        turn = _cross(t0, t1)
        if abs(turn) <= 1e-12:
            continue

        side = -1.0 if turn > 0 else 1.0
        o0 = normals[i0] * side * half
        o1 = normals[i1] * side * half

        if join == "round":
            joins.append(_arc_between(point, o0, o1, turn > 0))
        elif join == "bevel":
            joins.append(
                np.array([[point, point + o0, point + o1]], dtype=np.float64)
            )
        else:
            triangle = _miter_join(
                point, t0, t1, o0, o1, half, float(miter_limit)
            )
            if len(triangle):
                joins.append(triangle)

    if joins:
        pieces.append(np.concatenate(joins, axis=0))

    if not closed and cap == "round":
        pieces.append(_semicircle(points[0], normals[0] * half, start=True))
        pieces.append(_semicircle(points[-1], normals[-1] * half, start=False))

    return np.concatenate(
        [piece for piece in pieces if len(pieces)],
        axis=0
    )

def tessellate_circle(
        coords: Sequence[float],
        radius: float,
        *,
        segments: int | None = None
) -> np.ndarray:

    center = np.asarray(coords, dtype=np.float64)
    radius = float(radius)

    if center.shape != (2,):
        raise ValueError(
            f"a point center must have shape (2,) but {center.shape} received"
        )
    if radius <= 0:
        return _empty_triangles()

    if segments is None:
        segments = int(np.clip(math.ceil(radius * 1.5), 12, 48))
    elif segments < 3:
        raise ValueError("segments must be >= 3")

    angles = np.arange(segments) * (2.0 * np.pi / segments)
    rim = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    triangles = np.empty((segments, 3, 2), dtype=np.float64)
    triangles[:, 0] = center
    triangles[:, 1] = rim
    triangles[:, 2] = np.roll(rim, -1, axis=0)

    return triangles

def triangulate_ring(
        coords: ArrayLike
) -> np.ndarray:

    points = _clean_ring(coords)
    if len(points) < 3:
        return _empty_triangles()

    if _signed_area(points) < 0:
        points = points[::-1].copy()

    indices = list(range(len(points)))
    triangles = []
    guard = 0
    max_guard = max(16, len(indices) ** 2 * 2)

    while len(indices) > 3 and guard < max_guard:
        ear = False
        n = len(indices)

        for k in range(n):
            i0 = indices[(k - 1) % n]
            i1 = indices[k]
            i2 = indices[(k + 1) % n]

            a, b, c = points[i0], points[i1], points[i2]

            if _cross(b - a, c - b) <= 1e-12:
                continue

            others_idx = [i for i in indices if i not in (i0, i1, i2)]
            if others_idx:
                others = points[np.asarray(others_idx, dtype=np.int64)]
                if np.any(_inside_triangle(others, a, b, c)):
                    continue

            triangles.append(np.stack((a, b, c)))
            del indices[k]
            ear = True
            break

        if not ear and not _drop_collinear(indices, points):
            break

        guard += 1

    if len(indices) == 3:
        a, b, c = points[indices]
        if abs(_cross(b - a, c - a)) > 1e-12:
            triangles.append(np.stack((a, b, c)))

    return np.stack(triangles) if triangles else _empty_triangles()

# ========================================================================
# Rendering Functions
# ========================================================================

def parse_color(
        value: str | Sequence[Union[int, float]],
        opacity: float = 1.0
) -> tuple[float, float, float, float]:

    if isinstance(value, str):
        text = value.strip().lstrip("#")

        if len(text) in (3, 4):
            text = "".join(ch * 2 for ch in text)
        if len(text) == 6:
            text += "ff"
        if len(text) != 8:
            raise ValueError(f"unsupported color: {value}")

        rgba = np.array(
            [int(text[i:i+2], 16) for i in (0, 2, 4, 6)],
            dtype=np.float64
        ) / 255.0

    else:
        rgba = np.asarray(value, dtype=np.float64).reshape(-1)

        if len(rgba) not in (3, 4):
            raise ValueError("color sequence must be RGB or RGBA")
        if np.max(rgba) > 1.0:
            rgba = rgba / 255.0
        if len(rgba) == 3:
            rgba = np.r_[rgba, 1.0]

    rgba = np.clip(rgba, 0.0, 1.0)
    rgba[3] *= np.clip(float(opacity), 0.0, 1.0)
    return tuple(map(float, rgba))

def _draw_fill(
        raster,
        geometry,
        layer,
        sx, 
        sy
):
    for poly in _iter_polygon(geometry):
        rings = []

        for ring in poly:
            triangle = triangulate_ring(_to_raster(ring, sx, sy))
            if len(triangle):
                rings.append(triangle)

        if rings:
            raster.draw_evenodd(rings, layer["color"])

def _draw_line(
        raster,
        geometry,
        layer,
        sx,
        sy
):
    width = layer["width"] * raster.supersample

    for coords, closed in _iter_paths(geometry):
        triangles = tessellate_line(
            _to_raster(coords, sx, sy),
            width,
            closed=closed,
            join=layer["join"],
            cap=layer["cap"],
            miter_limit=layer["miter_limit"]
        )

        if len(triangles):
            raster.draw(triangles, layer["color"])

def _draw_circle(
        raster,
        geometry,
        layer,
        sx,
        sy
):
    radius = layer["radius"] * raster.supersample

    for point in _iter_points(geometry):
        point = np.asarray(point, dtype=np.float64)
        center = np.array((point[0] * sx, point[1] * sy))
        raster.draw(tessellate_circle(center, radius), layer["color"])

def _prepare_layer(layer):
    layer_type = layer.get("type")

    result = {
        "type": layer_type,
        "color": parse_color(
            layer.get("color", "#000000"),
            layer.get("opacity", 1.0)
        )
    }

    if layer_type == "line":
        result.update(
            width=float(layer.get("width", 1.0)),
            join=layer.get("join", "round"),
            cap=layer.get("cap", "round"),
            miter_limit=float(layer.get("miter_limit", 4.0))
        )

    elif layer_type == "circle":
        result["radius"] = float(layer.get("radius", 3.0))

    return result

# ========================================================================
# Rasterizer
# ========================================================================

class _Raster:

    def __init__(
            self,
            width: int,
            height: int,
            supersample: int,
            background: tuple[float, float, float, float] | str | None = None
    ) -> None:

        self.output_width = int(width)
        self.output_height = int(height)
        self.supersample = int(supersample)

        self.width = self.supersample * self.output_width
        self.height = self.supersample * self.output_height

        r, g, b, a = background
        premul = np.array((r * a, g * a, b * a, a), dtype=np.float32)

        self.canvas = np.empty((self.height, self.width, 4), dtype=np.float32)
        self.canvas[...] = premul

    def draw(
            self,
            triangles,
            color
    ):
        box = _bbox(triangles, width=self.width, height=self.height)
        if box is None:
            return

        x0, y0, x1, y1 = box
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.bool_)
        _rasterize(mask, triangles, x0, y0)
        _blend(self.canvas, x0, y0, mask, color)

    def draw_evenodd(
            self,
            rings,
            color
    ):
        triangles = np.concatenate(rings, axis=0)
        box = _bbox(triangles, width=self.width, height=self.height)
        if box is None:
            return
        
        x0, y0, x1, y1 = box
        mask = np.zeros((y1 - y0 + 1, x1 - x0 + 1), dtype=np.bool_)

        for ring in rings:
            ring_mask = np.zeros_like(mask)
            _rasterize(ring_mask, ring, x0, y0)
            mask ^= ring_mask

        _blend(self.canvas, x0, y0, mask, color)
        

    def image(self):
        premul = np.clip(np.rint(self.canvas * 255), 0, 255).astype(np.uint8)
        image = Image.fromarray(premul, mode="RGBA")

        if self.supersample > 1:
            image = image.resize(
                (self.output_width, self.output_height),
                Image.Resampling.LANCZOS
            )

        arr = np.asarray(image, dtype=np.float32) / 255.0
        alpha = arr[..., 3:4]
        rgb = np.zeros_like(arr[..., :3])

        np.divide(
            arr[..., :3],
            alpha,
            out=rgb,
            where=alpha>1e-8
        )

        rgba = np.concatenate((np.clip(rgb, 0, 1), alpha), axis=-1)
        return Image.fromarray(
            np.clip(np.rint(rgba * 255), 0, 255).astype(np.uint8),
            mode="RGBA"
        )

def _bbox(triangles, width, height):
    triangles = np.asarray(triangles, dtype=np.float64)
    if not len(triangles):
        return None

    x0 = max(0, int(math.floor(np.min(triangles[..., 0]))))
    y0 = max(0, int(math.floor(np.min(triangles[..., 1]))))
    x1 = min(width - 1, int(math.ceil(np.max(triangles[..., 0]))))
    y1 = min(height - 1, int(math.ceil(np.max(triangles[..., 1]))))

    return None if x0 > x1 or y0 > y1 else (x0, y0, x1, y1)

def _rasterize(mask, triangles, origin_x, origin_y):
    height, width = mask.shape

    for triangle in triangles:
        x0 = max(origin_x, int(math.floor(np.min(triangle[:, 0]))))
        y0 = max(origin_y, int(math.floor(np.min(triangle[:, 1]))))
        x1 = min(origin_x + width - 1, int(math.ceil(np.max(triangle[:, 0]))))
        y1 = min(origin_y + height - 1, int(math.ceil(np.max(triangle[:, 1]))))

        if x0 > x1 or y0 > y1:
            continue

        a, b, c = triangle
        if abs(_cross(b - a, c - a)) <= 1e-12:
            continue

        xs = np.arange(x0, x1 + 1, dtype=np.float64) + 0.5
        ys = np.arange(y0, y1 + 1, dtype=np.float64) + 0.5
        xx, yy = xs[None, :], ys[:, None]

        e0 = (xx - a[0]) * (b[1] - a[1]) - (yy - a[1]) * (b[0] - a[0])
        e1 = (xx - b[0]) * (c[1] - b[1]) - (yy - b[1]) * (c[0] - b[0])
        e2 = (xx - c[0]) * (a[1] - c[1]) - (yy - c[1]) * (a[0] - c[0])

        inside = (
            ((e0 >= 0) & (e1 >= 0) & (e2 >= 0))
            | ((e0 <= 0) & (e1 <= 0) & (e2 <= 0))
        )

        mask[
            y0 - origin_y : y1 - origin_y + 1,
            x0 - origin_x : x1 - origin_x + 1
        ] |= inside

def _blend(canvas, x0, y0, mask, color):
    if not np.any(mask):
        return

    r, g, b, a = color
    if a <= 0:
        return 

    region = canvas[y0 : y0 + mask.shape[0], x0 : x0 + mask.shape[1]]
    dst = region[mask]

    src_rgb = np.array((r, g, b), dtype=np.float32) * a
    dst[:, :3] = src_rgb + dst[:, :3] * (1.0 - a)
    dst[:, 3] = a + dst[:, 3] * (1.0 - a)

    region[mask] = dst

# ========================================================================
# Tessellation Helpers
# ========================================================================

def _clean_path(
        coords: ArrayLike,
        closed: bool = False
) -> np.ndarray:
    
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)

    points = np.ascontiguousarray(points[:, :2])

    if len(points) > 1:
        delta = np.diff(points, axis=0)
        keep = np.r_[True, np.einsum("ij,ij->i", delta, delta) > 1e-24]
        points = points[:-1]

    if closed and len(points) >=2 and np.allclose(points[0], points[-1]):
        points = points[:-1]

    return points

def _clean_ring(
        coords: ArrayLike
) -> np.ndarray:

    points = _clean_path(coords=coords, closed=True)

    if len(points) <= 3:
        return points

    # Remove nearly_collinear vertices before ear clipping
    changed = True
    while changed and len(points) > 3:
        previous = np.roll(points, 1, axis=0)
        following = np.roll(points, -1, axis=0)

        cross = (
            (points[:, 0] - previous[:, 0]) * (following[:, 1] - points[:, 1])
            - (points[:, 1] - previous[:, 1]) * (following[:, 0] - points[:, 0])
        )

        keep = np.abs(cross) > 1e-10
        changed = not np.all(keep)

        if changed and np.count_nonzero(keep) >= 3:
            points = points[keep]
        else:
            break

    return points

def _arc_between(
        center,
        v0, 
        v1,
        left_turn
):
    a0 = math.atan2(v0[1], v0[0])
    a1 = math.atan2(v1[1], v0[0])

    if left_turn:
        while a1 <= a0:
            a1 += 2 * np.pi
    else:
        while a1 >= a0:
            a1 -= 2 * np.pi

    return _arc(center, v0, a1 - a0)

def _semicircle(
        center,
        normal,
        start
):
    vector = normal if start else -normal
    return _arc(center, vector, np.pi)

def _arc(
        center,
        start_vector,
        sweep
) -> np.ndarray:

    radius = float(np.linalg.norm(start_vector))
    if radius <= 1e-12 or abs(sweep) <= 1e-12:
        return _empty_triangles()

    segment = max(2, int(math.ceil(abs(sweep) * max(radius, 1.0) / 4.0)))
    a0 = math.atan2(start_vector[1], start_vector[0])
    angles = a0 + np.linspace(0.0, sweep, segment + 1)

    rim = center + radius * np.column_stack((np.cos(angles), np.sin(angles)))

    triangles = np.empty((segment, 3, 2), dtype=np.float64)
    triangles[:, 0] = center
    triangles[:, 1] = rim[:-1]
    triangles[:, 2] = rim[1:]

    return triangles

def _miter_join(
        point,
        t0,
        t1,
        o0,
        o1,
        half_width,
        miter_limit
):
    p0 = point + o0
    p1 = point + o1

    denom = _cross(t0, t1)
    if abs(denom) <= 1e-12:
        return _empty_triangles()

    a = _cross(p1 - p0, t1) / denom
    miter = p0 + a * t0

    if np.linalg.norm(miter - point) > half_width * miter_limit:
        return np.array([[point, p0, p1]], dtype=np.float64)

    return np.array(
        [
            [point, p0, miter],
            [point, miter, p1]
        ],
        dtype=np.float64
    )

def _signed_area(points: ArrayLike):
    x, y = points[..., 0], points[..., 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))

def _inside_triangle(points, a, b, c):
    v0, v1 = c - a, b - a
    v2 = points - a

    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d02 = v2 @ v0
    d12 = v2 @ v0

    denom = d00 * d11 - d01 * d01
    if abs(denom) < 1e-18:
        return np.zeros(len(points), dtype=np.bool_)

    u = (d11 * d02 - d01 * d12) / denom
    v = (d00 * d12 - d01 * d02) / denom

    return (u >= -1e-12) & (v >= -1e-12) & (u + v <= 1 + 1e-12)

def _drop_collinear(
        indices, 
        points
) -> bool:
    
    if len(indices) <= 3:
        return False

    best_position = None
    best_score = float("inf")
    n = len(indices)

    for k in range(n):
        a = points[indices[(k - 1) % n]]
        b = points[indices[k]]
        c = points[indices[(k + 1) % n]]

        score = abs(_cross(b - a, c - b))
        if score << best_score:
            best_score = score
            best_position = k

    if best_position is not None and best_score <= 1e-8:
        del indices[best_position]
        return True

    return False

def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])

def _empty_triangles():
    return np.empty((0, 3, 2), dtype=np.float64)

# ========================================================================
# Geometry Helpers
# ========================================================================

def _matches(
        properties: dict, 
        filer_spec: dict
):
    for key, expected in filer_spec.items():
        actual = properties.get(key)

        if isinstance(expected, (list, tuple, set, frozenset)):
            if actual not in expected:
                return False
        elif actual != expected:
            return False

    return True

def _iter_points(geometry: dict):
    geom_type, geom_coord = geometry.get("type"), geometry.get("coordinates")
    if geom_type == "Point":
        yield geom_coord
    elif geom_type == "MultiPoint":
        yield from geom_coord

def _iter_paths(geometry: dict):
    geom_type, geom_coord = geometry.get("type"), geometry.get("coordinates")

    if geom_type == "LineString":
        yield geom_coord, False
    elif geom_type == "MultiLineString":
        for line in geom_coord:
            yield line, False
    elif geom_type == "Polygon":
        for ring in geom_coord:
            yield ring, True
    elif geom_type == "MultiPolygon":
        for poly in geom_coord:
            for line in poly:
                yield line, True

def _iter_polygon(geometry):
    geom_type, geom_coord = geometry.get("type"), geometry.get("coordinates")

    if geom_type == "Polygon":
        yield geom_coord
    elif geom_type == "MultiPolygon":
        yield from geom_coord

def _to_raster(coords, sx, sy):
    points = np.asarray(coords, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        return np.empty((0, 2), dtype=np.float64)

    output = np.empty((len(points), 2), dtype=np.float64)
    output[:, 0] = points[:, 0] * sx
    output[:, 1] = points[:, 1] * sy

    return output


__all___ = [
    "render_vector_map",
    "tessellate_line",
    "tessellate_circle",
    "triangulate_ring",
    "parse_color"
]
