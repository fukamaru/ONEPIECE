from __future__ import annotations

import math
import numpy as np
from os import PathLike
from dataclasses import dataclass
from PIL import Image, ImageColor, ImageDraw
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union, Tuple


ICON_LAYER_TYPE = "icon"
DEFAULT_ICON_SIZE = 8.0

ColorLike = Union[
    str,
    Tuple[int, int, int],
    Tuple[int, int, int, int],
    List[int]
]
FilterFn = Callable[[Mapping[str, Any]], bool]
IconLike = Union[str, PathLike, Image.Image]


# ================================================================
# Styles / Layers
# ================================================================

def parse_color(color: ColorLike, opacity: float = 1.0) -> Tuple[int, int, int, int]:
    """Convert CSS-like color or RGB/RGBA tuple into RBGA."""
    if isinstance(color, str):
        rgba = ImageColor.getcolor(color, "RGBA")
    else:
        values = tuple(int(v) for v in color)
        if len(values) == 3:
            rgba = (*values, 255)
        elif len(values) == 4:
            rgba = values
        else:
            raise ValueError("color tuple/list must be RGB or RGBA.")

    alpha = int(np.clip(round(rgba[3] * float(opacity)), 0, 255))
    return rgba[0], rgba[1], rgba[2], alpha


@dataclass
class FillStyle:
    color: ColorLike = "#4C78A8"
    opacity: float = 1.0
    outline_color: Optional[ColorLike] = None
    outline_width: float = 0.0


@dataclass
class LineStyle:
    color: ColorLike = "#2F2F2F"
    width: float = 1.0
    opacity: float = 1.0
    join: str = "miter"        # miter / bevel / round
    cap: str = "butt"          # butt / square / round
    miter_limit: float = 2.0   # miter limit / max ratio of half-width
    round_segments: int = 12   # round join/cap tessellation


@dataclass
class CircleStyle:
    radius: float = 3.0
    color: ColorLike = "#D62728"
    opacity: float = 1.0
    stroke_color: Optional[ColorLike] = None
    stroke_width: float = 0.0


@dataclass
class Layer:
    """Style layer in map engines."""
    id: str
    type: str               # fill / line / circle / icon
    source: str
    paint: FillStyle | LineStyle | CircleStyle | Mapping[str, Any]
    filter: FilterFn | None = None
    minzoom: float = 0.0
    maxzoom: float = 24.0
    visible: bool = True 


# ================================================================
# Geometry normalize
# ================================================================

def _feature_geometry_and_properties(obj: Mapping[str, Any]) -> Tuple[Optional[Mapping[str, Any]], Mapping[str, Any]]:
    if obj.get("type") == "Feature":
        return obj.get("geometry"), obj.get("properties") or {}
    return obj, {}


def _iter_geometries(obj: Mapping[str, Any]) -> Iterable[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    """Expand FeatureCollection / GeometryCollection."""
    typ = obj.get("type")
    if typ == "FeatureCollection":
        for feature in obj.get("features", []):
            yield from _iter_geometries(feature)
        return

    geom, props = _feature_geometry_and_properties(obj)
    if geom is None:
        return

    if geom.get("type") == "GeometryCollection":
        for sub in geom.get("geometries", []):
            yield sub, props
    else:
        yield geom, props


def _to_points(coords: Any, scale: float = 1.0) -> List[Tuple[float, float]]:
    arr = np.asarray(coords, dtype=np.float64)
    if arr.size == 0:
        return []
    arr = arr.reshape(-1, arr.shape[-1])[:, :2] * scale
    return [tuple(xy) for xy in arr]


# ================================================================
# Polygon tessellation: ear clipping
# ================================================================

def _signed_area(ring: np.ndarray) -> float:
    x = ring[:, 0]
    y = ring[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def _cross(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))


def _point_in_triangle(p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """boundary-included point-in-triangle test"""
    c1 = _cross(a, b, p)
    c2 = _cross(b, c, p)
    c3 = _cross(c, a, p)
    has_neg = (c1 < 0) or (c2 < 0) or (c3 < 0)
    has_pos = (c1 > 0) or (c2 > 0) or (c3 > 0)
    return not (has_neg and has_pos)


def tessellate_ring(ring: Any) -> np.ndarray:
    """
    Ear-clipping triangulation for simple (non-hole) polygon rings。

    Notes
    -----
    - this is a lightning-weight implementation without capacity to deal with following
    - complex polygons with self-interaction and holes, and MultiPolygon
    """
    pts = np.asarray(ring, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("ring must have shape (N, 2)")
    pts = pts[:, :2]

    if len(pts) >= 2 and np.allclose(pts[0], pts[-1]):
        pts = pts[:-1]
    if len(pts) < 3:
        return np.empty((0, 3, 2), dtype=np.float64)

    if _signed_area(pts) < 0:
        pts = pts[::-1].copy()

    indices = list(range(len(pts)))
    triangles: List[np.ndarray] = []

    guard = 0
    max_guard = len(indices) * len(indices) + 16

    while len(indices) > 3 and guard < max_guard:
        ear_found = False
        m = len(indices)

        for i in range(m):
            i_prev = indices[(i - 1) % m]
            i_curr = indices[i]
            i_next = indices[(i + 1) % m]

            a, b, c = pts[i_prev], pts[i_curr], pts[i_next]
            if _cross(a, b, c) <= 1e-12:
                continue

            contains = False
            for j in indices:
                if j in (i_prev, i_curr, i_next):
                    continue
                if _point_in_triangle(pts[j], a, b, c):
                    contains = True
                    break

            if contains:
                continue

            triangles.append(np.stack((a, b, c)))
            del indices[i]
            ear_found = True
            break

        if not ear_found:
            # return empty result for complex polygons
            return np.empty((0, 3, 2), dtype=np.float64)

        guard += 1

    if len(indices) == 3:
        triangles.append(pts[indices])

    if not triangles:
        return np.empty((0, 3, 2), dtype=np.float64)
    return np.stack(triangles)


# ================================================================
# LineString Stroke Tessellation
# ================================================================

def _cross2(a: np.ndarray, b: np.ndarray) -> float:
    """2-D vector cross product."""
    return float(a[0] * b[1] - a[1] * b[0])


def _normalize(v: np.ndarray, eps: float = 1e-12) -> Optional[np.ndarray]:
    """Return a unit vector, or None for a degenerate vector."""
    length = float(np.hypot(v[0], v[1]))
    if length <= eps:
        return None
    return v / length


def _clean_polyline(coords: Any, *, closed: Optional[bool] = None) -> Tuple[np.ndarray, bool]:
    """Normalize a polyline and remove consecutive duplicate vertices.

    Parameters
    ----------
    coords:
        ``(N, 2+)`` coordinates in final image/pixel coordinate space.
    closed:
        ``None`` means auto-detect by comparing first/last vertices.

    Returns
    -------
    points, is_closed
        For a closed line, the duplicated final vertex is removed; closure is
        represented by ``is_closed=True`` instead.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] < 2:
        raise ValueError("line coordinates must have shape (N, 2+)")
    pts = pts[:, :2]
    if len(pts) == 0:
        return pts.copy(), False
    if not np.all(np.isfinite(pts)):
        raise ValueError("line coordinates contain NaN or infinity")

    # Remove consecutive duplicates. They otherwise create zero-length
    # segments whose normal/join is undefined.
    if len(pts) > 1:
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(np.abs(np.diff(pts, axis=0)) > 1e-12, axis=1)
        pts = pts[keep]

    auto_closed = len(pts) >= 3 and np.allclose(pts[0], pts[-1], atol=1e-12, rtol=0.0)
    is_closed = auto_closed if closed is None else bool(closed)

    if is_closed and len(pts) >= 2 and np.allclose(pts[0], pts[-1], atol=1e-12, rtol=0.0):
        pts = pts[:-1]

    return pts, is_closed


def _disk_triangles(center: np.ndarray, radius: float, segments: int) -> List[np.ndarray]:
    """Tessellate a full disk as a triangle fan.

    A full disk is intentionally used for round joins: the union of segment
    rectangles and a radius-r disk at a vertex is exactly the usual round
    stroke construction (Minkowski sum of the centerline with a disk).
    """
    if radius <= 0:
        return []
    segments = max(8, int(segments))
    angles = np.linspace(0.0, 2.0 * math.pi, segments + 1)
    ring = np.column_stack((np.cos(angles), np.sin(angles))) * radius + center
    return [np.stack((center, ring[i], ring[i + 1])) for i in range(segments)]


def _line_intersection(
    p: np.ndarray,
    dp: np.ndarray,
    q: np.ndarray,
    dq: np.ndarray,
    eps: float = 1e-12,
) -> Optional[np.ndarray]:
    """Intersection of two infinite 2-D lines ``p+t*dp`` and ``q+u*dq``."""
    denom = _cross2(dp, dq)
    if abs(denom) <= eps:
        return None
    t = _cross2(q - p, dq) / denom
    out = p + t * dp
    if not np.all(np.isfinite(out)):
        return None
    return out


def tessellate_line(
    coords: Any,
    width: float,
    *,
    join: str = "miter",
    cap: str = "butt",
    miter_limit: float = 2.0,
    round_segments: int = 12,
    closed: Optional[bool] = None,
) -> np.ndarray:
    """Tessellate a stroked polyline into triangles.

    The tessellation is performed in the same coordinate space as ``coords``.
    In this renderer that space is the final output image/pixel space; the
    RasterCanvas later multiplies triangle positions by the supersampling
    factor before rasterization.

    Parameters
    ----------
    coords:
        ``(N, 2+)`` polyline coordinates.
    width:
        Stroke width in output pixels.
    join:
        ``"miter"``, ``"bevel"`` or ``"round"``.
    cap:
        ``"butt"``, ``"square"`` or ``"round"``. Ignored for closed lines.
    miter_limit:
        Maximum ``distance(join_vertex, miter_point) / half_width``. When a
        miter exceeds this value, the join falls back to bevel.
    round_segments:
        Triangle count used for a full round join/cap disk. Larger values are
        smoother before supersampling.
    closed:
        ``None`` auto-detects a repeated first/last vertex. Polygon outlines
        should explicitly pass ``True``.

    Returns
    -------
    np.ndarray
        Triangle array with shape ``(T, 3, 2)``.
    """
    join = str(join).lower()
    cap = str(cap).lower()
    if join not in {"miter", "bevel", "round"}:
        raise ValueError("line join must be 'miter', 'bevel' or 'round'")
    if cap not in {"butt", "square", "round"}:
        raise ValueError("line cap must be 'butt', 'square' or 'round'")
    width = float(width)
    if not math.isfinite(width) or width <= 0:
        return np.empty((0, 3, 2), dtype=np.float64)
    miter_limit = float(miter_limit)
    if not math.isfinite(miter_limit) or miter_limit <= 0:
        raise ValueError("miter_limit must be a positive finite value")

    pts, is_closed = _clean_polyline(coords, closed=closed)
    min_vertices = 3 if is_closed else 2
    if len(pts) < min_vertices:
        return np.empty((0, 3, 2), dtype=np.float64)

    half = width * 0.5
    n_vertices = len(pts)
    n_segments = n_vertices if is_closed else n_vertices - 1

    directions: List[np.ndarray] = []
    normals: List[np.ndarray] = []
    valid_segment = np.ones(n_segments, dtype=bool)

    for i in range(n_segments):
        j = (i + 1) % n_vertices
        d = _normalize(pts[j] - pts[i])
        if d is None:
            valid_segment[i] = False
            directions.append(np.array([1.0, 0.0], dtype=np.float64))
            normals.append(np.array([0.0, 1.0], dtype=np.float64))
        else:
            directions.append(d)
            normals.append(np.array([-d[1], d[0]], dtype=np.float64))

    triangles: List[np.ndarray] = []

    # 1) Segment bodies: each segment is a quad -> two triangles.
    for i in range(n_segments):
        if not valid_segment[i]:
            continue
        j = (i + 1) % n_vertices
        d = directions[i]
        n = normals[i]
        p0 = pts[i].copy()
        p1 = pts[j].copy()

        # Square cap extends the centerline by half the stroke width. Only the
        # first/last segment of an open polyline is extended.
        if not is_closed and cap == "square":
            if i == 0:
                p0 = p0 - d * half
            if i == n_segments - 1:
                p1 = p1 + d * half

        off = n * half
        l0 = p0 + off
        r0 = p0 - off
        l1 = p1 + off
        r1 = p1 - off

        triangles.append(np.stack((l0, r0, r1)))
        triangles.append(np.stack((l0, r1, l1)))

    # 2) Joins.
    join_indices = range(n_vertices) if is_closed else range(1, n_vertices - 1)
    for vi in join_indices:
        prev_seg = (vi - 1) % n_segments
        next_seg = vi % n_segments
        if not valid_segment[prev_seg] or not valid_segment[next_seg]:
            continue

        p = pts[vi]
        d0 = directions[prev_seg]
        d1 = directions[next_seg]
        n0 = normals[prev_seg]
        n1 = normals[next_seg]
        turn = _cross2(d0, d1)
        dot = float(np.dot(d0, d1))

        # Straight-through segments need no join geometry. A 180-degree
        # reversal has no unique miter/bevel side; a disk is a stable fallback
        # that avoids a crack at the reversal point.
        if abs(turn) <= 1e-12:
            if dot < 0.0:
                triangles.extend(_disk_triangles(p, half, round_segments))
            continue

        if join == "round":
            triangles.extend(_disk_triangles(p, half, round_segments))
            continue

        # For a positive turn the outer side is the -normal side; for a
        # negative turn it is the +normal side.
        outer_sign = -1.0 if turn > 0.0 else 1.0
        a = p + n0 * half * outer_sign
        b = p + n1 * half * outer_sign

        if join == "bevel":
            triangles.append(np.stack((p, a, b)))
            continue

        # Miter join = intersection of the two outer offset lines.
        miter = _line_intersection(a, d0, b, d1)
        if miter is None:
            triangles.append(np.stack((p, a, b)))
            continue

        miter_ratio = float(np.linalg.norm(miter - p) / half)
        if (not math.isfinite(miter_ratio)) or miter_ratio > miter_limit:
            triangles.append(np.stack((p, a, b)))
            continue

        # Fill p-a-miter-b as two triangles. Segment bodies already fill the
        # region toward the centerline; these triangles fill only the outer gap.
        triangles.append(np.stack((p, a, miter)))
        triangles.append(np.stack((p, miter, b)))

    # 3) Round caps are independent full endpoint disks. Butt needs nothing;
    # square was handled by segment extension above.
    if not is_closed and cap == "round":
        triangles.extend(_disk_triangles(pts[0], half, round_segments))
        triangles.extend(_disk_triangles(pts[-1], half, round_segments))

    if not triangles:
        return np.empty((0, 3, 2), dtype=np.float64)
    return np.stack(triangles).astype(np.float64, copy=False)


# ================================================================
# Rasterization
# ================================================================

class RasterCanvas:
    """Rasterization Canvas.

    Create a temporal canvas with higher resolution firstly and the render on it,
    finally resize into a smaller one via LANCZOS.

    This is to get smoother edge with lower cost.
    """

    def __init__(
        self,
        width: int,
        height: int,
        *,
        background: ColorLike = (255, 255, 255, 0),
        antialias: int = 2,
    ) -> None:
        if width <= 0 or height <= 0:
            raise ValueError("width / height must be positive.")
        if antialias < 1:
            raise ValueError("antialias must >= 1.")

        self.width = int(width)
        self.height = int(height)
        self.antialias = int(antialias)
        self.scale = float(self.antialias)

        size = (self.width * self.antialias, self.height * self.antialias)
        self.image = Image.new("RGBA", size, parse_color(background))

    def _draw(self) -> ImageDraw.ImageDraw:
        return ImageDraw.Draw(self.image, "RGBA")

    def draw_triangles(self, triangles: np.ndarray, color: ColorLike, opacity: float = 1.0) -> None:
        """Rasterization for tessellated triangles."""
        if triangles.size == 0:
            return
        draw = self._draw()
        fill = parse_color(color, opacity)
        for tri in triangles:
            pts = [(float(x) * self.scale, float(y) * self.scale) for x, y in tri]
            draw.polygon(pts, fill=fill)

    def draw_polygon_masked(
        self,
        rings: Sequence[Any],
        color: ColorLike,
        opacity: float = 1.0,
    ) -> None:
        """Rasterize polygons using alpha masks to support holes."""
        if not rings:
            return

        mask = Image.new("L", self.image.size, 0)
        mdraw = ImageDraw.Draw(mask)

        exterior = _to_points(rings[0], self.scale)
        if len(exterior) >= 3:
            mdraw.polygon(exterior, fill=255)

        for hole in rings[1:]:
            pts = _to_points(hole, self.scale)
            if len(pts) >= 3:
                mdraw.polygon(pts, fill=0)

        rgba = parse_color(color, opacity)
        overlay = Image.new("RGBA", self.image.size, rgba)

        if rgba[3] < 255:
            alpha_arr = np.asarray(mask, dtype=np.uint16)
            alpha_arr = (alpha_arr * rgba[3] // 255).astype(np.uint8)
            mask = Image.fromarray(alpha_arr, mode="L")
            overlay.putalpha(mask)
        else:
            overlay.putalpha(mask)

        self.image = Image.alpha_composite(self.image, overlay)

    def draw_line(self, coords: Any, style: LineStyle, *, closed: Optional[bool] = None) -> None:
        """Rasterize a LineString through triangle stroke tessellation.

        No ``ImageDraw.line`` fallback is used here. The complete stroke
        geometry (segment bodies, joins and caps) is converted to triangles by
        :func:`tessellate_line`, then follows the same triangle rasterization
        path as polygon tessellation.
        """
        triangles = tessellate_line(
            coords,
            style.width,
            join=style.join,
            cap=style.cap,
            miter_limit=style.miter_limit,
            round_segments=style.round_segments,
            closed=closed,
        )
        self.draw_triangles(triangles, style.color, style.opacity)

    def draw_circle(self, coord: Any, style: CircleStyle) -> None:
        xy = np.asarray(coord, dtype=np.float64).reshape(-1)[:2] * self.scale
        x, y = map(float, xy)
        r = max(0.0, float(style.radius) * self.scale)
        box = (x - r, y - r, x + r, y + r)

        draw = self._draw()
        fill = parse_color(style.color, style.opacity)
        outline = parse_color(style.stroke_color, style.opacity) if style.stroke_color else None
        width = max(1, int(round(style.stroke_width * self.scale))) if style.stroke_width > 0 else 1
        draw.ellipse(box, fill=fill, outline=outline, width=width)

    @staticmethod
    def _anchor_fraction(anchor: Union[str, Sequence[float]] = 'center') -> Tuple[float, float]:
        """
        Convert icon anchor to [0,1] normalized coordinates.
        
        Here, 0/1 correspond to PNG pixel centers of left/right and top/bottom.
        Therefore, 'bottom-center' will put the center of PNG bottom line on the target point.
        """
        if isinstance(anchor, str):
            anchors = {
                "top-left": (0.0, 0.0),
                "top-center": (0.5, 0.0),
                "top-right": (1.0, 0.0),
                "left-center": (0.0, 0.5),
                "center": (0.5, 0.5),
                "right-center": (1.0, 0.5),
                "bottom-left": (0.0, 1.0),
                "bottom-center": (0.5, 1.0),
                "bottom-right": (1.0, 1.0),
            }
            try:
                return anchors[anchor]
            except KeyError as exc:
                raise ValueError(
                    f"unknown icon anchor {anchor!r}; supported: {sorted(anchors)}"
                ) from exc

        arr = np.asarray(anchor, dtype=np.float64).reshape(-1)
        if arr.size != 2 or np.any(~np.isfinite(arr)) or np.any((arr < 0) | (arr > 1)):
            raise ValueError("custom icon anchor must be a finite (x, y) pair in [0, 1]")
        return float(arr[0]), float(arr[1])

    def draw_icon(
        self,
        coord: Any,
        icon: Image.Image,
        *,
        size: Optional[Sequence[float]] = None,
        anchor: Union[str, Sequence[float]] = "center",
        offset: Sequence[float] = (0.0, 0.0),
    ) -> None:
        """
        Accurately attach RGBA PNG/icon on a pixel coordinate point, correspond to anchor.
        """
        xy = np.asarray(coord, dtype=np.float64).reshape(-1)
        if xy.size < 2:
            raise ValueError("icon coordinate must contain x/y")

        off = np.asarray(offset, dtype=np.float64).reshape(-1)
        if off.size != 2:
            raise ValueError("icon offset must be (dx, dy)")

        target_x = (float(xy[0]) + float(off[0])) * self.scale
        target_y = (float(xy[1]) + float(off[1])) * self.scale

        rgba = icon.convert("RGBA")
        if size is None:
            out_w, out_h = rgba.size
        else:
            sz = np.asarray(size, dtype=np.float64).reshape(-1)
            if sz.size != 2 or np.any(~np.isfinite(sz)) or np.any(sz <= 0):
                raise ValueError("icon size must be positive (width, height)")
            out_w, out_h = float(sz[0]), float(sz[1])

        internal_w = max(1, int(round(out_w * self.scale)))
        internal_h = max(1, int(round(out_h * self.scale)))
        if rgba.size != (internal_w, internal_h):
            rgba = rgba.resize((internal_w, internal_h), Image.Resampling.LANCZOS)

        ax, ay = self._anchor_fraction(anchor)
        # compute anchor on final output pixel and than supersampling
        # this is for avoid pixel drift
        anchor_out_x = ax * (out_w - 1.0)
        anchor_out_y = ay * (out_h - 1.0)
        left = int(round((float(xy[0]) + float(off[0]) - anchor_out_x) * self.scale))
        top = int(round((float(xy[1]) + float(off[1]) - anchor_out_y) * self.scale))

        self.image.alpha_composite(rgba, dest=(left, top))

    def draw_polygon_outline(self, rings: Sequence[Any], style: FillStyle) -> None:
        if style.outline_color is None or style.outline_width <= 0:
            return
        line_style = LineStyle(
            color=style.outline_color,
            width=style.outline_width,
            opacity=style.opacity,
        )
        for ring in rings:
            self.draw_line(ring, line_style)

    def finish(self) -> Image.Image:
        if self.antialias == 1:
            return self.image
        return self.image.resize((self.width, self.height), Image.Resampling.LANCZOS)


# ================================================================
# Vector Renderer
# ================================================================

class VectorRenderer:
    """
    Lightning-Weight Vector Map Renderer.
    The usage is analogue to common map style engines:

    1. ''add_source(name, geojson_or_features)''
    2. ''add_layer(Layer(...))''
    3. ''render()''
    """

    def __init__(
        self,
        width: int = 256,
        height: int = 256,
        *,
        extent: float | None = 4096.0,
        background: ColorLike = (255, 255, 255, 0),
        antialias: int = 2,
    ) -> None:
        self.width = int(width)
        self.height = int(height)
        self.extent = extent
        self.background = background
        self.antialias = int(antialias)
        self.sources: Dict[str, Any] = {}
        self.layers: List[Layer] = []
        # only open icon file once
        self._icon_cache: Dict[str, Image.Image] = {}

    def _tile_to_image_xy(self, coords: Any) -> np.ndarray:
        """Convert tile coordinate to image pixel coordinate."""
        arr = np.asarray(coords, dtype=np.float64)
        if arr.shape[-1] < 2:
            raise ValueError("coordinates last dimension must contain x/y")
        out = arr[..., :2].copy()
        if self.extent is None:
            return out
        out[..., 0] *= self.width / self.extent
        out[..., 1] *= self.height / self.extent
        return out

    def _geometry_to_image(self, geom: Mapping[str, Any]) -> Mapping[str, Any]:
        typ = geom.get("type")
        coords = geom.get("coordinates")

        if typ == "Point":
            mapped = self._tile_to_image_xy(coords)
        elif typ in {"MultiPoint", "LineString"}:
            mapped = self._tile_to_image_xy(coords)
        elif typ in {"MultiLineString", "Polygon"}:
            mapped = [self._tile_to_image_xy(part) for part in coords]
        elif typ == "MultiPolygon":
            mapped = [
                [self._tile_to_image_xy(ring) for ring in polygon]
                for polygon in coords
            ]
        else:
            return geom

        return {**geom, "coordinates": mapped}

    def add_source(self, name: str, data: Any) -> None:
        """
        Data source registeration:
        - GeoJSON FeatureCollection
        - Feature / Geometry
        - Feature / Geometry list
        """
        self.sources[name] = data

    def add_layer(self, layer: Layer) -> None:
        if layer.type not in ("fill", "line", "circle", ICON_LAYER_TYPE):
            raise ValueError("Layer.type only supports fill / line / circle / line-endpoint")
        if layer.source not in self.sources:
            raise KeyError(f"source {layer.source!r} has not registered")
        self.layers.append(layer)

    def _iter_source_items(self, source: Any) -> Iterable[Mapping[str, Any]]:
        if isinstance(source, Mapping):
            if source.get("type") == "FeatureCollection":
                yield from source.get("features", [])
            else:
                yield source
        else:
            yield from source

    def _load_icon(self, icon: IconLike) -> Image.Image:
        """Load PNG / icon."""
        if isinstance(icon, Image.Image):
            return icon
        if isinstance(icon, (str, PathLike)):
            path = str(icon)
            cached = self._icon_cache.get(path)
            if cached is not None:
                return cached
            with Image.open(path) as im:
                loaded = im.convert("RGBA")
                loaded.load()
            self._icon_cache[path] = loaded
            return loaded
        raise TypeError("icon must be a PNG/image path or PIL.Image.Image")

    def _draw_icon_instance(
        self,
        canvas: RasterCanvas,
        coord: Any,
        paint: Mapping[str, Any],
        *,
        prefix: str = "",
    ) -> bool:
        """Draw one icon using common paint keys plus an optional prefix.

        Common keys are ``icon``, ``size``, ``anchor`` and ``offset``.
        Endpoint placement can override them with ``start_icon``, ``end_icon``
        etc. without changing the generic PNG rendering primitive.
        """
        key = lambda name: f"{prefix}_{name}" if prefix else name
        icon_value = paint.get(key("icon"), paint.get("icon"))
        if icon_value is None:
            return False
        canvas.draw_icon(
            coord,
            self._load_icon(icon_value),
            size=paint.get(key("size"), paint.get("size")),
            anchor=paint.get(key("anchor"), paint.get("anchor", "bottom-center")),
            offset=paint.get(key("offset"), paint.get("offset", (0.0, 0.0))),
        )
        return True

    def _draw_line_icon_placement(
        self,
        canvas: RasterCanvas,
        coords: Any,
        paint: Mapping[str, Any],
        placement: str,
    ) -> None:
        """Extract anchor coordinates from one LineString and draw icons."""
        arr = np.asarray(coords, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] < 2:
            return
        arr = arr[:, :2]

        if placement == "line-start":
            self._draw_icon_instance(canvas, arr[0], paint, prefix="start")
        elif placement == "line-end":
            self._draw_icon_instance(canvas, arr[-1], paint, prefix="end")
        elif placement == "line-endpoints":
            self._draw_icon_instance(canvas, arr[0], paint, prefix="start")
            self._draw_icon_instance(canvas, arr[-1], paint, prefix="end")
        elif placement == "vertices":
            for coord in arr:
                self._draw_icon_instance(canvas, coord, paint)
        else:
            raise ValueError(
                "icon placement for LineString must be 'line-start', 'line-end', "
                "'line-endpoints' or 'vertices'"
            )

    def _render_icon_geometry(
        self,
        canvas: RasterCanvas,
        geom: Mapping[str, Any],
        paint: Mapping[str, Any],
    ) -> None:
        """Render PNG icons independently from line/fill/circle rendering.

        Point/MultiPoint
            ``placement`` defaults to ``"point"`` and uses the common
            ``icon/size/anchor/offset`` keys.

        LineString/MultiLineString
            Set ``placement`` to ``"line-start"``, ``"line-end"``,
            ``"line-endpoints"`` or ``"vertices"``. For endpoints the common
            icon can be overridden with ``start_*`` and ``end_*`` keys.
        """
        typ = geom.get("type")
        coords = geom.get("coordinates")
        placement = str(paint.get("placement", "point")).lower()

        if typ == "Point":
            if placement not in {"point", "points", "auto"}:
                return
            self._draw_icon_instance(canvas, coords, paint)
            return

        if typ == "MultiPoint":
            if placement not in {"point", "points", "auto"}:
                return
            for coord in coords:
                self._draw_icon_instance(canvas, coord, paint)
            return

        if typ == "LineString":
            if placement in {"point", "points", "auto"}:
                return
            self._draw_line_icon_placement(canvas, coords, paint, placement)
            return

        if typ == "MultiLineString":
            if placement in {"point", "points", "auto"}:
                return
            mode = str(paint.get("multiline_mode", "each")).lower()
            lines = [np.asarray(line, dtype=np.float64) for line in coords if len(line) > 0]
            if not lines:
                return
            if mode == "each" or placement == "vertices":
                for line in lines:
                    self._draw_line_icon_placement(canvas, line, paint, placement)
                return
            if mode != "overall":
                raise ValueError("multiline_mode must be 'each' or 'overall'")

            # Overall means the whole MultiLineString is treated as one route
            # for start/end placement. Vertices still use each component above.
            if placement == "line-start":
                self._draw_icon_instance(canvas, lines[0][0], paint, prefix="start")
            elif placement == "line-end":
                self._draw_icon_instance(canvas, lines[-1][-1], paint, prefix="end")
            elif placement == "line-endpoints":
                self._draw_icon_instance(canvas, lines[0][0], paint, prefix="start")
                self._draw_icon_instance(canvas, lines[-1][-1], paint, prefix="end")
            return

    def render(self, *, zoom: float = 0.0) -> Image.Image:
        canvas = RasterCanvas(
            self.width,
            self.height,
            background=self.background,
            antialias=self.antialias,
        )

        for layer in self.layers:
            if not layer.visible or not (layer.minzoom <= zoom < layer.maxzoom):
                continue

            source = self.sources[layer.source]
            for item in self._iter_source_items(source):
                for geom, props in _iter_geometries(item):
                    if layer.filter is not None and not layer.filter(props):
                        continue
                    self._render_geometry(canvas, self._geometry_to_image(geom), layer)

        return canvas.finish()

    def _render_geometry(self, canvas: RasterCanvas, geom: Mapping[str, Any], layer: Layer) -> None:
        typ = geom.get("type")
        coords = geom.get("coordinates")

        if layer.type == ICON_LAYER_TYPE:
            if not isinstance(layer.paint, Mapping):
                raise TypeError("paint of icon layer must be Mapping/dict")
            self._render_icon_geometry(canvas, geom, layer.paint)
            return

        if layer.type == "circle":
            style = layer.paint
            if not isinstance(style, CircleStyle):
                raise TypeError("circle layer requires CircleStyle")
            if typ == "Point":
                canvas.draw_circle(coords, style)
            elif typ == "MultiPoint":
                for p in coords:
                    canvas.draw_circle(p, style)
            return

        if layer.type == "line":
            style = layer.paint
            if not isinstance(style, LineStyle):
                raise TypeError("line layer requires LineStyle")

            if typ == "LineString":
                canvas.draw_line(coords, style)
            elif typ == "MultiLineString":
                for line in coords:
                    canvas.draw_line(line, style)
            elif typ == "Polygon":
                # Polygon boundary is closed stroke, unneed cap。
                for ring in coords:
                    canvas.draw_line(ring, style, closed=True)
            elif typ == "MultiPolygon":
                for polygon in coords:
                    for ring in polygon:
                        canvas.draw_line(ring, style, closed=True)
            return

        if layer.type == "fill":
            style = layer.paint
            if not isinstance(style, FillStyle):
                raise TypeError("fill layer requires FillStyle")

            if typ == "Polygon":
                self._render_polygon(canvas, coords, style)
            elif typ == "MultiPolygon":
                for polygon in coords:
                    self._render_polygon(canvas, polygon, style)
            return

    @staticmethod
    def _render_polygon(canvas: RasterCanvas, rings: Sequence[Any], style: FillStyle) -> None:
        if not rings:
            return
        
        if len(rings) == 1:
            triangles = tessellate_ring(rings[0])
            if len(triangles) > 0:
                canvas.draw_triangles(triangles, style.color, style.opacity)
            else:
                canvas.draw_polygon_masked(rings, style.color, style.opacity)
        else:
            canvas.draw_polygon_masked(rings, style.color, style.opacity)

        canvas.draw_polygon_outline(rings, style)


__all__ = [
    "FillStyle",
    "LineStyle",
    "CircleStyle",
    "Layer",
    "IconLike",
    "RasterCanvas",
    "VectorRenderer",
    "parse_color",
    "tessellate_ring",
    "tessellate_line",
]
