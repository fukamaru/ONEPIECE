from __future__ import annotations

import numpy as np
from typing import Any
from collections.abc import Mapping, Sequence

EXTENT = 4096
_SUPPORTED = {"Point", "LineString", "Polygon"}


# ==========================================================
# Numeric Coodinate Conversion
# ==========================================================

def _as_xy_array(coords: Any, copy_array: bool = False) -> np.ndarray:
    """Return coordinates as float64 ``(..., 2)`` without changing shape.

    Z/M values are intentionally ignored by this 2-D renderer.
    """
    arr = (
        np.array(coords, dtype=np.float64, copy=True)
        if copy_array
        else np.asarray(coords, dtype=np.float64)
    )
    if arr.ndim < 1 or arr.shape[-1] < 2:
        raise ValueError(f"coordinates must have shape (..., 2+), got {arr.shape}")
    out = arr[..., :2]
    if not np.all(np.isfinite(out)):
        raise ValueError("coordinates contain NaN or infinity")
    return out


def _broadcast_leading(value: Any, coord_shape: tuple[int, ...], name: str) -> np.ndarray:
    """Broadcast a scalar/batch parameter over LEADING coordinate dimensions.

    Example: coordinates ``(B, N, 2)`` and extent ``(B,)`` become ``(B, N)``.
    NumPy's default trailing-axis broadcast would not express that intent.
    """
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim == 0:
        return np.broadcast_to(arr, coord_shape)
    if arr.ndim > len(coord_shape):
        raise ValueError(f"{name} shape {arr.shape} has too many dimensions for {coord_shape}")
    leading = coord_shape[:arr.ndim]
    for actual, expected in zip(arr.shape, leading):
        if actual not in (1, expected):
            raise ValueError(
                f"{name} shape {arr.shape} cannot match leading coordinate dimensions {leading}"
            )
    reshaped = arr.reshape(arr.shape + (1,) * (len(coord_shape) - arr.ndim))
    return np.broadcast_to(reshaped, coord_shape)


def _broadcast_bounds(bounds: Any, coord_shape: tuple[int, ...]) -> np.ndarray:
    """Broadcast ``(...,4)`` bounds over leading coordinate dimensions.

    Examples
    --------
    ``coords.shape == (N,2)``, ``bounds.shape == (4,)`` -> ``(N,4)``

    ``coords.shape == (B,N,2)``, ``bounds.shape == (B,4)`` -> ``(B,N,4)``
    """
    arr = np.asarray(bounds, dtype=np.float64)
    if arr.ndim < 1 or arr.shape[-1] != 4:
        raise ValueError(f"bounds must have shape (4,) or (...,4), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError("bounds contain NaN or infinity")

    batch_shape = arr.shape[:-1]
    if len(batch_shape) > len(coord_shape):
        raise ValueError(
            f"bounds batch shape {batch_shape} has more dimensions than coordinate shape {coord_shape}"
        )
    leading = coord_shape[:len(batch_shape)]
    for actual, expected in zip(batch_shape, leading):
        if actual not in (1, expected):
            raise ValueError(
                f"bounds batch shape {batch_shape} cannot match leading coordinate dimensions {leading}"
            )

    reshape_shape = batch_shape + (1,) * (len(coord_shape) - len(batch_shape)) + (4,)
    return np.broadcast_to(arr.reshape(reshape_shape), coord_shape + (4,))


def source_to_tile(
    coords: Any,
    bounds: Any,
    extent: int | float = EXTENT,
) -> np.ndarray:
    """Convert source/geographic coordinates to local tile coordinates.

    The transform is purely affine.  Coordinates outside ``bounds`` are
    intentionally allowed and therefore may map outside ``[0, extent]``.
    Geometry clipping is a separate operation performed by ``preprocess``.
    """
    xy = _as_xy_array(coords)
    shape = xy.shape[:-1]
    b = _broadcast_bounds(bounds, shape)
    e = _broadcast_leading(extent, shape, "extent")

    if np.any(e <= 0) or not np.all(np.isfinite(e)):
        raise ValueError("extent must be positive and finite")

    xmin, ymin, xmax, ymax = (b[..., i] for i in range(4))
    sx = xmax - xmin
    sy = ymax - ymin
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("bounds require xmax > xmin and ymax > ymin")

    tx = (xy[..., 0] - xmin) / sx * e
    # Tile/image convention: y=0 is the top (maximum source y).
    ty = (ymax - xy[..., 1]) / sy * e
    return np.stack((tx, ty), axis=-1)


def tile_to_source(
    tile_coords: Any,
    bounds: Any,
    extent: int | float = EXTENT,
) -> np.ndarray:
    """Inverse of :func:`source_to_tile`.

    This converts local tile coordinates back to the source/geographic
    coordinate system described by ``bounds``.  Values outside the tile are
    extrapolated rather than clipped, exactly mirroring ``source_to_tile``.
    """
    xy = _as_xy_array(tile_coords)
    shape = xy.shape[:-1]
    b = _broadcast_bounds(bounds, shape)
    e = _broadcast_leading(extent, shape, "extent")

    if np.any(e <= 0) or not np.all(np.isfinite(e)):
        raise ValueError("extent must be positive and finite")

    xmin, ymin, xmax, ymax = (b[..., i] for i in range(4))
    sx = xmax - xmin
    sy = ymax - ymin
    if np.any(sx <= 0) or np.any(sy <= 0):
        raise ValueError("bounds require xmax > xmin and ymax > ymin")

    x = xmin + xy[..., 0] / e * sx
    y = ymax - xy[..., 1] / e * sy
    return np.stack((x, y), axis=-1)


# Explicit geographic names retained because geographic lon/lat is a common
# source coordinate system.  No projection is implied by the aliases.
geographic_to_tile = source_to_tile
tile_to_geographic = tile_to_source

# Compatibility names for the earlier functional prototype.
_lonlat_to_tile_custom = source_to_tile
_tile_to_lonlat_custom = tile_to_source

# ==========================================================
# Explicit Input Adapters
# ==========================================================

def _optional_shapely_mapping(obj: Any) -> Mapping[str, Any] | None:
    """Return GeoJSON mapping for a Shapely geometry, or None.

    Importing Shapely is optional and only attempted when needed.
    """
    try:
        from shapely.geometry.base import BaseGeometry
        from shapely.geometry import mapping as shapely_mapping
    except Exception:
        return None
    if isinstance(obj, BaseGeometry):
        return shapely_mapping(obj)
    return None


def _optional_geopandas_features(obj: Any) -> tuple[list[dict[str, Any]], Any] | None:
    """Convert GeoSeries/GeoDataFrame explicitly; returns (features, crs)."""
    try:
        import geopandas as gpd
        from shapely.geometry import mapping as shapely_mapping
    except Exception:
        return None

    if isinstance(obj, gpd.GeoSeries):
        features = []
        for geom in obj:
            if geom is None or geom.is_empty:
                continue
            features.append({
                "type": "Feature",
                "properties": {},
                "geometry": shapely_mapping(geom),
            })
        return features, obj.crs

    if isinstance(obj, gpd.GeoDataFrame):
        geometry_name = obj.geometry.name
        features = []
        for _, row in obj.iterrows():
            geom = row[geometry_name]
            if geom is None or geom.is_empty:
                continue
            props = {
                str(k): v
                for k, v in row.items()
                if k != geometry_name
            }
            features.append({
                "type": "Feature",
                "properties": props,
                "geometry": shapely_mapping(geom),
            })
        return features, obj.crs
    return None


def _infer_array_type(arr: np.ndarray) -> str:
    if arr.ndim == 1 and arr.size >= 2:
        return "Point"
    if arr.ndim == 2 and arr.shape[-1] >= 2:
        if len(arr) == 1:
            return "Point"
        if len(arr) >= 4 and np.array_equal(arr[0, :2], arr[-1, :2]):
            return "Polygon"
        return "LineString"
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        closed = [len(r) >= 4 and np.array_equal(r[0, :2], r[-1, :2]) for r in arr]
        return "Polygon" if all(closed) else "MultiLineString"
    raise ValueError(f"cannot infer ndarray geometry type from shape {arr.shape}")


def _numeric_geometry_to_geojson(data: Any, geometry_type: str | None = None) -> dict[str, Any]:
    arr = _as_xy_array(data)
    typ = geometry_type or _infer_array_type(arr)
    if typ not in _SUPPORTED:
        raise ValueError(f"unsupported geometry_type={typ!r}")

    if typ == "Point":
        coords = arr.reshape(-1, 2)[0].tolist()
    else:
        coords = arr.tolist()
    return {"type": typ, "coordinates": coords}


def _is_numeric_nested(value: Any) -> bool:
    try:
        arr = np.asarray(value)
    except Exception:
        return False
    return arr.dtype != object and arr.ndim >= 1 and arr.shape[-1] >= 2


def to_feature_collection(
    data: Any,
    geometry_type: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Normalize supported input into a standard GeoJSON FeatureCollection.

    Supported inputs: one GeoJSON Feature/Geometry, FeatureCollection,
    ndarray/numeric geometry, Shapely geometry, GeoSeries, GeoDataFrame, or a
    sequence of those objects.  A single feature never needs to be wrapped in
    a list.
    """
    # GeoPandas must be checked before generic Sequence handling.
    gpd_result = _optional_geopandas_features(data)
    if gpd_result is not None:
        features, crs = gpd_result
        fc: dict[str, Any] = {"type": "FeatureCollection", "features": features}
        if crs is not None:
            fc["crs"] = str(crs)
        return fc

    if isinstance(data, Mapping):
        typ = data.get("type")
        if typ == "FeatureCollection":
            # Make a shallow standardized copy; preserve CRS/coordinate_space metadata.
            out = dict(data)
            out["features"] = list(data.get("features", []) or [])
            return out
        if typ == "Feature":
            return {"type": "FeatureCollection", "features": [dict(data)]}
        if typ in _SUPPORTED:
            return {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": dict(data)}],
            }
        if typ == "GeometryCollection":
            return {
                "type": "FeatureCollection",
                "features": [
                    {"type": "Feature", "properties": {}, "geometry": dict(g)}
                    for g in data.get("geometries", []) or []
                ],
            }
        raise TypeError(f"unsupported GeoJSON type {typ!r}")

    shapely_geom = _optional_shapely_mapping(data)
    if shapely_geom is not None:
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": dict(shapely_geom)}],
        }

    if isinstance(data, np.ndarray):
        if isinstance(geometry_type, Sequence) and not isinstance(geometry_type, str):
            raise ValueError("geometry_type sequence is only valid for a sequence of geometry objects")
        geom = _numeric_geometry_to_geojson(data, geometry_type if isinstance(geometry_type, str) else None)
        return {
            "type": "FeatureCollection",
            "features": [{"type": "Feature", "properties": {}, "geometry": geom}],
        }

    if isinstance(data, Sequence) and not isinstance(data, (str, bytes)):
        items = list(data)

        # A list of explicit geometry objects is a batch even if equally-shaped
        # ndarray items could be stacked into one 3-D numeric array.  Pure
        # numeric nesting (lists/tuples of numbers) remains one geometry.
        object_batch = any(
            isinstance(item, (np.ndarray, Mapping))
            or _optional_shapely_mapping(item) is not None
            for item in items
        )
        if not object_batch and _is_numeric_nested(data):
            if isinstance(geometry_type, Sequence) and not isinstance(geometry_type, str):
                raise ValueError("geometry_type sequence cannot describe one numeric geometry")
            geom = _numeric_geometry_to_geojson(
                data, geometry_type if isinstance(geometry_type, str) else None
            )
            return {
                "type": "FeatureCollection",
                "features": [{"type": "Feature", "properties": {}, "geometry": geom}],
            }

        if geometry_type is None or isinstance(geometry_type, str):
            types = [geometry_type] * len(items)
        else:
            types = list(geometry_type)
            if len(types) != len(items):
                raise ValueError("geometry_type sequence length must match input sequence length")

        features: list[dict[str, Any]] = []
        for item, typ_hint in zip(items, types):
            sub = to_feature_collection(item, geometry_type=typ_hint)
            features.extend(sub.get("features", []))
        return {"type": "FeatureCollection", "features": features}

    raise TypeError(f"unsupported geometry input: {type(data)!r}")


def coordinate_arrays(data: Any, geometry_type: str | Sequence[str] | None = None) -> list[np.ndarray]:
    """Extract all coordinate arrays from supported input for inspection."""
    fc = to_feature_collection(data, geometry_type=geometry_type)
    out: list[np.ndarray] = []

    def collect(geom: Mapping[str, Any]) -> None:
        typ = geom.get("type")
        c = geom.get("coordinates")
        if typ == "Point":
            out.append(_as_xy_array(c).reshape(1, 2))
        elif typ in {"MultiPoint", "LineString"}:
            out.append(_as_xy_array(c).reshape(-1, 2))
        elif typ in {"MultiLineString", "Polygon"}:
            out.extend(_as_xy_array(part).reshape(-1, 2) for part in c)
        elif typ == "MultiPolygon":
            for poly in c:
                out.extend(_as_xy_array(ring).reshape(-1, 2) for ring in poly)

    for feat in fc.get("features", []):
        geom = feat.get("geometry")
        if geom is not None:
            collect(geom)
    return out


# ==========================================================
# Geometry-Level Coordinate Transforms
# ==========================================================

def _map_coordinate_structure(coords: Any, fn) -> Any:
    """Apply a vectorized XY transform while preserving GeoJSON nesting."""
    try:
        arr = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.ndim >= 1 and arr.shape[-1] >= 2:
        return fn(arr[..., :2]).tolist()
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes)):
        return [_map_coordinate_structure(item, fn) for item in coords]
    raise TypeError("invalid coordinate structure")


def _transform_geojson_geometry(geom: Mapping[str, Any], fn) -> dict[str, Any]:
    typ = geom.get("type")
    if typ not in _SUPPORTED:
        raise ValueError(f"unsupported geometry type {typ!r}")
    out = dict(geom)
    out["coordinates"] = _map_coordinate_structure(geom.get("coordinates"), fn)
    return out


def source_geometry_to_tile(
    data: Any,
    bounds: Any,
    extent: int | float = EXTENT,
    geometry_type: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Convert supported geometry input to tile-coordinate GeoJSON.

    Unlike ``preprocess``, this function performs coordinate conversion only:
    no clipping, simplification or quantization.
    """
    fc = to_feature_collection(data, geometry_type=geometry_type)
    features = []
    for feat in fc.get("features", []):
        out_feat = dict(feat)
        geom = feat.get("geometry")
        out_feat["geometry"] = None if geom is None else _transform_geojson_geometry(
            geom, lambda a: source_to_tile(a, bounds, extent)
        )
        features.append(out_feat)
    out = {"type": "FeatureCollection", "features": features}
    out["coordinate_space"] = "tile"
    out["extent"] = float(extent)
    out["source_bounds"] = np.asarray(bounds, dtype=np.float64).tolist()
    return out


def tile_geometry_to_source(
    data: Any,
    bounds: Any | None = None,
    extent: int | float | None = None,
    geometry_type: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Convert tile-coordinate geometry back to source/geographic coordinates.

    If ``data`` is a FeatureCollection produced by ``preprocess`` or
    ``source_geometry_to_tile``, ``bounds`` and ``extent`` can be omitted and
    are read from ``source_bounds`` / ``extent`` metadata.
    """
    fc = to_feature_collection(data, geometry_type=geometry_type)
    if bounds is None and isinstance(data, Mapping):
        bounds = data.get("source_bounds")
    if extent is None and isinstance(data, Mapping):
        extent = data.get("extent")
    if bounds is None:
        raise ValueError("bounds are required unless data contains source_bounds metadata")
    if extent is None:
        extent = EXTENT

    features = []
    for feat in fc.get("features", []):
        out_feat = dict(feat)
        geom = feat.get("geometry")
        out_feat["geometry"] = None if geom is None else _transform_geojson_geometry(
            geom, lambda a: tile_to_source(a, bounds, extent)
        )
        features.append(out_feat)
    return {
        "type": "FeatureCollection",
        "coordinate_space": "source",
        "features": features,
    }


# ==========================================================
# Simplification / Clipping
# ==========================================================

def douglas_peucker(points: Any, tolerance: float) -> np.ndarray:
    pts = _as_xy_array(points).reshape(-1, 2)
    tolerance = float(tolerance)
    if tolerance <= 0 or len(pts) <= 2:
        return pts.copy()

    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    tol2 = tolerance * tolerance

    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b = pts[i], pts[j]
        seg = b - a
        seg2 = float(seg @ seg)
        middle = pts[i + 1:j]
        if seg2 == 0:
            d2 = np.sum((middle - a) ** 2, axis=1)
        else:
            t = np.clip(((middle - a) @ seg) / seg2, 0.0, 1.0)
            proj = a + t[:, None] * seg
            d2 = np.sum((middle - proj) ** 2, axis=1)
        krel = int(np.argmax(d2))
        if d2[krel] > tol2:
            k = i + 1 + krel
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return pts[keep]


def _close_ring(points: np.ndarray) -> np.ndarray:
    a = np.asarray(points)
    if len(a) and not np.array_equal(a[0], a[-1]):
        a = np.vstack((a, a[0]))
    return np.ascontiguousarray(a)


def _clip_segment(p0: np.ndarray, p1: np.ndarray, extent: float) -> tuple[np.ndarray, np.ndarray] | None:
    """Liang-Barsky segment clipping against [0,extent]^2."""
    x0, y0 = map(float, p0)
    x1, y1 = map(float, p1)
    dx, dy = x1 - x0, y1 - y0
    p = (-dx, dx, -dy, dy)
    q = (x0, extent - x0, y0, extent - y0)
    u0, u1 = 0.0, 1.0
    for pi, qi in zip(p, q):
        if abs(pi) < 1e-15:
            if qi < 0:
                return None
            continue
        t = qi / pi
        if pi < 0:
            u0 = max(u0, t)
        else:
            u1 = min(u1, t)
        if u0 > u1:
            return None
    a = np.array([x0 + u0 * dx, y0 + u0 * dy], dtype=np.float64)
    b = np.array([x0 + u1 * dx, y0 + u1 * dy], dtype=np.float64)
    return a, b


def _clip_line(points: np.ndarray, extent: float) -> list[np.ndarray]:
    pts = _as_xy_array(points).reshape(-1, 2)
    if len(pts) < 2:
        return []
    pieces: list[list[np.ndarray]] = []
    current: list[np.ndarray] = []
    for i in range(len(pts) - 1):
        clipped = _clip_segment(pts[i], pts[i + 1], extent)
        if clipped is None:
            if len(current) >= 2:
                pieces.append(current)
            current = []
            continue
        a, b = clipped
        if not current:
            current = [a, b]
        elif np.allclose(current[-1], a, atol=1e-9):
            if not np.allclose(current[-1], b, atol=1e-9):
                current.append(b)
        else:
            if len(current) >= 2:
                pieces.append(current)
            current = [a, b]
    if len(current) >= 2:
        pieces.append(current)
    return [np.asarray(p, dtype=np.float64) for p in pieces]


def _clip_ring(points: np.ndarray, extent: float) -> np.ndarray:
    """Sutherland-Hodgman polygon ring clipping against the tile rectangle."""
    poly = _as_xy_array(points).reshape(-1, 2)
    if len(poly) and np.array_equal(poly[0], poly[-1]):
        poly = poly[:-1]
    if len(poly) < 3:
        return np.empty((0, 2), dtype=np.float64)

    def edge(vertices, inside, intersect):
        if len(vertices) == 0:
            return vertices
        out = []
        prev = vertices[-1]
        prev_in = inside(prev)
        for cur in vertices:
            cur_in = inside(cur)
            if cur_in:
                if not prev_in:
                    out.append(intersect(prev, cur))
                out.append(cur)
            elif prev_in:
                out.append(intersect(prev, cur))
            prev, prev_in = cur, cur_in
        return np.asarray(out, dtype=np.float64)

    def xi(x):
        def fn(a, b):
            t = 0.0 if b[0] == a[0] else (x - a[0]) / (b[0] - a[0])
            return np.array([x, a[1] + t * (b[1] - a[1])], dtype=np.float64)
        return fn

    def yi(y):
        def fn(a, b):
            t = 0.0 if b[1] == a[1] else (y - a[1]) / (b[1] - a[1])
            return np.array([a[0] + t * (b[0] - a[0]), y], dtype=np.float64)
        return fn

    poly = edge(poly, lambda p: p[0] >= 0, xi(0.0))
    poly = edge(poly, lambda p: p[0] <= extent, xi(extent))
    poly = edge(poly, lambda p: p[1] >= 0, yi(0.0))
    poly = edge(poly, lambda p: p[1] <= extent, yi(extent))
    return _close_ring(poly) if len(poly) >= 3 else np.empty((0, 2), dtype=np.float64)


def _remove_adjacent_duplicates(a: np.ndarray) -> np.ndarray:
    if len(a) <= 1:
        return a
    keep = np.ones(len(a), dtype=bool)
    keep[1:] = np.any(a[1:] != a[:-1], axis=1)
    return a[keep]


def _process_piece(piece: np.ndarray, *, is_ring: bool, simplify: float, quantize: bool, extent: int) -> np.ndarray:
    if len(piece) == 0:
        return piece
    if simplify > 0:
        if is_ring:
            ring = _close_ring(piece)
            open_ring = ring[:-1]
            if len(open_ring) >= 3:
                # Simplify a closed ring by temporarily closing it after DP.
                simp = douglas_peucker(np.vstack((open_ring, open_ring[0])), simplify)
                piece = _close_ring(simp[:-1] if len(simp) > 1 else open_ring)
        else:
            piece = douglas_peucker(piece, simplify)
    if quantize:
        piece = np.rint(piece).astype(np.int32)
    piece = np.clip(piece, 0, extent)
    piece = _remove_adjacent_duplicates(piece)
    if is_ring:
        piece = _close_ring(piece)
    return piece


def _process_geometry(geom: Mapping[str, Any], extent: int, simplify: float, clip: bool, quantize: bool) -> dict[str, Any] | None:
    typ = geom.get("type")
    c = geom.get("coordinates")

    if typ == "Point":
        p = _as_xy_array(c).reshape(1, 2)
        if clip and not (0 <= p[0, 0] <= extent and 0 <= p[0, 1] <= extent):
            return None
        if quantize:
            p = np.rint(p).astype(np.int32)
        p = np.clip(p, 0, extent)
        return {"type": "Point", "coordinates": p[0].tolist()}

    if typ == "MultiPoint":
        pts = _as_xy_array(c).reshape(-1, 2)
        if clip:
            m = ((pts[:, 0] >= 0) & (pts[:, 0] <= extent) & (pts[:, 1] >= 0) & (pts[:, 1] <= extent))
            pts = pts[m]
        if len(pts) == 0:
            return None
        if quantize:
            pts = np.rint(pts).astype(np.int32)
        return {"type": "MultiPoint", "coordinates": np.clip(pts, 0, extent).tolist()}

    if typ == "LineString":
        pts = _as_xy_array(c).reshape(-1, 2)
        pieces = _clip_line(pts, float(extent)) if clip else [pts]
        out = [_process_piece(p, is_ring=False, simplify=simplify, quantize=quantize, extent=extent) for p in pieces]
        out = [p for p in out if len(p) >= 2]
        if not out:
            return None
        if len(out) == 1:
            return {"type": "LineString", "coordinates": out[0].tolist()}
        return {"type": "MultiLineString", "coordinates": [p.tolist() for p in out]}

    if typ == "MultiLineString":
        lines = []
        for line in c:
            pts = _as_xy_array(line).reshape(-1, 2)
            pieces = _clip_line(pts, float(extent)) if clip else [pts]
            for p in pieces:
                p = _process_piece(p, is_ring=False, simplify=simplify, quantize=quantize, extent=extent)
                if len(p) >= 2:
                    lines.append(p.tolist())
        return None if not lines else {"type": "MultiLineString", "coordinates": lines}

    if typ == "Polygon":
        rings = []
        for ring in c:
            pts = _as_xy_array(ring).reshape(-1, 2)
            p = _clip_ring(pts, float(extent)) if clip else _close_ring(pts)
            p = _process_piece(p, is_ring=True, simplify=simplify, quantize=quantize, extent=extent)
            if len(p) >= 4:
                rings.append(p.tolist())
        return None if not rings else {"type": "Polygon", "coordinates": rings}

    if typ == "MultiPolygon":
        polygons = []
        for poly in c:
            rings = []
            for ring in poly:
                pts = _as_xy_array(ring).reshape(-1, 2)
                p = _clip_ring(pts, float(extent)) if clip else _close_ring(pts)
                p = _process_piece(p, is_ring=True, simplify=simplify, quantize=quantize, extent=extent)
                if len(p) >= 4:
                    rings.append(p.tolist())
            if rings:
                polygons.append(rings)
        return None if not polygons else {"type": "MultiPolygon", "coordinates": polygons}

    raise ValueError(f"unsupported geometry type {typ!r}")


# ==========================================================
# High-Throughput Preprocessing Facade
# ==========================================================

def _validated_bbox(value: Any) -> tuple[float, float, float, float]:
    b = np.asarray(value, dtype=np.float64)
    if b.shape != (4,) or not np.all(np.isfinite(b)):
        raise ValueError("bounds must be (xmin, ymin, xmax, ymax)")
    xmin, ymin, xmax, ymax = map(float, b)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds require xmax > xmin and ymax > ymin")
    return xmin, ymin, xmax, ymax


def _infer_non_degenerate_bounds(fc: Mapping[str, Any]) -> tuple[float, float, float, float]:
    arrays = coordinate_arrays(fc)
    if not arrays:
        raise ValueError("cannot infer bounds from empty geometry")
    xy = np.concatenate(arrays, axis=0)
    xmin, ymin = xy.min(axis=0)
    xmax, ymax = xy.max(axis=0)
    ref = max(abs(xmin), abs(ymin), abs(xmax), abs(ymax), xmax - xmin, ymax - ymin, 1.0)
    pad = max(float(ref) * 1e-6, 1e-9)
    if xmax <= xmin:
        xmin -= pad; xmax += pad
    if ymax <= ymin:
        ymin -= pad; ymax += pad
    return float(xmin), float(ymin), float(xmax), float(ymax)


def preprocess(
    data: Any,
    bounds: Any = None,
    *,
    extent: int = EXTENT,
    geometry_type: str | Sequence[str] | None = None,
    simplify: float = 0.0,
    clip: bool = True,
    quantize: bool = True,
) -> list[dict[str, Any]]:
    """Convert source geometry into one or more tile-coordinate FeatureCollections.

    ``bounds`` can be one bbox shared by all features or ``(N,4)`` with one
    bbox per feature.  Features with identical bounds are grouped together.
    """
    extent = int(extent)
    if extent <= 0:
        raise ValueError("extent must be positive")
    simplify = float(simplify)
    if simplify < 0:
        raise ValueError("simplify must be >= 0")

    fc = to_feature_collection(data, geometry_type=geometry_type)
    features = list(fc.get("features", []) or [])
    if not features:
        return []
    n = len(features)

    if bounds is None:
        one = _infer_non_degenerate_bounds(fc)
        per_bounds = np.repeat(np.asarray(one, dtype=np.float64)[None, :], n, axis=0)
    else:
        b = np.asarray(bounds, dtype=np.float64)
        if b.shape == (4,):
            one = _validated_bbox(b)
            per_bounds = np.repeat(np.asarray(one, dtype=np.float64)[None, :], n, axis=0)
        elif b.shape == (n, 4):
            per_bounds = np.vstack([_validated_bbox(row) for row in b])
        else:
            raise ValueError(f"bounds must have shape (4,) or ({n},4), got {b.shape}")

    # Transform feature-by-feature into tile coordinates.  The numeric
    # source_to_tile function itself is fully vectorized and also supports
    # batched bounds.  Geometry topology is preserved here rather than being
    # flattened into a custom container.
    tile_features: list[dict[str, Any] | None] = []
    for i, feat in enumerate(features):
        geom = feat.get("geometry")
        if geom is None:
            tile_features.append(None)
            continue
        tile_geom = _transform_geojson_geometry(
            geom,
            lambda a, b=per_bounds[i]: source_to_tile(a, b, extent),
        )
        processed = _process_geometry(tile_geom, extent, simplify, clip, quantize)
        if processed is None:
            tile_features.append(None)
            continue
        tile_features.append({
            "type": "Feature",
            "properties": dict(feat.get("properties") or {}),
            "geometry": processed,
        })

    unique_bounds, inverse = np.unique(per_bounds, axis=0, return_inverse=True)
    outputs = []
    for group_i, group_bounds in enumerate(unique_bounds):
        group_features = [
            tile_features[i]
            for i in np.flatnonzero(inverse == group_i)
            if tile_features[i] is not None
        ]
        outputs.append({
            "type": "FeatureCollection",
            "coordinate_space": "tile",
            "extent": extent,
            "source_bounds": [float(v) for v in group_bounds],
            "features": group_features,
        })
    return outputs


__all__ = [
    "EXTENT",
    "VECTOR_PREPROCESS_API",
    "source_to_tile",
    "tile_to_source",
    "geographic_to_tile",
    "tile_to_geographic",
    "source_geometry_to_tile",
    "tile_geometry_to_source",
    "to_feature_collection",
    "coordinate_arrays",
    "douglas_peucker",
    "preprocess",
]
