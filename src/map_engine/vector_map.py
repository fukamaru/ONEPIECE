from __future__ import annotations

import os
import numpy as np
from PIL import Image
from pyproj import CRS, Transformer
from typing import Any, Mapping, Sequence

from process import EXTENT, _lonlat_to_tile_custom
from tile_system import XYZTileSystem, MapTileSystem
from vector_render import VectorLayer, VectorRenderer, LineStyle, FillStyle, CircleStyle
from vector_preprocess import EXTENT, coordinate_arrays, preprocess, to_feature_collection

_DEFAULT_SOURCE = "_default"
_DEFAULT_SOURCE_NAME = "__vector_map_source__"
_VALID_SPACES = {"geographic", "source", "tile", "image"}

# =====================================================================
# Input / CRS Inspection
# =====================================================================

def _geojson_crs(value: Mapping[str, Any]) -> Any:
    crs = value.get("crs")
    if crs is None:
        return None
    if isinstance(crs, str):
        return crs
    if isinstance(crs, Mapping):
        props = crs.get("properties") or {}
        return props.get("name") or props.get("href") or crs.get("name")
    return crs


def _parse_crs(value: Any):
    if value is None or (isinstance(value, str) and value.lower() == "auto"):
        return None
    if CRS is None:
        raise ImportError("CRS parsing/conversion requires pyproj")
    try:
        return CRS.from_user_input(value)
    except Exception as exc:
        raise ValueError(f"cannot parse CRS {value!r}") from exc


def _metadata(data: Any) -> tuple[Any, str | None]:
    """Read explicit CRS/coordinate-space metadata without magic adapters."""
    crs_values: list[Any] = []
    space_values: list[str] = []

    # GeoPandas exposes CRS directly as .crs.  Reading that attribute is
    # explicit and does not change the geometry representation.
    obj_crs = getattr(data, "crs", None)
    if obj_crs is not None:
        crs_values.append(obj_crs)

    def visit_mapping(obj: Mapping[str, Any]) -> None:
        c = _geojson_crs(obj)
        if c is not None:
            crs_values.append(c)
        for key in ("coordinate_space", "input_space"):
            v = obj.get(key)
            if isinstance(v, str) and v.lower() in _VALID_SPACES:
                space_values.append(v.lower())
        typ = obj.get("type")
        if typ == "FeatureCollection":
            for feat in obj.get("features", []) or []:
                if isinstance(feat, Mapping):
                    visit_mapping(feat)
        elif typ == "Feature":
            geom = obj.get("geometry")
            if isinstance(geom, Mapping):
                visit_mapping(geom)

    if isinstance(data, Mapping):
        visit_mapping(data)

    parsed = [_parse_crs(v) for v in crs_values]
    parsed = [v for v in parsed if v is not None]
    metadata_crs = parsed[0] if parsed else None
    for value in parsed[1:]:
        if value != metadata_crs:
            raise ValueError("conflicting CRS metadata in input")

    spaces = set(space_values)
    if len(spaces) > 1:
        raise ValueError(f"conflicting coordinate_space metadata: {sorted(spaces)}")
    metadata_space = next(iter(spaces)) if spaces else None
    return metadata_crs, metadata_space


def _stats(data: Any, geometry_type: Any = None) -> dict[str, Any]:
    arrays = [a for a in coordinate_arrays(data, geometry_type=geometry_type) if a.size]
    if not arrays:
        return {"count": 0, "min_x": None, "min_y": None, "max_x": None, "max_y": None}
    xy = np.concatenate(arrays, axis=0)
    xy = xy[np.isfinite(xy).all(axis=1)]
    if len(xy) == 0:
        raise ValueError("all coordinates are NaN/Inf")
    return {
        "count": int(len(xy)),
        "min_x": float(xy[:, 0].min()),
        "min_y": float(xy[:, 1].min()),
        "max_x": float(xy[:, 0].max()),
        "max_y": float(xy[:, 1].max()),
    }


def _is_lonlat(stats: Mapping[str, Any]) -> bool:
    return bool(stats.get("count")) and (
        -180 <= stats["min_x"] <= 180
        and -180 <= stats["max_x"] <= 180
        and -90 <= stats["min_y"] <= 90
        and -90 <= stats["max_y"] <= 90
    )


def _fits_tile(stats: Mapping[str, Any], extent: float) -> bool:
    return bool(stats.get("count")) and (
        0 <= stats["min_x"] <= extent
        and 0 <= stats["max_x"] <= extent
        and 0 <= stats["min_y"] <= extent
        and 0 <= stats["max_y"] <= extent
    )


def _bounds_array(bounds: Any) -> np.ndarray | None:
    if bounds is None:
        return None
    b = np.asarray(bounds, dtype=np.float64)
    if b.shape == (4,):
        b = b.reshape(1, 4)
    elif not (b.ndim == 2 and b.shape[1] == 4):
        raise ValueError(f"bounds must be (4,) or (N,4), got {b.shape}")
    if not np.all(np.isfinite(b)):
        raise ValueError("bounds contain NaN/Inf")
    if np.any(b[:, 2] <= b[:, 0]) or np.any(b[:, 3] <= b[:, 1]):
        raise ValueError("bounds require xmax > xmin and ymax > ymin")
    return b


def _bounds_look_lonlat(bounds: Any) -> bool:
    b = _bounds_array(bounds)
    return b is not None and bool(
        np.all((-180 <= b[:, 0]) & (b[:, 2] <= 180))
        and np.all((-90 <= b[:, 1]) & (b[:, 3] <= 90))
    )


def inspect_input(
    data: Any,
    bounds: Any = None,
    *,
    extent: int = EXTENT,
    input_space: str = "auto",
    crs: Any = "auto",
    bounds_crs: Any = "auto",
    geometry_type: Any = None,
) -> dict[str, Any]:
    """Inspect one feature/geometry/batch before preprocessing or rendering.

    Priority:
      1. explicit input_space / crs;
      2. explicit metadata (.crs, GeoJSON crs, coordinate_space);
      3. coordinate ranges.

    Numeric ranges can distinguish lon/lat from obvious tile/projected ranges,
    but cannot uniquely infer an arbitrary projected EPSG code.
    """
    mode = str(input_space).lower()
    if mode not in {"auto", *_VALID_SPACES}:
        raise ValueError("input_space must be auto/source/tile/image")

    metadata_crs, metadata_space = _metadata(data)
    explicit_crs = _parse_crs(crs)
    source_crs = explicit_crs or metadata_crs
    stats = _stats(data, geometry_type=geometry_type)

    if mode != "auto":
        space, reason, confidence = mode, "input_space explicitly supplied", "high"
    elif metadata_space:
        space, reason, confidence = metadata_space, "coordinate_space metadata", "high"
    elif source_crs is not None:
        space, reason, confidence = "source", "CRS metadata/parameter identifies source coordinates", "high"
    elif bounds is not None:
        b = _bounds_array(bounds)
        canonical_tile = b is not None and len(b) == 1 and np.allclose(b[0], [0, 0, extent, extent])
        if canonical_tile and _fits_tile(stats, extent):
            space, reason, confidence = "tile", "bounds equal [0,extent]^2", "high"
        else:
            space, reason, confidence = "source", "source bounds supplied", "high"
    elif _is_lonlat(stats):
        space, reason, confidence = "source", "coordinates fit longitude/latitude ranges", "medium"
    elif _fits_tile(stats, extent):
        space, reason, confidence = "tile", "coordinates fit [0,extent] but not lon/lat", "high"
    else:
        space, reason, confidence = "source", "projected/local numeric range", "medium"

    ambiguous = bool(_is_lonlat(stats) and _fits_tile(stats, extent) and source_crs is None and metadata_space is None)
    if space == "source" and source_crs is None and _is_lonlat(stats) and CRS is not None:
        source_crs = CRS.from_epsg(4326)

    explicit_bounds_crs = _parse_crs(bounds_crs)
    resolved_bounds_crs = explicit_bounds_crs
    if resolved_bounds_crs is None and bounds is not None:
        if _bounds_look_lonlat(bounds) and source_crs is not None and not source_crs.is_geographic and CRS is not None:
            resolved_bounds_crs = CRS.from_epsg(4326)
        else:
            resolved_bounds_crs = source_crs

    return {
        "input_space": space,
        "source_crs": None if source_crs is None else source_crs.to_string(),
        "bounds_crs": None if resolved_bounds_crs is None else resolved_bounds_crs.to_string(),
        "metadata_space": metadata_space,
        "metadata_crs": None if metadata_crs is None else metadata_crs.to_string(),
        "coordinate_range": stats,
        "confidence": confidence,
        "reason": reason,
        "ambiguous_with_tile": ambiguous,
    }

# =====================================================================
# CRS Transform Helpers
# =====================================================================

def _map_coordinate_structure(coords: Any, transformer: Any) -> Any:
    try:
        arr = np.asarray(coords, dtype=np.float64)
    except (TypeError, ValueError):
        arr = None
    if arr is not None and arr.ndim >= 1 and arr.shape[-1] >= 2:
        out = arr[..., :2].copy()
        x, y = transformer.transform(out[..., 0], out[..., 1])
        out[..., 0] = x
        out[..., 1] = y
        return out.tolist()
    if isinstance(coords, Sequence) and not isinstance(coords, (str, bytes)):
        return [_map_coordinate_structure(c, transformer) for c in coords]
    raise TypeError("invalid coordinate structure")


def _transform_fc(fc: Mapping[str, Any], transformer: Any) -> dict[str, Any]:
    features = []
    for feat in fc.get("features", []) or []:
        out_feat = dict(feat)
        geom = feat.get("geometry")
        if geom is not None:
            out_geom = dict(geom)
            out_geom["coordinates"] = _map_coordinate_structure(geom.get("coordinates"), transformer)
            out_feat["geometry"] = out_geom
        features.append(out_feat)
    return {"type": "FeatureCollection", "features": features}


def _transform_bounds(bounds: Any, src: Any, dst: Any) -> Any:
    if bounds is None or src is None or dst is None or src == dst:
        return bounds
    if Transformer is None:
        raise ImportError("CRS conversion requires pyproj")
    b = _bounds_array(bounds)
    tr = Transformer.from_crs(src, dst, always_xy=True)
    out = [tr.transform_bounds(*map(float, row), densify_pts=21) for row in b]
    return tuple(map(float, out[0])) if np.asarray(bounds).shape == (4,) else np.asarray(out)


def _infer_bounds(data: Any, geometry_type: Any = None) -> tuple[float, float, float, float]:
    s = _stats(data, geometry_type=geometry_type)
    if not s["count"]:
        raise ValueError("cannot infer bounds from empty geometry")
    xmin, ymin, xmax, ymax = s["min_x"], s["min_y"], s["max_x"], s["max_y"]
    ref = max(xmax - xmin, ymax - ymin, abs(xmin), abs(xmax), abs(ymin), abs(ymax), 1.0)
    pad = max(ref * 1e-6, 1e-9)
    if xmax <= xmin:
        xmin -= pad; xmax += pad
    if ymax <= ymin:
        ymin -= pad; ymax += pad
    return float(xmin), float(ymin), float(xmax), float(ymax)


def _prepare_source(
    data: Any,
    bounds: Any,
    *,
    extent: int,
    input_space: str,
    crs: Any,
    bounds_crs: Any,
    target_crs: Any,
    geometry_type: Any,
):
    info = inspect_input(
        data, bounds,
        extent=extent,
        input_space=input_space,
        crs=crs,
        bounds_crs=bounds_crs,
        geometry_type=geometry_type,
    )
    if info["input_space"] != "source":
        return data, bounds, info

    source_crs = _parse_crs(info["source_crs"])
    bcrs = _parse_crs(info["bounds_crs"])
    target = _parse_crs(target_crs)
    working = target or bcrs or source_crs

    work_data: Any = data
    work_bounds = bounds
    if source_crs is not None and working is not None and source_crs != working:
        if Transformer is None:
            raise ImportError("CRS conversion requires pyproj")
        fc = to_feature_collection(data, geometry_type=geometry_type)
        tr = Transformer.from_crs(source_crs, working, always_xy=True)
        work_data = _transform_fc(fc, tr)
        # Already normalized, so geometry_type no longer needs to guide ndarray parsing.
        geometry_type = None

    if work_bounds is not None and bcrs is not None and working is not None and bcrs != working:
        work_bounds = _transform_bounds(work_bounds, bcrs, working)

    if work_bounds is None:
        work_bounds = _infer_bounds(work_data, geometry_type=geometry_type)
    return work_data, work_bounds, info

# =====================================================================
# Style Normalization
# =====================================================================

def _pick(mapping: Mapping[str, Any], *names: str, default: Any = None) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _line_style(paint: Any) -> LineStyle:
    if isinstance(paint, LineStyle):
        return paint
    if paint is None:
        paint = {}
    if not isinstance(paint, Mapping):
        raise TypeError("line paint must be a mapping/dict or LineStyle")
    return LineStyle(
        color=_pick(paint, "color", "line-color", default="#2864FF"),
        width=float(_pick(paint, "width", "line-width", default=2.0)),
        opacity=float(_pick(paint, "opacity", "line-opacity", default=1.0)),
        join=str(_pick(paint, "join", "line-join", default="miter")),
        cap=str(_pick(paint, "cap", "line-cap", default="butt")),
        miter_limit=float(
            _pick(paint, "miter_limit", "miter-limit", "line-miter-limit", default=2.0)
        ),
        round_segments=int(
            _pick(paint, "round_segments", "round-segments", default=12)
        ),
    )

def _fill_style(paint: Any) -> FillStyle:
    if isinstance(paint, FillStyle):
        return paint
    if paint is None:
        paint = {}
    if not isinstance(paint, Mapping):
        raise TypeError("fill paint must be a mapping/dict or FillStyle")
    return FillStyle(
        color=_pick(paint, "color", "fill-color", default="#D8E4D4"),
        opacity=float(_pick(paint, "opacity", "fill-opacity", default=1.0)),
        outline_color=_pick(
            paint, "outline_color", "outline-color", "fill-outline-color", default=None
        ),
        outline_width=float(
            _pick(paint, "outline_width", "outline-width", "fill-outline-width", default=0.0)
        ),
    )


def _circle_style(paint: Any) -> CircleStyle:
    if isinstance(paint, CircleStyle):
        return paint
    if paint is None:
        paint = {}
    if not isinstance(paint, Mapping):
        raise TypeError("circle paint must be a mapping/dict or CircleStyle")
    return CircleStyle(
        radius=float(_pick(paint, "radius", "circle-radius", default=4.0)),
        color=_pick(paint, "color", "circle-color", default="#D62728"),
        opacity=float(_pick(paint, "opacity", "circle-opacity", default=1.0)),
        stroke_color=_pick(
            paint, "stroke_color", "stroke-color", "circle-stroke-color", default=None
        ),
        stroke_width=float(
            _pick(paint, "stroke_width", "stroke-width", "circle-stroke-width", default=0.0)
        ),
    )

def _normalize_paint(layer_type: str, paint: Any) -> Any:
    if layer_type == "line":
        return _line_style(paint)
    if layer_type == "fill":
        return _fill_style(paint)
    if layer_type == "circle":
        return _circle_style(paint)
    if layer_type == "icon":
        if paint is None:
            return {}
        if not isinstance(paint, Mapping):
            raise TypeError("icon paint must be a mapping/dict")
        return dict(paint)
    raise ValueError("style layer type must be fill / line / circle / icon")

def _default_style() -> list[dict[str, Any]]:
    """A conservative default style that can display point/line/polygon data."""
    return [
        {
            "type": "fill",
            "paint": {
                "color": "#DDE7DA",
                "outline_color": "#A6B4A2",
                "outline_width": 1.0,
            },
        },
        {
            "type": "line",
            "paint": {
                "color": "#2864FF",
                "width": 2.5,
                "join": "miter",
                "cap": "round",
                "miter_limit": 2.0,
            },
        },
        {
            "type": "circle",
            "paint": {
                "radius": 4.0,
                "color": "#D62728",
            },
        },
    ]

def _normalize_style(style: Any, source_name: str) -> list[VectorLayer]:
    if style is None:
        specs: list[Any] = _default_style()
    elif isinstance(style, Mapping):
        specs = [style]
    elif isinstance(style, Sequence) and not isinstance(style, (str, bytes)):
        specs = list(style)
    else:
        raise TypeError("style must be a layer mapping or a sequence of layer mappings")

    layers: list[VectorLayer] = []
    for index, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError(f"style[{index}] must be a mapping/dict")

        layer_type = str(spec.get("type", "")).lower()
        if layer_type not in {"fill", "line", "circle", "icon"}:
            raise ValueError(
                f"style[{index}]['type'] must be fill / line / circle / icon"
            )

        filter_fn = spec.get("filter")
        if filter_fn is not None and not callable(filter_fn):
            raise TypeError(f"style[{index}]['filter'] must be callable or None")

        layers.append(
            VectorLayer(
                id=str(spec.get("id", f"layer-{index}-{layer_type}")),
                type=layer_type,
                source=source_name,
                paint=_normalize_paint(layer_type, spec.get("paint")),
                filter=filter_fn,
                minzoom=float(spec.get("minzoom", 0.0)),
                maxzoom=float(spec.get("maxzoom", 24.0)),
                visible=bool(spec.get("visible", True)),
            )
        )
    return layers


def _pick(mapping: Mapping[str, Any], *names: str, default=None):
    for name in names:
        if name in mapping:
            return mapping[name]
    return default


def _paint(layer_type: str, paint: Any):
    if layer_type == "line":
        if isinstance(paint, LineStyle):
            return paint
        p = {} if paint is None else paint
        return LineStyle(
            color=_pick(p, "color", "line-color", default="#2864FF"),
            width=float(_pick(p, "width", "line-width", default=2.5)),
            opacity=float(_pick(p, "opacity", "line-opacity", default=1.0)),
            join=str(_pick(p, "join", "line-join", default="miter")),
            cap=str(_pick(p, "cap", "line-cap", default="round")),
            miter_limit=float(_pick(p, "miter_limit", "line-miter-limit", default=2.0)),
            round_segments=int(_pick(p, "round_segments", "line-round-segments", default=8)),
        )
    if layer_type == "fill":
        if isinstance(paint, FillStyle):
            return paint
        p = {} if paint is None else paint
        return FillStyle(
            color=_pick(p, "color", "fill-color", default="#DDE7DA"),
            opacity=float(_pick(p, "opacity", "fill-opacity", default=1.0)),
            outline_color=_pick(p, "outline_color", "fill-outline-color", default=None),
            outline_width=float(_pick(p, "outline_width", "fill-outline-width", default=0.0)),
        )
    if layer_type == "circle":
        if isinstance(paint, CircleStyle):
            return paint
        p = {} if paint is None else paint
        return CircleStyle(
            radius=float(_pick(p, "radius", "circle-radius", default=4.0)),
            color=_pick(p, "color", "circle-color", default="#D62728"),
            opacity=float(_pick(p, "opacity", "circle-opacity", default=1.0)),
            stroke_color=_pick(p, "stroke_color", "circle-stroke-color", default=None),
            stroke_width=float(_pick(p, "stroke_width", "circle-stroke-width", default=0.0)),
        )
    if layer_type == "icon":
        if paint is None:
            return {}
        if not isinstance(paint, Mapping):
            raise TypeError("icon paint must be a dict")
        return dict(paint)
    raise ValueError("layer type must be fill/line/circle/icon")


def _layers(style: Any, source: str) -> list[VectorLayer]:
    if style is None:
        specs = [
            {"type": "fill", "paint": {"color": "#DDE7DA"}},
            {"type": "line", "paint": {"color": "#2864FF", "width": 2.5, "join": "miter", "cap": "round"}},
            {"type": "circle", "paint": {"radius": 4, "color": "#D62728"}},
        ]
    elif isinstance(style, Mapping):
        specs = [style]
    else:
        specs = list(style)

    out = []
    for i, spec in enumerate(specs):
        if not isinstance(spec, Mapping):
            raise TypeError("each style layer must be a dict")
        typ = str(spec.get("type", "")).lower()
        filt = spec.get("filter")
        if filt is not None and not callable(filt):
            raise TypeError("style filter must be callable")
        out.append(VectorLayer(
            id=str(spec.get("id", f"layer-{i}-{typ}")),
            type=typ,
            source=source,
            paint=_paint(typ, spec.get("paint")),
            filter=filt,
            minzoom=float(spec.get("minzoom", 0)),
            maxzoom=float(spec.get("maxzoom", 24)),
            visible=bool(spec.get("visible", True)),
        ))
    return out


def _render_source(source_data: Any, *, size, extent, style, background, antialias, zoom):
    width, height = map(int, size)
    renderer = VectorRenderer(
        width=width,
        height=height,
        extent=extent,
        background=background,
        antialias=antialias,
    )
    renderer.add_source(_DEFAULT_SOURCE, source_data)
    for layer in _layers(style, _DEFAULT_SOURCE):
        renderer.add_layer(layer)
    return renderer.render(zoom=zoom)

# =====================================================================
# Input Preparation
# =====================================================================

def _make_tile_geojson() -> Mapping[str, Any] | Sequence[Mapping[str, Any]]:
    pass

def _make_renderer(
    *,
    size: tuple[int, int] = (512, 512),
    extent: int | float | None = 4096.0,
    style: Any = _default_style(),
    background: Any = "white",
    antialias: int = 4,
    source_data: Any = None,
    zoom: float = 0.0,
) -> Image.Image:
    width, height = map(int, size)
    if width <= 0 or height <= 0:
        raise ValueError("size must contain positive width/height")

    renderer = VectorRenderer(
        width=width,
        height=height,
        extent=extent,
        background=background,
        antialias=antialias,
    )
    renderer.add_source(_DEFAULT_SOURCE_NAME, source_data)
    for layer in _normalize_style(style, _DEFAULT_SOURCE_NAME):
        renderer.add_layer(layer)
    return renderer.render(zoom=zoom)

# =====================================================================
# Public Facade
# =====================================================================

def render_timeline() -> tuple[Image.Image]:
    pass
