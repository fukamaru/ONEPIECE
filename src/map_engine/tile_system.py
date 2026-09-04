from __future__ import annotations

import math
import numpy as np
from typing import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (
            self.min_lon, self.min_lat,
            self.max_lon, self.max_lat
        )

    def contains(self, other: "BBox", atol: float = 1e-12) -> bool:
        return (
            self.min_lon <= other.min_lon + atol
            and self.min_lat <= other.min_lat + atol
            and self.max_lon >= other.max_lon + atol
            and self.max_lat >= other.max_lat + atol
        )

@dataclass(frozen=True)
class MapTile:
    z: int
    x: int
    y: int
    quadkey: str
    bbox: BBox

@dataclass(frozen=True)
class MapTileBatch:
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    quadkey: np.ndarray
    bbox: np.ndarray

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(self, idx: int | slice):
        if isinstance(idx, slice):
            return MapTileBatch(
                z=self.z[idx],
                x=self.x[idx],
                y=self.y[idx],
                quadkey=self.quadkey[idx],
                bbox=self.bbox[idx]
            )

        b = self.bbox[idx]
        return MapTile(
            z=int(self.z[idx]),
            x=int(self.x[idx]),
            y=int(self.y[idx]),
            quadkey=self.quadkey[idx].decode(),
            bbox=BBox(
                float(b[0]), float(b[1]),
                float(b[2]), float(b[3])
            )
        )


@dataclass(frozen=True)
class LooseMapTile(MapTile):
    """A canonical tile plus its expanded loose coverage bbox."""
    loose_bbox: BBox


@dataclass(frozen=True)
class LooseMapTileBatch(MapTileBatch):

    loose_bbox: np.ndarray

    def __getitem__(self, index: int | slice):
        if isinstance(index, slice):
            return LooseMapTileBatch(
                z=self.z[index],
                x=self.x[index],
                y=self.y[index],
                quadkey=self.quadkey[index],
                bbox=self.bbox[index],
                loose_bbox=self.loose_bbox[index],
            )

        b = self.bbox[index]
        lb = self.loose_bbox[index]
        return LooseMapTile(
            z=int(self.z[index]),
            x=int(self.x[index]),
            y=int(self.y[index]),
            quadkey=self.quadkey[index].decode(),
            bbox=BBox(
                float(b[0]),
                float(b[1]),
                float(b[2]),
                float(b[3]),
            ),
            loose_bbox=BBox(
                float(lb[0]),
                float(lb[1]),
                float(lb[2]),
                float(lb[3]),
            ),
        )


class BaseTileSystem:
    R = 6378137.0
    MAX_LAT = 85.0511287798066
    HALF_WORLD = math.pi * R
    WORLD_SIZE = 2.0 * HALF_WORLD
    MAX_SUPPORTED_ZOOM = 30

    # tolerance measured in max-zoom tile-index units
    # this is only used to snap values that should mathematically lie exactly on tile boundaries
    _INDEX_SNAP_ATOL = 1e-7

    def __init__(
            self,
            *,
            root_xy: Sequence[float],
            max_zoom: int
    ):
        self._validate_zoom(max_zoom)

        root_xy = np.asarray(root_xy, dtype=np.float64)
        if root_xy.shape != (4,):
            raise ValueError("root_xy must have shape (4,)")

        xmin, ymin, xmax, ymax = map(float, root_xy)
        width = xmax - xmin
        height = ymax - ymin

        if width <= 0 or height <= 0:
            raise ValueError("root_xy must have positive width and height")

        if not math.isclose(width, height, rel_tol=1e-12, abs_tol=1e-9):
            raise ValueError("root_xy must be square")

        self.max_zoom = int(max_zoom)
        self.root_xy = root_xy
        self.root_size = float(width)
        self.tile_size = self.root_size / (1 << self.max_zoom)

        self._root_xmin = xmin
        self._root_ymin = ymin
        self._root_xmax = xmax
        self._root_ymax = ymax

        self._n_max = np.int64(1 << self.max_zoom)
        self._scale_max = self._n_max / self.root_size

        root_bbox_array = self._xy_bbox_to_lonlat(root_xy.reshape(1, 4))[0]
        self.root_bbox = BBox(*map(float, root_bbox_array))

    # ========================================================================
    # Public shared API
    # ========================================================================
    def query_single(
            self,
            bbox: BBox | tuple[float, float, float, float] | Sequence[BBox] | np.ndarray,
            *,
            max_zoom: int | None = None
    ) -> MapTile | MapTileBatch:
        """Return the deepest single tile that fully contains each bbox."""

        query, single = self._prepare_query(bbox)
        query_zoom = self.max_zoom if max_zoom is None else int(max_zoom)
        self._validate_query_zoom(query_zoom)

        result = self._query_single_batch(query, query_zoom)
        return result[0] if single else result

    def get_tile(
            self,
            z: int,
            x: int,
            y: int
    ) -> MapTile:
        
        self._validate_query_zoom(z)

        n = 1 << z
        if not (0 <= x < n and 0 <= y < n):
            raise ValueError(f"invalid tile index for z={z}: x={x}, y={y}")

        size = self.root_size / n
        xmin = self._root_xmin + x * size
        xmax = xmin + size
        ymax = self._root_ymax - y * size
        ymin = ymax - size

        bbox_array = self._xy_bbox_to_lonlat(
            np.array([[xmin, ymin, xmax, ymax]], dtype=np.float64)
        )[0]

        return MapTile(
            z=int(z),
            x=int(x),
            y=int(y),
            quadkey=self.to_quadkey(z, x, y),
            bbox=BBox(*map(float, bbox_array)),
        )

    def point_to_tile(
            self,
            lon: float,
            lat: float,
            z: int
    ) -> MapTile:
        
        self._validate_query_zoom(z)
        self._validate_lonlat(lon, lat)

        px, py = self._lonlat_to_xy(lon, lat)
        xy_tol = max(self.root_size * 1e-12, 1e-7)

        if not (
            self._root_xmin - xy_tol <= px <= self._root_xmax + xy_tol
            and self._root_ymin - xy_tol <= py <= self._root_ymax + xy_tol
        ):
            raise ValueError("point is outside the tile-system root")

        n = 1 << z
        scale = n / self.root_size

        fx = self._snap_to_integer(
            np.asarray([(px - self._root_xmin) * scale], dtype=np.float64)
        )[0]
        fy = self._snap_to_integer(
            np.asarray([(self._root_ymax - py) * scale], dtype=np.float64)
        )[0]

        x = int(math.floor(fx))
        y = int(math.floor(fy))

        x = min(max(x, 0), n - 1)
        y = min(max(y, 0), n - 1)
        return self.get_tile(z, x, y)

    def tile_range(
            self,
            bbox: BBox | tuple[float, float, float, float],
            z: int
    ) -> tuple[int, int, int, int]:
        """Return inclusive x/y tile range: (x0, x1, y0, y1)."""

        self._validate_query_zoom(z)
        query, _ = self._prepare_query(bbox)
        if len(query) != 1:
            raise ValueError("tile_range accepts a single bbox")

        xmin, ymin = self._lonlat_to_xy(query[:, 0], query[:, 1])
        xmax, ymax = self._lonlat_to_xy(query[:, 2], query[:, 3])
        self._validate_query_inside_root(xmin, ymin, xmax, ymax)

        x0, x1, y0, y1 = self._bbox_to_tile_range_batch(
            xmin, ymin, xmax, ymax, z
        )
        return int(x0[0]), int(x1[0]), int(y0[0]), int(y1[0])

    def tile_shape(
            self,
            z: int
    ) -> tuple[int, int]:
        
        self._validate_query_zoom(z)
        n = 1 << z
        return n, n

    def tile_count(
            self,
            z: int
    ) -> int:
        
        self._validate_query_zoom(z)
        return 1 << (2 * z)

    def tile_size_at_zoom(
            self,
            z: int
    ) -> float:
        
        self._validate_query_zoom(z)
        return self.root_size / (1 << z)

    def zoom_level_info(self) -> dict[str, np.ndarray]:
        z = np.arange(self.max_zoom + 1, dtype=np.int64)
        tiles_per_axis = np.left_shift(np.int64(1), z)
        return {
            "zoom": z,
            "tiles_per_axis": tiles_per_axis,
            "tile_count": tiles_per_axis**2,
            "tile_size": self.root_size / tiles_per_axis,
        }

    @staticmethod
    def to_quadkey(
        z: int,
        x: int,
        y: int
    ) -> str:
        
        if z < 0:
            raise ValueError("z must be non-negative")
        if not (0 <= x < (1 << z) and 0 <= y < (1 << z)):
            raise ValueError(f"invalid tile index for z={z}: x={x}, y={y}")

        chars: list[str] = []
        for bit in range(z - 1, -1, -1):
            digit = ((x >> bit) & 1) | (((y >> bit) & 1) << 1)
            chars.append(str(digit))
        return "".join(chars)

    @staticmethod
    def from_quadkey(
        quadkey: str
    ) -> tuple[int, int, int]:
        
        x = 0
        y = 0
        z = len(quadkey)

        for char in quadkey:
            if char not in "0123":
                raise ValueError(f"invalid QuadKey digit: {char!r}")
            digit = ord(char) - ord("0")
            x = (x << 1) | (digit & 1)
            y = (y << 1) | ((digit >> 1) & 1)

        return z, x, y

    # ========================================================================
    # Vectorized query core
    # ========================================================================

    def _query_single_batch(
            self,
            bboxs: np.ndarray,
            max_zoom: int
    ) -> MapTileBatch:
        xmin, ymin = self._lonlat_to_xy(bboxs[:, 0], bboxs[:, 1])
        xmax, ymax = self._lonlat_to_xy(bboxs[:, 2], bboxs[:, 3])

        self._validate_query_inside_root(xmin, ymin, xmax, ymax)

        x0, x1, y0, y1 = self._bbox_to_tile_range_batch(
            xmin, ymin, xmax, ymax, max_zoom
        )

        diff = np.bitwise_or(
            np.bitwise_xor(x0, x1),
            np.bitwise_xor(y0, y1),
        )

        # For positive integers frexp exponent equals bit_length; for 0 it is 0.
        shift = np.frexp(diff.astype(np.float64))[1].astype(np.int64)
        z = max_zoom - shift
        x = np.right_shift(x0, shift)
        y = np.right_shift(y0, shift)

        tile_bboxs = self._tile_bboxs_batch(z, x, y)
        quadkey = self._quadkey_batch(z, x, y)

        return MapTileBatch(
            z=z.astype(np.uint8, copy=False),
            x=x.astype(np.uint32, copy=False),
            y=y.astype(np.uint32, copy=False),
            quadkey=quadkey,
            bbox=tile_bboxs,
        )

    def _bbox_to_tile_range_batch(
            self,
            xmin: np.ndarray,
            ymin: np.ndarray,
            xmax: np.ndarray,
            ymax: np.ndarray,
            z: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        n = np.int64(1 << z)
        scale = n / self.root_size

        fx0 = (xmin - self._root_xmin) * scale
        fx1 = (xmax - self._root_xmin) * scale
        fy0 = (self._root_ymax - ymax) * scale
        fy1 = (self._root_ymax - ymin) * scale

        fx0 = self._snap_to_integer(fx0)
        fx1 = self._snap_to_integer(fx1)
        fy0 = self._snap_to_integer(fy0)
        fy1 = self._snap_to_integer(fy1)

        # BBox upper/right edges are exclusive for tile enumeration.
        x0 = np.floor(fx0).astype(np.int64)
        x1 = (np.ceil(fx1).astype(np.int64) - 1)
        y0 = np.floor(fy0).astype(np.int64)
        y1 = (np.ceil(fy1).astype(np.int64) - 1)

        maximum = n - 1
        np.clip(x0, 0, maximum, out=x0)
        np.clip(x1, 0, maximum, out=x1)
        np.clip(y0, 0, maximum, out=y0)
        np.clip(y1, 0, maximum, out=y1)

        if np.any((x1 < x0) | (y1 < y0)):
            raise RuntimeError("invalid internal tile range")

        return x0, x1, y0, y1

    def _tile_bboxs_batch(
            self,
            z: np.ndarray,
            x: np.ndarray,
            y: np.ndarray
    ) -> np.ndarray:
        
        n = np.left_shift(np.int64(1), z)
        size = self.root_size / n

        xmin = self._root_xmin + x * size
        xmax = xmin + size
        ymax = self._root_ymax - y * size
        ymin = ymax - size

        return self._xy_bbox_to_lonlat(
            np.column_stack((xmin, ymin, xmax, ymax))
        )

    def _quadkey_batch(
            self,
            z: np.ndarray,
            x: np.ndarray,
            y: np.ndarray
    ) -> np.ndarray:
        output = np.empty(len(z), dtype=f"S{self.max_zoom}")

        for zz_raw in np.unique(z):
            zz = int(zz_raw)
            indices = np.flatnonzero(z == zz_raw)

            if zz == 0:
                output[indices] = b""
                continue

            xx = x[indices].astype(np.uint64, copy=False)
            yy = y[indices].astype(np.uint64, copy=False)
            bits = np.arange(zz - 1, -1, -1, dtype=np.uint64)

            digits = (
                ((xx[:, None] >> bits) & 1)
                | (((yy[:, None] >> bits) & 1) << 1)
            ).astype(np.uint8)
            digits += ord("0")

            strings = (
                np.ascontiguousarray(digits)
                .view(f"S{zz}")
                .reshape(-1)
            )
            output[indices] = strings

        return output

    # ========================================================================
    # Web-Mercator conversion
    # ========================================================================

    @classmethod
    def _lonlat_to_xy(
        cls,
        lon: float,
        lat: float
    ) -> tuple[float, float]:
        
        lat = np.clip(lat, -cls.MAX_LAT, cls.MAX_LAT)
        x = cls.R * np.deg2rad(lon)
        lat_rad = np.deg2rad(lat)
        y = cls.R * np.log(np.tan(np.pi / 4.0 + lat_rad / 2.0))
        return x, y

    @classmethod
    def _xy_bbox_to_lonlat(
        cls,
        bbox: np.ndarray
    ) -> np.ndarray:
        
        xmin = bbox[:, 0]
        ymin = bbox[:, 1]
        xmax = bbox[:, 2]
        ymax = bbox[:, 3]

        min_lon = np.clip(np.rad2deg(xmin / cls.R), -180.0, 180.0)
        max_lon = np.clip(np.rad2deg(xmax / cls.R), -180.0, 180.0)
        min_lat = np.clip(
            np.rad2deg(
                2.0 * np.arctan(np.exp(ymin / cls.R)) - np.pi / 2.0
            ),
            -cls.MAX_LAT,
            cls.MAX_LAT,
        )
        max_lat = np.clip(
            np.rad2deg(
                2.0 * np.arctan(np.exp(ymax / cls.R)) - np.pi / 2.0
            ),
            -cls.MAX_LAT,
            cls.MAX_LAT,
        )

        return np.column_stack((min_lon, min_lat, max_lon, max_lat))

    # ========================================================================
    # Input / validation helper
    # ========================================================================

    def _prepare_query(
            self,
            bboxs: BBox | np.ndarray | Sequence[BBox] | Sequence
    ):
        if isinstance(bboxs, BBox):
            array = np.asarray([bboxs.as_tuple()], dtype=np.float64)
            single = True

        elif isinstance(bboxs, np.ndarray):
            array = np.asarray(bboxs, dtype=np.float64)
            if array.ndim == 1:
                if array.shape != (4,):
                    raise ValueError("bbox must have shape (4,) or (N, 4)")
                array = array.reshape(1, 4)
                single = True
            elif array.ndim == 2 and array.shape[1] == 4:
                single = False
            else:
                raise ValueError("bbox must have shape (4,) or (N, 4)")

        elif (
            isinstance(bboxs, Sequence)
            and len(bboxs) > 0
            and isinstance(bboxs[0], BBox)
        ):
            array = np.asarray([b.as_tuple() for b in bboxs], dtype=np.float64)
            single = False

        else:
            array = np.asarray(bboxs, dtype=np.float64)
            if array.ndim == 1:
                if array.shape != (4,):
                    raise ValueError("bbox must have shape (4,) or (N, 4)")
                array = array.reshape(1, 4)
                single = True
            elif array.ndim == 2 and array.shape[1] == 4:
                single = False
            else:
                raise ValueError("bbox must have shape (4,) or (N, 4)")

        self._validate_bbox_array(array)
        return array, single

    @classmethod
    def _validate_bbox_array(
        cls,
        bboxs: np.ndarray
    ) -> None:
        invalid = (
            (bboxs[:, 0] >= bboxs[:, 2])
            | (bboxs[:, 1] >= bboxs[:, 3])
            | (bboxs[:, 0] < -180.0)
            | (bboxs[:, 2] > 180.0)
            | (bboxs[:, 1] < -cls.MAX_LAT)
            | (bboxs[:, 3] > cls.MAX_LAT)
        )

        if np.any(invalid):
            rows = np.flatnonzero(invalid)
            raise ValueError(f"invalid bbox at rows {rows[:10].tolist()}")

    @classmethod
    def _validate_bbox(
        cls,
        bbox: BBox
    ) -> None:
        cls._validate_bbox_array(
            np.asarray([bbox.as_tuple()], dtype=np.float64)
        )

    @classmethod
    def _validate_lonlat(
        cls, 
        lon: float,
        lat: float
    ) -> None:
        if not -180.0 <= lon <= 180.0:
            raise ValueError("longitude must be in [-180, 180]")
        if not -cls.MAX_LAT <= lat <= cls.MAX_LAT:
            raise ValueError(
                f"latitude must be in [-{cls.MAX_LAT}, {cls.MAX_LAT}]"
            )

    @classmethod
    def _validate_zoom(cls, z: int) -> None:
        if not 0 <= int(z) <= cls.MAX_SUPPORTED_ZOOM:
            raise ValueError(
                f"zoom must be in [0, {cls.MAX_SUPPORTED_ZOOM}]"
            )

    def _validate_query_zoom(self, z: int) -> None:
        self._validate_zoom(z)
        if z > self.max_zoom:
            raise ValueError(
                f"zoom {z} exceeds this tile system's max_zoom={self.max_zoom}"
            )

    def _validate_query_inside_root(
            self,
            xmin: np.ndarray,
            ymin: np.ndarray,
            xmax: np.ndarray,
            ymax: np.ndarray
    ) -> None:
        xy_tol = max(self.root_size * 1e-12, 1e-7)
        outside = (
            (xmin < self._root_xmin - xy_tol)
            | (ymin < self._root_ymin - xy_tol)
            | (xmax > self._root_xmax + xy_tol)
            | (ymax > self._root_ymax + xy_tol)
        )

        if np.any(outside):
            rows = np.flatnonzero(outside)
            raise ValueError(
                f"query bbox is outside root at rows {rows[:10].tolist()}"
            )

    def _snap_to_integer(self, values: np.ndarray) -> np.ndarray:
        nearest = np.rint(values)
        ulp_tol = 8.0 * np.abs(np.spacing(values))
        tol = np.maximum(self._INDEX_SNAP_ATOL, ulp_tol)
        return np.where(np.abs(values - nearest) <= tol, nearest, values)


class LooseTileSystem(BaseTileSystem):
    """Adds loose-quadtree queries on top of a canonical quadtree grid.

    Canonical tile geometry and QuadKeys are never changed. ``loose_factor``
    only enlarges the coverage region used to decide whether a bbox can be
    assigned to a tile. A factor of 1.0 is exactly the strict behavior.
    """

    @staticmethod
    def _validate_loose_factor(loose_factor: float) -> float:
        loose_factor = float(loose_factor)
        if not math.isfinite(loose_factor) or loose_factor < 1.0:
            raise ValueError("loose_factor must be finite and >= 1.0")
        return loose_factor

    def query_single_loose(
        self,
        bbox: BBox | tuple[float, float, float, float] | Sequence[BBox] | np.ndarray,
        *,
        loose_factor: float = 1.5,
        max_zoom: int | None = None,
    ) -> LooseMapTile | LooseMapTileBatch:
        """Return the deepest tile whose loose coverage contains each bbox.

        At a given zoom there can be multiple overlapping loose tiles that
        contain the same bbox. The returned tile is chosen deterministically:
        prefer the candidate containing the query center, otherwise choose the
        nearest valid candidate index independently on x/y.
        """

        loose_factor = self._validate_loose_factor(loose_factor)
        query, single = self._prepare_query(bbox)
        query_zoom = self.max_zoom if max_zoom is None else int(max_zoom)
        self._validate_query_zoom(query_zoom)

        result = self._query_single_loose_batch(
            query,
            max_zoom=query_zoom,
            loose_factor=loose_factor,
        )
        return result[0] if single else result

    def get_loose_tile(
        self,
        z: int,
        x: int,
        y: int,
        *,
        loose_factor: float = 1.5,
    ) -> LooseMapTile:
        """Return a canonical tile together with its loose coverage bbox."""

        loose_factor = self._validate_loose_factor(loose_factor)
        tile = self.get_tile(z, x, y)
        loose_bbox_array = self._loose_tile_bboxs_batch(
            np.asarray([z], dtype=np.int64),
            np.asarray([x], dtype=np.int64),
            np.asarray([y], dtype=np.int64),
            loose_factor,
        )[0]
        return LooseMapTile(
            z=tile.z,
            x=tile.x,
            y=tile.y,
            quadkey=tile.quadkey,
            bbox=tile.bbox,
            loose_bbox=BBox(*map(float, loose_bbox_array)),
        )

    def _query_single_loose_batch(
        self,
        bboxs: np.ndarray,
        *,
        max_zoom: int,
        loose_factor: float,
    ) -> LooseMapTileBatch:
        xmin, ymin = self._lonlat_to_xy(bboxs[:, 0], bboxs[:, 1])
        xmax, ymax = self._lonlat_to_xy(bboxs[:, 2], bboxs[:, 3])
        self._validate_query_inside_root(xmin, ymin, xmax, ymax)

        count = len(bboxs)
        if count == 0:
            empty_i64 = np.empty(0, dtype=np.int64)
            empty_bbox = np.empty((0, 4), dtype=np.float64)
            return LooseMapTileBatch(
                z=empty_i64.astype(np.uint8),
                x=empty_i64.astype(np.uint32),
                y=empty_i64.astype(np.uint32),
                quadkey=np.empty(0, dtype=f"S{self.max_zoom}"),
                bbox=empty_bbox.copy(),
                loose_bbox=empty_bbox.copy(),
            )

        # Feasibility is monotone with zoom for loose_factor >= 1:
        # if a child loose tile contains a bbox, its parent loose tile does too.
        # Find the deepest feasible zoom for every query with one vectorized
        # binary search over the batch. z=0 is always feasible for in-root bboxs.
        low = np.zeros(count, dtype=np.int64)
        high = np.full(count, max_zoom + 1, dtype=np.int64)

        while np.any(high - low > 1):
            active = high - low > 1
            mid = (low + high) // 2
            feasible, _, _, _, _ = self._loose_candidates_at_zoom(
                xmin, ymin, xmax, ymax, mid, loose_factor
            )
            low = np.where(active & feasible, mid, low)
            high = np.where(active & ~feasible, mid, high)

        z = low
        feasible, x_lo, x_hi, y_lo, y_hi = self._loose_candidates_at_zoom(
            xmin, ymin, xmax, ymax, z, loose_factor
        )
        if not np.all(feasible):
            raise RuntimeError("internal loose-quadtree search failure")

        n = np.left_shift(np.int64(1), z)
        scale = n / self.root_size

        # Prefer the canonical tile containing the projected query center.
        center_x = ((xmin + xmax) * 0.5 - self._root_xmin) * scale
        center_y = (self._root_ymax - (ymin + ymax) * 0.5) * scale
        center_x = self._snap_to_integer(center_x)
        center_y = self._snap_to_integer(center_y)
        x_center = np.floor(center_x).astype(np.int64)
        y_center = np.floor(center_y).astype(np.int64)
        maximum = n - 1
        x_center = np.minimum(np.maximum(x_center, 0), maximum)
        y_center = np.minimum(np.maximum(y_center, 0), maximum)

        x = np.minimum(np.maximum(x_center, x_lo), x_hi)
        y = np.minimum(np.maximum(y_center, y_lo), y_hi)

        tile_bboxs = self._tile_bboxs_batch(z, x, y)
        loose_bboxs = self._loose_tile_bboxs_batch(
            z, x, y, loose_factor
        )
        quadkey = self._quadkey_batch(z, x, y)

        return LooseMapTileBatch(
            z=z.astype(np.uint8, copy=False),
            x=x.astype(np.uint32, copy=False),
            y=y.astype(np.uint32, copy=False),
            quadkey=quadkey,
            bbox=tile_bboxs,
            loose_bbox=loose_bboxs,
        )

    def _loose_candidates_at_zoom(
        self,
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
        z: np.ndarray,
        loose_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.int64)
        n = np.left_shift(np.int64(1), z)
        scale = n / self.root_size
        expansion = (loose_factor - 1.0) * 0.5

        fx0 = (xmin - self._root_xmin) * scale
        fx1 = (xmax - self._root_xmin) * scale
        fy0 = (self._root_ymax - ymax) * scale
        fy1 = (self._root_ymax - ymin) * scale

        # Tile i has loose interval [i-expansion, i+1+expansion]
        # in tile-coordinate space. Solve the containment inequalities for i.
        x_lo_f = self._snap_to_integer(fx1 - 1.0 - expansion)
        x_hi_f = self._snap_to_integer(fx0 + expansion)
        y_lo_f = self._snap_to_integer(fy1 - 1.0 - expansion)
        y_hi_f = self._snap_to_integer(fy0 + expansion)

        x_lo = np.ceil(x_lo_f).astype(np.int64)
        x_hi = np.floor(x_hi_f).astype(np.int64)
        y_lo = np.ceil(y_lo_f).astype(np.int64)
        y_hi = np.floor(y_hi_f).astype(np.int64)

        minimum = np.zeros_like(n)
        maximum = n - 1
        x_lo = np.maximum(x_lo, minimum)
        x_hi = np.minimum(x_hi, maximum)
        y_lo = np.maximum(y_lo, minimum)
        y_hi = np.minimum(y_hi, maximum)

        feasible = (x_lo <= x_hi) & (y_lo <= y_hi)
        return feasible, x_lo, x_hi, y_lo, y_hi

    def _loose_tile_bboxs_batch(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        loose_factor: float,
    ) -> np.ndarray:
        z = np.asarray(z, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)

        n = np.left_shift(np.int64(1), z)
        size = self.root_size / n
        pad = (loose_factor - 1.0) * 0.5 * size

        xmin = self._root_xmin + x * size - pad
        xmax = self._root_xmin + (x + 1) * size + pad
        ymax = self._root_ymax - y * size + pad
        ymin = self._root_ymax - (y + 1) * size - pad

        # Geographic output is limited to the valid Web-Mercator world.
        xmin = np.clip(xmin, -self.HALF_WORLD, self.HALF_WORLD)
        xmax = np.clip(xmax, -self.HALF_WORLD, self.HALF_WORLD)
        ymin = np.clip(ymin, -self.HALF_WORLD, self.HALF_WORLD)
        ymax = np.clip(ymax, -self.HALF_WORLD, self.HALF_WORLD)

        return self._xy_bbox_to_lonlat(
            np.column_stack((xmin, ymin, xmax, ymax))
        )


class LooseQueryMixin:
    """Mixin adding loose-quadtree queries to a canonical tile system.

    Canonical tile geometry and QuadKeys are never changed. ``loose_factor``
    only enlarges the coverage region used to decide whether a bbox can be
    assigned to a tile. A factor of 1.0 is exactly the strict behavior.
    """

    @staticmethod
    def _validate_loose_factor(loose_factor: float) -> float:
        loose_factor = float(loose_factor)
        if not math.isfinite(loose_factor) or loose_factor < 1.0:
            raise ValueError("loose_factor must be finite and >= 1.0")
        return loose_factor

    def query_single_loose(
        self,
        bbox: BBox | tuple[float, float, float, float] | Sequence[BBox] | np.ndarray,
        *,
        loose_factor: float = 1.5,
        max_zoom: int | None = None,
    ) -> LooseMapTile | LooseMapTileBatch:
        """Return the deepest tile whose loose coverage contains each bbox.

        At a given zoom there can be multiple overlapping loose tiles that
        contain the same bbox. The returned tile is chosen deterministically:
        prefer the candidate containing the query center, otherwise choose the
        nearest valid candidate index independently on x/y.
        """

        loose_factor = self._validate_loose_factor(loose_factor)
        query, single = self._prepare_query(bbox)
        query_zoom = self.max_zoom if max_zoom is None else int(max_zoom)
        self._validate_query_zoom(query_zoom)

        result = self._query_single_loose_batch(
            query,
            max_zoom=query_zoom,
            loose_factor=loose_factor,
        )
        return result[0] if single else result

    def get_loose_tile(
        self,
        z: int,
        x: int,
        y: int,
        *,
        loose_factor: float = 1.5,
    ) -> LooseMapTile:
        """Return a canonical tile together with its loose coverage bbox."""

        loose_factor = self._validate_loose_factor(loose_factor)
        tile = self.get_tile(z, x, y)
        loose_bbox_array = self._loose_tile_bboxs_batch(
            np.asarray([z], dtype=np.int64),
            np.asarray([x], dtype=np.int64),
            np.asarray([y], dtype=np.int64),
            loose_factor,
        )[0]
        return LooseMapTile(
            z=tile.z,
            x=tile.x,
            y=tile.y,
            quadkey=tile.quadkey,
            bbox=tile.bbox,
            loose_bbox=BBox(*map(float, loose_bbox_array)),
        )

    def _query_single_loose_batch(
        self,
        bboxs: np.ndarray,
        *,
        max_zoom: int,
        loose_factor: float,
    ) -> LooseMapTileBatch:
        xmin, ymin = self._lonlat_to_xy(bboxs[:, 0], bboxs[:, 1])
        xmax, ymax = self._lonlat_to_xy(bboxs[:, 2], bboxs[:, 3])
        self._validate_query_inside_root(xmin, ymin, xmax, ymax)

        count = len(bboxs)
        if count == 0:
            empty_i64 = np.empty(0, dtype=np.int64)
            empty_bbox = np.empty((0, 4), dtype=np.float64)
            return LooseMapTileBatch(
                z=empty_i64.astype(np.uint8),
                x=empty_i64.astype(np.uint32),
                y=empty_i64.astype(np.uint32),
                quadkey=np.empty(0, dtype=f"S{self.max_zoom}"),
                bbox=empty_bbox.copy(),
                loose_bbox=empty_bbox.copy(),
            )

        # Feasibility is monotone with zoom for loose_factor >= 1:
        # if a child loose tile contains a bbox, its parent loose tile does too.
        # Find the deepest feasible zoom for every query with one vectorized
        # binary search over the batch. z=0 is always feasible for in-root bboxs.
        low = np.zeros(count, dtype=np.int64)
        high = np.full(count, max_zoom + 1, dtype=np.int64)

        while np.any(high - low > 1):
            active = high - low > 1
            mid = (low + high) // 2
            feasible, _, _, _, _ = self._loose_candidates_at_zoom(
                xmin, ymin, xmax, ymax, mid, loose_factor
            )
            low = np.where(active & feasible, mid, low)
            high = np.where(active & ~feasible, mid, high)

        z = low
        feasible, x_lo, x_hi, y_lo, y_hi = self._loose_candidates_at_zoom(
            xmin, ymin, xmax, ymax, z, loose_factor
        )
        if not np.all(feasible):
            raise RuntimeError("internal loose-quadtree search failure")

        n = np.left_shift(np.int64(1), z)
        scale = n / self.root_size

        # Prefer the canonical tile containing the projected query center.
        center_x = ((xmin + xmax) * 0.5 - self._root_xmin) * scale
        center_y = (self._root_ymax - (ymin + ymax) * 0.5) * scale
        center_x = self._snap_to_integer(center_x)
        center_y = self._snap_to_integer(center_y)
        x_center = np.floor(center_x).astype(np.int64)
        y_center = np.floor(center_y).astype(np.int64)
        maximum = n - 1
        x_center = np.minimum(np.maximum(x_center, 0), maximum)
        y_center = np.minimum(np.maximum(y_center, 0), maximum)

        x = np.minimum(np.maximum(x_center, x_lo), x_hi)
        y = np.minimum(np.maximum(y_center, y_lo), y_hi)

        tile_bboxs = self._tile_bboxs_batch(z, x, y)
        loose_bboxs = self._loose_tile_bboxs_batch(
            z, x, y, loose_factor
        )
        quadkey = self._quadkey_batch(z, x, y)

        return LooseMapTileBatch(
            z=z.astype(np.uint8, copy=False),
            x=x.astype(np.uint32, copy=False),
            y=y.astype(np.uint32, copy=False),
            quadkey=quadkey,
            bbox=tile_bboxs,
            loose_bbox=loose_bboxs,
        )

    def _loose_candidates_at_zoom(
        self,
        xmin: np.ndarray,
        ymin: np.ndarray,
        xmax: np.ndarray,
        ymax: np.ndarray,
        z: np.ndarray,
        loose_factor: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        z = np.asarray(z, dtype=np.int64)
        n = np.left_shift(np.int64(1), z)
        scale = n / self.root_size
        expansion = (loose_factor - 1.0) * 0.5

        fx0 = (xmin - self._root_xmin) * scale
        fx1 = (xmax - self._root_xmin) * scale
        fy0 = (self._root_ymax - ymax) * scale
        fy1 = (self._root_ymax - ymin) * scale

        # Tile i has loose interval [i-expansion, i+1+expansion]
        # in tile-coordinate space. Solve the containment inequalities for i.
        x_lo_f = self._snap_to_integer(fx1 - 1.0 - expansion)
        x_hi_f = self._snap_to_integer(fx0 + expansion)
        y_lo_f = self._snap_to_integer(fy1 - 1.0 - expansion)
        y_hi_f = self._snap_to_integer(fy0 + expansion)

        x_lo = np.ceil(x_lo_f).astype(np.int64)
        x_hi = np.floor(x_hi_f).astype(np.int64)
        y_lo = np.ceil(y_lo_f).astype(np.int64)
        y_hi = np.floor(y_hi_f).astype(np.int64)

        minimum = np.zeros_like(n)
        maximum = n - 1
        x_lo = np.maximum(x_lo, minimum)
        x_hi = np.minimum(x_hi, maximum)
        y_lo = np.maximum(y_lo, minimum)
        y_hi = np.minimum(y_hi, maximum)

        feasible = (x_lo <= x_hi) & (y_lo <= y_hi)
        return feasible, x_lo, x_hi, y_lo, y_hi

    def _loose_tile_bboxs_batch(
        self,
        z: np.ndarray,
        x: np.ndarray,
        y: np.ndarray,
        loose_factor: float,
    ) -> np.ndarray:
        z = np.asarray(z, dtype=np.int64)
        x = np.asarray(x, dtype=np.int64)
        y = np.asarray(y, dtype=np.int64)

        n = np.left_shift(np.int64(1), z)
        size = self.root_size / n
        pad = (loose_factor - 1.0) * 0.5 * size

        xmin = self._root_xmin + x * size - pad
        xmax = self._root_xmin + (x + 1) * size + pad
        ymax = self._root_ymax - y * size + pad
        ymin = self._root_ymax - (y + 1) * size - pad

        # Geographic output is limited to the valid Web-Mercator world.
        xmin = np.clip(xmin, -self.HALF_WORLD, self.HALF_WORLD)
        xmax = np.clip(xmax, -self.HALF_WORLD, self.HALF_WORLD)
        ymin = np.clip(ymin, -self.HALF_WORLD, self.HALF_WORLD)
        ymax = np.clip(ymax, -self.HALF_WORLD, self.HALF_WORLD)

        return self._xy_bbox_to_lonlat(
            np.column_stack((xmin, ymin, xmax, ymax))
        )


class MapTileSystem(
    LooseQueryMixin,
    BaseTileSystem
):
    """Custon tile system."""
    def __init__(
            self,
            bbox: BBox | tuple[float, float, float, float],
            max_zoom: int | None = None,
            min_tile_size: float | None = None
    ) -> None:
        if not isinstance(bbox, BBox):
            bbox = BBox(*bbox)
        self._validate_bbox(bbox)

        if max_zoom is None and min_tile_size is None:
            raise ValueError("max_zoom and min_tile_size cannot both be None")
        if max_zoom is not None:
            self._validate_zoom(max_zoom)
        if min_tile_size is not None and min_tile_size <= 0:
            raise ValueError("min_tile_size must be positive")

        self.anchor_bbox = bbox
        self.requested_max_zoom = max_zoom
        self.requested_min_tile_size = min_tile_size

        xmin, ymin = self._lonlat_to_xy(bbox.min_lon, bbox.min_lat)
        xmax, ymax = self._lonlat_to_xy(bbox.max_lon, bbox.max_lat)

        anchor_size = max(float(xmax - xmin), float(ymax - ymin))
        effective_zoom, root_size = self._resolve_grid(
            anchor_size=anchor_size,
            max_zoom=max_zoom,
            min_tile_size=min_tile_size,
        )

        root_xy = self._build_root(
            xmin=float(xmin),
            ymin=float(ymin),
            xmax=float(xmax),
            ymax=float(ymax),
            size=root_size,
        )

        super().__init__(root_xy=root_xy, max_zoom=effective_zoom)

    @classmethod
    def _resolve_grid(
        cls,
        anchor_size: float,
        max_zoom: int | None = None,
        min_tile_size: float | None = None
    ) -> tuple[int, float]:
        if min_tile_size is None:
            return int(max_zoom), anchor_size

        if anchor_size <= min_tile_size:
            required_zoom = 0
        else:
            required_zoom = math.ceil(math.log2(anchor_size / min_tile_size))

        if required_zoom > cls.MAX_SUPPORTED_ZOOM and max_zoom is None:
            raise ValueError(
                f"min_tile_size requires zoom {required_zoom}, exceeding "
                f"supported maximum {cls.MAX_SUPPORTED_ZOOM}"
            )

        zoom = (
            required_zoom
            if max_zoom is None
            else min(required_zoom, int(max_zoom))
        )

        if zoom == required_zoom:
            root_size = min_tile_size * (1 << zoom)
        else:
            root_size = anchor_size

        if root_size > cls.WORLD_SIZE:
            raise ValueError("adjusted custom root exceeds Web-Mercator world")

        return int(zoom), float(root_size)

    @classmethod
    def _build_root(
        cls,
        xmin: float, 
        ymin: float,
        xmax: float, 
        ymax: float,
        size: float
    ) -> np.ndarray:
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        root_xmin = cx - size * 0.5
        root_ymin = cy - size * 0.5

        x_low = max(xmax - size, -cls.HALF_WORLD)
        x_high = min(xmin, cls.HALF_WORLD - size)
        y_low = max(ymax - size, -cls.HALF_WORLD)
        y_high = min(ymin, cls.HALF_WORLD - size)

        if x_low > x_high or y_low > y_high:
            raise ValueError("cannot fit custom root inside Web-Mercator world")

        root_xmin = float(np.clip(root_xmin, x_low, x_high))
        root_ymin = float(np.clip(root_ymin, y_low, y_high))

        return np.array(
            [
                root_xmin,
                root_ymin,
                root_xmin + size,
                root_ymin + size,
            ],
            dtype=np.float64,
        )


class XYZTileSystem(
    LooseQueryMixin,
    BaseTileSystem
):
    """Standard global XYZ/Web-Mercator tile system.

    No geographic initialization is required. ``max_zoom`` only controls the
    deepest level exposed by this object and defaults to 30.
    """

    def __init__(self, max_zoom: int = BaseTileSystem.MAX_SUPPORTED_ZOOM) -> None:
        self._validate_zoom(max_zoom)
        root_xy = np.array(
            [
                -self.HALF_WORLD,
                -self.HALF_WORLD,
                self.HALF_WORLD,
                self.HALF_WORLD,
            ],
            dtype=np.float64,
        )
        super().__init__(root_xy=root_xy, max_zoom=max_zoom)