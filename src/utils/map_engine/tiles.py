from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, Sequence


@dataclass
class BBox:
    min_lon: float
    min_lat: float
    max_lon: float
    max_lat: float

    def as_tuple(
            self
    ) -> tuple[float, float, float, float]:
        return (
            self.min_lon,
            self.min_lat,
            self.max_lon,
            self.max_lat
        )


@dataclass(frozen=True)
class MapTile:
    z: int
    x: int 
    y: int
    quadkey: str
    bbox: BBox


@dataclass
class MapTileBatch:
    z: np.ndarray
    x: np.ndarray
    y: np.ndarray
    quadkey: np.ndarray
    bbox: np.ndarray

    def __len__(self) -> int:
        return len(self.z)

    def __getitem__(
            self, 
            i
    ) -> MapTile:
        b = self.bbox[i]

        return MapTile(
            z=int(self.z[i]),
            x=int(self.x[i]),
            y=int(self.y[i]),
            quadkey=self.quadkey[i].decode(),
            bbox=BBox(
                min_lon=float(b[0]),
                min_lat=float(b[1]),
                max_lon=float(b[2]),
                max_lat=float(b[3])
            )
        )


class MapTileSystem:
    """
    Custom Quadtree Map Tile System.
    """

    R = 6378137.0
    MAX_LAT = 85.0511287798066
    HALF_WORLD = math.pi * R
    WORLD_SIZE = 2.0 * HALF_WORLD

    def __init__(
            self,
            bbox: BBox | tuple[float, float, float, float],
            max_zoom: int | None = None,
            min_tile_size: float | None = None,
    ) -> None:
        if not isinstance(bbox, BBox):
            bbox = BBox(*bbox)

        self._validate_bbox(bbox)

        if max_zoom is None and min_tile_size is None:
            raise ValueError(
                "max_zoom and min_tile_size cannot both be None"
            )

        if max_zoom is not None and not 0 <= max_zoom <= 30:
            raise ValueError(
                f"max_zoom must be in [0, 30], but get {max_zoom}"
            )

        if min_tile_size is not None and min_tile_size <= 0:
            raise ValueError(
                "min_tile_size must be positive"
            )

        self.anchor_bbox = bbox
        self.requested_max_zoom = max_zoom
        self.requested_min_tile_size = min_tile_size

        # Anchor extent in Web--Mercator Coordinates
        xmin, ymin = self._lonlat_to_xy(bbox.min_lon, bbox.min_lat)
        xmax, ymax = self._lonlat_to_xy(bbox.max_lon, bbox.max_lat)

        anchor_width = xmax - xmin
        anchor_height = ymax- ymin
        anchor_size = max(anchor_width, anchor_height)

        self.max_zoom, self.root_size = self._resolve_grid(
            anchor_size=anchor_size,
            max_zoom=max_zoom,
            min_tile_size=min_tile_size
        )

        self.root_xy = self._build_root(
            xmin=xmin,
            ymin=ymin,
            xmax=xmax,
            ymax=ymax,
            size=self.root_size
        )

        self.root_bbox = self._xy_bbox_to_lonlat(
            self.root_xy.reshape(1, 4)
        )[0]
        self.root_bbox = BBox(*self.root_bbox)

        self.tile_size = (
            self.root_size / (1 << self.max_zoom)
        )

        self._n_max = np.int64(1 << self.max_zoom)
        self._scale_max = (
            self._n_max / self.root_size
        )

        self._root_xmin = self.root_xy[0]
        self._root_ymin = self.root_xy[1]
        self._root_xmax = self.root_xy[2]
        self._root_ymax = self.root_xy[3]

    # ==================================================================
    # Tile System Initialization
    # ==================================================================  
    @staticmethod
    def _resolve_grid(
        anchor_size: float,
        max_zoom: int | None,
        min_tile_size: float | None
    ) -> tuple[int, float]:
        if min_tile_size is None:
            return (
                int(max_zoom),
                anchor_size,
            ) 

        if anchor_size <= min_tile_size:
            required_zoom = 0
        else:
            required_zoom = math.ceil(
                math.log2(anchor_size / min_tile_size)
            )

        zoom = (
            required_zoom
            if max_zoom is None
            else min(
                required_zoom,
                int(max_zoom)
            )
        )
        if zoom == required_zoom:
            root_size = (
                min_tile_size * (1 << zoom)
            )
        else:
            root_size = anchor_size

        if root_size > MapTileSystem.WORLD_SIZE:
            raise ValueError(
                f"The adjusted root {root_size} exceeds the Web-Mercator world extent"
            )

        return zoom, root_size

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

        root_xmin = np.clip(
            root_xmin,
            max(xmax - size, -cls.HALF_WORLD),
            min(xmin, cls.HALF_WORLD - size)
        )

        root_ymin = np.clip(
            root_ymin,
            max(ymax - size, -cls.HALF_WORLD), min(ymin, cls.HALF_WORLD - size)
        )

        return np.array(
            [
                root_xmin,
                root_ymin,
                root_xmin + size,
                root_ymin + size
            ],
            dtype=np.float64
        )

    def zoom_level_info(
            self
    ) -> dict:
        z = np.arange(
            self.max_zoom + 1,
            dtype=np.int64
        )

        tiles_per_axis = np.left_shift(np.int64(1), z)

        tile_count = tiles_per_axis ** 2

        tile_size = (
            self.root_size / tiles_per_axis
        )

        return {
            "zoom": z,
            "tiles_per_axis": tiles_per_axis,
            "tile_count": tile_count,
            "tile_size": tile_size
        }

    # ==================================================================
    # Coordinate Conversion
    # ==================================================================
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

        xy_bboxs = np.column_stack(
            (xmin, ymin, xmax, ymax)
        )

        return self._xy_bbox_to_lonlat(xy_bboxs)

    @classmethod
    def _lonlat_to_xy(
        cls,
        lon,
        lat
    ) -> tuple[float, float]:
        lat = np.clip(
            lat, -cls.MAX_LAT, cls.MAX_LAT
        )
        x = (
            cls.R * np.deg2rad(lon)
        )
        lat = np.deg2rad(lat)
        y = (
            cls.R * np.log(np.tan(np.pi / 4 + lat / 2))
        )
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

        min_lon = np.rad2deg(
            xmin / cls.R
        )
        max_lon = np.rad2deg(
            xmax / cls.R
        )
        min_lat = np.rad2deg(
            2 * np.arctan(np.exp(ymin / cls.R)) - np.pi / 2
        )
        max_lat = np.rad2deg(
            2 * np.arctan(np.exp(ymax / cls.R)) - np.pi / 2
        )

        return np.column_stack(
            (min_lon, min_lat, max_lon, max_lat)
        )

    # ==================================================================
    # Tile Query Functions
    # ==================================================================
    def query_single(
            self,
            bbox: (
                BBox
                | tuple[float, float, float, float]
                | Sequence[BBox]
                | np.ndarray
            )
    ) -> MapTile | MapTileBatch:
        '''
        Query the deepest single map tile that fully covers each bbox.
        '''
        query, single = self._prepare_query(bbox)

        result = self._query_single_batch(query)

        if single:
            return result[0]
        
        return result

    def get_tile(
            self,
            z: int, 
            x: int,
            y: int,
    ) -> MapTile:
        '''
        Get one tile directly from (z, x, y).
        '''
        if not 0 <= z <= self.max_zoom:
            raise ValueError(f"invalid zoom {z}")

        n = 1 << z

        if not (
            0 <= x < n
            and 0 <= y < n
        ):
            raise ValueError("invalid tile index.")

        size = self.root_size / n

        xmin = self._root_xmin + x * size
        xmax = xmin + size
        ymax = self._root_ymax - y * size
        ymin = ymax - size

        bbox = self._xy_bbox_to_lonlat(
            np.array(
                [[xmin, ymin, xmax, ymax]],
                dtype=np.float64
            )
        )[0]

        return MapTile(
            z=z,
            x=x,
            y=y,
            quadkey=self.to_quadkey(z,x,y),
            bbox=BBox(*bbox)
        )

    def _query_single_batch(
            self,
            bboxs: np.ndarray
    ) -> MapTileBatch:
        xmin, ymin = self._lonlat_to_xy(
            bboxs[:, 0], bboxs[:, 1]
        )
        xmax, ymax = self._lonlat_to_xy(
            bboxs[:, 2], bboxs[:, 3]
        )

        outside = (
            (xmin < self._root_xmin)
            | (ymin < self._root_ymin)
            | (xmax > self._root_xmax)
            | (ymax > self._root_ymax)
        )

        if np.any(outside):
            rows = np.flatnonzero(outside)
            raise ValueError(
                f"query bboxs outside root: {rows[:10].tolist()}"
            )

        ## Tile range occupied by each query at max_zoom
        x0 = np.floor(
            (xmin - self._root_xmin) * self._scale_max
        ).astype(np.int64)

        x1 = np.floor(
            np.nextafter(
                (xmax - self._root_xmin) * self._scale_max, - np.inf
            )
        ).astype(np.int64)

        y0 = np.floor(
            (self._root_ymax - ymax) * self._scale_max
        ).astype(np.int64)

        y1 = np.floor(
            np.nextafter(
                (self._root_ymax - ymax) * self._scale_max, - np.inf
            )
        ).astype(np.int64)

        np.clip(x0, 0, self._n_max - 1, out=x0)
        np.clip(x1, 0, self._n_max - 1, out=x1)
        np.clip(y0, 0, self._n_max - 1, out=y0)
        np.clip(y1, 0, self._n_max - 1, out=y1)

        # Longest common quadtree prefix
        diff = (
            np.bitwise_xor(x0, x1)
            |
            np.bitwise_xor(y0, y1)
        )

        exponent = np.frexp(
            diff.astype(np.float64)
        )[1].astype(np.int64)

        z = (
            self.max_zoom - exponent
        )

        shift = (
            self.max_zoom - z
        )

        x = np.right_shift(
            x0, shift
        )
        y = np.right_shift(
            y0, shift
        )

        tile_bboxs = self._tile_bboxs_batch(
            z, x, y
        )
        quadkey = self._quadkey_batch(
            z, x, y
        )

        return MapTileBatch(
            z=z.astype(np.uint8, copy=False),
            x=x.astype(np.uint32, copy=False),
            y=y.astype(np.uint32, copy=False),
            quadkey=quadkey,
            bbox=tile_bboxs
        )

    def _quadkey_batch(
            self,
            z: np.ndarray,
            x: np.ndarray,
            y: np.ndarray
    ) -> np.ndarray:
        '''
        Generate QuadKeys by zoom group.
        '''
        output = np.empty(
            len(z), 
            dtype=f"S{self.max_zoom}"
        )

        for zz in np.unique(z):
            indices = np.flatnonzero(z == zz)
            if zz == 0:
                output[indices] = b""
                continue
            xx = x[indices].astype(np.uint64, copy=False)
            yy = y[indices].astype(np.uint64, copy=False)
            bits = np.arange(
                zz - 1, -1, -1, dtype=np.uint64
            )
            digits = (
                (
                    (xx[:, None] >> bits)
                    & 1
                )
                |
                (
                    (
                        (yy[:, None] >> bits)
                        & 1
                    ) 
                    << 1
                )
            ).astype(np.uint8)
            digits += ord("0")

            strings = (
                np.ascontiguousarray(digits)
                .view(f"S{zz}")
                .reshape(-1)
            )

            output[indices] = strings

        return output

    def _prepare_query(
            self,
            bboxs
    ) -> tuple[np.ndarray, bool]:
        
        if isinstance(bboxs, BBox):
            return (
                np.asarray(
                    [bboxs.as_tuple()],
                    dtype=np.float64
                ),
                True
            )

        if isinstance(bboxs, np.ndarray):
            array = np.asarray(bboxs, dtype=np.float64)
        elif (
            isinstance(bboxs, Sequence)
            and len(bboxs) > 0
            and isinstance(bboxs[0], BBox)
        ):
            array = np.asarray(
                [
                    b.as_tuple()
                    for b in bboxs
                ],
                dtype=np.float64
            )
        else:
            array = np.array(bboxs, dtype=np.float64)

        if array.ndim == 1:
            if array.shape != (4,):
                raise ValueError("BBox must have shape (4,) or (N,4)")
            array = array.reshape(1, 4)
            single = True

        elif (
            array.ndim == 2
            and array.shape[1] == 4
        ):
            single = False

        else:
            raise ValueError(f"BBox must have shape (4,) or (N,4)")

        self._validate_bbox_array(array)
        return array, single

    @staticmethod
    def to_quadkey(
        z: int, 
        x: int,
        y: int
    ) -> str:
        chars = []

        for bit in range(z-1, -1, -1):
            digit = (
                ((x >> bit) & 1)
                |
                (((y >> bit) & 1) << 1)
            )
            chars.append(str(digit))

        return "".join(chars)

    # ==================================================================
    # Validation
    # ==================================================================

    @classmethod
    def _validate_bbox(
        cls, 
        bbox: BBox
    ) -> None:
        if not (
            bbox.min_lat < bbox.max_lat 
            and bbox.min_lat < bbox.max_lat
        ):
            raise ValueError(f"Invalid bounds: {bbox}")

        if not (
            - 180 <= bbox.min_lon <= 180
            and - 180 <= bbox.max_lat <= 180
        ):
            raise ValueError(
                f"Longitude out of range: {bbox}"
            )

        if not (
            - cls.MAX_LAT <= bbox.min_lat <= cls.MAX_LAT
            and - cls.MAX_LAT <= bbox.max_lat <= cls.MAX_LAT
        ):
            raise ValueError(
                f"Latitudue out of Web-Mercator range: {bbox}"
            )

    @classmethod
    def _validate_bbox_array(
        cls,
        bbox: np.array,
    ) -> None:
        pass
