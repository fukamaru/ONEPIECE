from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
from typing import Literal, Optional, Tuple, List, Sequence

from ..utils.data_utils import SerializableDataclass


@dataclass
class MVT(SerializableDataclass):
    uid: int | str | None = None
    bid: int | str | None = None

    quadkey: str
    geo_type: Literal["Point", "LineString", "Polygon"]
    geometry: Sequence[Tuple[int, int]] | Sequence[List[int, int]]
    timestamp: Tuple[datetime, datetime] | datetime | None

    positions: Sequence[Tuple[float, float]] | Sequence[List[float, float]]


@dataclass
class RMVT(SerializableDataclass):
    tsinfo: Literal["xyz", "custom"]
    tsdetail: dict | None = None
    mobility: Sequence[MVT]
