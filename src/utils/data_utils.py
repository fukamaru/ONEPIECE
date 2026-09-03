"""
TODO
- 将Mobility统一处理成GeoJSON形式的功能：处理完的结果包含基本属性+Geometry+经纬度范围（这是为了方便索引）
"""
from __future__ import annotations

import os
import json
import uuid
import types
import base64
import numpy as np

from enum import Enum
from pathlib import Path
from datetime import date, datetime, time
from dataclasses import fields, is_dataclass
from shapely.geometry import Point, LineString, Polygon
from typing import Any, Iterator, Union, Literal, get_origin, get_args, get_type_hints


# ===============================================================================================
# Save/Load Mobility Data as/from GeoJSON like format
# ===============================================================================================

def load_mobility_data(
        filepath,
        feature_only: bool = False
):
    ext = os.path.splitext(filepath)[-1]

    if (
        ext == '.json'
        or ext == '.jsonl'
        or ext == '.geojson'
    ):
        with open(filepath, 'r') as f:
            data = json.load(f)

        if feature_only:
            return data['features'], None
        else:
            return (
                data['features'],
                {k: v for k, v in data.items() if k != 'features'}
            )

def _to_geometry(
        lonlat: np.ndarray,
        properties: dict,
        geom_type: Literal["Point", "LineString", "Polygon"]
) -> dict:
    if geom_type == "Point":
        return {
            "properties": properties,
            "geometry": Point(lonlat[0, 0], lonlat[0, 1])
        }
    elif geom_type == "LineString":
        return {
            "properties": properties,
            "geometry": LineString(lonlat)
        }
    elif geom_type == "Polygon":
        return {
            "properties": properties,
            "geometry": Polygon(lonlat)
        }
    else:
        raise ValueError(f"Unsupported Geometry Type: {geom_type}")


def _from_geometry(
        geojson_like: dict
) -> tuple[dict, np.ndarray]:
    geom = geojson_like.get("geometry")
    properties = geojson_like.get("properties")
    return properties, geom


# ===============================================================================================
# Serializable Dataclass for Saving/Loading Mobility Data in GeoJSON-like format
# ===============================================================================================

class SerializableDataclass:

    # ================================================
    # Basic Conversion
    # ================================================

    @staticmethod
    def _to_primitive(
        obj: Any
    ) -> Any:
        '''
        Python / dataclass -> basic Python classes.
        '''

        if obj is None:
            return None

        # dataclass
        if is_dataclass(obj) and not isinstance(obj, type):
            return {
                field.name: SerializableDataclass._to_primitive(
                    getattr(obj, field.name)
                ) 
                for field in fields
            }

        # Enum
        if isinstance(obj, Enum):
            return obj.value

        # datetime
        if isinstance(obj, datetime):
            return obj.isoformat()

        if isinstance(obj, date):
            return obj.isoformat()

        if isinstance(obj, time):
            return obj.isoformat()

        # Path
        if isinstance(obj, Path):
            return str(obj)

        # UUID
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # bytes
        if isinstance(obj, bytes):
            return {
                "__type__": "bytes",
                "data": base64.b64encode(obj).decode(("ascii"))
            }

        # dict
        if isinstance(obj, dict):
            return {
                str(k): SerializableDataclass._to_primitive(v)
                for k, v in obj.items()
            }

        # list / tuple / set
        if isinstance(obj, (list, tuple, set)):
            return [
                SerializableDataclass._to_primitive(v)
                for v in obj
            ]

        # basic
        if isinstance(obj, (str, int, float, bool)):
            return obj

    @staticmethod
    def _from_primitive(
        value: Any,
        target_type: Any
    ) -> Any:

        if target_type is Any:
            return value

        if value is None:
            return None

        origin = get_origin(target_type)
        args = get_args(target_type)

        # ========================================================
        # Union / Optional
        # ========================================================

        if origin in (Union, types.UnionType):

            candidates = [
                t
                for t in args
                if t is not type(None)
            ]

            # Optional[T]
            if len(candidates) == 1:
                return SerializableDataclass._from_primitive(
                    value,
                    candidates[0],
                )

            last_error = None

            for candidate in candidates:
                try:
                    return SerializableDataclass._from_primitive(
                        value,
                        candidate,
                    )
                except (
                    TypeError,
                    ValueError,
                    KeyError,
                ) as exc:
                    last_error = exc

            raise TypeError(
                f"Cannot convert {value!r} "
                f"to {target_type!r}"
            ) from last_error

        # ========================================================
        # list
        # ========================================================

        if origin is list:
            item_type = args[0] if args else Any

            return [
                SerializableDataclass._from_primitive(
                    item,
                    item_type,
                )
                for item in value
            ]

        # ========================================================
        # set
        # ========================================================

        if origin is set:
            item_type = args[0] if args else Any

            return {
                SerializableDataclass._from_primitive(
                    item,
                    item_type,
                )
                for item in value
            }

        # ========================================================
        # tuple
        # ========================================================

        if origin is tuple:

            # tuple[T, ...]
            if (
                len(args) == 2
                and args[1] is Ellipsis
            ):
                item_type = args[0]

                return tuple(
                    SerializableDataclass._from_primitive(
                        item,
                        item_type,
                    )
                    for item in value
                )

            # tuple[int, str, ...]
            if args:
                return tuple(
                    SerializableDataclass._from_primitive(
                        item,
                        item_type,
                    )
                    for item, item_type
                    in zip(value, args)
                )

            return tuple(value)

        # ========================================================
        # dict
        # ========================================================

        if origin is dict:

            key_type = (
                args[0]
                if args
                else Any
            )

            value_type = (
                args[1]
                if len(args) > 1
                else Any
            )

            return {
                SerializableDataclass._from_primitive(
                    key,
                    key_type,
                ):
                SerializableDataclass._from_primitive(
                    val,
                    value_type,
                )
                for key, val in value.items()
            }

        # ========================================================
        # dataclass
        # ========================================================

        if (
            isinstance(target_type, type)
            and is_dataclass(target_type)
        ):
            hints = get_type_hints(target_type)

            kwargs = {}

            for field in fields(target_type):

                if field.name not in value:
                    # 缺失字段交给 default /
                    # default_factory 处理
                    continue

                field_type = hints.get(
                    field.name,
                    Any,
                )

                kwargs[field.name] = (
                    SerializableDataclass
                    ._from_primitive(
                        value[field.name],
                        field_type,
                    )
                )

            return target_type(**kwargs)

        # ========================================================
        # Enum
        # ========================================================

        if (
            isinstance(target_type, type)
            and issubclass(target_type, Enum)
        ):
            return target_type(value)

        # ========================================================
        # datetime / date / time
        # ========================================================

        if target_type is datetime:

            if isinstance(value, datetime):
                return value

            return datetime.fromisoformat(value)

        if target_type is date:

            if (
                isinstance(value, date)
                and not isinstance(value, datetime)
            ):
                return value

            return date.fromisoformat(value)

        if target_type is time:

            if isinstance(value, time):
                return value

            return time.fromisoformat(value)

        # ========================================================
        # Path
        # ========================================================

        if target_type is Path:
            return Path(value)

        # ========================================================
        # UUID
        # ========================================================

        if target_type is uuid.UUID:

            if isinstance(value, uuid.UUID):
                return value

            return uuid.UUID(value)

        # ========================================================
        # bytes
        # ========================================================

        if target_type is bytes:

            if isinstance(value, bytes):
                return value

            if (
                isinstance(value, dict)
                and value.get("__type__") == "bytes"
            ):
                return base64.b64decode(
                    value["data"]
                )

        # ========================================================
        # basic types
        # ========================================================

        if target_type is str:
            return str(value)

        if target_type is int:
            return int(value)

        if target_type is float:
            return float(value)

        if target_type is bool:
            return bool(value)

        return value

    # ================================================
    # dict API
    # ================================================

    def to_dict(self) -> dict[str, Any]:
        return self._to_primitive(self)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any]
    ) -> Self:
        return cls._from_primitive(data, cls)

    # ================================================
    # JSONL
    # ================================================

    @classmethod
    def save_jsonl(
        cls,
        objects,
        path: str | Path
    ) -> None:

        path = Path(path)

        with path.open("w", encoding="utf-8") as f:
            for obj in objects:
                f.write(
                    json.dumps(
                        obj.to_dict(),
                        ensure_ascii=False
                    )
                )
                f.write("\n")

    @classmethod
    def iter_jsonl(
        cls,
        path: str | Path
    ) -> Iterator[Self]:
        
        path = Path(path)

        with path.open(
            "r",
            encoding="utf-8"
        ) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield cls.from_dict(
                    json.loads(line)
                )

    # ================================================
    # Parquet
    # ================================================

    @classmethod
    def save_parquet(
        cls,
        objects,
        path: str | Path,
        *,
        compression: str = "zstd"
    ) -> None:

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "please install pyarrow"
                "pip install pyarrow"
            ) from exc

        records = [
            obj.to_dict()
            for obj in objects
        ]

        table = pa.Table.from_pylist(records)
        pq.write_table(
            table,
            Path(path),
            compression=compression
        )

    @classmethod
    def load_parquet(
        cls,
        path: str | Path
    ) -> list[Self]:

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "please install pyarrow:"
                "pip install pyarrow"
            ) from exc

        table = pq.read_table(Path(path))

        return [
            cls.from_dict(record)
            for record in table.to_pylist()
        ]

    @classmethod
    def iter_parquet(
        cls,
        path: str | Path,
        *,
        batch_size: int = 1_000_000
    ) -> Iterator[Self]:
        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "please install pyarrow:"
                "pip install pyarrow"
            ) from exc

        parquet_file = pq.ParquetFile(Path(path))

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            for record in batch.to_pylist():
                yield cls.from_dict(record)

    @classmethod
    def migrate(
        cls,
        data: dict[str, Any],
        from_version: int
    ) -> dict[str, Any]:
        return data

# ===============================================================================================
# Serializable Dataclass for Saving/Loading Mobility Data in GeoJSON-like format
# ===============================================================================================


__all__ = [
    "SerializableDataclass"
]
    