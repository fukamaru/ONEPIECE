import math
import os
import time

import numpy as np
import pytest

from tiles import (
    BBox,
    MapTile,
    MapTileBatch,
    MapTileSystem,
)


# ============================================================
# Test configuration
# ============================================================

TEST_BOUNDS = BBox(
    min_lon=122.93270464,
    min_lat=30.725239,
    max_lon=146.2692261,
    max_lat=45.5228577,
)


TEST_QUERIES = np.array(
    [
        [139.50, 35.50, 139.90, 35.90],
        [135.30, 34.50, 135.70, 34.90],
        [130.20, 32.50, 130.60, 32.90],
    ],
    dtype=np.float64,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def tile_system():
    return MapTileSystem(
        bbox=TEST_BOUNDS,
        max_zoom=20,
        min_tile_size=500.0,
    )


# ============================================================
# Helper functions
# ============================================================

def assert_bbox_close(
    a: BBox,
    b: BBox,
    atol: float = 1e-10,
):
    assert math.isclose(
        a.min_lon,
        b.min_lon,
        abs_tol=atol,
    )

    assert math.isclose(
        a.min_lat,
        b.min_lat,
        abs_tol=atol,
    )

    assert math.isclose(
        a.max_lon,
        b.max_lon,
        abs_tol=atol,
    )

    assert math.isclose(
        a.max_lat,
        b.max_lat,
        abs_tol=atol,
    )


def bbox_union(
    a: BBox,
    b: BBox,
) -> BBox:
    return BBox(
        min_lon=min(
            a.min_lon,
            b.min_lon,
        ),
        min_lat=min(
            a.min_lat,
            b.min_lat,
        ),
        max_lon=max(
            a.max_lon,
            b.max_lon,
        ),
        max_lat=max(
            a.max_lat,
            b.max_lat,
        ),
    )


def reference_query_single(
    ts: MapTileSystem,
    bbox: BBox,
):
    """
    Slow reference implementation.

    It searches from max_zoom down to z=0 and is used to verify
    the optimized longest-common-prefix implementation.
    """

    xmin, ymin = ts._lonlat_to_xy(
        bbox.min_lon,
        bbox.min_lat,
    )

    xmax, ymax = ts._lonlat_to_xy(
        bbox.max_lon,
        bbox.max_lat,
    )

    root_xmin = ts.root_xy[0]
    root_ymax = ts.root_xy[3]

    for z in range(
        ts.max_zoom,
        -1,
        -1,
    ):
        n = 1 << z
        scale = n / ts.root_size

        x0 = math.floor(
            (xmin - root_xmin)
            * scale
        )

        x1 = math.floor(
            math.nextafter(
                (xmax - root_xmin)
                * scale,
                -math.inf,
            )
        )

        y0 = math.floor(
            (root_ymax - ymax)
            * scale
        )

        y1 = math.floor(
            math.nextafter(
                (root_ymax - ymin)
                * scale,
                -math.inf,
            )
        )

        x0 = min(
            max(x0, 0),
            n - 1,
        )

        x1 = min(
            max(x1, 0),
            n - 1,
        )

        y0 = min(
            max(y0, 0),
            n - 1,
        )

        y1 = min(
            max(y1, 0),
            n - 1,
        )

        if (
            x0 == x1
            and y0 == y1
        ):
            return z, x0, y0

    raise AssertionError(
        "z=0 must contain every valid query"
    )


# ============================================================
# BBox
# ============================================================

def test_bbox():

    bbox = BBox(
        122.93270464,
        30.725239,
        146.2692261,
        45.5228577,
    )

    assert bbox.min_lon == 122.93270464
    assert bbox.min_lat == 30.725239
    assert bbox.max_lon == 146.2692261
    assert bbox.max_lat == 45.5228577

    assert bbox.as_tuple() == (
        122.93270464,
        30.725239,
        146.2692261,
        45.5228577,
    )


# ============================================================
# Initialization
# ============================================================

def test_initialization_with_min_tile_size_only():

    ts = MapTileSystem(
        bbox=TEST_BOUNDS,
        min_tile_size=500.0,
    )

    assert ts.max_zoom >= 0

    assert math.isclose(
        ts.tile_size,
        500.0,
        rel_tol=0.0,
        abs_tol=1e-9,
    )

    assert math.isclose(
        ts.root_size,
        ts.tile_size
        * (1 << ts.max_zoom),
        rel_tol=1e-12,
    )


def test_initialization_with_max_zoom_only():

    ts = MapTileSystem(
        bbox=TEST_BOUNDS,
        max_zoom=8,
    )

    assert ts.max_zoom == 8

    assert math.isclose(
        ts.tile_size,
        ts.root_size / (1 << 8),
        rel_tol=1e-12,
    )


def test_initialization_with_both_constraints():

    ts = MapTileSystem(
        bbox=TEST_BOUNDS,
        max_zoom=30,
        min_tile_size=500.0,
    )

    assert ts.max_zoom <= 30

    assert math.isclose(
        ts.tile_size,
        500.0,
        abs_tol=1e-9,
    )


def test_max_zoom_can_limit_min_tile_size():

    ts = MapTileSystem(
        bbox=TEST_BOUNDS,
        max_zoom=2,
        min_tile_size=500.0,
    )

    assert ts.max_zoom == 2

    assert ts.tile_size > 500.0


def test_root_is_square_in_internal_coordinates(
    tile_system,
):

    root = tile_system.root_xy

    width = (
        root[2]
        - root[0]
    )

    height = (
        root[3]
        - root[1]
    )

    assert math.isclose(
        width,
        height,
        rel_tol=1e-12,
    )

    assert math.isclose(
        width,
        tile_system.root_size,
        rel_tol=1e-12,
    )


def test_root_contains_anchor(
    tile_system,
):

    xmin, ymin = tile_system._lonlat_to_xy(
        TEST_BOUNDS.min_lon,
        TEST_BOUNDS.min_lat,
    )

    xmax, ymax = tile_system._lonlat_to_xy(
        TEST_BOUNDS.max_lon,
        TEST_BOUNDS.max_lat,
    )

    root = tile_system.root_xy

    eps = 1e-7

    assert xmin >= root[0] - eps
    assert ymin >= root[1] - eps
    assert xmax <= root[2] + eps
    assert ymax <= root[3] + eps


def test_root_bounds_contains_anchor(
    tile_system,
):

    root = tile_system.root_bbox

    assert root.min_lon <= TEST_BOUNDS.min_lon
    assert root.min_lat <= TEST_BOUNDS.min_lat

    assert root.max_lon >= TEST_BOUNDS.max_lon
    assert root.max_lat >= TEST_BOUNDS.max_lat


def test_leaf_tile_size_consistency(
    tile_system,
):

    expected = (
        tile_system.root_size
        / (1 << tile_system.max_zoom)
    )

    assert math.isclose(
        tile_system.tile_size,
        expected,
        rel_tol=1e-12,
    )


# ============================================================
# Invalid initialization
# ============================================================

@pytest.mark.parametrize(
    "bbox",
    [
        # Reversed longitude.
        BBox(
            146.0,
            31.0,
            123.0,
            45.0,
        ),

        # Reversed latitude.
        BBox(
            123.0,
            45.0,
            146.0,
            31.0,
        ),

        # Longitude below -180.
        BBox(
            -181.0,
            30.0,
            140.0,
            40.0,
        ),

        # Longitude above 180.
        BBox(
            123.0,
            30.0,
            181.0,
            40.0,
        ),

        # Latitude outside Web-Mercator range.
        BBox(
            123.0,
            -86.0,
            140.0,
            40.0,
        ),
    ],
)
def test_invalid_initial_bounds(
    bbox,
):

    with pytest.raises(ValueError):

        MapTileSystem(
            bbox=bbox,
            max_zoom=10,
        )


def test_missing_zoom_and_tile_size():

    with pytest.raises(ValueError):

        MapTileSystem(
            bbox=TEST_BOUNDS,
        )


@pytest.mark.parametrize(
    "value",
    [
        -1,
        31,
    ],
)
def test_invalid_max_zoom(
    value,
):

    with pytest.raises(ValueError):

        MapTileSystem(
            bbox=TEST_BOUNDS,
            max_zoom=value,
        )


@pytest.mark.parametrize(
    "value",
    [
        0,
        -1,
        -500,
    ],
)
def test_invalid_min_tile_size(
    value,
):

    with pytest.raises(ValueError):

        MapTileSystem(
            bbox=TEST_BOUNDS,
            min_tile_size=value,
        )


# ============================================================
# Tile matrix structure
# ============================================================

@pytest.mark.parametrize(
    "z",
    [
        0,
        1,
        2,
        5,
        10,
    ],
)
def test_number_of_tiles_per_zoom(
    tile_system,
    z,
):

    if z > tile_system.max_zoom:
        pytest.skip(
            "zoom exceeds TileSystem max_zoom"
        )

    tiles_per_axis = (
        1 << z
    )

    total_tiles = (
        1 << (2 * z)
    )

    assert tiles_per_axis == 2 ** z
    assert total_tiles == 4 ** z


def test_tile_size_per_zoom(
    tile_system,
):

    for z in range(
        tile_system.max_zoom + 1
    ):

        expected_size = (
            tile_system.root_size
            / (1 << z)
        )

        assert expected_size > 0

        if z == tile_system.max_zoom:

            assert math.isclose(
                expected_size,
                tile_system.tile_size,
                rel_tol=1e-12,
            )


# ============================================================
# QuadKey
# ============================================================

@pytest.mark.parametrize(
    "z, x, y, expected",
    [
        (0, 0, 0, ""),
        (1, 0, 0, "0"),
        (1, 1, 0, "1"),
        (1, 0, 1, "2"),
        (1, 1, 1, "3"),
        (2, 3, 2, "31"),
        (2, 2, 3, "32"),
    ],
)
def test_quadkey(
    z,
    x,
    y,
    expected,
):

    assert (
        MapTileSystem.to_quadkey(
            z,
            x,
            y,
        )
        == expected
    )


def test_quadkey_length(
    tile_system,
):

    for z in range(
        tile_system.max_zoom + 1
    ):

        n = 1 << z

        quadkey = (
            tile_system.to_quadkey(
                z,
                n - 1,
                n - 1,
            )
        )

        assert len(quadkey) == z


def test_quadkey_hierarchy():

    parent = MapTileSystem.to_quadkey(
        4,
        6,
        9,
    )

    child_x = 6 << 1
    child_y = 9 << 1

    children = [
        MapTileSystem.to_quadkey(
            5,
            child_x,
            child_y,
        ),
        MapTileSystem.to_quadkey(
            5,
            child_x + 1,
            child_y,
        ),
        MapTileSystem.to_quadkey(
            5,
            child_x,
            child_y + 1,
        ),
        MapTileSystem.to_quadkey(
            5,
            child_x + 1,
            child_y + 1,
        ),
    ]

    for child in children:

        assert child.startswith(
            parent
        )

        assert len(child) == (
            len(parent) + 1
        )


# ============================================================
# get_tile
# ============================================================

def test_get_root_tile(
    tile_system,
):

    tile = tile_system.get_tile(
        0,
        0,
        0,
    )

    assert isinstance(
        tile,
        MapTile,
    )

    assert tile.z == 0
    assert tile.x == 0
    assert tile.y == 0
    assert tile.quadkey == ""

    assert_bbox_close(
        tile.bbox,
        tile_system.root_bbox,
    )


def test_get_tile_indices(
    tile_system,
):

    tile = tile_system.get_tile(
        3,
        4,
        5,
    )

    assert tile.z == 3
    assert tile.x == 4
    assert tile.y == 5

    assert (
        tile.quadkey
        == tile_system.to_quadkey(
            3,
            4,
            5,
        )
    )


def test_adjacent_tiles_share_longitude_boundary(
    tile_system,
):

    left = tile_system.get_tile(
        4,
        4,
        7,
    )

    right = tile_system.get_tile(
        4,
        5,
        7,
    )

    assert math.isclose(
        left.bbox.max_lon,
        right.bbox.min_lon,
        abs_tol=1e-10,
    )


def test_adjacent_tiles_share_latitude_boundary(
    tile_system,
):

    north = tile_system.get_tile(
        4,
        4,
        6,
    )

    south = tile_system.get_tile(
        4,
        4,
        7,
    )

    assert math.isclose(
        north.bbox.min_lat,
        south.bbox.max_lat,
        abs_tol=1e-10,
    )


def test_four_children_cover_parent(
    tile_system,
):

    z = 4
    x = 5
    y = 6

    parent = tile_system.get_tile(
        z,
        x,
        y,
    )

    children = [
        tile_system.get_tile(
            z + 1,
            2 * x,
            2 * y,
        ),
        tile_system.get_tile(
            z + 1,
            2 * x + 1,
            2 * y,
        ),
        tile_system.get_tile(
            z + 1,
            2 * x,
            2 * y + 1,
        ),
        tile_system.get_tile(
            z + 1,
            2 * x + 1,
            2 * y + 1,
        ),
    ]

    combined = BBox(
        min(
            t.bbox.min_lon
            for t in children
        ),
        min(
            t.bbox.min_lat
            for t in children
        ),
        max(
            t.bbox.max_lon
            for t in children
        ),
        max(
            t.bbox.max_lat
            for t in children
        ),
    )

    assert_bbox_close(
        combined,
        parent.bbox,
    )


def test_invalid_get_tile(
    tile_system,
):

    with pytest.raises(ValueError):

        tile_system.get_tile(
            tile_system.max_zoom + 1,
            0,
            0,
        )

    with pytest.raises(ValueError):

        tile_system.get_tile(
            2,
            4,
            0,
        )

    with pytest.raises(ValueError):

        tile_system.get_tile(
            2,
            0,
            4,
        )


# ============================================================
# query_single: basic behavior
# ============================================================

def test_query_entire_root_returns_zoom_zero(
    tile_system,
):

    tile = tile_system.query_single(
        tile_system.root_bbox
    )

    assert isinstance(
        tile,
        MapTile,
    )

    assert tile.z == 0
    assert tile.x == 0
    assert tile.y == 0
    assert tile.quadkey == ""


@pytest.mark.parametrize(
    "z, x, y",
    [
        (1, 0, 0),
        (1, 1, 1),
        (2, 2, 1),
        (4, 5, 7),
        (6, 17, 31),
    ],
)
def test_query_exact_tile_bounds(
    tile_system,
    z,
    x,
    y,
):

    if z > tile_system.max_zoom:
        pytest.skip(
            "zoom exceeds TileSystem max_zoom"
        )

    original = (
        tile_system.get_tile(
            z,
            x,
            y,
        )
    )

    result = (
        tile_system.query_single(
            original.bbox
        )
    )

    assert result.z == original.z
    assert result.x == original.x
    assert result.y == original.y
    assert (
        result.quadkey
        == original.quadkey
    )


def test_query_small_area_inside_one_leaf_tile(
    tile_system,
):

    z = tile_system.max_zoom

    n = 1 << z

    x = n // 2
    y = n // 2

    tile = tile_system.get_tile(
        z,
        x,
        y,
    )

    lon_margin = (
        tile.bbox.max_lon
        - tile.bbox.min_lon
    ) * 0.25

    lat_margin = (
        tile.bbox.max_lat
        - tile.bbox.min_lat
    ) * 0.25

    query = BBox(
        tile.bbox.min_lon
        + lon_margin,

        tile.bbox.min_lat
        + lat_margin,

        tile.bbox.max_lon
        - lon_margin,

        tile.bbox.max_lat
        - lat_margin,
    )

    result = tile_system.query_single(
        query
    )

    assert result.z == z
    assert result.x == x
    assert result.y == y


def test_query_two_horizontal_siblings_returns_parent(
    tile_system,
):

    z = min(
        6,
        tile_system.max_zoom,
    )

    left = tile_system.get_tile(
        z,
        20,
        18,
    )

    right = tile_system.get_tile(
        z,
        21,
        18,
    )

    query = bbox_union(
        left.bbox,
        right.bbox,
    )

    result = tile_system.query_single(
        query
    )

    assert result.z == z - 1

    assert result.x == (
        20 >> 1
    )

    assert result.y == (
        18 >> 1
    )


def test_query_two_vertical_siblings_returns_parent(
    tile_system,
):

    z = min(
        6,
        tile_system.max_zoom,
    )

    north = tile_system.get_tile(
        z,
        20,
        18,
    )

    south = tile_system.get_tile(
        z,
        20,
        19,
    )

    query = bbox_union(
        north.bbox,
        south.bbox,
    )

    result = tile_system.query_single(
        query
    )

    assert result.z == z - 1

    assert result.x == (
        20 >> 1
    )

    assert result.y == (
        18 >> 1
    )


def test_query_four_siblings_returns_parent(
    tile_system,
):

    z = min(
        6,
        tile_system.max_zoom,
    )

    parent_x = 10
    parent_y = 9

    children = [
        tile_system.get_tile(
            z,
            parent_x * 2,
            parent_y * 2,
        ),
        tile_system.get_tile(
            z,
            parent_x * 2 + 1,
            parent_y * 2,
        ),
        tile_system.get_tile(
            z,
            parent_x * 2,
            parent_y * 2 + 1,
        ),
        tile_system.get_tile(
            z,
            parent_x * 2 + 1,
            parent_y * 2 + 1,
        ),
    ]

    query = BBox(
        min(
            t.bbox.min_lon
            for t in children
        ),
        min(
            t.bbox.min_lat
            for t in children
        ),
        max(
            t.bbox.max_lon
            for t in children
        ),
        max(
            t.bbox.max_lat
            for t in children
        ),
    )

    result = tile_system.query_single(
        query
    )

    assert result.z == z - 1
    assert result.x == parent_x
    assert result.y == parent_y


def test_query_outside_root(
    tile_system,
):

    root = tile_system.root_bbox

    outside = BBox(
        min_lon=root.min_lon - 1.0,
        min_lat=root.min_lat,
        max_lon=root.min_lon - 0.5,
        max_lat=root.max_lat,
    )

    with pytest.raises(ValueError):

        tile_system.query_single(
            outside
        )


# ============================================================
# query_single: reference comparison
# ============================================================

def test_query_single_random_against_reference(
    tile_system,
):

    rng = np.random.default_rng(
        42
    )

    root = tile_system.root_bbox

    lon_span = (
        root.max_lon
        - root.min_lon
    )

    lat_span = (
        root.max_lat
        - root.min_lat
    )

    for _ in range(1000):

        lon0, lon1 = np.sort(
            rng.uniform(
                root.min_lon
                + lon_span * 1e-5,
                root.max_lon
                - lon_span * 1e-5,
                2,
            )
        )

        lat0, lat1 = np.sort(
            rng.uniform(
                root.min_lat
                + lat_span * 1e-5,
                root.max_lat
                - lat_span * 1e-5,
                2,
            )
        )

        if (
            lon1 - lon0 < 1e-12
            or lat1 - lat0 < 1e-12
        ):
            continue

        query = BBox(
            lon0,
            lat0,
            lon1,
            lat1,
        )

        expected = (
            reference_query_single(
                tile_system,
                query,
            )
        )

        actual = (
            tile_system.query_single(
                query
            )
        )

        assert (
            actual.z,
            actual.x,
            actual.y,
        ) == expected


# ============================================================
# Batch query
# ============================================================

def test_batch_query_returns_tile_batch(
    tile_system,
):

    result = tile_system.query_single(
        TEST_QUERIES
    )

    assert isinstance(
        result,
        MapTileBatch,
    )

    assert len(result) == len(
        TEST_QUERIES
    )


def test_batch_matches_individual_queries(
    tile_system,
):

    queries = [
        BBox(
            139.50,
            35.50,
            139.90,
            35.90,
        ),
        BBox(
            135.30,
            34.50,
            135.70,
            34.90,
        ),
        BBox(
            130.20,
            32.50,
            130.60,
            32.90,
        ),
    ]

    batch = tile_system.query_single(
        queries
    )

    for i, query in enumerate(
        queries
    ):

        single = (
            tile_system.query_single(
                query
            )
        )

        batch_tile = batch[i]

        assert (
            batch_tile.z
            == single.z
        )

        assert (
            batch_tile.x
            == single.x
        )

        assert (
            batch_tile.y
            == single.y
        )

        assert (
            batch_tile.quadkey
            == single.quadkey
        )

        assert_bbox_close(
            batch_tile.bbox,
            single.bbox,
        )


def test_batch_preserves_input_order(
    tile_system,
):

    queries = np.array(
        [
            [124.0, 32.0, 124.1, 32.1],
            [144.0, 43.0, 144.1, 43.1],
            [139.0, 35.0, 139.1, 35.1],
        ],
        dtype=np.float64,
    )

    batch = tile_system.query_single(
        queries
    )

    for i in range(
        len(queries)
    ):

        single = tile_system.query_single(
            queries[i]
        )

        assert (
            batch[i].quadkey
            == single.quadkey
        )


def test_sequence_of_bbox_query(
    tile_system,
):

    queries = [
        BBox(
            139.5,
            35.5,
            139.9,
            35.9,
        ),
        BBox(
            135.3,
            34.5,
            135.7,
            34.9,
        ),
    ]

    result = tile_system.query_single(
        queries
    )

    assert isinstance(
        result,
        MapTileBatch,
    )

    assert len(result) == 2


def test_large_random_batch_matches_reference(
    tile_system,
):

    rng = np.random.default_rng(
        1234
    )

    root = tile_system.root_bbox

    n = 10_000

    lon = rng.uniform(
        root.min_lon,
        root.max_lon,
        size=(n, 2),
    )

    lat = rng.uniform(
        root.min_lat,
        root.max_lat,
        size=(n, 2),
    )

    lon.sort(axis=1)
    lat.sort(axis=1)

    queries = np.column_stack(
        [
            lon[:, 0],
            lat[:, 0],
            lon[:, 1],
            lat[:, 1],
        ]
    )

    valid = (
        (
            queries[:, 2]
            - queries[:, 0]
            > 1e-10
        )
        &
        (
            queries[:, 3]
            - queries[:, 1]
            > 1e-10
        )
    )

    queries = queries[valid]

    result = tile_system.query_single(
        queries
    )

    assert len(result) == len(
        queries
    )

    sample = rng.choice(
        len(queries),
        size=min(
            500,
            len(queries),
        ),
        replace=False,
    )

    for i in sample:

        bbox = BBox(
            *queries[i]
        )

        expected = (
            reference_query_single(
                tile_system,
                bbox,
            )
        )

        assert (
            int(result.z[i]),
            int(result.x[i]),
            int(result.y[i]),
        ) == expected


# ============================================================
# TileBatch
# ============================================================

def test_tile_batch_index_returns_tile(
    tile_system,
):

    result = tile_system.query_single(
        TEST_QUERIES
    )

    tile = result[0]

    assert isinstance(
        tile,
        MapTile,
    )

    assert isinstance(
        tile.bbox,
        BBox,
    )

    assert isinstance(
        tile.quadkey,
        str,
    )


def test_batch_array_lengths(
    tile_system,
):

    result = tile_system.query_single(
        TEST_QUERIES
    )

    n = len(TEST_QUERIES)

    assert len(result.z) == n
    assert len(result.x) == n
    assert len(result.y) == n
    assert len(result.quadkey) == n

    assert result.bbox.shape == (
        n,
        4,
    )


def test_batch_quadkey_length(
    tile_system,
):

    result = tile_system.query_single(
        TEST_QUERIES
    )

    for i in range(
        len(result)
    ):

        quadkey = (
            result.quadkey[i]
            .decode()
        )

        assert (
            len(quadkey)
            == int(result.z[i])
        )


# ============================================================
# Numerical boundary behavior
# ============================================================

def test_query_near_southwest_root_corner(
    tile_system,
):

    root = tile_system.root_bbox

    lon_eps = (
        root.max_lon
        - root.min_lon
    ) * 1e-8

    lat_eps = (
        root.max_lat
        - root.min_lat
    ) * 1e-8

    query = BBox(
        root.min_lon,
        root.min_lat,
        root.min_lon + lon_eps,
        root.min_lat + lat_eps,
    )

    result = tile_system.query_single(
        query
    )

    assert (
        0
        <= result.z
        <= tile_system.max_zoom
    )


def test_query_near_northeast_root_corner(
    tile_system,
):

    root = tile_system.root_bbox

    lon_eps = (
        root.max_lon
        - root.min_lon
    ) * 1e-8

    lat_eps = (
        root.max_lat
        - root.min_lat
    ) * 1e-8

    query = BBox(
        root.max_lon - lon_eps,
        root.max_lat - lat_eps,
        root.max_lon,
        root.max_lat,
    )

    result = tile_system.query_single(
        query
    )

    assert (
        0
        <= result.z
        <= tile_system.max_zoom
    )


def test_level_one_tiles_cover_root(
    tile_system,
):

    tiles = [
        tile_system.get_tile(
            1,
            0,
            0,
        ),
        tile_system.get_tile(
            1,
            1,
            0,
        ),
        tile_system.get_tile(
            1,
            0,
            1,
        ),
        tile_system.get_tile(
            1,
            1,
            1,
        ),
    ]

    combined = BBox(
        min(
            t.bbox.min_lon
            for t in tiles
        ),
        min(
            t.bbox.min_lat
            for t in tiles
        ),
        max(
            t.bbox.max_lon
            for t in tiles
        ),
        max(
            t.bbox.max_lat
            for t in tiles
        ),
    )

    assert_bbox_close(
        combined,
        tile_system.root_bbox,
    )


# ============================================================
# Performance
# ============================================================

@pytest.mark.performance
def test_batch_query_performance(
    tile_system,
):
    """
    Benchmark vectorized query_single() with 100,000 bboxes.
    """

    rng = np.random.default_rng(2026)

    n = 100_000

    # Work directly in the TileSystem's internal XY space.
    root_xmin, root_ymin, root_xmax, root_ymax = map(
        float,
        tile_system.root_xy,
    )

    root_size = float(
        tile_system.root_size
    )

    # Keep generated queries away from the root boundary.
    margin = root_size * 0.01

    center_x = rng.uniform(
        root_xmin + margin,
        root_xmax - margin,
        size=n,
    )

    center_y = rng.uniform(
        root_ymin + margin,
        root_ymax - margin,
        size=n,
    )

    # Generate bboxes of different sizes.
    min_half_size = root_size * 1e-7
    max_half_size = root_size * 1e-4

    half_width = rng.uniform(
        min_half_size,
        max_half_size,
        size=n,
    )

    half_height = rng.uniform(
        min_half_size,
        max_half_size,
        size=n,
    )

    xy_queries = np.column_stack(
        (
            center_x - half_width,
            center_y - half_height,
            center_x + half_width,
            center_y + half_height,
        )
    )

    # Convert all query bboxes back to lon/lat in one vectorized operation.
    queries = tile_system._xy_bbox_to_lonlat(
        xy_queries
    )

    assert queries.shape == (
        n,
        4,
    )

    # Warm-up run to avoid measuring first-call overhead.
    tile_system.query_single(
        queries[:100]
    )

    start = time.perf_counter()

    result = tile_system.query_single(
        queries
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    assert isinstance(
        result,
        MapTileBatch,
    )

    assert len(result) == n

    throughput = (
        n / elapsed
    )

    print(
        f"\n"
        f"{n:,} bbox queries\n"
        f"elapsed:    {elapsed:.6f} s\n"
        f"throughput: {throughput:,.0f} queries/s"
    )

    max_seconds = os.getenv(
        "TILE_TEST_MAX_SECONDS"
    )

    if max_seconds is not None:
        assert elapsed <= float(
            max_seconds
        )