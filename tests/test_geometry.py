"""Tests for the pure geometry helpers in ``compute_grid_weights``.

These functions have no I/O and decide where every sample point lands, so they are
both the fastest and the highest-leverage thing to pin down.
"""

import numpy as np
import pytest
from shapely.geometry import Point

from factories import MINI_LATS, MINI_LONS, mini_coordinates
from netcdf2region.compute_grid_weights import (
    cell_bounds,
    generate_point_distribution,
    random_points_in_cell,
    uniform_points_in_cell,
)

UNIT_CELL = (0.0, 0.0, 1.0, 1.0)


# --------------------------------------------------------------------------- #
# uniform_points_in_cell
# --------------------------------------------------------------------------- #

def test_uniform_points_form_a_square_lattice():
    """100 points in a unit cell must be the 10x10 lattice at 0.05, 0.15, ..."""
    points = uniform_points_in_cell(UNIT_CELL, 100)

    assert len(points) == 100
    expected = np.round(np.arange(0.05, 1.0, 0.1), 6)
    assert np.array_equal(np.unique(np.round([p.x for p in points], 6)), expected)
    assert np.array_equal(np.unique(np.round([p.y for p in points], 6)), expected)


def test_uniform_points_lie_strictly_inside_the_cell():
    """No sample point may sit on a cell edge, where membership is ambiguous."""
    minx, miny, maxx, maxy = UNIT_CELL
    for point in uniform_points_in_cell(UNIT_CELL, 100):
        assert minx < point.x < maxx
        assert miny < point.y < maxy


def test_uniform_points_are_symmetric_about_the_centre():
    """The lattice must not be biased towards one side of the cell."""
    points = uniform_points_in_cell(UNIT_CELL, 100)
    assert np.isclose(np.mean([p.x for p in points]), 0.5)
    assert np.isclose(np.mean([p.y for p in points]), 0.5)


def test_uniform_points_tolerate_inverted_vertical_bounds():
    """``generate_point_distribution`` passes bounds as (lon1, lat2, lon2, lat1).

    That puts the upper latitude in the ``miny`` slot. The linspace then descends
    and the offset goes negative, so the points still tile the cell correctly -
    surprising, but relied upon by the whole pipeline.
    """
    upright = uniform_points_in_cell((0.0, 0.0, 1.0, 1.0), 100)
    inverted = uniform_points_in_cell((0.0, 1.0, 1.0, 0.0), 100)

    as_set = lambda pts: sorted((round(p.x, 6), round(p.y, 6)) for p in pts)  # noqa: E731
    assert as_set(upright) == as_set(inverted)


@pytest.mark.parametrize("n", [1, 4, 9, 25, 50, 99, 100, 101])
def test_uniform_points_return_exactly_n_points(n):
    """The caller assumes ``len(pts) == points_per_cell`` when building cell ids."""
    assert len(uniform_points_in_cell(UNIT_CELL, n)) == n


def test_uniform_points_are_truncated_when_n_is_not_a_perfect_square():
    """Document the sampling bias for non-square ``points_per_cell``.

    Points are generated column by column and then cut to length, so with n=50
    (nx=7, ny=8 -> 56 generated) the final column is only partly present and the
    cell is sampled unevenly. This is why the default of 100 matters; the test
    locks in the current behaviour rather than endorsing it.
    """
    points = uniform_points_in_cell(UNIT_CELL, 50)
    assert len(points) == 50

    per_column = {}
    for point in points:
        per_column[round(point.x, 6)] = per_column.get(round(point.x, 6), 0) + 1

    assert len(set(per_column.values())) > 1, "expected an uneven lattice"
    assert sorted(per_column.values()) == [2] + [8] * 6


# --------------------------------------------------------------------------- #
# random_points_in_cell
# --------------------------------------------------------------------------- #

def test_random_points_count_and_containment():
    np.random.seed(0)
    points = random_points_in_cell(UNIT_CELL, 250)

    assert len(points) == 250
    assert all(0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0 for p in points)


def test_random_points_are_reproducible_under_a_seed():
    """Random sampling is only testable because the global RNG can be seeded."""
    np.random.seed(42)
    first = [(p.x, p.y) for p in random_points_in_cell(UNIT_CELL, 10)]
    np.random.seed(42)
    second = [(p.x, p.y) for p in random_points_in_cell(UNIT_CELL, 10)]

    assert first == second


# --------------------------------------------------------------------------- #
# cell_bounds
# --------------------------------------------------------------------------- #

def test_cell_bounds_centered_surrounds_the_coordinate():
    coords = mini_coordinates("centered")
    lo_x, lo_y, hi_x, hi_y = cell_bounds(0, 0, coords)

    assert (lo_x, lo_y, hi_x, hi_y) == (0.0, 0.0, 1.0, 1.0)
    assert lo_x < MINI_LONS[0] < hi_x
    assert lo_y < MINI_LATS[0] < hi_y


def test_cell_bounds_top_left_spans_adjacent_coordinates():
    coords = mini_coordinates("top-left")
    assert cell_bounds(0, 0, coords) == (0.5, 0.5, 1.5, 1.5)
    assert cell_bounds(1, 2, coords) == (2.5, 1.5, 3.5, 2.5)


def test_cell_bounds_normalises_descending_latitudes():
    """Latitudes often descend in real NetCDF files; bounds must still be ordered."""
    coords = {
        "lat": np.array([5.5, 4.5, 3.5, 2.5, 1.5, 0.5]),
        "lon": MINI_LONS.copy(),
        "lat_res": 1.0,
        "lon_res": 1.0,
        "coordinate_origin": "centered",
    }
    for i in range(len(coords["lat"])):
        lo_x, lo_y, hi_x, hi_y = cell_bounds(i, 0, coords)
        assert lo_x < hi_x
        assert lo_y < hi_y


def test_cell_bounds_tile_the_grid_without_gaps():
    coords = mini_coordinates("centered")
    left = cell_bounds(0, 0, coords)
    right = cell_bounds(0, 1, coords)
    assert left[2] == right[0]


# --------------------------------------------------------------------------- #
# generate_point_distribution
# --------------------------------------------------------------------------- #

def test_generate_point_distribution_centered_cell_count():
    """A centred grid treats every coordinate as a cell centre."""
    coords = mini_coordinates("centered")
    point_gdf, cell_coords = generate_point_distribution(None, coords, 4, "uniform")

    n_cells = len(MINI_LATS) * len(MINI_LONS)
    assert len(cell_coords) == n_cells
    assert len(point_gdf) == n_cells * 4


def test_generate_point_distribution_top_left_has_one_fewer_cell_per_axis():
    """A top-left grid treats coordinates as edges, so there is one cell fewer."""
    coords = mini_coordinates("top-left")
    point_gdf, cell_coords = generate_point_distribution(None, coords, 4, "uniform")

    n_cells = (len(MINI_LATS) - 1) * (len(MINI_LONS) - 1)
    assert len(cell_coords) == n_cells
    assert len(point_gdf) == n_cells * 4


def test_generate_point_distribution_maps_cells_to_their_coordinates():
    """``cell_coords`` is what turns a cell index back into the CSV's lat/lon."""
    coords = mini_coordinates("centered")
    _, cell_coords = generate_point_distribution(None, coords, 1, "uniform")

    assert cell_coords[(0, 0)] == (MINI_LATS[0], MINI_LONS[0])
    assert cell_coords[(3, 5)] == (MINI_LATS[3], MINI_LONS[5])


def test_generate_point_distribution_points_fall_in_their_own_cell():
    coords = mini_coordinates("centered")
    point_gdf, _ = generate_point_distribution(None, coords, 9, "uniform")

    for (i, j), group in point_gdf.groupby("cell"):
        lo_x, lo_y, hi_x, hi_y = cell_bounds(i, j, coords)
        assert group.geometry.x.between(lo_x, hi_x).all()
        assert group.geometry.y.between(lo_y, hi_y).all()


def test_generate_point_distribution_skips_masked_cells():
    """A land-sea mask must remove exactly the cells it marks as non-land."""
    coords = mini_coordinates("centered")
    mask = np.zeros((len(MINI_LATS), len(MINI_LONS)))
    mask[0, 0] = 1
    mask[2, 3] = 1

    point_gdf, cell_coords = generate_point_distribution(mask, coords, 4, "uniform")

    assert set(cell_coords) == {(0, 0), (2, 3)}
    assert len(point_gdf) == 2 * 4


def test_generate_point_distribution_rejects_unknown_distribution():
    coords = mini_coordinates("centered")
    with pytest.raises(ValueError, match="Unknown point distribution"):
        generate_point_distribution(None, coords, 4, "gaussian")


def test_generate_point_distribution_returns_geodataframe_in_wgs84():
    coords = mini_coordinates("centered")
    point_gdf, _ = generate_point_distribution(None, coords, 4, "uniform")

    assert point_gdf.crs == "EPSG:4326"
    assert isinstance(point_gdf.geometry.iloc[0], Point)
