"""Tests for the two functions that turn geometry into weights.

``calc_cell_region_overlap`` is the sampling estimator; ``assign_missing_regions``
is the exact-geometry rescue path added to stop small regions from vanishing.
"""

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import MultiPolygon, Point, box

from factories import make_regions, mini_coordinates, mini_polygons
from netcdf2region.sampled_weights import (
    assign_missing_regions,
    calc_cell_region_overlap,
    generate_point_distribution,
)

import geopandas as gpd


def _points(pairs):
    """Build the point GeoDataFrame ``calc_cell_region_overlap`` expects."""
    return gpd.GeoDataFrame(
        {
            "geometry": [Point(x, y) for (x, y), _ in pairs],
            "cell": [cell for _, cell in pairs],
        },
        crs="EPSG:4326",
    )


def _regions(polygons):
    return make_regions(polygons).rename(columns={"GID_1": "region_id"})


# --------------------------------------------------------------------------- #
# calc_cell_region_overlap
# --------------------------------------------------------------------------- #

def test_overlap_fraction_is_hits_over_points_per_cell():
    """Four of ten points inside the region must give a weight of 0.4."""
    inside = [((0.1 * k + 0.05, 0.5), (0, 0)) for k in range(4)]
    outside = [((5.0 + 0.1 * k, 0.5), (0, 0)) for k in range(6)]
    point_gdf = _points(inside + outside)
    regions = _regions({"R": box(0.0, 0.0, 1.0, 1.0)})

    grouped = calc_cell_region_overlap(point_gdf, regions, 10, n_jobs=1, backend="threading")

    assert len(grouped) == 1
    assert grouped.iloc[0]["region_id"] == "R"
    assert grouped.iloc[0]["count"] == 4
    assert np.isclose(grouped.iloc[0]["percent_overlap"], 0.4)


def test_points_outside_every_region_are_dropped():
    """The left join yields NaN region ids for ocean points; they must not survive."""
    point_gdf = _points([((50.0, 50.0), (0, 0)), ((0.5, 0.5), (0, 0))])
    regions = _regions({"R": box(0.0, 0.0, 1.0, 1.0)})

    grouped = calc_cell_region_overlap(point_gdf, regions, 2, n_jobs=1, backend="threading")

    assert list(grouped["region_id"]) == ["R"]
    assert grouped["region_id"].isna().sum() == 0


def test_a_point_in_two_overlapping_regions_counts_for_both():
    """Overlapping source polygons are legitimate - DOSE has them."""
    point_gdf = _points([((0.5, 0.5), (0, 0))])
    regions = _regions({"R1": box(0, 0, 1, 1), "R2": box(0, 0, 1, 1)})

    grouped = calc_cell_region_overlap(point_gdf, regions, 1, n_jobs=1, backend="threading")

    assert set(grouped["region_id"]) == {"R1", "R2"}
    assert grouped["percent_overlap"].sum() == 2.0


def test_result_is_independent_of_the_number_of_jobs():
    """Batching must not change the answer, only how fast it arrives."""
    coords = mini_coordinates()
    point_gdf, _ = generate_point_distribution(coords, 25, "uniform")
    regions = _regions(mini_polygons())

    serial = calc_cell_region_overlap(point_gdf, regions, 25, n_jobs=1, backend="threading")
    parallel = calc_cell_region_overlap(point_gdf, regions, 25, n_jobs=4, backend="threading")

    sort = lambda df: df.sort_values(["cell", "region_id"]).reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(sort(serial), sort(parallel))


def test_result_is_independent_of_the_batch_count():
    coords = mini_coordinates()
    point_gdf, _ = generate_point_distribution(coords, 16, "uniform")
    regions = _regions(mini_polygons())

    few = calc_cell_region_overlap(
        point_gdf, regions, 16, n_jobs=1, backend="threading", n_batches=1
    )
    many = calc_cell_region_overlap(
        point_gdf, regions, 16, n_jobs=1, backend="threading", n_batches=17
    )

    sort = lambda df: df.sort_values(["cell", "region_id"]).reset_index(drop=True)  # noqa: E731
    pd.testing.assert_frame_equal(sort(few), sort(many))


# --------------------------------------------------------------------------- #
# assign_missing_regions
# --------------------------------------------------------------------------- #

def test_no_rows_when_every_region_was_already_sampled():
    coords = mini_coordinates()
    regions = _regions({"A": box(0, 0, 1, 1)})

    result = assign_missing_regions({"A"}, regions, coords)

    assert result.empty
    assert list(result.columns) == ["lat", "lon", "region_id", "percent_overlap"]


def test_missing_region_is_rescued_with_its_exact_area_fraction():
    """A region covering exactly half a cell must be rescued at 0.5, not sampled."""
    coords = mini_coordinates()
    regions = _regions({"H": box(0.0, 0.0, 0.5, 1.0)})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 1
    row = result.iloc[0]
    assert (row["lat"], row["lon"]) == (0.5, 0.5)
    assert row["percent_overlap"] == 0.5


def test_sub_cell_sliver_is_rescued_rather_than_dropped():
    """This is the failure the rescue path exists for: too small to ever be sampled."""
    coords = mini_coordinates()
    regions = _regions({"D": box(7.20, 4.20, 7.22, 4.22)})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 1
    row = result.iloc[0]
    assert (row["lat"], row["lon"]) == (4.5, 7.5)
    assert row["percent_overlap"] == pytest.approx(0.0004)


def test_a_speck_too_small_to_round_gets_the_positive_floor():
    """Every overlap rounding to zero must still leave the region represented.

    Without the floor the region would carry no weight at all and downstream
    weighted means would divide by zero.
    """
    coords = mini_coordinates()
    regions = _regions({"E": box(6.30, 1.30, 6.3005, 1.3005)})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 1, "exactly the largest-overlap cell is kept"
    assert result.iloc[0]["percent_overlap"] == 1e-6


def test_rescued_weights_never_include_a_zero():
    """Zero-weight rows are useless downstream and must be filtered out."""
    coords = mini_coordinates()
    regions = _regions(mini_polygons())

    result = assign_missing_regions(set(), regions, coords)

    assert (result["percent_overlap"] > 0).all()


def test_region_spanning_several_cells_is_split_across_them():
    coords = mini_coordinates()
    regions = _regions({"C": box(4.5, 2.5, 5.5, 3.5)})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 4
    assert set(zip(result["lat"], result["lon"])) == {
        (2.5, 4.5), (2.5, 5.5), (3.5, 4.5), (3.5, 5.5)
    }
    assert np.allclose(result["percent_overlap"], 0.25)
    assert result["percent_overlap"].sum() == pytest.approx(1.0)


def test_multipart_region_is_dissolved_not_double_counted():
    """Two rows sharing a region id, and a MultiPolygon, must give one weight per cell."""
    coords = mini_coordinates()
    multi = MultiPolygon([box(0.0, 0.0, 0.5, 1.0), box(2.0, 0.0, 2.5, 1.0)])
    regions = _regions({"M": multi})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 2
    assert result["region_id"].tolist() == ["M", "M"]
    assert np.allclose(sorted(result["percent_overlap"]), [0.5, 0.5])


def test_duplicate_rows_for_one_region_are_merged():
    coords = mini_coordinates()
    regions = gpd.GeoDataFrame(
        {"region_id": ["M", "M"]},
        geometry=[box(0.0, 0.0, 0.5, 1.0), box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:4326",
    )

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 1, "the two parts must be unioned, not counted twice"
    assert result.iloc[0]["percent_overlap"] == 1.0


def test_region_outside_the_grid_produces_no_rows():
    coords = mini_coordinates()
    regions = _regions({"Far": box(100.0, 50.0, 101.0, 51.0)})

    result = assign_missing_regions(set(), regions, coords)

    assert result.empty


def test_empty_and_none_geometries_are_skipped():
    coords = mini_coordinates()
    regions = gpd.GeoDataFrame(
        {"region_id": ["Empty", "Real"]},
        geometry=[box(0, 0, 0, 0), box(0.0, 0.0, 1.0, 1.0)],
        crs="EPSG:4326",
    )

    result = assign_missing_regions(set(), regions, coords)

    assert list(result["region_id"]) == ["Real"]


def test_only_regions_absent_from_present_ids_are_considered():
    coords = mini_coordinates()
    regions = _regions({"A": box(0, 0, 1, 1), "B": box(2.0, 0.0, 2.5, 1.0)})

    result = assign_missing_regions({"A"}, regions, coords)

    assert set(result["region_id"]) == {"B"}


def test_rescue_works_on_a_top_left_grid():
    """The rescue path must mirror the cell layout sampling uses."""
    coords = mini_coordinates("top-left")
    # Cell (0, 0) of a top-left grid spans lon [0.5, 1.5], lat [0.5, 1.5].
    regions = _regions({"T": box(0.5, 0.5, 1.0, 1.5)})

    result = assign_missing_regions(set(), regions, coords)

    assert len(result) == 1
    assert (result.iloc[0]["lat"], result.iloc[0]["lon"]) == (0.5, 0.5)
    assert result.iloc[0]["percent_overlap"] == 0.5


def test_coordinates_are_rounded_to_three_decimals():
    """The CSV writes 3 decimals, and ``aggregate`` matches on exact equality."""
    coords = {
        "lat": np.array([0.123456, 1.123456]),
        "lon": np.array([0.654321, 1.654321]),
        "lat_res": 1.0,
        "lon_res": 1.0,
        "coordinate_origin": "centered",
    }
    regions = _regions({"R": box(0.2, -0.3, 0.6, 0.5)})

    result = assign_missing_regions(set(), regions, coords)

    assert result.iloc[0]["lat"] == 0.123
    assert result.iloc[0]["lon"] == 0.654
