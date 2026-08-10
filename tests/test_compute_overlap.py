"""End-to-end tests for the weight computation pipeline and its CLI."""

import os
import subprocess
import sys

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from factories import (
    MINI_EXPECTED_RESCUED,
    MINI_EXPECTED_SAMPLED,
    MINI_POINTS_PER_CELL,
    make_grid_nc,
    make_regions,
    mini_polygons,
    write_regions,
)
from netcdf2region.compute_grid_weights import compute_overlap, load_regions
from netcdf2region.validate_weights import validate_weights

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(netcdf, shapefile, output, **overrides):
    """Call ``compute_overlap`` with the mini defaults, overridable per test."""
    kwargs = dict(
        netcdf_path=netcdf,
        shapefile_path=shapefile,
        output_csv=str(output),
        points_per_cell=MINI_POINTS_PER_CELL,
        simplify=False,
        mask_nc_path=None,
        shapefile_layer=None,
        point_distribution="uniform",
        backend="threading",
        n_jobs=1,
        yes=True,
        coordinate_origin="centered",
        default_gid="GID_1",
        shift_longitudes=False,
    )
    kwargs.update(overrides)
    compute_overlap(**kwargs)
    return pd.read_csv(str(output))


# --------------------------------------------------------------------------- #
# The golden regression guard
# --------------------------------------------------------------------------- #

def test_output_matches_the_committed_golden_file(mini_weights_csv, golden_weights_path):
    """Uniform sampling is deterministic, so the CSV must reproduce byte for byte.

    This is the regression guard: any change to sampling, rounding, or the rescue
    path shows up here as a diff instead of silently altering the shipped weights.
    """
    with open(mini_weights_csv) as fresh, open(golden_weights_path) as golden:
        assert fresh.read() == golden.read()


def test_repeated_runs_are_identical(tmp_path, mini_grid_nc, mini_regions_geojson):
    first = _run(mini_grid_nc, mini_regions_geojson, tmp_path / "a.csv")
    second = _run(mini_grid_nc, mini_regions_geojson, tmp_path / "b.csv")

    pd.testing.assert_frame_equal(first, second)


# --------------------------------------------------------------------------- #
# Hand-computed expectations
# --------------------------------------------------------------------------- #

def test_sampled_weights_match_the_hand_computed_values(mini_weights_csv):
    """Every sampled weight in the mini case is derivable from the geometry."""
    df = pd.read_csv(mini_weights_csv)
    actual = {
        (row.region_id, row.lat, row.lon): row.percent_overlap
        for row in df.itertuples()
    }

    for key, expected in MINI_EXPECTED_SAMPLED.items():
        assert key in actual, f"missing weight for {key}"
        assert actual[key] == pytest.approx(expected), key


def test_unsampled_regions_are_rescued_with_exact_geometry(mini_weights_csv):
    df = pd.read_csv(mini_weights_csv)
    actual = {
        (row.region_id, row.lat, row.lon): row.percent_overlap
        for row in df.itertuples()
    }

    for key, expected in MINI_EXPECTED_RESCUED.items():
        assert key in actual, f"region {key[0]} was dropped instead of rescued"
        assert actual[key] == pytest.approx(expected, rel=1e-9), key


def test_every_region_in_the_shapefile_reaches_the_output(mini_weights_csv):
    df = pd.read_csv(mini_weights_csv)
    assert set(df["region_id"]) == set(mini_polygons())


def test_no_zero_weight_rows_are_written(mini_weights_csv):
    df = pd.read_csv(mini_weights_csv)
    assert (df["percent_overlap"] > 0).all()


def test_coordinates_are_grid_coordinates(mini_weights_csv, mini_coords):
    df = pd.read_csv(mini_weights_csv)
    assert set(df["lat"]) <= set(np.round(mini_coords["lat"], 3))
    assert set(df["lon"]) <= set(np.round(mini_coords["lon"], 3))


def test_output_passes_full_validation(mini_weights_csv, mini_coords, mini_regions_gdf):
    report = validate_weights(
        mini_weights_csv, coordinates=mini_coords, regions=mini_regions_gdf
    )
    assert report.ok, report.summary()


def test_partial_cell_weight_reflects_the_sampled_fraction(tmp_path, mini_grid_nc):
    """A vertical cut at 0.37 catches the 4 lattice columns below it, of 10."""
    regions = make_regions({"P": box(0.0, 0.0, 0.37, 1.0)})
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    df = _run(mini_grid_nc, shapefile, tmp_path / "w.csv")

    assert len(df) == 1
    assert df.iloc[0]["percent_overlap"] == 0.4


def test_sampled_weights_are_rounded_to_two_decimals(tmp_path, mini_grid_nc):
    """Only rescued rows carry more than 2 decimals; sampled ones are rounded.

    With a 3x3 lattice a single hit is 1/9 = 0.1111..., which must be written as
    0.11. A 10x10 lattice cannot show this, since every count/100 is already exact
    at 2 decimals.
    """
    regions = make_regions({"P": box(0.0, 0.0, 0.25, 0.25)})
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    df = _run(mini_grid_nc, shapefile, tmp_path / "w.csv", points_per_cell=9)

    assert len(df) == 1
    assert df.iloc[0]["percent_overlap"] == 0.11


# --------------------------------------------------------------------------- #
# Grid variants
# --------------------------------------------------------------------------- #

def test_shifted_longitudes_place_weights_on_the_right_cell(tmp_path):
    """An ERA5-style [0, 360] grid must land weights at the shifted coordinate."""
    grid = make_grid_nc(
        tmp_path / "g.nc",
        lats=np.arange(0.5, 6.0, 1.0),
        lons=np.arange(0.5, 360.0, 1.0),
    )
    regions = make_regions({"W": box(-10.0, 0.0, -9.0, 1.0)})
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    df = _run(grid, shapefile, tmp_path / "w.csv", shift_longitudes=True)

    assert len(df) == 1
    assert df.iloc[0]["lon"] == -9.5
    assert df.iloc[0]["lat"] == 0.5
    assert df.iloc[0]["percent_overlap"] == 1.0


def test_top_left_origin_offsets_the_cells(tmp_path, mini_grid_nc):
    """With edge coordinates, cell (0, 0) spans [0.5, 1.5] rather than [0, 1]."""
    regions = make_regions({"T": box(0.5, 0.5, 1.5, 1.5)})
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    df = _run(mini_grid_nc, shapefile, tmp_path / "w.csv", coordinate_origin="top-left")

    assert len(df) == 1
    assert (df.iloc[0]["lat"], df.iloc[0]["lon"]) == (0.5, 0.5)
    assert df.iloc[0]["percent_overlap"] == 1.0


def test_masked_cells_are_not_sampled_but_regions_are_still_rescued(tmp_path):
    """A land-sea mask removes cells from sampling, never regions from the output.

    ``assign_missing_regions`` works on true geometry, so a region sitting entirely
    in masked-out water still gets a weight rather than disappearing.
    """
    lats = np.arange(0.5, 40.0, 1.0)
    lons = np.arange(0.5, 40.0, 1.0)
    grid = make_grid_nc(tmp_path / "g.nc", lats=lats, lons=lons)

    mask_values = np.zeros((len(lats), len(lons)))
    mask_values[5, 5] = 1
    mask = make_grid_nc(
        tmp_path / "m.nc", lats=lats, lons=lons, var="lsm", values=mask_values
    )

    regions = make_regions({
        "OnLand": box(5.0, 5.0, 6.0, 6.0),
        "AtSea": box(30.0, 30.0, 31.0, 31.0),
    })
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    df = _run(
        grid, shapefile, tmp_path / "w.csv",
        mask_nc_path=mask, points_per_cell=4,
    )

    assert set(df["region_id"]) == {"OnLand", "AtSea"}
    at_sea = df[df["region_id"] == "AtSea"].iloc[0]
    assert (at_sea["lat"], at_sea["lon"]) == (30.5, 30.5)
    assert at_sea["percent_overlap"] == 1.0


def test_random_distribution_produces_valid_weights(tmp_path, mini_grid_nc, mini_regions_geojson):
    """Random sampling is not reproducible, so only invariants are asserted."""
    np.random.seed(7)
    df = _run(
        mini_grid_nc, mini_regions_geojson, tmp_path / "w.csv",
        point_distribution="random",
    )

    assert set(df["region_id"]) == set(mini_polygons())
    assert (df["percent_overlap"] > 0).all()
    assert (df["percent_overlap"] <= 1.0).all()
    # A whole-cell region must be found by any sensible sample.
    assert df[df["region_id"] == "A"].iloc[0]["percent_overlap"] == 1.0


# --------------------------------------------------------------------------- #
# load_regions
# --------------------------------------------------------------------------- #

def test_load_regions_renames_the_gid_column(tmp_path):
    shapefile = write_regions(make_regions(mini_polygons()), tmp_path / "r.geojson")

    regions = load_regions(shapefile, False, None, "GID_1", yes=True)

    assert "region_id" in regions.columns
    assert set(regions["region_id"]) == set(mini_polygons())


def test_load_regions_falls_back_to_gid_0(tmp_path):
    """Admin-0 shapefiles carry GID_0, and the default --gid is GID_1."""
    regions = make_regions({"DEU": box(0, 0, 1, 1)}, gid_col="GID_0")
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    loaded = load_regions(shapefile, False, None, "GID_1", yes=True)

    assert list(loaded["region_id"]) == ["DEU"]


def test_load_regions_uses_the_first_column_under_yes(tmp_path):
    regions = make_regions({"X": box(0, 0, 1, 1)}, gid_col="my_id")
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    loaded = load_regions(shapefile, False, None, "GID_1", yes=True)

    assert "region_id" in loaded.columns
    assert list(loaded["region_id"]) == ["X"]


def test_load_regions_reprojects_to_wgs84(tmp_path):
    """Weights are computed in degrees, so anything else must be reprojected."""
    regions = make_regions({"R": box(0, 0, 1, 1)}).to_crs("EPSG:3857")
    shapefile = write_regions(regions, tmp_path / "r.gpkg")

    loaded = load_regions(shapefile, False, None, "GID_1", yes=True)

    assert loaded.crs == "EPSG:4326"


def test_load_regions_reads_a_named_geopackage_layer(tmp_path):
    """GADM ships admin levels as layers, selected with --shapefile_layer."""
    path = str(tmp_path / "multi.gpkg")
    make_regions({"L0": box(0, 0, 1, 1)}, gid_col="GID_0").to_file(
        path, driver="GPKG", layer="level0"
    )
    make_regions({"L1": box(0, 0, 1, 1)}).to_file(path, driver="GPKG", layer="level1")

    level0 = load_regions(path, False, 0, "GID_1", yes=True)
    level1 = load_regions(path, False, 1, "GID_1", yes=True)

    assert list(level0["region_id"]) == ["L0"]
    assert list(level1["region_id"]) == ["L1"]


def test_load_regions_can_simplify_geometries(tmp_path):
    circle = gpd.GeoSeries([box(0, 0, 1, 1).buffer(0.5)], crs="EPSG:4326").iloc[0]
    shapefile = write_regions(make_regions({"C": circle}), tmp_path / "r.geojson")

    detailed = load_regions(shapefile, False, None, "GID_1", yes=True)
    simplified = load_regions(shapefile, True, None, "GID_1", yes=True)

    n_detailed = len(detailed.geometry.iloc[0].exterior.coords)
    n_simplified = len(simplified.geometry.iloc[0].exterior.coords)
    assert n_simplified < n_detailed


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #

def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "netcdf2region.compute_grid_weights", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_writes_the_same_output_as_the_api(tmp_path, mini_grid_nc,
                                               mini_regions_geojson, golden_weights_path):
    out = tmp_path / "cli.csv"
    result = _cli(
        mini_grid_nc, mini_regions_geojson, str(out),
        "--yes", "--n_jobs=1", f"--points_per_cell={MINI_POINTS_PER_CELL}",
    )

    assert result.returncode == 0, result.stderr
    with open(out) as produced, open(golden_weights_path) as golden:
        assert produced.read() == golden.read()


def test_cli_rejects_a_360_grid_without_the_shift_flag(tmp_path, mini_regions_geojson):
    grid = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))

    result = _cli(grid, mini_regions_geojson, str(tmp_path / "o.csv"), "--yes")

    assert result.returncode != 0
    assert "shift_longitudes" in result.stderr


def test_cli_accepts_a_geopackage_layer(tmp_path, mini_grid_nc):
    path = str(tmp_path / "multi.gpkg")
    make_regions({"L0": box(0, 0, 1, 1)}, gid_col="GID_0").to_file(
        path, driver="GPKG", layer="level0"
    )
    out = tmp_path / "o.csv"

    result = _cli(
        mini_grid_nc, path, str(out), "--yes", "--shapefile_layer=0", "--n_jobs=1",
    )

    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(out)["region_id"]) == ["L0"]


def test_cli_reports_a_missing_netcdf(tmp_path, mini_regions_geojson):
    result = _cli(
        str(tmp_path / "absent.nc"), mini_regions_geojson, str(tmp_path / "o.csv"), "--yes"
    )

    assert result.returncode != 0
    assert "NetCDF file not found" in result.stderr
