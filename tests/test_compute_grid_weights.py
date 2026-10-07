"""End-to-end tests for ``compute_weights`` and the ``compute_grid_weights`` command."""

import os
import subprocess
import sys

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
from netcdf2region.compute_grid_weights import compute_weights
from netcdf2region.validate_weights import validate_weights

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _run(netcdf, shapefile, output, **overrides):
    """Call ``compute_weights`` with the mini sampling defaults, overridable per test."""
    kwargs = dict(
        method="sampling",
        points_per_cell=MINI_POINTS_PER_CELL,
        point_distribution="uniform",
        n_jobs=1,
    )
    kwargs.update(overrides)
    compute_weights(netcdf, shapefile, str(output), **kwargs)
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


def test_coordinates_are_grid_coordinates(mini_weights_csv, mini_template_grid):
    df = pd.read_csv(mini_weights_csv)
    assert set(df["lat"]) <= set(np.round(mini_template_grid.lat, 3))
    assert set(df["lon"]) <= set(np.round(mini_template_grid.lon, 3))


def test_output_passes_full_validation(mini_weights_csv, mini_template_grid, mini_regions_gdf):
    report = validate_weights(
        mini_weights_csv, grid=mini_template_grid, regions=mini_regions_gdf
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
# compute_weights
# --------------------------------------------------------------------------- #

def test_exact_is_the_default_method(tmp_path, mini_grid_nc, mini_regions_geojson):
    df = compute_weights(mini_grid_nc, mini_regions_geojson, n_jobs=1)

    assert "cell_area_km2" in df.columns


def test_output_csv_is_optional(tmp_path, mini_grid_nc, mini_regions_geojson):
    df = compute_weights(mini_grid_nc, mini_regions_geojson, method="sampling", n_jobs=1)

    assert set(df["region_id"]) == set(mini_polygons())
    assert not list(tmp_path.iterdir())


def test_unknown_method_is_rejected(mini_grid_nc, mini_regions_geojson):
    with pytest.raises(ValueError, match="Unknown method"):
        compute_weights(mini_grid_nc, mini_regions_geojson, method="guess")


def _break_weights(monkeypatch):
    """Make the exact method return a weight above 1, which validation must reject."""
    import netcdf2region.compute_grid_weights as cgw
    real = cgw.exact_weights

    def broken(*args, **kwargs):
        df = real(*args, **kwargs)
        df.loc[0, "percent_overlap"] = 1.5
        return df

    monkeypatch.setattr(cgw, "exact_weights", broken)


def test_validation_runs_by_default_and_logs_the_report(mini_grid_nc, mini_regions_geojson, caplog):
    caplog.set_level("INFO", logger="netcdf2region.compute_grid_weights")
    compute_weights(mini_grid_nc, mini_regions_geojson, n_jobs=1)

    assert "Weights validation" in caplog.text
    assert "[PASS] region_coverage" in caplog.text


def test_failed_validation_raises_after_writing(tmp_path, monkeypatch, mini_grid_nc,
                                                mini_regions_geojson):
    _break_weights(monkeypatch)
    out = tmp_path / "w.csv"

    with pytest.raises(ValueError, match="weight_range"):
        compute_weights(mini_grid_nc, mini_regions_geojson, str(out), n_jobs=1)
    assert out.exists()


def test_validation_can_be_skipped(monkeypatch, mini_grid_nc, mini_regions_geojson):
    _break_weights(monkeypatch)

    df = compute_weights(mini_grid_nc, mini_regions_geojson, n_jobs=1, validate=False)

    assert df["percent_overlap"].max() == 1.5


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
        "--n-jobs=1", "--method=sampling", f"--points-per-cell={MINI_POINTS_PER_CELL}",
    )

    assert result.returncode == 0, result.stderr
    with open(out) as produced, open(golden_weights_path) as golden:
        assert produced.read() == golden.read()


def test_cli_validates_by_default_and_can_skip_it(tmp_path, mini_grid_nc, mini_regions_geojson):
    default = _cli(mini_grid_nc, mini_regions_geojson, str(tmp_path / "a.csv"), "--n-jobs=1")
    skipped = _cli(mini_grid_nc, mini_regions_geojson, str(tmp_path / "b.csv"), "--n-jobs=1",
                   "--no-validate")

    assert default.returncode == 0, default.stderr
    assert "Weights validation" in default.stderr
    assert skipped.returncode == 0, skipped.stderr
    assert "Weights validation" not in skipped.stderr


def test_cli_rejects_a_360_grid_without_the_shift_flag(tmp_path, mini_regions_geojson):
    grid = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))

    result = _cli(grid, mini_regions_geojson, str(tmp_path / "o.csv"))

    assert result.returncode != 0
    assert "shift-longitudes" in result.stderr


def test_cli_accepts_a_geopackage_layer(tmp_path, mini_grid_nc):
    path = str(tmp_path / "multi.gpkg")
    make_regions({"L0": box(0, 0, 1, 1)}, gid_col="GID_0").to_file(
        path, driver="GPKG", layer="level0"
    )
    out = tmp_path / "o.csv"

    result = _cli(
        mini_grid_nc, path, str(out), "--shapefile-layer=0", "--gid=GID_0", "--n-jobs=1",
    )

    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(out)["region_id"]) == ["L0"]


def test_cli_reports_a_missing_netcdf(tmp_path, mini_regions_geojson):
    result = _cli(
        str(tmp_path / "absent.nc"), mini_regions_geojson, str(tmp_path / "o.csv")
    )

    assert result.returncode != 0
    assert "NetCDF file not found" in result.stderr


def test_cli_accepts_the_old_underscore_spellings(tmp_path, mini_grid_nc,
                                                  mini_regions_geojson, golden_weights_path):
    """Options were spelled with underscores before; existing scripts must keep working."""
    out = tmp_path / "cli.csv"
    result = _cli(
        mini_grid_nc, mini_regions_geojson, str(out),
        "--n_jobs=1", "--method=sampling", f"--points_per_cell={MINI_POINTS_PER_CELL}",
    )

    assert result.returncode == 0, result.stderr
    with open(out) as produced, open(golden_weights_path) as golden:
        assert produced.read() == golden.read()
