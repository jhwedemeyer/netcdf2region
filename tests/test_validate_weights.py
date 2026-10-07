"""Tests for the weights validator.

Each check gets a passing and a failing case, so a rule that silently stops firing
is caught. This matters most for the checks that only run on the cluster
(``coords_on_grid``, ``region_coverage``, ``region_areas``), which would otherwise
never be exercised on a laptop.
"""

import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from factories import make_regions, make_weights_df, mini_grid, mini_polygons
from netcdf2region.validate_weights import ERROR, WARNING, Report, validate_weights

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GOOD_ROWS = [
    (0.5, 0.5, "A", 1.0),
    (0.5, 2.5, "B", 0.5),
    (2.5, 4.5, "C", 0.25),
]


def _check(report, name):
    """Return the named check from a report."""
    matches = [c for c in report.checks if c.name == name]
    assert matches, f"no check named {name} in {[c.name for c in report.checks]}"
    return matches[0]


def _regions_gdf(polygons=None):
    gdf = make_regions(polygons if polygons is not None else mini_polygons())
    return gdf.rename(columns={"GID_1": "region_id"})


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #

def test_accepts_a_dataframe():
    report = validate_weights(make_weights_df(GOOD_ROWS))

    assert report.ok
    assert report.source == "<dataframe>"


def test_accepts_a_path(tmp_path):
    path = tmp_path / "w.csv"
    make_weights_df(GOOD_ROWS).to_csv(path, index=False)

    report = validate_weights(str(path))

    assert report.ok
    assert report.source == str(path)


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError, match="Weights file not found"):
        validate_weights("/nonexistent/weights.csv")


def test_an_empty_file_is_an_error():
    report = validate_weights(make_weights_df([]))

    assert not report.ok
    assert _check(report, "non_empty").message == "weights file contains no rows"


def test_does_not_modify_the_input_dataframe():
    df = make_weights_df(GOOD_ROWS)
    before = df.copy()

    validate_weights(df)

    pd.testing.assert_frame_equal(df, before)


# --------------------------------------------------------------------------- #
# Check 1: schema
# --------------------------------------------------------------------------- #

def test_missing_column_is_reported_and_stops_further_checks():
    df = make_weights_df(GOOD_ROWS).drop(columns=["percent_overlap"])

    report = validate_weights(df)

    assert not report.ok
    assert "percent_overlap" in _check(report, "schema").message
    # Later checks index that column, so they must not have run.
    assert [c.name for c in report.checks] == ["schema"]


def test_unexpected_extra_column_is_reported():
    df = make_weights_df(GOOD_ROWS)
    df["notes"] = "x"

    report = validate_weights(df)

    assert not report.ok
    assert "notes" in _check(report, "schema").message


def test_non_numeric_weight_column_is_reported():
    df = make_weights_df(GOOD_ROWS)
    df["percent_overlap"] = ["a", "b", "c"]

    report = validate_weights(df)

    assert not report.ok
    assert "non-numeric" in _check(report, "schema").message


# --------------------------------------------------------------------------- #
# Check 2: missing values
# --------------------------------------------------------------------------- #

def test_nan_weight_is_an_error():
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "D", np.nan)])

    report = validate_weights(df)

    assert not report.ok
    assert not _check(report, "no_missing_values").passed


def test_blank_region_id_is_an_error():
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "   ", 0.5)])

    report = validate_weights(df)

    assert not report.ok
    assert "blank" in _check(report, "no_missing_values").message


# --------------------------------------------------------------------------- #
# Check 3: weight range
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("bad_weight", [0.0, -0.5, 1.5])
def test_weights_outside_the_unit_interval_are_errors(bad_weight):
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "D", bad_weight)])

    report = validate_weights(df)

    assert not report.ok
    assert not _check(report, "weight_range").passed


def test_the_rescue_floor_is_an_acceptable_weight():
    """1e-6 is what ``assign_missing_regions`` writes for a sub-pixel region."""
    df = make_weights_df(GOOD_ROWS + [(1.5, 6.5, "E", 1e-6)])

    report = validate_weights(df)

    assert report.ok
    assert _check(report, "weight_range").passed


# --------------------------------------------------------------------------- #
# Check 4: duplicates
# --------------------------------------------------------------------------- #

def test_duplicate_cell_region_pair_is_an_error():
    df = make_weights_df(GOOD_ROWS + [(0.5, 0.5, "A", 0.3)])

    report = validate_weights(df)

    assert not report.ok
    assert "duplicated" in _check(report, "no_duplicates").message


def test_the_same_cell_for_two_regions_is_fine():
    df = make_weights_df([(0.5, 0.5, "A", 0.4), (0.5, 0.5, "B", 0.6)])

    report = validate_weights(df)

    assert report.ok


# --------------------------------------------------------------------------- #
# Check 6: cell sums
# --------------------------------------------------------------------------- #

def test_cell_sum_above_one_is_a_warning_not_an_error():
    """Overlapping polygons are legitimate, so this must never fail a run."""
    df = make_weights_df([
        (0.5, 0.5, "ESH.1_1", 1.0),
        (0.5, 0.5, "MAR.26_1", 1.0),
    ])

    report = validate_weights(df)

    assert report.ok, "an over-subscribed cell must not be an error"
    check = _check(report, "cell_sums")
    assert not check.passed
    assert check.severity == WARNING
    assert check.details["max_sum"] == 2.0


def test_the_cell_sum_ceiling_is_configurable():
    df = make_weights_df([(0.5, 0.5, "A", 1.0), (0.5, 0.5, "B", 1.0)])

    assert not _check(validate_weights(df, max_cell_sum=1.01), "cell_sums").passed
    assert _check(validate_weights(df, max_cell_sum=2.0), "cell_sums").passed


def test_rounding_slack_does_not_trip_the_default_ceiling():
    """Three regions rounded to 2 decimals can sum to 1.02 in a fully covered cell."""
    df = make_weights_df([
        (0.5, 0.5, "A", 0.34), (0.5, 0.5, "B", 0.34), (0.5, 0.5, "C", 0.33),
    ])

    assert _check(validate_weights(df), "cell_sums").passed


# --------------------------------------------------------------------------- #
# Check 7: coordinates on the grid
# --------------------------------------------------------------------------- #

def test_coordinates_are_checked_against_the_template_grid():
    report = validate_weights(make_weights_df(GOOD_ROWS), grid=mini_grid())

    assert report.ok
    assert _check(report, "coords_on_grid").passed


def test_a_coordinate_off_the_grid_is_an_error():
    """This is what makes ``aggregate`` fail, so it must be caught here first."""
    df = make_weights_df(GOOD_ROWS + [(0.75, 0.5, "D", 0.5)])

    report = validate_weights(df, grid=mini_grid())

    assert not report.ok
    check = _check(report, "coords_on_grid")
    assert 0.75 in check.details["bad_lats"]


def test_the_grid_check_is_skipped_without_a_template():
    report = validate_weights(make_weights_df(GOOD_ROWS))

    check = _check(report, "coords_on_grid")
    assert check.skipped
    assert report.ok


# --------------------------------------------------------------------------- #
# Check 8: region coverage
# --------------------------------------------------------------------------- #

def test_a_region_absent_from_the_weights_is_an_error():
    """The exact failure ``assign_missing_regions`` was written to prevent."""
    df = make_weights_df(GOOD_ROWS)  # covers A, B, C but not D or E

    report = validate_weights(df, regions=_regions_gdf())

    assert not report.ok
    check = _check(report, "region_coverage")
    assert set(check.details["missing"]) == {"D", "E"}


def test_a_region_not_in_the_shapefile_is_an_error():
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "GHOST", 0.5)])

    report = validate_weights(df, regions=_regions_gdf({"A": box(0, 0, 1, 1)}))

    assert not report.ok
    assert "GHOST" in _check(report, "region_coverage").details["unexpected"]


def test_full_coverage_passes():
    rows = [(0.5, 0.5, name, 0.5) for name in mini_polygons()]
    report = validate_weights(make_weights_df(rows), regions=_regions_gdf())

    assert _check(report, "region_coverage").passed


def test_regions_can_be_supplied_as_a_path(tmp_path):
    from factories import write_regions

    path = write_regions(make_regions(mini_polygons()), tmp_path / "r.geojson")
    rows = [(0.5, 0.5, name, 0.5) for name in mini_polygons()]

    report = validate_weights(make_weights_df(rows), regions=path)

    assert _check(report, "region_coverage").passed


# --------------------------------------------------------------------------- #
# Check 9: region areas
# --------------------------------------------------------------------------- #

def test_area_reconstruction_passes_for_correct_weights(mini_weights_csv):
    """The pipeline's own output must reconstruct its regions' areas."""
    report = validate_weights(
        mini_weights_csv, grid=mini_grid(), regions=_regions_gdf()
    )

    assert report.ok
    # E is a speck kept alive by the 1e-6 floor, which says nothing about its size,
    # so it must be skipped. Everything else must be within tolerance.
    check = _check(report, "region_areas")
    assert check.passed, check.message
    assert check.details["n_skipped"] == 1
    assert check.details["n_regions"] == 4


def test_badly_scaled_weights_are_flagged_as_an_area_warning():
    """A region given ten times its true footprint must be noticed."""
    rows = [
        (0.5, 0.5, "A", 1.0),
        (1.5, 0.5, "A", 1.0), (2.5, 0.5, "A", 1.0), (3.5, 0.5, "A", 1.0),
        (4.5, 0.5, "A", 1.0), (0.5, 1.5, "A", 1.0), (1.5, 1.5, "A", 1.0),
    ]
    report = validate_weights(
        make_weights_df(rows),
        grid=mini_grid(),
        regions=_regions_gdf({"A": box(0, 0, 1, 1)}),
    )

    check = _check(report, "region_areas")
    assert not check.passed
    assert check.severity == WARNING
    assert "A" in check.details["worst_regions"]


def test_the_area_check_needs_both_a_grid_and_regions():
    report = validate_weights(make_weights_df(GOOD_ROWS), regions=_regions_gdf())

    assert _check(report, "region_areas").skipped


# --------------------------------------------------------------------------- #
# Check 10: statistics
# --------------------------------------------------------------------------- #

def test_statistics_counts_rescued_rows():
    """Rescued rows carry more than 2 decimals; sampled ones do not."""
    df = make_weights_df(GOOD_ROWS + [(1.5, 6.5, "E", 1e-6), (4.5, 7.5, "D", 0.0004)])

    details = _check(validate_weights(df), "statistics").details

    assert details["n_rescued_rows"] == 2
    assert details["n_regions"] == 5
    assert details["n_rows"] == 5


def test_statistics_never_fails_a_run():
    report = validate_weights(make_weights_df(GOOD_ROWS))

    assert _check(report, "statistics").passed
    assert report.ok


# --------------------------------------------------------------------------- #
# Report behaviour
# --------------------------------------------------------------------------- #

def test_raise_if_failed_is_silent_when_ok():
    validate_weights(make_weights_df(GOOD_ROWS)).raise_if_failed()


def test_raise_if_failed_names_the_broken_checks():
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "D", 5.0)])

    with pytest.raises(ValueError, match="weight_range"):
        validate_weights(df).raise_if_failed()


def test_warnings_do_not_make_a_report_fail():
    df = make_weights_df([(0.5, 0.5, "A", 1.0), (0.5, 0.5, "B", 1.0)])
    report = validate_weights(df)

    assert report.ok
    assert len(report.warnings) == 1
    assert report.failures == []


def test_summary_mentions_every_check():
    report = validate_weights(make_weights_df(GOOD_ROWS))
    summary = report.summary()

    for check in report.checks:
        assert check.name in summary
    assert "OK" in summary


def test_summary_of_a_failing_report_says_failed():
    df = make_weights_df(GOOD_ROWS + [(1.5, 1.5, "D", -1.0)])

    assert "FAILED" in validate_weights(df).summary()


def test_empty_report_is_ok():
    assert Report().ok


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #

def _cli(*args):
    return subprocess.run(
        [sys.executable, "-m", "netcdf2region.validate_weights", *args],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )


def test_cli_exits_zero_for_a_valid_file(tmp_path):
    path = tmp_path / "w.csv"
    make_weights_df(GOOD_ROWS).to_csv(path, index=False)

    result = _cli(str(path))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "OK" in result.stdout


def test_cli_exits_nonzero_for_an_invalid_file(tmp_path):
    path = tmp_path / "w.csv"
    make_weights_df(GOOD_ROWS + [(1.5, 1.5, "D", 9.0)]).to_csv(path, index=False)

    result = _cli(str(path))

    assert result.returncode == 1
    assert "FAILED" in result.stdout


def test_cli_strict_mode_fails_on_warnings(tmp_path):
    path = tmp_path / "w.csv"
    make_weights_df([
        (0.5, 0.5, "A", 1.0), (0.5, 0.5, "B", 1.0),
    ]).to_csv(path, index=False)

    assert _cli(str(path)).returncode == 0
    assert _cli(str(path), "--strict").returncode == 1


def test_cli_checks_against_a_template_grid(tmp_path, mini_grid_nc):
    path = tmp_path / "w.csv"
    make_weights_df(GOOD_ROWS + [(0.75, 0.5, "D", 0.5)]).to_csv(path, index=False)

    result = _cli(str(path), "--netcdf", mini_grid_nc)

    assert result.returncode == 1
    assert "coords_on_grid" in result.stdout


def test_cli_checks_region_coverage(tmp_path, mini_regions_geojson):
    path = tmp_path / "w.csv"
    make_weights_df(GOOD_ROWS).to_csv(path, index=False)

    result = _cli(str(path), "--shapefile", mini_regions_geojson)

    assert result.returncode == 1
    assert "region_coverage" in result.stdout


def test_cli_validates_the_pipeline_output_end_to_end(
    mini_weights_csv, mini_grid_nc, mini_regions_geojson
):
    """The command the SLURM scripts run must pass on freshly computed weights."""
    result = _cli(
        mini_weights_csv,
        "--netcdf", mini_grid_nc,
        "--shapefile", mini_regions_geojson,
    )

    assert result.returncode == 0, result.stdout + result.stderr
