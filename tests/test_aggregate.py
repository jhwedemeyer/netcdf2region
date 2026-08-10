"""Tests for the aggregation side: turning a weights CSV plus a NetCDF into a table.

The mini temperature field holds ``100*t + 10*i + j`` at cell (i, j) and time t, so
every expected aggregate below is arithmetic rather than a recorded number.
"""

import os

import numpy as np
import pandas as pd
import pytest

from factories import MINI_LATS, MINI_LONS, make_grid_nc, make_weights_df
from netcdf2region.netcdf2region import (
    aggregate,
    correct_weights,
    load_netcdf,
    precalc_weights_path,
    validate_secondary_weights,
)


def _value(t, i, j):
    """The value the mini temperature field holds at time t, cell (i, j)."""
    return 100 * t + 10 * i + j


def _weights_file(tmp_path, rows, name="w.csv"):
    path = tmp_path / name
    make_weights_df(rows).to_csv(path, index=False)
    return str(path)


def _aggregate(tmp_path, netcdf, weights, method="mean", var="tas", **kwargs):
    out = tmp_path / "out.csv"
    aggregate(
        netcdf=netcdf, weights=weights, method=method, output=str(out), var=var, **kwargs
    )
    return pd.read_csv(out)


# --------------------------------------------------------------------------- #
# load_netcdf
# --------------------------------------------------------------------------- #

def test_load_netcdf_returns_values_and_coordinates(mini_tas_nc):
    values, lats, lons, times = load_netcdf(mini_tas_nc, "tas")

    assert values.shape == (3, len(MINI_LATS), len(MINI_LONS))
    assert np.array_equal(lats, MINI_LATS)
    assert np.array_equal(lons, MINI_LONS)
    assert len(times) == 3
    assert values[2, 3, 4] == _value(2, 3, 4)


def test_load_netcdf_accepts_latitude_longitude_naming(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc", n_time=2, lat_name="latitude", lon_name="longitude"
    )
    values, lats, lons, _ = load_netcdf(path, "tas")

    assert values.shape == (2, len(MINI_LATS), len(MINI_LONS))
    assert np.array_equal(lats, MINI_LATS)


def test_load_netcdf_transposes_to_time_lat_lon(tmp_path):
    """Downstream indexing assumes (time, lat, lon) regardless of storage order."""
    import xarray as xr

    values = np.arange(2 * len(MINI_LATS) * len(MINI_LONS), dtype=float)
    values = values.reshape(len(MINI_LATS), len(MINI_LONS), 2)
    ds = xr.Dataset(
        {"tas": (("lat", "lon", "time"), values)},
        coords={
            "lat": MINI_LATS, "lon": MINI_LONS,
            "time": pd.date_range("2001-01-01", periods=2, freq="YS"),
        },
    )
    path = str(tmp_path / "g.nc")
    ds.to_netcdf(path)
    ds.close()

    loaded, _, _, _ = load_netcdf(path, "tas")

    assert loaded.shape == (2, len(MINI_LATS), len(MINI_LONS))


def test_load_netcdf_allows_a_spatial_only_variable(mini_pop_nc):
    values, _, _, times = load_netcdf(mini_pop_nc, "pop", require_time=False)

    assert values.shape == (len(MINI_LATS), len(MINI_LONS))
    assert times is None


def test_load_netcdf_rejects_a_spatial_only_variable_when_time_is_required(mini_pop_nc):
    with pytest.raises(ValueError, match="has no time dimension"):
        load_netcdf(mini_pop_nc, "pop", require_time=True)


def test_load_netcdf_rejects_an_unknown_variable(mini_tas_nc):
    with pytest.raises(KeyError, match="not found"):
        load_netcdf(mini_tas_nc, "does_not_exist")


def test_load_netcdf_rejects_an_unknown_time_variable(mini_tas_nc):
    with pytest.raises(KeyError, match="not found"):
        load_netcdf(mini_tas_nc, "tas", time_var="epoch")


def test_load_netcdf_cannot_guess_between_two_non_spatial_dims(tmp_path):
    """Ambiguity must be reported, with the flag that resolves it."""
    import xarray as xr

    values = np.zeros((2, 3, len(MINI_LATS), len(MINI_LONS)))
    ds = xr.Dataset(
        {"tas": (("time", "level", "lat", "lon"), values)},
        coords={
            "time": pd.date_range("2001-01-01", periods=2, freq="YS"),
            "level": [1, 2, 3], "lat": MINI_LATS, "lon": MINI_LONS,
        },
    )
    path = str(tmp_path / "g.nc")
    ds.to_netcdf(path)
    ds.close()

    with pytest.raises(ValueError, match="time_var"):
        load_netcdf(path, "tas")


# --------------------------------------------------------------------------- #
# correct_weights
# --------------------------------------------------------------------------- #

def test_correct_weights_rescales_a_partly_covered_cell():
    """A cell 50% covered by land must have its weights scaled up to sum to 1."""
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.3),
        (0.5, 0.5, "R2", 0.2),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(sorted(corrected["percent_overlap"]), [0.4, 0.6])
    assert corrected["percent_overlap"].sum() == pytest.approx(1.0)


def test_correct_weights_leaves_a_fully_covered_cell_alone():
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.75),
        (0.5, 0.5, "R2", 0.25),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [0.75, 0.25])


def test_correct_weights_scales_overlapping_regions_down():
    """Overlapping source polygons make cell sums exceed 1, and this divides by them.

    The shipped DOSE weights contain cells summing to exactly 2.0 where two regions
    both claim the cell in full. Correcting such a file halves both weights, which
    is a real consequence of using --correct_weights with overlapping regions.
    """
    weights = make_weights_df([
        (0.5, 0.5, "ESH.1_1", 1.0),
        (0.5, 0.5, "MAR.26_1", 1.0),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [0.5, 0.5])


def test_correct_weights_treats_cells_independently():
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.5),
        (1.5, 1.5, "R2", 0.25),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [1.0, 1.0])


def test_correct_weights_drops_its_helper_column():
    corrected = correct_weights(make_weights_df([(0.5, 0.5, "R", 0.5)]))
    assert list(corrected.columns) == ["lat", "lon", "region_id", "percent_overlap"]


# --------------------------------------------------------------------------- #
# aggregate - core methods
# --------------------------------------------------------------------------- #

def test_mean_of_a_single_full_cell_is_that_cell(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])

    df = _aggregate(tmp_path, mini_tas_nc, weights)

    assert list(df.columns) == ["region_id", "time", "tas"]
    assert len(df) == 3
    assert list(df["tas"]) == [_value(t, 0, 0) for t in range(3)]


def test_mean_is_weighted_not_arithmetic(tmp_path, mini_tas_nc):
    """Cell (0,0)=0 at weight 0.25 and cell (0,4)=4 at weight 0.75 -> 3.0, not 2.0."""
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.25),
        (MINI_LATS[0], MINI_LONS[4], "R", 0.75),
    ])

    df = _aggregate(tmp_path, mini_tas_nc, weights)

    expected = 0.25 * _value(0, 0, 0) + 0.75 * _value(0, 0, 4)
    assert df.iloc[0]["tas"] == pytest.approx(expected)
    assert df.iloc[0]["tas"] == pytest.approx(3.0)


def test_sum_multiplies_values_by_weights(tmp_path, mini_tas_nc):
    """Unlike mean, sum does not normalise by the total weight."""
    weights = _weights_file(tmp_path, [
        (MINI_LATS[1], MINI_LONS[2], "R", 0.5),
        (MINI_LATS[1], MINI_LONS[3], "R", 0.5),
    ])

    df = _aggregate(tmp_path, mini_tas_nc, weights, method="sum")

    expected = 0.5 * _value(0, 1, 2) + 0.5 * _value(0, 1, 3)
    assert df.iloc[0]["tas"] == pytest.approx(expected)


def test_std_is_the_weighted_standard_deviation(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.5),
        (MINI_LATS[0], MINI_LONS[2], "R", 0.5),
    ])

    df = _aggregate(tmp_path, mini_tas_nc, weights, method="std")

    values = np.array([_value(0, 0, 0), _value(0, 0, 2)])
    assert df.iloc[0]["tas"] == pytest.approx(values.std())


def test_std_of_a_uniform_region_is_zero(tmp_path):
    """Two cells holding the same value must have zero spread."""
    values = np.full((1, len(MINI_LATS), len(MINI_LONS)), 7.0)
    netcdf = make_grid_nc(tmp_path / "flat.nc", n_time=1, values=values)
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.5),
        (MINI_LATS[1], MINI_LONS[1], "R", 0.5),
    ])

    df = _aggregate(tmp_path, netcdf, weights, method="std")

    assert df.iloc[0]["tas"] == pytest.approx(0.0)


def test_unknown_method_is_rejected(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "R", 1.0)])

    with pytest.raises(ValueError, match="Unknown aggregation method"):
        _aggregate(tmp_path, mini_tas_nc, weights, method="median")


def test_every_region_appears_for_every_time_step(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "A", 1.0),
        (MINI_LATS[1], MINI_LONS[1], "B", 1.0),
        (MINI_LATS[2], MINI_LONS[2], "C", 1.0),
    ])

    df = _aggregate(tmp_path, mini_tas_nc, weights)

    assert len(df) == 3 * 3
    assert set(df["region_id"]) == {"A", "B", "C"}
    assert df.groupby("region_id").size().tolist() == [3, 3, 3]


def test_time_column_carries_the_netcdf_time_stamps(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])

    df = _aggregate(tmp_path, mini_tas_nc, weights)

    assert list(pd.to_datetime(df["time"]).dt.year) == [2001, 2002, 2003]


def test_correct_weights_flag_changes_the_result(tmp_path, mini_tas_nc):
    """A half-covered cell aggregates the same, but a two-region cell does not."""
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.25),
        (MINI_LATS[0], MINI_LONS[4], "R", 0.25),
    ])

    plain = _aggregate(tmp_path, mini_tas_nc, weights, method="sum")
    corrected = _aggregate(
        tmp_path, mini_tas_nc, weights, method="sum", correct_weights_flag=True
    )

    assert plain.iloc[0]["tas"] == pytest.approx(0.25 * (0 + 4))
    assert corrected.iloc[0]["tas"] == pytest.approx(1.0 * (0 + 4))


# --------------------------------------------------------------------------- #
# aggregate - error reporting
# --------------------------------------------------------------------------- #

def test_weights_off_the_grid_give_an_actionable_error(tmp_path, mini_tas_nc):
    """Using the wrong weights file for a grid is the most likely user error.

    It used to surface as a bare IndexError from deep inside the loop.
    """
    weights = _weights_file(tmp_path, [(99.5, 0.5, "R", 1.0)])

    with pytest.raises(ValueError, match="do not exist in"):
        _aggregate(tmp_path, mini_tas_nc, weights)


def test_the_error_names_the_offending_region_and_cell(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(0.5, 0.5, "Good", 1.0), (0.5, 12.5, "Bad", 1.0)])

    with pytest.raises(ValueError) as excinfo:
        _aggregate(tmp_path, mini_tas_nc, weights)

    message = str(excinfo.value)
    assert "Bad" in message
    assert "12.5" in message
    assert "validate_weights" in message


# --------------------------------------------------------------------------- #
# aggregate - secondary weights
# --------------------------------------------------------------------------- #

def test_uniform_secondary_weights_do_not_change_the_mean(tmp_path, mini_tas_nc):
    flat = make_grid_nc(
        tmp_path / "flat_pop.nc", var="pop",
        values=np.full((len(MINI_LATS), len(MINI_LONS)), 5.0),
    )
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.5),
        (MINI_LATS[0], MINI_LONS[4], "R", 0.5),
    ])

    plain = _aggregate(tmp_path, mini_tas_nc, weights)
    weighted = _aggregate(
        tmp_path, mini_tas_nc, weights,
        secondary_weights_var="pop", secondary_weights_netcdf=flat,
    )

    assert weighted.iloc[0]["tas"] == pytest.approx(plain.iloc[0]["tas"])


def test_secondary_weights_pull_the_mean_towards_the_populated_cell(tmp_path, mini_tas_nc):
    pop = np.ones((len(MINI_LATS), len(MINI_LONS)))
    pop[0, 4] = 999.0
    pop_nc = make_grid_nc(tmp_path / "pop.nc", var="pop", values=pop)
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 0.5),
        (MINI_LATS[0], MINI_LONS[4], "R", 0.5),
    ])

    df = _aggregate(
        tmp_path, mini_tas_nc, weights,
        secondary_weights_var="pop", secondary_weights_netcdf=pop_nc,
    )

    expected = (0.5 * 1 * _value(0, 0, 0) + 0.5 * 999 * _value(0, 0, 4)) / (0.5 + 0.5 * 999)
    assert df.iloc[0]["tas"] == pytest.approx(expected)
    assert df.iloc[0]["tas"] > 3.9  # dominated by cell (0, 4), whose value is 4


def test_nan_secondary_weights_do_not_poison_the_result(tmp_path, mini_tas_nc, mini_pop_nc):
    """NaN cells are documented as being replaced by 1.0; the result must be finite.

    The replacement previously assigned to a local inside the validator, so the
    NaNs survived and every affected region aggregated to NaN.
    """
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "HasNaN", 1.0),
        (MINI_LATS[1], MINI_LONS[1], "Clean", 1.0),
    ])

    df = _aggregate(
        tmp_path, mini_tas_nc, weights,
        secondary_weights_var="pop", secondary_weights_netcdf=mini_pop_nc,
    )

    assert df["tas"].notna().all(), "NaN secondary weights leaked into the output"
    nan_region = df[df["region_id"] == "HasNaN"]
    assert list(nan_region["tas"]) == [_value(t, 0, 0) for t in range(3)]


def test_validate_secondary_weights_returns_the_cleaned_array():
    data = np.array([[1.0, np.nan], [3.0, 4.0]])

    cleaned = validate_secondary_weights(
        lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]),
        secondary_data=data, secondary_lats=np.array([0.0, 1.0]),
        secondary_lons=np.array([0.0, 1.0]), secondary_weights_var="pop",
    )

    assert not np.isnan(cleaned).any()
    assert cleaned[0, 1] == 1.0


def test_secondary_weights_on_a_different_grid_are_rejected(tmp_path, mini_tas_nc):
    pop_nc = make_grid_nc(
        tmp_path / "pop.nc", var="pop",
        lats=MINI_LATS + 10.0, lons=MINI_LONS,
    )
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "R", 1.0)])

    with pytest.raises(ValueError, match="Latitude coordinates mismatch"):
        _aggregate(
            tmp_path, mini_tas_nc, weights,
            secondary_weights_var="pop", secondary_weights_netcdf=pop_nc,
        )


def test_secondary_weights_with_a_single_timestep_are_squeezed(tmp_path, mini_tas_nc):
    pop_nc = make_grid_nc(
        tmp_path / "pop.nc", var="pop", n_time=1,
        values=np.ones((1, len(MINI_LATS), len(MINI_LONS))),
    )
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "R", 1.0)])

    df = _aggregate(
        tmp_path, mini_tas_nc, weights,
        secondary_weights_var="pop", secondary_weights_netcdf=pop_nc,
    )

    assert df["tas"].notna().all()


def test_secondary_weights_with_multiple_timesteps_are_rejected(tmp_path, mini_tas_nc):
    pop_nc = make_grid_nc(
        tmp_path / "pop.nc", var="pop", n_time=3,
        values=np.ones((3, len(MINI_LATS), len(MINI_LONS))),
    )
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "R", 1.0)])

    with pytest.raises(ValueError, match="time dimension must be 1"):
        _aggregate(
            tmp_path, mini_tas_nc, weights,
            secondary_weights_var="pop", secondary_weights_netcdf=pop_nc,
        )


def test_integer_weights_are_not_truncated_by_secondary_weighting(tmp_path, mini_tas_nc):
    """An all-integer percent_overlap column reads back as int64.

    Multiplying in place would then truncate the secondary weighting to zero.
    """
    pop = np.full((len(MINI_LATS), len(MINI_LONS)), 0.5)
    pop_nc = make_grid_nc(tmp_path / "pop.nc", var="pop", values=pop)
    weights = _weights_file(tmp_path, [
        (MINI_LATS[0], MINI_LONS[0], "R", 1),
        (MINI_LATS[0], MINI_LONS[4], "R", 1),
    ])
    assert pd.read_csv(weights)["percent_overlap"].dtype.kind == "i"

    df = _aggregate(
        tmp_path, mini_tas_nc, weights, method="sum",
        secondary_weights_var="pop", secondary_weights_netcdf=pop_nc,
    )

    assert df.iloc[0]["tas"] == pytest.approx(0.5 * (0 + 4))


# --------------------------------------------------------------------------- #
# Package data access and realistic input
# --------------------------------------------------------------------------- #

def test_precalc_weights_path_resolves_shipped_data():
    path = precalc_weights_path("ISIMIP_to_DOSE_admin1_weights.csv")

    assert os.path.exists(str(path))
    assert list(pd.read_csv(path).columns) == [
        "lat", "lon", "region_id", "percent_overlap"
    ]


def test_precalc_weights_path_reports_unknown_names():
    with pytest.raises(ValueError, match="not found in package data"):
        precalc_weights_path("no_such_weights.csv")


def test_aggregation_over_a_real_shipped_slice(tmp_path, shipped_slice_csv):
    """Exercise the full path on genuine weights, not just synthetic ones."""
    weights = pd.read_csv(shipped_slice_csv)
    lats = np.sort(weights["lat"].unique())
    lons = np.sort(weights["lon"].unique())
    netcdf = make_grid_nc(tmp_path / "real.nc", lats=lats, lons=lons, n_time=2)

    df = _aggregate(tmp_path, netcdf, shipped_slice_csv)

    assert set(df["region_id"]) == set(weights["region_id"])
    assert len(df) == weights["region_id"].nunique() * 2
    assert df["tas"].notna().all()


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #

def _cli(*args):
    import subprocess
    import sys

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return subprocess.run(
        [sys.executable, "-m", "netcdf2region.netcdf2region", *args],
        capture_output=True, text=True, cwd=repo_root,
    )


def test_cli_aggregates_to_the_expected_values(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])
    out = tmp_path / "out.csv"

    result = _cli(
        "--netcdf", mini_tas_nc, "--weights", weights,
        "--method", "mean", "--output", str(out), "--var", "tas",
    )

    assert result.returncode == 0, result.stderr
    assert list(pd.read_csv(out)["tas"]) == [_value(t, 0, 0) for t in range(3)]


def test_cli_supports_correct_weights(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 0.5)])
    out = tmp_path / "out.csv"

    result = _cli(
        "--netcdf", mini_tas_nc, "--weights", weights,
        "--method", "sum", "--output", str(out), "--var", "tas", "--correct_weights",
    )

    assert result.returncode == 0, result.stderr
    assert pd.read_csv(out).iloc[0]["tas"] == pytest.approx(_value(0, 0, 0))


def test_cli_requires_a_weights_source(tmp_path, mini_tas_nc):
    result = _cli(
        "--netcdf", mini_tas_nc, "--method", "mean",
        "--output", str(tmp_path / "o.csv"), "--var", "tas",
    )

    assert result.returncode == 2
    assert "required" in result.stderr


def test_cli_rejects_both_weights_sources_at_once(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])

    result = _cli(
        "--netcdf", mini_tas_nc, "--weights", weights,
        "--precalc_weights", "ISIMIP_to_DOSE_admin1_weights.csv",
        "--method", "mean", "--output", str(tmp_path / "o.csv"), "--var", "tas",
    )

    assert result.returncode == 2
    assert "not allowed with" in result.stderr


def test_cli_reports_an_unknown_precalc_name(tmp_path, mini_tas_nc):
    result = _cli(
        "--netcdf", mini_tas_nc, "--precalc_weights", "nope.csv",
        "--method", "mean", "--output", str(tmp_path / "o.csv"), "--var", "tas",
    )

    assert result.returncode != 0
    assert "not found in package data" in result.stderr


def test_cli_requires_both_secondary_weight_options(tmp_path, mini_tas_nc, mini_pop_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])
    common = [
        "--netcdf", mini_tas_nc, "--weights", weights, "--method", "mean",
        "--output", str(tmp_path / "o.csv"), "--var", "tas",
    ]

    only_var = _cli(*common, "--secondary_weights_var", "pop")
    only_file = _cli(*common, "--secondary_weights_netcdf", mini_pop_nc)

    assert only_var.returncode != 0
    assert "requires --secondary_weights_netcdf" in only_var.stderr
    assert only_file.returncode != 0
    assert "requires --secondary_weights_var" in only_file.stderr


def test_cli_rejects_an_unsupported_method(tmp_path, mini_tas_nc):
    weights = _weights_file(tmp_path, [(MINI_LATS[0], MINI_LONS[0], "A", 1.0)])

    result = _cli(
        "--netcdf", mini_tas_nc, "--weights", weights,
        "--method", "median", "--output", str(tmp_path / "o.csv"), "--var", "tas",
    )

    assert result.returncode == 2
    assert "invalid choice" in result.stderr
