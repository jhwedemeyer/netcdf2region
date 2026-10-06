"""Tests for reading the template grid."""

import os

import numpy as np
import pytest

from factories import MINI_LATS, MINI_LONS, make_grid_nc
from netcdf2region.load_grid import load_template_grid


def _load(path, origin="centered", shift=False):
    return load_template_grid(path, coordinate_origin=origin, shift_longitudes=shift)


# --------------------------------------------------------------------------- #
# load_template_grid - happy paths
# --------------------------------------------------------------------------- #

def test_reads_lat_lon_and_resolution(tmp_path):
    coords = _load(make_grid_nc(tmp_path / "g.nc"))

    assert np.array_equal(coords["lat"], MINI_LATS)
    assert np.array_equal(coords["lon"], MINI_LONS)
    assert coords["lat_res"] == 1.0
    assert coords["lon_res"] == 1.0
    assert coords["coordinate_origin"] == "centered"


def test_accepts_latitude_longitude_naming(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc", lat_name="latitude", lon_name="longitude"
    )
    coords = _load(path)

    assert np.array_equal(coords["lat"], MINI_LATS)
    assert np.array_equal(coords["lon"], MINI_LONS)


def test_takes_the_first_time_step_of_a_time_varying_file(tmp_path):
    """The grid is all that matters, so a time dimension must be collapsed."""
    coords = _load(make_grid_nc(tmp_path / "g.nc", n_time=5))

    assert np.array_equal(coords["lat"], MINI_LATS)


def test_resolution_of_a_finer_grid(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc",
        lats=np.arange(0.125, 4.0, 0.25),
        lons=np.arange(0.125, 4.0, 0.25),
    )
    coords = _load(path)

    assert np.isclose(coords["lat_res"], 0.25)
    assert np.isclose(coords["lon_res"], 0.25)


def test_descending_latitudes_are_accepted_with_positive_resolution(tmp_path):
    """Latitudes descend in many real products; only longitudes must ascend."""
    path = make_grid_nc(tmp_path / "g.nc", lats=MINI_LATS[::-1])
    coords = _load(path)

    assert coords["lat_res"] == 1.0
    assert coords["lat"][0] > coords["lat"][-1]


def test_top_left_origin_is_accepted(tmp_path):
    coords = _load(make_grid_nc(tmp_path / "g.nc"), origin="top-left")
    assert coords["coordinate_origin"] == "top-left"


# --------------------------------------------------------------------------- #
# load_template_grid - longitude shifting
# --------------------------------------------------------------------------- #

def test_zero_to_360_longitudes_are_rejected_without_the_flag(tmp_path):
    """The error must name the flag that fixes it - this is the ERA5 case."""
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))

    with pytest.raises(ValueError, match="shift_longitudes"):
        _load(path, shift=False)


def test_zero_to_360_longitudes_are_remapped_with_the_flag(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))
    coords = _load(path, shift=True)

    assert coords["lon"].min() >= -180.0
    assert coords["lon"].max() <= 180.0
    assert np.all(np.diff(coords["lon"]) > 0), "shifted longitudes must stay sorted"
    assert len(coords["lon"]) == 360
    assert np.isclose(coords["lon_res"], 1.0)


def test_longitudes_outside_both_conventions_are_rejected(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(400.0, 408.0, 1.0))

    with pytest.raises(ValueError, match="not in EPSG:4326 range"):
        _load(path, shift=True)


# --------------------------------------------------------------------------- #
# load_template_grid - error paths
# --------------------------------------------------------------------------- #

def test_missing_file_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError, match="NetCDF file not found"):
        _load(str(tmp_path / "absent.nc"))


def test_missing_latitude_raises_key_error(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lat_name="y")

    with pytest.raises(KeyError, match="lat"):
        _load(path)


def test_missing_longitude_raises_key_error(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lon_name="x")

    with pytest.raises(KeyError, match="lon"):
        _load(path)


def test_out_of_range_latitude_raises(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lats=np.arange(88.0, 94.0, 1.0))

    with pytest.raises(ValueError, match="Latitude values"):
        _load(path)


def test_unsorted_longitudes_are_rejected(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.array([3.5, 0.5, 1.5, 2.5]))

    with pytest.raises(ValueError, match="increasing order"):
        _load(path)


def test_invalid_coordinate_origin_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Invalid coordinate origin"):
        _load(make_grid_nc(tmp_path / "g.nc"), origin="bottom-right")


def test_rejects_a_grid_finer_than_the_stored_coordinate_precision(tmp_path):
    """Coordinates are rounded to 3 decimals, so finer cells would collide."""
    path = make_grid_nc(
        tmp_path / "g.nc",
        lats=np.arange(0.0, 0.004, 0.0005),
        lons=MINI_LONS,
    )

    with pytest.raises(ValueError, match="finer than the supported resolution"):
        _load(path)


def test_accepts_a_grid_at_the_stored_coordinate_precision(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc",
        lats=np.arange(0.0005, 0.008, 0.001),
        lons=np.arange(0.0005, 0.008, 0.001),
    )

    coords = _load(path)

    assert np.isclose(coords["lat_res"], 0.001)
