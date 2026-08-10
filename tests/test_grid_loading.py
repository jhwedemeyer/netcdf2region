"""Tests for reading the template grid and the optional land-sea mask."""

import os

import numpy as np
import pytest

from factories import MINI_LATS, MINI_LONS, make_grid_nc
from netcdf2region.compute_grid_weights import load_land_sea_mask, load_template_grid


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


# --------------------------------------------------------------------------- #
# load_land_sea_mask
# --------------------------------------------------------------------------- #

def _coords_for(lats, lons):
    return {
        "lat": np.asarray(lats),
        "lon": np.asarray(lons),
        "lat_res": 1.0,
        "lon_res": 1.0,
        "coordinate_origin": "centered",
    }


def test_mask_is_read_and_matched_against_the_grid(tmp_path):
    values = np.zeros((len(MINI_LATS), len(MINI_LONS)))
    values[0, 0] = 1
    path = make_grid_nc(tmp_path / "m.nc", var="lsm", values=values)

    mask = load_land_sea_mask(path, _coords_for(MINI_LATS, MINI_LONS), yes=True)

    assert mask.shape == (len(MINI_LATS), len(MINI_LONS))


def test_mask_is_dilated_so_coastal_cells_are_kept(tmp_path):
    """A ``maximum_filter`` of size 10 grows every land cell by ~5 cells.

    Without the dilation, a coarse mask clips regions that only touch the edge of
    a land cell.
    """
    lats = np.arange(0.5, 40.0, 1.0)
    lons = np.arange(0.5, 40.0, 1.0)
    values = np.zeros((len(lats), len(lons)))
    values[20, 20] = 1
    path = make_grid_nc(tmp_path / "m.nc", lats=lats, lons=lons, var="lsm", values=values)

    mask = load_land_sea_mask(path, _coords_for(lats, lons), yes=True)

    assert mask[20, 20] == 1
    assert mask[18, 18] == 1, "neighbouring cells must be pulled in"
    assert mask[35, 35] == 0, "distant cells must stay excluded"
    assert mask.sum() > values.sum()


def test_mask_with_wrong_shape_is_rejected(tmp_path):
    path = make_grid_nc(
        tmp_path / "m.nc", lats=np.arange(0.5, 3.0, 1.0), lons=np.arange(0.5, 3.0, 1.0),
        var="lsm",
    )

    with pytest.raises(ValueError, match="does not match grid shape"):
        load_land_sea_mask(path, _coords_for(MINI_LATS, MINI_LONS), yes=True)


def test_mask_on_a_different_grid_is_rejected(tmp_path):
    """A mask sharing the grid's shape but not its coordinates must be refused.

    Same latitudes, longitudes shifted by 40 degrees: the two grids describe
    completely different parts of the world, and silently accepting this would
    assign land cells to the wrong places.
    """
    shifted_lons = MINI_LONS + 40.0
    path = make_grid_nc(
        tmp_path / "m.nc", lats=MINI_LATS, lons=shifted_lons, var="lsm"
    )

    with pytest.raises(ValueError, match="must match grid lat/lon"):
        load_land_sea_mask(path, _coords_for(MINI_LATS, MINI_LONS), yes=True)


def test_missing_mask_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Land-sea mask file not found"):
        load_land_sea_mask(
            str(tmp_path / "absent.nc"), _coords_for(MINI_LATS, MINI_LONS), yes=True
        )
