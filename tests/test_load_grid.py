"""Tests for ``load_grid``: reading the template and data grids, and the ``Grid`` cells."""

import numpy as np
import pandas as pd
import pytest

from factories import MINI_LATS, MINI_LONS, make_grid_nc, mini_grid
from netcdf2region.load_grid import Grid, load_data_grid, load_template_grid


def _value(t, i, j):
    """The value the mini temperature field holds at time t, cell (i, j)."""
    return 100 * t + 10 * i + j


def _load(path, origin="centered", shift=False):
    return load_template_grid(path, coordinate_origin=origin, shift_longitudes=shift)


# --------------------------------------------------------------------------- #
# load_template_grid - happy paths
# --------------------------------------------------------------------------- #

def test_reads_lat_lon_and_resolution(tmp_path):
    coords = _load(make_grid_nc(tmp_path / "g.nc"))

    assert np.array_equal(coords.lat, MINI_LATS)
    assert np.array_equal(coords.lon, MINI_LONS)
    assert coords.lat_res == 1.0
    assert coords.lon_res == 1.0
    assert coords.coordinate_origin == "centered"


def test_accepts_latitude_longitude_naming(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc", lat_name="latitude", lon_name="longitude"
    )
    coords = _load(path)

    assert np.array_equal(coords.lat, MINI_LATS)
    assert np.array_equal(coords.lon, MINI_LONS)


def test_takes_the_first_time_step_of_a_time_varying_file(tmp_path):
    """The grid is all that matters, so a time dimension must be collapsed."""
    coords = _load(make_grid_nc(tmp_path / "g.nc", n_time=5))

    assert np.array_equal(coords.lat, MINI_LATS)


def test_resolution_of_a_finer_grid(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc",
        lats=np.arange(0.125, 4.0, 0.25),
        lons=np.arange(0.125, 4.0, 0.25),
    )
    coords = _load(path)

    assert np.isclose(coords.lat_res, 0.25)
    assert np.isclose(coords.lon_res, 0.25)


def test_descending_latitudes_are_accepted_with_positive_resolution(tmp_path):
    """Latitudes descend in many real products; only longitudes must ascend."""
    path = make_grid_nc(tmp_path / "g.nc", lats=MINI_LATS[::-1])
    coords = _load(path)

    assert coords.lat_res == 1.0
    assert coords.lat[0] > coords.lat[-1]


def test_top_left_origin_is_accepted(tmp_path):
    coords = _load(make_grid_nc(tmp_path / "g.nc"), origin="top-left")
    assert coords.coordinate_origin == "top-left"


# --------------------------------------------------------------------------- #
# load_template_grid - longitude shifting
# --------------------------------------------------------------------------- #

def test_zero_to_360_longitudes_are_rejected_without_the_flag(tmp_path):
    """The error must name the flag that fixes it - this is the ERA5 case."""
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))

    with pytest.raises(ValueError, match="shift-longitudes"):
        _load(path, shift=False)


def test_zero_to_360_longitudes_are_remapped_with_the_flag(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0))
    coords = _load(path, shift=True)

    assert coords.lon.min() >= -180.0
    assert coords.lon.max() <= 180.0
    assert np.all(np.diff(coords.lon) > 0), "shifted longitudes must stay sorted"
    assert len(coords.lon) == 360
    assert np.isclose(coords.lon_res, 1.0)


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

    assert np.isclose(coords.lat_res, 0.001)


# --------------------------------------------------------------------------- #
# load_data_grid
# --------------------------------------------------------------------------- #

def test_load_data_grid_returns_values_and_coordinates(mini_tas_nc):
    values, lats, lons, times = load_data_grid(mini_tas_nc, "tas")

    assert values.shape == (3, len(MINI_LATS), len(MINI_LONS))
    assert np.array_equal(lats, MINI_LATS)
    assert np.array_equal(lons, MINI_LONS)
    assert len(times) == 3
    assert values[2, 3, 4] == _value(2, 3, 4)


def test_load_data_grid_accepts_latitude_longitude_naming(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc", n_time=2, lat_name="latitude", lon_name="longitude"
    )
    values, lats, lons, _ = load_data_grid(path, "tas")

    assert values.shape == (2, len(MINI_LATS), len(MINI_LONS))
    assert np.array_equal(lats, MINI_LATS)


def test_load_data_grid_transposes_to_time_lat_lon(tmp_path):
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

    loaded, _, _, _ = load_data_grid(path, "tas")

    assert loaded.shape == (2, len(MINI_LATS), len(MINI_LONS))


def test_load_data_grid_allows_a_spatial_only_variable(mini_pop_nc):
    # mini_pop holds NaN cells, so loading it at all needs the opt-out.
    values, _, _, times = load_data_grid(
        mini_pop_nc, "pop", require_time=False, allow_nan=True
    )

    assert values.shape == (len(MINI_LATS), len(MINI_LONS))
    assert times is None


def test_load_data_grid_rejects_a_spatial_only_variable_when_time_is_required(mini_pop_nc):
    with pytest.raises(ValueError, match="has no time dimension"):
        load_data_grid(mini_pop_nc, "pop", require_time=True, allow_nan=True)


def test_load_data_grid_rejects_nan_values_by_default(mini_pop_nc):
    """NaNs silently corrupt every region that touches them, so refuse by default."""
    with pytest.raises(ValueError, match="contains NaN values"):
        load_data_grid(mini_pop_nc, "pop", require_time=False)


def test_load_data_grid_keeps_nan_values_when_allowed(mini_pop_nc):
    values, _, _, _ = load_data_grid(
        mini_pop_nc, "pop", require_time=False, allow_nan=True
    )

    # The opt-out is not a repair: the NaNs are handed back untouched.
    assert np.isnan(values).sum() == 2


def test_load_data_grid_rejects_an_unknown_variable(mini_tas_nc):
    with pytest.raises(KeyError, match="not found"):
        load_data_grid(mini_tas_nc, "does_not_exist")


def test_load_data_grid_rejects_an_unknown_time_variable(mini_tas_nc):
    with pytest.raises(KeyError, match="not found"):
        load_data_grid(mini_tas_nc, "tas", time_var="epoch")


def test_load_data_grid_cannot_guess_between_two_non_spatial_dims(tmp_path):
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

    with pytest.raises(ValueError, match="time-var"):
        load_data_grid(path, "tas")


# --------------------------------------------------------------------------- #
# load_data_grid - longitude shifting
# --------------------------------------------------------------------------- #

def test_load_data_grid_rejects_zero_to_360_longitudes_without_the_flag(tmp_path):
    """Weights are written in [-180, 180]; the error must name the flag."""
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0), n_time=1)

    with pytest.raises(ValueError, match="shift-longitudes"):
        load_data_grid(path, "tas")


def test_load_data_grid_shifts_zero_to_360_longitudes_with_the_flag(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(0.5, 360.0, 1.0), n_time=1)

    _, _, lons, _ = load_data_grid(path, "tas", shift_longitudes=True)

    assert lons.min() == -179.5 and lons.max() == 179.5
    assert np.all(np.diff(lons) > 0), "shifted longitudes must stay sorted"


def test_load_data_grid_reorders_the_data_with_the_longitudes(tmp_path):
    """Values must follow their own longitude, not stay at their old column."""
    lons = np.arange(0.5, 360.0, 1.0)
    path = make_grid_nc(
        tmp_path / "g.nc", lons=lons, lats=[0.5],
        values=lons.reshape(1, 1, len(lons)), n_time=1,
    )

    values, _, shifted, _ = load_data_grid(path, "tas", shift_longitudes=True)

    # Each cell held its own original longitude, so after the shift every value
    # must equal its new coordinate mapped back into [0, 360].
    assert np.array_equal(values[0, 0, :], shifted % 360)


def test_load_data_grid_rejects_longitudes_outside_both_conventions(tmp_path):
    path = make_grid_nc(tmp_path / "g.nc", lons=np.arange(400.0, 408.0, 1.0), n_time=1)

    with pytest.raises(ValueError, match="not in EPSG:4326 range"):
        load_data_grid(path, "tas", shift_longitudes=True)


# --------------------------------------------------------------------------- #
# load_data_grid - coordinate precision
# --------------------------------------------------------------------------- #

def test_load_data_grid_rejects_a_grid_finer_than_the_stored_coordinate_precision(tmp_path):
    path = make_grid_nc(
        tmp_path / "g.nc", lons=np.arange(0.0, 0.004, 0.0005), n_time=1
    )

    with pytest.raises(ValueError, match="finer than the supported resolution"):
        load_data_grid(path, "tas")


# --------------------------------------------------------------------------- #
# Grid
# --------------------------------------------------------------------------- #

def test_cell_bounds_centered_surrounds_the_coordinate():
    coords = mini_grid("centered")
    lo_x, lo_y, hi_x, hi_y = coords.cell_bounds(0, 0)

    assert (lo_x, lo_y, hi_x, hi_y) == (0.0, 0.0, 1.0, 1.0)
    assert lo_x < MINI_LONS[0] < hi_x
    assert lo_y < MINI_LATS[0] < hi_y


def test_cell_bounds_top_left_spans_adjacent_coordinates():
    coords = mini_grid("top-left")
    assert coords.cell_bounds(0, 0) == (0.5, 0.5, 1.5, 1.5)
    assert coords.cell_bounds(1, 2) == (2.5, 1.5, 3.5, 2.5)


def test_cell_bounds_normalises_descending_latitudes():
    """Latitudes often descend in real NetCDF files; bounds must still be ordered."""
    coords = Grid(np.array([5.5, 4.5, 3.5, 2.5, 1.5, 0.5]), MINI_LONS.copy(), 1.0, 1.0)
    for i in range(len(coords.lat)):
        lo_x, lo_y, hi_x, hi_y = coords.cell_bounds(i, 0)
        assert lo_x < hi_x
        assert lo_y < hi_y


def test_cell_bounds_tile_the_grid_without_gaps():
    coords = mini_grid("centered")
    left = coords.cell_bounds(0, 0)
    right = coords.cell_bounds(0, 1)
    assert left[2] == right[0]


def test_shape_counts_cells_not_coordinates():
    assert mini_grid("centered").shape == (len(MINI_LATS), len(MINI_LONS))
    assert mini_grid("top-left").shape == (len(MINI_LATS) - 1, len(MINI_LONS) - 1)


@pytest.mark.parametrize("origin", ["centered", "top-left"])
@pytest.mark.parametrize("lats", [MINI_LATS, MINI_LATS[::-1]])
def test_edges_agree_with_cell_bounds(origin, lats):
    """edges() is the vectorised form of cell_bounds, so they must never disagree."""
    grid = Grid(lats.copy(), MINI_LONS.copy(), 1.0, 1.0, origin)
    lat_lo, lat_hi, lon_lo, lon_hi = grid.edges()
    n_lat, n_lon = grid.shape

    for i in range(n_lat):
        for j in range(n_lon):
            assert grid.cell_bounds(i, j) == (lon_lo[j], lat_lo[i], lon_hi[j], lat_hi[i])


def test_edges_are_clamped_to_the_poles():
    grid = Grid(np.array([-89.75, 89.75]), np.array([0.25]), 0.6, 0.5)
    lat_lo, lat_hi, _, _ = grid.edges()

    assert lat_lo.min() == -90 and lat_hi.max() == 90


def test_grid_rejects_an_unknown_coordinate_origin():
    with pytest.raises(ValueError, match="Invalid coordinate origin"):
        Grid(MINI_LATS, MINI_LONS, 1.0, 1.0, "bottom-right")
