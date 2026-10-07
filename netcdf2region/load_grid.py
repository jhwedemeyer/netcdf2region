"""Loading NetCDF grids: the template grid the weights are computed on, and the data
grid :mod:`aggregate` aggregates, plus describing the grid's cells.

Used by both :mod:`sampled_weights` and :mod:`exact_weights`, by ``validate_weights``
and by :mod:`aggregate`.
"""

import logging
import os
from dataclasses import dataclass

import numpy as np
import xarray as xr

#: Weights CSVs store coordinates at 3 decimals, so grid axes are rounded to the same
#: precision before matching.
COORD_DECIMALS = 3

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Grid:
    """A regular lat/lon grid, as the weights are computed on.

    Attributes:
        lat, lon: Coordinate axes in degrees (EPSG:4326). Longitudes ascend; latitudes
            may run either way.
        lat_res, lon_res: Spacing between neighbouring coordinates.
        coordinate_origin: ``"centered"`` if the coordinates are cell centres,
            ``"top-left"`` if they are cell edges (so one more edge than cells per axis).
    """

    lat: np.ndarray
    lon: np.ndarray
    lat_res: float
    lon_res: float
    coordinate_origin: str = "centered"

    def __post_init__(self):
        if self.coordinate_origin not in ("centered", "top-left"):
            raise ValueError("Invalid coordinate origin. Must be 'centered' or 'top-left'.")

    @property
    def shape(self):
        """Number of cells along (lat, lon)."""
        if self.coordinate_origin == "centered":
            return len(self.lat), len(self.lon)
        return len(self.lat) - 1, len(self.lon) - 1

    def cell_bounds(self, i, j):
        """Return the (lon1, lat1, lon2, lat2) edges of grid cell (i, j)."""
        lats, lons = self.lat, self.lon
        if self.coordinate_origin == "centered":
            lat1, lat2 = lats[i] - self.lat_res / 2, lats[i] + self.lat_res / 2
            lon1, lon2 = lons[j] - self.lon_res / 2, lons[j] + self.lon_res / 2
        else:  # top-left: coordinates are cell edges
            lat1, lat2 = lats[i], lats[i + 1]
            lon1, lon2 = lons[j], lons[j + 1]
        return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)

    def edges(self):
        """Return the lower and upper cell edges along each axis.

        Returns:
            tuple: ``(lat_lo, lat_hi, lon_lo, lon_hi)``, indexed like the grid's own
            ``lat`` and ``lon`` axes (so ascending or descending order is preserved).
        """
        # The grid is regular in lon/lat, so the latitude edges depend only on the row
        # and the longitude edges only on the column: the same values cell_bounds gives,
        # computed once per axis.
        if self.coordinate_origin == "centered":
            lat_a, lat_b = self.lat - self.lat_res / 2, self.lat + self.lat_res / 2
            lon_a, lon_b = self.lon - self.lon_res / 2, self.lon + self.lon_res / 2
        else:  # top-left: the coordinates are cell edges
            lat_a, lat_b = self.lat[:-1], self.lat[1:]
            lon_a, lon_b = self.lon[:-1], self.lon[1:]
        # Clamp to the globe: a cell centred on +-89.75 with 0.5 deg resolution ends at
        # +-90, but rounding can push the edge marginally beyond it.
        return (
            np.clip(np.minimum(lat_a, lat_b), -90, 90),
            np.clip(np.maximum(lat_a, lat_b), -90, 90),
            np.minimum(lon_a, lon_b),
            np.maximum(lon_a, lon_b),
        )


def check_coordinate_resolution(lats, lons):
    """Raise if a grid axis is finer than ``COORD_DECIMALS`` can represent.

    Coordinates are rounded to ``COORD_DECIMALS`` decimals for matching, so on a finer
    grid neighbouring cells would collapse onto the same coordinate and their weights
    would be silently mixed up.
    """
    min_spacing = 10.0 ** -COORD_DECIMALS
    for name, axis in (("Latitude", lats), ("Longitude", lons)):
        if len(axis) < 2:
            continue
        spacing = np.abs(np.diff(axis)).min()
        # Tolerance so a 0.001 grid stored with float error is still accepted.
        if spacing < min_spacing * (1 - 1e-6):
            raise ValueError(
                f"{name} spacing of {spacing:g} degrees is finer than the supported "
                f"resolution of {min_spacing:g} degrees: coordinates are stored at "
                f"{COORD_DECIMALS} decimals, so neighbouring cells would become "
                "indistinguishable. Regrid the data to a coarser grid first."
            )


def _open_dataset(netcdf_path):
    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")
    return xr.open_dataset(netcdf_path)


def _read_axes(ds, shift_longitudes):
    """Find the lat/lon axes of ``ds`` and bring them into EPSG:4326.

    Weights are always written in [-180, 180], so a [0, 360] grid is shifted when
    ``shift_longitudes`` is set and rejected otherwise.

    Returns:
        tuple: ``(lat_name, lon_name, lats, lons, lon_order)``, where ``lon_order``
        reorders data along the longitude axis to match the shifted, sorted ``lons``
        (``None`` when no shift was needed).
    """
    lat_name = "lat" if "lat" in ds else "latitude"
    if lat_name not in ds:
        raise KeyError(f"NetCDF file must contain 'lat' or 'latitude' variable. Available variables: {list(ds.variables)}")

    lon_name = "lon" if "lon" in ds else "longitude"
    if lon_name not in ds:
        raise KeyError(f"NetCDF file must contain 'lon' or 'longitude' variable. Available variables: {list(ds.variables)}")

    lats = ds[lat_name].values
    lons = ds[lon_name].values

    if not np.all((lats >= -90) & (lats <= 90)):
        raise ValueError("Latitude values are not in EPSG:4326 range (-90 to 90).")

    lon_order = None
    if not np.all((lons >= -180) & (lons <= 180)):
        if not shift_longitudes:
            raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360). You might want to enable longitude shifting with the --shift-longitudes flag if your data uses [0, 360] range.")
        if not np.all((lons >= 0) & (lons <= 360)):
            raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360).")
        logger.info("Shifting longitudes from [0, 360] to [-180, 180]. Double check if this works for your data!")
        lons = ((lons + 180) % 360) - 180
        lon_order = np.argsort(lons)
        lons = lons[lon_order]

    check_coordinate_resolution(lats, lons)

    return lat_name, lon_name, lats, lons, lon_order


def load_template_grid(netcdf_path, coordinate_origin="centered", shift_longitudes=False):
    """Load the grid of a NetCDF file as a :class:`Grid`."""
    logger.info(f"Loading NetCDF: {netcdf_path}")

    with _open_dataset(netcdf_path) as ds:
        _, _, lats, lons, _ = _read_axes(ds, shift_longitudes)

    lat_res = np.abs(lats[1] - lats[0])
    lon_res = np.abs(lons[1] - lons[0])
    logger.info(f"Grid resolution: {lat_res:.2f}° latitude × {lon_res:.2f}° longitude")

    # Cell edges are derived from neighbouring longitudes, so they must ascend.
    if not np.all(np.diff(lons) > 0):
        raise ValueError("Longitudes must be in increasing order")

    return Grid(lats, lons, lat_res, lon_res, coordinate_origin)


def load_data_grid(
        netcdf_path,
        var,
        time_var=None,
        require_time=True,
        allow_nan=False,
        shift_longitudes=False,
    ):
    """ Load NetCDF file and extract variable, lat, lon, and optional time. """
    with _open_dataset(netcdf_path) as ds:
        if var not in ds:
            raise KeyError(f"Variable '{var}' not found in NetCDF file. Available variables: {list(ds.variables)}")

        lat_name, lon_name, lat_values, lon_values, lon_order = _read_axes(ds, shift_longitudes)

        data_var = ds[var]
        if lat_name not in data_var.dims or lon_name not in data_var.dims:
            raise ValueError(
                f"Variable '{var}' must include both latitude and longitude dimensions. "
                f"Found dimensions: {data_var.dims}"
            )

        if time_var is not None:
            if time_var not in ds.variables:
                raise KeyError(f"Variable '{time_var}' not found in NetCDF file. Available variables: {list(ds.variables)}")
            if time_var not in data_var.dims:
                raise ValueError(
                    f"Provided time variable '{time_var}' is not a dimension of variable '{var}'. "
                    f"Found dimensions: {data_var.dims}"
                )
            time_name = time_var
        else:
            candidate_time_dims = [d for d in data_var.dims if d not in {lat_name, lon_name}]
            if len(candidate_time_dims) == 1:
                time_name = candidate_time_dims[0]
                if require_time:
                    logger.info(f"Using time variable: {time_name}")
            elif len(candidate_time_dims) == 0:
                time_name = None
            else:
                raise ValueError(
                    f"Could not infer a unique time dimension for variable '{var}'. "
                    f"Found non-spatial dimensions: {candidate_time_dims}. Please provide --time-var."
                )

        if require_time and time_name is None:
            raise ValueError(
                f"Variable '{var}' has no time dimension. Please provide a variable with dimensions "
                f"(time, {lat_name}, {lon_name})."
            )

        if time_name is None:
            data_var = data_var.transpose(lat_name, lon_name)
            time_values = None
        else:
            data_var = data_var.transpose(time_name, lat_name, lon_name)
            time_values = ds[time_name].values if time_name in ds.variables else None

        if lon_order is not None:
            data_var = data_var.isel({lon_name: lon_order})

        # Read the data before the dataset is closed.
        values = data_var.values

    # Put the axes in the same precision the weights files use. This has to come after the
    # shift, because the modulo arithmetic there introduces error of its own.
    lat_values = np.round(lat_values, COORD_DECIMALS)
    lon_values = np.round(lon_values, COORD_DECIMALS)

    if np.isnan(values).any() and not allow_nan:
        raise ValueError(f"Variable '{var}' in file '{netcdf_path}' contains NaN values. This may affect the aggregation results. Add --allow-nan to proceed anyway.")

    return values, lat_values, lon_values, time_values

