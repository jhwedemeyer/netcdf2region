"""Builders for the synthetic NetCDF grids and region sets the tests run on.

The repository ships no shapefile and no small NetCDF, and the cluster inputs the
precalculation scripts point at are not available locally. Everything the tests
need is therefore constructed here, which has the side benefit of letting us cover
grid variants the shipped data never exercises: ``latitude``/``longitude`` naming,
``[0, 360]`` longitudes, descending latitudes and the ``top-left`` origin.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import xarray as xr
from shapely.geometry import box

#: The grid every "mini" fixture is built on: 1 degree cells, centred coordinates,
#: spanning lon [0, 8] and lat [0, 6].
MINI_LATS = np.arange(0.5, 6.0, 1.0)
MINI_LONS = np.arange(0.5, 8.0, 1.0)

#: Points per cell used throughout. 100 gives an exact 10x10 lattice at offsets
#: 0.05, 0.15, ... 0.95 within a 1 degree cell, which is what makes the expected
#: weights below computable by hand.
MINI_POINTS_PER_CELL = 100


def make_grid_nc(
    path,
    lats=None,
    lons=None,
    var="tas",
    n_time=0,
    lat_name="lat",
    lon_name="lon",
    values=None,
    start="2001-01-01",
):
    """Write a NetCDF file describing a lat/lon grid.

    Args:
        path: Destination path.
        lats: Latitude coordinates (defaults to the mini grid).
        lons: Longitude coordinates (defaults to the mini grid).
        var: Name of the data variable.
        n_time: Number of time steps; 0 writes a purely spatial variable.
        lat_name: ``"lat"`` or ``"latitude"``.
        lon_name: ``"lon"`` or ``"longitude"``.
        values: Explicit data array; generated deterministically when omitted.
        start: First time stamp when ``n_time`` is non-zero.

    Returns:
        str: ``path``, for convenient chaining.
    """
    lats = MINI_LATS if lats is None else np.asarray(lats, dtype=float)
    lons = MINI_LONS if lons is None else np.asarray(lons, dtype=float)

    if n_time:
        shape = (n_time, len(lats), len(lons))
        dims = ("time", lat_name, lon_name)
        coords = {
            "time": pd.date_range(start, periods=n_time, freq="YS"),
            lat_name: lats,
            lon_name: lons,
        }
    else:
        shape = (len(lats), len(lons))
        dims = (lat_name, lon_name)
        coords = {lat_name: lats, lon_name: lons}

    if values is None:
        # Deterministic, distinct per cell and per time step, so a weighted mean
        # over any subset of cells has an unmistakable expected value.
        values = np.arange(np.prod(shape), dtype=float).reshape(shape)
    else:
        values = np.asarray(values, dtype=float).reshape(shape)

    ds = xr.Dataset({var: (dims, values)}, coords=coords)
    ds.to_netcdf(path)
    ds.close()
    return str(path)


def make_regions(polygons, gid_col="GID_1", extra_columns=None):
    """Build a regions GeoDataFrame from ``{region_id: geometry}``.

    Args:
        polygons: Mapping of region id to shapely geometry.
        gid_col: Name of the identifier column, matching ``--gid``.
        extra_columns: Optional mapping of extra column name to list of values.

    Returns:
        geopandas.GeoDataFrame: In EPSG:4326, with ``gid_col`` and ``geometry``.
    """
    data = {gid_col: list(polygons.keys())}
    if extra_columns:
        data.update(extra_columns)
    return gpd.GeoDataFrame(
        data, geometry=list(polygons.values()), crs="EPSG:4326"
    )


def write_regions(gdf, path, layer=None):
    """Write a regions GeoDataFrame to disk, as GeoPackage or GeoJSON.

    Args:
        gdf: The GeoDataFrame to write.
        path: Destination path; the suffix selects the driver.
        layer: Optional layer name (GeoPackage only).

    Returns:
        str: ``path``.
    """
    path = str(path)
    if path.endswith(".gpkg"):
        gdf.to_file(path, driver="GPKG", layer=layer or "regions")
    else:
        gdf.to_file(path, driver="GeoJSON")
    return path


def mini_polygons():
    """The mini region set, chosen so every expected weight is exact.

    On the 1 degree mini grid with a 10x10 point lattice per cell (sample offsets
    0.05, 0.15, ... 0.95):

    * ``A`` covers cell (0, 0) exactly           -> sampled weight 1.00
    * ``B`` covers the western half of cell (0, 2) -> sampled weight 0.50
    * ``C`` straddles the corner shared by cells (2, 4), (2, 5), (3, 4), (3, 5),
      taking a quarter of each                    -> sampled weight 0.25 x 4
    * ``D`` is a 0.02 degree sliver placed between sample points in cell (4, 7),
      so sampling never sees it and ``assign_missing_regions`` must rescue it
      with its exact area fraction of 0.0004
    * ``E`` is a 0.0005 degree speck in cell (1, 6) whose area fraction rounds to
      zero at 6 decimals, exercising the ``1e-6`` floor of the rescue path

    Returns:
        dict: Mapping of region id to shapely polygon.
    """
    return {
        "A": box(0.0, 0.0, 1.0, 1.0),
        "B": box(2.0, 0.0, 2.5, 1.0),
        "C": box(4.5, 2.5, 5.5, 3.5),
        "D": box(7.20, 4.20, 7.22, 4.22),
        "E": box(6.30, 1.30, 6.3005, 1.3005),
    }


#: Sampled weights the mini case must produce, keyed by ``(region_id, lat, lon)``.
#: Derived by hand from the geometry above; see :func:`mini_polygons`.
MINI_EXPECTED_SAMPLED = {
    ("A", 0.5, 0.5): 1.00,
    ("B", 0.5, 2.5): 0.50,
    ("C", 2.5, 4.5): 0.25,
    ("C", 2.5, 5.5): 0.25,
    ("C", 3.5, 4.5): 0.25,
    ("C", 3.5, 5.5): 0.25,
}

#: Regions that point sampling must miss entirely, forcing the rescue path.
MINI_EXPECTED_RESCUED = {
    ("D", 4.5, 7.5): 0.0004,
    ("E", 1.5, 6.5): 1e-6,
}


def mini_coordinates(coordinate_origin="centered"):
    """The mini grid as a ``load_template_grid``-style coordinates dict."""
    return {
        "lat": MINI_LATS.copy(),
        "lon": MINI_LONS.copy(),
        "lat_res": 1.0,
        "lon_res": 1.0,
        "coordinate_origin": coordinate_origin,
    }


def make_weights_df(rows):
    """Build a weights DataFrame from ``(lat, lon, region_id, percent_overlap)``."""
    return pd.DataFrame(rows, columns=["lat", "lon", "region_id", "percent_overlap"])
