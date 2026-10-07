"""Aggregate gridded NetCDF data to regions using spatial weights.

The main entry points::

    from netcdf2region import aggregate, compute_weights, load_template_grid, validate_weights

    compute_weights("grid.nc", "regions.gpkg", "weights.csv")
    validate_weights("weights.csv", grid=load_template_grid("grid.nc")).raise_if_failed()
    aggregate("data.nc", "weights.csv", method="mean", output="out.csv", var="tas")
"""

from importlib.metadata import PackageNotFoundError, version

from .aggregate import aggregate
from .compute_grid_weights import compute_weights
from .load_grid import Grid, load_template_grid
from .load_regions import load_regions
from .validate_weights import validate_weights
from .weights import precalc_weights_path

try:
    __version__ = version("netcdf2region")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "unknown"

__all__ = [
    "Grid",
    "aggregate",
    "compute_weights",
    "load_regions",
    "load_template_grid",
    "precalc_weights_path",
    "validate_weights",
]
