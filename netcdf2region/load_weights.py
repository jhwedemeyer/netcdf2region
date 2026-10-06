"""Loading and checking the weights used by ``netcdf2region``: the weights CSV (or a
precalculated one from the package data), optional sea correction and area weighting,
checks against the data grid, and optional secondary weights from a NetCDF file.
"""

import logging
from importlib.resources import files

import numpy as np
import pandas as pd

from .load_grid import COORD_DECIMALS

logger = logging.getLogger(__name__)

def precalc_weights_path(name: str):
    """ Get the path to a precalculated weights file from the package data. """
    data = files("netcdf2region.data")
    path = data.joinpath(name)
    # joinpath() never raises for a missing name, so check explicitly rather than
    # letting the failure surface much later as a confusing read_csv error.
    if not path.is_file():
        available = sorted(p.name for p in data.iterdir() if p.name.endswith(".csv"))
        raise ValueError(
            f"Precalculated weights file '{name}' not found in package data. "
            f"Available: {available}"
        )
    return path


def load_weights(weights, method, correct_weights_flag=False, area_weighted=False):
    """Read a weights CSV (columns lat, lon, region_id, percent_overlap and cell_area_km2) and prepare it for aggregation.
    """
    weights_df = pd.read_csv(weights)
    # compute_grid_weights already writes 3 decimals, but a hand-edited or externally
    # produced file may not, and correct_weights below groups on these columns.
    weights_df[["lat", "lon"]] = weights_df[["lat", "lon"]].round(COORD_DECIMALS)
    if correct_weights_flag:
        weights_df = correct_weights(weights_df)

    # With area weighting, a mean or standard deviation of an intensive quantity
    # (temperature, precipitation rate, ...) weights cells by the area they contribute,
    # i.e. covered fraction x cell area. A sum of per-cell totals (population, ...) must
    # use the covered fraction alone, so area weighting does not apply to it.
    if area_weighted:
        if method not in ("mean", "std"):
            raise ValueError("Area weighting only applies to --method mean or std, not " + method + ".")
        if "cell_area_km2" not in weights_df.columns:
            raise ValueError(
                "Area weighting needs a 'cell_area_km2' column in the weights file; "
                "compute the weights with `compute_grid_weights --method exact`."
            )
        logger.info("Computing an area-weighted " + method + ".")

    return weights_df


def correct_weights(weights):
    """Adjusts grid cell weights so that no values are attributed to the sea.

    When a grid cell is split among multiple regions (e.g., due to partial overlaps), the sum of its weights
    across all regions may be less than 1 because some parts may fall into the sea. This function rescales 
    the weights so that, for each grid cell, the sum of its weights across all regions is exactly 1. 
    This ensures that the entire value of the grid cell is fully attributed to the regions it overlaps, 
    which is important for variables where the total value (e.g., population, GDP) should be preserved.

    The correction is performed by dividing each weight by the total overlap fraction for its grid cell.

    weight = old_weight / sum_of_weights_for_cell

    """

    land_fraction = weights.groupby(["lat", "lon"]).percent_overlap.sum().reset_index()

    land_fraction['correction_factor'] = 1.0 / land_fraction['percent_overlap']
    weights = weights.merge(land_fraction[['lat', 'lon', 'correction_factor']], on=['lat', 'lon'], how='left')
    weights['percent_overlap'] = weights['percent_overlap'] * weights['correction_factor']

    return weights.drop(columns=['correction_factor'])


def _describe_axis(values):
    """One-line description of a coordinate axis, for error messages."""
    if len(values) == 0:
        return "0 values"
    return f"{len(values)} values from {values[0]} to {values[-1]}"


def _check_axis_match(name, primary, secondary):
    """Raise if a secondary-weights axis is not the primary axis.

    The length is checked first: np.allclose on axes of different length raises
    a broadcast error from inside numpy rather than the message we want.
    """
    if len(primary) != len(secondary) or not np.allclose(primary, secondary):
        raise ValueError(
            f"{name} coordinates mismatch between primary data and secondary "
            f"weights file. Primary grid has {_describe_axis(primary)}, "
            f"secondary weights have {_describe_axis(secondary)}. The secondary "
            "weights file must be on the exact same grid as the input NetCDF; "
            "crop or regrid it to that grid first."
        )


def check_weights_on_grid(weights_df, lats, lons, netcdf):
    """Raise if the weights file references coordinates the grid does not have.

    Checked once up front rather than inside the region loop: using the wrong weights
    file for a grid is the most likely user error, and reporting it here says which
    axis is wrong instead of blaming whichever region happened to come first.

    Both sides are already rounded to COORD_DECIMALS, so a mismatch here is a genuinely
    different grid rather than a float representation artefact.
    """
    for name, column, axis in (("Latitude", "lat", lats), ("Longitude", "lon", lons)):
        missing = sorted(set(weights_df[column].tolist()) - set(axis.tolist()))
        if not missing:
            continue

        # The [-180, 180] vs [0, 360] mix-up is the usual cause of an unmatched
        # longitude, so name it rather than making the user guess.
        hint = ""
        if column == "lon":
            hint = (
                "Weights are always written in the -180 to 180 convention; if your data "
                "uses 0 to 360, re-run with --shift_longitudes. Otherwise the "
            )
        else:
            hint = "The "

        raise ValueError(
            f"{name} values in weights file do not match the grid in '{netcdf}'. "
            f"{len(missing)} unmatched value(s), e.g. {missing[:5]}. The NetCDF file has "
            f"{name.lower()}s {_describe_axis(axis)}. "
            f"{hint}weights file must have been computed for this exact grid. "
            f"Check it with: validate_weights <weights.csv> --netcdf <file.nc>"
        )


def validate_secondary_weights(
        lats,
        lons,
        secondary_data,
        secondary_lats,
        secondary_lons,
        secondary_weights_var,
    ):
    """Validate secondary weights as 2D spatial data on the primary grid.
    
    Accepts 2D (lat, lon) or 3D (time, lat, lon) with single timestep.
    3D data with time=1 will be squeezed to 2D by the caller.
    """
    _check_axis_match("Latitude", lats, secondary_lats)
    _check_axis_match("Longitude", lons, secondary_lons)

    if secondary_data.ndim == 3:
        if secondary_data.shape[0] != 1:
            raise ValueError(
                f"Secondary weights variable '{secondary_weights_var}' has time dimension size {secondary_data.shape[0]}. "
                "If provided as 3D data, time dimension must be 1."
            )
    elif secondary_data.ndim != 2:
        raise ValueError(
            f"Secondary weights variable '{secondary_weights_var}' must have dimensions "
            "(lat, lon) or (time=1, lat, lon)."
        )


def prepare_secondary_weights(
        lats,
        lons,
        secondary_data,
        secondary_lats,
        secondary_lons,
        secondary_weights_var,
    ):
    """Validate secondary weights loaded from NetCDF and return them as a 2D (lat, lon) array."""
    if np.isnan(secondary_data).any():
        logger.warning(
            f"Secondary weight variable '{secondary_weights_var}' contains NaN values. "
            "Regions covering a NaN secondary weight will aggregate to NaN."
        )

    validate_secondary_weights(
        lats=lats,
        lons=lons,
        secondary_data=secondary_data,
        secondary_lats=secondary_lats,
        secondary_lons=secondary_lons,
        secondary_weights_var=secondary_weights_var,
    )

    # Squeeze 3D data with single timestep to 2D
    # validate_secondary_weights() already checked that the time dimension is 1 if present, so this is safe.
    if secondary_data.ndim == 3:
        return secondary_data[0, :, :]
    return secondary_data


def region_cell_weights(region_weights, lat_indices, lon_indices, area_weighted=False,
                        secondary_weights_data=None):
    """Weight of each grid cell of one region: covered fraction, times cell area with
    area weighting, times the secondary weight if given."""
    cell_weights = region_weights['percent_overlap'].to_numpy()

    # Force float: an all-integer weights column would otherwise truncate the
    #  multiplication below.
    cell_weights = np.array(cell_weights, dtype=float)

    if area_weighted:
        cell_weights = cell_weights * region_weights['cell_area_km2'].to_numpy()

    if secondary_weights_data is not None:
        cell_weights = cell_weights * secondary_weights_data[lat_indices, lon_indices]

    return cell_weights
