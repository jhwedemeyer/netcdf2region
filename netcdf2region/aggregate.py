"""Aggregating gridded NetCDF data to regions: :func:`aggregate` and the
``netcdf2region`` command.
"""

import argparse
import logging

import pandas as pd
import numpy as np
from tqdm import tqdm

from netcdf2region.cli import add_option, add_shift_longitudes, setup_logging
from netcdf2region.load_grid import load_data_grid
from netcdf2region.weights import (
    check_weights_on_grid,
    load_weights,
    precalc_weights_path,
    prepare_secondary_weights,
    region_cell_weights,
)

logger = logging.getLogger(__name__)


def aggregate(
    netcdf,
    weights,
    method,
    output,
    var,
    correct_weights_flag=False,
    time_var_name=None,
    secondary_weights_var=None,
    secondary_weights_netcdf=None,
    allow_nan=False,
    shift_longitudes=False,
    exclude_nan_cells=False,
    fill_nans=None,
    area_weighted=False,
):
    """ Aggregate gridded NetCDF data to regions using weights. """

    logger.info(f"Loading weights file: {weights}")
    weights_df = load_weights(weights, method, correct_weights_flag, area_weighted)

    # Load NetCDF data. Choosing a NaN handling mode implies NaNs are expected in
    # the variable (but not in the secondary weights, which these modes don't touch).
    logger.info(f"Loading NetCDF file: {netcdf}")
    var_data, lats, lons, time_var = load_data_grid(
        netcdf,
        var,
        time_var=time_var_name,
        require_time=True,
        allow_nan=allow_nan or exclude_nan_cells or fill_nans is not None,
        shift_longitudes=shift_longitudes,
    )

    # Print NaN handling summary
    if np.isnan(var_data).any():
        steps = []
        if exclude_nan_cells:
            steps.append("cells that are NaN at every time step are excluded")
        if fill_nans is not None:
            remaining = "remaining " if exclude_nan_cells else ""
            steps.append(f"{remaining}NaN values are replaced with {fill_nans}")
        else:
            steps.append("affected regions aggregate to NaN")
        handling = "; then ".join(steps)
        logger.warning(f"Variable '{var}' contains NaN values. {handling[0].upper()}{handling[1:]}.")

    check_weights_on_grid(weights_df, lats, lons, netcdf)

    # If secondary weights are provided, load and validate them
    secondary_weights_data = None
    if secondary_weights_var is not None:
        secondary_data, secondary_lats, secondary_lons, _ = load_data_grid(
            secondary_weights_netcdf,
            secondary_weights_var,
            time_var=None,
            require_time=False,
            allow_nan=allow_nan,
            shift_longitudes=shift_longitudes,
        )
        secondary_weights_data = prepare_secondary_weights(
            lats=lats,
            lons=lons,
            secondary_data=secondary_data,
            secondary_lats=secondary_lats,
            secondary_lons=secondary_lons,
            secondary_weights_var=secondary_weights_var,
        )

    # initialize list to store results for each region
    results = []

    # Loop over each unique region
    for region_id in tqdm(weights_df['region_id'].unique(), desc="Aggregating regions"):

        # Select weights for the current region
        region_weights = weights_df[weights_df['region_id'] == region_id]

        # Resolve to grid indices. check_weights_on_grid has already established that
        # every coordinate in the file is on the grid, so these lookups cannot miss.
        lat_indices = np.array([np.where(lats == lat)[0][0] for lat in region_weights['lat']])
        lon_indices = np.array([np.where(lons == lon)[0][0] for lon in region_weights['lon']])

        weights_arr = region_cell_weights(
            region_weights, lat_indices, lon_indices, area_weighted, secondary_weights_data
        )

        # Extract the relevant grid cells for all time steps at once
        # Shape: (num_cells, num_times)
        region_values = var_data[:, lat_indices, lon_indices].T  # shape: (num_cells, num_times)

        # Cells that are NaN at every time step are structural (e.g. ocean cells):
        # drop them by zeroing their weight. Their values must be zeroed too, since
        # NaN * 0 is still NaN. This runs before filling, so the fill only reaches
        # the cells that are kept.
        if exclude_nan_cells:
            cells_all_nan = np.isnan(region_values).all(axis=1)
            region_values[cells_all_nan, :] = 0
            weights_arr[cells_all_nan] = 0

        # Replace any remaining NaNs with the fill value, keeping the cell's weight.
        # Note this pulls mean/std towards the fill value.
        if fill_nans is not None:
            region_values = np.nan_to_num(region_values, nan=fill_nans)

        # Vectorized aggregation over all time steps
        # region_values and effective_weights: shape (num_cells, num_times)
        if exclude_nan_cells and cells_all_nan.all():
            # No data left in the region: NaN for every method, so sum doesn't report 0
            agg = np.full(region_values.shape[1], np.nan)
        elif method == "mean":
            try:
                agg = np.average(region_values, axis=0, weights=weights_arr)
            except ZeroDivisionError:
                logger.warning(f"All weights are zero for region_id {region_id}. Setting mean to NaN.")
                agg = np.full(region_values.shape[1], np.nan)
        elif method == "sum":
            agg = np.sum(region_values * weights_arr[:, None], axis=0)
        elif method == "std":
            try:
                mean = np.average(region_values, axis=0, weights=weights_arr)
                variance = np.average((region_values - mean) ** 2, axis=0, weights=weights_arr)
                agg = np.sqrt(variance)
            except ZeroDivisionError:
                logger.warning(f"All weights are zero for region_id {region_id}. Setting standard deviation to NaN.")
                agg = np.full(region_values.shape[1], np.nan)
        else:
            raise ValueError("Unknown aggregation method")

        # Store the result for all time steps for this region (vectorized, no loop)
        region_results = pd.DataFrame({
            "region_id": region_id,
            "time": pd.to_datetime(time_var),
            var: agg
        })
        results.append(region_results)

    # Convert results to DataFrame and write to CSV
    logger.info(f"Merging results and writing to output CSV {output}")
    out_df = pd.concat(results)
    out_df.to_csv(output, index=False)


def main():
    """ Main function to parse arguments and run aggregation. """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Aggregate gridded NetCDF data to regions using weights.")
    parser.add_argument("--netcdf", required=True, help="Input NetCDF file")
    
    weight_group = parser.add_mutually_exclusive_group(required=True)
    add_option(weight_group, "--weights", help="CSV file with grid weights (lat,lon,region_id,percent_overlap)")
    add_option(weight_group, "--precalc-weights", legacy="--precalc_weights",
               help="Name of precalculated weights (e.g., ERA5_to_GADM36_admin1_weights.csv)")

    parser.add_argument("--method", required=True, choices=["mean", "sum", "std"], help="Aggregation method")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--var", required=True, help="Variable name in NetCDF to aggregate")
    add_option(parser, "--correct-weights", legacy="--correct_weights", action="store_true",
               help="Correct weights to ensure no values are attributed to the sea")
    add_option(parser, "--time-var", legacy="--time_var", default=None,
               help="Optional time variable name in NetCDF (if not automatically detected)")
    add_option(
        parser, "--secondary-weights-var", legacy="--secondary_weights_var",
        default=None,
        help=(
            "Optional NetCDF variable name for secondary grid weights. "
            "Must have dimensions (lat, lon) and is multiplied with overlap weights."
        ),
    )
    add_option(
        parser, "--secondary-weights-netcdf", legacy="--secondary_weights_netcdf",
        default=None,
        help=(
            "Optional NetCDF file path containing secondary grid weights. "
            "Required when --secondary-weights-var is provided."
        ),
    )
    add_option(
        parser, "--allow-nan", legacy="--allow_nan",
        action="store_true",
        help=(
            "Proceed even if the NetCDF variable contains NaN values. "
            "Regions covering a NaN grid cell will aggregate to NaN."
        ),
    )
    add_option(
        parser, "--exclude-nan-cells", legacy="--exclude_nan_cells",
        action="store_true",
        help=(
            "Exclude grid cells that are NaN at every time step (e.g. ocean cells) from "
            "the aggregation. Implies --allow-nan for the variable."
        ),
    )
    add_option(
        parser, "--fill-nans", legacy="--fill_nans",
        type=float,
        default=None,
        metavar="VALUE",
        help=(
            "Replace NaN values with VALUE while keeping the cell's weight. Runs after "
            "--exclude-nan-cells if both are given. Biases mean/std towards VALUE. "
            "Implies --allow-nan for the variable."
        ),
    )
    add_shift_longitudes(parser)
    add_option(
        parser, "--area-weighted", legacy="--area_weighted",
        action="store_true",
        help=(
            "Weight grid cells by their area (covered fraction x cell area) for mean/std. "
            "Requires weights with a cell_area_km2 column (compute_grid_weights --method exact)."
        ),
    )
    args = parser.parse_args()
    setup_logging()

    if args.secondary_weights_netcdf and not args.secondary_weights_var:
        parser.error("--secondary-weights-netcdf requires --secondary-weights-var.")
    if args.secondary_weights_var and not args.secondary_weights_netcdf:
        parser.error("--secondary-weights-var requires --secondary-weights-netcdf.")

    if args.precalc_weights:
        weights = precalc_weights_path(args.precalc_weights)
    else:
        weights = args.weights
    
    # Run aggregation
    aggregate(
        netcdf=args.netcdf,
        weights=weights,
        method=args.method,
        output=args.output,
        var=args.var,
        correct_weights_flag=args.correct_weights,
        time_var_name=args.time_var,
        secondary_weights_var=args.secondary_weights_var,
        secondary_weights_netcdf=args.secondary_weights_netcdf,
        allow_nan=args.allow_nan,
        shift_longitudes=args.shift_longitudes,
        exclude_nan_cells=args.exclude_nan_cells,
        fill_nans=args.fill_nans,
        area_weighted=args.area_weighted,
    )

if __name__ == "__main__":
    main()
