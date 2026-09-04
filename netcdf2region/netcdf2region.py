#%%
import argparse
import pandas as pd
import numpy as np
import xarray as xr
from importlib.resources import files
from tqdm import tqdm

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

def load_netcdf(
        netcdf_path,
        var,
        time_var=None,
        require_time=True,
        allow_nan=False,
        shift_longitudes=False,
    ):
    """ Load NetCDF file and extract variable, lat, lon, and optional time. """
    # Load NetCDF dataset
    ds = xr.open_dataset(netcdf_path)

    # Get variable names
    ds_vars = {
        "var" : None,
        "lat" : None,
        "lon" : None,
        "time" : None
    }

    ds_vars["var"] = var
    if ds_vars["var"] not in ds:
        raise KeyError(f"Variable '{ds_vars['var']}' not found in NetCDF file. Available variables: {list(ds.variables)}")

    ds_vars["lat"] = "lat" if "lat" in ds else "latitude"
    if ds_vars["lat"] not in ds:
        raise KeyError(f"NetCDF file must contain 'lat' or 'latitude' variable. Available variables: {list(ds.variables)}")

    ds_vars["lon"] = "lon" if "lon" in ds else "longitude"
    if ds_vars["lon"] not in ds:
        raise KeyError(f"NetCDF file must contain 'lon' or 'longitude' variable. Available variables: {list(ds.variables)}")

    data_var = ds[ds_vars["var"]]
    if ds_vars["lat"] not in data_var.dims or ds_vars["lon"] not in data_var.dims:
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
        ds_vars["time"] = time_var
    else:
        candidate_time_dims = [d for d in data_var.dims if d not in {ds_vars["lat"], ds_vars["lon"]}]
        if len(candidate_time_dims) == 1:
            ds_vars["time"] = candidate_time_dims[0]
            if require_time:
                print(f"Using time variable: {ds_vars['time']}")
        elif len(candidate_time_dims) == 0:
            ds_vars["time"] = None
        else:
            raise ValueError(
                f"Could not infer a unique time dimension for variable '{var}'. "
                f"Found non-spatial dimensions: {candidate_time_dims}. Please provide --time_var."
            )

    if require_time and ds_vars["time"] is None:
        raise ValueError(
            f"Variable '{var}' has no time dimension. Please provide a variable with dimensions "
            f"(time, {ds_vars['lat']}, {ds_vars['lon']})."
        )

    if ds_vars["time"] is None:
        data_var = data_var.transpose(ds_vars["lat"], ds_vars["lon"])
        time_values = None
    else:
        data_var = data_var.transpose(ds_vars["time"], ds_vars["lat"], ds_vars["lon"])
        time_values = ds[ds_vars["time"]].values if ds_vars["time"] in ds.variables else None

    lat_values = ds[ds_vars["lat"]].values
    lon_values = ds[ds_vars["lon"]].values

    # Weights are always written in [-180, 180], so a file on a [0, 360] grid must
    # be shifted here or none of its cells would be found.
    if not np.all((lon_values >= -180) & (lon_values <= 180)):
        if shift_longitudes:
            if np.all((lon_values >= 0) & (lon_values <= 360)):
                print("Shifting longitudes from [0, 360] to [-180, 180].")
                print("Double check if this works for your data!")
                lon_values = ((lon_values + 180) % 360) - 180
                # Sort longitudes and reorder the data along the same axis.
                sort_idx = np.argsort(lon_values)
                lon_values = lon_values[sort_idx]
                data_var = data_var.isel({ds_vars["lon"]: sort_idx})
            else:
                raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360).")
        else:
            raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360). You might want to enable longitude shifting with the --shift_longitudes flag if your data uses [0, 360] range.")

    if np.isnan(data_var.values).any():
        if not allow_nan:
            raise ValueError(f"Variable '{var}' in file '{netcdf_path}' contains NaN values. This may affect the aggregation results. Add --allow_nan to proceed anyway.")
        print(f"Warning: Variable '{var}' in file '{netcdf_path}' contains NaN values. Affected regions will aggregate to NaN.")

    return (
        data_var.values,
        lat_values,
        lon_values,
        time_values,
    )

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
):
    """ Aggregate gridded NetCDF data to regions using weights. """

    # Load weights CSV file (should contain columns: lat, lon, region_id, percent_overlap)
    print(f"Loading weights file: {weights}")
    weights_df = pd.read_csv(weights)
    if correct_weights_flag:
        weights_df = correct_weights(weights_df)

    # Load NetCDF data
    print(f"Loading NetCDF file: {netcdf}")
    var_data, lats, lons, time_var = load_netcdf(
        netcdf,
        var,
        time_var=time_var_name,
        require_time=True,
        allow_nan=allow_nan,
        shift_longitudes=shift_longitudes,
    )

    # If secondary weights are provided, load and validate them
    secondary_weights_data = None
    if secondary_weights_var is not None:
        secondary_data, secondary_lats, secondary_lons, _ = load_netcdf(
            secondary_weights_netcdf,
            secondary_weights_var,
            time_var=None,
            require_time=False,
            allow_nan=allow_nan,
            shift_longitudes=shift_longitudes,
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
        if secondary_data.ndim == 3:
            secondary_weights_data = secondary_data[0, :, :]
        else:
            secondary_weights_data = secondary_data

    # initialize list to store results for each region
    results = []

    # Loop over each unique region
    for region_id in tqdm(weights_df['region_id'].unique(), desc="Aggregating regions"):

        # Select weights for the current region
        region_weights = weights_df[weights_df['region_id'] == region_id]

        # Build mask and weights for each grid cell in the region
        mask = list(zip(region_weights['lat'], region_weights['lon']))
        cell_weights = region_weights['percent_overlap'].tolist()

        # Prepare arrays for indices and weights. Coordinates are matched by exact
        # equality, so a weights file built for a different grid fails here; report
        # which cell is at fault instead of an unexplained IndexError.
        try:
            lat_indices = np.array([np.where(lats == lat)[0][0] for lat, _ in mask])
            lon_indices = np.array([np.where(lons == lon)[0][0] for _, lon in mask])
        except IndexError:
            unmatched = [
                (lat, lon) for lat, lon in mask
                if not len(np.where(lats == lat)[0]) or not len(np.where(lons == lon)[0])
            ]
            raise ValueError(
                f"Region '{region_id}' references grid cell(s) that do not exist in "
                f"'{netcdf}', e.g. {unmatched[:3]}. The NetCDF file has corrdinates like "
                f"{list(zip(lats, lons))[:3]}. The weights file must have been "
                f"computed for this exact grid. Check it with: "
                f"validate_weights <weights.csv> --netcdf <file.nc>"
            ) from None
        
        # Force float: an all-integer weights column would otherwise truncate the
        # secondary-weight multiplication below.
        weights_arr = np.array(cell_weights, dtype=float)

        # If secondary weights are provided, multiply the region weights by the secondary weights
        if secondary_weights_data is not None:
            secondary_region = secondary_weights_data[lat_indices, lon_indices][:, None]
            weights_arr[:, None] = weights_arr[:, None] * secondary_region

        # Extract the relevant grid cells for all time steps at once
        # Shape: (num_cells, num_times)
        region_values = var_data[:, lat_indices, lon_indices].T  # shape: (num_cells, num_times)

        # Vectorized aggregation over all time steps
        # region_values and effective_weights: shape (num_cells, num_times)
        if method == "mean":
            try:
                agg = np.average(region_values, axis=0, weights=weights_arr)
            except ZeroDivisionError:
                print(f"Warning: All weights are zero for region_id {region_id}. Setting mean to NaN.")
                agg = np.full(region_values.shape[1], np.nan)
        elif method == "sum":
            agg = np.sum(region_values * weights_arr[:, None], axis=0)
        elif method == "std":
            try:
                mean = np.average(region_values, axis=0, weights=weights_arr)
                variance = np.average((region_values - mean) ** 2, axis=0, weights=weights_arr)
                agg = np.sqrt(variance)
            except ZeroDivisionError:
                print(f"Warning: All weights are zero for region_id {region_id}. Setting standard deviation to NaN.")
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
    print("Merging results and writing to output CSV to ", output)
    out_df = pd.concat(results)
    out_df.to_csv(output, index=False)

def main():
    """ Main function to parse arguments and run aggregation. """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Aggregate gridded NetCDF data to regions using weights.")
    parser.add_argument("--netcdf", required=True, help="Input NetCDF file")
    
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument("--weights", help="CSV file with grid weights (lat,lon,region_id,percent_overlap)")
    weight_group.add_argument("--precalc_weights", help="Name of precalculated weights (e.g., ERA5_to_GADM36_admin1_weights.csv)")

    parser.add_argument("--method", required=True, choices=["mean", "sum", "std"], help="Aggregation method")
    parser.add_argument("--output", required=True, help="Output CSV file")
    parser.add_argument("--var", required=True, help="Variable name in NetCDF to aggregate")
    parser.add_argument("--correct_weights", action="store_true", help="Correct weights to ensure no values are attributed to the sea")
    parser.add_argument("--time_var", default=None, help="Optional time variable name in NetCDF (if not automatically detected)")
    parser.add_argument(
        "--secondary_weights_var",
        default=None,
        help=(
            "Optional NetCDF variable name for secondary grid weights. "
            "Must have dimensions (lat, lon) and is multiplied with overlap weights."
        ),
    )
    parser.add_argument(
        "--secondary_weights_netcdf",
        default=None,
        help=(
            "Optional NetCDF file path containing secondary grid weights. "
            "Required when --secondary_weights_var is provided."
        ),
    )
    parser.add_argument(
        "--allow_nan",
        action="store_true",
        help=(
            "Proceed even if the NetCDF variable contains NaN values. "
            "Regions covering a NaN grid cell will aggregate to NaN."
        ),
    )
    parser.add_argument(
        "--shift_longitudes",
        action="store_true",
        help="Shift NetCDF longitudes from [0, 360] to [-180, 180], as the weights use.",
    )
    args = parser.parse_args()

    if args.secondary_weights_netcdf and not args.secondary_weights_var:
        raise ValueError("--secondary_weights_netcdf requires --secondary_weights_var.")
    if args.secondary_weights_var and not args.secondary_weights_netcdf:
        raise ValueError("--secondary_weights_var requires --secondary_weights_netcdf.")

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
    )

if __name__ == "__main__":
    main()