#%%
import argparse
import pandas as pd
import numpy as np
import xarray as xr
from importlib.resources import files
from tqdm import tqdm

def precalc_weights_path(name: str):
    """ Get the path to a precalculated weights file from the package data. """
    try:
        path = files("netcdf2region.data").joinpath(name)
        return path
    except FileNotFoundError:
        raise ValueError(f"Precalculated weights file '{name}' not found in package data.")

def load_netcdf(netcdf_path, var, time_var=None):
    """ Load NetCDF file and extract variable, lat, lon, and time. """
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

    # Try to automatically detect the time variable
    if time_var is None:
        exclude_names = ['lat', 'lon', 'latitude', 'longitude', var]
        for v in ds.variables:
            if v not in exclude_names:
                ds_vars["time"] = v
                print(f"Using time variable: {ds_vars['time']}")
                break
        if ds_vars["time"] is None:
            print(f"Variables found in mask NetCDF: {list(ds.variables)}")
            ds_vars["time"] = input("Could not automatically find time variable. Please enter the variable name to use as time: ")
            if ds_vars["time"] not in ds.variables:
                raise KeyError(f"Variable '{ds_vars.time}' not found in NetCDF file.")
    else:
        ds_vars["time"] = time_var
        if ds_vars["time"] not in ds.variables:
            raise KeyError(f"Variable '{ds_vars['time']}' not found in NetCDF file. Available variables: {list(ds.variables)}")
        
    return ds[ds_vars["var"]].values, ds[ds_vars["lat"]].values, ds[ds_vars["lon"]].values, ds[ds_vars["time"]].values

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


def aggregate(netcdf, weights, method, output, var, correct_weights_flag=False, time_var_name=None):
    """ Aggregate gridded NetCDF data to regions using weights. """

    # Load weights CSV file (should contain columns: lat, lon, region_id, percent_overlap)
    print(f"Loading weights file: {weights}")
    weights_df = pd.read_csv(weights)
    if correct_weights_flag:
        weights_df = correct_weights(weights_df)

    # Load NetCDF data
    print(f"Loading NetCDF file: {netcdf}")
    var_data, lats, lons, time_var = load_netcdf(netcdf, var, time_var=time_var_name)

    results = []


    # Loop over each unique region
    for region_id in tqdm(weights_df['region_id'].unique(), desc="Aggregating regions"):

        # Select weights for the current region
        region_weights = weights_df[weights_df['region_id'] == region_id]

        # Build mask and weights for each grid cell in the region
        mask = list(zip(region_weights['lat'], region_weights['lon']))
        cell_weights = region_weights['percent_overlap'].tolist()

        # Prepare arrays for indices and weights
        lat_indices = np.array([np.where(lats == lat)[0][0] for lat, _ in mask])
        lon_indices = np.array([np.where(lons == lon)[0][0] for _, lon in mask])
        weights_arr = np.array(cell_weights)

        # Extract the relevant grid cells for all time steps at once
        # Shape: (num_cells, num_times)
        region_values = var_data[:, lat_indices, lon_indices].T  # shape: (num_cells, num_times)

        # Vectorized aggregation over all time steps
        # region_values: shape (num_cells, num_times)
        if method == "mean":
            agg = np.average(region_values, axis=0, weights=weights_arr)
        elif method == "sum":
            agg = np.sum(region_values * weights_arr[:, None], axis=0)
        elif method == "std":
            mean = np.average(region_values, axis=0, weights=weights_arr)
            variance = np.average((region_values - mean) ** 2, axis=0, weights=weights_arr)
            agg = np.sqrt(variance)
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
    args = parser.parse_args()

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
        time_var_name=args.time_var
    )

if __name__ == "__main__":
    main()