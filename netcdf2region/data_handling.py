"""Data handling shared by the weight calculations: loading the NetCDF template grid
and the region geometries, and describing the grid's cells.

Used by both :mod:`sampled_weights` and :mod:`exact_weights`, and by ``validate_weights``.
"""

import os

import geopandas as gpd
import numpy as np
import xarray as xr


def load_template_grid(netcdf_path, coordinate_origin, shift_longitudes):
    """Load the NetCDF grid and infer its properties."""

    # Load NetCDF grid
    print(f"Loading NetCDF: {netcdf_path}")

    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")

    ds = xr.open_dataset(netcdf_path)
    if 'time' in ds.dims:
        ds = ds.isel(time=0)

    if 'lat' in ds:
        lats = ds['lat'].values
    elif 'latitude' in ds:
        lats = ds['latitude'].values
    else:
        raise KeyError("NetCDF file must contain 'lat' or 'latitude' variable.")

    if 'lon' in ds:
        lons = ds['lon'].values
    elif 'longitude' in ds:
        lons = ds['longitude'].values
    else:
        raise KeyError("NetCDF file must contain 'lon' or 'longitude' variable.")    

    # Check if lat/lon are in EPSG:4326 range, shift if necessary
    if not np.all((lats >= -90) & (lats <= 90)):
        raise ValueError("Latitude values are not in EPSG:4326 range (-90 to 90).")
    if not np.all((lons >= -180) & (lons <= 180)):
        # Try shifting longitude if out of range
        if shift_longitudes:
            if np.all((lons >= 0) & (lons <= 360)):
                print("Shifting longitudes from [0, 360] to [-180, 180].")
                print("Double check if this works for your data!")
                lons = ((lons + 180) % 360) - 180
                # Sort longitudes and reorder data accordingly
                sort_idx = np.argsort(lons)
                lons = lons[sort_idx]
                ds = ds.assign_coords(lon=lons)
                ds = ds.sortby('lon')
            else:
                raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360).")
        else: 
            raise ValueError("Longitude values are not in EPSG:4326 range (-180 to 180 or 0 to 360). You might want to enable longitude shifting with the --shift_longitudes flag if your data uses [0, 360] range.")
        
    # Calculate grid resolution
    lat_res = np.abs(lats[1] - lats[0])
    lon_res = np.abs(lons[1] - lons[0])
    print(f"Grid resolution: {lat_res:.2f}° latitude × {lon_res:.2f}° longitude")

    # Check coordinate origin
    if coordinate_origin not in ["centered", "top-left"]:
        raise ValueError("Invalid coordinate origin. Must be 'centered' or 'top-left'.")

    # check if lats/lons are sorted
    if not np.all(np.diff(lons) > 0):
        raise ValueError("Longitudes must be in increasing order")

    coordinates = {
        "lat": lats,
        "lon": lons,
        "lat_res": lat_res,
        "lon_res": lon_res,
        "coordinate_origin": coordinate_origin
    }

    return coordinates

def load_regions(shapefile_path, simplify, shapefile_layer, default_gid, yes=False):
    """Load regions from a shapefile and optionally simplify geometries."""

    regions = gpd.read_file(shapefile_path, layer=shapefile_layer).to_crs("EPSG:4326")
    if default_gid in regions.columns:
        regions = regions.rename(columns={default_gid: 'region_id'})
    else:
        print(f"Column {default_gid} not found in shapefile.")
        print(f"Available columns: {list(regions.columns)}")
        if not yes:
            region_col = input("Please enter the column name to use as region identifier: ")
        else:
            region_col = list(regions.columns)[0]
            print(f"Using first column as region identifier: {region_col}")
        if region_col not in regions.columns:
            raise KeyError(f"Column '{region_col}' not found in shapefile.")
        regions = regions.rename(columns={region_col: 'region_id'})

    if simplify:
        print("Simplifying region geometries...")
        regions["geometry"] = regions["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    return regions

def cell_bounds(i, j, coordinates):
    """Return the (lon1, lat1, lon2, lat2) edges of grid cell (i, j)."""
    lats = coordinates['lat']
    lons = coordinates['lon']
    lat_res = coordinates['lat_res']
    lon_res = coordinates['lon_res']
    if coordinates['coordinate_origin'] == "centered":
        lat1, lat2 = lats[i] - lat_res / 2, lats[i] + lat_res / 2
        lon1, lon2 = lons[j] - lon_res / 2, lons[j] + lon_res / 2
    else:  # top-left: coordinates are cell edges
        lat1, lat2 = lats[i], lats[i + 1]
        lon1, lon2 = lons[j], lons[j + 1]
    return min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2)
