#!/usr/bin/env python3

import argparse
import xarray as xr
import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import box, Point
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
from scipy.ndimage import maximum_filter
import os


def load_template_grid(netcdf_path, coordinate_origin = "centered", yes=False):
    """Load the NetCDF grid and infer its properties."""

    # Load NetCDF grid
    print(f"Loading NetCDF: {netcdf_path}")

    if not os.path.exists(netcdf_path):
        raise FileNotFoundError(f"NetCDF file not found: {netcdf_path}")

    ds = xr.open_dataset(netcdf_path)
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

def load_land_sea_mask(mask_nc_path, coordinates, yes=False):
    """Load the land-sea mask NetCDF and verify its shape matches the grid."""
    lats = coordinates['lat']
    lons = coordinates['lon']

    if not os.path.exists(mask_nc_path):
        raise FileNotFoundError(f"Land-sea mask file not found: {mask_nc_path}")

    mask_ds = xr.open_dataset(mask_nc_path)

    # Use the variable that is not 'time', 'lon', or 'lat'
    exclude_names = {"time", "lon", "lat", "longitude", "latitude"}
    mask_var = None
    for v in mask_ds.variables:
        if v not in exclude_names:
            mask_var = v
            print(f"Using mask variable: {mask_var}")
            break
    if mask_var is None:
        print(f"Variables found in mask NetCDF: {list(mask_ds.variables)}")
        if not yes:
            mask_var = input("Could not automatically find mask variable. Please enter the variable name to use as mask: ")
        else:
            mask_var = list(mask_ds.variables)[0]
        if mask_var not in mask_ds.variables:
            raise KeyError(f"Variable '{mask_var}' not found in mask NetCDF file.")
    mask = mask_ds[mask_var].values

    # Verify mask shape matches grid shape
    if mask.shape != (len(lats), len(lons)):
        raise ValueError(f"Mask shape {mask.shape} does not match grid shape {(len(lats), len(lons))}")
    mask_lats = mask_ds['lat'].values if 'lat' in mask_ds else mask_ds['latitude'].values
    mask_lons = mask_ds['lon'].values if 'lon' in mask_ds else mask_ds['longitude'].values
    if mask_lats.min() != lats.min() and mask_lats.max() != lats.max() and mask_lons.min() != lons.min() and mask_lons.max() != lons.max():
        raise ValueError("Mask lat/lon values must match grid lat/lon values exactly. Please check your mask NetCDF file.")
    
    # Expand mask so that all neighboring cells (including diagonals) of mask==1 are also set to 1 (fast numpy version)
    mask = maximum_filter(mask, size=10, mode='constant', cval=0)
    
    return mask

def load_regions(shapefile_path, simplify=True, shapefile_layer=None, yes=False):
    """Load regions from a shapefile and optionally simplify geometries."""

    regions = gpd.read_file(shapefile_path, layer=shapefile_layer).to_crs("EPSG:4326")
    if 'GID_1' in regions.columns:
        regions = regions.rename(columns={'GID_1': 'region_id'})
    elif 'GID_0' in regions.columns:
        regions = regions.rename(columns={'GID_0': 'region_id'})
    else:
        print("Column 'GID_1' not found in shapefile.")
        print(f"Available columns: {list(regions.columns)}")
        if not yes:
            region_col = input("Please enter the column name to use as region identifier: ")
        else:
            region_col = list(regions.columns)[0]
        if region_col not in regions.columns:
            raise KeyError(f"Column '{region_col}' not found in shapefile.")
        regions = regions.rename(columns={region_col: 'region_id'})

    if simplify:
        print("Simplifying region geometries...")
        regions["geometry"] = regions["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    return regions

def random_points_in_cell(cell_bounds, n):
    """Generate n random points inside the bounding box."""
    minx, miny, maxx, maxy = cell_bounds
    xs = np.random.uniform(minx, maxx, n)
    ys = np.random.uniform(miny, maxy, n)
    return [Point(x, y) for x, y in zip(xs, ys)]

def uniform_points_in_cell(cell_bounds, n):
    """Generate n points uniformly distributed inside the bounding box."""
    minx, miny, maxx, maxy = cell_bounds
    nx = int(np.sqrt(n))
    ny = int(np.ceil(n / nx))
    xs = np.linspace(minx, maxx, nx, endpoint=False) + (maxx - minx) / (2 * nx)
    ys = np.linspace(miny, maxy, ny, endpoint=False) + (maxy - miny) / (2 * ny)
    points = [Point(x, y) for x in xs for y in ys]
    return points[:n]

def generate_point_distribution(mask, coordinates, points_per_cell, point_distribution):
    """Generate points for each grid cell based on the specified distribution."""

    lats = coordinates['lat']
    lons = coordinates['lon']
    lat_res = coordinates['lat_res']
    lon_res = coordinates['lon_res']
    coordinate_origin = coordinates['coordinate_origin']

    points = []
    cell_ids = []
    cell_coords = {}
    
    if mask is not None:
        n_land_cells = np.sum(mask == 1)
        print(f"Total grid cells: {len(lats)-1} x {len(lons)-1} = {(len(lats)-1)*(len(lons)-1)}")
        print(f"Land cells (mask=1): {n_land_cells}")
        print("Warning: Using a land-sea mask could result in lower weight quality in areas containing a large number of ocean cells.")

    # Loop over grid cells
    for i in tqdm(range(len(lats) - 1)):
        for j in range(len(lons) - 1):
            if mask is not None:
                if mask[i, j] != 1:
                    # skip ocean/non-land cells
                    continue
            if coordinate_origin == "centered":
                lat1, lat2 = lats[i] - lat_res / 2, lats[i] + lat_res / 2
                lon1, lon2 = lons[j] - lon_res / 2, lons[j] + lon_res / 2
            elif coordinate_origin == "top-left":
                lat1, lat2 = lats[i], lats[i+1]
                lon1, lon2 = lons[j], lons[j+1]

            bounds = (lon1, lat2, lon2, lat1)
            if point_distribution == "random":
                pts = random_points_in_cell(bounds, points_per_cell)
            elif point_distribution == "uniform":
                pts = uniform_points_in_cell(bounds, points_per_cell)
            else:
                raise ValueError(f"Unknown point distribution: {point_distribution}. Use 'random' or 'uniform'.")
            points.extend(pts)
            cell_ids.extend([(i, j)] * points_per_cell)
            cell_coords[(i, j)] = (lats[i], lons[j])

    point_gdf = gpd.GeoDataFrame({
        "geometry": points,
        "cell": cell_ids
    }, crs="EPSG:4326")
    
    return point_gdf, cell_coords

def calc_cell_region_overlap(point_gdf, regions, points_per_cell, n_jobs=8, n_batches=None, backend='threading'):
    """Calculate overlaps between grid cells and regions using point sampling."""

    def sjoin_batch(batch):
        return gpd.sjoin(batch, regions, how="left")

    if n_batches is None:
        n_batches = n_jobs * 2  # default to double the number of jobs for better load balancing
    batches = np.array_split(point_gdf, n_batches)
    results = Parallel(n_jobs=n_jobs, backend=backend)(delayed(sjoin_batch)(b) for b in batches)
    joined = pd.concat(results)

    print("Aggregating region overlaps per grid cell...")
    grouped = (
        joined.groupby(["cell", "region_id"])
        .size()
        .reset_index(name="count")
        .dropna(subset=["region_id"])
    )
    grouped["percent_overlap"] = grouped["count"] / points_per_cell

    return grouped

def compute_overlap(netcdf_path, shapefile_path, output_csv, points_per_cell=100, \
                    simplify=True, mask_nc_path=None, shapefile_layer=None, \
                    point_distribution="uniform", backend='threading', n_jobs=8, \
                    yes=False, coordinate_origin="centered"):
    """Estimate region overlaps with NetCDF grid cells via point sampling."""
    if yes:
        print("Automatic 'yes' mode enabled. All prompts will be answered with 'yes'.")

    # Load the NetCDF grid
    coordinates = load_template_grid(netcdf_path, coordinate_origin=coordinate_origin, yes=yes)

    # Load land-sea mask if available
    if mask_nc_path is not None:
        print(f"Loading land-sea mask NetCDF: {mask_nc_path}")
        mask = load_land_sea_mask(mask_nc_path, coordinates, yes=yes)
    else: 
        mask = None

    # Load regions
    print(f"Loading shapefile: {shapefile_path}")
    regions = load_regions(shapefile_path, simplify=simplify, shapefile_layer=shapefile_layer, yes=yes)

    # Generate points for each grid cell
    print(f"Generating {point_distribution}-distributed points for all {len(coordinates['lat'])-1}×{len(coordinates['lon'])-1} grid cells...")
    point_gdf, cell_coords = generate_point_distribution(mask, coordinates, points_per_cell, point_distribution)

    # Estiamte overlaps with regions
    print("Calculating overlaps between grid cells and regions...")
    grouped = calc_cell_region_overlap(point_gdf, regions, points_per_cell, backend=backend, n_jobs=n_jobs)

    # Building result table
    rows = []
    for _, row in grouped.iterrows():
        cell = row["cell"]
        lat, lon = cell_coords[cell]
        region_id = row["region_id"]
        percent = round(row["percent_overlap"], 2)
        rows.append({
            "lat": round(lat, 3),
            "lon": round(lon, 3),
            "region_id": region_id,
            "percent_overlap": percent
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")         

def main():
    """ Main function to parse arguments and run the overlap computation. """
    parser = argparse.ArgumentParser(description="Estimate region overlaps with NetCDF grid cells via point sampling.")
    parser.add_argument("netcdf", help="Path to NetCDF file with lat/lon grid")
    parser.add_argument("shapefile", help="Path to shapefile (e.g., .shp)")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument("--points_per_cell", type=int, default=100, help="Number of random points per grid cell (default: 100)")
    parser.add_argument("--simplify", action="store_true", help="Simplify region geometries")
    parser.add_argument("--land_sea_mask", default=None, help="Path to land-sea mask NetCDF file (optional)")
    parser.add_argument("--shapefile_layer", default=None, type=int, help="Layer number in shapefile to use (default: None, uses first layer)")
    parser.add_argument("--backend", default="threading", help="Parallel backend to use (default: threading)")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs (default: 8)")
    parser.add_argument("--yes", action="store_true", help="Automatically answer yes to prompts")
    parser.add_argument("--coordinate_origin", choices=["centered", "top-left"], default="centered",
                        help="Coordinate origin of the grid (default: centered)")
    args = parser.parse_args()

    compute_overlap(
        netcdf_path=args.netcdf,
        shapefile_path=args.shapefile,
        output_csv=args.output,
        points_per_cell=args.points_per_cell,
        simplify=args.simplify,
        mask_nc_path=args.land_sea_mask,
        shapefile_layer=args.shapefile_layer,
        point_distribution="uniform",
        backend=args.backend,
        n_jobs=args.n_jobs,
        yes=args.yes,
        coordinate_origin=args.coordinate_origin
    )

if __name__ == "__main__":
    main()