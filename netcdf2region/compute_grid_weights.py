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

def load_land_sea_mask(mask_nc_path, coordinates, yes=False):
    """Load the land-sea mask NetCDF and verify its shape matches the grid."""
    lats = coordinates['lat']
    lons = coordinates['lon']

    print("Warning: Using a land-sea mask could result in lower weight quality in areas containing a large number of ocean cells. Consider using a higher resolution mask or no mask if you encounter issues with weight quality.")

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
    # Any single mismatched edge means the mask describes a different grid, so
    # these must be combined with `or`: with `and` the check only fired when all
    # four edges differed, silently accepting masks that overlapped on one side.
    if mask_lats.min() != lats.min() or mask_lats.max() != lats.max() or mask_lons.min() != lons.min() or mask_lons.max() != lons.max():
        raise ValueError("Mask lat/lon values must match grid lat/lon values exactly. Please check your mask NetCDF file.")
    
    # Expand mask so that all neighboring cells (including diagonals) of mask==1 are also set to 1 (fast numpy version)
    mask = maximum_filter(mask, size=10, mode='constant', cval=0)
    
    return mask

def load_regions(shapefile_path, simplify, shapefile_layer, default_gid, yes=False):
    """Load regions from a shapefile and optionally simplify geometries."""

    regions = gpd.read_file(shapefile_path, layer=shapefile_layer).to_crs("EPSG:4326")
    if default_gid in regions.columns:
        regions = regions.rename(columns={default_gid: 'region_id'})
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

    # Number of cells depends on the coordinate origin: for "centered" grids every
    # coordinate is a cell center, so there are len(lats)/len(lons) cells. For "top-left"
    # grids the coordinates are cell edges, so there is one fewer cell than coordinates.
    if coordinate_origin == "centered":
        n_lat, n_lon = len(lats), len(lons)
    else:  # top-left
        n_lat, n_lon = len(lats) - 1, len(lons) - 1

    if mask is not None:
        n_land_cells = np.sum(mask == 1)
        print(f"Total grid cells: {n_lat} x {n_lon} = {n_lat*n_lon}")
        print(f"Land cells (mask=1): {n_land_cells}")
        print("Warning: Using a land-sea mask could result in lower weight quality in areas containing a large number of ocean cells.")

    # Loop over grid cells
    for i in tqdm(range(n_lat)):
        for j in range(n_lon):
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

def calc_cell_region_overlap(point_gdf, regions, points_per_cell, n_jobs, backend, n_batches=None):
    """Calculate overlaps between grid cells and regions using point sampling."""

    def sjoin_batch(batch):
        return gpd.sjoin(batch, regions, how="left")

    if n_batches is None:
        n_batches = n_jobs * 2  # default to double the number of jobs for better load balancing

    # Convert to list of indices instead of using np.array_split which can trigger deprecated methods
    #batches = np.array_split(point_gdf, n_batches)
    indices = np.array_split(np.arange(len(point_gdf)), n_batches)
    batches = [point_gdf.iloc[idx] for idx in indices]

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

def assign_missing_regions(present_ids, regions, coordinates):
    """Assign weights to regions with no sampled overlap via exact geometry.

    Point sampling misses regions that are smaller than the point spacing (no sample
    point ever falls inside them), so they never appear in the sampled weights and would
    be silently dropped from the CSV. For each region whose id is not in ``present_ids``
    this computes the exact geometric overlap fraction ``area(region ∩ cell) / area(cell)``
    with every grid cell it intersects and returns those rows so the region is represented.

    Operates on true geometry, so a region is assigned even to a cell that a land-sea
    mask excluded from sampling. Cells whose overlap is a negligible sliver (rounds to
    zero) are dropped rather than emitted as useless ``0.0`` rows; if that would leave a
    region with no cell at all, its single largest-overlap cell is kept with a small
    positive floor so the region stays represented (and downstream mean/std don't hit a
    ZeroDivisionError). Returns a DataFrame with columns
    ``lat, lon, region_id, percent_overlap`` (empty if no regions are missing).
    """
    lats = coordinates['lat']
    lons = coordinates['lon']
    coordinate_origin = coordinates['coordinate_origin']

    # Number of cells per axis, mirroring generate_point_distribution.
    if coordinate_origin == "centered":
        n_lat, n_lon = len(lats), len(lons)
    else:  # top-left
        n_lat, n_lon = len(lats) - 1, len(lons) - 1

    missing = regions[~regions["region_id"].isin(present_ids)]
    if missing.empty:
        return pd.DataFrame(columns=["lat", "lon", "region_id", "percent_overlap"])

    # Union multi-row / multipart regions into a single geometry per region_id.
    missing = missing.dissolve(by="region_id").reset_index()

    # Precompute the lower/upper edge of every cell along each axis (in ascending order
    # of the edges, regardless of whether lats/lons ascend or descend).
    lat_lo = np.array([cell_bounds(i, 0, coordinates)[1] for i in range(n_lat)])
    lat_hi = np.array([cell_bounds(i, 0, coordinates)[3] for i in range(n_lat)])
    lon_lo = np.array([cell_bounds(0, j, coordinates)[0] for j in range(n_lon)])
    lon_hi = np.array([cell_bounds(0, j, coordinates)[2] for j in range(n_lon)])

    rows = []
    for _, region in missing.iterrows():
        geom = region["geometry"]
        region_id = region["region_id"]
        if geom is None or geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds

        # Candidate cells are those whose extent overlaps the region bounding box.
        cand_i = np.where((lat_hi >= miny) & (lat_lo <= maxy))[0]
        cand_j = np.where((lon_hi >= minx) & (lon_lo <= maxx))[0]

        region_rows = []
        best_row = None
        best_area = 0.0
        for i in cand_i:
            for j in cand_j:
                lo_x, lo_y, hi_x, hi_y = cell_bounds(i, j, coordinates)
                cell = box(lo_x, lo_y, hi_x, hi_y)
                if not geom.intersects(cell):
                    continue
                inter_area = geom.intersection(cell).area
                if inter_area <= 0:
                    continue
                row = {
                    "lat": round(lats[i], 3),
                    "lon": round(lons[j], 3),
                    "region_id": region_id,
                    "percent_overlap": round(inter_area / cell.area, 6),
                }
                if inter_area > best_area:
                    best_area, best_row = inter_area, row
                # Drop negligible slivers that round to zero: a 0.0 weight is meaningless
                # and only pollutes the output.
                if row["percent_overlap"] > 0:
                    region_rows.append(row)

        if best_row is None:
            continue  # region does not actually intersect any cell

        # If every overlapping cell was a negligible sliver, keep the largest one with a
        # small positive floor so the region is still represented.
        if not region_rows:
            best_row["percent_overlap"] = 1e-6
            region_rows.append(best_row)

        rows.extend(region_rows)

    return pd.DataFrame(rows, columns=["lat", "lon", "region_id", "percent_overlap"])

def compute_overlap(netcdf_path, shapefile_path, output_csv, points_per_cell, \
                    simplify, mask_nc_path, shapefile_layer, \
                    point_distribution, backend, n_jobs, \
                    yes, coordinate_origin, default_gid,
                    shift_longitudes):
    """Estimate region overlaps with NetCDF grid cells via point sampling."""

    # Load the NetCDF grid
    coordinates = load_template_grid(netcdf_path, coordinate_origin=coordinate_origin, shift_longitudes=shift_longitudes)

    # Load land-sea mask if available
    if mask_nc_path is not None:
        print(f"Loading land-sea mask NetCDF: {mask_nc_path}")
        mask = load_land_sea_mask(mask_nc_path, coordinates, yes=yes)
    else: 
        mask = None

    # Load regions
    print(f"Loading shapefile: {shapefile_path}")
    regions = load_regions(shapefile_path, simplify=simplify, shapefile_layer=shapefile_layer, yes=yes, default_gid=default_gid)

    # Generate points for each grid cell
    print(f"Generating {point_distribution}-distributed points for grid cells...")
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

    df = pd.DataFrame(rows, columns=["lat", "lon", "region_id", "percent_overlap"])

    # Drop zero-weight rows: with a large --points_per_cell a single-point overlap can
    # round to 0.0, which is a meaningless weight (and yields NaN in downstream mean/std).
    df = df[df["percent_overlap"] > 0].reset_index(drop=True)

    # Rescue any region not represented by a positive sampled weight (regions too small to
    # catch a sample point, or whose only overlaps rounded to zero above) by computing
    # their exact geometric overlap with the grid.
    print("Checking for regions with no sampled overlap...")
    present_ids = set(df["region_id"].unique())
    missing_rows = assign_missing_regions(present_ids, regions, coordinates)
    if not missing_rows.empty:
        print(f"Rescued {missing_rows['region_id'].nunique()} region(s) with no sampled overlap via exact geometry.")
        df = pd.concat([df, missing_rows], ignore_index=True)

    df.to_csv(output_csv, index=False)
    print(f"Done. Results saved to {output_csv}")         

def main():
    """ Main function to parse arguments and run the overlap computation. """
    parser = argparse.ArgumentParser(description="Estimate region overlaps with NetCDF grid cells via point sampling.")
    parser.add_argument("netcdf", help="Path to NetCDF file with lat/lon grid")
    parser.add_argument("shapefile", help="Path to shapefile (e.g., .shp)")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument("--points_per_cell", type=int, default=100, help="Number of random points per grid cell (default: 100)")
    parser.add_argument("--point_distribution", choices=["random", "uniform"], default="uniform", help="Distribution of points within grid cells (default: uniform)")
    parser.add_argument("--simplify", action="store_true", help="Simplify region geometries")
    parser.add_argument("--land_sea_mask", default=None, help="Path to land-sea mask NetCDF file (optional)")
    parser.add_argument("--shapefile_layer", default=None, type=int, help="Layer number in shapefile to use (default: None, uses first layer)")
    parser.add_argument("--backend", default="threading", help="Parallel backend to use (default: threading)")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs (default: 8)")
    parser.add_argument("--gid", type=str, default="GID_1", help="Geo identifier in the shapefile (default: GID_1)")
    parser.add_argument("--yes", action="store_true", help="Automatically answer yes to prompts")
    parser.add_argument("--coordinate_origin", choices=["centered", "top-left"], default="centered",
                        help="Coordinate origin of the grid (default: centered)")
    parser.add_argument("--shift_longitudes", action="store_true", help="Shift longitudes from [0, 360] to [-180, 180]")    
    args = parser.parse_args()

    if args.yes:
        print("Automatic 'yes' mode enabled. All prompts will be answered with 'yes'. This may lead to incorrect assumptions if the script cannot automatically determine certain parameters. Use with caution.")


    compute_overlap(
        netcdf_path=args.netcdf,
        shapefile_path=args.shapefile,
        output_csv=args.output,
        points_per_cell=args.points_per_cell,
        simplify=args.simplify,
        mask_nc_path=args.land_sea_mask,
        shapefile_layer=args.shapefile_layer,
        point_distribution=args.point_distribution,
        backend=args.backend,
        n_jobs=args.n_jobs,
        yes=args.yes,
        coordinate_origin=args.coordinate_origin,
        default_gid=args.gid,
        shift_longitudes=args.shift_longitudes
    )

if __name__ == "__main__":
    main()
