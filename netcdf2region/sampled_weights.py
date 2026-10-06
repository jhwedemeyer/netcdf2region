"""Grid-cell/region weights estimated by point sampling.

Every grid cell is filled with ``points_per_cell`` points (on a regular lattice or at
random), the points are spatially joined with the regions, and a cell's
``percent_overlap`` for a region is the share of its points that fall inside it.
Regions too small to catch any point are rescued afterwards with their exact geometric
overlap (see :func:`assign_missing_regions`).
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shapely.geometry import Point, box
from tqdm import tqdm

from .data_handling import cell_bounds, load_regions, load_template_grid


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

def generate_point_distribution(coordinates, points_per_cell, point_distribution):
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

    # Loop over grid cells
    for i in tqdm(range(n_lat)):
        for j in range(n_lon):
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

def assign_missing_regions(present_ids, regions, coordinates):
    """Assign weights to regions with no sampled overlap via exact geometry.

    Point sampling misses regions that are smaller than the point spacing (no sample
    point ever falls inside them), so they never appear in the sampled weights and would
    be silently dropped from the CSV. For each region whose id is not in ``present_ids``
    this computes the exact geometric overlap fraction ``area(region ∩ cell) / area(cell)``
    with every grid cell it intersects and returns those rows so the region is represented.

    Cells whose overlap is a negligible sliver (rounds to
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
                    simplify, shapefile_layer, \
                    point_distribution, backend, n_jobs, \
                    yes, coordinate_origin, default_gid,
                    shift_longitudes):
    """Estimate region overlaps with NetCDF grid cells via point sampling."""

    # Load the NetCDF grid
    coordinates = load_template_grid(netcdf_path, coordinate_origin=coordinate_origin, shift_longitudes=shift_longitudes)

    # Load regions
    print(f"Loading shapefile: {shapefile_path}")
    regions = load_regions(shapefile_path, simplify=simplify, shapefile_layer=shapefile_layer, yes=yes, default_gid=default_gid)

    # Generate points for each grid cell
    print(f"Generating {point_distribution}-distributed points for grid cells...")
    point_gdf, cell_coords = generate_point_distribution(coordinates, points_per_cell, point_distribution)

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
