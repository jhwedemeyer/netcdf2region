#!/usr/bin/env python3
"""Command-line entry point for computing grid-cell/region weights.

The weights themselves are computed in :mod:`sampled_weights` (point sampling, the
default) or :mod:`exact_weights` (exact geometric overlap plus cell areas); the input
loading they share lives in :mod:`load_grid` and :mod:`load_regions`.
"""

import argparse
import logging

from .exact_weights import compute_exact_overlap
from .sampled_weights import compute_overlap


def main():
    """ Main function to parse arguments and run the overlap computation. """
    parser = argparse.ArgumentParser(description="Compute region overlaps with NetCDF grid cells, by point sampling or exactly.")
    parser.add_argument("netcdf", help="Path to NetCDF file with lat/lon grid")
    parser.add_argument("shapefile", help="Path to shapefile (e.g., .shp)")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument("--method", choices=["exact", "sampling"], default="exact",
                        help="'exact' intersects the geometries and also writes each cell's area, so means "
                             "become area-weighted; 'sampling' estimates overlaps from points per cell "
                             "(default: exact)")
    parser.add_argument("--points_per_cell", type=int, default=100, help="Number of random points per grid cell, only with --method sampling (default: 100)")
    parser.add_argument("--point_distribution", choices=["random", "uniform"], default="uniform", help="Distribution of points within grid cells, only with --method sampling (default: uniform)")
    parser.add_argument("--simplify", action="store_true", help="Simplify region geometries")
    parser.add_argument("--shapefile_layer", default=None, type=int, help="Layer number in shapefile to use (default: None, uses first layer)")
    parser.add_argument("--backend", default="threading", help="Parallel backend to use (default: threading)")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of parallel jobs (default: 8)")
    parser.add_argument("--gid", type=str, default="GID_1", help="Geo identifier in the shapefile (default: GID_1)")
    parser.add_argument("--yes", action="store_true", help="If the --gid column is missing, use the shapefile's first column as region identifier")
    parser.add_argument("--coordinate_origin", choices=["centered", "top-left"], default="centered",
                        help="Coordinate origin of the grid (default: centered)")
    parser.add_argument("--shift_longitudes", action="store_true", help="Shift longitudes from [0, 360] to [-180, 180]")    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.method == "exact":
        compute_exact_overlap(
            netcdf_path=args.netcdf,
            shapefile_path=args.shapefile,
            output_csv=args.output,
            shapefile_layer=args.shapefile_layer,
            default_gid=args.gid,
            coordinate_origin=args.coordinate_origin,
            shift_longitudes=args.shift_longitudes,
            simplify=args.simplify,
            n_jobs=args.n_jobs,
            backend=args.backend,
            yes=args.yes,
        )
    elif args.method == "sampling":
        compute_overlap(
            netcdf_path=args.netcdf,
            shapefile_path=args.shapefile,
            output_csv=args.output,
            points_per_cell=args.points_per_cell,
            simplify=args.simplify,
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
