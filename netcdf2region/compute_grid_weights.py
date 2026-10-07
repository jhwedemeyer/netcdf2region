#!/usr/bin/env python3
"""Computing grid-cell/region weights: :func:`compute_weights` and the
``compute_grid_weights`` command.

The weights themselves are computed in :mod:`exact_weights` (exact geometric overlap
plus cell areas, the default) or :mod:`sampled_weights` (point sampling); the input
loading they share lives in :mod:`load_grid` and :mod:`load_regions`.
"""

import argparse
import logging

from .cli import add_grid_options, add_option, add_region_options, setup_logging
from .exact_weights import exact_weights
from .load_grid import load_template_grid
from .load_regions import load_regions
from .sampled_weights import sampled_weights
from .validate_weights import validate_weights

logger = logging.getLogger(__name__)

METHODS = ("exact", "sampling")


def compute_weights(
    netcdf_path,
    shapefile_path,
    output_csv=None,
    method="exact",
    *,
    gid="GID_1",
    shapefile_layer=None,
    simplify=False,
    coordinate_origin="centered",
    shift_longitudes=False,
    points_per_cell=100,
    point_distribution="uniform",
    n_jobs=1,
    backend="threading",
    validate=True,
    max_cell_sum=1.01,
):
    """Compute the weights of every grid cell for every region.

    Args:
        netcdf_path: NetCDF file whose lat/lon grid the weights are computed for.
        shapefile_path: Region file (anything geopandas reads).
        output_csv: Where to write the weights; ``None`` only returns them.
        method: ``"exact"`` intersects the geometries and adds each cell's area
            (``cell_area_km2``); ``"sampling"`` estimates overlaps from sample points.
        gid: Column holding the region identifier.
        shapefile_layer: Layer of a multi-layer file such as a GeoPackage.
        simplify: Simplify the region outlines first (faster, less exact).
        coordinate_origin: ``"centered"`` or ``"top-left"``, see
            :class:`~netcdf2region.load_grid.Grid`.
        shift_longitudes: Shift a [0, 360] grid to [-180, 180].
        points_per_cell: Sample points per cell, only for ``"sampling"``.
        point_distribution: ``"uniform"`` or ``"random"``, only for ``"sampling"``.
        n_jobs: Parallel jobs.
        backend: joblib backend.
        validate: Check the weights with :func:`~netcdf2region.validate_weights.validate_weights`
            against the grid and regions they were computed from. The report is logged and
            the CSV is still written, but a failed error-level check then raises
            ``ValueError``. Warnings (e.g. cell sums above ``max_cell_sum``) only get logged.
        max_cell_sum: Ceiling for the per-cell weight sum warning of the validation.

    Returns:
        pandas.DataFrame: ``lat, lon, region_id, percent_overlap`` (plus
        ``cell_area_km2`` for ``"exact"``).
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method '{method}'. Use one of {METHODS}.")

    grid = load_template_grid(netcdf_path, coordinate_origin=coordinate_origin,
                              shift_longitudes=shift_longitudes)
    logger.info(f"Loading shapefile: {shapefile_path}")
    regions = load_regions(shapefile_path, simplify=simplify, shapefile_layer=shapefile_layer,
                           default_gid=gid)

    if method == "exact":
        logger.info("Calculating exact overlaps between grid cells and regions...")
        df = exact_weights(regions, grid, n_jobs=n_jobs, backend=backend)
        # Fractions keep 8 significant digits, so even tiny regions keep a non-zero weight.
        float_format = "%.8g"
    else:
        df = sampled_weights(regions, grid, points_per_cell=points_per_cell,
                             point_distribution=point_distribution, n_jobs=n_jobs,
                             backend=backend)
        float_format = None

    # Both methods represent every region that touches the grid (sampling rescues the
    # ones it misses), so a region can only be absent if it lies outside the grid.
    missing = sorted(set(regions["region_id"]) - set(df["region_id"]))
    if missing:
        logger.warning(f"{len(missing)} region(s) do not overlap the grid at all: {missing[:10]}")

    if output_csv is not None:
        df.to_csv(output_csv, index=False, float_format=float_format)
        logger.info(f"Done. {len(df)} rows for {df['region_id'].nunique()} regions saved to {output_csv}")

    if validate:
        # Runs after writing, so a long computation is not lost if a check fails.
        logger.info("Validating the weights...")
        report = validate_weights(df, grid=grid, regions=regions, max_cell_sum=max_cell_sum,
                                  method=method)
        report.source = output_csv or report.source
        logger.info("\n" + report.summary())
        report.raise_if_failed()
    return df


def main():
    """ Main function to parse arguments and run the overlap computation. """
    parser = argparse.ArgumentParser(description="Compute region overlaps with NetCDF grid cells, exactly or by point sampling.")
    parser.add_argument("netcdf", help="Path to NetCDF file with lat/lon grid")
    parser.add_argument("shapefile", help="Path to shapefile (e.g., .shp)")
    parser.add_argument("output", help="Path to output CSV file")
    add_option(parser, "--method", choices=METHODS, default="exact",
               help="'exact' intersects the geometries and also writes each cell's area, which "
                    "netcdf2region --area-weighted uses; 'sampling' estimates overlaps from points "
                    "per cell (default: exact)")
    add_option(parser, "--points-per-cell", legacy="--points_per_cell", type=int, default=100,
               help="Number of points per grid cell, only with --method sampling (default: 100)")
    add_option(parser, "--point-distribution", legacy="--point_distribution",
               choices=["random", "uniform"], default="uniform",
               help="Distribution of points within grid cells, only with --method sampling (default: uniform)")
    add_option(parser, "--simplify", action="store_true", help="Simplify region geometries")
    add_region_options(parser)
    add_grid_options(parser)
    add_option(parser, "--n-jobs", legacy="--n_jobs", type=int, default=8,
               help="Number of parallel jobs (default: 8)")
    add_option(parser, "--backend", default="threading", help="Parallel backend to use (default: threading)")
    add_option(parser, "--no-validate", dest="validate", action="store_false",
               help="Skip validating the computed weights (by default they are checked against "
                    "the grid and regions, and the command fails if an error-level check fails)")
    add_option(parser, "--max-cell-sum", type=float, default=1.01,
               help="Validation warns when a cell's weights sum above this (default: 1.01)")
    args = parser.parse_args()
    setup_logging()

    compute_weights(
        args.netcdf,
        args.shapefile,
        args.output,
        method=args.method,
        gid=args.gid,
        shapefile_layer=args.shapefile_layer,
        simplify=args.simplify,
        coordinate_origin=args.coordinate_origin,
        shift_longitudes=args.shift_longitudes,
        points_per_cell=args.points_per_cell,
        point_distribution=args.point_distribution,
        n_jobs=args.n_jobs,
        backend=args.backend,
        validate=args.validate,
        max_cell_sum=args.max_cell_sum,
    )


if __name__ == "__main__":
    main()
