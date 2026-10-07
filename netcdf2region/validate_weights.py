#!/usr/bin/env python3
"""Sanity checks for grid weight CSVs produced by ``compute_grid_weights``.

A weights CSV (``lat,lon,region_id,percent_overlap``) is easy to produce and hard
to check: the shipped files run to hundreds of thousands of rows. This module
collects the invariants such a file must satisfy so a precalculation run can
verify its own output before anything downstream consumes it.

The checks split into two groups. Checks 1-6 and 10 need nothing but the CSV, so
they run anywhere. Checks 7-9 compare the CSV against the grid and the regions it
was built from, so they need the template NetCDF and the shapefile - typically
only available where the weights were computed.

Use from the shell::

    validate_weights weights.csv --netcdf grid.nc --shapefile regions.gpkg --strict

or from Python::

    report = validate_weights("weights.csv")
    print(report.summary())
    report.raise_if_failed()
"""

import argparse
import os
import sys
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from netcdf2region.cli import add_grid_options, add_option, add_region_options, setup_logging
from netcdf2region.load_grid import COORD_DECIMALS, load_template_grid
from netcdf2region.load_regions import load_regions

#: Columns every weights CSV must have, in this order.
REQUIRED_COLUMNS = ["lat", "lon", "region_id", "percent_overlap"]
#: Written by `compute_grid_weights --method exact`; makes `aggregate` area-weight means.
OPTIONAL_COLUMNS = ["cell_area_km2"]

#: Weights are rounded to 2 decimals when sampled and 6 when rescued via exact
#: geometry, so anything above 1 by more than a rounding step is a real error.
WEIGHT_TOLERANCE = 1e-9

ERROR = "error"
WARNING = "warning"
INFO = "info"

_SEVERITY_ORDER = {ERROR: 0, WARNING: 1, INFO: 2}


@dataclass
class Check:
    """The outcome of a single validation rule."""

    name: str
    passed: bool
    severity: str
    message: str
    details: dict = field(default_factory=dict)

    @property
    def skipped(self):
        """True when the check could not run for lack of an optional input."""
        return self.details.get("skipped", False)


@dataclass
class Report:
    """The outcome of a full validation run."""

    checks: list = field(default_factory=list)
    source: str = "<dataframe>"

    @property
    def ok(self):
        """True when no error-severity check failed."""
        return not any(
            c.severity == ERROR and not c.passed and not c.skipped for c in self.checks
        )

    @property
    def failures(self):
        """Error-severity checks that failed."""
        return [
            c for c in self.checks
            if c.severity == ERROR and not c.passed and not c.skipped
        ]

    @property
    def warnings(self):
        """Warning-severity checks that failed."""
        return [
            c for c in self.checks
            if c.severity == WARNING and not c.passed and not c.skipped
        ]

    def summary(self):
        """Render the report as a human-readable block of text."""
        symbols = {True: "PASS", False: "FAIL"}
        lines = [f"Weights validation: {self.source}", "-" * 72]
        for check in self.checks:
            if check.skipped:
                status = "SKIP"
            elif check.severity == INFO:
                status = "INFO"
            elif check.passed:
                status = "PASS"
            else:
                status = symbols[False] if check.severity == ERROR else "WARN"
            lines.append(f"[{status}] {check.name}: {check.message}")
        lines.append("-" * 72)
        n_err, n_warn = len(self.failures), len(self.warnings)
        lines.append(
            f"{'OK' if self.ok else 'FAILED'} - {n_err} error(s), {n_warn} warning(s)"
        )
        return "\n".join(lines)

    def raise_if_failed(self):
        """Raise ``ValueError`` if any error-severity check failed."""
        if self.ok:
            return
        detail = "\n".join(f"  - {c.name}: {c.message}" for c in self.failures)
        raise ValueError(
            f"Weights validation failed for {self.source}:\n{detail}"
        )


def _worst_offenders(series, n=5):
    """Return the ``n`` largest entries of a Series as a plain dict."""
    return {str(k): float(v) for k, v in series.nlargest(n).items()}


def _check_schema(df):
    """Check 1: the CSV has exactly the expected columns with usable dtypes."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in REQUIRED_COLUMNS + OPTIONAL_COLUMNS]
    if missing:
        return Check(
            "schema", False, ERROR,
            f"missing column(s): {missing}; found {list(df.columns)}",
            {"missing": missing, "extra": extra},
        )

    # An empty frame has object-dtype columns; emptiness is reported separately so
    # it is not misdiagnosed as a dtype problem.
    bad_dtype = []
    if not df.empty:
        for col in ["lat", "lon", "percent_overlap"]:
            if not pd.api.types.is_numeric_dtype(df[col]):
                bad_dtype.append(col)
    if bad_dtype:
        return Check(
            "schema", False, ERROR,
            f"non-numeric column(s): {bad_dtype}",
            {"bad_dtype": bad_dtype},
        )

    msg = f"{len(df)} rows, columns {REQUIRED_COLUMNS}"
    if extra:
        msg += f" (plus unexpected {extra})"
    return Check("schema", not extra, ERROR if extra else INFO, msg, {"extra": extra})


def _check_no_missing_values(df):
    """Check 2: no NaN anywhere, no blank region ids."""
    na_counts = {c: int(df[c].isna().sum()) for c in REQUIRED_COLUMNS}
    blank = int((df["region_id"].astype(str).str.strip() == "").sum())
    total = sum(na_counts.values()) + blank
    if total:
        return Check(
            "no_missing_values", False, ERROR,
            f"{total} missing value(s): NaN per column {na_counts}, {blank} blank region_id",
            {"na_counts": na_counts, "blank_region_ids": blank},
        )
    return Check("no_missing_values", True, ERROR, "no NaN or blank region ids")


def _check_weight_range(df):
    """Check 3: every weight lies in (0, 1]."""
    non_positive = df[df["percent_overlap"] <= 0]
    too_large = df[df["percent_overlap"] > 1 + WEIGHT_TOLERANCE]
    n_bad = len(non_positive) + len(too_large)
    if n_bad:
        return Check(
            "weight_range", False, ERROR,
            f"{len(non_positive)} weight(s) <= 0 and {len(too_large)} weight(s) > 1 "
            f"(range {df['percent_overlap'].min()} .. {df['percent_overlap'].max()})",
            {
                "n_non_positive": len(non_positive),
                "n_too_large": len(too_large),
                "min": float(df["percent_overlap"].min()),
                "max": float(df["percent_overlap"].max()),
            },
        )
    return Check(
        "weight_range", True, ERROR,
        f"all weights in (0, 1] "
        f"(min {df['percent_overlap'].min():.6g}, max {df['percent_overlap'].max():.6g})",
    )


def _check_no_duplicates(df):
    """Check 4: a (cell, region) pair appears at most once."""
    dup_mask = df.duplicated(["lat", "lon", "region_id"], keep=False)
    n_dup = int(dup_mask.sum())
    if n_dup:
        sample = df[dup_mask].head(5).to_dict("records")
        return Check(
            "no_duplicates", False, ERROR,
            f"{n_dup} duplicated (lat, lon, region_id) row(s)",
            {"n_duplicate_rows": n_dup, "sample": sample},
        )
    return Check("no_duplicates", True, ERROR, "no duplicated (lat, lon, region_id) rows")


def _check_regions_represented(df):
    """Check 5: every region present in the file carries positive weight.

    Rows with a weight of zero are dropped by ``compute_weights``, so a region
    whose every overlap rounded away would vanish entirely. This catches the case
    where such rows were written anyway.
    """
    per_region = df.groupby("region_id")["percent_overlap"].sum()
    empty = per_region[per_region <= 0]
    if len(empty):
        return Check(
            "regions_represented", False, ERROR,
            f"{len(empty)} region(s) have a total weight of zero: {list(empty.index[:5])}",
            {"regions": list(empty.index)},
        )
    return Check(
        "regions_represented", True, ERROR,
        f"all {len(per_region)} regions carry positive weight "
        f"(smallest total {per_region.min():.6g})",
    )


def _check_cell_sums(df, max_cell_sum):
    """Check 6: no grid cell is over-subscribed beyond ``max_cell_sum``.

    Deliberately a warning. Overlapping source polygons make sums above 1
    legitimate - the shipped DOSE weights reach exactly 2.0 where Western Sahara
    (ESH.1_1) and Morocco (MAR.26_1) both claim the same cell in full. Rounding
    each weight to 2 decimals also lets a fully-covered cell land slightly above 1.
    """
    sums = df.groupby(["lat", "lon"])["percent_overlap"].sum()
    over = sums[sums > max_cell_sum]
    if len(over):
        return Check(
            "cell_sums", False, WARNING,
            f"{len(over)} of {len(sums)} cell(s) sum above {max_cell_sum} "
            f"(worst {sums.max():.4f}); this is expected when source polygons overlap",
            {
                "n_over": len(over),
                "n_cells": len(sums),
                "max_sum": float(sums.max()),
                "worst_cells": _worst_offenders(over),
            },
        )
    return Check(
        "cell_sums", True, WARNING,
        f"all {len(sums)} cell sums <= {max_cell_sum} (max {sums.max():.4f})",
    )


def _check_coords_on_grid(df, grid):
    """Check 7: every coordinate in the file is an actual grid coordinate.

    ``aggregate`` resolves weights to grid indices by equality after rounding both
    sides to ``COORD_DECIMALS`` (``np.where(lats == lat)``), which is what this check
    reproduces. A coordinate that is off by more than a rounding step is therefore not
    a cosmetic problem - it makes aggregation fail outright.
    """
    if grid is None:
        return Check(
            "coords_on_grid", True, ERROR,
            "skipped (no template grid supplied)", {"skipped": True},
        )

    grid_lats = set(np.round(grid.lat, COORD_DECIMALS).tolist())
    grid_lons = set(np.round(grid.lon, COORD_DECIMALS).tolist())

    file_lats = np.round(df["lat"].unique(), COORD_DECIMALS)
    file_lons = np.round(df["lon"].unique(), COORD_DECIMALS)

    bad_lats = sorted(set(file_lats.tolist()) - grid_lats)
    bad_lons = sorted(set(file_lons.tolist()) - grid_lons)

    if bad_lats or bad_lons:
        return Check(
            "coords_on_grid", False, ERROR,
            f"{len(bad_lats)} latitude(s) and {len(bad_lons)} longitude(s) are not on "
            f"the template grid, e.g. lat {bad_lats[:3]}, lon {bad_lons[:3]}",
            {"bad_lats": bad_lats[:50], "bad_lons": bad_lons[:50]},
        )
    return Check(
        "coords_on_grid", True, ERROR,
        f"all {len(file_lats)} latitudes and {len(file_lons)} longitudes lie on the grid",
    )


def _check_region_coverage(df, regions):
    """Check 8: the CSV covers exactly the regions in the shapefile.

    A region missing here is the failure ``assign_missing_regions`` exists to
    prevent: regions smaller than the point spacing never catch a sample point and
    would silently disappear from the weights.
    """
    if regions is None:
        return Check(
            "region_coverage", True, ERROR,
            "skipped (no shapefile supplied)", {"skipped": True},
        )

    shapefile_ids = set(regions["region_id"].dropna().astype(str))
    csv_ids = set(df["region_id"].astype(str))

    missing = sorted(shapefile_ids - csv_ids)
    unexpected = sorted(csv_ids - shapefile_ids)

    if missing or unexpected:
        return Check(
            "region_coverage", False, ERROR,
            f"{len(missing)} region(s) from the shapefile are absent from the weights "
            f"(e.g. {missing[:5]}) and {len(unexpected)} region(s) in the weights are "
            f"not in the shapefile (e.g. {unexpected[:5]})",
            {
                "missing": missing[:200],
                "unexpected": unexpected[:200],
                "n_missing": len(missing),
                "n_unexpected": len(unexpected),
            },
        )
    return Check(
        "region_coverage", True, ERROR,
        f"all {len(shapefile_ids)} shapefile regions are represented",
    )


def _cell_area_km2(lat, lat_res, lon_res):
    """Approximate the area of a grid cell centred at ``lat`` in square km."""
    earth_radius_km = 6371.0088
    deg = np.pi / 180.0
    lat_north = np.minimum(lat + lat_res / 2.0, 90.0)
    lat_south = np.maximum(lat - lat_res / 2.0, -90.0)
    return (
        earth_radius_km ** 2
        * lon_res * deg
        * np.abs(np.sin(lat_north * deg) - np.sin(lat_south * deg))
    )


def _check_region_areas(df, regions, grid, area_rtol):
    """Check 9: weights reconstruct each region's true area, where that is testable.

    ``sum(weight * cell_area)`` over a region's cells should approach the region's
    own area. Point sampling and 2-decimal rounding make this approximate, so it
    is a warning that reports the worst offenders rather than a hard failure.

    Regions whose area the weights cannot express - above all anything small enough
    to sit inside a single grid cell - are skipped and reported instead of counted;
    see ``skip_reasons`` below. One bias remains invisible here: a region stored
    unsplit across +/-180 has a meaningless bounding box.
    """
    if regions is None or grid is None:
        return Check(
            "region_areas", True, WARNING,
            "skipped (needs both a shapefile and a template grid)", {"skipped": True},
        )

    lat_res = grid.lat_res
    lon_res = grid.lon_res

    weighted = df.copy()
    weighted["cell_area"] = _cell_area_km2(weighted["lat"].values, lat_res, lon_res)
    weighted["area"] = weighted["percent_overlap"] * weighted["cell_area"]
    # Rescued rows carry 6 decimals, sampled ones 2: the resolution a region's share
    # of a cell is known to, and with it the smallest area error the file can express.
    weighted["quantum"] = np.where(
        weighted["percent_overlap"].round(2) == weighted["percent_overlap"], 0.01, 1e-6
    )
    # A cell reaching past the pole is clamped to it, but only the half inside the
    # globe can catch a sample point, so its region always looks too small.
    weighted["polar"] = weighted["lat"].abs() + lat_res / 2 > 90.0
    est = weighted.groupby("region_id").agg(
        area=("area", "sum"),
        quantum=("quantum", "max"),
        cell_area=("cell_area", "mean"),
        n_cells=("cell_area", "size"),
        polar=("polar", "any"),
    )
    # Smallest area error the file can express for a region: one rounding step of a
    # cell, over the cells it touches. The steps are independent, so they add in
    # quadrature - summing them would call every long, thin region unverifiable.
    est["resolution"] = est["quantum"] * est["cell_area"] * np.sqrt(est["n_cells"])

    # Equal-area projection (World Cylindrical Equal Area) for the reference area.
    dissolved = regions.dissolve(by="region_id")
    true_area = dissolved.to_crs("ESRI:54034").geometry.area / 1e6

    common = est.index.intersection(true_area.index)
    if len(common) == 0:
        return Check(
            "region_areas", True, WARNING,
            "skipped (no region ids in common)", {"skipped": True},
        )

    est = est.loc[common]
    ref = true_area.loc[common]
    bounds = dissolved.bounds.loc[common]

    # Regions whose reconstructed area is wrong for reasons that have nothing to do
    # with the weights. Their deviation is noise, so they are skipped and reported
    # rather than counted. The first reason wins where several apply.
    skip_reasons = {
        "invalid or zero-area geometry":
            ~dissolved.geometry.loc[common].is_valid.values | ~(ref > 0),
        "cell clamped at the pole": est["polar"],
        "extends beyond the template grid":
            (bounds["miny"] < grid.lat.min() - lat_res / 2)
            | (bounds["maxy"] > grid.lat.max() + lat_res / 2)
            | (bounds["minx"] < grid.lon.min() - lon_res / 2)
            | (bounds["maxx"] > grid.lon.max() + lon_res / 2),
        # A region small enough to sit inside one cell is described by a weight
        # rounded to 2 decimals, so its area comes in steps of a whole percent of
        # that cell: for a region covering a fraction of a percent of it the
        # deviation is hundreds of percent and measures rounding, not the weights.
        "below the weight resolution": est["resolution"] > area_rtol * ref,
    }
    skipped = pd.Series(None, index=common, dtype=object)
    for reason, mask in skip_reasons.items():
        skipped = skipped.mask(skipped.isna() & np.asarray(mask), reason)

    counts = skipped.value_counts()
    note = ""
    details = {"n_skipped": int(skipped.notna().sum())}
    if len(counts):
        note = (
            f"; {details['n_skipped']} region(s) skipped as not verifiable ("
            + ", ".join(f"{n} {reason}" for reason, n in counts.items()) + ")"
        )
        details["skipped_by_reason"] = {r: int(n) for r, n in counts.items()}

    keep = skipped.isna()
    if not keep.any():
        return Check(
            "region_areas", True, WARNING,
            f"skipped (no region's area is verifiable){note}",
            {"skipped": True, **details},
        )

    rel_err = ((est["area"][keep] - ref[keep]) / ref[keep]).abs()
    over = rel_err[rel_err > area_rtol]
    details.update({
        "n_over": len(over),
        "n_regions": len(rel_err),
        "max_rel_err": float(rel_err.max()),
    })
    if len(over):
        details["worst_regions"] = _worst_offenders(over)
        return Check(
            "region_areas", False, WARNING,
            f"{len(over)} of {len(rel_err)} verifiable region(s) deviate from their "
            f"true area by more than {area_rtol:.0%} (worst {rel_err.max():.1%}){note}",
            details,
        )
    return Check(
        "region_areas", True, WARNING,
        f"all {len(rel_err)} verifiable region areas within {area_rtol:.0%} "
        f"(worst {rel_err.max():.1%}){note}",
        details,
    )


def _check_statistics(df):
    """Check 10: descriptive statistics, always reported, never failing."""
    weights = df["percent_overlap"]
    # Rescued rows are written at 6 decimals; sampled rows at 2.
    rescued = int((weights.round(2) != weights).sum())
    n_cells = len(df.groupby(["lat", "lon"]))
    histogram, _ = np.histogram(weights, bins=[0, 0.01, 0.1, 0.25, 0.5, 0.75, 1.0001])
    details = {
        "n_rows": len(df),
        "n_regions": int(df["region_id"].nunique()),
        "n_cells": n_cells,
        "min_weight": float(weights.min()),
        "max_weight": float(weights.max()),
        "mean_weight": float(weights.mean()),
        "n_rescued_rows": rescued,
        "histogram": [int(h) for h in histogram],
    }
    return Check(
        "statistics", True, INFO,
        f"{len(df)} rows, {details['n_regions']} regions, {n_cells} cells, "
        f"weights {details['min_weight']:.6g}..{details['max_weight']:.6g}, "
        f"{rescued} rescued row(s)",
        details,
    )


def validate_weights(
    weights,
    grid=None,
    regions=None,
    max_cell_sum=1.01,
    area_rtol=0.05,
):
    """Run every applicable check against a weights table.

    Args:
        weights: Path to a weights CSV, or a DataFrame with the weights columns.
        grid: Optional template :class:`~netcdf2region.load_grid.Grid`, e.g. from
            :func:`~netcdf2region.load_grid.load_template_grid`.
            Enables the on-grid coordinate check.
        regions: Optional GeoDataFrame (with a ``region_id`` column) or path to a
            shapefile. Enables the region coverage and area checks.
        max_cell_sum: Ceiling for the per-cell weight sum warning. Values above 1
            are legitimate when the source polygons overlap.
        area_rtol: Relative tolerance for the region area warning.

    Returns:
        Report: The outcome of every check, in severity-relevant order.
    """
    if isinstance(weights, pd.DataFrame):
        df = weights.copy()
        source = "<dataframe>"
    else:
        source = str(weights)
        if not os.path.exists(source):
            raise FileNotFoundError(f"Weights file not found: {source}")
        df = pd.read_csv(source)

    if isinstance(regions, str):
        regions = load_regions(
            regions, simplify=False, shapefile_layer=None, default_gid="GID_1"
        )

    report = Report(source=source)

    schema = _check_schema(df)
    report.checks.append(schema)
    if not schema.passed and schema.severity == ERROR:
        # Every later check indexes the required columns; bail out cleanly.
        return report

    if df.empty:
        report.checks.append(
            Check("non_empty", False, ERROR, "weights file contains no rows")
        )
        return report

    report.checks.append(_check_no_missing_values(df))
    report.checks.append(_check_weight_range(df))
    report.checks.append(_check_no_duplicates(df))
    report.checks.append(_check_regions_represented(df))
    report.checks.append(_check_cell_sums(df, max_cell_sum))
    report.checks.append(_check_coords_on_grid(df, grid))
    report.checks.append(_check_region_coverage(df, regions))
    report.checks.append(_check_region_areas(df, regions, grid, area_rtol))
    report.checks.append(_check_statistics(df))

    return report


def main():
    """Validate a weights CSV from the command line."""
    parser = argparse.ArgumentParser(
        description="Validate a grid weights CSV produced by compute_grid_weights."
    )
    parser.add_argument("weights", help="Path to the weights CSV to validate")
    parser.add_argument(
        "--netcdf", default=None,
        help="Template NetCDF the weights were computed for (enables the on-grid check)",
    )
    parser.add_argument(
        "--shapefile", default=None,
        help="Shapefile the weights were computed from (enables the coverage/area checks)",
    )
    add_region_options(parser)
    add_grid_options(parser)
    add_option(
        parser, "--max-cell-sum", type=float, default=1.01,
        help="Warn when a cell's weights sum above this (default: 1.01)",
    )
    add_option(
        parser, "--area-rtol", type=float, default=0.05,
        help="Relative tolerance for the region area check (default: 0.05)",
    )
    add_option(
        parser, "--strict", action="store_true",
        help="Exit non-zero when any check fails, including warnings",
    )
    args = parser.parse_args()
    setup_logging()

    grid = None
    if args.netcdf:
        grid = load_template_grid(
            args.netcdf,
            coordinate_origin=args.coordinate_origin,
            shift_longitudes=args.shift_longitudes,
        )

    regions = None
    if args.shapefile:
        regions = load_regions(
            args.shapefile,
            simplify=False,
            shapefile_layer=args.shapefile_layer,
            default_gid=args.gid,
        )

    report = validate_weights(
        args.weights,
        grid=grid,
        regions=regions,
        max_cell_sum=args.max_cell_sum,
        area_rtol=args.area_rtol,
    )
    print(report.summary())

    if not report.ok:
        return 1
    if args.strict and report.warnings:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
