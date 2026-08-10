#!/usr/bin/env python3
"""Regenerate (or verify) the committed test fixtures in this directory.

Every fixture here is derived deterministically from code in ``tests/factories.py``
or from the shipped weights in ``netcdf2region/data/``. Committing them rather than
building them at test time is what makes ``golden_mini_weights.csv`` a regression
guard: if a change to ``compute_grid_weights.py`` alters the numbers, the test
fails instead of quietly regenerating a new "expected" answer.

Usage::

    python tests/data/make_fixtures.py           # write the fixtures
    python tests/data/make_fixtures.py --check   # verify they are up to date
"""

import argparse
import io
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS_DIR = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(TESTS_DIR)
SHIPPED_DIR = os.path.join(REPO_ROOT, "netcdf2region", "data")

sys.path.insert(0, TESTS_DIR)
sys.path.insert(0, REPO_ROOT)

from factories import (  # noqa: E402
    MINI_LATS,
    MINI_LONS,
    MINI_POINTS_PER_CELL,
    make_grid_nc,
    make_regions,
    mini_polygons,
    write_regions,
)
from netcdf2region.compute_grid_weights import compute_overlap  # noqa: E402

MINI_GRID_NC = os.path.join(HERE, "mini_grid.nc")
MINI_REGIONS = os.path.join(HERE, "mini_regions.geojson")
GOLDEN_WEIGHTS = os.path.join(HERE, "golden_mini_weights.csv")
MINI_TAS_NC = os.path.join(HERE, "mini_tas.nc")
MINI_POP_NC = os.path.join(HERE, "mini_pop.nc")
SHIPPED_SLICE = os.path.join(HERE, "shipped_slice_ISIMIP_DOSE.csv")
SHIPPED_STATS = os.path.join(HERE, "shipped_stats.json")

#: The shipped file the slice fixture is cut from.
SLICE_SOURCE = "ISIMIP_to_DOSE_admin1_weights.csv"

#: Cells whose secondary weight is NaN, to exercise the NaN handling in
#: ``validate_secondary_weights``. (0, 0) is the sole cell of region A, so a
#: mishandled NaN turns A's aggregate into NaN and nothing else can mask it.
POP_NAN_CELLS = [(0, 0), (3, 5)]


def build_mini_grid():
    """Write the bare 6x8 template grid used to compute the golden weights."""
    make_grid_nc(MINI_GRID_NC, n_time=1, var="template")
    return MINI_GRID_NC


def build_mini_regions():
    """Write the five mini regions as GeoJSON."""
    gdf = make_regions(mini_polygons())
    write_regions(gdf, MINI_REGIONS)
    return MINI_REGIONS


def build_golden_weights():
    """Run the real pipeline on the mini case and record its output.

    ``point_distribution="uniform"`` makes this fully deterministic, so the result
    can be compared byte for byte.
    """
    compute_overlap(
        netcdf_path=MINI_GRID_NC,
        shapefile_path=MINI_REGIONS,
        output_csv=GOLDEN_WEIGHTS,
        points_per_cell=MINI_POINTS_PER_CELL,
        simplify=False,
        mask_nc_path=None,
        shapefile_layer=None,
        point_distribution="uniform",
        backend="threading",
        n_jobs=1,
        yes=True,
        coordinate_origin="centered",
        default_gid="GID_1",
        shift_longitudes=False,
    )
    return GOLDEN_WEIGHTS


def build_mini_tas():
    """Write a 3-timestep temperature-like field on the mini grid.

    Values are ``100 * t + 10 * i + j`` so that any weighted combination of cells
    has an obvious closed form.
    """
    n_time = 3
    values = np.zeros((n_time, len(MINI_LATS), len(MINI_LONS)))
    for t in range(n_time):
        for i in range(len(MINI_LATS)):
            for j in range(len(MINI_LONS)):
                values[t, i, j] = 100 * t + 10 * i + j
    make_grid_nc(MINI_TAS_NC, var="tas", n_time=n_time, values=values)
    return MINI_TAS_NC


def build_mini_pop():
    """Write a 2-D population-like field on the mini grid, including NaNs."""
    values = np.full((len(MINI_LATS), len(MINI_LONS)), 10.0)
    # A pronounced hotspot lets a population-weighted mean be told apart from an
    # area-weighted one at a glance.
    values[2, 4] = 1000.0
    for i, j in POP_NAN_CELLS:
        values[i, j] = np.nan
    make_grid_nc(MINI_POP_NC, var="pop", n_time=0, values=values)
    return MINI_POP_NC


def build_shipped_slice():
    """Cut a small, complete-region slice out of a shipped weights file.

    Whole regions are kept rather than an arbitrary row range, so the slice is
    internally coherent and can drive a real aggregation test.
    """
    src = os.path.join(SHIPPED_DIR, SLICE_SOURCE)
    df = pd.read_csv(src)
    counts = df.groupby("region_id").size()
    # Deterministic pick: alphabetically first regions of a manageable size.
    candidates = sorted(counts[(counts >= 3) & (counts <= 20)].index)[:8]
    sliced = (
        df[df["region_id"].isin(candidates)]
        .sort_values(["region_id", "lat", "lon"])
        .reset_index(drop=True)
    )
    sliced.to_csv(SHIPPED_SLICE, index=False)
    return SHIPPED_SLICE


def _stats_for(path):
    """Summarise a weights CSV into the numbers worth regression-testing."""
    df = pd.read_csv(path)
    cell_sums = df.groupby(["lat", "lon"])["percent_overlap"].sum()
    return {
        "n_rows": int(len(df)),
        "n_regions": int(df["region_id"].nunique()),
        "n_cells": int(len(cell_sums)),
        "min_weight": float(df["percent_overlap"].min()),
        "max_weight": float(df["percent_overlap"].max()),
        "max_cell_sum": float(cell_sums.max()),
        "n_cells_over_one": int((cell_sums > 1.01).sum()),
        "n_rescued_rows": int((df["percent_overlap"].round(2) != df["percent_overlap"]).sum()),
    }


def build_shipped_stats():
    """Record summary statistics for every shipped weights file."""
    stats = {}
    for name in sorted(os.listdir(SHIPPED_DIR)):
        if name.endswith("_weights.csv"):
            stats[name] = _stats_for(os.path.join(SHIPPED_DIR, name))
    with open(SHIPPED_STATS, "w") as fh:
        json.dump(stats, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return SHIPPED_STATS


#: Fixtures that are byte-comparable in ``--check`` mode. NetCDF files are not:
#: the library stamps metadata that varies between writes.
_TEXT_FIXTURES = [GOLDEN_WEIGHTS, MINI_REGIONS, SHIPPED_SLICE, SHIPPED_STATS]


def build_all():
    """Build every fixture, in dependency order."""
    build_mini_grid()
    build_mini_regions()
    build_golden_weights()
    build_mini_tas()
    build_mini_pop()
    build_shipped_slice()
    build_shipped_stats()


def check_all():
    """Rebuild the text fixtures in memory and diff them against what is committed.

    Returns:
        int: 0 when everything matches, 1 otherwise.
    """
    before = {}
    for path in _TEXT_FIXTURES:
        if not os.path.exists(path):
            print(f"MISSING: {path}")
            return 1
        with open(path) as fh:
            before[path] = fh.read()

    build_all()

    failed = False
    for path in _TEXT_FIXTURES:
        with open(path) as fh:
            after = fh.read()
        if after != before[path]:
            print(f"CHANGED: {os.path.relpath(path, REPO_ROOT)}")
            failed = True
        else:
            print(f"ok:      {os.path.relpath(path, REPO_ROOT)}")

    if failed:
        print(
            "\nFixtures are out of date. If the change is intended, commit the "
            "regenerated files; otherwise investigate the behaviour change."
        )
        return 1
    print("\nAll fixtures reproduce byte-identically.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check", action="store_true",
        help="Verify the committed fixtures instead of overwriting them",
    )
    args = parser.parse_args()

    if args.check:
        return check_all()

    # Silence the pipeline's progress output; only the result matters here.
    stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        build_all()
    finally:
        sys.stdout = stdout
    print(f"Fixtures written to {HERE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
