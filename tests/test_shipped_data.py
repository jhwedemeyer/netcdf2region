"""Invariants and regression guards for the weights shipped as package data.

These files are what ``--precalc-weights`` hands to users. They are computed by
``netcdf2region/precalc_weights/*.sh`` with ``--method exact``, so they carry a
``cell_area_km2`` column for ``--area-weighted``. The tests here make a regeneration
a deliberate, reviewable event rather than a silent change; ``test_cluster.py``
checks the files against the real grids and shapefiles.
"""

import os

import numpy as np
import pandas as pd
import pytest

from netcdf2region.weights import precalc_weights_path
from netcdf2region.validate_weights import validate_weights

SHIPPED_NAMES = [
    "ERA5_to_DOSE_admin1_weights.csv",
    "ERA5_to_GADM36_admin0_weights.csv",
    "ERA5_to_GADM36_admin1_weights.csv",
    "ISIMIP_to_DOSE_admin1_weights.csv",
    "ISIMIP_to_GADM36_admin0_weights.csv",
    "ISIMIP_to_GADM36_admin1_weights.csv",
]

#: Grid resolution each family of files was computed on.
EXPECTED_RESOLUTION = {"ERA5": 0.25, "ISIMIP": 0.5}


@pytest.fixture(scope="module")
def shipped(request):
    """Cache each shipped CSV once per module rather than per test."""
    cache = {}

    def load(name):
        if name not in cache:
            cache[name] = pd.read_csv(str(precalc_weights_path(name)))
        return cache[name]

    return load


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #

def test_every_expected_weights_file_is_shipped(shipped_weights_files):
    assert sorted(os.path.basename(p) for p in shipped_weights_files) == sorted(SHIPPED_NAMES)


@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_each_file_is_reachable_through_precalc_weights_path(name):
    assert os.path.exists(str(precalc_weights_path(name)))


# --------------------------------------------------------------------------- #
# Invariants
# --------------------------------------------------------------------------- #

@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_shipped_file_passes_validation(name, shipped):
    """Only the file-local checks; the grid and shapefile live on the cluster."""
    report = validate_weights(shipped(name), max_cell_sum=2.0)

    assert report.ok, report.summary()


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_coordinates_sit_on_a_regular_grid(name, shipped):
    """A ragged coordinate set would break ``aggregate``'s exact-equality lookup."""
    df = shipped(name)
    resolution = EXPECTED_RESOLUTION["ERA5" if name.startswith("ERA5") else "ISIMIP"]

    for axis in ["lat", "lon"]:
        spacings = np.diff(np.sort(df[axis].unique()))
        # Coordinates are absent where no region falls, so only check that every
        # gap is a whole multiple of the resolution.
        multiples = spacings / resolution
        assert np.allclose(multiples, np.round(multiples)), f"{name}: ragged {axis}"


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_coordinates_are_within_the_earth(name, shipped):
    df = shipped(name)

    assert df["lat"].between(-90, 90).all()
    assert df["lon"].between(-180, 180).all()


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_admin1_ids_look_like_gadm_ids(name, shipped):
    if "admin1" not in name:
        pytest.skip("admin0 files carry plain country codes")
    ids = shipped(name)["region_id"]

    assert ids.str.contains(r"^[A-Z]{3}\.\d+_\d+$", regex=True).mean() > 0.95


@pytest.mark.slow
def test_admin0_ids_are_three_letter_country_codes(shipped):
    ids = shipped("ISIMIP_to_GADM36_admin0_weights.csv")["region_id"]

    assert ids.str.fullmatch(r"[A-Z]{3}").all()


# --------------------------------------------------------------------------- #
# Regression guards
# --------------------------------------------------------------------------- #

def test_the_committed_slice_still_matches_the_shipped_file(shipped_slice_csv):
    """A few-KB slice stands in for the 2.5 MB file it was cut from.

    If this fails, the shipped weights were regenerated: check the diff is what
    you intended, then rerun ``python tests/data/make_fixtures.py``.
    """
    committed = pd.read_csv(shipped_slice_csv)
    live = pd.read_csv(str(precalc_weights_path("ISIMIP_to_DOSE_admin1_weights.csv")))

    subset = (
        live[live["region_id"].isin(committed["region_id"].unique())]
        .sort_values(["region_id", "lat", "lon"])
        .reset_index(drop=True)
    )

    pd.testing.assert_frame_equal(committed, subset)


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_summary_statistics_are_unchanged(name, shipped, shipped_stats):
    """Turn a regeneration into a readable diff instead of an opaque failure."""
    df = shipped(name)
    cell_sums = df.groupby(["lat", "lon"])["percent_overlap"].sum()
    actual = {
        "n_rows": int(len(df)),
        "n_regions": int(df["region_id"].nunique()),
        "n_cells": int(len(cell_sums)),
        "min_weight": float(df["percent_overlap"].min()),
        "max_weight": float(df["percent_overlap"].max()),
        "max_cell_sum": float(cell_sums.max()),
        "n_cells_over_one": int((cell_sums > 1.01).sum()),
    }

    assert actual == shipped_stats[name], (
        f"{name} changed. If this was an intended regeneration, rerun "
        f"`python tests/data/make_fixtures.py` and review the diff."
    )


@pytest.mark.slow
def test_dose_weights_contain_the_known_overlapping_cell(shipped):
    """Guards the assumption behind the soft cell-sum ceiling.

    Western Sahara and Morocco both claim (24.75, -14.25) in full. If this ever
    stops being true, the cell-sum check could be tightened to a hard invariant.
    """
    df = shipped("ISIMIP_to_DOSE_admin1_weights.csv")
    cell = df[(df["lat"] == 24.75) & (df["lon"] == -14.25)]

    assert set(cell["region_id"]) == {"ESH.1_1", "MAR.26_1"}
    assert cell["percent_overlap"].sum() == pytest.approx(2.0)


@pytest.mark.slow
def test_gadm_weights_do_not_over_subscribe_cells(shipped):
    """GADM regions do not overlap, so their cell sums must stay at or below 1."""
    for name in SHIPPED_NAMES:
        if "GADM" not in name:
            continue
        cell_sums = shipped(name).groupby(["lat", "lon"])["percent_overlap"].sum()
        assert cell_sums.max() <= 1.02, f"{name} sums to {cell_sums.max()}"


@pytest.mark.slow
@pytest.mark.parametrize("name", SHIPPED_NAMES)
def test_shipped_files_are_exact_weights(name, shipped):
    """The shipped files must support ``--area-weighted``, i.e. come from ``--method exact``."""
    df = shipped(name)

    assert "cell_area_km2" in df.columns
    assert (df["cell_area_km2"] > 0).all()
    # Sampled weights are multiples of 1/100; exact ones are not.
    assert (df["percent_overlap"].round(2) != df["percent_overlap"]).any()
