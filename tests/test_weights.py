"""Tests for ``weights``: preparing a weights table for aggregation."""

import os

import numpy as np
import pandas as pd
import pytest

from factories import make_weights_df
from netcdf2region.weights import correct_weights, precalc_weights_path, validate_secondary_weights


# --------------------------------------------------------------------------- #
# correct_weights
# --------------------------------------------------------------------------- #

def test_correct_weights_rescales_a_partly_covered_cell():
    """A cell 50% covered by land must have its weights scaled up to sum to 1."""
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.3),
        (0.5, 0.5, "R2", 0.2),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(sorted(corrected["percent_overlap"]), [0.4, 0.6])
    assert corrected["percent_overlap"].sum() == pytest.approx(1.0)


def test_correct_weights_leaves_a_fully_covered_cell_alone():
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.75),
        (0.5, 0.5, "R2", 0.25),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [0.75, 0.25])


def test_correct_weights_scales_overlapping_regions_down():
    """Overlapping source polygons make cell sums exceed 1, and this divides by them.

    The shipped DOSE weights contain cells summing to exactly 2.0 where two regions
    both claim the cell in full. Correcting such a file halves both weights, which
    is a real consequence of using --correct-weights with overlapping regions.
    """
    weights = make_weights_df([
        (0.5, 0.5, "ESH.1_1", 1.0),
        (0.5, 0.5, "MAR.26_1", 1.0),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [0.5, 0.5])


def test_correct_weights_treats_cells_independently():
    weights = make_weights_df([
        (0.5, 0.5, "R1", 0.5),
        (1.5, 1.5, "R2", 0.25),
    ])

    corrected = correct_weights(weights)

    assert np.allclose(corrected["percent_overlap"], [1.0, 1.0])


def test_correct_weights_drops_its_helper_column():
    corrected = correct_weights(make_weights_df([(0.5, 0.5, "R", 0.5)]))
    assert list(corrected.columns) == ["lat", "lon", "region_id", "percent_overlap"]


# --------------------------------------------------------------------------- #
# precalc_weights_path
# --------------------------------------------------------------------------- #

def test_precalc_weights_path_resolves_shipped_data():
    path = precalc_weights_path("ISIMIP_to_DOSE_admin1_weights.csv")

    assert os.path.exists(str(path))
    assert list(pd.read_csv(path).columns) == [
        "lat", "lon", "region_id", "percent_overlap", "cell_area_km2"
    ]


def test_precalc_weights_path_reports_unknown_names():
    with pytest.raises(ValueError, match="not found in package data"):
        precalc_weights_path("no_such_weights.csv")


# --------------------------------------------------------------------------- #
# validate_secondary_weights
# --------------------------------------------------------------------------- #

def test_validate_secondary_weights_does_not_alter_the_array():
    """The validator only checks; NaN handling is load_data_grid's decision now."""
    data = np.array([[1.0, np.nan], [3.0, 4.0]])

    returned = validate_secondary_weights(
        lats=np.array([0.0, 1.0]), lons=np.array([0.0, 1.0]),
        secondary_data=data, secondary_lats=np.array([0.0, 1.0]),
        secondary_lons=np.array([0.0, 1.0]), secondary_weights_var="pop",
    )

    assert returned is None
    assert np.isnan(data[0, 1])
