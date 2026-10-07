"""Tests that need the real PIK inputs the precalculation scripts point at.

Skipped automatically off-cluster (see ``pytest_collection_modifyitems`` in
``conftest.py``). Run them after a precalculation job with::

    python -m pytest tests -m cluster -v

They are the acceptance gate for regenerated weights: the coverage check below
fails on the currently shipped files, because those predate
``assign_missing_regions``, and must pass on freshly computed ones.
"""

import os

import numpy as np
import pytest

from netcdf2region.load_grid import load_template_grid
from netcdf2region.load_regions import load_regions
from netcdf2region.compute_grid_weights import compute_weights
from netcdf2region.weights import precalc_weights_path
from netcdf2region.validate_weights import validate_weights

pytestmark = pytest.mark.cluster

#: Which template grid and shapefile each shipped weights file was built from,
#: mirroring netcdf2region/precalc_weights/*.sh.
WEIGHTS_PROVENANCE = {
    "ERA5_to_DOSE_admin1_weights.csv": ("era5", "dose", None, True),
    "ERA5_to_GADM36_admin0_weights.csv": ("era5", "gadm", 0, True),
    "ERA5_to_GADM36_admin1_weights.csv": ("era5", "gadm", 1, True),
    "ISIMIP_to_DOSE_admin1_weights.csv": ("isimip", "dose", None, False),
    "ISIMIP_to_GADM36_admin0_weights.csv": ("isimip", "gadm", 0, False),
    "ISIMIP_to_GADM36_admin1_weights.csv": ("isimip", "gadm", 1, False),
}


def _require(path):
    if not os.path.exists(path):
        pytest.skip(f"not available: {path}")
    return path


# --------------------------------------------------------------------------- #
# Reading the real inputs
# --------------------------------------------------------------------------- #

def test_era5_grid_is_quarter_degree_after_shifting(cluster_paths):
    coords = load_template_grid(
        _require(cluster_paths["era5"]), coordinate_origin="centered", shift_longitudes=True
    )

    assert np.isclose(coords.lat_res, 0.25)
    assert np.isclose(coords["lon_res"], 0.25)
    assert coords["lon"].min() >= -180
    assert coords["lon"].max() <= 180
    assert np.all(np.diff(coords["lon"]) > 0)


def test_era5_grid_is_rejected_without_the_shift_flag(cluster_paths):
    """Confirms the ERA5 scripts genuinely need --shift-longitudes."""
    with pytest.raises(ValueError, match="shift_longitudes"):
        load_template_grid(
            _require(cluster_paths["era5"]),
            coordinate_origin="centered", shift_longitudes=False,
        )


def test_isimip_grid_is_half_degree_without_shifting(cluster_paths):
    """Confirms the ISIMIP scripts are right to omit --shift-longitudes."""
    coords = load_template_grid(
        _require(cluster_paths["isimip"]), coordinate_origin="centered", shift_longitudes=False
    )

    assert np.isclose(coords.lat_res, 0.5)
    assert np.isclose(coords["lon_res"], 0.5)


def test_dose_shapefile_provides_gid_1(cluster_paths):
    """The DOSE scripts pass no --gid, so the default GID_1 must be present."""
    regions = load_regions(
        _require(cluster_paths["dose"]), simplify=False, shapefile_layer=None,
        default_gid="GID_1",
    )

    assert "region_id" in regions.columns
    assert regions.crs == "EPSG:4326"
    assert regions["region_id"].notna().all()
    assert len(regions) > 1000


@pytest.mark.parametrize("layer,expected_gid", [(0, "GID_0"), (1, "GID_1")])
def test_gadm_layers_load(cluster_paths, layer, expected_gid):
    regions = load_regions(
        _require(cluster_paths["gadm"]), simplify=False, shapefile_layer=layer,
        default_gid="GID_1",
    )

    assert "region_id" in regions.columns
    assert regions.crs == "EPSG:4326"
    assert regions["region_id"].nunique() == len(regions)


def test_dose_regions_genuinely_overlap(cluster_paths):
    """Justifies treating cell sums above 1 as a warning rather than an error."""
    regions = load_regions(
        _require(cluster_paths["dose"]), simplify=False, shapefile_layer=None,
        default_gid="GID_1",
    )
    esh = regions[regions["region_id"] == "ESH.1_1"]
    mar = regions[regions["region_id"] == "MAR.26_1"]
    if esh.empty or mar.empty:
        pytest.skip("expected region ids not present in this DOSE version")

    assert esh.geometry.iloc[0].intersects(mar.geometry.iloc[0])


# --------------------------------------------------------------------------- #
# Full validation of the shipped weights against their real provenance
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", sorted(WEIGHTS_PROVENANCE))
def test_shipped_weights_sit_on_their_template_grid(name, cluster_paths):
    """Every coordinate must exist in the grid, or ``aggregate`` cannot use it."""
    grid_key, _, _, shift = WEIGHTS_PROVENANCE[name]
    coords = load_template_grid(
        _require(cluster_paths[grid_key]), coordinate_origin="centered", shift_longitudes=shift
    )

    report = validate_weights(
        str(precalc_weights_path(name)), grid=coords, max_cell_sum=2.0
    )

    assert report.ok, report.summary()


@pytest.mark.parametrize("name", sorted(WEIGHTS_PROVENANCE))
def test_shipped_weights_cover_every_region(name, cluster_paths):
    """The acceptance gate for regenerated weights.

    Expected to FAIL on the files shipped today: they were produced before
    ``assign_missing_regions`` existed, so regions smaller than the sampling
    spacing are missing. After rerunning precalc_weights/*.sh this must pass.
    """
    _, shape_key, layer, _ = WEIGHTS_PROVENANCE[name]
    regions = load_regions(
        _require(cluster_paths[shape_key]), simplify=False, shapefile_layer=layer,
        default_gid="GID_1",
    )

    report = validate_weights(
        str(precalc_weights_path(name)), regions=regions, max_cell_sum=2.0
    )
    coverage = next(c for c in report.checks if c.name == "region_coverage")

    assert coverage.passed, (
        f"{name}: {coverage.message}\n"
        f"Rerun netcdf2region/precalc_weights/ to regenerate with the small-region fix."
    )


@pytest.mark.parametrize("name", sorted(WEIGHTS_PROVENANCE))
def test_shipped_weights_reconstruct_region_areas(name, cluster_paths):
    """Weighted cell areas should add up to each region's true area."""
    grid_key, shape_key, layer, shift = WEIGHTS_PROVENANCE[name]
    coords = load_template_grid(
        _require(cluster_paths[grid_key]), coordinate_origin="centered", shift_longitudes=shift
    )
    regions = load_regions(
        _require(cluster_paths[shape_key]), simplify=False, shapefile_layer=layer,
        default_gid="GID_1",
    )

    report = validate_weights(
        str(precalc_weights_path(name)),
        grid=coords, regions=regions, max_cell_sum=2.0, area_rtol=0.25,
    )
    areas = next(c for c in report.checks if c.name == "region_areas")

    # Small regions are dominated by rounding, so judge the bulk, not the worst case.
    fraction_bad = areas.details.get("n_over", 0) / max(areas.details.get("n_regions", 1), 1)
    assert fraction_bad < 0.05, report.summary()


# --------------------------------------------------------------------------- #
# A real end-to-end run, cropped to stay fast
# --------------------------------------------------------------------------- #

def test_end_to_end_on_real_data_cropped(tmp_path, cluster_paths):
    """Run the real pipeline on real geometry, cropped to a few hundred cells.

    Synthetic fixtures use clean rectangles; this exercises the messy parts of a
    genuine shapefile (multipart countries, self-touching rings, tiny islands).
    """
    import geopandas as gpd
    import xarray as xr

    from factories import make_grid_nc

    era5 = _require(cluster_paths["era5"])
    gadm = _require(cluster_paths["gadm"])

    # A 5x5 degree window over central Europe, on the real ERA5 0.25 degree grid.
    ds = xr.open_dataset(era5)
    lat_name = "lat" if "lat" in ds else "latitude"
    lon_name = "lon" if "lon" in ds else "longitude"
    lats = np.sort(ds[lat_name].values)
    lons = ((np.sort(ds[lon_name].values) + 180) % 360) - 180
    lons = np.sort(lons)
    ds.close()

    lats = lats[(lats >= 47.0) & (lats <= 52.0)]
    lons = lons[(lons >= 6.0) & (lons <= 11.0)]
    assert len(lats) > 5 and len(lons) > 5

    grid = make_grid_nc(tmp_path / "crop.nc", lats=lats, lons=lons)

    from shapely.geometry import box

    regions = load_regions(gadm, False, 1, "GID_1")
    window = regions[regions.intersects(box(6.0, 47.0, 11.0, 52.0))].copy()
    if window.empty:
        pytest.skip("no GADM regions in the crop window")
    shapefile = str(tmp_path / "window.gpkg")
    window.to_file(shapefile, driver="GPKG")

    out = tmp_path / "weights.csv"
    compute_weights(
        grid, shapefile, str(out), method="sampling",
        points_per_cell=25, n_jobs=4, gid="region_id",
    )

    coords = load_template_grid(grid, coordinate_origin="centered", shift_longitudes=False)
    report = validate_weights(str(out), grid=coords, max_cell_sum=2.0)

    assert report.ok, report.summary()
