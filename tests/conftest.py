"""Shared fixtures for the netcdf2region test suite."""

import json
import os

import geopandas as gpd
import pytest

from factories import (
    MINI_POINTS_PER_CELL,
    make_regions,
    mini_coordinates,
    mini_polygons,
)

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHIPPED_DIR = os.path.join(REPO_ROOT, "netcdf2region", "data")

#: Presence of this directory is what distinguishes the PIK cluster from a laptop.
CLUSTER_MARKER_PATH = "/p/projects/impactee"

#: Real inputs the cluster tests run against, mirroring precalc_weights/*.sh.
CLUSTER_PATHS = {
    "era5": "/p/projects/climate_data_central/reanalysis/ERA5/2m_temperature/2m_temperature_200010.nc",
    "isimip": "/p/projects/isimip/isimip/ISIMIP2b/InputData/gdp/rcp26soc/gdp_rcp26soc_0p5deg_annual_2006-2099.nc4",
    "dose": "/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/DOSE_REGIONS.gpkg",
    "gadm": "/p/tmp/lochner/gadm36_levels.gpkg",
}


def pytest_collection_modifyitems(config, items):
    """Skip cluster-marked tests when the PIK data directories are not mounted."""
    if os.path.isdir(CLUSTER_MARKER_PATH):
        return
    skip = pytest.mark.skip(
        reason=f"cluster data not available ({CLUSTER_MARKER_PATH} not found)"
    )
    for item in items:
        if "cluster" in item.keywords:
            item.add_marker(skip)


# --------------------------------------------------------------------------- #
# Committed fixture files
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="session")
def data_dir():
    """Directory holding the committed test fixtures."""
    return DATA_DIR


@pytest.fixture(scope="session")
def mini_grid_nc():
    """Bare 6x8 template grid the golden weights were computed from."""
    return os.path.join(DATA_DIR, "mini_grid.nc")


@pytest.fixture(scope="session")
def mini_regions_geojson():
    """The five mini regions on disk."""
    return os.path.join(DATA_DIR, "mini_regions.geojson")


@pytest.fixture(scope="session")
def golden_weights_path():
    """Committed expected output of ``compute_overlap`` on the mini case."""
    return os.path.join(DATA_DIR, "golden_mini_weights.csv")


@pytest.fixture(scope="session")
def mini_tas_nc():
    """3-timestep field on the mini grid, values ``100*t + 10*i + j``."""
    return os.path.join(DATA_DIR, "mini_tas.nc")


@pytest.fixture(scope="session")
def mini_pop_nc():
    """2-D population-like field on the mini grid, containing NaN cells."""
    return os.path.join(DATA_DIR, "mini_pop.nc")


@pytest.fixture(scope="session")
def shipped_slice_csv():
    """Small slice of complete regions cut from a shipped weights file."""
    return os.path.join(DATA_DIR, "shipped_slice_ISIMIP_DOSE.csv")


@pytest.fixture(scope="session")
def shipped_stats():
    """Recorded summary statistics for every shipped weights file."""
    with open(os.path.join(DATA_DIR, "shipped_stats.json")) as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def shipped_weights_files():
    """Paths of all weights CSVs shipped as package data."""
    return sorted(
        os.path.join(SHIPPED_DIR, name)
        for name in os.listdir(SHIPPED_DIR)
        if name.endswith("_weights.csv")
    )


# --------------------------------------------------------------------------- #
# In-memory mini case
# --------------------------------------------------------------------------- #

@pytest.fixture
def mini_coords():
    """The mini grid as a ``load_template_grid``-style dict."""
    return mini_coordinates()


@pytest.fixture
def mini_regions_gdf():
    """The five mini regions as a GeoDataFrame with a ``region_id`` column."""
    gdf = make_regions(mini_polygons())
    return gdf.rename(columns={"GID_1": "region_id"})


@pytest.fixture(scope="session")
def mini_weights_csv(tmp_path_factory, mini_grid_nc, mini_regions_geojson):
    """Run the real pipeline on the mini case once and reuse the output.

    Distinct from ``golden_weights_path``: this is computed fresh during the test
    session, so comparing the two is what detects a behaviour change.
    """
    from netcdf2region.compute_grid_weights import compute_overlap

    out = tmp_path_factory.mktemp("mini") / "weights.csv"
    compute_overlap(
        netcdf_path=mini_grid_nc,
        shapefile_path=mini_regions_geojson,
        output_csv=str(out),
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
    return str(out)


@pytest.fixture(scope="session")
def cluster_paths():
    """Absolute paths of the real cluster inputs, as used by precalc_weights/*.sh."""
    return dict(CLUSTER_PATHS)


@pytest.fixture(scope="session")
def gadm_regions():
    """GADM admin1 regions from the cluster, loaded once per session."""
    from netcdf2region.compute_grid_weights import load_regions

    path = CLUSTER_PATHS["gadm"]
    if not os.path.exists(path):
        pytest.skip(f"GADM GeoPackage not available at {path}")
    return load_regions(
        path, simplify=False, shapefile_layer=1, default_gid="GID_1", yes=True
    )
