"""Tests for ``load_regions``: reading region geometries from a shapefile."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from factories import make_regions, mini_polygons, write_regions
from netcdf2region.load_regions import load_regions


# --------------------------------------------------------------------------- #
# load_regions
# --------------------------------------------------------------------------- #

def test_load_regions_renames_the_gid_column(tmp_path):
    shapefile = write_regions(make_regions(mini_polygons()), tmp_path / "r.geojson")

    regions = load_regions(shapefile, False, None, "GID_1")

    assert "region_id" in regions.columns
    assert set(regions["region_id"]) == set(mini_polygons())


def test_load_regions_lists_the_columns_when_the_gid_is_missing(tmp_path):
    """There is no fallback column: the error must name the columns to pick from."""
    regions = make_regions({"X": box(0, 0, 1, 1)}, gid_col="my_id")
    shapefile = write_regions(regions, tmp_path / "r.geojson")

    with pytest.raises(KeyError, match="my_id"):
        load_regions(shapefile, False, None, "GID_1")


def test_load_regions_reprojects_to_wgs84(tmp_path):
    """Weights are computed in degrees, so anything else must be reprojected."""
    regions = make_regions({"R": box(0, 0, 1, 1)}).to_crs("EPSG:3857")
    shapefile = write_regions(regions, tmp_path / "r.gpkg")

    loaded = load_regions(shapefile, False, None, "GID_1")

    assert loaded.crs == "EPSG:4326"


def test_load_regions_reads_a_named_geopackage_layer(tmp_path):
    """GADM ships admin levels as layers, selected with --shapefile-layer."""
    path = str(tmp_path / "multi.gpkg")
    make_regions({"L0": box(0, 0, 1, 1)}, gid_col="GID_0").to_file(
        path, driver="GPKG", layer="level0"
    )
    make_regions({"L1": box(0, 0, 1, 1)}).to_file(path, driver="GPKG", layer="level1")

    level0 = load_regions(path, False, 0, "GID_0")
    level1 = load_regions(path, False, 1, "GID_1")

    assert list(level0["region_id"]) == ["L0"]
    assert list(level1["region_id"]) == ["L1"]


def test_load_regions_can_simplify_geometries(tmp_path):
    circle = gpd.GeoSeries([box(0, 0, 1, 1).buffer(0.5)], crs="EPSG:4326").iloc[0]
    shapefile = write_regions(make_regions({"C": circle}), tmp_path / "r.geojson")

    detailed = load_regions(shapefile, False, None, "GID_1")
    simplified = load_regions(shapefile, True, None, "GID_1")

    n_detailed = len(detailed.geometry.iloc[0].exterior.coords)
    n_simplified = len(simplified.geometry.iloc[0].exterior.coords)
    assert n_simplified < n_detailed
