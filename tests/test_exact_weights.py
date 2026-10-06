"""Tests for the exact, area-aware weights (``compute_grid_weights --method exact``)."""

import numpy as np
import pandas as pd
import pytest
import shapely
from shapely.geometry import MultiPolygon, Point, box

from factories import (
    MINI_EXPECTED_SAMPLED,
    make_grid_nc,
    make_regions,
    mini_coordinates,
    mini_polygons,
    write_regions,
)
from netcdf2region.exact_weights import (
    EARTH_RADIUS_KM,
    cell_area_km2,
    compute_exact_overlap,
    exact_weights,
    grid_edges,
    region_cell_fractions,
)
from netcdf2region.aggregate import aggregate


def _regions(polygons):
    return make_regions(polygons).rename(columns={"GID_1": "region_id"})


def _lookup(df):
    return {(r.region_id, r.lat, r.lon): r.percent_overlap for r in df.itertuples()}


# --------------------------------------------------------------------------- #
# Fractions
# --------------------------------------------------------------------------- #

def test_mini_regions_get_their_exact_fractions():
    """Full, half and quarter cells come out exactly, like the sampled golden values."""
    df = exact_weights(_regions(mini_polygons()), mini_coordinates(), progress=False)
    got = _lookup(df)
    for key, expected in MINI_EXPECTED_SAMPLED.items():
        assert got[key] == pytest.approx(expected, abs=1e-12)


def test_slivers_that_sampling_misses_get_their_exact_area():
    """D and E fall between sample points; exact intersection still sees them."""
    got = _lookup(exact_weights(_regions(mini_polygons()), mini_coordinates(), progress=False))
    assert got[("D", 4.5, 7.5)] == pytest.approx(0.02 * 0.02, rel=1e-9)
    assert got[("E", 1.5, 6.5)] == pytest.approx(0.0005 * 0.0005, rel=1e-6)


def test_every_region_is_represented_and_no_zero_rows():
    df = exact_weights(_regions(mini_polygons()), mini_coordinates(), progress=False)
    assert set(df.region_id) == set(mini_polygons())
    assert (df.percent_overlap > 0).all()
    assert (df.percent_overlap <= 1).all()


def test_polygon_with_hole_and_multipart():
    """A hole is subtracted and the parts of a multipolygon are combined."""
    ring = box(0.0, 0.0, 2.0, 2.0).difference(box(0.25, 0.25, 0.75, 0.75))   # hole of 0.25 in cell (0, 0)
    multi = MultiPolygon([box(4.0, 0.0, 4.5, 1.0), box(6.0, 0.0, 6.25, 1.0)])
    got = _lookup(exact_weights(_regions({"H": ring, "M": multi}), mini_coordinates(), progress=False))
    assert got[("H", 0.5, 0.5)] == pytest.approx(0.75)
    assert got[("H", 1.5, 1.5)] == pytest.approx(1.0)
    assert got[("M", 0.5, 4.5)] == pytest.approx(0.5)
    assert got[("M", 0.5, 6.5)] == pytest.approx(0.25)


def test_bisection_clipping_matches_naive_per_cell_intersection():
    """Recursive bisection must give the same fractions as clipping each cell directly."""
    rng = np.random.default_rng(1)
    blob = Point(4.1, 2.9).buffer(2.3, quad_segs=64)
    blob = shapely.affinity.rotate(blob, 30).union(Point(1.2, 4.7).buffer(0.8))
    lat_lo, lat_hi, lon_lo, lon_hi = grid_edges(mini_coordinates())
    r, c, f = region_cell_fractions(blob, lat_lo, lat_hi, lon_lo, lon_hi)
    for ri, ci, fi in zip(r, c, f):
        cell = box(lon_lo[ci], lat_lo[ri], lon_hi[ci], lat_hi[ri])
        assert fi == pytest.approx(blob.intersection(cell).area / cell.area, rel=1e-9)
    # and no overlapping cell was skipped
    n_naive = sum(
        blob.intersection(box(lon_lo[j], lat_lo[i], lon_hi[j], lat_hi[i])).area > 0
        for i in range(len(lat_lo)) for j in range(len(lon_lo))
    )
    assert len(f) == n_naive


def test_descending_latitudes_give_the_same_weights():
    coords = mini_coordinates()
    flipped = dict(coords, lat=coords["lat"][::-1].copy())
    a = exact_weights(_regions(mini_polygons()), coords, progress=False)
    b = exact_weights(_regions(mini_polygons()), flipped, progress=False)
    key = ["region_id", "lat", "lon"]
    pd.testing.assert_frame_equal(a.sort_values(key).reset_index(drop=True),
                                  b.sort_values(key).reset_index(drop=True))


# --------------------------------------------------------------------------- #
# Cell areas
# --------------------------------------------------------------------------- #

def test_global_cell_areas_sum_to_the_sphere():
    lats = np.arange(-89.75, 90, 0.5)
    lons = np.arange(-179.75, 180, 0.5)
    coords = {"lat": lats, "lon": lons, "lat_res": 0.5, "lon_res": 0.5, "coordinate_origin": "centered"}
    lat_lo, lat_hi, lon_lo, lon_hi = grid_edges(coords)
    total = cell_area_km2(lat_lo[:, None], lat_hi[:, None], lon_lo[None, :], lon_hi[None, :]).sum()
    assert total == pytest.approx(4 * np.pi * EARTH_RADIUS_KM ** 2, rel=1e-12)


def test_cell_area_shrinks_with_latitude():
    eq = cell_area_km2(0.0, 0.5, 0.0, 0.5)
    hi = cell_area_km2(60.0, 60.5, 0.0, 0.5)
    assert hi / eq == pytest.approx(np.cos(np.deg2rad(60.25)), rel=1e-3)


# --------------------------------------------------------------------------- #
# End to end: CSV and aggregation
# --------------------------------------------------------------------------- #

def test_csv_roundtrip_and_area_weighted_mean(tmp_path):
    """A region spanning a low- and a high-latitude cell: with area_weighted the mean
    is area-weighted; without it, and for the sum, cells count by covered fraction only."""
    lats = np.arange(0.5, 61.0, 1.0)   # regular 1 degree grid; the region uses rows 0 and 60
    lons = np.array([0.5, 1.5])
    values = np.zeros((1, len(lats), len(lons)))
    values[0, 0, 0], values[0, 60, 0] = 10.0, 20.0
    nc_grid = make_grid_nc(tmp_path / "grid.nc", lats=lats, lons=lons)
    nc_data = make_grid_nc(tmp_path / "data.nc", lats=lats, lons=lons, n_time=1, values=values)
    regions = write_regions(make_regions({"R": MultiPolygon([box(0, 0, 1, 1), box(0, 60, 1, 61)])}),
                            tmp_path / "r.gpkg")
    w = tmp_path / "w.csv"
    compute_exact_overlap(nc_grid, regions, str(w), n_jobs=1)

    weights = pd.read_csv(w)
    assert list(weights.columns) == ["lat", "lon", "region_id", "percent_overlap", "cell_area_km2"]
    a_lo, a_hi = weights.set_index("lat").cell_area_km2.loc[[0.5, 60.5]]

    out = tmp_path / "mean.csv"
    aggregate(netcdf=nc_data, weights=str(w), method="mean", output=str(out), var="tas",
              area_weighted=True)
    mean = pd.read_csv(out).tas.iloc[0]
    assert mean == pytest.approx((10 * a_lo + 20 * a_hi) / (a_lo + a_hi))
    assert mean < 15  # the smaller high-latitude cell counts less than in a plain mean

    out = tmp_path / "plain_mean.csv"
    aggregate(netcdf=nc_data, weights=str(w), method="mean", output=str(out), var="tas")
    assert pd.read_csv(out).tas.iloc[0] == pytest.approx(15.0)

    with pytest.raises(ValueError, match="mean or std"):
        aggregate(netcdf=nc_data, weights=str(w), method="sum", output=str(out), var="tas",
                  area_weighted=True)

    out = tmp_path / "sum.csv"
    aggregate(netcdf=nc_data, weights=str(w), method="sum", output=str(out), var="tas")
    assert pd.read_csv(out).tas.iloc[0] == pytest.approx(30.0)


def test_area_weighted_requires_cell_area_column(tmp_path):
    nc_data = make_grid_nc(tmp_path / "data.nc", lats=np.array([0.5]), lons=np.array([0.5]),
                           n_time=1, values=np.ones((1, 1, 1)))
    w = tmp_path / "w.csv"
    pd.DataFrame({"lat": [0.5], "lon": [0.5], "region_id": ["R"],
                  "percent_overlap": [1.0]}).to_csv(w, index=False)
    with pytest.raises(ValueError, match="cell_area_km2"):
        aggregate(netcdf=nc_data, weights=str(w), method="mean",
                  output=str(tmp_path / "o.csv"), var="tas", area_weighted=True)


def test_validate_weights_accepts_cell_area_column(tmp_path):
    from netcdf2region.validate_weights import _check_schema
    df = pd.DataFrame({"lat": [0.5], "lon": [0.5], "region_id": ["R"],
                       "percent_overlap": [1.0], "cell_area_km2": [3086.0]})
    assert _check_schema(df).passed


def test_fast_clipping_matches_exact_clipping():
    """clip_by_rect with validity fallback must agree with plain intersections."""
    blob = Point(4.1, 2.9).buffer(2.3, quad_segs=64).union(Point(1.2, 4.7).buffer(0.8))
    islands = MultiPolygon([Point(0.3 + 0.37 * k, 0.4 + 0.29 * (k % 7)).buffer(0.11) for k in range(18)])
    edges = grid_edges(mini_coordinates())
    for geom in (blob, islands, box(0, 0, 8, 6).difference(blob)):
        a = region_cell_fractions(geom, *edges, fast=True)
        b = region_cell_fractions(geom, *edges, fast=False)
        da = dict(zip(zip(a[0], a[1]), a[2])); db = dict(zip(zip(b[0], b[1]), b[2]))
        assert da.keys() == db.keys()
        assert max(abs(da[k] - db[k]) for k in da) < 1e-12


def test_area_check_falls_back_to_exact_clipping(monkeypatch):
    """If fast clipping returns a wrong piece, the area check must catch it."""
    import netcdf2region.exact_weights as ew
    blob = Point(4.1, 2.9).buffer(2.3, quad_segs=64)
    edges = grid_edges(mini_coordinates())
    expected = ew.region_cell_fractions(blob, *edges, fast=False)

    def broken_clip(piece, x0, y0, x1, y1):  # returns the whole rectangle: wrong area
        return shapely.box(x0, y0, x1, y1)

    monkeypatch.setattr(ew, "_clip_fast", broken_clip)
    got = ew.region_cell_fractions(blob, *edges, fast=True)
    assert np.allclose(np.sort(got[2]), np.sort(expected[2]))
