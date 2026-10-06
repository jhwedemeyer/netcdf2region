"""Exact, area-aware grid-cell/region weights (alternative to point sampling).

For every grid cell and every region overlapping it, this computes

* ``percent_overlap``: the exact fraction of the cell (in lon/lat space) covered by the
  region, from the shapely intersection of the region with the cell rectangle, and
* ``cell_area_km2``: the cell's area on the sphere.

The output has the same layout as the point-sampled weights plus the ``cell_area_km2``
column. ``aggregate`` uses ``percent_overlap * cell_area_km2`` for means and standard
deviations (an area-weighted mean, so high-latitude cells are not over-weighted) and
``percent_overlap`` alone for sums, which is the right weight for per-cell totals such as
population counts.

Clipping a large, detailed region directly against thousands of cells is slow, because
every intersection walks the whole outline. Regions are therefore clipped by recursive
bisection (see :func:`region_cell_fractions`).
"""

import logging

import numpy as np
import pandas as pd
import shapely
from joblib import Parallel, delayed
from tqdm import tqdm

from .load_grid import COORD_DECIMALS, cell_bounds, load_template_grid
from .load_regions import load_regions

logger = logging.getLogger(__name__)

#: Mean Earth radius (IUGG), the same value validate_weights uses for its area check.
EARTH_RADIUS_KM = 6371.0088


def grid_edges(coordinates):
    """Return the lower and upper cell edges along each axis.

    Returns:
        tuple: ``(lat_lo, lat_hi, lon_lo, lon_hi)``, indexed like the grid's own
        ``lat`` and ``lon`` axes (so ascending or descending order is preserved).
    """
    lats, lons = coordinates["lat"], coordinates["lon"]
    if coordinates["coordinate_origin"] == "centered":
        # One coordinate per cell (its centre), so there are as many cells as coordinates.
        n_lat, n_lon = len(lats), len(lons)
    else:  # top-left: the coordinates are cell edges
        # N edges enclose N - 1 cells.
        n_lat, n_lon = len(lats) - 1, len(lons) - 1
    # The grid is regular in lon/lat, so the latitude edges depend only on the row and
    # the longitude edges only on the column. Compute them once along each axis (with a
    # fixed other index) instead of for every cell. cell_bounds returns
    # (lon_lo, lat_lo, lon_hi, lat_hi), hence the column picks below.
    rows = np.array([cell_bounds(i, 0, coordinates) for i in range(n_lat)])
    cols = np.array([cell_bounds(0, j, coordinates) for j in range(n_lon)])
    # Clamp to the globe: a cell centred on +-89.75 with 0.5 deg resolution ends at +-90,
    # but rounding can push the edge marginally beyond it.
    return (np.clip(rows[:, 1], -90, 90), np.clip(rows[:, 3], -90, 90), cols[:, 0], cols[:, 2])


def cell_area_km2(lat_lo, lat_hi, lon_lo, lon_hi):
    """Area of the lon/lat rectangle(s) on a sphere of radius ``EARTH_RADIUS_KM``.

    Uses the closed form ``R^2 * dlon * |sin(lat_hi) - sin(lat_lo)|`` (angles in
    radians). The ``sin`` term is why cells shrink towards the poles: equal steps in
    latitude cover less and less area. Works element-wise on arrays.
    """
    deg = np.pi / 180.0  # degrees -> radians
    return (
        EARTH_RADIUS_KM ** 2
        * np.abs(np.asarray(lon_hi) - np.asarray(lon_lo)) * deg
        * np.abs(np.sin(np.asarray(lat_hi) * deg) - np.sin(np.asarray(lat_lo) * deg))
    )


def _clip_fast(piece, x0, y0, x1, y1):
    """Clip to a rectangle with GEOS's fast rectangle clipper, falling back to a full
    intersection if its output is invalid.

    ``clip_by_rect`` is ~100x faster than a general intersection on detailed outlines,
    but on some island-rich regions (e.g. GADM's Nunavut) it returns invalid pieces,
    and clipping those again gives grossly wrong areas (whole cells filled or emptied).
    Checking each piece and redoing only the invalid ones keeps the speed and gives
    the same fractions as exact intersection (to ~1e-14 in tests on GADM admin-1).
    """
    out = shapely.clip_by_rect(piece, x0, y0, x1, y1)
    if out.is_empty or shapely.is_valid(out):
        return out
    # clip_by_rect produced an invalid geometry: redo just this piece the slow, robust way.
    return shapely.intersection(piece, shapely.box(x0, y0, x1, y1))


def _clip_exact(piece, x0, y0, x1, y1):
    """Clip to a rectangle with a full (slower, always valid) GEOS intersection."""
    return shapely.intersection(piece, shapely.box(x0, y0, x1, y1))


def region_cell_fractions(geom, lat_lo, lat_hi, lon_lo, lon_hi, fast=True):
    """Exact covered fraction of every grid cell a region overlaps.

    The region is clipped by recursive bisection: its block of candidate cells is split
    in half along the longer side, the outline is clipped to each half, and the halves
    are processed in turn until single cells remain. Each level only walks the pieces
    produced by the level above, so a detailed outline costs about
    ``vertices * log(cells)`` instead of ``vertices * cells``. A block whose clipped
    area equals the block's own area lies entirely inside the region, and all its cells
    get the fraction 1 without further clipping.

    Args:
        geom: Region geometry (shapely, lon/lat).
        lat_lo, lat_hi, lon_lo, lon_hi: Cell edges as returned by :func:`grid_edges`.
        fast: Clip with :func:`_clip_fast` (default). The result is then checked: the
            covered cell areas must add up to the region's own area, otherwise the
            region is recomputed with plain intersections (``fast=False``).

    Returns:
        tuple: ``(row_idx, col_idx, fraction)`` arrays, one entry per overlapped cell,
        with indices into the grid's own ``lat`` and ``lon`` axes.
    """
    empty = (np.empty(0, int), np.empty(0, int), np.empty(0))
    if geom is None or geom.is_empty:
        return empty
    # Candidate cells: only rows/columns whose extent overlaps the region's bounding box
    # (strict inequalities, so cells that merely touch the box edge are skipped).
    minx, miny, maxx, maxy = geom.bounds
    rows = np.nonzero((lat_hi > miny) & (lat_lo < maxy))[0]
    cols = np.nonzero((lon_hi > minx) & (lon_lo < maxx))[0]
    if len(rows) == 0 or len(cols) == 0:
        return empty
    # Work in ascending edge order, so every block of cells is one contiguous rectangle.
    # (Many grids store latitude north-to-south; ``rows``/``cols`` keep the original
    # grid indices, so results still map back to the grid's own axis order.)
    rows = rows[np.argsort(lat_lo[rows])]
    cols = cols[np.argsort(lon_lo[cols])]
    # Edges of the candidate cells, in block-local (sorted) order: position i in these
    # arrays corresponds to the grid row rows[i], and likewise for columns.
    ylo, yhi = lat_lo[rows], lat_hi[rows]
    xlo, xhi = lon_lo[cols], lon_hi[cols]

    # Results are accumulated here by the recursion below.
    out_r, out_c, out_f = [], [], []
    clip = _clip_fast if fast else _clip_exact

    # A "block" is the half-open index range [i0, i1) x [j0, j1) of candidate cells; its
    # rectangle runs from the lower edge of the first cell to the upper edge of the last.
    def block_clip(piece, i0, i1, j0, j1):
        return clip(piece, xlo[j0], ylo[i0], xhi[j1 - 1], yhi[i1 - 1])

    def visit(piece, i0, i1, j0, j1):
        # ``piece`` is the part of the region that lies inside this block. Areas are in
        # squared degrees (planar lon/lat); only their ratio matters for the fraction.
        block_area = (xhi[j1 - 1] - xlo[j0]) * (yhi[i1 - 1] - ylo[i0])
        piece_area = shapely.area(piece)
        if piece_area <= 0:
            # Only a line or point touches the block (e.g. a shared edge): no overlap.
            return
        if (i1 - i0) * (j1 - j0) == 1:
            # Base case, a single cell: its fraction is the covered share of its area.
            # Capped at 1 to absorb floating-point rounding in the clipped area.
            out_r.append(rows[i0]); out_c.append(cols[j0])
            out_f.append(min(piece_area / block_area, 1.0))
            return
        if piece_area >= block_area * (1 - 1e-12):
            # The whole block is covered: every cell in it gets the fraction 1.
            # (This shortcut is what makes large regions cheap: their interior is
            # resolved in a few big blocks instead of cell by cell.)
            ii, jj = np.meshgrid(rows[i0:i1], cols[j0:j1], indexing="ij")
            out_r.extend(ii.ravel()); out_c.extend(jj.ravel()); out_f.extend([1.0] * ii.size)
            return
        # Partially covered block: split it in half along its longer side (in cells),
        # which keeps blocks roughly square and the recursion depth ~log2(cells).
        if (i1 - i0) >= (j1 - j0):
            m = (i0 + i1) // 2
            halves = ((i0, m, j0, j1), (m, i1, j0, j1))
        else:
            m = (j0 + j1) // 2
            halves = ((i0, i1, j0, m), (i0, i1, m, j1))
        for h in halves:
            # Clip the current piece (not the full region) to the half, so each level
            # only processes the vertices that actually fall inside it.
            part = block_clip(piece, *h)
            if not part.is_empty:
                visit(part, *h)

    n_i, n_j = len(rows), len(cols)
    # The first clip, to the region's whole candidate block, always uses the exact
    # intersection: its area is the reference for the consistency check below.
    root = _clip_exact(geom, xlo[0], ylo[0], xhi[n_j - 1], yhi[n_i - 1])
    visit(root, 0, n_i, 0, n_j)
    if not out_r:
        return empty
    r_idx = np.asarray(out_r, dtype=int)
    c_idx = np.asarray(out_c, dtype=int)
    frac = np.asarray(out_f, dtype=float)

    if fast:
        # Independent check: the covered parts of the cells must add up to the region.
        # If _clip_fast silently lost or duplicated area anywhere, this sum disagrees with
        # the exactly clipped region area, and the whole region is redone exactly.
        covered = (frac * (lon_hi[c_idx] - lon_lo[c_idx]) * (lat_hi[r_idx] - lat_lo[r_idx])).sum()
        if not np.isclose(covered, shapely.area(root), rtol=1e-9, atol=1e-12):
            return region_cell_fractions(geom, lat_lo, lat_hi, lon_lo, lon_hi, fast=False)
    return r_idx, c_idx, frac


def exact_weights(regions, coordinates, n_jobs=1, backend="threading", progress=True):
    """Exact overlap fractions and cell areas for every region/cell pair.

    Args:
        regions: GeoDataFrame with ``region_id`` and ``geometry`` (EPSG:4326).
        coordinates: Grid description as returned by ``load_template_grid``.
        n_jobs: Parallel jobs. shapely releases the GIL in its vectorised operations,
            so the default threading backend scales without copying the geometries.
        backend: joblib backend.
        progress: Show a progress bar.

    Returns:
        pandas.DataFrame: ``lat, lon, region_id, percent_overlap, cell_area_km2``.
    """
    lat_lo, lat_hi, lon_lo, lon_hi = grid_edges(coordinates)

    regions = regions[["region_id", "geometry"]]
    if regions["region_id"].duplicated().any():
        # Several rows per region (e.g. multipart regions stored row-wise): merge them.
        regions = regions.dissolve(by="region_id").reset_index()
    # Repair self-intersections etc. up front: GEOS clipping can fail or give wrong areas
    # on invalid input geometries.
    geoms = shapely.make_valid(regions.geometry.values)
    ids = regions["region_id"].to_numpy()

    # Process regions in order of decreasing vertex count (a proxy for clipping cost).
    # Starting the slow ones first stops a single large region from running alone at
    # the end while the other workers sit idle.
    order = np.argsort(-shapely.get_num_coordinates(geoms))
    tasks = (delayed(region_cell_fractions)(geoms[k], lat_lo, lat_hi, lon_lo, lon_hi) for k in order)
    # return_as="generator" yields results in task order as they finish, so the progress
    # bar advances steadily and results don't all have to be held at once.
    results = Parallel(n_jobs=n_jobs, backend=backend, return_as="generator")(tasks)
    if progress:
        results = tqdm(results, total=len(order), desc="Exact overlaps")

    # Results come back in ``order``, so zip with it to recover each region's id.
    parts = []
    for k, (r, c, f) in zip(order, results):
        if len(r):
            parts.append(pd.DataFrame({"row": r, "col": c, "region_id": ids[k], "percent_overlap": f}))
    df = pd.concat(parts, ignore_index=True)

    # Turn grid indices into the output columns: the cell's spherical area, and its
    # lat/lon coordinates rounded like the point-sampled weights, so both CSV kinds can
    # be joined against the same NetCDF coordinates.
    lats, lons = coordinates["lat"], coordinates["lon"]
    df["cell_area_km2"] = cell_area_km2(lat_lo[df.row], lat_hi[df.row], lon_lo[df.col], lon_hi[df.col])
    df.insert(0, "lat", np.round(lats[df.row], COORD_DECIMALS))
    df.insert(1, "lon", np.round(lons[df.col], COORD_DECIMALS))
    return df.drop(columns=["row", "col"])


def compute_exact_overlap(netcdf_path, shapefile_path, output_csv, shapefile_layer=None,
                          default_gid="GID_1", coordinate_origin="centered",
                          shift_longitudes=False, simplify=False, n_jobs=1,
                          backend="threading", yes=False):
    """Compute exact, area-aware weights for a NetCDF grid and a region file and write a CSV."""
    coordinates = load_template_grid(netcdf_path, coordinate_origin=coordinate_origin,
                                     shift_longitudes=shift_longitudes)
    logger.info(f"Loading shapefile: {shapefile_path}")
    regions = load_regions(shapefile_path, simplify=simplify, shapefile_layer=shapefile_layer,
                           default_gid=default_gid, yes=yes)
    logger.info("Calculating exact overlaps between grid cells and regions...")
    df = exact_weights(regions, coordinates, n_jobs=n_jobs, backend=backend)

    # Unlike point sampling, exact clipping never misses small regions, so a region can
    # only be absent here if it lies entirely outside the grid's extent.
    missing = sorted(set(regions["region_id"]) - set(df["region_id"]))
    if missing:
        logger.warning(f"{len(missing)} region(s) do not overlap the grid at all: {missing[:10]}")

    # Fractions keep 8 significant digits, so even tiny regions keep a non-zero weight.
    df.to_csv(output_csv, index=False, float_format="%.8g")
    logger.info(f"Done. {len(df)} rows for {df['region_id'].nunique()} regions saved to {output_csv}")
    return df
