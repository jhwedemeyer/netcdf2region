"""Loading the region geometries from a shapefile.

Used by both :mod:`sampled_weights` and :mod:`exact_weights`, and by ``validate_weights``.
"""

import logging

import geopandas as gpd

logger = logging.getLogger(__name__)


def load_regions(shapefile_path, simplify, shapefile_layer, default_gid, yes=False):
    """Load regions from a shapefile and optionally simplify geometries.

    The ``default_gid`` column becomes ``region_id``. If the shapefile has no such
    column, ``yes`` falls back to the first column; otherwise a ``KeyError`` lists the
    available columns so the caller can pick one.
    """

    regions = gpd.read_file(shapefile_path, layer=shapefile_layer).to_crs("EPSG:4326")
    if default_gid in regions.columns:
        region_col = default_gid
    elif yes:
        region_col = regions.columns[0]
        logger.warning(
            f"Column '{default_gid}' not found in shapefile. "
            f"Using first column as region identifier: {region_col}"
        )
    else:
        raise KeyError(
            f"Column '{default_gid}' not found in shapefile '{shapefile_path}'. "
            f"Available columns: {list(regions.columns)}. Pass the region identifier "
            "column with --gid, or --yes to use the first column."
        )
    regions = regions.rename(columns={region_col: 'region_id'})

    if simplify:
        logger.info("Simplifying region geometries...")
        regions["geometry"] = regions["geometry"].simplify(tolerance=0.01, preserve_topology=True)

    return regions
