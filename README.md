<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/netcdf2region_logo2_dark.svg">
  <img src="docs/_static/netcdf2region_logo2_light.svg" alt="netcdf2region" width="520">
</picture>

[![Tests](https://github.com/jhwedemeyer/netcdf2region/actions/workflows/tests.yml/badge.svg)](https://github.com/jhwedemeyer/netcdf2region/actions/workflows/tests.yml)
[![PyPI](https://img.shields.io/pypi/v/netcdf2region.svg)](https://pypi.org/project/netcdf2region/)
[![Python Versions](https://img.shields.io/pypi/pyversions/netcdf2region.svg)](https://pypi.org/project/netcdf2region/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22062731.svg)](https://doi.org/10.5281/zenodo.22062731)

`netcdf2region` is a Python package providing command-line tools for aggregating gridded NetCDF files over regions defined by shapefiles. It is designed to facilitate spatial analysis and extraction of regional statistics from large climate or geospatial datasets.

> **Disclaimer:** This package is still in the early stages of development. Please check that all outputs are valid, as errors during calculation may still occur.

## Features

- Aggregate gridded NetCDF data by custom regions (e.g., administrative boundaries, ecozones).
- Support for shapefile-based region definitions.
- Efficient processing of large datasets.
- Command-line interface for easy integration into workflows.

## Installation

```bash
pip install netcdf2region
```

## Workflow

Aggregating NetCDF data over regions is a two-step process:

1. **`compute_grid_weights`** – compute how much each grid cell overlaps each region, once per grid/shapefile combination, and write the result to a weights CSV. The package also ships [precomputed weights](#precomputed-weights) for common grids and shapefiles, so this step can often be skipped.
2. **`netcdf2region`** – aggregate one or more NetCDF variables over the regions using those weights. This step is cheap, so it is the one to repeat for each new NetCDF file on the same grid.

A third command, **`validate_weights`**, checks a weights CSV independently — useful for auditing precomputed weights or weights produced by other tools.

## Commands

Every option below also accepts `--help` on the command itself for the authoritative, up-to-date list. Options are spelled with hyphens (e.g. `--shift-longitudes`); the old underscore spellings (e.g. `--shift_longitudes`) still work but no longer show up in `--help`.

### `compute_grid_weights`

Estimates the overlap between NetCDF grid cells and the regions defined in a shapefile.

```bash
compute_grid_weights <netcdf> <shapefile> <output> [options]
```

**Positional arguments**

| Argument      | Description                                           |
|---------------|--------------------------------------------------------|
| `<netcdf>`    | Path to the NetCDF file with a latitude/longitude grid. |
| `<shapefile>` | Path to the shapefile defining regions (e.g., `.shp`, `.gpkg`). |
| `<output>`    | Path to the output CSV file for the computed grid weights. |

**Options**

| Option | Default | Description |
|---|---|---|
| `--method <exact\|sampling>` | `exact` | `exact` intersects each region with the grid cells and also writes each cell's area (`cell_area_km2`), which `netcdf2region --area-weighted` uses to compute area-weighted means and standard deviations. `sampling` estimates overlaps by sampling points in each cell. |
| `--points-per-cell <int>` | `100` | Number of points per grid cell. Only used with `--method sampling`. |
| `--point-distribution <uniform\|random>` | `uniform` | Distribution of points within grid cells. Only used with `--method sampling`. |
| `--simplify` | off | Simplify region geometries for faster computation. |
| `--shapefile-layer <int>` | first layer | Layer number in the shapefile to use. |
| `--gid <str>` | `GID_1` | Column in the shapefile with the region identifier. |
| `--coordinate-origin <centered\|top-left>` | `centered` | Whether the grid coordinates are cell centres or top-left cell edges. |
| `--shift-longitudes` | off | Shift longitudes from `[0, 360]` to `[-180, 180]`. |
| `--backend <str>` | `threading` | Parallel backend to use. |
| `--n-jobs <int>` | `8` | Number of parallel jobs. |
| `--no-validate` | off | Skip validating the computed weights. By default the weights are checked against the grid and the regions (the same checks as `validate_weights`), the report is printed, and the command exits with an error if an error-level check fails (the CSV is still written). |
| `--max-cell-sum <float>` | `1.01` | Validation warns when a cell's weights sum above this. Raise it for shapefiles with overlapping polygons. |

### `netcdf2region`

Aggregates gridded NetCDF data over regions using precomputed grid weights (either your own, from `compute_grid_weights`, or one of the package's [precomputed weights](#precomputed-weights)).

```bash
netcdf2region \
    --netcdf <input.nc> \
    (--weights <weights.csv> | --precalc-weights <precalc_weights_name>) \
    --method <mean|sum|std> \
    --output <output.csv> \
    --var <variable_name> \
    [options]
```

**Required arguments**

| Option | Description |
|---|---|
| `--netcdf <path>` | Path to the input NetCDF file. |
| `--weights <path>` | Path to the CSV file with grid weights (`lat,lon,region_id,percent_overlap`). Mutually exclusive with `--precalc-weights`; exactly one of the two is required. |
| `--precalc-weights <name>` | Name of a predefined weights file (e.g., `ERA5_to_GADM36_admin1_weights.csv`). Mutually exclusive with `--weights`; exactly one of the two is required. |
| `--method <mean\|sum\|std>` | Aggregation method. |
| `--output <path>` | Path to the output CSV file. |
| `--var <name>` | Name of the variable in the NetCDF file to aggregate. |

**Optional arguments**

| Option | Default | Description |
|---|---|---|
| `--correct-weights` | off | Correct weights to ensure no values are attributed to the sea. |
| `--time-var <name>` | auto-detected | Name of the time variable in the NetCDF file, if it can't be detected automatically. |
| `--secondary-weights-var <name>` | none | Name of a secondary weight variable in a secondary NetCDF file. Must have dimensions `(lat, lon)` on the exact same grid as `--netcdf`; the run aborts if the coordinates differ, so crop or regrid the secondary file first. Values are multiplied with `percent_overlap` from the weights CSV during aggregation. Requires `--secondary-weights-netcdf`. |
| `--secondary-weights-netcdf <path>` | none | Path to the NetCDF file containing `--secondary-weights-var`. Required when `--secondary-weights-var` is given. |
| `--allow-nan` | off | By default the command aborts if the aggregated variable (or the secondary weight variable) contains NaN values, since these silently corrupt the regional results. Set this flag to proceed anyway: a warning is printed instead, and every region that overlaps a NaN grid cell aggregates to NaN. |
| `--exclude-nan-cells` | off | Implies `--allow-nan` for the aggregated variable (not for the secondary weights). Excludes grid cells that are NaN at every time step from the aggregation. Cells that are NaN at only some time steps are not affected, so regions still aggregate to NaN at those time steps unless `--fill-nans` is also set. A region whose cells are all excluded aggregates to NaN. |
| `--fill-nans <value>` | none | Implies `--allow-nan` for the aggregated variable (not for the secondary weights). Replaces NaN values with `<value>` (e.g. `--fill-nans 0`) while keeping the cell's weight, so missing values no longer turn the regional aggregate into NaN. If `--exclude-nan-cells` is also given, the exclusion runs first, so only NaNs in the remaining cells are filled. Note that for `mean`/`std` this pulls the result towards the fill value at the affected cells and time steps. |
| `--shift-longitudes` | off | Shift the input longitudes from `[0, 360]` to `[-180, 180]` and reorder the data accordingly. Weights files are always written in `[-180, 180]`, so a NetCDF on a `[0, 360]` grid (e.g. raw ERA5) needs this flag; without it the command aborts. Same meaning as in `compute_grid_weights`. |
| `--area-weighted` | off | Weight each grid cell by the area it contributes to the region (covered fraction × cell area) instead of by the covered fraction alone. Use this for `mean`/`std` of intensive quantities (temperature, precipitation rate, ...) on regular lat/lon grids, where cells shrink towards the poles. Requires a weights file with a `cell_area_km2` column, as written by `compute_grid_weights --method exact`. Not allowed with `sum`, which should add up per-cell totals by covered fraction only. |

#### Example

```bash
netcdf2region \
    --netcdf ERA5_sample.nc \
    --precalc-weights ERA5_to_GADM36_admin1_weights.csv \
    --method mean \
    --output ERA5_admin1_mean.csv \
    --var temperature
```

This computes the mean of the `temperature` variable for each region defined in the GADM 3.6 Admin 1 shapefile, using the precomputed weights.

### `validate_weights`

Validates a weights CSV — the same checks `compute_grid_weights` runs by default — independently of computing it. Useful for auditing precomputed weights or weights produced elsewhere.

```bash
validate_weights <weights> [options]
```

**Positional arguments**

| Argument    | Description                             |
|-------------|------------------------------------------|
| `<weights>` | Path to the weights CSV to validate.     |

**Options**

| Option | Default | Description |
|---|---|---|
| `--netcdf <path>` | none | Template NetCDF the weights were computed for. Enables the on-grid check. |
| `--shapefile <path>` | none | Shapefile the weights were computed from. Enables the coverage/area checks. |
| `--shapefile-layer <int>` | first layer | Layer index in the shapefile. |
| `--gid <str>` | `GID_1` | Column with the region identifier. |
| `--coordinate-origin <centered\|top-left>` | `centered` | Whether the grid coordinates are cell centres or top-left cell edges. |
| `--shift-longitudes` | off | Shift longitudes from `[0, 360]` to `[-180, 180]`. |
| `--max-cell-sum <float>` | `1.01` | Warn when a cell's weights sum above this. |
| `--area-rtol <float>` | `0.05` | Relative tolerance for the region area check. |
| `--strict` | off | Exit non-zero when any check fails, including warnings (by default only error-level checks cause a non-zero exit). |

## Python API

The same steps are available from Python:

```python
from netcdf2region import aggregate, compute_weights, load_template_grid, validate_weights

compute_weights("grid.nc", "regions.gpkg", "weights.csv", method="exact")
validate_weights("weights.csv", grid=load_template_grid("grid.nc")).raise_if_failed()
aggregate(netcdf="data.nc", weights="weights.csv", method="mean", output="out.csv", var="tas")
```

## Precomputed weights

The package provides precomputed weights for standard grids and shapefiles that can be used immediately with the `netcdf2region` command via `--precalc-weights`.

| Grid         | Shapefile         | Weights file                                 |
|--------------|-------------------|----------------------------------------------|
| ISIMIP 0.5°  | GADM 3.6 Admin 0  | `ISIMIP_to_GADM36_admin0_weights.csv`        |
| ISIMIP 0.5°  | GADM 3.6 Admin 1  | `ISIMIP_to_GADM36_admin1_weights.csv`        |
| ISIMIP 0.5°  | DOSE V2 Admin 1   | `ISIMIP_to_DOSE_admin1_weights.csv`          |
| ERA5 0.25°   | GADM 3.6 Admin 0  | `ERA5_to_GADM36_admin0_weights.csv`          |
| ERA5 0.25°   | GADM 3.6 Admin 1  | `ERA5_to_GADM36_admin1_weights.csv`          |
| ERA5 0.25°   | DOSE Admin 1      | `ERA5_to_DOSE_admin1_weights.csv`            |

See the [`netcdf2region` example](#example) above for how to use one of these.

## Running on an HPC with Slurm

This example assumes that the Python environment `netcdf2region_env` is available, with the package installed:

```bash
conda create -n netcdf2region_env -c anaconda
pip install netcdf2region
```

SLURM script to be executed with `sbatch`:

```bash
#!/bin/bash
#SBATCH --qos=priority
#SBATCH --account=<account_name>
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000

# Important to be able to find netcdf2region
export PATH="$HOME/.local/bin:$PATH"

# Setup python environment
module purge
module load anaconda
source activate netcdf2region_env

# Setup paths
NETCDF_GRID="ERA5_sample.nc"
SHAPEFILE="gadm36_levels.gpkg"
SHAPEFILE_LAYER=1
WEIGHTS_OUTPUT="ERA5_to_GADM36_admin1_weights.csv"

# Settings for parallel computation
NUMBER_OF_JOBS=16 # Number of parallel jobs set to the number of CPU cores

# Run the Python script
compute_grid_weights $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --shapefile-layer=$SHAPEFILE_LAYER --n-jobs=$NUMBER_OF_JOBS
```

## Requirements

- Python 3.7+
- numpy
- pandas
- xarray
- geopandas
- shapely
- joblib
- tqdm
- netCDF4

## Acknowledgements

This package builds on prior work of Leonie Wenz and Maximilian Kotz, whose approach to aggregating gridded climate data onto administrative regions forms the methodological basis for `netcdf2region`.

Thanks to Anna Reckwitz, Katharina Ziegler, and Maik Hoffmann for early testing and for their help in improving the package.

Special thanks go to Olalla Castro for designing the logo.

## Contributing

Contributions are welcome! Please open issues or pull requests.
