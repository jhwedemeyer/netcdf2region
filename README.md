<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/_static/logo-horizontal-dark.svg">
  <img src="docs/_static/logo-horizontal-light.svg" alt="netcdf2region" width="520">
</picture>

`netcdf2region` is a Python package providing command-line tools for aggregating gridded NetCDF files over regions defined by shapefiles. It is designed to facilitate spatial analysis and extraction of regional statistics from large climate or geospatial datasets.

## Features

- Aggregate gridded NetCDF data by custom regions (e.g., administrative boundaries, ecozones).
- Support for shapefile-based region definitions.
- Efficient processing of large datasets.
- Command-line interface for easy integration into workflows.

## Installation

```bash
pip install netcdf2region
```
## Disclaimer

This package is still in the early stages of development. Please check that all outputs are valid, as errors during calculation may still occur.

## Usage

### 1. Compute Grid Weights

The `compute_grid_weights` command estimates the overlap between NetCDF grid cells and regions defined in a shapefile, by default exactly from the geometries. This step is performed separately to enable efficient repeated aggregations. The package also provides precomputed weights for common grids, such as ERA5, and shapefiles, such as GADM 3.6. See the section 'Precomputed weights' for more information.

```bash
compute_grid_weights <netcdf> <shapefile> <output> [options]
```

#### Positional Arguments

- `<netcdf>`: Path to the NetCDF file with latitude/longitude grid.
- `<shapefile>`: Path to the shapefile defining regions (e.g., `.shp`).
- `<output>`: Path to the output CSV file for computed grid weights.

#### Options

- `--method <exact|sampling>`: How overlaps are computed (default: exact). `exact` intersects each region with the grid cells and also writes each cell's area (`cell_area_km2`), so `netcdf2region` computes area-weighted means and standard deviations. `sampling` estimates overlaps by sampling points in each cell.
- `--points_per_cell <int>`: Number of points per grid cell, only with `--method sampling` (default: 100).
- `--point_distribution <uniform|random>`: Distribution of points within grid cells, only with `--method sampling` (default: uniform)
- `--simplify`: Simplify region geometries for faster computation.
- `--shapefile_layer <int>`: Layer number in the shapefile to use (default: first layer).
- `--backend <str>`: Parallel backend to use (`threading` by default).
- `--n_jobs <int>`: Number of parallel jobs (default: 8).
- `--coordinate_origin <str>`: Coordinate origin of the grid (default: centered).
- `--gid <str>`: Geo identifier of the shapefile (default: GID_1)
- `--shift_longitudes`: Shift longitudes from [0, 360] to [-180, 180]
- `--yes`: Automatically answer yes to prompts.

See `--help` for more details.

### 2. Aggregate NetCDF Data
The `netcdf2region` command aggregates gridded NetCDF data over regions using precomputed grid weights, which are estimated via the `compute_grid_weights` command. The package also provides precomputed weights for common grids, such as ERA5, and shapefiles, such as GADM 3.6. These can be used directly with the `--precalc_weights` option (see the section 'Precomputed weights').

```bash
netcdf2region \
    --netcdf <input.nc> \
    (--weights <weights.csv> | --precalc_weights <precalc_weights_name>) \
    --method <mean|sum|std> \
    --output <output.csv> \
    --var <variable_name> \
    [--correct_weights] \
    [--time_var <time_variable>] \
    [--secondary_weights_var <secondary_weight_variable>] \
    [--secondary_weights_netcdf <secondary_weights.nc>] \
    [--allow_nan] \
    [--exclude_nan_cells] \
    [--fill_nans <value>] \
    [--shift_longitudes]
```

#### Arguments

- `--netcdf`: Path to the input NetCDF file.
- `--weights`: Path to the CSV file with grid weights (`lat,lon,region_id,percent_overlap`). Mutually exclusive with `--precalc_weights`.
- `--precalc_weights`: Name of a predefined weights file (e.g., `ERA5_to_GADM36_admin1_weights.csv`). Mutually exclusive with `--weights`.
- `--method`: Aggregation method (`mean`, `sum`, or `std`).
- `--output`: Path to the output CSV file.
- `--var`: Name of the variable in the NetCDF file to aggregate.
- `--correct_weights`: (Optional) If set, corrects weights to ensure no values are attributed to the sea.
- `--time_var`: (Optional) Name of the time variable in the NetCDF file (if not automatically detected).
- `--secondary_weights_var`: (Optional) Name of a secondary weight variable in the secondary NetCDF file. Must have dimensions `(lat, lon)` on the exact same grid as `--netcdf`; the run aborts if the coordinates differ, so crop or regrid the secondary file first. Values are multiplied with `percent_overlap` from the weights CSV during aggregation.
- `--secondary_weights_netcdf`: Path to the NetCDF file containing `--secondary_weights_var`. Required when `--secondary_weights_var` is provided.
- `--allow_nan`: (Optional) By default the command aborts if the aggregated variable (or the secondary weight variable) contains NaN values, since these silently corrupt the regional results. Set this flag to proceed anyway: a warning is printed instead, and every region that overlaps a NaN grid cell aggregates to NaN.
- `--exclude_nan_cells`: (Optional) Implies `--allow_nan` for the aggregated variable (not for the secondary weights). Excludes grid cells that are NaN at every time step from the aggregation. Cells that are NaN at only some time steps are not affected, so regions still aggregate to NaN at those time steps unless `--fill_nans` is also set. A region whose cells are all excluded aggregates to NaN.
- `--fill_nans <value>`: (Optional) Implies `--allow_nan` for the aggregated variable (not for the secondary weights). Replaces NaN values with `<value>` (e.g. `--fill_nans 0`) while keeping the cell's weight, so missing values no longer turn the regional aggregate into NaN. If `--exclude_nan_cells` is also given, the exclusion runs first, so only NaNs in the remaining cells are filled. Note that for `mean`/`std` this pulls the result towards the fill value at the affected cells and time steps.
- `--shift_longitudes`: (Optional) Shift the input longitudes from `[0, 360]` to `[-180, 180]` and reorder the data accordingly. Weights files are always written in `[-180, 180]`, so a NetCDF on a `[0, 360]` grid (e.g. raw ERA5) needs this flag; without it the command aborts. Same meaning as in `compute_grid_weights`.

See the command-line help (`--help`) for more options and details.

## Precomputed weights
The package provides precomputed weights for standard grids and shapefiles that can be used immediately with the `netcdf2region` command.

| Grid         | Shapefile         | Weights file                                 |
|--------------|-------------------|----------------------------------------------|
| ISIMIP 0.5°  | GADM 3.6 Admin 0  | ISIMIP_to_GADM36_admin0_weights.csv          |
| ISIMIP 0.5°  | GADM 3.6 Admin 1  | ISIMIP_to_GADM36_admin1_weights.csv          |
| ISIMIP 0.5°  | DOSE V2 Admin 1   | ISIMIP_to_DOSE_admin1_weights.csv            |
| ERA5 0.25°   | GADM 3.6 Admin 0  | ERA5_to_GADM36_admin0_weights.csv            |
| ERA5 0.25°   | GADM 3.6 Admin 1  | ERA5_to_GADM36_admin1_weights.csv            |
| ERA5 0.25°   | DOSE Admin 1      | ERA5_to_DOSE_admin1_weights.csv              |


### Example: Using Precomputed Weights

To aggregate NetCDF data using precomputed weights, specify the `--precalc_weights` option with the name of the weights file. For example, to aggregate ERA5 data over GADM 3.6 Admin 1 regions:

```bash
netcdf2region \
    --netcdf ERA5_sample.nc \
    --precalc_weights ERA5_to_GADM36_admin1_weights.csv \
    --method mean \
    --output ERA5_admin1_mean.csv \
    --var temperature
```

This command computes the mean of the `temperature` variable for each region defined in the GADM 3.6 Admin 1 shapefile using the precomputed weights.


## Example of Executing Weight Calculations on a HPC with Slurm Workload Manager

This example assumes that the Python environment 'netcdf2region_env' is available, with the package installed:
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
compute_grid_weights $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --yes --shapefile_layer=$SHAPEFILE_LAYER --n_jobs=$NUMBER_OF_JOBS
```

## Requirements

- Python 3.7+
- numpy
- pandas
- xarray
- rasterio
- geopandas
- matplotlib
- shapely
- joblib
- scipy
- tqdm
- netCDF4

## Acknowledgements

This package builds on prior work of Leonie Wenz and Maximilian Kotz, whose approach to aggregating gridded climate data onto administrative regions forms the methodological basis for `netcdf2region`.

Thanks to Anna Reckwitz, Katharina Ziegler, and Maik Hoffmann for early testing and for their help in improving the package.

## Contributing

Contributions are welcome! Please open issues or pull requests.
