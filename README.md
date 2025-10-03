# netcdf2region

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

The `compute_grid_weights` command estimates the overlap between NetCDF grid cells and regions defined in a shapefile via point sampling. This step is performed separately to enable efficient repeated aggregations. The package also provides precomputed weights for common grids, such as ERA5, and shapefiles, such as GADM 3.6. See the section 'Precomputed weights' for more information.

```bash
compute_grid_weights <netcdf> <shapefile> <output> [options]
```

#### Positional Arguments

- `<netcdf>`: Path to the NetCDF file with latitude/longitude grid.
- `<shapefile>`: Path to the shapefile defining regions (e.g., `.shp`).
- `<output>`: Path to the output CSV file for computed grid weights.

#### Options

- `--points_per_cell <int>`: Number of random points per grid cell (default: 100).
- `--simplify`: Simplify region geometries for faster computation.
- `--land_sea_mask <file>`: Path to a land-sea mask NetCDF file for faster computation (optional).
- `--shapefile_layer <int>`: Layer number in the shapefile to use (default: first layer).
- `--backend <str>`: Parallel backend to use (`threading` by default).
- `--n_jobs <int>`: Number of parallel jobs (default: 8).
- `--coordinate_origin <str>`: Coordinate origin of the grid (default: centered).
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
    [--time_var <time_variable>]
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
^^
See the command-line help (`--help`) for more options and details.

## Precomputed weights
The package provides precomputed weights for standard grids and shapefiles that can be used immediately with the `netcdf2region` command.

| Grid         | Shapefile         | Weights file                                 |
|--------------|-------------------|----------------------------------------------|
| ISIMIP 0.5°  | GADM 3.6 Admin 0  | ISIMIP_to_GADM36_admin0_weights.csv          |
| ISIMIP 0.5°  | GADM 3.6 Admin 1  | ISIMIP_to_GADM36_admin1_weights.csv          |
| ISIMIP 0.5°  | DOSE Admin 1      | ISIMIP_to_DOSE_admin1_weights.csv            |
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
pip install /path/to/netcdf2region
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

## Contributing

Contributions are welcome! Please open issues or pull requests.
