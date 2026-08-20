#!/bin/bash 
#SBATCH --qos=short
#SBATCH --account=impactee
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --output=temp/weights-%j.out
#SBATCH --error=temp/weights-%j.err

# Setup python environment
module purge
module load anaconda
source ~/.venvs/netcdf2region-dev/bin/activate

# Setup paths
NETCDF_GRID="/p/projects/climate_data_central/reanalysis/ERA5/2m_temperature/2m_temperature_200010.nc"
SHAPEFILE="/p/tmp/lochner/gadm36_levels.gpkg"
SHAPEFILE_LAYER=1
WEIGHTS_OUTPUT="data/ERA5_to_GADM36_admin1_weights.csv"

# Settings for parallel computation
NUMBER_OF_JOBS=16 # Number of parallel jobs set to the number of CPU cores

# Run the Python script
compute_grid_weights $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --yes --shapefile_layer=$SHAPEFILE_LAYER --n_jobs=$NUMBER_OF_JOBS --shift_longitudes

if [ $? -ne 0 ]; then
    echo "compute_grid_weights failed; not validating." >&2
    exit 1
fi

# Check the freshly computed weights before anything downstream consumes them.
# Errors (regions missing entirely, coordinates off the grid, weights out of
# range) exit non-zero and fail the job; cell sums above the ceiling are only
# reported, since overlapping source polygons make them legitimate.
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shapefile_layer=$SHAPEFILE_LAYER \
    --shift_longitudes \
    --max-cell-sum 1.01
