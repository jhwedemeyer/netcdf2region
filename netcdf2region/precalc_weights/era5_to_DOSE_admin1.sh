#!/bin/bash 
#SBATCH --qos=short
#SBATCH --account=impactee
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=100000
#SBATCH --output=temp/weights-%j.out
#SBATCH --error=temp/weights-%j.err

# Setup python environment
module purge
module load anaconda
source ~/.venvs/netcdf2region-dev/bin/activate

# Setup paths
NETCDF_GRID="/p/projects/climate_data_central/reanalysis/ERA5/2m_temperature/2m_temperature_200010.nc"
SHAPEFILE="/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/DOSE_REGIONS.gpkg"
WEIGHTS_OUTPUT="data/ERA5_to_DOSE_admin1_weights.csv"

# Settings for parallel computation
NUMBER_OF_JOBS=16 # Number of parallel jobs set to the number of CPU cores

# compute_grid_weights validates the weights it writes before exiting. Errors
# (regions missing entirely, coordinates off the grid, weights out of range) exit
# non-zero and fail the job; cell sums above --max-cell-sum are only reported,
# since overlapping source polygons make them legitimate.
compute_grid_weights $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --n-jobs=$NUMBER_OF_JOBS --shift-longitudes --max-cell-sum 2.0
