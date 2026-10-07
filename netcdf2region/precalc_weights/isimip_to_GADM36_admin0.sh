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
NETCDF_GRID="/p/projects/isimip/isimip/ISIMIP2b/InputData/gdp/rcp26soc/gdp_rcp26soc_0p5deg_annual_2006-2099.nc4"
SHAPEFILE="/p/tmp/lochner/gadm36_levels.gpkg"
SHAPEFILE_LAYER=0
WEIGHTS_OUTPUT="data/ISIMIP_to_GADM36_admin0_weights.csv"

# Settings for parallel computation
NUMBER_OF_JOBS=16 # Number of parallel jobs set to the number of CPU cores

# compute_grid_weights validates the weights it writes before exiting. Errors
# (regions missing entirely, coordinates off the grid, weights out of range) exit
# non-zero and fail the job; cell sums above --max-cell-sum are only reported,
# since overlapping source polygons make them legitimate.
compute_grid_weights $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --gid=GID_0 --shapefile-layer=$SHAPEFILE_LAYER --n-jobs=$NUMBER_OF_JOBS --max-cell-sum 1.01
