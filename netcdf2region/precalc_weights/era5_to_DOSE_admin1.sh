#!/bin/bash 
#SBATCH --qos=priority
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

# Use the compute_grid_weights of this package, not a system wide installation.
# Under sbatch the script runs from a spool copy, so resolve via the submit
# directory (the package dir, which the relative paths below assume anyway)
# and only fall back to the script's own location when run directly.
if [ -n "$SLURM_SUBMIT_DIR" ]; then
    PACKAGE_DIR="$SLURM_SUBMIT_DIR"
else
    PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
COMPUTE_GRID_WEIGHTS="$PACKAGE_DIR/compute_grid_weights.py"

if [ ! -f "$COMPUTE_GRID_WEIGHTS" ]; then
    echo "compute_grid_weights.py not found at $COMPUTE_GRID_WEIGHTS" >&2
    exit 1
fi
echo "Using $COMPUTE_GRID_WEIGHTS"


# Setup paths
NETCDF_GRID="/p/projects/climate_data_central/reanalysis/ERA5/2m_temperature/2m_temperature_200010.nc"
SHAPEFILE="/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/DOSE_REGIONS.gpkg"
WEIGHTS_OUTPUT="data/ERA5_to_DOSE_admin1_weights.csv"

# Settings for parallel computation
NUMBER_OF_JOBS=16 # Number of parallel jobs set to the number of CPU cores

# Run the Python script
python "$COMPUTE_GRID_WEIGHTS" $NETCDF_GRID $SHAPEFILE $WEIGHTS_OUTPUT --yes --n_jobs=$NUMBER_OF_JOBS --shift_longitudes

if [ $? -ne 0 ]; then
    echo "compute_grid_weights failed; not validating." >&2
    exit 1
fi

# Check the freshly computed weights before anything downstream consumes them.
# Errors (regions missing entirely, coordinates off the grid, weights out of
# range) exit non-zero and fail the job; cell sums above the ceiling are only
# reported, since overlapping source polygons make them legitimate.
# Use the local package copy, matching the compute step above.
export PYTHONPATH="$(dirname "$PACKAGE_DIR")${PYTHONPATH:+:$PYTHONPATH}"
python -m netcdf2region.validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shift_longitudes \
    --max-cell-sum 2.0
