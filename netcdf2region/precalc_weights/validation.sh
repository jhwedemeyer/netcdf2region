#!/bin/bash 
#SBATCH --qos=short
#SBATCH --account=impactee
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1 
#SBATCH --cpus-per-task=16
#SBATCH --mem=60000
#SBATCH --output=temp/weights-valid-%j.out
#SBATCH --error=temp/weights-valid-%j.err

module purge
module load anaconda
source ~/.venvs/netcdf2region-dev/bin/activate

## Testing ERA5 grids
NETCDF_GRID="/p/projects/climate_data_central/reanalysis/ERA5/2m_temperature/2m_temperature_200010.nc"
SHAPEFILE="/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/DOSE_REGIONS.gpkg"
WEIGHTS_OUTPUT="data/ERA5_to_DOSE_admin1_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shift-longitudes \
    --max-cell-sum 2.0


SHAPEFILE="/p/tmp/lochner/gadm36_levels.gpkg"
SHAPEFILE_LAYER=0
WEIGHTS_OUTPUT="data/ERA5_to_GADM36_admin0_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shapefile-layer=$SHAPEFILE_LAYER \
    --shift-longitudes \
    --max-cell-sum 1.03

SHAPEFILE_LAYER=1
WEIGHTS_OUTPUT="data/ERA5_to_GADM36_admin1_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shapefile-layer=$SHAPEFILE_LAYER \
    --shift-longitudes \
    --max-cell-sum 1.03

# Testing isimip grids

NETCDF_GRID="/p/projects/isimip/isimip/ISIMIP2b/InputData/gdp/rcp26soc/gdp_rcp26soc_0p5deg_annual_2006-2099.nc4"
SHAPEFILE="/p/projects/impactee/DOSE_creation/DoseV2Creation/shapefile/DOSE_REGIONS.gpkg"
WEIGHTS_OUTPUT="data/ISIMIP_to_DOSE_admin1_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --max-cell-sum 2.0

SHAPEFILE="/p/tmp/lochner/gadm36_levels.gpkg"
SHAPEFILE_LAYER=0
WEIGHTS_OUTPUT="data/ISIMIP_to_GADM36_admin0_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shapefile-layer=$SHAPEFILE_LAYER \
    --max-cell-sum 1.03

SHAPEFILE_LAYER=1
WEIGHTS_OUTPUT="data/ISIMIP_to_GADM36_admin1_weights.csv"
validate_weights "$WEIGHTS_OUTPUT" \
    --netcdf "$NETCDF_GRID" \
    --shapefile "$SHAPEFILE" \
    --shapefile-layer=$SHAPEFILE_LAYER \
    --max-cell-sum 1.03
