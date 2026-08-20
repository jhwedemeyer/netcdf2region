.ONESHELL:
precalc_weights:
	cd netcdf2region
	sbatch precalc_weights/era5_to_DOSE_admin1.sh
	sbatch precalc_weights/era5_to_GADM36_admin0.sh
	sbatch precalc_weights/era5_to_GADM36_admin1.sh
	sbatch precalc_weights/isimip_to_DOSE_admin1.sh
	sbatch precalc_weights/isimip_to_GADM36_admin0.sh
	sbatch precalc_weights/isimip_to_GADM36_admin1.sh
