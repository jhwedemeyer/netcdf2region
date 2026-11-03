# Changelog

All notable changes to `netcdf2region` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.10] - 2026-10-26

### Added
- Exact, area-aware grid weights (`compute_grid_weights --method exact`), now the default method.
- `--area-weighted` option for `netcdf2region` to weight grid cells by covered area instead of covered fraction only.
- `compute_grid_weights` validates the weights it writes by default.
- Precomputed exact weights are shipped with the package.
- Optional NaN handling for aggregation, split into two separate flags (contributed by Anna Reckwitz).
- GitHub Actions workflows for tests (Python 3.11 and 3.12) and for releases to PyPI and Zenodo.
- Logo, CI and DOI badges in the README.

### Changed
- CLI options use hyphens (e.g. `--area-weighted` instead of `--area_weighted`).
- Weight computation and loading split into separate modules.
- Aggregation runs over blocks of time steps to keep memory use bounded.
- Grids finer than 0.001° are rejected.
- The land-sea mask was removed from weight computation.
- Zenodo citation now uses the concept DOI.

### Fixed
- Exact weights ignore floating-point slivers (overlaps below 1e-9 of a grid cell) along shared region borders; regions smaller than this threshold are kept.
- Improved precision handling of longitude/latitude coordinates.
- Improved error handling, including for custom region identifiers (GIDs).

## [0.1.9] - 2026-08-20

### Added
- `validate_weights` module and command-line tool.
- Pytest suite covering weight computation and aggregation.
- SLURM scripts to regenerate the precomputed weights.
- Check for the correct longitude shift in `netcdf2region`.
- Citation file, license and acknowledgements.

### Changed
- Strict NaN handling: aggregation fails on NaNs unless explicitly allowed with a flag.
- Regenerated and validated the precomputed weights.

### Fixed
- Four defects in weight computation and aggregation.
- Handling of very small regions and very small weights.
- The validation step now uses the checked-out script instead of an installed copy.

## Earlier versions (0.1.0 – 0.1.8)

Untagged development releases between August 2025 and July 2026. Highlights:

- Initial release with `compute_grid_weights` and `netcdf2region` (2025-08-01).
- Precomputed weights for common grids and region sets.
- Coordinate origin option, progress bar and improved logging.
- Geo-identifier (GID) option for regions.
- Batch splitting with a default number of batches.
- Secondary weights.

[Unreleased]: https://github.com/jhwedemeyer/netcdf2region/compare/v0.1.10...HEAD
[0.1.10]: https://github.com/jhwedemeyer/netcdf2region/compare/v0.1.9...v0.1.10
[0.1.9]: https://github.com/jhwedemeyer/netcdf2region/releases/tag/v0.1.9
