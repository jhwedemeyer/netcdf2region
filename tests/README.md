# netcdf2region test suite

Run it:

```bash
pip install -e ".[test]"          # or: pip install pytest pytest-cov
pytest                            # local suite, no cluster data needed (~11 s)
pytest -m ""                      # also collect the cluster tests (skipped off-cluster)
pytest --cov=netcdf2region --cov-report=term-missing
```

`pytest` alone deselects `-m cluster` via `addopts` in `pyproject.toml`. The
`slow` marker is *not* deselected — it only means "reads the full shipped weights"
and adds about a second.

## What is covered

| File | Covers |
|---|---|
| `test_load_grid.py` | `load_template_grid` and `load_data_grid` (incl. `--shift-longitudes`, naming variants, every error path) and the `Grid` cells |
| `test_load_regions.py` | `load_regions` |
| `test_sampled_weights.py` | `uniform_points_in_cell`, `random_points_in_cell`, `generate_point_distribution`, `calc_cell_region_overlap`, `assign_missing_regions` (the small-region rescue path) |
| `test_exact_weights.py` | exact fractions, cell areas, and area-weighted aggregation of the result |
| `test_compute_grid_weights.py` | `compute_weights` end to end, the `compute_grid_weights` CLI, and the golden regression check |
| `test_weights.py` | `correct_weights`, `precalc_weights_path`, `validate_secondary_weights` |
| `test_aggregate.py` | `aggregate` (mean/sum/std, secondary weights, longitude conventions) and the `netcdf2region` CLI |
| `test_validate_weights.py` | every rule in `netcdf2region/validate_weights.py`, plus its CLI and exit codes |
| `test_cli.py` | the option helpers the three commands share (hyphenated flags, hidden underscore aliases) |
| `test_shipped_data.py` | invariants and regression guards for the six shipped `data/*_weights.csv` |
| `test_cluster.py` | the real ERA5/ISIMIP/GADM/DOSE inputs — skipped unless `/p/projects/impactee` exists |

## The mini case

Everything local is built on a synthetic 6x8 grid of 1-degree cells (`tests/factories.py`).
With `points_per_cell=100` each cell is sampled by a 10x10 lattice at offsets
0.05, 0.15, ... 0.95, which makes every expected weight exact rather than recorded:

| Region | Geometry | Expected weight |
|---|---|---|
| `A` | covers cell (0, 0) exactly | 1.00, sampled |
| `B` | western half of cell (0, 2) | 0.50, sampled |
| `C` | straddles the corner of four cells | 0.25 in each, sampled |
| `D` | 0.02° sliver between sample points | 0.0004, **rescued** by exact geometry |
| `E` | 0.0005° speck, rounds to zero | 1e-6 floor, **rescued** |

`D` and `E` are the reason `assign_missing_regions` exists: point sampling can
never see them, and without the rescue path they would vanish from the output.

## Fixtures

`tests/data/` is committed, and regenerated with:

```bash
python tests/data/make_fixtures.py            # rewrite
python tests/data/make_fixtures.py --check    # verify byte-identical
```

`golden_mini_weights.csv` is the real regression guard: uniform sampling is
deterministic, so a fresh `compute_weights(method="sampling")` run must reproduce it byte for byte.
If it changes, something about sampling, rounding, or the rescue path changed.

`shipped_slice_ISIMIP_DOSE.csv` is a small cut of complete regions from
`ISIMIP_to_DOSE_admin1_weights.csv`, and `shipped_stats.json` records summary
statistics for all six shipped files. Both turn a regeneration of the shipped
weights into a readable diff rather than an opaque failure.

## Two properties worth knowing before you change a test

**Per-cell weights do not have to sum to 1.** `ERA5_to_DOSE_admin1_weights.csv`
has 13 697 cells above 1.01, peaking at 2.03: DOSE polygons overlap, and
cell (24.75, -14.25) is claimed in full by both `ESH.1_1` (Western Sahara) and
`MAR.26_1`. `validate_weights` therefore treats the cell sum as a *warning* with a
configurable ceiling. It also means `--correct-weights` scales those cells **down**.

**The shipped weights are exact.** All six files were computed with
`--method exact` (October 2026) and carry `cell_area_km2`;
`test_shipped_files_are_exact_weights` guards that. After rerunning
`netcdf2region/precalc_weights/*.sh`:

1. rerun `python tests/data/make_fixtures.py` and review the diff of
   `shipped_stats.json` and the slice,
2. run `pytest -m cluster` on the cluster — `test_shipped_weights_cover_every_region`
   is the acceptance gate.
