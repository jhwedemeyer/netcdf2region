# Contributing to netcdf2region

Thanks for your interest in `netcdf2region`! Bug reports, questions, documentation fixes and code contributions are all welcome.

By taking part in this project you agree to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting help

- **Questions about usage**: open an [issue](https://github.com/jhwedemeyer/netcdf2region/issues) and describe what you are trying to do.
- **Bug reports**: open an issue and include
  - the `netcdf2region` version (`pip show netcdf2region`) and Python version,
  - the exact command or Python code you ran,
  - the full error message or the unexpected output,
  - if possible, a small input file or a description of the grid and regions involved.
- **Feature requests**: open an issue describing the use case before starting on code, so we can agree on the approach.

## Contributing code

1. Fork the repository and create a branch from `main`.
2. Install the package in editable mode with the test dependencies:

   ```bash
   pip install -e .[test]
   ```

3. Make your change. Please
   - add or update tests in `tests/` for any change in behaviour,
   - update the README if you change a command-line option or the Python API,
   - add a short entry under `[Unreleased]` in [CHANGELOG.md](CHANGELOG.md).
4. Run the test suite:

   ```bash
   pytest
   ```

   Tests marked `cluster` need input data on the PIK cluster and are skipped elsewhere.
5. Open a pull request against `main`. Describe what the change does and why. CI runs the tests on Python 3.11 and 3.12 and must pass before merging.

## Releases

Releases follow [Semantic Versioning](https://semver.org/). To release, the maintainer bumps the version in `pyproject.toml` and `CITATION.cff`, moves the `[Unreleased]` entries in the changelog to the new version, and pushes a `v*` tag. The release workflow then publishes the package to PyPI and archives it on Zenodo.

## Project governance and support

`netcdf2region` is maintained by Jakob Wedemeyer ([@jhwedemeyer](https://github.com/jhwedemeyer), Potsdam Institute for Climate Impact Research). The maintainer reviews issues and pull requests and makes the final decision on what is merged and released.

The project is maintained on a best-effort basis alongside research work. We aim to respond to new issues and pull requests within few weeks. Bug fixes that affect the correctness of aggregated results take priority.

For matters that should not be discussed in public, contact the maintainer at jakob.wedemeyer [at] pik-potsdam.de.
