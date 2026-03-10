#!/usr/bin/env bash
set -e

# Check that the version in pyproject.toml differs from the latest on PyPI
LOCAL_VERSION=$(grep '^version' pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
PYPI_VERSION=$(curl -s https://pypi.org/pypi/netcdf2region/json | python3 -c "import sys,json; print(json.load(sys.stdin)['info']['version'])" 2>/dev/null || echo "none")

echo "Local version:  $LOCAL_VERSION"
echo "PyPI version:   $PYPI_VERSION"

if [ "$LOCAL_VERSION" = "$PYPI_VERSION" ]; then
    echo "Version $LOCAL_VERSION is already published on PyPI."
    read -rp "Auto-increment patch version? (y/N) " ANSWER
    if [[ "$ANSWER" =~ ^[Yy]$ ]]; then
        IFS='.' read -r MAJOR MINOR PATCH <<< "$LOCAL_VERSION"
        NEW_VERSION="$MAJOR.$MINOR.$((PATCH + 1))"
        sed -i "s/^version = \"$LOCAL_VERSION\"/version = \"$NEW_VERSION\"/" pyproject.toml
        LOCAL_VERSION="$NEW_VERSION"
        echo "Version bumped to $LOCAL_VERSION"
    else
        echo "Aborting. Please bump the version in pyproject.toml manually."
        exit 1
    fi
fi

echo "Cleaning previous builds..."
rm -rf dist/ build/ *.egg-info/

echo "Building package..."
python -m build

echo "Uploading to PyPI..."
# Use the twine executable directly so it works regardless of which venv is active
TWINE=$(command -v twine || true)
if [ -z "$TWINE" ]; then
    echo "Error: twine not found. Install it with: pip install twine"
    exit 1
fi
"$TWINE" upload dist/*

echo "Done! Package published to PyPI."
