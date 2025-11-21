#!/usr/bin/env bash

# Create and publish a new version by creating and pushing a git tag for
# the version and publishing the version to PyPI. Also perform some
# basic checks to avoid mistakes in releases, for example tags not
# matching PyPI.
# Usage: ./release.sh 0.1.12

set -e

version="$1"

if [ -z "$version" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

# ----------------------------------------------------------------------
# Check version in pyproject.toml matches the requested version
# ----------------------------------------------------------------------
pyproject_version="$(python - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:
    import toml as tomllib  # fallback if tomllib is missing

import sys
from pathlib import Path

data = tomllib.loads(Path("pyproject.toml").read_text())
print(data["project"]["version"])
PY
)"

if [ "$pyproject_version" != "$version" ]; then
    echo "pyproject.toml has version $pyproject_version, not $version."
    echo "Aborting."
    exit 1
fi

# ----------------------------------------------------------------------
# Check working directory is clean (ignores untracked files)
# ----------------------------------------------------------------------
if [ -n "$(git status --untracked-files=no --porcelain)" ]; then
    echo "You have changes that have yet to be committed."
    echo "Aborting."
    exit 1
fi

# ----------------------------------------------------------------------
# Check that we are on main
# ----------------------------------------------------------------------
branch="$(git rev-parse --abbrev-ref HEAD)"
if [ "$branch" != "main" ]; then
    echo "You are on $branch but should be on main for releases."
    echo "Aborting."
    exit 1
fi

# ----------------------------------------------------------------------
# Create and push git tag
# ----------------------------------------------------------------------
git tag -m "Version v$version" "v$version"
git push --tags

# ----------------------------------------------------------------------
# Build and publish package using pyproject.toml
# ----------------------------------------------------------------------
rm -rf dist build *.egg-info

python -m build

# Optional but recommended: verify metadata / README rendering
twine check dist/*

# Upload to PyPI
twine upload dist/*

echo "Version v$version has been published on PyPI and has a git tag."
