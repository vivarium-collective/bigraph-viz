#!/usr/bin/env bash
# Local-development helper: install the canonical, frozen dependency
# tree (so versions match CI), then swap in editable sibling checkouts
# of bigraph-schema and process-bigraph.

set -e

uv sync --frozen

for sibling in ../bigraph-schema ../process-bigraph; do
    if [ -d "$sibling" ]; then
        echo "Linking editable: $sibling"
        uv pip install -e "$sibling" --no-deps --reinstall
    else
        echo "Skipping $sibling (not present)"
    fi
done

echo
echo "Active versions:"
uv pip list 2>/dev/null | grep -E '^(bigraph-schema|process-bigraph|bigraph-viz) '
