#!/usr/bin/env python
"""Generate the Notebooks section for README.md from notebooks.json.

Usage:
    python notebooks/generate_readme.py          # print to stdout
    python notebooks/generate_readme.py --update  # update README.md in place
"""

import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOKS_DIR = REPO_ROOT / 'notebooks'
REGISTRY_PATH = NOTEBOOKS_DIR / 'notebooks.json'
README_PATH = REPO_ROOT / 'README.md'

PAGES_BASE = 'https://vivarium-collective.github.io/bigraph-viz/notebooks'
GITHUB_BASE = 'https://github.com/vivarium-collective/bigraph-viz/blob/main/notebooks'

BEGIN_MARKER = '<!-- BEGIN NOTEBOOKS -->'
END_MARKER = '<!-- END NOTEBOOKS -->'


def load_registry():
    raw = REGISTRY_PATH.read_text()
    # Strip trailing commas before ] or } to tolerate editor auto-formatting
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    return json.loads(raw)


def discover_unlisted(registry):
    """Find .ipynb files not mentioned in the registry."""
    listed = {e['name'] for e in registry.get('notebooks', [])}
    unlisted = []
    for path in sorted(NOTEBOOKS_DIR.glob('*.ipynb')):
        if path.stem not in listed:
            unlisted.append(path.stem)
    return unlisted


def pretty_name(name):
    return name.replace('_', ' ').replace('-', ' ').title()


def generate_section():
    registry = load_registry()
    unlisted = discover_unlisted(registry)

    lines = [
        BEGIN_MARKER,
        '## Notebooks',
        '',
        f'All notebooks are tested in CI and published as HTML to GitHub Pages: '
        f'**[Browse all notebooks]({PAGES_BASE}/)**',
        '',
        '| Notebook | Description |',
        '|----------|-------------|',
    ]

    for entry in registry.get('notebooks', []):
        name = entry['name']
        desc = entry.get('description', '')
        if entry.get('skip'):
            link = f'[{pretty_name(name)}]({GITHUB_BASE}/{name}.ipynb)'
            desc = f'{desc} *(source only)*' if desc else '*(source only)*'
        else:
            link = f'[{pretty_name(name)}]({PAGES_BASE}/{name}.html)'
        lines.append(f'| {link} | {desc} |')

    for name in unlisted:
        link = f'[{pretty_name(name)}]({GITHUB_BASE}/{name}.ipynb)'
        lines.append(f'| {link} | |')

    lines.append(END_MARKER)
    return '\n'.join(lines)


def update_readme():
    section = generate_section()
    readme = README_PATH.read_text()

    if BEGIN_MARKER in readme and END_MARKER in readme:
        before = readme[:readme.index(BEGIN_MARKER)]
        after = readme[readme.index(END_MARKER) + len(END_MARKER):]
        readme = before + section + after
    else:
        # Insert before ## License or at the end
        if '## License' in readme:
            idx = readme.index('## License')
            readme = readme[:idx] + section + '\n\n' + readme[idx:]
        else:
            readme = readme.rstrip() + '\n\n' + section + '\n'

    README_PATH.write_text(readme)
    print(f'Updated {README_PATH}')


if __name__ == '__main__':
    if '--update' in sys.argv:
        update_readme()
    else:
        print(generate_section())
