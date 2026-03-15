import json
import subprocess
import sys

import pytest

from pathlib import Path


NOTEBOOKS_DIR = Path(__file__).resolve().parent.parent / 'notebooks'
REGISTRY_PATH = NOTEBOOKS_DIR / 'notebooks.json'


def get_notebook_list():
    """Return list of notebook filenames to test from notebooks.json,
    plus any .ipynb files in the notebooks dir not listed in the registry."""
    registered = []
    skip_names = set()

    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        for entry in registry.get('notebooks', []):
            name = entry['name']
            if entry.get('skip'):
                skip_names.add(name)
            else:
                registered.append(f'{name}.ipynb')

    # Auto-discover notebooks not in the registry
    registered_names = {Path(nb).stem for nb in registered} | skip_names
    for path in sorted(NOTEBOOKS_DIR.glob('*.ipynb')):
        if path.stem not in registered_names:
            registered.append(path.name)

    return registered


NOTEBOOKS = get_notebook_list()


@pytest.mark.parametrize('notebook', NOTEBOOKS)
def test_notebook_runs(notebook):
    path = NOTEBOOKS_DIR / notebook
    assert path.exists(), f'Notebook not found: {path}'
    result = subprocess.run(
        [
            sys.executable, '-m', 'jupyter', 'nbconvert',
            '--to', 'notebook',
            '--execute',
            '--ExecutePreprocessor.timeout=120',
            '--stdout',
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f'Notebook {notebook} failed:\n{result.stderr}'
    )
