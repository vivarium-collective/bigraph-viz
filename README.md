# Bigraph-viz

[![PyPI](https://img.shields.io/pypi/v/bigraph-viz.svg)](https://pypi.org/project/bigraph-viz/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Tutorial-brightgreen)](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html)

**Bigraph-viz** is an easy-to-use plotting tool for compositional bigraph schema, built on top of Graphviz. 

<p align="center">
    <img src="https://github.com/vivarium-collective/bigraph-viz/blob/main/doc/_static/cell_structure_function.png?raw=true" width="600" alt="Bigraph-viz example">
</p>

## Compositional Schemas and Milner Bigraphs

Compositional Bigraph Schemas (CBS) are based on a mathematical formalism introduced by 
<a href="https://www.google.com/search?q=the+space+and+motion+of+communicating+agents+by+robin+milner" target="_blank">Robin Milner in 2009</a>
, and used in Vivarium. 
Bigraphs consist of networks with embeddable nodes that can be placed *within* other nodes and dynamically restructured.
CBS reimagines the bigraph concept. Variables are contained within Stores (the circles in the figure), which can be embedded
in a hierarchy, as shown by the dark edges. Instead of Milner's hyperedges, CBS employs Processes (the rectangles) which 
connect via wires (dashed edges) to shared variables within the Stores. Processes and Stores form a type of bipartite graph, 
as illustrated by the dashed edges.

In Vivarium, the CBS is used to structure a modular multiscale simulation. Bigraph-viz is part of an effort to
standardize this structure, so that it can be used as an exchange format for multiscale simulations.

## Getting Started


### Dependencies

Before installing `bigraph-viz`, make sure you have [Graphviz](https://pypi.org/project/graphviz/).

#### Ubuntu/Debian

```console
sudo apt-get install -y graphviz
```
#### macOS

```console
brew install graphviz
```

#### Windows
Download and install the Graphviz installer from the [official website](https://graphviz.org/download/). 
Make sure to add the Graphviz bin directory to your system's PATH.

### Installation

Once Graphviz is installed, you can install `bigraph-viz` using pip:

```console
pip install bigraph-viz
```

<!-- BEGIN NOTEBOOKS -->
## Notebooks

All notebooks are tested in CI and published as HTML to GitHub Pages: **[Browse all notebooks](https://vivarium-collective.github.io/bigraph-viz/notebooks/)**

| Notebook | Description |
|----------|-------------|
| [Basics](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html) | Tutorial covering stores, processes, wires, and composites |
| [Basics2](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics2.html) | Extended basics with cell structure example |
| [Basics Pres](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics_pres.html) | Presentation-style examples |
| [Bigraph Example](https://vivarium-collective.github.io/bigraph-viz/notebooks/bigraph_example.html) | Transcription/translation and multicellular models |
| [Cell Atlas](https://vivarium-collective.github.io/bigraph-viz/notebooks/cell_atlas.html) | Kidney glomerulus model with OpenVT processes |
| [Ecoli](https://vivarium-collective.github.io/bigraph-viz/notebooks/ecoli.html) | Whole-cell E. coli wiring diagram |
| [Ecoli2](https://vivarium-collective.github.io/bigraph-viz/notebooks/ecoli2.html) | E. coli with plasmid replication |
| [Ecoli2026](https://vivarium-collective.github.io/bigraph-viz/notebooks/ecoli2026.html) | E. coli biomanufacturing model |
| [Format](https://vivarium-collective.github.io/bigraph-viz/notebooks/format.html) | Styling options for bigraph figures |
| [Gut Microbiome](https://vivarium-collective.github.io/bigraph-viz/notebooks/gut_microbiome.html) | Multi-region gut model with cdFBA-CRM |
| [Ccb](https://github.com/vivarium-collective/bigraph-viz/blob/main/notebooks/ccb.ipynb) | CCB model *(source only)* |
| [Ecoli Biomanufacturing](https://github.com/vivarium-collective/bigraph-viz/blob/main/notebooks/ecoli_biomanufacturing.ipynb) | E. coli biomanufacturing with WCM *(source only)* |
<!-- END NOTEBOOKS -->

## License

Bigraph-viz is open-source software released under the [Apache 2 License](https://github.com/vivarium-collective/bigraph-viz/blob/main/LICENSE).
