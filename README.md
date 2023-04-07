# Bigraph-viz

[![PyPI](https://img.shields.io/pypi/v/bigraph-viz.svg)](https://pypi.org/project/bigraph-viz/)
[![GitHub Pages](https://img.shields.io/badge/GitHub%20Pages-Tutorial-brightgreen)](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html)

**Bigraph-viz** is an easy-to-use plotting tool for compositional bigraph schema, built on top of Graphviz. 

<p align="center">
    <img src="https://github.com/vivarium-collective/bigraph-viz/blob/main/doc/_static/nested_composite.png?raw=true" width="340" alt="Bigraph-viz example">
</p>

## Compositional Schemas and Milner Bigraphs

Compositional Bigraph Schemas (CIP) are based on a mathematical formalism introduced by 
[Robin Milner in 2009](https://www.google.com/search?q=the+space+and+motion+of+communicating+agents+by+robin+milner), and
used in Vivarium. 
Bigraphs consist of networks with embeddable nodes that can be placed *within* other nodes and dynamically restructured.
CIP reimagines the bigraph concept. Variables are contained within Stores (the circles in the figure), which can be embedded
in a hierarchy, as shown by the dark edges with inner nodes depicted spatially beneath the outer nodes.
Instead of Milner's hyperedges, CIP employs Processes (the rectangles), which connect 
to shared variables within the Stores. Processes and Stores form a type of bipartite graph, as illustrated by the 
dashed edges. Collapsing the Process into a vertex of a hyperedge would result in Milner's bigraph.


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

## Tutorial

To get started with Bigraph-viz, explore the [Bigraph Schema Basics](https://vivarium-collective.github.io/bigraph-viz/notebooks/basics.html) tutorial.
