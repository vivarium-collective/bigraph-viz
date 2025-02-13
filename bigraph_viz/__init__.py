import pprint
from bigraph_viz.visualize_types import VisualizeTypes, plot_bigraph, get_graphviz_fig
from bigraph_viz.dict_utils import replace_regex_recursive


pretty = pprint.PrettyPrinter(indent=2)


def pf(x):
    return pretty.pformat(x)
