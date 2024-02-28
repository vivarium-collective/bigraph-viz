import pprint
from bigraph_viz.diagram import plot_bigraph
from bigraph_viz.dict_utils import replace_regex_recursive
from bigraph_schema import TypeSystem


pretty = pprint.PrettyPrinter(indent=2)


def pf(x):
    return pretty.pformat(x)
