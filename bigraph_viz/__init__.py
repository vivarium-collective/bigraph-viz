import pprint
# from bigraph_viz.visualize_types import VisualizeTypes, plot_bigraph, get_graphviz_fig
from bigraph_viz.visualize_types import plot_bigraph, get_graphviz_fig
from bigraph_viz.dict_utils import replace_regex_recursive
from bigraph_viz.methods.generate_graph_dict import generate_graph_dict


pretty = pprint.PrettyPrinter(indent=2)


def pf(x):
    return pretty.pformat(x)



def register_types(core):
    core.register_method('generate_graph_dict', generate_graph_dict)

    return core
