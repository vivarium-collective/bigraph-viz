import pprint
from bigraph_viz.plot import plot_bigraph, plot_flow, plot_multitimestep
from bigraph_viz.dict_utils import pp, pf, schema_state_to_dict
from bigraph_viz.convert import convert_vivarium_composite


pretty = pprint.PrettyPrinter(indent=2)


def pp(x):
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x):
    """Format ``x`` for display."""
    return pretty.pformat(x)