from plum import dispatch

from bigraph_schema.schema import (
    Node,
    Atom,
    Union,
    Tuple,
    Boolean,
    Or,
    And,
    Xor,
    Number,
    Integer,
    Float,
    Delta,
    Nonnegative,
    String,
    Enum,
    Wrap,
    Maybe,
    Overwrite,
    List,
    Map,
    Tree,
    Array,
    Key,
    Path,
    Wires,
    Schema,
    Link,
)


@dispatch
def graphviz(core, schema, state, path, options, graph):
    import ipdb; ipdb.set_trace()


def generate_graph_dict(core, schema, state, path=(), options=None, graph=None):
    path = path or ()
    graph = graph or {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'input_edges': [],
        'output_edges': [],
        'bidirectional_edges': [],
        'disconnected_input_edges': [],
        'disconnected_output_edges': []}

    graphviz(
        core,
        schema,
        state,
        path,
        options,
        graph)
    
