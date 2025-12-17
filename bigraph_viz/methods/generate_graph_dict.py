from plum import dispatch

from bigraph_schema import is_schema_key, hierarchy_depth
from bigraph_schema.schema import (
    Node,
    Atom,
    Empty,
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

from process_bigraph import CompositeLink

from bigraph_viz.dict_utils import absolute_path


PROCESS_SCHEMA_KEYS = [
    'config', 'address', 'interval', 'inputs', 'outputs', 'instance', 'bridge']

# Append a single port wire connection to graph_dict

def get_single_wire(edge_path, graph_dict, port, schema_key, wire):
    """
    Add a connection from a port to its wire target.

    Parameters:
        edge_path (tuple): Path to the process
        graph_dict (dict): Current graph dict
        port (str): Name of the port
        schema_key (str): Either 'inputs' or 'outputs'
        wire (str|list): Wire connection(s)

    Returns:
        Updated graph_dict
    """
    if isinstance(wire, str):
        wire = [wire]
    else:
        wire = [item for item in wire if isinstance(item, str)]

    target_path = absolute_path(edge_path[:-1], tuple(wire))
    edge_key = 'input_edges' if schema_key == 'inputs' else 'output_edges'
    graph_dict[edge_key].append({
        'edge_path': edge_path,
        'target_path': target_path,
        'port': port,
        'type': schema_key
    })
    return graph_dict


def get_graph_wires(ports_schema, wires, graph_dict, schema_key, edge_path, bridge_wires=None):
    """
    Traverse the port wiring and append wire edges or disconnected ports to graph_dict.

    Parameters:
        ports_schema (dict): Schema for ports (inputs or outputs)
        wires (dict): Wiring structure from the process
        graph_dict (dict): Accumulated graph
        schema_key (str): Either 'inputs' or 'outputs'
        edge_path (tuple): Path of the process node
        bridge_wires (dict, optional): Optional rewiring via 'bridge' dict

    Returns:
        graph_dict (dict): Updated graph dict
    """
    wires = wires or {}
    ports_schema = ports_schema or {}
    inferred_ports = set(ports_schema.keys()) | set(wires.keys())

    for port in inferred_ports:
        wire = wires.get(port)
        bridge = bridge_wires.get(port) if bridge_wires else None

        if not wire:
            # If not connected, mark as disconnected
            edge_type = 'disconnected_input_edges' if schema_key == 'inputs' else 'disconnected_output_edges'
            graph_dict[edge_type].append({
                'edge_path': edge_path,
                'port': port,
                'type': schema_key
            })
        elif isinstance(wire, (list, tuple, str)):
            graph_dict = get_single_wire(edge_path, graph_dict, port, schema_key, wire)
        elif isinstance(wire, dict):
            for subpath, subwire in hierarchy_depth(wires).items():
                subport = '/'.join(subpath)
                graph_dict = get_single_wire(edge_path, graph_dict, subport, schema_key, subwire)
        else:
            raise ValueError(f"Unexpected wire type: {wires}")

        # Handle optional bridge wiring
        if bridge:
            target_path = absolute_path(edge_path, tuple(bridge))
            edge_key = 'input_edges' if schema_key == 'inputs' else 'output_edges'
            graph_dict[edge_key].append({
                'edge_path': edge_path,
                'target_path': target_path,
                'port': f'bridge_{port}',
                'type': f'bridge_{schema_key}'
            })

    return graph_dict


def graphviz_map(core, schema, state, path, options, graph):
    """Visualize mappings by traversing key–value pairs."""

    value_type = schema._value
    # value_type = core._find_parameter(schema, 'value')

    # Add node for the map container itself
    if path:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': None,
            'type': core.render(schema)
        }
        graph['state_nodes'].append(node_spec)

    # Add place edge to parent
    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                graph = core.call_method('generate_graph_dict',
                    value_type,
                    value,
                    path + (key,),
                    options,
                    graph
                )

    return graph


def graphviz_link(core, schema: Link, state, path, options, graph):
    """Visualize a process node with input/output/bridge wiring."""
    schema = schema or {}
    node_spec = {
        'name': path[-1],
        'path': path,
        'value': None,
        'type': core.render(schema)
    }

    if state.get('address') == 'local:Composite' and node_spec not in graph['process_nodes']:
        graph['process_nodes'].append(node_spec)
        return graphviz_composite(core, schema, state, path, options, graph)

    graph['process_nodes'].append(node_spec)

    # Wiring
    graph = get_graph_wires(schema._inputs, state.get('inputs', {}), graph, 'inputs', path,
                            state.get('bridge', {}).get('inputs', {}))
    graph = get_graph_wires(schema._outputs, state.get('outputs', {}), graph, 'outputs', path,
                            state.get('bridge', {}).get('outputs', {}))
    # # Wiring
    # graph = get_graph_wires(schema.get('_inputs', {}), state.get('inputs', {}), graph, 'inputs', path,
    #                         state.get('bridge', {}).get('inputs', {}))
    # graph = get_graph_wires(schema.get('_outputs', {}), state.get('outputs', {}), graph, 'outputs', path,
    #                         state.get('bridge', {}).get('outputs', {}))

    # Merge bidirectional edges
    def key(edge):
        return (tuple(edge['edge_path']), tuple(edge['target_path']), edge['port'])

    input_set = {key(e): e for e in graph['input_edges']}
    output_set = {key(e): e for e in graph['output_edges']}
    shared_keys = input_set.keys() & output_set.keys()
    for k in shared_keys:
        graph['bidirectional_edges'].append({
            'edge_path': k[0], 'target_path': k[1], 'port': k[2],
            'type': (input_set[k]['type'], output_set[k]['type'])
        })
    graph['input_edges'] = [e for k, e in input_set.items() if k not in shared_keys]
    graph['output_edges'] = [e for k, e in output_set.items() if k not in shared_keys]

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    return graph

def graphviz_composite(core, schema, state, path, options, graph):
    """Visualize composite nodes by recursing into their internal structure."""
    graph = graphviz_link(core, schema, state, path, options, graph)

    inner_state = state.get('config', {}).get('state', {}) # or state
    inner_schema = state.get('config', {}).get('composition', {}) # or schema
    inner_schema, inner_state = core.deserialize(inner_schema, inner_state)
    # inner_schema, inner_state = core.generate(inner_schema, inner_state)

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    for key, value in inner_state.items():
        if not is_schema_key(key) and key not in PROCESS_SCHEMA_KEYS:
            graph = core.call_method('generate_graph_dict',
                inner_schema.get(key),
                value,
                path + (key,),
                options,
                graph
            )

    return graph

def graphviz_node(core, schema: Node, state, path, options, graph):
    """Visualize any type (generic node)."""
    schema = schema or {}

    if path:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': state if not isinstance(state, dict) else None,
            'type': core.render(schema)
        }
        graph['state_nodes'].append(node_spec)

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                attr = Empty()
                if hasattr(schema, key):
                    attr = getattr(schema, key)

                graph = core.call_method('generate_graph_dict',
                    attr,
                    value,
                    path + (key,),
                    options,
                    graph
                )

    return graph


def graphviz_dict(core, schema, state, path, options, graph):
    """Visualize any type (generic node)."""
    schema = schema or {}

    if path:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': state if not isinstance(state, dict) else None,
            'type': core.render(schema)
        }
        graph['state_nodes'].append(node_spec)

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                graph = core.call_method('generate_graph_dict',
                    schema.get(key, {}),
                    value,
                    path + (key,),
                    options,
                    graph
                )

    return graph


@dispatch
def graphviz(core, schema: Empty, state, path, options, graph):
    """No-op visualizer for nodes with no visualization."""
    return graph

@dispatch
def graphviz(core, schema: Map, state, path, options, graph):
    return graphviz_map(core, schema, state, path, options, graph)

@dispatch
def graphviz(core, schema: Link, state, path, options, graph):
    return graphviz_link(core, schema, state, path, options, graph)

@dispatch
def graphviz(core, schema: CompositeLink, state, path, options, graph):
    return graphviz_composite(core, schema, state, path, options, graph)

@dispatch
def graphviz(core, schema: Node, state, path, options, graph):
    return graphviz_node(core, schema, state, path, options, graph)

@dispatch
def graphviz(core, schema: dict, state, path, options, graph):
    return graphviz_dict(core, schema, state, path, options, graph)

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

    if schema is None:
        schema = Empty()

    return graphviz(
        core,
        schema,
        state,
        path,
        options,
        graph)
    
