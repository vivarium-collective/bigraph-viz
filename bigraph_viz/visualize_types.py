import os
import inspect
import graphviz
import numpy as np

from itertools import islice
from bigraph_schema import TypeSystem, is_schema_key, hierarchy_depth
from bigraph_viz.dict_utils import absolute_path

# Constants
PROCESS_SCHEMA_KEYS = [
    'config', 'address', 'interval', 'inputs', 'outputs', 'instance', 'bridge']

# Utility: Label formatting
def make_label(label):
    """Wrap a label in angle brackets for Graphviz HTML rendering."""
    return f'<{label}>'

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

# Plot a labeled edge from a port to a process
def plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='false'):
    """
    Add an edge between a target (state node) and process node.
    If target not already rendered, add it to the graph.
    """
    process_name = str(edge['edge_path'])
    target_name = str(edge['target_path'])
    label = make_label(edge['port']) if port_labels else ''

    if target_name not in graph.body:
        label_text = make_label(edge['target_path'][-1])
        graph.node(target_name, label=label_text, **state_node_spec)

    with graph.subgraph(name=process_name) as sub:
        sub.edge(target_name, process_name, constraint=constraint, label=label,
                 labelloc="t", fontsize=port_label_size)

# Add a node to the graph with optional value/type
def add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits):
    """
    Add a state node to the Graphviz graph.

    Parameters:
        graph: The Graphviz object
        node: Dict representing the node
        state_node_spec: Style options
        show_values (bool): Whether to show the node value
        show_types (bool): Whether to show the node type
        significant_digits (int): Digits to round values
    """
    node_path = node['path']
    node_name = str(node_path)
    label = node_path[-1]
    label_info = ''

    if show_values and (val := node.get('value')) is not None:
        if isinstance(val, float):
            val = int(val) if val.is_integer() else round(val, significant_digits)
        label_info += f":{val}"

    if show_types and (typ := node.get('type')):
        label_info += f"<br/>[{typ if len(typ) <= 20 else '...'}]"

    full_label = make_label(label + label_info) if label_info else make_label(label)
    graph.attr('node', **state_node_spec)
    graph.node(node_name, label=full_label)
    return node_name


# make the Graphviz figure
def get_graphviz_fig(
    graph_dict,
    label_margin='0.05',
    node_label_size='12pt',
    process_label_size=None,
    size='16,10',
    rankdir='TB',
    aspect_ratio='auto',
    dpi='70',
    significant_digits=2,
    undirected_edges=False,
    show_values=False,
    show_types=False,
    port_labels=True,
    port_label_size='10pt',
    invisible_edges=None,
    remove_process_place_edges=False,
    node_border_colors=None,
    node_fill_colors=None,
    node_groups=None,
):
    """
    Generate a Graphviz Digraph from a graph_dict describing a simulation architecture.
    """
    invisible_edges = invisible_edges or []
    node_groups = node_groups or []
    process_label_size = process_label_size or node_label_size

    graph = graphviz.Digraph(name='bigraph', engine='dot')
    graph.attr(size=size, overlap='false', rankdir=rankdir, dpi=dpi, ratio=aspect_ratio, splines='true')

    # === Style Specifications ===
    state_node_spec = {
        'shape': 'circle', 'penwidth': '2', 'constraint': 'false',
        'margin': label_margin, 'fontsize': node_label_size
    }
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false',
        'margin': label_margin, 'fontsize': process_label_size
    }
    edge_styles = {
        'input': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'forward'},
        'output': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'back'},
        'bidirectional': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'both'},
        'place': {'arrowhead': 'none', 'penwidth': '2'}
    }

    if undirected_edges:
        for spec in edge_styles.values():
            spec['dir'] = 'none'

    node_names = []

    # === Add State Nodes ===
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        name = add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits)
        node_names.append(name)

    # === Add Process Nodes ===
    graph.attr('node', **process_node_spec)
    process_paths = []
    for node in graph_dict['process_nodes']:
        node_path = node['path']
        name = str(node_path)
        label = make_label(node_path[-1])
        graph.node(name, label=label)
        process_paths.append(node_path)
        node_names.append(name)

    # === Add Place Edges ===
    for edge in graph_dict['place_edges']:
        show_edge = not (
            (remove_process_place_edges and edge['child'] in process_paths) or
            (edge in invisible_edges)
        )
        style = 'filled' if show_edge else 'invis'
        graph.attr('edge', style=style)
        graph.edge(str(edge['parent']), str(edge['child']), **edge_styles['place'], constraint='true')

    # === Add Input/Output/Bidirectional Edges ===
    for edge_group, style_key in [('input_edges', 'input'), ('output_edges', 'output'), ('bidirectional_edges', 'bidirectional')]:
        for edge in graph_dict[edge_group]:
            if 'bridge_outputs' in edge['type']:
                style = 'input'
                constraint = 'false'
            elif 'bridge_inputs' in edge['type']:
                style = 'output'
                constraint = 'false'
            else:
                style = style_key
                constraint = 'true'
            graph.attr('edge', **edge_styles[style])
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint=constraint)

    # === Re-Apply State Node Style (Hack for correct shape) ===
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        name = add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits)
        node_names.append(name)

    # === Disconnected Port Edges ===
    for direction, style_key in [('disconnected_input_edges', 'input'), ('disconnected_output_edges', 'output')]:
        for edge in graph_dict[direction]:
            process_path = edge['edge_path']
            port = edge['port']
            suffix = '_input' if 'input' in direction else '_output'
            dummy_name = str(absolute_path(process_path, port)) + suffix
            graph.node(dummy_name, label='', style='invis', width='0')
            edge['target_path'] = dummy_name
            graph.attr('edge', **edge_styles[style_key])
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')

    # === Grouped Nodes ===
    for group in node_groups:
        group = [tuple(g) for g in group]
        with graph.subgraph(name=str(group)) as sg:
            sg.attr(rank='same')
            prev = None
            for path in group:
                name = str(path)
                if name in node_names:
                    sg.node(name)
                    if prev:
                        sg.edge(prev, name, style='invis', ordering='out')
                    prev = name

    # === Apply Node Colors ===
    if node_border_colors:
        for name, color in node_border_colors.items():
            graph.node(str(name), color=color)
    if node_fill_colors:
        for name, color in node_fill_colors.items():
            graph.node(str(name), color=color, style='filled')

    return graph


def plot_bigraph(
    state,
    schema=None,
    core=None,
    out_dir=None,
    filename=None,
    file_format='png',
    **kwargs
):
    """
    Create and render a bigraph visualization using Graphviz from a given state and optional schema.

    Parameters:
        state (dict): The simulation state.
        schema (dict): Optional schema defining the structure of the state.
        core (VisualizeTypes): Visualization engine.
        out_dir (str): Directory to write output.
        filename (str): Name of the output file.
        file_format (str): Output format (e.g., 'png', 'svg').
        **kwargs: Additional arguments for styling or traversal.

    Returns:
        graphviz.Digraph: Rendered graph object.
    """
    # Separate kwargs into rendering and traversal arguments
    graphviz_sig = inspect.signature(get_graphviz_fig)
    render_kwargs = {k: v for k, v in kwargs.items() if k in graphviz_sig.parameters}
    traversal_kwargs = {k: v for k, v in kwargs.items() if k not in graphviz_sig.parameters}

    # Defaults
    core = core or VisualizeTypes()
    schema = schema or {}
    schema, state = core.generate(schema, state)

    graph_dict = core.generate_graph_dict(schema, state, (), options=traversal_kwargs)

    return core.plot_graph(
        graph_dict,
        filename=filename,
        out_dir=out_dir,
        file_format=file_format,
        options=render_kwargs
    )


# Visualize Types
def graphviz_any(core, schema, state, path, options, graph):
    """Visualize any type (generic node)."""
    schema = schema or {}

    if path:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': state if not isinstance(state, dict) else None,
            'type': core.representation(schema)
        }
        graph['state_nodes'].append(node_spec)

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                graph = core.get_graph_dict(
                    schema.get(key, {}),
                    value,
                    path + (key,),
                    options,
                    graph
                )

    return graph

def graphviz_edge(core, schema, state, path, options, graph):
    """Visualize a process node with input/output/bridge wiring."""
    node_spec = {
        'name': path[-1],
        'path': path,
        'value': None,
        'type': core.representation(schema)
    }

    if state.get('address') == 'local:composite' and node_spec not in graph['process_nodes']:
        graph['process_nodes'].append(node_spec)
        return graphviz_composite(core, schema, state, path, options, graph)

    graph['process_nodes'].append(node_spec)

    # Wiring
    graph = get_graph_wires(schema.get('_inputs', {}), state.get('inputs', {}), graph, 'inputs', path, state.get('bridge', {}).get('inputs', {}))
    graph = get_graph_wires(schema.get('_outputs', {}), state.get('outputs', {}), graph, 'outputs', path, state.get('bridge', {}).get('outputs', {}))

    # Merge bidirectional edges
    def key(edge): return (tuple(edge['edge_path']), tuple(edge['target_path']), edge['port'])
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


def graphviz_none(core, schema, state, path, options, graph):
    """No-op visualizer for nodes with no visualization."""
    return graph


def graphviz_composite(core, schema, state, path, options, graph):
    """Visualize composite nodes by recursing into their internal structure."""
    graph = graphviz_edge(core, schema, state, path, options, graph)

    inner_state = state.get('config', {}).get('state') or state
    inner_schema = state.get('config', {}).get('composition') or schema
    inner_schema, inner_state = core.generate(inner_schema, inner_state)

    if len(path) > 1:
        graph['place_edges'].append({'parent': path[:-1], 'child': path})

    for key, value in inner_state.items():
        if not is_schema_key(key) and key not in PROCESS_SCHEMA_KEYS:
            graph = core.get_graph_dict(
                inner_schema.get(key),
                value,
                path + (key,),
                options,
                graph
            )

    return graph


# dict with different types and their graphviz functions
visualize_types = {
    'any': {
        '_graphviz': graphviz_any
    },
    'edge': {
        '_graphviz': graphviz_edge
    },
    'quote': {
        '_graphviz': graphviz_none,
    },
    'step': {
        '_inherit': ['edge']
    },
    'process': {
        '_inherit': ['edge']
    },
    'composite': {
        '_inherit': ['process'],
        '_graphviz': graphviz_composite,
    },
}

# TODO: we want to visualize things that are not yet complete

class VisualizeTypes(TypeSystem):
    def __init__(self):
        super().__init__()

        self.update_types(visualize_types)

    def get_graph_dict(self, schema, state, path, options, graph=None):
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

        graphviz_function = self.choose_method(
            schema,
            state,
            'graphviz')

        if options.get('remove_nodes') and path in options['remove_nodes']:
            return graph

        return graphviz_function(
            self,
            schema,
            state,
            path,
            options,
            graph)

    def generate_graph_dict(self, schema, state, path, options):
        full_schema, full_state = self.generate(schema, state)
        return self.get_graph_dict(full_schema, full_state, path, options)

    def plot_graph(self,
                   graph_dict,
                   out_dir='out',
                   filename=None,
                   file_format='png',
                   print_source=False,
                   options=None
                   ):
        # make a figure
        options = options or {}
        graph = get_graphviz_fig(
            graph_dict,
            **options)

        # display or save results
        if print_source:
            print(graph.source)

        if filename is not None:
            out_dir = out_dir or 'out'
            os.makedirs(out_dir, exist_ok=True)
            fig_path = os.path.join(out_dir, filename)
            print(f"Writing {fig_path}")
            graph.render(filename=fig_path, format=file_format)

        return graph


# Begin Tests
###############

plot_settings = {
    # 'out_dir': 'out',
    'dpi': '150',
}


def test_simple_store():
    simple_store_state = {
        'store1': 1.0,
    }
    plot_bigraph(simple_store_state,
                 **plot_settings,
                 show_values=True,
                 filename='simple_store')


def test_forest():
    forest = {
        'v0': {
            'v1': {},
            'v2': {
                'v3': {}
            },
        },
        'v4': {
            'v5': {},
        },
    }
    plot_bigraph(forest, **plot_settings, filename='forest')


def test_nested_composite():
    state = {
        'environment': {
            '0': {
                'mass': 1.0,
                'grow_divide': {
                    '_type': 'process',
                    'inputs': {
                        'mass': ['mass']},
                    'outputs': {
                        'mass': ['mass'],
                        'environment': ['..']},
                    'interval': 1.0,
                    'address': 'local:composite',
                    'config': {'_type': 'quote',
                               'state': {'grow': {'_type': 'process',
                                                  'address': 'local:grow',
                                                  'config': {'rate': 0.03},
                                                  'inputs': {'mass': ['mass']},
                                                  'outputs': {'mass': ['mass']}},
                                         'divide': {'_type': 'process',
                                                    'address': 'local:divide',
                                                    'config': {'agent_id': '0',
                                                               'agent_schema': {'mass': 'float'},
                                                               'threshold': 2.0,
                                                               'divisions': 2},
                                                    'inputs': {'trigger': ['mass']},
                                                    'outputs': {'environment': ['environment']}},
                                         'global_time': 0.0},
                               'bridge': {'inputs': {'mass': ['mass']},
                                          'outputs': {'mass': ['mass'],
                                                      'environment': ['environment']}},
                               'composition': {'global_time': 'float'},
                               'interface': {'inputs': {}, 'outputs': {}},
                               'emitter': {'path': ['emitter'],
                                           'address': 'local:ram-emitter',
                                           'config': {},
                                           'mode': 'none',
                                           'emit': {}},
                               'global_time_precision': None}
                }}}}
    plot_bigraph(state,
                 filename='nested_composite',
                 **plot_settings)


def test_graphviz():
    cell = {
        'config': {
            '_type': 'map[float]',
            'a': 11.0,  # {'_type': 'float', '_value': 11.0},
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            # 'config': {},
            # 'address': 'local:cell',   # TODO -- this is where the ports/inputs/outputs come from
            'internal': 1.0,
            '_inputs': {
                'nutrients': 'float',
            },
            '_outputs': {
                'secretions': 'float',
                'biomass': 'float',
            },
            'inputs': {
                'nutrients': ['down', 'nutrients_store'],
            },
            'outputs': {
                # 'secretions': ['secretions_store'],
                'biomass': ['biomass_store'],
            }
        }
    }

    core = VisualizeTypes()
    graph_dict = core.generate_graph_dict(
        {},
        cell,
        (),
        options={'dpi': '150'})

    core.plot_graph(
        graph_dict,
        out_dir='out',
        filename='test_graphviz'
    )


def test_bigraph_cell():
    cell = {
        'config': {
            '_type': 'map[float]',
            'a': 11.0,  # {'_type': 'float', '_value': 11.0},
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            'config': {},
            'address': 'local:cell',  # TODO -- this is where the ports/inputs/outputs come from
            'internal': 1.0,
            '_inputs': {
                'nutrients': 'float',
            },
            '_outputs': {
                'secretions': 'float',
                'biomass': 'float',
            },
            'inputs': {
                'nutrients': ['down', 'nutrients_store'],
            },
            'outputs': {
                # 'secretions': ['secretions_store'],
                'biomass': ['biomass_store'],
            }
        }
    }

    plot_bigraph(cell,
                 filename='bigraph_cell',
                 show_values=True,
                 # show_types=True,
                 **plot_settings
                 )


def test_bio_schema():
    core = VisualizeTypes()
    b = {
        'environment': {
            'cells': {
                'cell1': {
                    'cytoplasm': {},
                    'nucleus': {
                        'chromosome': {},
                        'transcription': {
                            '_type': 'process',
                            '_inputs': {'DNA': 'any'},
                            '_outputs': {'RNA': 'any'},
                            'inputs': {
                                'DNA': ['chromosome']
                            },
                            'outputs': {
                                'RNA': ['..', 'cytoplasm']
                            }
                        }
                    },

                }
            },
            'fields': {},
            'barriers': {},
            'diffusion': {
                '_type': 'process',
                '_inputs': {'fields': 'any'},
                '_outputs': {'fields': 'any'},
                'inputs': {
                    'fields': ['fields', ]
                },
                'outputs': {
                    'fields': ['fields', ]
                }
            }
        }}

    plot_bigraph(b, core=core, filename='bio_schema', show_process_schema_keys=[],
                 **plot_settings)


def test_flat_composite():
    flat_composite_spec = {
        'store1.1': 'float',
        'store1.2': 'int',
        'process1': {
            '_type': 'process',
            'outputs': {
                'port1': ['store1.1'],
                'port2': ['store1.2'],
            }
        },
        'process2': {
            '_type': 'process',
            '_inputs': {
                'port1': 'any',
                'port2': 'any',
            },
            'inputs': {
                'port1': ['store1.1'],
                'port2': ['store1.2'],
            }
        },
    }
    plot_bigraph(flat_composite_spec,
                 rankdir='RL',
                 filename='flat_composite',
                 **plot_settings)


def test_multi_processes():
    process_schema = {
        '_type': 'process',
        '_inputs': {
            'port1': 'Any',
        },
        '_outputs': {
            'port2': 'Any'
        },
    }

    processes_spec = {
        'process1': process_schema,
        'process2': process_schema,
        'process3': process_schema,
    }
    plot_bigraph(processes_spec,
                 rankdir='BT',
                 filename='multiple_processes',
                 **plot_settings)


def test_nested_processes():
    nested_process_spec = {
        'store1': {
            'store1.1': 'float',
            'store1.2': 'int',
            'process1': {
                '_type': 'process',
                'inputs': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            },
            'process2': {
                '_type': 'process',
                'outputs': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            },
        },
        'process3': {
            '_type': 'process',
            'inputs': {
                'port1': ['store1'],
            }
        }
    }
    plot_bigraph(nested_process_spec,
                 **plot_settings,
                 filename='nested_processes')


def test_cell_hierarchy():
    core = VisualizeTypes()

    core.register('concentrations', 'float')
    core.register('sequences', 'float')
    core.register('membrane', {
        'transporters': 'concentrations',
        'lipids': 'concentrations',
        'transmembrane transport': {
            '_type': 'process',
            '_inputs': {},
            '_outputs': {
                'transporters': 'concentrations',
                'internal': 'concentrations',
                'external': 'concentrations'
            }
        }
    })

    core.register('cytoplasm', {
        'metabolites': 'concentrations',
        'ribosomal complexes': 'concentrations',
        'transcript regulation complex': {
            'transcripts': 'concentrations'},
        'translation': {
            '_type': 'process',
            '_outputs': {
                'p1': 'concentrations',
                'p2': 'concentrations'}}})

    core.register('nucleoid', {
        'chromosome': {
            'genes': 'sequences'}})

    core.register('cell', {
        'membrane': 'membrane',
        'cytoplasm': 'cytoplasm',
        'nucleoid': 'nucleoid'})

    # state
    cell_struct_state = {
        'cell': {
            'membrane': {
                'transmembrane transport': {
                    'outputs': {
                        'transporters': ['transporters'],
                        'internal': ['..', 'cytoplasm', 'metabolites']}}},
            'cytoplasm': {
                'translation': {
                    'outputs': {
                        'p1': ['ribosomal complexes'],
                        'p2': ['transcript regulation complex', 'transcripts']}}}}}

    plot_bigraph(
        cell_struct_state,
        schema={'cell': 'cell'},
        core=core,
        filename='cell_hierarchy',
        **plot_settings)


def test_multiple_disconnected_ports():
    core = VisualizeTypes()

    spec = {
        'process': {
            '_type': 'process',
            '_inputs': {
                'port1': 'Any',
                'port2': 'Any',
            },
            '_outputs': {
                'port1': 'Any',
                'port2': 'Any',
            },
        },
    }

    plot_bigraph(
        spec,
        core=core,
        # out_dir='out',
        filename='multiple_disconnected_ports',
        **plot_settings)


def test_composite_process():
    core = VisualizeTypes()

    spec = {
        'composite': {
            '_type': 'composite',
            '_inputs': {'port1': 'any'},
            '_outputs': {'port2': 'any'},
            'inputs': {'port1': ['external store']},
            'store1': 'any',
            'store2': 'any',
            'bridge': {
                'inputs': {'port1': ['store1']},
                'outputs': {'port2': ['store2']}},
            'process1': {
                '_type': 'process',
                '_inputs': {'port3': 'any'},
                '_outputs': {'port4': 'any', },
                'inputs': {'port3': ['store1']},
                'outputs': {'port4': ['store2']}}}}

    plot_bigraph(
        spec,
        core=core,
        filename='composite_process',
        **plot_settings)


def test_bidirectional_edges():
    core = VisualizeTypes()

    spec = {
        'process1': {
            '_type': 'process',
            '_inputs': {'port1': 'any'},
            '_outputs': {'port1': 'any'},
            'inputs': {'port1': ['external store']},
            'outputs': {'port1': ['external store']}},
        'process2': {
            '_type': 'process',
            '_inputs': {'port3': 'any'},
            '_outputs': {'port4': 'any'},
            'inputs': {'port3': ['external store']},
            'outputs': {'port4': ['external store']}
        }
    }

    plot_bigraph(
        spec,
        core=core,
        filename='bidirectional_edges',
        **plot_settings)

def generate_spec_and_schema(n_rows, n_cols):
    spec = {}
    fields = {
        'acetate': np.zeros((n_rows, n_cols)),
        'biomass': np.zeros((n_rows, n_cols)),
        'glucose': np.zeros((n_rows, n_cols)),
    }

    for i in range(n_rows):
        for j in range(n_cols):
            name = f'dFBA[{i},{j}]'
            cell_spec = {
                '_type': 'process',
                'address': 'local:DynamicFBA',
                'inputs': {
                    'substrates': {
                        'acetate': ['fields', 'acetate', i, j],
                        'biomass': ['fields', 'biomass', i, j],
                        'glucose': ['fields', 'glucose', i, j],
                    }
                },
                'outputs': {
                    'substrates': {
                        'acetate': ['fields', 'acetate', i, j],
                        'biomass': ['fields', 'biomass', i, j],
                        'glucose': ['fields', 'glucose', i, j],
                    }
                }
            }
            spec[name] = cell_spec

    # Add fields to spec
    spec['fields'] = fields

    # Generate schema
    schema = {
        'fields': {
            mol: {
                '_type': 'array',
                '_shape': (n_rows, n_cols),
                '_data': 'float'
            } for mol in ['acetate', 'biomass', 'glucose']
        }
    }

    return spec, schema

def test_array_paths():
    core = VisualizeTypes()

    n_rows, n_cols = 2, 1  # or any desired shape
    spec, schema = generate_spec_and_schema(n_rows, n_cols)

    plot_bigraph(
        spec,
        schema=schema,
        core=core,
        filename='array_paths',
        **plot_settings)


def test_complex_bigraph():
    core = VisualizeTypes()

    n_rows, n_cols = 10, 10 # or any desired shape
    spec, schema = generate_spec_and_schema(n_rows, n_cols)

    plot_bigraph(
        spec,
        schema=schema,
        core=core,
        filename='complex_bigraph',
        max_nodes_per_row=10,
        **plot_settings)


if __name__ == '__main__':
    test_simple_store()
    test_forest()
    test_nested_composite()
    test_graphviz()
    test_bigraph_cell()
    test_bio_schema()
    test_flat_composite()
    test_multi_processes()
    test_nested_processes()
    test_cell_hierarchy()
    test_multiple_disconnected_ports()
    test_composite_process()
    test_bidirectional_edges()
    test_array_paths()
    test_complex_bigraph()
