import os
import inspect
import graphviz
import numpy as np

from bigraph_schema import TypeSystem, is_schema_key, hierarchy_depth
from bigraph_viz.dict_utils import absolute_path


PROCESS_SCHEMA_KEYS = [
    'config',
    'address',
    'interval',
    'inputs',
    'outputs',
    'instance',
    'bridge',
]


def make_label(label):
    # Insert line breaks after every max_length characters
    # max_length = 25
    # lines = [label[i:i+max_length] for i in range(0, len(label), max_length)]
    # label = '<br/>'.join(lines)
    return f'<{label}>'


def get_graph_wires(
        ports_schema,     # the ports schema
        wires,      # the wires, from port to path
        graph_dict, # the current graph dict that is being built
        schema_key, # inputs or outputs
        edge_path,  # the path up to this process
        bridge_wires=None,
):
    """
    TODO -- support subwires with advanced wiring. This currently assumes each port has a simple wire.
    """
    wires = wires or {}
    ports_schema = ports_schema or {}
    inferred_ports = set(list(ports_schema.keys()) + list(wires.keys()))

    for port in inferred_ports:
        wire = wires.get(port)
        bridge = bridge_wires.get(port) if bridge_wires else None

        if not wire:
            # there is no wire for this port, it is disconnected
            if schema_key == 'inputs':
                graph_dict['disconnected_input_edges'].append({
                    'edge_path': edge_path,
                    'port': port,
                    'type': schema_key})
            elif schema_key == 'outputs':
                graph_dict['disconnected_output_edges'].append({
                    'edge_path': edge_path,
                    'port': port,
                    'type': schema_key})

        elif isinstance(wire, (list, tuple, str)):
            graph_dict = get_single_wire(edge_path, graph_dict, port, schema_key, wire)
        elif isinstance(wire, dict):
            flat_wires = hierarchy_depth(wires)
            for subpath, subwire in flat_wires.items():
                subport = '/'.join(subpath)
                graph_dict = get_single_wire(edge_path, graph_dict, subport, schema_key, subwire)

        else:
            raise ValueError(f"Unexpected wire type: {wires}")

        if bridge:
            target_path = absolute_path(edge_path, tuple(bridge))
            if schema_key == 'inputs':
                graph_dict['input_edges'].append({
                    'edge_path': edge_path ,
                    'target_path': target_path,
                    'port': f'bridge_{port}',
                    'type': f'bridge_{schema_key}'})
            elif schema_key == 'outputs':
                graph_dict['output_edges'].append({
                    'edge_path': edge_path,
                    'target_path': target_path,
                    'port': f'bridge_{port}',
                    'type': f'bridge_{schema_key}'})

    return graph_dict


def get_single_wire(edge_path, graph_dict, port, schema_key, wire):
    # the wire is defined, add it to edges
    if isinstance(wire, str):
        wire = [wire]
    elif isinstance(wire, (list, tuple)):
        # only use strings in the wire
        # TODO -- make this more general so it only skips integers if they go into an array
        wire = [item for item in wire if isinstance(item, str)]

    target_path = absolute_path(edge_path[:-1], tuple(wire))  # TODO -- make sure this resolves ".."
    if schema_key == 'inputs':
        edge_key = 'input_edges'
    elif schema_key == 'outputs':
        edge_key = 'output_edges'
    else:
        raise Exception(f'invalid schema key {schema_key}')
    graph_dict[edge_key].append({
        'edge_path': edge_path,
        'target_path': target_path,
        'port': port,
        'type': schema_key})
    return graph_dict


def plot_edges(
        graph,
        edge,
        port_labels,
        port_label_size,
        state_node_spec,
        constraint='false',
):
    process_path = edge['edge_path']
    process_name = str(process_path)
    target_path = edge['target_path']
    port = edge['port']
    target_name = str(target_path)

    # place it in the graph
    if target_name not in graph.body:  # is the source node already in the graph?
        label = make_label(target_path[-1])
        graph.node(target_name, label=label, **state_node_spec)
    # port label
    label = ''
    if port_labels:
        label = make_label(port)
    with graph.subgraph(name=process_name) as c:
        c.edge(
            target_name,
            process_name,
            constraint=constraint,
            label=label,
            labelloc="t",
            fontsize=port_label_size)


def add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits):
    node_path = node['path']
    node_name = str(node_path)
    # make the label
    label = node_path[-1]
    schema_label = None
    if show_values:
        if node.get('value'):
            v = node['value']
            if isinstance(v, float):
                v = round(v, significant_digits)
                if v.is_integer():
                    v = int(v)
            if not schema_label:
                schema_label = ''
            schema_label += f":{v}"
    if show_types:
        if node.get('type'):
            if not schema_label:
                schema_label = '<br/>'
            ntype = node['type']
            if len(ntype) > 20:  # don't show the full type if it's too long
                ntype = '...'
            schema_label += f"[{ntype}]"
    if schema_label:
        label += schema_label
    label = make_label(label)

    graph.attr('node', **state_node_spec)
    graph.node(str(node_name), label=label)
    return node_name


def get_graphviz_fig(
        graph_dict,
        label_margin='0.05',
        node_label_size='12pt',
        process_label_size=None,
        size='16,10',
        rankdir='TB',
        aspect_ratio='auto', # 'compress', 'expand', 'auto', 'fill'
        dpi='70',
        significant_digits=2,
        undirected_edges=False,
        show_values=False,
        show_types=False,
        port_labels=True,
        port_label_size='10pt',
        invisible_edges=False,
        remove_process_place_edges=False,
        node_border_colors=None,
        node_fill_colors=None,
        node_groups=False,
):
    """make a graphviz figure from a graph_dict"""
    node_groups = node_groups or []
    node_names = []
    invisible_edges = invisible_edges or []
    process_label_size = process_label_size or node_label_size

    # node specs
    state_node_spec = {
        'shape': 'circle', 'penwidth': '2', 'constraint': 'false', 'margin': label_margin, 'fontsize': node_label_size}
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false', 'margin': label_margin, 'fontsize': process_label_size}
    input_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'forward'}
    output_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'back'}
    bidirectional_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'both'}

    if undirected_edges:
        input_edge_spec['dir'] = 'none'
        output_edge_spec['dir'] = 'none'
        bidirectional_edge_spec['dir'] = 'none'

    # initialize graph
    graph = graphviz.Digraph(name='bigraph', engine='dot')
    graph.attr(size=size, overlap='false', rankdir=rankdir, dpi=dpi,
               ratio=aspect_ratio,  # "fill",
               splines = 'true',
               )

    # state nodes
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        node_path = node['path']
        node_name = add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits)
        node_names.append(node_name)

    # process nodes
    process_paths = []
    graph.attr('node', **process_node_spec)
    for node in graph_dict['process_nodes']:
        node_path = node['path']
        process_paths.append(node_path)
        node_name = str(node_path)
        node_names.append(node_name)
        label = make_label(node_path[-1])
        graph.node(node_name, label=label)

    # place edges
    graph.attr('edge', arrowhead='none', penwidth='2')
    for edge in graph_dict['place_edges']:
        # show edge or not
        show_edge = True
        if remove_process_place_edges and edge['child'] in process_paths:
            show_edge = False
        elif edge in invisible_edges:
            show_edge = False
        if show_edge:
            graph.attr('edge', style='filled')
        else:
            graph.attr('edge', style='invis')
        parent_node = str(edge['parent'])
        child_node = str(edge['child'])
        graph.edge(parent_node, child_node,
                   dir='forward', constraint='true'
                   )

    # input edges
    for edge in graph_dict['input_edges']:
        if edge['type'] == 'bridge_inputs':
            graph.attr('edge', **output_edge_spec) # reverse arrow direction to go from composite to store
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='false')
        else:
            graph.attr('edge', **input_edge_spec)
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')
    # output edges
    for edge in graph_dict['output_edges']:
        if edge['type'] == 'bridge_outputs':
            graph.attr('edge', **input_edge_spec) # reverse arrow direction to go from store to composite
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='false')
        else:
            graph.attr('edge', **output_edge_spec)
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')
    # bidirectional edges
    for edge in graph_dict['bidirectional_edges']:
        if 'bridge_outputs' not in edge['type'] and 'bridge_inputs' not in edge['type']:
            graph.attr('edge', **bidirectional_edge_spec)
            plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')
        else:
            if 'bridge_outputs' in edge['type']:
                graph.attr('edge', **input_edge_spec) # reverse arrow direction to go from store to composite
                plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='false')
            if 'bridge_inputs' in edge['type']:
                graph.attr('edge', **output_edge_spec) # reverse arrow direction to go from composite to store
                plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='false')

    # state nodes again
    # TODO -- this is a hack to make sure the state nodes show up as circles
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        node_path = node['path']
        node_name = add_node_to_graph(graph, node, state_node_spec, show_values, show_types, significant_digits)
        node_names.append(node_name)

    # disconnected input edges
    for edge in graph_dict['disconnected_input_edges']:
        process_path = edge['edge_path']
        port = edge['port']
        # add invisible node for port
        node_name2 = str(absolute_path(process_path, port)) + '_input'
        graph.node(node_name2, label='', style='invis', width='0')
        edge['target_path'] = node_name2
        graph.attr('edge', **input_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')
    # disconnected output edges
    for edge in graph_dict['disconnected_output_edges']:
        process_path = edge['edge_path']
        port = edge['port']
        # add invisible node for port
        node_name2 = str(absolute_path(process_path, port)) + '_output'
        graph.node(node_name2, label='', style='invis', width='0')
        edge['target_path'] = node_name2
        graph.attr('edge', **output_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec, constraint='true')

    # grouped nodes
    for group in node_groups:
        # convert lists to tuples
        group = [tuple(item) for item in group]

        group_name = str(group)
        with graph.subgraph(name=group_name) as c:
            c.attr(rank='same')
            previous_node = None
            for path in group:
                node_name = str(path)
                if node_name in node_names:
                    c.node(node_name)
                    if previous_node:
                        # out them in the order declared in the group
                        c.edge(previous_node, node_name, style='invis', ordering='out')
                    previous_node = node_name
                else:
                    print(f'node {node_name} not in graph')
    # formatting
    if node_border_colors:
        for node_name, color in node_border_colors.items():
            graph.node(str(node_name), color=color)
    if node_fill_colors:
        for node_name, color in node_fill_colors.items():
            graph.node(str(node_name), color=color, style='filled')
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
    # inspect the signature of plot_bigraph
    get_graphviz_fig_signature = inspect.signature(get_graphviz_fig)

    # Filter kwargs to only include those accepted by get_graphviz_fig
    get_graphviz_kwargs = {
        k: v for k, v in kwargs.items()
        if k in get_graphviz_fig_signature.parameters}

    # get the remaining kwargs
    viztype_kwargs = {
        k: v for k, v in kwargs.items()
        if k not in get_graphviz_kwargs}

    # set defaults if none provided
    core = core or VisualizeTypes()
    schema = schema or {}
    schema, state = core.generate(schema, state)

    graph_dict = core.generate_graph_dict(
        schema,
        state,
        (),
        options=viztype_kwargs   # TODO
    )

    return core.plot_graph(
        graph_dict,
        filename=filename,
        out_dir=out_dir,
        file_format=file_format,
        options=get_graphviz_kwargs)


# Visualize Types
def graphviz_any(core, schema, state, path, options, graph):
    schema = schema or {}
    if len(path) > 0:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': None,
            'type': core.representation(schema)}

        if not isinstance(state, dict):
            node_spec['value'] = state

        graph['state_nodes'].append(node_spec)

    if len(path) > 1:
        graph['place_edges'].append({
            'parent': path[:-1],
            'child': path})

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                subpath = path + (key,)                
                graph = core.get_graph_dict(
                    schema.get(key, {}),
                    value,
                    subpath,
                    options,
                    graph)

    return graph
        

def graphviz_edge(core, schema, state, path, options, graph):
    # add process node to graph
    node_spec = {
        'name': path[-1],
        'path': path,
        'value': None,
        'type': core.representation(schema)}

    # check if this is actually a composite node
    if state.get('address') == 'local:composite' and node_spec not in graph['process_nodes']:
        graph['process_nodes'].append(node_spec)
        return graphviz_composite(core, schema, state, path, options, graph)

    graph['process_nodes'].append(node_spec)

    # get the wires and ports
    input_wires = state.get('inputs', {})
    output_wires = state.get('outputs', {})
    input_ports = state.get('_inputs', schema.get('_inputs', {}))
    output_ports = state.get('_outputs', schema.get('_outputs', {}))

    # bridge
    bridge_wires = state.get('bridge', {})
    bridge_inputs = bridge_wires.get('inputs', {})
    bridge_outputs = bridge_wires.get('outputs', {})

    # get the input wires
    graph = get_graph_wires(
        input_ports, input_wires, graph,
        schema_key='inputs', edge_path=path,
        bridge_wires=bridge_inputs)

    # get the output wires
    graph = get_graph_wires(
        output_ports, output_wires, graph,
        schema_key='outputs', edge_path=path,
        bridge_wires=bridge_outputs)

    # get bidirectional wires
    input_edges_to_remove = []
    output_edges_to_remove = []
    for input_edge in graph['input_edges']:
        for output_edge in graph['output_edges']:
            if (input_edge['target_path'] == output_edge['target_path']) and \
                    (input_edge['port'] == output_edge['port']) and \
                    (input_edge['edge_path'] == output_edge['edge_path']):

                graph['bidirectional_edges'].append({
                    'edge_path': input_edge['edge_path'],
                    'target_path': input_edge['target_path'],
                    'port': input_edge['port'],
                    'type': (input_edge['type'], output_edge['type']),
                    # 'type': 'bidirectional'
                })
                input_edges_to_remove.append(input_edge)
                output_edges_to_remove.append(output_edge)
                break  # prevent matching the same input_edge with multiple output_edges

    # Remove matched edges after iteration
    for edge in input_edges_to_remove:
        graph['input_edges'].remove(edge)

    for edge in output_edges_to_remove:
        graph['output_edges'].remove(edge)

    # get the input and output bridge wires
    if bridge_wires:
        # check that the bridge wires connect to valid ports
        assert set(bridge_wires.keys()).issubset({'inputs', 'outputs'})

    # add the process node path
    if len(path) > 1:
        graph['place_edges'].append({
            'parent': path[:-1],
            'child': path})

    return graph

def graphviz_none(core, schema, state, path, options, graph):
    return graph

def graphviz_composite(core, schema, state, path, options, graph):
    # add the composite edge
    graph = graphviz_edge(core, schema, state, path, options, graph)

    # get the inner state and schema
    inner_state = state.get('config', {}).get('state')
    inner_schema = state.get('config', {}).get('composition')
    if inner_state is None:
        inner_state = state
        inner_schema = schema
    inner_schema, inner_state = core.generate(inner_schema, inner_state)

    # add the process node path
    if len(path) > 1:
        graph['place_edges'].append({
            'parent': path[:-1],
            'child': path})

    # add the inner nodes and edges
    for key, value in inner_state.items():
        if not is_schema_key(key) and key not in PROCESS_SCHEMA_KEYS:
            subpath = path + (key,)
            graph = core.get_graph_dict(
                inner_schema.get(key),
                value,
                subpath,
                options,
                graph)

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
            'a': 11.0,  #{'_type': 'float', '_value': 11.0},
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
            'a': 11.0,  #{'_type': 'float', '_value': 11.0},
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            'config': {},
            'address': 'local:cell',   # TODO -- this is where the ports/inputs/outputs come from
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
                    'fields': ['fields',]
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
                '_outputs': {'port4': 'any',},
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

def test_array_paths():
    core = VisualizeTypes()

    spec = {
        'dFBA[0,0]': {
            '_type': 'process',
            'address': 'local:DynamicFBA',
            # 'config': {},
            'inputs': {
                'substrates': {
                    'acetate': ['fields', 'acetate',
                                0, 0
                                ],
                    'biomass': ['fields', 'biomass',
                                0, 0
                                ],
                    'glucose': ['fields', 'glucose',
                                0, 0
                                ]}},
            'outputs': {
                'substrates': {
                    'acetate': ['fields', 'acetate',
                                0, 0
                                ],
                    'biomass': ['fields', 'biomass',
                                0, 0
                                ],
                    'glucose': ['fields', 'glucose',
                                0, 0
                                ]}},
        },
        'dFBA[1,0]': {
            '_type': 'process',
            'address': 'local:DynamicFBA',
            # 'config': {},
            'inputs': {
                'substrates': {
                    'acetate': ['fields', 'acetate',
                                1, 0
                                ],
                    'biomass': ['fields', 'biomass',
                                1, 0
                                ],
                    'glucose': ['fields', 'glucose',
                                1, 0
                                ]}},
            'outputs': {
                'substrates': {
                    'acetate': ['fields', 'acetate',
                                1, 0
                                ],
                    'biomass': ['fields', 'biomass',
                                1, 0
                                ],
                    'glucose': ['fields', 'glucose',
                                1, 0
                                ]}},
        },
        'fields': {
            'acetate': np.array([[1.0], [2.0]]),
            'biomass': np.array([[3.0],[4.0]]),
            'glucose': np.array([[5.0],[6.0]])
        }
    }

    schema = {
        'fields': {
            'acetate': {
                '_type': 'array',
                '_shape': (2, 1),
                '_data': 'float'
            },
            'biomass': {
                '_type': 'array',
                '_shape': (2, 1),
                '_data': 'float',
            },
            'glucose': {
                '_type': 'array',
                '_shape': (2, 1),
                '_data': 'float',
            }
        }
    }

    plot_bigraph(
        spec,
        schema=schema,
        core=core,
        filename='array_paths',
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
