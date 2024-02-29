"""
Bigraph diagram
"""
import os
from bigraph_schema import TypeSystem, Edge
from bigraph_viz.plot_old import absolute_path, make_label, check_if_path_in_removed_nodes
import graphviz


PROCESS_SCHEMA_KEYS = [
    'config',
    'address',
    'interval',
    'inputs',
    'outputs', 
    'instance',
]

REMOVE_KEYS = ['global_time']

updated_path_type = {
        '_type': 'path',
        '_inherit': 'list[string]~string',
        # '_apply': apply_path
}

step_type = {
    '_type': 'step',
    '_inherit': 'edge',
    'address': 'string',
    # '_default': {
    #     'inputs': {},
    #     'outputs': {}},
    # '_inputs': 'string~tuple',
    # '_outputs': 'string~tuple',
    'config': 'schema'}


process_type = {
    '_type': 'process',
    '_inherit': 'step',
    'interval': 'float'}

def plot_edges(graph, edge, port_labels, port_label_size, state_node_spec):
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
            target_name, process_name,
            label=label,
            labelloc="t",
            fontsize=port_label_size)


def get_graph_wires(
        schema,     # the ports schema
        wires,      # the wires, from port to path
        graph_dict, # the current graph dict that is being built
        schema_key, # inputs or outputs
        edge_path,  # the path up to this process
        port,       # the port id
):
    """
    TODO -- support subwires with advanced wiring. This currently assumes each port has a simple wire.
    """
    for port, subschema in schema.items():
        wire = wires.get(port)
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
            # the wire is defined, add it to edges
            if isinstance(wire, str):
                wire = [wire]
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
        else:
            raise ValueError(f"Unexpected wire type: {wires}")

    return graph_dict


def get_graph_dict(
        schema,
        state,
        core,
        graph_dict=None,
        path=None,
        top_state=None,
        retain_type_keys=False,
        retain_process_keys=False,
        remove_nodes=None,
        show_process_schema_keys=None,
):
    path = path or ()
    top_state = top_state or state
    remove_nodes = remove_nodes or []
    show_process_schema_keys = show_process_schema_keys or []
    removed_process_keys = list(set(PROCESS_SCHEMA_KEYS) - set(show_process_schema_keys))

    # initialize bigraph
    graph_dict = graph_dict or {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'input_edges': [],
        'output_edges': [],
        'disconnected_input_edges': [],
        'disconnected_output_edges': [],
        'bridges': [],
    }

    for key, value in state.items():
        subschema = schema.get(key, schema)

        if key.startswith('_') and not retain_type_keys:
            continue

        subpath = path + (key,)
        if check_if_path_in_removed_nodes(subpath, remove_nodes):
            # skip node if path in removed_nodes
            continue

        node_spec = {
            'name': key,
            'path': subpath,
            'value': None,
            'type': None
        }

        is_edge = core.check('edge', value)
        if is_edge:  # this is a process/edge node
            if key in removed_process_keys and not retain_process_keys:
                continue

            graph_dict['process_nodes'].append(node_spec)

            # this is an edge, get its inputs and outputs
            input_wires = value.get('inputs', {})
            output_wires = value.get('outputs', {})
            input_schema = subschema.get('_inputs') or value.get('_inputs', {})
            output_schema = subschema.get('_outputs') or value.get('_outputs', {})

            # get the input and output wires
            graph_dict = get_graph_wires(
                input_schema, input_wires, graph_dict, schema_key='inputs', edge_path=subpath, port=())
            graph_dict = get_graph_wires(
                output_schema, output_wires, graph_dict, schema_key='outputs', edge_path=subpath, port=())

        else:  # this is a state node
            if key in REMOVE_KEYS:
                continue
            if not isinstance(value, dict):  # this is a leaf node
                node_spec['value'] = value
                node_spec['type'] = schema.get(key, {}).get('_type')
            else:
                # node_spec['value'] = str(value)
                node_spec['type'] = schema.get(key, {}).get('_type')
            graph_dict['state_nodes'].append(node_spec)

        if isinstance(value, dict):  # get subgraph
            if is_edge:
                removed_process_schema_keys = [subpath + (schema_key,) for schema_key in removed_process_keys]
                remove_nodes.extend(removed_process_schema_keys)

            graph_dict = get_graph_dict(
                schema=schema.get(key, schema),
                state=value,
                core=core,
                graph_dict=graph_dict,
                path=subpath,
                top_state=top_state,
                remove_nodes=remove_nodes
            )

            # get the place edge
            for node in value.keys():
                if node.startswith('_') and not retain_type_keys:
                    continue

                child_path = subpath + (node,)
                if check_if_path_in_removed_nodes(child_path, remove_nodes):
                    continue
                graph_dict['place_edges'].append({
                    'parent': subpath,
                    'child': child_path})

    return graph_dict


def get_graphviz_fig(
        graph_dict,
        label_margin='0.05',
        node_label_size='12pt',
        size='16,10',
        rankdir='TB',
        dpi='70',
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

    # node specs
    state_node_spec = {
        'shape': 'circle', 'penwidth': '2', 'margin': label_margin, 'fontsize': node_label_size}
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false', 'margin': label_margin, 'fontsize': node_label_size}
    hyper_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'dot', 'arrowsize': '0.5'}
    input_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0'}
    output_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'back'}

    # initialize graph
    graph = graphviz.Digraph(name='bigraph', engine='dot')
    graph.attr(size=size, overlap='false', rankdir=rankdir, dpi=dpi)

    # state nodes
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        node_path = node['path']
        node_name = str(node_path)
        node_names.append(node_name)

        # make the label
        label = node_path[-1]
        schema_label = None
        if show_values:
            if node.get('value'):
                if not schema_label:
                    schema_label = ''
                schema_label += f": {node['value']}"
        if show_types:
            if node.get('type'):
                if not schema_label:
                    schema_label = '<br/>'
                schema_label += f"[{node['type']}]"
        if schema_label:
            label += schema_label
        label = make_label(label)
        graph.node(node_name, label=label)

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
        graph.edge(parent_node, child_node)

    # input edges
    for edge in graph_dict['input_edges']:
        graph.attr('edge', **input_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec)

    # output edges
    for edge in graph_dict['output_edges']:
        graph.attr('edge', **output_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec)

    # disconnected input edges
    for edge in graph_dict['disconnected_input_edges']:
        process_path = edge['edge_path']
        port = edge['port']

        # add invisible node for port
        node_name2 = str(absolute_path(process_path, port))
        graph.node(node_name2, label='', style='invis', width='0')
        edge['target_path'] = node_name2

        graph.attr('edge', **input_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec)

    # disconnected output edges
    for edge in graph_dict['disconnected_output_edges']:
        process_path = edge['edge_path']
        port = edge['port']

        # add invisible node for port
        node_name2 = str(absolute_path(process_path, port))
        graph.node(node_name2, label='', style='invis', width='0')
        edge['target_path'] = node_name2

        graph.attr('edge', **output_edge_spec)
        plot_edges(graph, edge, port_labels, port_label_size, state_node_spec)

    # grouped nodes
    for group in node_groups:
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
        size='16,10',
        node_label_size='12pt',
        show_values=False,
        show_types=False,
        port_labels=True,
        port_label_size='10pt',
        rankdir='TB',
        print_source=False,
        dpi='70',
        label_margin='0.05',
        # show_process_schema=False,
        # collapse_processes=False,
        node_border_colors=None,
        node_fill_colors=None,
        node_groups=False,
        remove_nodes=None,
        invisible_edges=False,
        # mark_top=False,
        remove_process_place_edges=False,
        show_process_schema_keys=['interval'],
):
    # get kwargs dict and remove plotting-specific kwargs
    kwargs = locals()
    state = kwargs.pop('state')
    schema = kwargs.pop('schema')
    core = kwargs.pop('core')
    file_format = kwargs.pop('file_format')
    out_dir = kwargs.pop('out_dir')
    filename = kwargs.pop('filename')
    print_source = kwargs.pop('print_source')
    remove_nodes = kwargs.pop('remove_nodes')
    show_process_schema_keys = kwargs.pop('show_process_schema_keys')
    remaining_kwargs = dict(kwargs)

    # set defaults if none provided
    core = core or TypeSystem()
    schema = schema or {}

    core.register('path', updated_path_type)
    if not core.exists('step'):
        core.register('step', step_type)
    if not core.exists('process'):
        core.register('process', process_type)

    schema, state = core.complete(schema, state)

    # parse out the network
    graph_dict = get_graph_dict(
        schema=schema,
        state=state,
        core=core,
        remove_nodes=remove_nodes,
        show_process_schema_keys=show_process_schema_keys,
    )

    # make a figure
    graph = get_graphviz_fig(graph_dict, **remaining_kwargs)

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


def test_diagram_plot():
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
    plot_bigraph(cell, filename='bigraph_cell',
                 show_values=True,
                 show_types=True,
                 # port_labels=False,
                 # rankdir='BT',
                 # remove_nodes=[
                 #     ('cell', 'address',),
                 #     ('cell', 'config'),
                 #     ('cell', 'interval'),
                 # ]
                 )

def test_bio_schema():
    b = {
        'environment': {
            'cells': {
                'cell1': {
                    'nucleus': {
                        'transcription': {
                            '_type': 'process',
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
                # '_inputs': {
                #     'fields': 'array'
                # },
                'inputs': {
                    'fields': ['fields',]
                },
                'outputs': {
                    'fields': ['fields', ]
                }
            }
        }}

    plot_bigraph(b, filename='bioschema', show_process_schema_keys=[])

def test_input_output():
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
    plot_bigraph(flat_composite_spec, rankdir='RL', filename='flat_composite')

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
    plot_bigraph(processes_spec, rankdir='BT', filename='multiple_processes')

def test_nested_processes():
    nested_composite_spec = {
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
    plot_bigraph(nested_composite_spec,
                 # **plot_settings,
                 filename='nested_composite')

if __name__ == '__main__':
    test_diagram_plot()
    test_bio_schema()
    test_input_output()
    test_multi_processes()
    test_nested_processes()