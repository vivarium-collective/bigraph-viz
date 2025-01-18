import os

from bigraph_schema import TypeSystem, is_schema_key
from bigraph_viz.dict_utils import absolute_path

import graphviz

REMOVE_KEYS = ['global_time']


def make_label(label):
    # label = label.replace(' ', '<br/>')  # replace spaces with new lines
    return f'<{label}>'


def check_if_path_in_removed_nodes(path, remove_nodes):
    if remove_nodes:
        return any(remove_path == path[:len(remove_path)] for remove_path in remove_nodes)
    return False


def get_graph_wires(
        schema,     # the ports schema
        wires,      # the wires, from port to path
        graph_dict, # the current graph dict that is being built
        schema_key, # inputs or outputs
        edge_path,  # the path up to this process
        bridge_wires=None,
):
    """
    TODO -- support subwires with advanced wiring. This currently assumes each port has a simple wire.
    """
    for port, subschema in schema.items():
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


def graphviz_any(core, schema, state, path, options, graph):
    if len(path) > 0:
        node_spec = {
            'name': path[-1],
            'path': path,
            'value': None,
            'type': core.representation(schema)}

        if not isinstance(state, dict):
            node_spec['value'] = state

        graph['state_nodes'].append(node_spec)

    if isinstance(state, dict):
        for key, value in state.items():
            if not is_schema_key(key):
                subpath = path + (key,)                
                graph = core.get_graph_dict(
                    schema.get(key),
                    value,
                    subpath,
                    options,
                    graph)

    return graph
        

def graphviz_edge(core, schema, state, path, options, graph):
    node_spec = {
        'name': path[-1],
        'path': path,
        'value': None,
        'type': core.representation(schema)}

            # if key in removed_process_keys and not retain_process_keys:
            #     continue

    graph['process_nodes'].append(node_spec)

    # this is an edge, get its inputs and outputs
    input_wires = state.get('inputs', {})
    output_wires = state.get('outputs', {})
    input_schema = state.get('_inputs', schema.get('_inputs', {}))
    output_schema = state.get('_outputs', schema.get('_outputs', {}))

    # bridge
    bridge_wires = state.get('bridge', {})
    bridge_inputs = bridge_wires.get('inputs', {})
    bridge_outputs = bridge_wires.get('outputs', {})

    # get the input and output wires
    graph = get_graph_wires(
        input_schema, input_wires, graph,
        schema_key='inputs', edge_path=path, bridge_wires=bridge_inputs)

    graph = get_graph_wires(
        output_schema, output_wires, graph,
        schema_key='outputs', edge_path=path, bridge_wires=bridge_outputs)

    # get the input and output bridge wires
    if bridge_wires:
        # check that the bridge wires connect to valid ports
        assert set(bridge_wires.keys()).issubset({'inputs', 'outputs'})

    return graph


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
    # TODO -- not sure this is working, it might be remaking the node
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
    # hyper_edge_spec = {
    #     'style': 'dashed', 'penwidth': '1', 'arrowhead': 'dot', 'arrowsize': '0.5'}
    input_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal', 'arrowsize': '1.0', 'dir': 'forward'}
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
        graph.node(str(node_name), label=label)

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
        show_process_schema_keys=[],  # ['interval']
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
    core = core or generate_types()
    schema = schema or {}
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


visualize_types = {
    'any': {
        '_graphviz': graphviz_any},
    'edge': {
        '_graphviz': graphviz_edge},
    'step': {
        '_inherit': ['edge']},
    'process': {
        '_inherit': ['edge']}}


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
            'disconnected_input_edges': [],
            'disconnected_output_edges': []}

        graphviz_function = self.choose_method(
            schema,
            state,
            'graphviz')

        return graphviz_function(
            self,
            schema,
            state,
            path,
            options,
            graph)
        

    def generate_graphviz(self, schema, state, path, options):
        full_schema, full_state = self.generate(schema, state)
        return self.get_graph_dict(full_schema, full_state, path, options)


    def plot_graph(self,
                   graph_dict,
                   out_dir='out',
                   filename=None,
                   file_format='png',
                   print_source=False,
                   graph_options={}
                   ):
        # make a figure
        graph = get_graphviz_fig(
            graph_dict,
            **graph_options)

        # display or save results
        if print_source:
            print(graph.source)

        if filename is not None:
            os.makedirs(out_dir, exist_ok=True)
            fig_path = os.path.join(out_dir, filename)
            print(f"Writing {fig_path}")
            graph.render(filename=fig_path, format=file_format)


    # def render_graphviz(self, schema, state, path, options):
    #     graph = self.get_graph_dict(
    #         schema,
    #         state,
    #         path,
    #         options['graphviz'])
    # 
    #     self.plot_bigraph(
    #         graph,
    #         options['plot'])



# Begin Tests
###############


plot_settings = {'out_dir': 'out',
                 'dpi': '150',}

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

    graph_settings = {
        'plot': {
            'out_dir': 'out',
            'filename': 'test_graphviz'},
        'graphviz': {
            'dpi': '150'}}


    core = VisualizeTypes()
    graphviz = core.generate_graphviz(
        {},
        cell,
        (),
        graph_settings['graphviz'])

    core.plot_graphviz(
        graphviz,
        graph_settings['plot'])

    import ipdb; ipdb.set_trace()


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
                 **plot_settings
                 # port_labels=False,
                 # rankdir='BT',
                 # remove_nodes=[
                 #     ('cell', 'address',),
                 #     ('cell', 'config'),
                 #     ('cell', 'interval'),
                 # ]
                 )

def test_bio_schema():
    core = generate_types()
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

    plot_bigraph(b, core=core, filename='bioschema', show_process_schema_keys=[],
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
    plot_bigraph(flat_composite_spec, rankdir='RL',
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
    plot_bigraph(processes_spec, rankdir='BT', filename='multiple_processes',
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


def test_multi_input_output():
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
    plot_bigraph(
        processes_spec, show_process_schema_keys=None,
        rankdir='BT', filename='multiple_processes',
        **plot_settings)


def test_cell_hierarchy():
    core = generate_types()

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
        remove_process_place_edges=True,
        filename='cell',
        **plot_settings)


def test_multiple_disconnected_ports():
    core = generate_types()

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
    core = generate_types()

    spec = {
        'composite': {
            '_type': 'composite',
            '_inputs': {
                'port1': 'any',
            },
            '_outputs': {
                'port2': 'any',
            },
            'inputs': {
                'port1': ['external store'],
            },
            'store1': 'any',
            'store2': 'any',
            'bridge': {
                'inputs': {
                    'port1': ['store1'],
                },
                'outputs': {
                    'port2': ['store2'],
                }
            },
            'process1': {
                '_type': 'process',
                '_inputs': {
                    'port3': 'any',
                },
                '_outputs': {
                    'port4': 'any',
                },
                'inputs': {
                    'port3': ['store1'],
                },
                'outputs': {
                    'port4': ['store2'],
                }
            },
        },
    }

    plot_bigraph(
        spec,
        core=core,
        filename='composite_process',
        **plot_settings)


if __name__ == '__main__':
    test_graphviz()
    # test_diagram_plot()
    # test_bio_schema()
    # test_flat_composite()
    # test_multi_processes()
    # test_nested_processes()
    # test_multi_input_output()
    # test_cell_hierarchy()
    # test_multiple_disconnected_ports()
    # test_composite_process()
