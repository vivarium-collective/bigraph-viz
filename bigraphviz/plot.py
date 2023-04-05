"""
====
Plot
====

plotting tool
"""
import copy
import os

from bigraphviz.dict_utils import absolute_path
import graphviz

special_keys = [
    '_value',
    '_process',
    '_config',
    '_wires',
    '_type',
    '_ports',
    '_tunnels',
]


def extend_bigraph(bigraph, bigraph2):
    for key, values in bigraph.items():
        if isinstance(values, list):
            bigraph[key] += bigraph2[key]
        elif isinstance(values, dict):
            bigraph[key].update(bigraph2[key])
    return bigraph


def state_path_tuple(state_path):
    if isinstance(state_path, str):
        state_path = [state_path]
    return state_path


def make_label(label):
    label = label.replace(' ', '<br/>')  # replace spaces with new lines
    return f'<{label}>'


def get_bigraph_network(bigraph_dict, path=None):
    path = path or ()

    # initialize bigraph
    bigraph = {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'hyper_edges': {},
        'disconnected_hyper_edges': {},
        'tunnels': {},
    }

    for key, child in bigraph_dict.items():
        if key not in special_keys:
            path_here = path + (key,)
            node = {'path': path_here}
            # add schema
            if '_value' in child:
                node['value'] = child['_value']
            if '_type' in child:
                node['type'] = child['_type']

            # what kind of node?
            if '_wires' in child or '_ports' in child:
                # this is a hyperedge/process
                if path_here not in bigraph['hyper_edges']:
                    bigraph['hyper_edges'][path_here] = {}
                if '_wires' in child:
                    for port, state_path in child['_wires'].items():
                        state_path = state_path_tuple(state_path)
                        state_path.insert(0, '..')  # go up one to the same level as the process
                        bigraph['hyper_edges'][path_here][port] = state_path

                # check for mismatch, there might be disconnected wires or mismatch with declared wires
                wire_ports = []
                if '_wires' in child:
                    wire_ports = child['_wires'].keys()
                if '_ports' in child:
                    # wires need to match schema
                    schema_ports = child['_ports'].keys()
                    assert all(item in schema_ports for item in wire_ports), \
                        f"attempting to wire undeclared process ports: " \
                        f"wire ports: {wire_ports}, schema ports: {schema_ports}"
                    disconnected_ports = [port_id for port_id in schema_ports if port_id not in wire_ports]
                    if disconnected_ports and path_here not in bigraph['disconnected_hyper_edges']:
                        bigraph['disconnected_hyper_edges'][path_here] = {}
                    for port in disconnected_ports:
                        bigraph['disconnected_hyper_edges'][path_here][port] = [port, ]
                if '_tunnels' in child:
                    tunnels = child['_tunnels']
                    if path_here not in bigraph['tunnels']:
                        bigraph['tunnels'][path_here] = {}
                    for port, state_path in tunnels.items():
                        assert port in bigraph['disconnected_hyper_edges'][path_here].keys() or \
                               port in bigraph['hyper_edges'][path_here].keys(), f"tunnel '{port}' " \
                                                                                 f"is not declared in ports"
                        bigraph['tunnels'][path_here][port] = state_path_tuple(state_path)

                bigraph['process_nodes'] += [node]
            else:
                bigraph['state_nodes'] += [node]

            # check inner states
            if isinstance(child, dict):
                child_bigraph_network = get_bigraph_network(child, path=path_here)
                bigraph = extend_bigraph(bigraph, child_bigraph_network)

                # add place edges to this layer
                bigraph['place_edges'] += [
                    (path_here, path_here + (node,)) for node in child.keys()
                    if node not in special_keys]

    return bigraph


def get_graphviz_graph(
        bigraph_network,
        size='16,10',
        node_label_size='12pt',
        plot_schema=False,
        port_labels=True,
        port_label_size='10pt',
        engine='dot',
        rankdir='TB',
        node_groups=False,
        invisible_edges=False,
        remove_process_place_edges=False,
):
    """make a graphviz figure from a bigraph_network"""
    node_groups = node_groups or []
    invisible_edges = invisible_edges or []
    node_names = []

    # initialize graph
    graph_name = 'bigraph'
    graph = graphviz.Digraph(graph_name, engine=engine)
    graph.attr(size=size, overlap='false')

    # state nodes
    graph.attr('node', shape='circle', penwidth='2', margin='0.05', fontsize=node_label_size)
    graph.attr(rankdir=rankdir)

    # check if multiple layers
    multilayer = False
    for node in bigraph_network['state_nodes']:
        node_path = node['path']
        if len(node_path) > 1:
            multilayer = True

    # get state nodes
    for node in bigraph_network['state_nodes']:
        node_path = node['path']
        node_name = str(node_path)
        node_names.append(node_name)

        # make the label
        label = node_path[-1]
        label = label.replace(' ', '<br/>')  # replace spaces with new lines
        if plot_schema:
            # add schema to label
            schema_label = None
            if 'value' in node:
                if not schema_label:
                    schema_label = '<br/>'
                schema_label += f"{node['value']}"
            if 'type' in node:
                if not schema_label:
                    schema_label = '<br/>'
                schema_label += f"::{node['type']}"
            if schema_label:
                label += schema_label
        label = f'<{label}>'

        if len(node_path) == 1 and multilayer:
            # the top node gets a double circle
            graph.node(node_name, label=label, peripheries='2')
        else:
            graph.node(node_name, label=label)

    # process nodes
    process_paths = []
    graph.attr('node', shape='box', penwidth='2', constraint='false')
    for node in bigraph_network['process_nodes']:
        node_path = node['path']
        process_paths.append(node_path)
        node_name = str(node_path)
        node_names.append(node_name)

        # make the label
        label = make_label(node_path[-1])

        # composite processes have sub-nodes
        composite_process = False
        for edge in bigraph_network['place_edges']:
            if edge[0] == node_path:
                composite_process = True
        if len(node_path) == 1 and composite_process:
            # composite processes gets a double box
            graph.node(node_name, label=label, peripheries='2')
        else:
            graph.node(node_name, label=label)

    # place edges
    graph.attr('edge', arrowhead='none', penwidth='2')
    for edge in bigraph_network['place_edges']:

        # show edge or not
        show_edge = True
        if remove_process_place_edges and edge[1] in process_paths:
            show_edge = False
        if edge in invisible_edges:
            show_edge = False
        if show_edge:
            graph.attr('edge', style='filled')
        else:
            graph.attr('edge', style='invis')

        node_name1 = str(edge[0])
        node_name2 = str(edge[1])
        graph.edge(node_name1, node_name2)

    # hyper edges
    graph.attr('edge', style='dashed', penwidth='1', arrowhead='dot', arrowsize='0.5')
    for node_path, wires in bigraph_network['hyper_edges'].items():
        node_name = str(node_path)
        with graph.subgraph(name=node_name) as c:
            for port, state_path in wires.items():
                absolute_state_path = absolute_path(node_path, state_path)
                node_name1 = str(node_path)
                node_name2 = str(absolute_state_path)
                if port_labels:
                    label = make_label(port)
                    c.edge(node_name2, node_name1, label=label, labelloc="t", fontsize=port_label_size)
                else:
                    c.edge(node_name2, node_name1)

    # disconnected hyper edges
    graph.attr('edge', style='dashed', penwidth='1', arrowhead='dot')
    for node_path, wires in bigraph_network['disconnected_hyper_edges'].items():
        node_name = str(node_path)
        with graph.subgraph(name=node_name) as c:
            # c.attr(rank='source', rankdir='TB')
            for port, state_path in wires.items():
                absolute_state_path = absolute_path(node_path, state_path)
                node_name1 = str(node_path)
                node_name2 = str(absolute_state_path)
                # add invisible node for port
                graph.node(node_name2, label='', style='invis', width='0')
                if port_labels:
                    label = make_label(port)
                    c.edge(node_name2, node_name1, label=label, labelloc="t", fontsize=port_label_size)
                else:
                    c.edge(node_name2, node_name1)

    # tunnel edges
    graph.attr('edge', style='dashed', penwidth='1', arrowtail='dot', arrowhead='none', dir='both')
    for node_path, wires in bigraph_network['tunnels'].items():
        node_name = str(node_path)
        with graph.subgraph(name=node_name) as c:
            for port, state_path in wires.items():
                absolute_state_path = absolute_path(node_path, state_path)
                node_name1 = str(node_path)
                node_name2 = str(absolute_state_path)
                if port_labels:
                    # label = make_label(port)
                    label = make_label(f'tunnel {port}')
                    c.edge(node_name1, node_name2, label=label, labelloc="t", fontsize=port_label_size)
                else:
                    c.edge(node_name1, node_name2)

    # grouped nodes
    for group in node_groups:
        group_name = str(group)
        with graph.subgraph(name=group_name) as c:
            c.attr(rank='same')
            for path in group:
                node_name = str(path)
                if node_name in node_names:
                    c.node(node_name)
                else:
                    print(f'node {node_name} not in graph')

    return graph


def plot_bigraph(
        bigraph_schema,
        size='16,10',
        node_label_size='12pt',
        plot_schema=False,
        port_labels=True,
        port_label_size='10pt',
        engine='dot',
        rankdir='TB',
        node_groups=False,
        invisible_edges=False,
        remove_process_place_edges=False,
        view=False,
        print_source=False,
        file_format='png',
        out_dir=None,
        filename=None,
):
    """
    Plot a bigraph from bigraph schema.

    Args:
        bigraph_schema (dict): The bigraph schema dict that will be plotted.
        size (str, optional): The size of the output figure (example: '16,10'). Default is '16,10'.
        node_label_size (str, optional): The font size for the node labels. Default is None.
        plot_schema (bool, optional): Turn on schema text in nodes. Default is False.
        port_labels (bool, optional): Turn on port labels. Default is False.
        port_label_size (str, optional): The font size of the port labels (example: '10pt'). Default is None.
        engine (str, optional): Graphviz graphing engine. Try 'dot' or 'neato'. Default is 'dot'.
        rankdir (str, optional): Sets direction of graph layout. 'TB'=top-to-bottom, 'LR'=left-to-right.
            Default is 'TB'.
        node_groups (list, optional): A list of lists of nodes.
            Each sub-list is a grouping of nodes that will be aligned at the same rank.
            For example: [[('path to', 'group1 node1',), ('path to', 'group1 node2',)], [another group]]
            Default is None.
        invisible_edges (list, optional): A list of edge tuples. The edge tuples have the (source, target) node
            according to the nodes' paths. For example: [(('top',), ('top', 'inner1')), (another edge)]
            Default is None.
        remove_process_place_edges (bool, optional): Turn off process place edges from plotting. Default is False.
        view (bool, optional): Open the rendered graph in a viewer. Default is False.
        print_source (bool, optional): Print the graphviz DOT source code as string. Default is False.
        file_format (str, optional): File format of the output image. Default is 'png'.
        out_dir (bool, optional): The output directory for the bigraph image. Default is None.
        filename (bool, optional): The file name for the bigraph image. Default is None.

    Notes:
        You can adjust node labels using HTML syntax for fonts, colors, sizes, subscript, superscript. For example:
            H<sub><font point-size="8">2</font></sub>O will print H2O with 2 as a subscript with smaller font.
    """

    # get kwargs dict and remove plotting-specific kwargs
    kwargs = locals()
    bigraph_schema_0 = kwargs.pop('bigraph_schema')
    bigraph_schema = copy.deepcopy(bigraph_schema_0)
    view = kwargs.pop('view')
    print_source = kwargs.pop('print_source')
    file_format = kwargs.pop('file_format')
    out_dir = kwargs.pop('out_dir')
    filename = kwargs.pop('filename')

    # get the nodes and edges from the composite
    bigraph_network = get_bigraph_network(bigraph_schema)

    # make graphviz network
    graph = get_graphviz_graph(bigraph_network, **kwargs)

    # display or save results
    if view:
        graph.view()
    if print_source:
        print(graph.source)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        graph.render(filename=fig_path, format=file_format)
    return graph


# testing functions
plot_settings_test = {'plot_schema': True, 'out_dir': 'out'}


def test_simple_spec():
    simple_store_spec = {
        'store1': {
            '_value': 1.0,
            '_type': 'float',
        },
    }
    plot_bigraph(simple_store_spec, **plot_settings_test, filename='simple_store')


def test_composite_spec():
    composite_spec = {
        'store1': {
            'store1.1': {
                '_value': 1.1,
                '_type': 'float'
            },
            'store1.2': {
                '_value': 2,
                '_type': 'int'
            },
            'process1': {
                '_ports': {
                    'port1': {'_type': 'type'},
                    'port2': {'_type': 'type'},
                },
                '_wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
        },
        'process3': {
            '_wires': {
                'port1': 'store1'
            }
        }  # TODO -- wires without ports should not work.
    }
    plot_bigraph(composite_spec, **plot_settings_test, filename='nested_composite', remove_process_place_edges=True)


def test_disconnected_process_spec():
    # disconnected processes
    process_schema = {
        '_ports': {
            'port1': {'_type': 'type'},
            'port2': {'_type': 'type'}
        }
    }
    process_spec = {
        'process1': process_schema,
        'process2': process_schema,
        'process3': process_schema,
    }
    plot_bigraph(process_spec, **plot_settings_test, rankdir='BT', filename='disconnected_processes')


def test_nested_spec():
    nested_processes = {
        'cell': {
            'membrane': {
                'transporters': {'_type': 'concentrations'},
                'lipids': {'_type': 'concentrations'},
                'transmembrane transport': {
                    '_value': {
                        '_process': 'transport URI',
                        '_config': {'parameter': 1}
                    },
                    '_wires': {
                        'transporters': 'transporters',
                        'internal': ['..', 'cytoplasm', 'metabolites']},
                    '_ports': {
                        'transporters': {'_type': 'concentrations'},
                        'internal': {'_type': 'concentrations'},
                        'external': {'_type': 'concentrations'}
                    }
                }
            },
            'cytoplasm': {
                'metabolites': {
                    '_value': 1.1,
                    '_type': 'concentrations'
                },
                'ribosomal complexes': {
                    '_value': 2.2,
                    '_type': 'concentrations'
                },
                'transcript regulation complex': {
                    '_value': 0.01,
                    '_type': 'concentrations',
                    'transcripts': {
                        '_value': 0.1,
                        '_type': 'concentrations'
                    }
                },
                'translation': {
                    '_wires': {
                        'p1': 'ribosomal complexes',
                        'p2': ['transcript regulation complex', 'transcripts']}}},
            'nucleoid': {
                'chromosome': {
                    'genes': 'sequences'
                }
            }
        }
    }
    plot_bigraph(nested_processes, **plot_settings_test, filename='nested_processes')


def test_composite_process_spec():
    composite_process_spec = {
        'composite_process': {
            'store1.1': {
                '_value': 1.1, '_type': 'float'
            },
            'store1.2': {
                '_value': 2, '_type': 'int'
            },
            'process1': {
                '_ports': {
                    'port1': 'type',
                    'port2': 'type',
                },
                '_wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
            'process2': {
                '_ports': {
                    'port1': {'_type': 'type'},
                    'port2': {'_type': 'type'},
                },
                '_wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
            '_ports': {
                'port1': {'_type': 'type'},
                'port2': {'_type': 'type'},
            },
            '_tunnels': {
                'port1': 'store1.1',
                'port2': 'store1.2',
            }
        }
    }
    plot_bigraph(composite_process_spec,
                 **plot_settings_test,
                 remove_process_place_edges=True,
                 filename='composite_process'
                 )


if __name__ == '__main__':
    # test_simple_spec()
    # test_composite_spec()
    # test_disconnected_process_spec()
    # test_nested_spec()
    test_composite_process_spec()
