"""
Bigraph diagram
"""
import os
from bigraph_schema import TypeSystem, Edge
from bigraph_schema.registry import type_schema_keys, function_keys
import graphviz


SCHEMA_KEYS = type_schema_keys | set(function_keys)
EDGE_TYPES = ['process', 'step', 'edge']


def absolute_path(path, relative):
    progress = list(path)
    for step in relative:
        if step == '..' and len(progress) > 0:
            progress = progress[:-1]
        else:
            progress.append(step)
    return tuple(progress)


def extend_bigraph(bigraph, bigraph2):
    for key, values in bigraph.items():
        if isinstance(values, list):
            bigraph[key] += bigraph2[key]
        elif isinstance(values, dict):
            bigraph[key].update(bigraph2[key])
    return bigraph


def check_if_path_in_removed_nodes(path, remove_nodes):
    if remove_nodes:
        return any(remove_path == path[:len(remove_path)] for remove_path in remove_nodes)
    return False


def make_label(label):
    return f'<{label}>'


def get_flattened_graph(
        schema,
        path=None,
        remove_nodes=None,
):
    path = path or ()

    # initialize bigraph
    bigraph = {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'hyper_edges': {},
        'disconnected_hyper_edges': {},
        'bridges': {},
        'flow': {},
    }

    for key, child in schema.items():
        path_here = path + (key,)
        if check_if_path_in_removed_nodes(path_here, remove_nodes):
            continue  # skip node if path in removed_nodes
        node = {'path': path_here}

        if not isinstance(child, dict):
            # this is a leaf node
            node['_value'] = child
            bigraph['state_nodes'] += [node]
            continue

        # what kind of node?
        if child.get('_type') in EDGE_TYPES:
            # this is a hyperedge/process
            bigraph['process_nodes'] += [node]
        else:
            bigraph['state_nodes'] += [node]

        # check inner states
        if isinstance(child, dict):
            # this is a branch node
            child_bigraph_network = get_flattened_graph(
                child,
                path=path_here,
                remove_nodes=remove_nodes)
            bigraph = extend_bigraph(bigraph, child_bigraph_network)

            # add place edges to this layer
            for node in child.keys():
                child_path = path_here + (node,)
                if node in SCHEMA_KEYS or check_if_path_in_removed_nodes(child_path, remove_nodes):
                    continue
                place_edge = [(path_here, child_path)]
                bigraph['place_edges'] += place_edge

    return bigraph


def get_graphviz_fig(
        bigraph_network,
        label_margin='0.05',
        node_label_size='12pt',
        size='16,10',
        rankdir='TB',
        dpi='70',
):
    """make a graphviz figure from a bigraph_network"""

    node_names = []

    # node specs
    state_node_spec = {
        'shape': 'circle', 'penwidth': '2', 'margin': label_margin, 'fontsize': node_label_size}
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false', 'margin': label_margin, 'fontsize': node_label_size}
    hyper_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'dot', 'arrowsize': '0.5'}

    # initialize graph
    graph_name = 'bigraph'
    graph = graphviz.Digraph(graph_name, engine='dot')
    graph.attr(size=size, overlap='false', rankdir=rankdir, dpi=dpi)

    # check if multiple layers
    multilayer = False
    for node in bigraph_network['state_nodes']:
        node_path = node['path']
        if len(node_path) > 1:
            multilayer = True

    # state nodes
    graph.attr('node', **state_node_spec)
    for node in bigraph_network['state_nodes']:
        node_path = node['path']
        node_name = str(node_path)
        node_names.append(node_name)

        # make the label
        label = node_path[-1]
        schema_label = None
        if schema_label:
            label += schema_label
        label = make_label(label)

        if len(node_path) == 1 and multilayer:
            # the top node gets a double circle
            graph.node(node_name, label=label, peripheries='2')
        else:
            graph.node(node_name, label=label)

    # process nodes
    process_paths = []
    graph.attr('node', **process_node_spec)
    for node in bigraph_network['process_nodes']:
        node_path = node['path']
        process_paths.append(node_path)
        node_name = str(node_path)
        node_names.append(node_name)
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

        # show edge
        graph.attr('edge', style='filled')

        node_name1 = str(edge[0])
        node_name2 = str(edge[1])
        graph.edge(node_name1, node_name2)

    # hyper edges
    graph.attr('edge', **hyper_edge_spec)
    for node_path, wires in bigraph_network['hyper_edges'].items():
        node_name = str(node_path)
        with graph.subgraph(name=node_name) as c:
            for port, state_path in wires.items():
                absolute_state_path = absolute_path(node_path, state_path)
                node_name1 = str(node_path)
                node_name2 = str(absolute_state_path)

                # are the nodes already in the graph?
                if node_name2 not in graph.body:
                    label = make_label(absolute_state_path[-1])
                    graph.node(node_name2, label=label, **state_node_spec)
                c.edge(node_name2, node_name1)

    # disconnected hyper edges
    graph.attr('edge', **hyper_edge_spec)
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
                c.edge(node_name2, node_name1)

    return graph


def plot_diagram(
        schema,
        out_dir=None,
        filename=None,
        file_format='png',
):
    # parse out the network
    bigraph_network = get_flattened_graph(schema)

    # make a figure
    graph = get_graphviz_fig(bigraph_network)

    # display or save results
    if filename is not None:
        out_dir = out_dir or 'out'
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        graph.render(filename=fig_path, format=file_format)
    return graph


def test_diagram_plot():
    cell_schema = {
        'cell': {
            '_type': 'process',
            'config': {},
            'address': 'local:cell',
            'inputs': {
                'nutrients': 'any',
            },
            'outputs': {
                'secretions': 'any',
                'biomass': 'any',
            },
        }
    }
    plot_diagram(cell_schema, filename='bigraph_cell')


if __name__ == '__main__':
    test_diagram_plot()
