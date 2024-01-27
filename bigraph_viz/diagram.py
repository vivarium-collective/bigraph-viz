"""
Bigraph diagram
"""
import os
from bigraph_schema import TypeSystem, Edge
# from process_bigraph import core
from bigraph_viz.plot import (
    absolute_path, extend_bigraph, get_state_path_extended,
    check_if_path_in_removed_nodes, make_label,
)
import graphviz



SCHEMA_KEYS = ['_type', 'config', 'address']  # get these from bigraph_schema
PROCESS_INTERFACE_KEYS = ['inputs', 'outputs']
EDGE_TYPES = ['process', 'step', 'edge']


def get_flattened_wires(wires, hyperedges):
    pass


def get_graph_dict(
        schema,
        state,
        core,
        graph_dict=None,
        path=None,
        # remove_nodes=None,
):
    path = path or ()

    # initialize bigraph
    graph_dict = graph_dict or {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'hyper_edges': [],
        'disconnected_hyper_edges': [],
        'bridges': [],
    }

    for key, value in state.items():
        subpath = path + (key,)
        node_spec = {'name': key,
                     'path': subpath}

        if core.check('edge', value):
            graph_dict['process_nodes'].append(node_spec)

            # this is an edge, get its inputs and outputs
            input_wires = value.get('inputs', {})
            output_wires = value.get('outputs', {})

            for port, input_wire in input_wires.items():
                target = input_wire  # todo get absolute path
                graph_dict['hyper_edges'].append({
                    'edge_path': subpath,
                    'target_path': target,
                    'port': port,
                    'type': 'input',
                })
            for port, output_wire in output_wires.items():
                target = output_wire  # todo get absolute path
                graph_dict['hyper_edges'].append({
                    'edge_path': subpath,
                    'target_path': target,
                    'port': port,
                    'type': 'output',
                })

        else:
            graph_dict['state_nodes'].append(node_spec)



    return graph_dict


def get_graphviz_fig(
        graph_dict,
        label_margin='0.05',
        node_label_size='12pt',
        size='16,10',
        rankdir='TB',
        dpi='70',
):
    """make a graphviz figure from a graph_dict"""
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

    # state nodes
    graph.attr('node', **state_node_spec)
    for node in graph_dict['state_nodes']:
        node_path = node['path']
        node_name = str(node_path)
        node_names.append(node_name)

        # make the label
        label = node_path[-1]
        schema_label = None
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

        # show edge
        graph.attr('edge', style='filled')
        node_name1 = str(edge[0])
        target_name = str(edge[1])
        graph.edge(node_name1, target_name)

    # hyper edges
    graph.attr('edge', **hyper_edge_spec)
    for edge in graph_dict['hyper_edges']:
        edge_path = edge['edge_path']
        edge_name = str(edge_path)
        target_path = edge['target_path']
        port = edge['port']
        edge_type = edge['type']  # input or output
        target_name = str(target_path)

        # place it in the graph
        if target_name not in graph.body: # is the source node already in the graph?
            label = make_label(target_path[-1])
            graph.node(target_name, label=label, **state_node_spec)
        else:
            with graph.subgraph(name=edge_name) as c:
                c.edge(target_name, edge_name)

    # disconnected hyper edges
    graph.attr('edge', **hyper_edge_spec)
    for edge in graph_dict['disconnected_hyper_edges']:
        pass

    return graph


def plot_bigraph(
        state,
        schema=None,
        core=None,
        out_dir=None,
        filename=None,
        file_format='png',
):
    core = core or TypeSystem()
    schema = schema or {}
    schema = core.infer_schema(schema, state)

    # parse out the network
    graph_dict = get_graph_dict(
        schema=schema,
        state=state,
        core=core)

    print(graph_dict)

    # make a figure
    graph = get_graphviz_fig(graph_dict)

    # # display or save results
    # if filename is not None:
    #     out_dir = out_dir or 'out'
    #     os.makedirs(out_dir, exist_ok=True)
    #     fig_path = os.path.join(out_dir, filename)
    #     print(f"Writing {fig_path}")
    #     graph.render(filename=fig_path, format=file_format)
    # return graph


def test_diagram_plot():
    # metabolic_process_type = {
    #     '_type': 'process',
    #     '_inputs': {
    #         'nutrients': 'any',
    #     },
    #     '_outputs': {
    #         'se'
    #     }
    # }
    # register('metabolic_process', metabolic_process_type)  # TODO -- register where?


    cell = {
        'cell': {
            '_type': 'edge',
            'config': {},
            'address': 'local:cell',   # TODO -- this is where the ports/inputs/outputs come from
            '_inputs': {
                'nutrients': 'any',
            },
            '_outputs': {
                'secretions': 'any',
                'biomass': 'any',
            },
            'inputs': {
                'nutrients': ['nutrients_store'],
            },
            'outputs': {
                'secretions': ['secretions_store'],
                'biomass': ['biomass_store'],
            }
        }
    }
    plot_bigraph(cell, filename='bigraph_cell')


if __name__ == '__main__':
    test_diagram_plot()
