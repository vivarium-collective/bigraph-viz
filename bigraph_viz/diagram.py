"""
Bigraph diagram
"""
import os
from bigraph_schema import TypeSystem, Edge
from bigraph_viz.plot import absolute_path, make_label
import graphviz


PROCESS_SCHEMA_KEYS = ['config', 'address', 'interval', 'inputs', 'outputs']


step_type = {
    '_type': 'step',
    '_inherit': 'edge',
    'address': 'string',
    'config': 'schema'}


process_type = {
    '_type': 'process',
    '_inherit': 'step',
    'interval': 'float'}


def get_graph_wires(schema, wires, graph_dict, schema_key, edge_path, port):

    if isinstance(wires, dict):
        for port, subwire in wires.items():
            subschema = schema.get(port, schema)
            graph_dict = get_graph_wires(
                subschema, subwire, graph_dict, schema_key, edge_path, port)
    elif isinstance(wires, (list, tuple)):
        target_path = absolute_path(edge_path[:-1], tuple(wires))   # TODO -- make sure this resolves ".."
        graph_dict['hyper_edges'].append({
            'edge_path': edge_path,
            'target_path': target_path,
            'port': port,
            'type': schema_key,
        })
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
):
    path = path or ()
    top_state = top_state or state

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
        if key.startswith('_') and not retain_type_keys:
            continue

        subpath = path + (key,)
        node_spec = {'name': key, 'path': subpath}

        if core.check('edge', value):  # this is a process/edge node
            if key in PROCESS_SCHEMA_KEYS and not retain_process_keys:
                continue

            graph_dict['process_nodes'].append(node_spec)

            # this is an edge, get its inputs and outputs
            input_wires = value.get('inputs', {})
            output_wires = value.get('outputs', {})
            input_schema = value.get('_inputs', {})
            output_schema = value.get('_outputs', {})

            # get the input and output wires
            graph_dict = get_graph_wires(
                input_schema, input_wires, graph_dict, schema_key='inputs', edge_path=subpath, port=())
            graph_dict = get_graph_wires(
                output_schema, output_wires, graph_dict, schema_key='outputs', edge_path=subpath, port=())

        else:  # this is a state node
            graph_dict['state_nodes'].append(node_spec)

        if isinstance(value, dict):  # get subgraph
            graph_dict = get_graph_dict(
                schema=schema,
                state=value,
                core=core,
                graph_dict=graph_dict,
                path=subpath,
                top_state=top_state,
            )

            # get the place edge
            for node in value.keys():
                if node.startswith('_') and not retain_type_keys:
                    continue
                if not retain_process_keys and core.check('edge', value):
                    continue

                child_path = subpath + (node,)
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
        graph.attr('edge', style='filled')
        parent_node = str(edge['parent'])
        child_node = str(edge['child'])
        graph.edge(parent_node, child_node)

    # hyper edges
    for edge in graph_dict['hyper_edges']:
        process_path = edge['edge_path']
        process_name = str(process_path)
        target_path = edge['target_path']
        port = edge['port']
        edge_type = edge['type']  # input or output
        target_name = str(target_path)

        # place it in the graph
        if target_name not in graph.body:  # is the source node already in the graph?
            label = make_label(target_path[-1])
            graph.node(target_name, label=label, **state_node_spec)

        if edge_type == 'inputs':
            graph.attr('edge', **input_edge_spec)
        elif edge_type == 'outputs':
            graph.attr('edge', **output_edge_spec)
        else:
            graph.attr('edge', **hyper_edge_spec)
        with graph.subgraph(name=process_name) as c:
            label = make_label(port)
            c.edge(target_name, process_name, label=label, labelloc="t")

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

    if not core.exists('step'):
        core.register('step', step_type)
    if not core.exists('process'):
        core.register('process', process_type)

    schema, state = core.complete(schema, state)

    # parse out the network
    graph_dict = get_graph_dict(
        schema=schema,
        state=state,
        core=core)

    # print(graph_dict)

    # make a figure
    graph = get_graphviz_fig(graph_dict)

    # display or save results
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
            # '_type': 'map[float]',
            'a': 11.0,
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            'config': {},
            'address': 'local:cell',   # TODO -- this is where the ports/inputs/outputs come from
            '_inputs': {
                'nutrients': 'float',
            },
            '_outputs': {
                'secretions': 'float',
                'biomass': 'float',
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
