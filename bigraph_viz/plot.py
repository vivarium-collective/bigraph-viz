"""
============
Plot Bigraph
============
"""
import copy
import os

from bigraph_viz.dict_utils import absolute_path, schema_keys, process_schema_keys
import graphviz
import warnings

warnings.warn(
    "The `plot_bigraph` method from `bigraph_viz.plot` is deprecated and will be removed in a future release. "
    "Please use the new `plot_bigraph` from `bigraph_viz.diagram`.",
    DeprecationWarning)


def extend_bigraph(bigraph, bigraph2):
    for key, values in bigraph.items():
        if isinstance(values, list):
            bigraph[key] += bigraph2[key]
        elif isinstance(values, dict):
            bigraph[key].update(bigraph2[key])
    return bigraph


def state_path_tuple(state_path):
    state_path = copy.deepcopy(state_path)
    if isinstance(state_path, str):
        state_path = [state_path]
    return state_path


def make_label(label):
    # label = label.replace(' ', '<br/>')  # replace spaces with new lines
    return f'<{label}>'


def check_if_path_in_removed_nodes(path, remove_nodes):
    if remove_nodes:
        return any(remove_path == path[:len(remove_path)] for remove_path in remove_nodes)
    return False


def get_state_path_extended(state_path):
    if isinstance(state_path, tuple):
        # support tuple paths
        state_path = list(state_path)
    state_path = state_path_tuple(state_path)
    state_path.insert(0, '..')  # go up one to the same level as the process
    return state_path


def get_bigraph_network(
        bigraph_dict,
        path=None,
        remove_nodes=None,
        show_process_schema=False
):
    path = path or ()

    # initialize bigraph
    bigraph = {
        'state_nodes': [],
        'process_nodes': [],
        'place_edges': [],
        'hyper_edges': {},
        'disconnected_hyper_edges': {},
        'tunnels': {},
        'flow': {},
    }

    for key, child in bigraph_dict.items():
        # skip process schema
        if not show_process_schema and key in process_schema_keys:
            continue
        if key in schema_keys:
            continue

        path_here = path + (key,)
        if check_if_path_in_removed_nodes(path_here, remove_nodes):
            # skip node if path in removed_nodes
            continue
        node = {'path': path_here}

        if not isinstance(child, dict):
            # this is a leaf node
            node['_value'] = child
            bigraph['state_nodes'] += [node]
            continue

        # add schema
        if '_value' in child:
            node['_value'] = child['_value']
        if '_type' in child:
            node['_type'] = child['_type']
        if '_sync_step' in child:
            node['sync_step'] = child['_sync_step']

        # what kind of node?
        if ('wires' in child or
            '_ports' in child or
                child.get('_type') == 'process' or
                child.get('_type') == 'step' or
                child.get('_type') == 'edge'
        ):
            # this is a hyperedge/process
            if path_here not in bigraph['hyper_edges']:
                bigraph['hyper_edges'][path_here] = {}
            if 'wires' in child:
                for port, state_path in child['wires'].items():
                    bigraph['hyper_edges'][path_here][port] = get_state_path_extended(state_path)
            if 'inputs' in child:
                for port, state_path in child['inputs'].items():
                    bigraph['hyper_edges'][path_here][port] = get_state_path_extended(state_path)
            if 'outputs' in child:
                for port, state_path in child['outputs'].items():
                    bigraph['hyper_edges'][path_here][port] = get_state_path_extended(state_path)
            if '_depends_on' in child:
                depends_on = child.get('_depends_on', [])
                if path_here not in bigraph['flow']:
                    bigraph['flow'][path_here] = []
                if isinstance(depends_on, str):
                    depends_on = [depends_on]
                for state_path in depends_on:
                    if isinstance(state_path, str):
                        state_path = [state_path]
                    state_path = state_path_tuple(state_path)
                    state_path.insert(0, '..')  # go up one to the same level as the process
                    bigraph['flow'][path_here].append(state_path)

            # check for mismatch, there might be disconnected wires or mismatch with declared wires
            wire_ports = []
            if 'wires' in child:
                wire_ports = child['wires'].keys()
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
            # this is a branch node
            child_bigraph_network = get_bigraph_network(
                child, 
                path=path_here, 
                remove_nodes=remove_nodes, 
                show_process_schema=show_process_schema)
            bigraph = extend_bigraph(bigraph, child_bigraph_network)

            # add place edges to this layer
            for node in child.keys():
                
                # skip process schema
                if not show_process_schema and node in process_schema_keys:
                    continue
                
                child_path = path_here + (node,)
                if node in schema_keys or check_if_path_in_removed_nodes(child_path, remove_nodes):
                    continue
                place_edge = [(path_here, child_path)]
                bigraph['place_edges'] += place_edge

    return bigraph


def get_graphviz_bigraph(
        bigraph_network,
        size='16,10',
        node_label_size='12pt',
        show_values=False,
        show_types=False,
        collapse_processes=False,
        port_labels=True,
        mark_top=False,
        port_label_size='10pt',
        rankdir='TB',
        dpi='70',
        label_margin='0.05',
        node_border_colors=None,
        node_fill_colors=None,
        node_groups=False,
        invisible_edges=False,
        remove_process_place_edges=False,
):
    """make a graphviz figure from a bigraph_network"""
    node_groups = node_groups or []
    invisible_edges = invisible_edges or []
    node_names = []

    # node specs
    state_node_spec = {
        'shape': 'circle', 'penwidth': '2', 'margin': label_margin, 'fontsize': node_label_size}
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false', 'margin': label_margin, 'fontsize': node_label_size}
    hyper_edge_spec = {
        'style': 'dashed', 'penwidth': '1', 'arrowhead': 'dot', 'arrowsize': '0.5'}
    if collapse_processes:
        process_node_spec = {
            'shape': 'circle', 'width': '0', 'height': '0', 'constraint': 'false', 'color': 'transparent',
        }
        hyper_edge_spec['arrowhead'] = 'none'

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
        if show_values:
            if '_value' in node:
                if not schema_label:
                    schema_label = '<br/>'
                schema_label += f"{node['_value']}"
        if show_types:
            if '_type' in node:
                if not schema_label:
                    schema_label = '<br/>'
                schema_label += f"::{node['_type']}"
        if schema_label:
            label += schema_label
        label = make_label(label)

        if len(node_path) == 1 and multilayer and mark_top:
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
            if collapse_processes:
                graph.node(node_name, label='', xlabel=label)
            else:
                graph.node(node_name, label=label)

    # place edges
    graph.attr('edge', arrowhead='none', penwidth='2')
    for edge in bigraph_network['place_edges']:

        # show edge or not
        show_edge = True
        if remove_process_place_edges and edge[1] in process_paths:
            show_edge = False
        elif edge in invisible_edges:
            show_edge = False

        if show_edge:
            graph.attr('edge', style='filled')
        else:
            graph.attr('edge', style='invis')

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

                if port_labels:
                    label = make_label(port)
                    c.edge(node_name2, node_name1, label=label, labelloc="t", fontsize=port_label_size)
                else:
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
        bigraph_schema,
        size='16,10',
        node_label_size='12pt',
        show_values=False,
        show_types=False,
        show_process_schema=False,
        collapse_processes=False,
        port_labels=True,
        port_label_size='10pt',
        rankdir='TB',
        node_border_colors=None,
        node_fill_colors=None,
        node_groups=False,
        remove_nodes=None,
        invisible_edges=False,
        mark_top=False,
        remove_process_place_edges=False,
        print_source=False,
        dpi='70',
        label_margin='0.05',
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
        show_values (bool, optional): Display on value info in node label. Default is False.
        show_types (bool, optional): Display on type info in node label. Default is False.
        show_process_schema (bool, optional): Display process schema info as internal nodes. Default is False.
        collapse_processes (bool, optional): Collapse rectangular process nodes to a hyperedge vertex. Default is False.
        port_labels (bool, optional): Turn on port labels. Default is True.
        port_label_size (str, optional): The font size of the port labels (example: '10pt'). Default is None.
        rankdir (str, optional): Sets direction of graph layout. 'TB'=top-to-bottom, 'LR'=left-to-right.
            Default is 'TB'.
        node_border_colors (dict, optional): Colors of node borders, with node path tuples mapped to the node color as
            the value: {('path', 'to', 'node',): 'color'}. Colors at https://graphviz.org/doc/info/colors.html
        node_fill_colors (dict, optional): Colors of node fills, with node path tuples mapped to the node color as
            the value: {('path', 'to', 'node',): 'color'}. Colors at https://graphviz.org/doc/info/colors.html
        node_groups (list, optional): A list of lists of nodes.
            Each sub-list is a grouping of nodes that will be aligned at the same rank.
            For example: [[('path to', 'group1 node1',), ('path to', 'group1 node2',)], [another group]]
            Default is None.
        remove_nodes (list, optional): A list of nodes to be removed.
        invisible_edges (list, optional): A list of edge tuples. The edge tuples have the (source, target) node
            according to the nodes' paths. For example: [(('top',), ('top', 'inner1')), (another edge)]
            Default is None.
        mark_top (bool). Turn on to mark the top nodes with a double outline.
        remove_process_place_edges (bool, optional): Turn off process place edges from plotting. Default is False.
        print_source (bool, optional): Print the graphviz DOT source code as string. Default is False.
        label_margin (str, optional): The margin size around a label. Default is '0.05'.
        file_format (str, optional): File format of the output image. Default is 'png'.
        out_dir (bool, optional): The output directory for the bigraph image. Default is None.
        filename (bool, optional): The file name for the bigraph image. Default is None.

    Notes:
        You can adjust node labels using HTML syntax for fonts, colors, sizes, subscript, superscript. For example:
            H<sub><font point-size="8">2</font></sub>O will print H2O with 2 as a subscript with smaller font.
        You can also make newlines by adding with <br/>
    """

    # get kwargs dict and remove plotting-specific kwargs
    kwargs = locals()
    bigraph_schema = kwargs.pop('bigraph_schema')
    print_source = kwargs.pop('print_source')
    file_format = kwargs.pop('file_format')
    out_dir = kwargs.pop('out_dir')
    filename = kwargs.pop('filename')
    remove_nodes = kwargs.pop('remove_nodes')
    show_process_schema = kwargs.pop('show_process_schema')

    # get the nodes and edges from the composite
    bigraph_network = get_bigraph_network(
        bigraph_schema,
        remove_nodes=remove_nodes,
        show_process_schema=show_process_schema,
)

    # make graphviz network
    graph = get_graphviz_bigraph(bigraph_network, **kwargs)
    # graph = graph.pipe(format='svg') # convert to svg

    # display or save results
    if print_source:
        print(graph.source)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        graph.render(filename=fig_path, format=file_format)
    return graph


def plot_flow(
        bigraph_schema,
        size='16,10',
        dpi='70',
        print_source=False,
        file_format='png',
        out_dir=None,
        filename=None,
):
    """plot the flow of a bigraph schema"""

    # get the bigraph network
    bigraph_network = get_bigraph_network(bigraph_schema)

    node_names = []
    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false'}
    dependency_edge_spec = {
        'arrowhead': 'normal', 'penwidth': '2', 'style': 'filled'}

    # initialize graph
    graph_name = 'flow'
    graph = graphviz.Digraph(graph_name, engine='dot')
    graph.attr(size=size, overlap='false', dpi=dpi)

    # process nodes
    process_paths = []
    graph.attr('node', **process_node_spec)
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

    # dependency edges
    graph.attr('edge', **dependency_edge_spec)
    for node_path, dependencies in bigraph_network['flow'].items():
        node_name1 = str(node_path)
        with graph.subgraph(name=node_name) as c:
            for state_path in dependencies:
                absolute_state_path = absolute_path(node_path, state_path)
                node_name2 = str(absolute_state_path)
                c.edge(node_name2, node_name1)

    # display or save results
    if print_source:
        print(graph.source)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        graph.render(filename=fig_path, format=file_format)
    return graph


def plot_multitimestep(
        bigraph_schema,
        total_time=10.0,
        size='16,10',
        dpi='70',
        print_source=False,
        file_format='png',
        out_dir=None,
        filename=None,
):
    """plot the timestepping of a bigraph schema"""
    total_time = float(total_time)

    # get the bigraph network
    bigraph_network = get_bigraph_network(bigraph_schema)

    process_node_spec = {
        'shape': 'box', 'penwidth': '2', 'constraint': 'false'}
    time_node_spec = {
        'style': 'filled', 'shape': 'circle', 'fontsize': '9', 'margin': '0',
        'fillcolor': 'white', 'color': 'none', 'width': '0'}
    end_node_spec = {
        'color': 'none', 'shape': 'point', 'width': '0', 'arrowsize': '0.5'}
    time_edge_spec = {
        'penwidth': '0.5', 'style': 'dashed', 'arrowhead': 'normal',
        'arrowtail': 'normal', 'arrowsize': '0.5', 'dir': 'both'}

    state_node_names = set()
    process_paths = []
    process_times = {}

    # initialize graph
    graph_name = 'flow'
    graph = graphviz.Digraph(graph_name, engine='fdp')
    graph.attr(size=size, overlap='false', rankdir='LR', dpi=dpi)

    # process nodes
    graph.attr('node', **process_node_spec)
    for idx, node in enumerate(bigraph_network['process_nodes']):
        y_pos = idx * 1.5
        node_path = node['path']
        process_paths.append(node_path)
        node_name = str(node_path)

        # set up timeline for this process
        sync_step = node['sync_step']
        steps = int(total_time / sync_step)
        scale_factor = 1 / sync_step
        steps_list = [round(i/scale_factor, 3) for i in range(1, steps + 1)]
        timebuffer = total_time/10
        previous_node = node_name
        process_times[node_name] = []

        # make the process node
        process_label = make_label(node_path[-1])
        graph.node(node_name, label=process_label, pos=f'{-1*timebuffer},{y_pos}!')

        # add the time nodes and connections
        graph.attr('edge', penwidth='2', arrowhead='none')
        for step in steps_list:
            time_node_name = f'{node_name} {step}'
            # place with precise positioning using the pos argument
            graph.node(time_node_name,
                       label=str(step),
                       pos=f'{step},{y_pos}!',
                       **time_node_spec
                       )
            graph.edge(previous_node, time_node_name, len=str(sync_step))
            process_times[node_name].append(time_node_name)
            previous_node = time_node_name
        # final arrow
        end_node = f'{node_name} end'
        graph.node(end_node, label='', pos=f'{step+timebuffer},{y_pos}!', **end_node_spec)
        graph.edge(previous_node, end_node, len=str(sync_step), arrowhead='normal', arrowsize='0.5')

    # time edges to shared states
    graph.attr('edge', **time_edge_spec)
    for node_path, wires in bigraph_network['hyper_edges'].items():
        node_name = str(node_path)
        with graph.subgraph(name=node_name) as c:
            for port, state_path in wires.items():
                absolute_state_path = absolute_path(node_path, state_path)
                process_node_name = str(node_path)
                state_node_name = str(absolute_state_path)
                state_node_names.add(state_node_name)

                # make the state node
                label = make_label(absolute_state_path[-1])
                graph.node(state_node_name, label=label, shape='circle', margin='0.05')

                # make edges between time nodes and the state node
                time_nodes = process_times[process_node_name]
                for time_node in time_nodes:
                    c.edge(time_node, state_node_name, constraint='false')

    # Create a subgraph to group state nodes at the top
    with graph.subgraph(name='state nodes') as top_nodes:
        top_nodes.attr(rank='source')
        for node in state_node_names:
            top_nodes.node(node)

    # display or save results
    if print_source:
        print(graph.source)
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        fig_path = os.path.join(out_dir, filename)
        print(f"Writing {fig_path}")
        graph.render(filename=fig_path, format=file_format)
    return graph
