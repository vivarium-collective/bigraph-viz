import os
import difflib
import re
from collections import defaultdict
import inspect
import graphviz
import numpy as np

from bigraph_schema import TypeSystem, is_schema_key, hierarchy_depth, Edge
from bigraph_viz.dict_utils import absolute_path
from process_bigraph import allocate_core

# Constants
PROCESS_SCHEMA_KEYS = [
    'config', 'address', 'interval', 'inputs', 'outputs', 'instance', 'bridge']


# Utility: Label formatting
def make_label(label):
    """Wrap a label in angle brackets for Graphviz HTML rendering."""
    return f'<{label}>'


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
def add_node_to_graph(
        graph,
        node,
        state_node_spec,
        show_values,
        show_types,
        significant_digits,
        value_char_limit,
        type_char_limit,
):
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
    value_char_limit = value_char_limit or 20
    type_char_limit = type_char_limit or 20

    rows = [label]  # the node's name, already plain text

    # Value row
    if show_values and (val := node.get('value')) is not None:
        if isinstance(val, float):
            val = int(val) if val.is_integer() else round(val, significant_digits)
        val_str = str(val)
        if len(val_str) > value_char_limit:
            val_str = val_str[:value_char_limit] + "…"
        rows.append(f"<FONT POINT-SIZE='10' COLOR='gray30'>value: {val_str}</FONT>")

    # Type row
    if show_types and (typ := node.get('type')):
        typ_str = str(typ)
        if len(typ_str) > type_char_limit:
            typ_str = typ_str[:type_char_limit] + "…"
        rows.append(f"<FONT POINT-SIZE='10' COLOR='gray40'>type: {typ_str}</FONT>")

    html_label = "<<TABLE BORDER='0' CELLBORDER='0' CELLSPACING='0'>"
    for r in rows:
        html_label += f"<TR><TD>{r}</TD></TR>"
    html_label += "</TABLE>>"

    # full_label = make_label(label + label_info) if label_info else make_label(label)
    graph.attr('node', **state_node_spec)
    graph.node(node_name, label=html_label)
    return node_name


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
        value_char_limit=20,
        type_char_limit=50,
        port_labels=True,
        port_label_size='10pt',
        invisible_edges=None,
        remove_process_place_edges=False,
        node_border_colors=None,
        node_fill_colors=None,
        node_groups=None,
        collapse_redundant_processes=False,
        collapse_paths=None,
        remove_paths=None,
):
    """
    Generate a Graphviz Digraph from a graph_dict describing a simulation bigraph.

    Parameters
    ----------
    graph_dict : dict
        Dictionary describing nodes and edges of a simulation bigraph.

    collapse_redundant_processes : bool | str | Iterable | dict
        Controls collapsing of processes that share identical port wiring:

        - False / None        : collapse no processes
        - True or "all"       : collapse all redundant processes
        - Iterable            : collapse only processes matching the selectors
        - {"exclude": it}     : collapse all redundant processes except those
                                matching selectors in `it`

        A selector can be:
          * str   : matches the leaf process name or str(path)
          * tuple : matches the exact process path

    collapse_paths : Iterable[Iterable[str]] | Iterable[str] | str | None
        Hierarchical paths to collapse as subtrees. For each prefix path P,
        all nodes whose path starts with P and is strictly deeper than P are
        removed, while the node at P is kept.

        Example:
            collapse_paths=[['particles']]
            keeps:   ['particles']
            removes: ['particles', 'abc'], ['particles', 'abc', 'def'], ...

    remove_paths : Iterable[Iterable[str]] | Iterable[str] | str | None
        Hierarchical paths to fully remove. For each prefix path P,
        nodes whose path starts with P (including P itself) are removed.

        Example:
            remove_paths=[['particles']]
            removes: ['particles'], ['particles', 'abc'], ...

    Returns
    -------
    graphviz.Digraph
    """

    invisible_edges = invisible_edges or []
    node_groups = node_groups or []
    process_label_size = process_label_size or node_label_size

    graph = graphviz.Digraph(name='bigraph', engine='dot')
    graph.attr(size=size, overlap='false', rankdir=rankdir, dpi=dpi,
               ratio=aspect_ratio, splines='true')

    # Use the same label_margin parameter, but circles get a smaller
    # effective margin so they don't balloon visually.
    state_margin = str(float(label_margin) * 0.1)
    process_margin = label_margin

    # Node styles
    state_node_spec = {
        'shape': 'circle',
        'penwidth': '2',
        'constraint': 'false',
        'margin': state_margin,
        'fontsize': node_label_size,
    }
    process_node_spec = {
        'shape': 'box',
        'penwidth': '2',
        'constraint': 'false',
        'margin': process_margin,
        'fontsize': process_label_size,
    }

    # Edge styles
    edge_styles = {
        'input': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal',
                  'arrowsize': '1.0', 'dir': 'forward'},
        'output': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal',
                   'arrowsize': '1.0', 'dir': 'back'},
        'bidirectional': {'style': 'dashed', 'penwidth': '1', 'arrowhead': 'normal',
                          'arrowsize': '1.0', 'dir': 'both'},
        'place': {'arrowhead': 'none', 'penwidth': '2'},
    }
    if undirected_edges:
        for spec in edge_styles.values():
            spec['dir'] = 'none'

    node_names = []

    # -------- collapse configuration: redundant processes -------------------

    def normalize_collapse_arg(arg):
        """
        Return (mode, selectors) where mode is:
          'none'       : collapse nothing
          'all'        : collapse all redundant processes
          'subset'     : collapse only selected processes
          'all_except' : collapse all except selected processes
        """
        if arg is False or arg is None:
            return 'none', set()

        if arg is True or arg == 'all':
            return 'all', set()

        if isinstance(arg, dict) and 'exclude' in arg:
            try:
                selectors = set(arg['exclude'])
            except TypeError:
                selectors = {arg['exclude']}
            return 'all_except', selectors

        # Iterable or single selector ⇒ subset
        try:
            selectors = set(arg)
        except TypeError:
            selectors = {arg}
        return 'subset', selectors

    collapse_mode, collapse_selectors = normalize_collapse_arg(
        collapse_redundant_processes
    )

    def process_matches_selector(entry, selector):
        """Check if a process entry matches a single selector."""
        path, path_str, name = entry

        if isinstance(selector, (tuple, list)):
            return tuple(selector) == tuple(path)

        return selector == name or selector == path_str

    def process_is_selected(entry):
        """Return True if this process is eligible to be collapsed."""
        if collapse_mode == 'all':
            return True
        if collapse_mode == 'none':
            return False
        if not collapse_selectors:
            return collapse_mode == 'all'

        matches = any(
            process_matches_selector(entry, sel)
            for sel in collapse_selectors
        )

        if collapse_mode == 'subset':
            return matches
        if collapse_mode == 'all_except':
            return not matches
        return False

    # -------- collapse/remove configuration: hierarchical subtrees ---------

    def normalize_prefixes(arg):
        """
        Normalize a 'paths' argument into a list of tuple prefixes.

        Examples:
          None                 -> []
          'particles'          -> [('particles',)]
          ['particles']        -> [('particles',)]
          [['particles']]      -> [('particles',)]
          [['a'], ['b', 'c']]  -> [('a',), ('b', 'c')]
        """
        if not arg:
            return []

        # Strings are treated as single-segment paths
        if isinstance(arg, (str, bytes)):
            return [(arg,)]

        # Try to interpret it as an iterable
        try:
            items = list(arg)
        except TypeError:
            return [(arg,)]

        if not items:
            return []

        # If first element is not a list/tuple, treat whole thing as one path
        if not isinstance(items[0], (list, tuple)):
            return [tuple(items)]

        # Otherwise each element is a path
        prefixes = []
        for p in items:
            if isinstance(p, (list, tuple)):
                prefixes.append(tuple(p))
            else:
                prefixes.append((p,))
        return prefixes

    collapsed_prefixes = normalize_prefixes(collapse_paths)
    removed_prefixes = normalize_prefixes(remove_paths)

    def path_is_hidden(path):
        """
        Return True if this path should be removed from the graph_dict
        based on collapse/remove prefixes.

        - remove_paths: hide root AND descendants
        - collapse_paths: hide ONLY descendants, keep root
        """
        # If paths are always lists/tuples, you can assert here.
        if isinstance(path, str):
            # If you ever represent paths as strings, customize as needed.
            # For now, treat them as non-hierarchical => never auto-hide.
            return False

        path_t = tuple(path)

        # 1) Removal: root + subtree
        for pref in removed_prefixes:
            if path_t[:len(pref)] == pref:
                return True

        # 2) Collapse: only descendants
        for pref in collapsed_prefixes:
            if len(path_t) > len(pref) and path_t[:len(pref)] == pref:
                return True

        return False

    def prune_subtrees():
        """
        Remove all nodes and edges that should be hidden according to
        collapse_paths and remove_paths.

        - For removed_prefixes: delete root and descendants.
        - For collapsed_prefixes: delete descendants only (root kept).
        """
        if not collapsed_prefixes and not removed_prefixes:
            return

        # State and process nodes
        graph_dict['state_nodes'] = [
            n for n in graph_dict.get('state_nodes', [])
            if not path_is_hidden(n['path'])
        ]
        graph_dict['process_nodes'] = [
            n for n in graph_dict.get('process_nodes', [])
            if not path_is_hidden(n['path'])
        ]

        def filter_edge_list(edges):
            new_edges = []
            for e in edges:
                if path_is_hidden(e['edge_path']):
                    continue
                tpath = e.get('target_path')
                if isinstance(tpath, (list, tuple)) and path_is_hidden(tpath):
                    continue
                new_edges.append(e)
            return new_edges

        for group in [
            'input_edges',
            'output_edges',
            'bidirectional_edges',
            'disconnected_input_edges',
            'disconnected_output_edges',
        ]:
            if group in graph_dict:
                graph_dict[group] = filter_edge_list(graph_dict[group])

        # Place edges
        if 'place_edges' in graph_dict:
            new_place_edges = []
            for e in graph_dict['place_edges']:
                if path_is_hidden(e['parent']) or path_is_hidden(e['child']):
                    continue
                new_place_edges.append(e)
            graph_dict['place_edges'] = new_place_edges

    # -------- core helpers --------------------------------------------------

    def get_name_template(names):
        """Create a generalized label for collapsed process names."""
        if len(names) == 1:
            return names[0]
        prefix = os.path.commonprefix(names)
        suffix = os.path.commonprefix([n[::-1] for n in names])[::-1]
        middle = '*' if prefix != names[0] or suffix != names[0] else ''
        return f"{prefix}{middle}{suffix}"

    def add_state_nodes():
        graph.attr('node', **state_node_spec)
        for node in graph_dict['state_nodes']:
            name = add_node_to_graph(
                graph, node, state_node_spec,
                show_values, show_types, significant_digits,
                value_char_limit, type_char_limit,
            )
            node_names.append(name)

    def add_process_nodes():
        """
        Add process nodes, with optional collapse of redundant ones
        according to collapse_mode and collapse_selectors.
        """
        graph.attr('node', **process_node_spec)
        process_fingerprints = defaultdict(list)

        # Group processes by connectivity "fingerprint"
        for node in graph_dict['process_nodes']:
            node_path = node['path']
            path_str = str(node_path)
            node_name = node_path[-1]

            fp = []
            for group, tag in [
                ('input_edges', 'in'),
                ('output_edges', 'out'),
                ('bidirectional_edges', 'both'),
            ]:
                for edge in graph_dict.get(group, []):
                    if edge['edge_path'] == node_path:
                        fp.append((tag, edge['port'], str(edge.get('target_path'))))
            fingerprint = tuple(sorted(fp))
            process_fingerprints[fingerprint].append((node_path, path_str, node_name))

        collapse_map = {}

        # For each fingerprint group, possibly collapse some or all
        for fingerprint, entries in process_fingerprints.items():
            selected = [e for e in entries if process_is_selected(e)]

            if len(selected) > 1:
                # Create collapsed representative
                names = [e[2] for e in selected]
                label = f"{get_name_template(names)} (x{len(selected)})"
                rep_path = selected[0][0]
                rep_str = str(rep_path)

                graph.node(rep_str, label=label)
                node_names.append(rep_str)

                # Map collapsed entries (except representative)
                for path, path_str, _ in selected[1:]:
                    collapse_map[str(path)] = rep_path

                # Draw remaining (unselected) entries individually
                remaining = [e for e in entries if e not in selected]
                for path, path_str, name in remaining:
                    graph.node(path_str, label=name)
                    node_names.append(path_str)
            else:
                # No effective collapse here: draw all individually
                for path, path_str, name in entries:
                    graph.node(path_str, label=name)
                    node_names.append(path_str)

        # Return original process paths and collapse map
        return [
            entry[0]
            for entries in process_fingerprints.values()
            for entry in entries
        ], collapse_map

    def rewrite_collapsed_edges(collapse_map):
        """Update edge endpoints to point to collapsed representatives."""
        removed_keys = set(collapse_map.keys())
        if not removed_keys:
            return

        for group in [
            'input_edges',
            'output_edges',
            'bidirectional_edges',
            'disconnected_input_edges',
            'disconnected_output_edges',
        ]:
            edges = graph_dict.get(group, [])
            new_edges, seen = [], set()
            for edge in edges:
                key = str(edge['edge_path'])
                if key in collapse_map:
                    edge['edge_path'] = collapse_map[key]
                if key not in removed_keys:
                    edge_key = (
                        group,
                        str(edge['edge_path']),
                        edge.get('port'),
                        str(edge.get('target_path')),
                    )
                    if edge_key not in seen:
                        seen.add(edge_key)
                        new_edges.append(edge)
            graph_dict[group] = new_edges

        # Remove place edges involving removed processes
        new_place_edges = []
        for edge in graph_dict.get('place_edges', []):
            parent_str = str(edge['parent'])
            child_str = str(edge['child'])
            if parent_str in removed_keys or child_str in removed_keys:
                continue
            new_place_edges.append(edge)
        graph_dict['place_edges'] = new_place_edges

    def add_edges(edge_groups):
        for group, style_key in edge_groups:
            for edge in graph_dict.get(group, []):
                if 'bridge_outputs' in edge['type']:
                    style, constraint = 'input', 'false'
                elif 'bridge_inputs' in edge['type']:
                    style, constraint = 'output', 'false'
                else:
                    style, constraint = style_key, 'true'
                graph.attr('edge', **edge_styles[style])
                plot_edges(
                    graph, edge, port_labels, port_label_size,
                    state_node_spec, constraint=constraint,
                )

    def add_place_edges(process_paths):
        for edge in graph_dict.get('place_edges', []):
            visible = not (
                (remove_process_place_edges and edge['child'] in process_paths)
                or (edge in invisible_edges)
            )
            graph.attr('edge', style='filled' if visible else 'invis')
            graph.edge(
                str(edge['parent']),
                str(edge['child']),
                **edge_styles['place'],
                constraint='true',
            )

    def add_disconnected_edges():
        for direction, style_key in [
            ('disconnected_input_edges', 'input'),
            ('disconnected_output_edges', 'output'),
        ]:
            for edge in graph_dict.get(direction, []):
                path = edge['edge_path']
                port = edge['port']
                suffix = '_input' if 'input' in direction else '_output'
                dummy = str(absolute_path(path, port)) + suffix
                graph.node(dummy, label='', style='invis', width='0')
                edge['target_path'] = dummy
                graph.attr('edge', **edge_styles[style_key])
                plot_edges(
                    graph, edge, port_labels, port_label_size,
                    state_node_spec, constraint='true',
                )

    def rank_node_groups():
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

    def apply_custom_colors():
        state_paths = [entry['path'] for entry in graph_dict['state_nodes']]
        process_paths = [entry['path'] for entry in graph_dict['process_nodes']]

        if node_border_colors:
            for name, color in node_border_colors.items():
                if name in state_paths or name in process_paths:
                    graph.node(str(name), color=color)
        if node_fill_colors:
            for name, color in node_fill_colors.items():
                if name in state_paths or name in process_paths:
                    graph.node(str(name), color=color, style='filled')

    # -------- build graph ---------------------------------------------------

    # First, apply hierarchical collapse/remove rules
    prune_subtrees()

    add_state_nodes()
    process_paths, collapse_map = add_process_nodes()
    rewrite_collapsed_edges(collapse_map)
    add_place_edges(process_paths)
    add_edges([
        ('input_edges', 'input'),
        ('output_edges', 'output'),
        ('bidirectional_edges', 'bidirectional'),
    ])
    # TODO: the second add_state_nodes() looks redundant
    # add_state_nodes()
    add_disconnected_edges()
    rank_node_groups()
    apply_custom_colors()

    return graph


def plot_graph(graph_dict,
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


def plot_bigraph(
        state,
        schema=None,
        core=None,
        out_dir=None,
        filename=None,
        file_format='png',
        show_compiled_state=False,
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
    core = core or allocate_core()

    schema = schema or {}
    compiled_schema, compiled_state = core.deserialize(schema, state)

    graph_dict = core.call_method(
        'generate_graph_dict',
        compiled_schema,
        compiled_state if show_compiled_state else state,
        (), options=traversal_kwargs)

    return plot_graph(
        graph_dict,
        filename=filename,
        out_dir=out_dir,
        file_format=file_format,
        options=render_kwargs
    )

    # graph_dict = core.generate_graph_dict(schema, state, (), options=traversal_kwargs)

# # Visualize Types
# def graphviz_any(core, schema, state, path, options, graph):
#     """Visualize any type (generic node)."""
#     schema = schema or {}

#     if path:
#         node_spec = {
#             'name': path[-1],
#             'path': path,
#             'value': state if not isinstance(state, dict) else None,
#             'type': core.representation(schema)
#         }
#         graph['state_nodes'].append(node_spec)

#     if len(path) > 1:
#         graph['place_edges'].append({'parent': path[:-1], 'child': path})

#     if isinstance(state, dict):
#         for key, value in state.items():
#             if not is_schema_key(key):
#                 graph = core.get_graph_dict(
#                     schema.get(key, {}),
#                     value,
#                     path + (key,),
#                     options,
#                     graph
#                 )

#     return graph


# def graphviz_edge(core, schema, state, path, options, graph):
#     """Visualize a process node with input/output/bridge wiring."""
#     schema = schema or {}
#     node_spec = {
#         'name': path[-1],
#         'path': path,
#         'value': None,
#         'type': core.representation(schema)
#     }

#     if state.get('address') == 'local:Composite' and node_spec not in graph['process_nodes']:
#         graph['process_nodes'].append(node_spec)
#         return graphviz_composite(core, schema, state, path, options, graph)

#     graph['process_nodes'].append(node_spec)

#     # Wiring
#     graph = get_graph_wires(schema.get('_inputs', {}), state.get('inputs', {}), graph, 'inputs', path,
#                             state.get('bridge', {}).get('inputs', {}))
#     graph = get_graph_wires(schema.get('_outputs', {}), state.get('outputs', {}), graph, 'outputs', path,
#                             state.get('bridge', {}).get('outputs', {}))

#     # Merge bidirectional edges
#     def key(edge):
#         return (tuple(edge['edge_path']), tuple(edge['target_path']), edge['port'])

#     input_set = {key(e): e for e in graph['input_edges']}
#     output_set = {key(e): e for e in graph['output_edges']}
#     shared_keys = input_set.keys() & output_set.keys()
#     for k in shared_keys:
#         graph['bidirectional_edges'].append({
#             'edge_path': k[0], 'target_path': k[1], 'port': k[2],
#             'type': (input_set[k]['type'], output_set[k]['type'])
#         })
#     graph['input_edges'] = [e for k, e in input_set.items() if k not in shared_keys]
#     graph['output_edges'] = [e for k, e in output_set.items() if k not in shared_keys]

#     if len(path) > 1:
#         graph['place_edges'].append({'parent': path[:-1], 'child': path})

#     return graph


# def graphviz_none(core, schema, state, path, options, graph):
#     """No-op visualizer for nodes with no visualization."""
#     return graph


# def graphviz_map(core, schema, state, path, options, graph):
#     """Visualize mappings by traversing key–value pairs."""

#     value_type = core._find_parameter(schema, 'value')

#     # Add node for the map container itself
#     if path:
#         node_spec = {
#             'name': path[-1],
#             'path': path,
#             'value': None,
#             'type': core.representation(schema)
#         }
#         graph['state_nodes'].append(node_spec)

#     # Add place edge to parent
#     if len(path) > 1:
#         graph['place_edges'].append({'parent': path[:-1], 'child': path})

#     if isinstance(state, dict):
#         for key, value in state.items():
#             if not is_schema_key(key):
#                 graph = core.get_graph_dict(
#                     value_type,
#                     value,
#                     path + (key,),
#                     options,
#                     graph
#                 )
#     return graph


# def graphviz_composite(core, schema, state, path, options, graph):
#     """Visualize composite nodes by recursing into their internal structure."""
#     graph = graphviz_edge(core, schema, state, path, options, graph)

#     inner_state = state.get('config', {}).get('state') or state
#     inner_schema = state.get('config', {}).get('composition') or schema
#     inner_schema, inner_state = core.generate(inner_schema, inner_state)

#     if len(path) > 1:
#         graph['place_edges'].append({'parent': path[:-1], 'child': path})

#     for key, value in inner_state.items():
#         if not is_schema_key(key) and key not in PROCESS_SCHEMA_KEYS:
#             graph = core.get_graph_dict(
#                 inner_schema.get(key),
#                 value,
#                 path + (key,),
#                 options,
#                 graph
#             )

#     return graph


# # dict with different types and their graphviz functions
# visualize_types = {
#     'any': {
#         '_graphviz': graphviz_any
#     },
#     'edge': {
#         '_graphviz': graphviz_edge
#     },
#     'quote': {
#         '_graphviz': graphviz_none,
#     },
#     'map': {
#         '_graphviz': graphviz_map,
#     },
#     'step': {
#         '_inherit': ['edge']
#     },
#     'process': {
#         '_inherit': ['edge']
#     },
#     'composite': {
#         '_inherit': ['process'],
#         '_graphviz': graphviz_composite,
#     },
# }


# TODO: we want to visualize things that are not yet complete

# class VisualizeTypes(TypeSystem):
#     def __init__(self):
#         super().__init__()

#         self.update_types(visualize_types)

#     def get_graph_dict(self, schema, state, path, options, graph=None):
#         path = path or ()
#         graph = graph or {
#             'state_nodes': [],
#             'process_nodes': [],
#             'place_edges': [],
#             'input_edges': [],
#             'output_edges': [],
#             'bidirectional_edges': [],
#             'disconnected_input_edges': [],
#             'disconnected_output_edges': []}

#         graphviz_function = self.choose_method(
#             schema,
#             state,
#             'graphviz')

#         if options.get('remove_nodes') and path in options['remove_nodes']:
#             return graph

#         return graphviz_function(
#             self,
#             schema,
#             state,
#             path,
#             options,
#             graph)

#     def generate_graph_dict(self, schema, state, path, options):
#         full_schema, full_state = self.generate(schema, state)
#         return self.get_graph_dict(full_schema, full_state, path, options)



# Begin Tests
###############

plot_settings = {
    # 'out_dir': 'out',
    'dpi': '150',
}


def test_simple_store(core):
    simple_store_state = {
        'store1': 1.0,
    }
    plot_bigraph(simple_store_state,
                 **plot_settings,
                 show_values=True,
                 filename='simple_store')


def test_forest(core):
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

    plot_bigraph(forest, **plot_settings, filename='forest', core=core)


def test_nested_composite(core):
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
                    'address': 'local:Composite',
                    'config': {'_type': 'node',
                               'state': {'grow': {'_type': 'process',
                                                  'address': 'local:Grow',
                                                  'config': {'rate': 0.03},
                                                  'inputs': {'mass': ['mass']},
                                                  'outputs': {'mass': ['mass']}},
                                         'divide': {'_type': 'process',
                                                    'address': 'local:Divide',
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
                                           'address': 'local:RAMEmitter',
                                           'config': {},
                                           'mode': 'none',
                                           'emit': {}},
                               'global_time_precision': None}
                }}}}

    plot_bigraph(
        state,
        core=core,
        filename='nested_composite',
        **plot_settings)


def test_graphviz(core):
    cell = {
        'config': {
            '_type': 'map[float]',
            'a': 11.0,  # {'_type': 'float', '_value': 11.0},
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            # 'config': {},
            # 'address': 'local:Cell',   # TODO -- this is where the ports/inputs/outputs come from
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

    graph_dict = core.call_method('generate_graph_dict',
        {},
        cell,
        (),
        options={'dpi': '150'})

    plot_graph(
        graph_dict,
        out_dir='out',
        filename='test_graphviz'
    )


class Cell(Edge):
    def inputs(self):
        return {
            'nutrients': 'float',
        }


    def outputs(self):
        return {
            'secretions': 'float',
            'biomass': 'float',
        }


def test_bigraph_cell(core):
    cell = {
        'config': {
            '_type': 'map[float]',
            'a': 11.0,  # {'_type': 'float', '_value': 11.0},
            'b': 3333.33},
        'cell': {
            '_type': 'process',  # TODO -- this should also accept process, step, but how in bigraph-schema?
            'config': {},
            'address': 'local:Cell',  # TODO -- this is where the ports/inputs/outputs come from
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
                 core=core,
                 show_values=True,
                 # show_types=True,
                 **plot_settings
                 )


def test_bio_schema(core):
    b = {
        'environment': {
            'cells': {
                'cell1': {
                    'cytoplasm': {},
                    'nucleus': {
                        'chromosome': {},
                        'transcription': {
                            '_type': 'process',
                            '_inputs': {'DNA': 'node'},
                            '_outputs': {'RNA': 'node'},
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
            'barriers': {
                '_type': 'node'},
            'diffusion': {
                '_type': 'process',
                '_inputs': {'fields': 'node'},
                '_outputs': {'fields': 'node'},
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


def test_flat_composite(core):
    flat_composite_spec = {
        'store1.1': 'float',
        'store1.2': 'integer',
        'process1': {
            '_type': 'process',
            '_outputs': {
                'port1': 'node',
                'port2': 'node',
            },
            'outputs': {
                'port1': ['store1.1'],
                'port2': ['store1.2'],
            }
        },
        'process2': {
            '_type': 'process',
            '_inputs': {
                'port1': 'node',
                'port2': 'node',
            },
            'inputs': {
                'port1': ['store1.1'],
                'port2': ['store1.2'],
            }
        },
    }

    plot_bigraph(flat_composite_spec,
                 core=core,
                 rankdir='RL',
                 filename='flat_composite',
                 **plot_settings)


def test_multi_processes(core):
    process_schema = {
        '_type': 'process',
        '_inputs': {
            'port1': 'node',
        },
        '_outputs': {
            'port2': 'node'
        },
    }

    processes_spec = {
        'process1': process_schema,
        'process2': process_schema,
        'process3': process_schema,
    }

    plot_bigraph(processes_spec,
                 core=core,
                 rankdir='BT',
                 filename='multiple_processes',
                 **plot_settings)


def test_nested_processes(core):
    nested_process_spec = {
        'store1': {
            'store1.1': 'float',
            'store1.2': 'integer',
            'process1': {
                '_type': 'process',
                '_inputs': {
                    'port1': 'node',
                    'port2': 'node',
                },
                'inputs': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            },
            'process2': {
                '_type': 'process',
                '_outputs': {
                    'port1': 'node',
                    'port2': 'node',
                },
                'outputs': {
                    'port1': ['store1.1'],
                    'port2': ['store1.2'],
                }
            },
        },
        'process3': {
            '_type': 'process',
            '_inputs': {
                'port1': 'node',
            },
            'inputs': {
                'port1': ['store1'],
            }
        }
    }

    plot_bigraph(nested_process_spec,
                 **plot_settings,
                 core=core,
                 filename='nested_processes')


def test_cell_hierarchy(core):
    core.register_type('concentrations', 'float')
    core.access('concentrations')

    core.register_type('sequences', 'float')
    core.register_type('membrane', {
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

    core.register_type('cytoplasm', {
        'metabolites': 'concentrations',
        'ribosomal complexes': 'concentrations',
        'transcript regulation complex': {
            'transcripts': 'concentrations'},
        'translation': {
            '_type': 'process',
            '_outputs': {
                'p1': 'concentrations',
                'p2': 'concentrations'}}})

    core.register_type('nucleoid', {
        'chromosome': {
            'genes': 'sequences'}})

    core.register_type('cell', {
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
        show_compiled_state=True,
        **plot_settings)


def test_multiple_disconnected_ports(core):
    spec = {
        'process': {
            '_type': 'process',
            '_inputs': {
                'port1': 'node',
                'port2': 'node',
            },
            '_outputs': {
                'port1': 'node',
                'port2': 'node',
            },
        },
    }

    plot_bigraph(
        spec,
        core=core,
        # out_dir='out',
        filename='multiple_disconnected_ports',
        **plot_settings)


def test_composite_process(core):
    spec = {
        'composite': {
            '_type': 'process',
            '_inputs': {'port1': 'node'},
            '_outputs': {'port2': 'node'},
            'address': 'local:Composite',
            'inputs': {'port1': ['external store']},
            'config': {
                'state': {
                    'store1': 'node',
                    'store2': 'node',
                    'process1': {
                        '_type': 'process',
                        '_inputs': {'port3': 'node'},
                        '_outputs': {'port4': 'node'},
                        'inputs': {'port3': ['store1']},
                        'outputs': {'port4': ['store2']}}},
                'bridge': {
                    'inputs': {'port1': ['store1']},
                    'outputs': {'port2': ['store2']}}}}}

    plot_bigraph(
        spec,
        core=core,
        filename='composite_process',
        **plot_settings)


def test_bidirectional_edges(core):
    spec = {
        'process1': {
            '_type': 'process',
            '_inputs': {'port1': 'node'},
            '_outputs': {'port1': 'node'},
            'inputs': {'port1': ['external store']},
            'outputs': {'port1': ['external store']}},
        'process2': {
            '_type': 'process',
            '_inputs': {'port3': 'node'},
            '_outputs': {'port4': 'node'},
            'inputs': {'port3': ['external store']},
            'outputs': {'port4': ['external store']}
        }
    }

    plot_bigraph(
        spec,
        core=core,
        filename='bidirectional_edges',
        **plot_settings)


class DynamicFBA(Edge):
    def inputs(self):
        return {
            'substrates': 'map[float]',
        }


    def outputs(self):
        return {
            'biomass': 'float',
            'substrates': 'map[float]',
        }


def generate_spec_and_schema(n_rows, n_cols):
    spec = {'cells': {}}
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
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'glucose': ['..', 'fields', 'glucose', i, j],
                    },
                    'biomass': ['..', 'fields', 'biomass', i, j]
                },
                'outputs': {
                    'substrates': {
                        'acetate': ['..', 'fields', 'acetate', i, j],
                        'glucose': ['..', 'fields', 'glucose', i, j],
                    },
                    'biomass': ['..', 'fields', 'biomass', i, j]
                }
            }
            spec['cells'][name] = cell_spec

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


def test_array_paths(core):
    n_rows, n_cols = 2, 1  # or any desired shape
    spec, schema = generate_spec_and_schema(n_rows, n_cols)

    plot_bigraph(
        spec,
        schema=schema,
        core=core,
        filename='array_paths',
        **plot_settings)


def test_complex_bigraph(core):
    n_rows, n_cols = 6, 6  # or any desired shape
    spec, schema = generate_spec_and_schema(n_rows, n_cols)

    plot_settings['dpi'] = '500'
    plot_bigraph(
        spec,
        schema=schema,
        core=core,
        filename='complex_bigraph',
        collapse_redundant_processes=True,
        # dpi='200',
        **plot_settings)


class Particles(Edge):
    def inputs(self):
        return {
            'particles': 'map[particle]',
            'fields': 'map[array]',
        }


    def outputs(self):
        return {
            'particles': 'map[particle]',
            'fields': 'map[array]',
        }


def test_nested_particle_process(core):
    core.register_type('particle', {
        'id': 'string',
        'position': 'tuple[float,float]',
        'size': 'float',
        'mass': 'float',
        'local': 'map[float]',
        'exchange': 'map[float]'
    })

    state = {
        "particles": {
            "rddyhz3IRHaZIKnpyROvGw": {
                "id": "rddyhz3IRHaZIKnpyROvGw",
                "position": ["1.6170202476993778", "2.6965198046441277"],
                "size": "0.0",
                "mass": "0.7708618958003092",
                "local": {
                    "glucose": "2.1129479416859507", "acetate": "0.0"},
                "exchange": {
                    "glucose": "0.0", "acetate": "0.0"},
                "dFBA": {
                    "inputs": {
                        "substrates": ["local"],
                        "biomass": ["mass"]},
                    "outputs": {
                        "substrates": ["exchange"],
                        "biomass": ["mass"]},
                    "interval": 1.0,
                    "address": "local:DynamicFBA",
                    "config": {
                        "model_file": "textbook",
                        "kinetic_params": {
                            "glucose": ["0.5", "1.0"],
                            "acetate": ["0.5", "2.0"]},
                        "substrate_update_reactions": {
                            "glucose": "EX_glc__D_e",
                            "acetate": "EX_ac_e"},
                        "bounds": {
                            "EX_o2_e": {
                                "lower": "-2.0",
                                "upper": "!nil"},
                            "ATPM": {
                                "lower": "1.0",
                                "upper": "1.0"}}},
                    "shared": {}
                }
            },
        },
        "global_time": "0.0",
        "particle_movement": {
            '_type': 'process',
            "inputs": {
                "particles": ["particles"],
                "fields": ["fields"]},
            "outputs": {
                "particles": ["particles"],
                "fields": ["fields"]},
            "interval": 1.0,
            "address": "local:Particles",
            "config": {},
        },
        "fields": {
            "glucose": {
                "list": [
                    [
                        6.682473038698228,
                        6.138508047074471,
                        5.932822055376635,
                        1.2275655546440918,
                        7.184289576021444,
                        5.802540321285436,
                        3.158370346023715,
                        6.191878825605585,
                        7.417057892118427,
                        9.619194357104389
                    ],
                    [
                        7.384059587748178,
                        8.640811575012702,
                        2.1129479416859507,
                        3.1057148920618385,
                        8.05289155553335,
                        4.399086558257299,
                        7.193948745260049,
                        5.035688862165517,
                        5.219923411699781,
                        4.539653707209219
                    ],
                    [
                        1.9082739815105203,
                        8.99864529956811,
                        6.5195511089756675,
                        1.957101992521509,
                        3.1907575508941806,
                        4.5623876367245195,
                        9.68212403622312,
                        4.419905700021853,
                        8.71921956750521,
                        8.163620432115913
                    ],
                    [
                        9.478709499822921,
                        8.675323729741326,
                        4.226239950967384,
                        1.6634413475982608,
                        4.2399161776694365,
                        1.8032704088212483,
                        8.029077191030485,
                        3.5987760418726706,
                        1.0827071604629657,
                        4.939077862941639
                    ],
                    [
                        2.2055610319311745,
                        8.052040372657313,
                        5.325939740703682,
                        5.915877670139153,
                        9.335157497844655,
                        5.8973761480277584,
                        7.503363745465629,
                        1.2530328598811558,
                        7.58703729093062,
                        2.4382507126877866
                    ]
                ],
                "data": "float",
                "shape": [
                    5,
                    10
                ]
            },
            "acetate": {
                "list": [
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0
                    ]
                ],
                "data": "float",
                "shape": [
                    5,
                    10
                ]
            }
        }
    }
    composition = {
        'particles': {
            '_type': 'map',
            '_value': {
                'dFBA': {'_type': 'process',
                         'address': {'_type': 'string', '_default': 'local:DynamicFBA'},
                         'config': {'_type': 'node', '_default': {
                             'model_file': 'textbook',
                             'kinetic_params': {'glucose': (0.5, 1), 'acetate': (0.5, 2)},
                             'substrate_update_reactions': {
                                 'glucose': 'EX_glc__D_e',
                                 'acetate': 'EX_ac_e'},
                             'bounds': {
                                 'EX_o2_e': {'lower': -2, 'upper': None},
                                 'ATPM': {'lower': 1, 'upper': 1}}}},
                         'inputs': {'_type': 'tree[wires]', '_default': {
                             'substrates': ['local'], 'biomass': ['mass']}},
                         'outputs': {'_type': 'tree[wires]', '_default': {
                             'substrates': ['exchange'],
                             'biomass': ['mass']}}}}}}

    plot_bigraph(state=state, schema=composition, core=core,
                 filename='nested_particle_process',
                 **plot_settings,
                 )


if __name__ == '__main__':
    core = allocate_core()

    test_simple_store(core)
    test_forest(core)
    test_nested_composite(core)
    test_graphviz(core)
    test_bigraph_cell(core)
    test_bio_schema(core)
    test_flat_composite(core)
    test_multi_processes(core)
    test_nested_processes(core)
    test_cell_hierarchy(core)
    test_multiple_disconnected_ports(core)
    test_composite_process(core)
    test_bidirectional_edges(core)
    test_array_paths(core)
    test_complex_bigraph(core)
    test_nested_particle_process(core)
