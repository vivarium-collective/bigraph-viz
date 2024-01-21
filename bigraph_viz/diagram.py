"""
Bigraph diagram
"""
from bigraph_schema import TypeSystem, Edge
from bigraph_schema.registry import type_schema_keys, function_keys

SCHEMA_KEYS = type_schema_keys | set(function_keys)
EDGE_TYPES = ['process', 'step', 'edge']


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


def plot_diagram(schema):
    bigraph_network = get_flattened_graph(schema)
    print(bigraph_network)


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
    plot_diagram(cell_schema)



if __name__ == '__main__':
    test_diagram_plot()
