import os

from bigraph_schema import TypeSystem, is_schema_key
from bigraph_viz.dict_utils import absolute_path
from bigraph_viz.diagram import generate_types, plot_bigraph, get_graphviz_fig

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

    core.plot_graph(
        graphviz,
        graph_settings['plot'])


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
