from bigraphviz import plot_bigraph

# testing functions
plot_settings_test = {
    'plot_schema': True,
    'out_dir': 'out'
}


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


def test_merging():
    from bigraphviz.dict_utils import compose, pf
    cell_structure1 = {
        'cell': {
            'membrane': {
                'transporters': {'_type': 'concentrations'},
                'lipids': {'_type': 'concentrations'},
            },
            'cytoplasm': {
                'metabolites': {
                    '_value': 1.1, '_type': 'concentrations'
                },
                'ribosomal complexes': {
                    '_value': 2.2, '_type': 'concentrations'
                },
                'transcript regulation complex': {
                    '_value': 0.01, '_type': 'concentrations',
                    'transcripts': {
                        '_value': 0.1, '_type': 'concentrations'
                    },
                },
            },
            'nucleoid': {
                'chromosome': {
                    'genes': 'sequences'
                }
            }
        }
    }

    # add processes
    transport_process = {
        'transmembrane transport': {
            '_wires': {
                'transporters': 'transporters',
                'internal': ['..', 'cytoplasm', 'metabolites'],
            }
        }
    }
    translation_process = {
        'translation': {
            '_wires': {
                'p1': 'ribosomal complexes',
                'p2': ['transcript regulation complex', 'transcripts'],
            }
        }
    }
    cell_with_transport1 = compose(cell_structure1, node=transport_process, path=('cell', 'membrane'))
    cell_with_transport2 = compose(cell_with_transport1, node=translation_process, path=('cell', 'cytoplasm'))

    print('BEFORE')
    print(pf(cell_with_transport2['cell']['membrane']['transmembrane transport']['_wires']))
    plot_bigraph(cell_with_transport2)
    print('AFTER')
    print(pf(cell_with_transport2['cell']['membrane']['transmembrane transport']['_wires']))


if __name__ == '__main__':
    test_simple_spec()
    test_composite_spec()
    test_disconnected_process_spec()
    test_nested_spec()
    test_composite_process_spec()
    test_merging()
