from bigraphviz import plot_bigraph, plot_flow, plot_multitimestep, pp
from bigraphviz.dict_utils import schema_state_to_dict


# testing functions
plot_settings_test = {
    'remove_process_place_edges': True,
    # 'plot_schema': True,
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
    plot_bigraph(composite_spec, **plot_settings_test, filename='nested_composite')


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


def test_schema_value_to_dict():
    schema = {
        'store1': {
            'store1.1': 'float',
            'store1.2': 'int'
        }
    }
    value = {
        'store1': {
            'store1.1': 1.1,
            'store1.2': 2
        }
    }
    expected = {
        'store1': {
            'store1.1': {
                '_value': 1.1,
                '_type': 'float'
            },
            'store1.2': {
                '_value': 2,
                '_type': 'int'
            }
        }
    }

    schema_state_dict = schema_state_to_dict(schema, value)
    assert schema_state_dict == expected

    schema = {
        'store1': {
            'store1.1': 'float',
            'store1.2': 'int',
            'process1': {
                '_ports': {
                    'port1': 'type',
                    'port2': 'type',
                },
            },
            'process2': {
                '_ports': {
                    'port1': 'type',
                    'port2': 'type',
                },
            },
        },
        'process3': {}
    }
    state = {
        'store1': {
            'store1.1': 1.1,
            'store1.2': 2,
            'process1': {
                '_wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
            'process2': {
                '_wires': {
                    'port1': 'store1.1',
                    'port2': 'store1.2',
                }
            },
        },
        'process3': {
            '_wires': {
                'port1': 'store1',
            }
        }
    }
    expected = {
        'store1': {
            'store1.1': {
                '_value': 1.1,
                '_type': 'float',
            },
            'store1.2': {
                '_value': 2,
                '_type': 'int',
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
        },
        'process3': {
            '_wires': {
                'port1': 'store1',
            }
        }
    }
    schema_state_dict = schema_state_to_dict(schema, state)
    pp(schema_state_dict)
    # assert schema_state_dict == expected


def test_flow():
    process_schema = {
        '_type': 'step_process',
        '_ports': {
            'port1': {'_type': 'type'},
            'port2': {'_type': 'type'}
        }
    }

    process_spec = {
        'process1': {'_depends_on': [], **process_schema},
        'process2': {'_depends_on': [], **process_schema},
        'process3': {'_depends_on': ['process1', 'process2'], **process_schema},
    }

    plot_flow(process_spec, out_dir='out', filename='flow')


def test_multitimestep():
    process_spec = {
        'process1': {
            '_ports': {'port1': {'_type': 'type'}},
            '_wires': {'port1': 'state1'},
            '_sync_step': 1.0,
        },
        'process2': {
            '_ports': {'port1': {'_type': 'type'}},
            '_wires': {'port1': 'state1'},
            '_sync_step': 0.5,
        },
        # 'process3': {
        #     '_ports': {'port1': {'_type': 'type'}},
        #     '_wires': {'port1': 'state1'},
        #     '_sync_step': 0.4,
        # },
    }
    plot_multitimestep(process_spec, total_time=3, out_dir='out', filename='multitimestep')


def test_multitimestep2():
    process_spec2 = {
        'A': {
            'process1': {
                '_ports': {'port1': {'_type': 'type'}},
                '_wires': {'port1': 'B'},
                '_sync_step': 1.0,
            },
        },
        'process2': {
            '_ports': {
                'port1': {'_type': 'type'},
                'port2': {'_type': 'type'}},
            '_wires': {
                'port1': ['A', 'B'],
                'port2': 'C',
            },
            '_sync_step': 0.5,
        },
        'process3': {
            '_ports': {'port1': {'_type': 'type'}},
            '_wires': {'port1': 'C'},
            '_sync_step': 0.6,
        },
        'D': {
            'process4': {
                '_ports': {'port1': {'_type': 'type'}},
                '_wires': {'port1': ['..', 'A', 'B']},
                '_sync_step': 0.8,
            },
        }
    }
    plot_multitimestep(process_spec2, total_time=4, out_dir='out', filename='multitimestep_2')


if __name__ == '__main__':
    # test_simple_spec()
    # test_composite_spec()
    # test_disconnected_process_spec()
    # test_nested_spec()
    # test_composite_process_spec()
    # test_merging()
    # # test_schema_value_to_dict()
    # test_flow()
    test_multitimestep()
    test_multitimestep2()
