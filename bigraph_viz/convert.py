"""
=======
Convert
=======
"""

import copy

from bigraph_viz import plot_bigraph
from bigraph_viz.dict_utils import deep_merge, absolute_path, nest_path


plot_settings_test = {
    'remove_process_place_edges': True,
    'show_values': True,
    'show_types': True,
    'dpi': '250',
    'out_dir': 'out'
}


def merge_wires(processes, topology, path=None):
    path = path or ()
    wired_processes = copy.deepcopy(processes)
    extended_state = {}
    for k, v in wired_processes.items():
        current_path = path + (k,)
        sub_topology = topology[k]
        if isinstance(v, dict):
            nested_processes = merge_wires(v, sub_topology, current_path)
            extended_state = deep_merge(nested_processes, extended_state)
        else:
            wired_processes[k] = {'wires': sub_topology}
            for port, port_path in sub_topology.items():
                absolute_state_path = absolute_path(path, port_path)
                nested_processes = nest_path({}, absolute_state_path)
                extended_state = deep_merge(nested_processes, extended_state)
    wired_processes = nest_path(wired_processes, path)
    wired_processes = deep_merge(wired_processes, extended_state)

    return wired_processes


def convert_vivarium_composite(vivarium_composite):
    composite_keys = set(vivarium_composite.keys())
    allowed_keys = set(['processes', 'topology', 'flow', 'steps', 'state', '_schema'])
    assert composite_keys.issubset(allowed_keys), f'Composite keys may only include {allowed_keys}'

    bigraph_spec = deep_merge(
        vivarium_composite.get('processes', {}),
        vivarium_composite.get('steps', {}))
    topology = vivarium_composite.get('topology', {})
    bigraph_spec = merge_wires(bigraph_spec, topology)
    bigraph_spec = deep_merge(
        bigraph_spec,
        vivarium_composite.get('state', {}))
    return bigraph_spec


def test_convert_flat():
    vivarium_composite = {
        'processes': {
            'ReactionBounds': '<dFBA_processes.ReactionBounds object at 0x166b91700>',
            'DynamicFBA': '<dFBA_processes.DynamicFBA object at 0x166b7acd0>',
            'BiomassCalculator': '<dFBA_processes.BiomassCalculator object at 0x169021b20>',
            'EnvCalculator': '<dFBA_processes.EnvCalculator object at 0x169021dc0>',
            'RegulatoryProtein': '<dFBA_processes.RegulatoryProtein object at 0x169021f70>',
            'GeneExpression': '<dFBA_processes.GeneExpression object at 0x169021a00>',
            'ProteinExpression': '<dFBA_processes.ProteinExpression object at 0x16842d820>'
        },
        'topology': {
            'ReactionBounds': {
                'reaction_bounds': ('reaction_bounds',),
                'current_v0': ('current_v0',),
                'concentration': ('concentration',),
                'enz_concentration': ('enz_concentration',)
            },
            'DynamicFBA': {
                'fluxes': ('fluxes_values',),
                'reactions': ('reactions_list',),
                'objective_flux': ('objective_flux_value',),
                'reaction_bounds': ('reaction_bounds',)},
            'BiomassCalculator': {
                'objective_flux': ('objective_flux_value',),
                'current_biomass': ('current_biomass_value',)},
            'EnvCalculator': {
                'current_biomass': ('current_biomass_value',),
                'fluxes_values': ('fluxes_values',),
                'current_env_consumption': ('current_env_consumption_value',),
                'concentration': ('concentration',)},
            'RegulatoryProtein': {
                'regulation_probability': ('regulation_probability',)},
            'GeneExpression': {
                'regulation_probability': ('regulation_probability',),
                'gene_expression': ('gene_expression',)},
            'ProteinExpression': {
                'gene_expression': ('gene_expression',),
                'enz_concentration': ('enz_concentration',)
            }
        },
        'steps': {},
        'flow': {},
        'state': {}
    }
    bigraph_spec = convert_vivarium_composite(vivarium_composite)
    plot_settings_test['rankdir'] = 'RL'
    plot_bigraph(bigraph_spec, **plot_settings_test, filename='vivarium_convert_flat')


def test_convert_nested():
    vivarium_composite = {
        'processes': {
            'microbiome': {
                'environment': {
                    'EnvCalculator': '<dFBA_processes.EnvCalculator object at 0x169021dc0>',
                },
                'bacteria': {
                    'ReactionBounds': '<dFBA_processes.ReactionBounds object at 0x166b91700>',
                    'DynamicFBA': '<dFBA_processes.DynamicFBA object at 0x166b7acd0>',
                    'BiomassCalculator': '<dFBA_processes.BiomassCalculator object at 0x169021b20>',
                    'RegulatoryProtein': '<dFBA_processes.RegulatoryProtein object at 0x169021f70>',
                    'GeneExpression': '<dFBA_processes.GeneExpression object at 0x169021a00>',
                    'ProteinExpression': '<dFBA_processes.ProteinExpression object at 0x16842d820>'
                }
            }
        },
        'topology': {
            'microbiome': {
                'environment': {
                    'EnvCalculator': {
                        'current_biomass': ('..', 'bacteria', 'current_biomass_value',),
                        'fluxes_values': ('..', 'bacteria', 'fluxes_values',),
                        'current_env_consumption': ('current_env_consumption_value',),
                        'concentration': ('concentration',)},
                },
                'bacteria': {
                    'ReactionBounds': {
                        'reaction_bounds': ('reaction_bounds',),
                        'current_v0': ('current_v0',),
                        'concentration': ('concentration',),
                        'enz_concentration': ('enz_concentration',)
                    },
                    'DynamicFBA': {
                        'fluxes': ('fluxes_values',),
                        'reactions': ('reactions_list',),
                        'objective_flux': ('objective_flux_value',),
                        'reaction_bounds': ('reaction_bounds',)},
                    'BiomassCalculator': {
                        'objective_flux': ('objective_flux_value',),
                        'current_biomass': ('current_biomass_value',)},
                    'RegulatoryProtein': {
                        'regulation_probability': ('regulation_probability',)},
                    'GeneExpression': {
                        'regulation_probability': ('regulation_probability',),
                        'gene_expression': ('gene_expression',)},
                    'ProteinExpression': {
                        'gene_expression': ('gene_expression',),
                        'enz_concentration': ('enz_concentration',)
                    }
                }
            }
        },
    }
    bigraph_spec = convert_vivarium_composite(vivarium_composite)
    plot_bigraph(bigraph_spec, **plot_settings_test, filename='vivarium_convert_nested')


def test_composite_process():
    vivarium_composite = {
        'processes': {
            'microbiome': {
                'environment': {
                    'EnvCalculator': '<dFBA_processes.EnvCalculator object at 0x169021dc0>',
                },
                'bacteria': {
                    'ReactionBounds': '<dFBA_processes.ReactionBounds object at 0x166b91700>',
                    'DynamicFBA': '<dFBA_processes.DynamicFBA object at 0x166b7acd0>',
                    'BiomassCalculator': '<dFBA_processes.BiomassCalculator object at 0x169021b20>',
                    'RegulatoryProtein': '<dFBA_processes.RegulatoryProtein object at 0x169021f70>',
                    'GeneExpression': '<dFBA_processes.GeneExpression object at 0x169021a00>',
                    'ProteinExpression': '<dFBA_processes.ProteinExpression object at 0x16842d820>'
                }
            }
        },
        'topology': {
            'microbiome': {
                'environment': {
                    'EnvCalculator': {
                        'current_biomass': ('..', 'bacteria', 'current_biomass_value',),
                        'fluxes_values': ('..', 'bacteria', 'fluxes_values',),
                        'current_env_consumption': ('current_env_consumption_value',),
                        'concentration': ('concentration',)},
                },
                'bacteria': {
                    'ReactionBounds': {
                        'reaction_bounds': ('reaction_bounds',),
                        'current_v0': ('current_v0',),
                        'concentration': ('concentration',),
                        'enz_concentration': ('enz_concentration',)
                    },
                    'DynamicFBA': {
                        'fluxes': ('fluxes_values',),
                        'reactions': ('reactions_list',),
                        'objective_flux': ('objective_flux_value',),
                        'reaction_bounds': ('reaction_bounds',)},
                    'BiomassCalculator': {
                        'objective_flux': ('objective_flux_value',),
                        'current_biomass': ('current_biomass_value',)},
                    'RegulatoryProtein': {
                        'regulation_probability': ('regulation_probability',)},
                    'GeneExpression': {
                        'regulation_probability': ('regulation_probability',),
                        'gene_expression': ('gene_expression',)},
                    'ProteinExpression': {
                        'gene_expression': ('gene_expression',),
                        'enz_concentration': ('enz_concentration',)
                    }
                }
            }
        },
    }
    bigraph_spec = convert_vivarium_composite(vivarium_composite)
    plot_bigraph(bigraph_spec, **plot_settings_test, filename='vivarium_composite_process')



if __name__ == '__main__':
    test_convert_flat()
    test_convert_nested()
    test_composite_process()
