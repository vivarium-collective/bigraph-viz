from bigraph_viz import plot_bigraph


ecoli = {
    'chromosome structure': {
        '_type': 'process',
        'inputs': {
            'fragmentBases': ['bulk molecules'],
            'molecules': ['bulk molecules'],
            'active tfs': ['bulk molecules'],
            'subunits': ['bulk molecules'],
            'amino acids': ['bulk molecules'],
            'active replisomes': ['unique molecules', 'active replisome'],
            'oriCs': ['unique molecules', 'oriC'],
            'chromosome domains': ['unique molecules', 'chromosome domain'],
            'active RNAPs': ['unique molecules', 'active RNAP'],
            'RNAs': ['unique molecules', 'RNA'],
            'active ribosome': ['unique molecules', 'active ribosome'],
            'full chromosomes': ['unique molecules', 'full chromosome'],
            'promoters': ['unique molecules', 'promoter'],
            'DnaA boxes': ['unique molecules', 'DnaA box']
        }
    },
    'metabolism': {
        '_type': 'process',
        'inputs': {
            'metabolites': ['bulk molecules'],
            'catalysts': ['bulk molecules'],
            'kinetics enzymes': ['bulk molecules'],
            'kinetics substrates': ['bulk molecules'],
            'amino acids': ['bulk molecules'],
            'environment': ['environment'],
            'amino acids total': ['bulk molecules']
        }
    },
    'tf binding': {
        '_type': 'process',
        'inputs': {
            'promoters': ['unique molecules', 'promoter'],
            'active tfs': ['bulk molecules'],
            'active tfs total': ['bulk molecules'],
            'inactive tfs total': ['bulk molecules'],
        }
    },
    'transcript initiation': {
        '_type': 'process',
        'inputs': {
            'environment': ['environment'],
            'full chromosomes': ['unique molecules', 'full chromosome'],
            'RNAs': ['unique molecules', 'RNA'],
            'active RNAPs': ['unique molecules', 'active RNAP'],
            'promoters': ['unique molecules', 'promoter'],
            'molecules': ['bulk molecules'],
        }
    },
    'transcript elongation': {
        '_type': 'process',
        'inputs': {
            'environment': ['environment'],
            'RNAs': ['unique molecules', 'RNA'],
            'active RNAPs': ['unique molecules', 'active RNAP'],
            'molecules': ['bulk molecules'],
            'bulk RNAs': ['bulk molecules'],
            'ntps': ['bulk molecules'],
        }
    },
    'rna degradation': {
        '_type': 'process',
        'inputs': {
            'charged trna': ['bulk molecules'],
            'bulk RNAs': ['bulk molecules'],
            'nmps': ['bulk molecules'],
            'fragmentMetabolites': ['bulk molecules'],
            'fragmentBases': ['bulk molecules'],
            'endoRnases': ['bulk molecules'],
            'exoRnases': ['bulk molecules'],
            'subunits': ['bulk molecules'],
            'molecules': ['bulk molecules'],
            'RNAs': ['unique molecules', 'RNA'],
            'active ribosome': ['unique molecules', 'active ribosome'],
        }
    },
    'polypeptide initiation': {
        '_type': 'process',
        'inputs': {
            'environment': ['environment'],
            'active ribosome': ['unique molecules', 'active ribosome'],
            'RNA': ['unique molecules', 'RNA'],
            'subunits': ['bulk molecules']
        }
    },
    'polypeptide elongation': {
        '_type': 'process',
        'inputs': {
            'environment': ['environment'],
            'active ribosome': ['unique molecules', 'active ribosome'],
            'molecules': ['bulk molecules'],
            'monomers': ['bulk molecules'],
            'amino acids': ['bulk molecules'],
            'ppgpp reaction metabolites': ['bulk molecules'],
            'uncharged trna': ['bulk molecules'],
            'charged trna': ['bulk molecules'],
            'charging molecules': ['bulk molecules'],
            'synthetases': ['bulk molecules'],
            'subunits': ['bulk molecules'],
            'molecules total': ['bulk molecules'],
            'amino acids total': ['bulk molecules'],
            'charged trna total': ['bulk molecules'],
            'uncharged trna total': ['bulk molecules']
        }
    },
    'complexation': {
        '_type': 'process',
        'inputs': {
            'molecules': ['bulk molecules'],
        }
    },
    'two component system': {
        '_type': 'process',
        'inputs': {
            'molecules': ['bulk molecules']
        }
    },
    'equilibrium': {
        '_type': 'process',
        'inputs': {
            'molecules': ['bulk molecules']
        }
    },
    'protein degradation': {
        '_type': 'process',
        'inputs': {
            'metabolites': ['bulk molecules'],
            'proteins': ['bulk molecules']
        }
    },
    'chromosome replication': {
        '_type': 'process',
        'inputs': {
            'replisome trimers': ['bulk molecules'],
            'replisome monomers': ['bulk molecules'],
            'dntps': ['bulk molecules'],
            'ppi': ['bulk molecules'],
            'active replisomes': ['unique molecules', 'active replisome'],
            'oriCs': ['unique molecules', 'oriC'],
            'chromosome domains': ['unique molecules', 'chromosome domain'],
            'full chromosomes': ['unique molecules', 'full chromosome'],
            'environment': ['environment'],
        }
    },
    # 'plasmid replication': {
    #     '_type': 'process',
    #     'inputs': {
    #         'replisome trimers': ['bulk molecules'],
    #         'replisome monomers': ['bulk molecules'],
    #         'dntps': ['bulk molecules'],
    #         'plasmid active replisomes': ['unique molecules', 'plasmid active replisome'],
    #         'oriVs': ['unique molecules', 'oriV'],
    #         'plasmid domains': ['unique molecules', 'plasmid domain'],
    #         'full plasmids': ['unique molecules', 'full plasmid'],
    #         'environment': ['environment'],
    #     }
    # },
    'unique molecules': {
        'chromosome domain': {},
        'full chromosome': {},
        'oriC': {},
        'active replisome': {},
        'RNA': {},
        'active ribosome': {},
        'DnaA box': {},
        'promoter': {},
        'plasmid domain': {},
        'full plasmid': {},
        'oriV': {},
        'plasmid active replisome': {},
    },
    'bulk molecules': {},
    'environment': {},
}


def test_ecoli_bigraph():
    plot_bigraph(
        ecoli,
        remove_process_place_edges=True,
        rankdir='RL',
        out_dir='out',
        filename='ecoli_bigraph',
        node_label_size='18pt',
        process_label_size='18pt',
        port_label_size='14pt',
        label_margin='0.2',
        dpi='150',
    )


def test_ecoli_reduced():
    ecoli_reduced = ecoli.copy()
    # ecoli_reduced['transciption translation'] = ecoli_reduced['polypeptide elongation'].copy()
    # ecoli_reduced['degradation'] = ecoli_reduced['rna degradation'].copy().update(ecoli_reduced['protein degradation'].copy())
    # del ecoli_reduced['transcript initiation']
    # del ecoli_reduced['transcript elongation']
    # del ecoli_reduced['rna degradation']
    # del ecoli_reduced['polypeptide initiation']
    # del ecoli_reduced['polypeptide elongation']
    # del ecoli_reduced['protein degradation']
    # del ecoli_reduced['complexation']

    ecoli_reduced['replication initiation control'] = {
        '_type': 'process',
        'inputs': {
            # 'fragmentBases': ['bulk molecules'],
            'metabolites': ['bulk molecules'],
            # 'active tfs': ['bulk molecules'],
            # 'subunits': ['bulk molecules'],
            # 'amino acids': ['bulk molecules'],
            'active replisomes': ['unique molecules', 'active replisome'],
            'oriCs': ['unique molecules', 'oriC'],
            'chromosome domains': ['unique molecules', 'chromosome domain'],
            'active RNAPs': ['unique molecules', 'active RNAP'],
            # 'RNAs': ['unique molecules', 'RNA'],
            # 'active ribosome': ['unique molecules', 'active ribosome'],
            'full chromosomes': ['unique molecules', 'full chromosome'],
            'promoters': ['unique molecules', 'promoter'],
            'DnaA boxes': ['unique molecules', 'DnaA box']
        }
    }

    plot_bigraph(
        ecoli_reduced,
        node_border_colors={('replication initiation control',): 'red'},
        node_border_widths={('replication initiation control',): '8'},
        node_label_sizes={('replication initiation control',): '40pt'},
        remove_process_place_edges=True,
        rankdir='RL',
        node_label_size='32pt',
        process_label_size='32pt',
        port_label_size='22pt',
        port_labels=False,
        label_margin='0.1',
        dpi='200',
        out_dir='out',
        filename='ecoli_reduced',
        file_format='png',
    )

def test_ecoli_separate_processes():
    plot_bigraph(
        ecoli,
        separate_processes=True,
        port_labels=False,
        out_dir='out',
        filename='ecoli_separate_processes',
        node_label_size='18pt',
        process_label_size='18pt',
        label_margin='0.2',
        dpi='150',
    )


if __name__ == '__main__':
    test_ecoli_bigraph()
    test_ecoli_reduced()
    test_ecoli_separate_processes()