"""
==========
Dict Utils
==========
"""

import collections.abc
import pprint
from typing import Any
import copy

from bigraph_schema.registry import type_schema_keys

pretty = pprint.PrettyPrinter(indent=2)

# TODO -- sync this up with bigraph-schema
schema_keys = [
    '_super',
    '_value',
    'wires',
    '_type',
    '_ports',
    '_tunnels',
    '_depends_on',
    '_sync_step',
]
schema_keys.extend(type_schema_keys)

# TODO -- this should come from process-bigraph
process_schema_keys = ['address', 'config', 'inputs', 'outputs', 'instance', 'interval']


def pp(x: Any) -> None:
    """Print ``x`` in a pretty format."""
    pretty.pprint(x)


def pf(x: Any) -> str:
    """Format ``x`` for display."""
    return pretty.pformat(x)


def deep_merge(dct, merge_dct):
    """ Recursive dict merge

    This mutates dct - the contents of merge_dct are added to dct (which is also returned).
    If you want to keep dct you could call it like deep_merge(copy.deepcopy(dct), merge_dct)
    """
    if dct is None:
        dct = {}
    if merge_dct is None:
        merge_dct = {}
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.abc.Mapping)):
            deep_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]
    return dct


def absolute_path(path, relative):
    progress = list(path)
    for step in relative:
        if step == '..' and len(progress) > 0:
            progress = progress[:-1]
        else:
            progress.append(step)
    return tuple(progress)


def nest_path(d, path):
    new_dict = {}
    if len(path) > 0:
        next_node = path[0]
        if len(path) > 1:
            remaining = path[1:]
            new_dict[next_node] = nest_path(d, remaining)
        else:
            new_dict[next_node] = d
    return new_dict


def compose(bigraph, node, path=None):
    path = path or ()
    new_bigraph = copy.deepcopy(bigraph)
    nested_node = nest_path(node, path)
    new_bigraph = deep_merge(new_bigraph, nested_node)
    return new_bigraph


def replace_regex_recursive(input_dict, match=' ', replacement='<br/>'):
    """
    Replace one string with another in keys and values of a nested dictionary. By default, this replaces whitespaces
    ' ' with newlines '<br/>'

    This function takes a nested dictionary as input and updates all keys and values
    that are strings by replacing whitespaces ' ' with '<br/>'. It uses a recursive
    function to process dictionaries within dictionaries.

    Args:
        input_dict (dict): The nested dictionary to be updated.
        match (str): The string with is replaced, whitespaces ' ' by default.
        replacement (str): The string that is inserted, newline '<br/>' by default.

    Returns:
        dict: The updated nested dictionary with match (whitespaces) replaced by replacement ('<br/>').
    """
    def replace_string(item):
        if isinstance(item, str):
            return item.replace(match, replacement)
        return item

    def recursive_replace(dictionary):
        updated_dict = {}
        for key, value in dictionary.items():
            new_key = replace_string(key)
            if isinstance(value, dict):
                new_value = recursive_replace(value)
            elif isinstance(value, list):
                new_value = []
                for v in value:
                    new_value.append(replace_string(v))
            elif isinstance(value, tuple):
                new_value = []
                for v in value:
                    new_value.append(replace_string(v))
                new_value = tuple(new_value)
            else:
                new_value = replace_string(value)
            updated_dict[new_key] = new_value
        return updated_dict

    return recursive_replace(input_dict)


def schema_state_to_dict(schema, state):
    schema_value_dict = {}
    for key, schema_value in schema.items():
        if key in schema_keys:
            # these are schema keys, just keep them as-is
            schema_value_dict[key] = schema_value
        else:
            state_value = state[key]
            if isinstance(schema_value, dict):
                schema_value_dict[key] = schema_state_to_dict(schema_value, state_value)
            else:
                assert isinstance(schema_value, str)
                if key not in schema_value_dict:
                    schema_value_dict[key] = {}
                schema_value_dict[key]['_type'] = schema_value

    for key, state_value in state.items():
        if key == 'wires':
            schema_value_dict[key] = state_value
        else:
            schema_value = schema.get(key, {})
            if isinstance(state_value, dict):
                schema_value_dict[key] = schema_state_to_dict(schema_value, state_value)
            else:
                if key not in schema_value_dict:
                    schema_value_dict[key] = {}
                schema_value_dict[key]['_value'] = state_value

    return schema_value_dict
