import collections.abc
import pprint
from typing import Any


pretty = pprint.PrettyPrinter(indent=2)


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
    new_bigraph = bigraph.copy()
    nested_node = nest_path(node, path)
    new_bigraph = deep_merge(new_bigraph, nested_node)
    return new_bigraph