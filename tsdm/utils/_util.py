r"""Utility functions.

TODO:  Module description
"""

# from __future__ import annotations


__all__ = [
    # Constants
    # Classes
    "Split",
    # Functions
    "deep_dict_update",
    "deep_kval_update",
    "flatten_dict",
    "flatten_nested",
    "initialize_from",
    "initialize_from_config",
    "is_partition",
    "now",
    "paths_exists",
    "prepend_path",
    "round_relative",
    "pairwise_disjoint",
    "pairwise_disjoint_masks",
    "interpolate_missing_limited",
]

import json
import os
import random
from collections.abc import Callable, Collection, Hashable, Iterable, Mapping, Sequence
from datetime import datetime
from functools import partial
from importlib import import_module
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional, Union, overload

import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from torch import nn

from tsdm.utils.types import AnyTypeVar, Nested, ObjectVar, PathType, ReturnVar
from tsdm.utils.types.abc import HashableType

__logger__ = getLogger(__name__)

EMPTY_PATH: Path = Path()
r"""Constant: Blank path."""


def pairwise_disjoint(sets: Iterable[set]) -> bool:
    r"""Check if sets are pairwise disjoint."""
    union = set().union(*sets)
    return len(union) == sum(len(s) for s in sets)


def pairwise_disjoint_masks(masks: Iterable[NDArray[np.bool_]]) -> bool:
    r"""Check if masks are pairwise disjoint."""
    return all(sum(masks) == 1)  # type: ignore[arg-type]


def flatten_dict(
        d: dict[Any, Any],
        /,
        *,
        join_string: Optional[str] = None,
        key_func: Optional[Callable[[Hashable, Hashable], Hashable]] = None,
        recursive: bool | int = True,
) -> dict[Any, Any]:
    r"""Flatten a dictionary containing iterables to a list of tuples.

    Parameters
    ----------
    d: Input Dictionary
    join_string:
        use this option in case of nested dicts with string keys,
    key_func:
        function that creates new key from old key and subkey
    recursive:
        If true applies flattening strategy recursively on nested dicts.
        If int, specifies the maximum depth of recursion.
    """
    if join_string is not None and key_func is not None:
        raise ValueError("Only one of join_string or key_func can be specified.")

    if join_string is not None:
        key_func = lambda k, sk: f"{k}{join_string}{sk}"  # noqa: E731
    elif key_func is None:
        key_func = lambda k, sk: (k, sk)  # noqa: E731

    result = {}
    for key, item in d.items():
        if isinstance(key, tuple):
            raise ValueError("Keys are not allowed to be tuples!")
        if isinstance(item, dict) and recursive:
            subdict = flatten_dict(
                item, recursive=True, key_func=key_func, join_string=join_string
            )
            for subkey, subitem in subdict.items():
                result[key_func(key, subkey)] = subitem
        else:
            result[key] = item
    return result


class Split(NamedTuple):
    r"""Holds indices for train/valid/test set."""

    train: int
    """The training set."""
    valid: Any
    """The validation set."""
    test: Any
    """The testing set."""


def round_relative(x: np.ndarray, decimals: int = 2) -> np.ndarray:
    r"""Round to relative precision."""
    order = np.where(x == 0, 0, np.floor(np.log10(x)))
    digits = decimals - order
    rounded = np.rint(x * 10 ** digits)
    return np.true_divide(rounded, 10 ** digits)


def now():
    r"""Return current time in iso format."""
    return datetime.now().isoformat(timespec="seconds")


def deep_dict_update(d: dict, new_kvals: Mapping) -> dict:
    r"""Update nested dictionary recursively in-place with new dictionary.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: Mapping
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in new_kvals.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_dict_update(d.get(key, {}), value)
        else:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def deep_kval_update(d: dict, **new_kvals: dict) -> dict:
    r"""Update nested dictionary recursively in-place with key-value pairs.

    Reference: https://stackoverflow.com/a/30655448/9318372

    Parameters
    ----------
    d: dict
    new_kvals: dict
    """
    # if not inplace:
    #     return deep_dict_update(deepcopy(d), new_kvals, inplace=False)

    for key, value in d.items():
        if isinstance(value, Mapping) and value:
            d[key] = deep_kval_update(d.get(key, {}), **new_kvals)
        elif key in new_kvals:
            # if value is not None or not safe:
            d[key] = new_kvals[key]
    return d


def apply_nested(
        nested: Nested[AnyTypeVar | None],
        kind: type[AnyTypeVar],
        func: Callable[[AnyTypeVar], ReturnVar],
) -> Nested[ReturnVar | None]:
    r"""Apply function to nested iterables of a given kind.

    Parameters
    ----------
    nested: Nested Data-Structure (Iterable, Mapping, ...)
    kind: The type of the leave nodes
    func: A function to apply to all leave Nodes
    """
    if nested is None:
        return None
    if isinstance(nested, kind):
        return func(nested)
    if isinstance(nested, Mapping):
        return {k: apply_nested(v, kind, func) for k, v in nested.items()}
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(nested, Collection):
        return [apply_nested(obj, kind, func) for obj in nested]  # type: ignore[misc]
    raise TypeError(f"Unsupported type: {type(nested)}")


@overload
def prepend_path(
        files: Nested[PathType],
        parent: Path,
        *,
        keep_none: bool = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
        files: Nested[Optional[PathType]],
        parent: Path,
        *,
        keep_none: Literal[False] = False,
) -> Nested[Path]:
    ...


@overload
def prepend_path(
        files: Nested[Optional[PathType]],
        parent: Path,
        *,
        keep_none: Literal[True] = True,
) -> Nested[Optional[Path]]:
    ...


def prepend_path(
        files: Nested[Optional[PathType]],
        parent: Path,
        *,
        keep_none: bool = False,
) -> Nested[Optional[Path]]:
    r"""Prepends path to all files in nested iterable.

    Parameters
    ----------
    files
        Nested datastructures with Path-objects at leave nodes.
    parent:
    keep_none:
        If True, None-values are kept.

    Returns
    -------
    Nested data-structure with prepended path.
    """
    # TODO: change it to apply_nested in python 3.10

    if files is None:
        return None if keep_none else parent
    if isinstance(files, (str, Path, os.PathLike)):
        return parent / Path(files)
    if isinstance(files, Mapping):
        return {
            k: prepend_path(v, parent, keep_none=keep_none) for k, v in files.items()  # type: ignore[arg-type]
        }
    # TODO https://github.com/python/mypy/issues/11615
    if isinstance(files, Collection):
        return [prepend_path(f, parent, keep_none=keep_none) for f in files]  # type: ignore[misc,arg-type]
    raise TypeError(f"Unsupported type: {type(files)}")


def flatten_nested(nested: Any, kind: type[HashableType]) -> set[HashableType]:
    r"""Flatten nested iterables of a given kind.

    Parameters
    ----------
    nested: Any
    kind: hashable

    Returns
    -------
    set[hashable]
    """
    if nested is None:
        return set()
    if isinstance(nested, kind):
        return {nested}
    if isinstance(nested, Mapping):
        return set.union(*(flatten_nested(v, kind) for v in nested.values()))
    if isinstance(nested, Iterable):
        return set.union(*(flatten_nested(v, kind) for v in nested))
    raise ValueError(f"{type(nested)} is not understood")


def initialize_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize `nn.Module` from config object."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


# partial from type
@overload
def initialize_from(  # type: ignore[misc]
        lookup_table: dict[str, type[ObjectVar]],
        /,
        __name__: str,
        **kwargs: Any,
) -> ObjectVar:
    ...


# partial from func
@overload
def initialize_from(
        lookup_table: dict[str, Callable[..., ReturnVar]],
        /,
        __name__: str,
        **kwargs: Any,
) -> Callable[..., ReturnVar]:
    ...


# partial from type
# @overload
# def initialize_from(
#     lookup_table: dict[str, Union[type[ObjectType], Callable[..., ReturnType]]],
#     /,
#     __name__: str,
#     **kwargs: Any,
# ) -> Union[ObjectType, Callable[..., ReturnType]]:
#     ...


def initialize_from(  # type: ignore[misc]
        lookup_table: Union[
            dict[str, type[ObjectVar]],
            dict[str, Callable[..., ReturnVar]],
            dict[str, type[ObjectVar] | Callable[..., ReturnVar]],
        ],
        /,
        __name__: str,
        **kwargs: Any,
) -> ObjectVar | Callable[..., ReturnVar]:
    r"""Lookup class/function from dictionary and initialize it.

    Roughly equivalent to:

    .. code-block:: python

        obj = lookup_table[__name__]
        if isclass(obj):
            return obj(**kwargs)
        return partial(obj, **kwargs)

    Parameters
    ----------
    lookup_table: dict[str, Callable]
    __name__: str
        The name of the class/function
    kwargs: Any
        Optional arguments to initialize class/function

    Returns
    -------
    object
        The initialized class/function
    """
    obj = lookup_table[__name__]
    assert callable(obj), f"Looked up object {obj} not callable class/function."

    # check that obj is a class, but not metaclass or instance.
    if isinstance(obj, type) and not issubclass(obj, type):
        initialized_object: ObjectVar = obj(**kwargs)
        return initialized_object
    # if it is function, fix kwargs
    initialized_callable: Callable[..., ReturnVar] = partial(obj, **kwargs)  # type: ignore[assignment]
    return initialized_callable


def is_dunder(name: str) -> bool:
    r"""Check if name is a dunder method.

    Parameters
    ----------
    name: str

    Returns
    -------
    bool
    """
    return name.isidentifier() and name.startswith("__") and name.endswith("__")


def is_partition(*partition: Collection, union: Optional[Sequence] = None) -> bool:
    r"""Check if partition is a valid partition of union."""
    if len(partition) == 1:
        return is_partition(*next(iter(partition)), union=union)

    sets = (set(p) for p in partition)
    part_union = set().union(*sets)

    if union is not None and part_union != set(union):
        return False
    return len(part_union) == sum(len(p) for p in partition)


def initialize_module_from_config(config: dict[str, Any]) -> nn.Module:
    r"""Initialize a class from a dictionary."""
    assert "__name__" in config, "__name__ not found in dict"
    assert "__module__" in config, "__module__ not found in dict"
    __logger__.debug("Initializing %s", config)
    config = config.copy()
    module = import_module(config.pop("__module__"))
    cls = getattr(module, config.pop("__name__"))
    opts = {key: val for key, val in config.items() if not is_dunder("key")}
    return cls(**opts)


def paths_exists(
        paths: Nested[Optional[PathType]],
        *,
        parent: Path = EMPTY_PATH,
) -> bool:
    r"""Check whether the files exist.

    The input can be arbitrarily nested data-structure with `Path` in leaves.

    Parameters
    ----------
    paths: None | Path | Collection[Path] | Mapping[Any, None | Path | Collection[Path]]
    parent: Path = Path(),

    Returns
    -------
    bool
    """
    if isinstance(paths, str):
        return Path(paths).exists()
    if paths is None:
        return True
    if isinstance(paths, Mapping):
        return all(paths_exists(f, parent=parent) for f in paths.values())
    if isinstance(paths, Collection):
        return all(paths_exists(f, parent=parent) for f in paths)
    if isinstance(paths, Path):
        return (parent / paths).exists()

    raise ValueError(f"Unknown type for rawdata_file: {type(paths)}")


import json
import os


def save_metadata(metadata: dict, filename: str = "interpolation_metadata.json"):
    filepath = os.path.join("./pic/interpolation_plots", filename)

    # Load existing metadata if file exists
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            try:
                existing = json.load(f)
            except json.JSONDecodeError:
                existing = []
    else:
        existing = []

    # Append new metadatac
    existing.append(metadata)

    # Save updated metadata
    with open(filepath, "w") as f:
        json.dump(existing, f, indent=2)

def interpolate_missing_limited(
    x_time: torch.Tensor,
    x_vals: torch.Tensor,
    x_mask: torch.Tensor,
    max_interp_points: int,
    plot: bool = False,
    plot_dir: str = "./pic/interpolation_plots",
    fallback_value: float = 0.0
) -> torch.Tensor:
    """
    Interpolate missing values at evenly spaced time indices. Keep only those values.
    Use original values if available, otherwise interpolate. Fall back to `fallback_value` if needed.

    Args:
        x_time (torch.Tensor): [time] Time indices
        x_vals (torch.Tensor): [batch, time, features]
        x_mask (torch.Tensor): [batch, time, features] boolean mask (True=observed)
        max_interp_points (int): Number of evenly spaced indices to keep per sample-feature
        plot (bool): Whether to generate plots
        plot_dir (str): Directory to save plots
        fallback_value (float): Value to use if interpolation is not possible

    Returns:
        torch.Tensor: [batch, max_interp_points, features]
    """
    if plot:
        os.makedirs(plot_dir, exist_ok=True)

    x_vals_np = x_vals.cpu().numpy()
    x_mask_np = x_mask.cpu().numpy()
    x_time_np = x_time.cpu().numpy()
    batch_size, time_steps, features = x_vals_np.shape

    # Evenly spaced time indices to retain (0-based)
    split_indices = np.floor(np.linspace(0, time_steps - 1, num=max_interp_points)).astype(int).tolist()

    # Output tensor initialized with fallback
    output_np = np.full((batch_size, max_interp_points, features), fallback_value, dtype=x_vals_np.dtype)

    for b in range(batch_size):
        for f in range(features):
            series = x_vals_np[b, :, f]
            mask = x_mask_np[b, :, f]
            valid_idx = np.where(mask & np.isfinite(series))[0]  # exclude NaNs/infs

            if len(valid_idx) < 1:
                continue  # No data to interpolate or fill

            values = []
            interp_flags = []

            for idx in split_indices:
                if mask[idx] and np.isfinite(series[idx]):
                    # Use original valid value
                    values.append(np.array(series[idx], dtype=series.dtype))
                    interp_flags.append(False)
                else:
                    # Find nearest valid neighbors
                    left_candidates = valid_idx[valid_idx < idx]
                    right_candidates = valid_idx[valid_idx > idx]

                    if len(left_candidates) > 0 and len(right_candidates) > 0:
                        x0 = left_candidates[-1]
                        x1 = right_candidates[0]
                        y0 = series[x0]
                        y1 = series[x1]
                        if np.isfinite(y0) and np.isfinite(y1) and x1 != x0:
                            interpolated = y0 + (idx - x0) / (x1 - x0) * (y1 - y0)
                            values.append(np.array(interpolated, dtype=series.dtype))
                        else:
                            values.append(np.array(fallback_value, dtype=series.dtype))
                        interp_flags.append(True)
                    elif len(left_candidates) > 0:
                        # Forward-fill using previous value
                        y0 = series[left_candidates[-1]]
                        values.append(np.array(y0, dtype=series.dtype))
                        interp_flags.append(True)
                    elif len(right_candidates) > 0:
                        # Backward-fill using next value
                        y1 = series[right_candidates[0]]
                        values.append(np.array(y1, dtype=series.dtype))
                        interp_flags.append(True)
                    else:
                        # No valid values found
                        values.append(np.array(fallback_value, dtype=series.dtype))
                        interp_flags.append(True)

            output_np[b, :, f] = np.array(values, dtype=series.dtype)

            # Plotting
            if plot:
                plt.figure(figsize=(10, 4))
                time = np.arange(time_steps)

                # Plot known values in blue
                plt.scatter(valid_idx, series[valid_idx], color="blue", label="Original Known", s=30)

                final_values = output_np[b, :, f]
                interp_idx = [i for i, flag in enumerate(interp_flags) if flag]
                orig_idx = [i for i, flag in enumerate(interp_flags) if not flag]

                interp_points = [split_indices[i] for i in interp_idx]
                orig_points = [split_indices[i] for i in orig_idx]

                combined_idx = sorted(interp_idx + orig_idx)
                combined_points = [split_indices[i] for i in combined_idx]
                combined_values = [final_values[i] for i in combined_idx]

                # Line through all final points
                plt.plot(combined_points,
                         combined_values,
                         color="red", linestyle='-', linewidth=2, alpha=0.7,
                         label="Interpolated & Original Line")

                plt.scatter(interp_points,
                            [final_values[i] for i in interp_idx],
                            color="red", marker='x', s=60, label="Interpolated")

                plt.scatter(orig_points,
                            [final_values[i] for i in orig_idx],
                            color="orange", marker='o', s=50, label="Selected Original")

                plt.xticks(split_indices)
                plt.xlabel("Time Step")
                plt.ylabel("Value")
                plt.grid(True, axis='y')
                plt.legend()
                plt.title(f"Sample {b}, Feature {f}")
                plt.tight_layout()

                filename = f"sample{b}_feature{f}.png"
                plt.savefig(os.path.join(plot_dir, filename))
                plt.close()

    return torch.tensor(output_np, dtype=x_vals.dtype, device=x_vals.device)
