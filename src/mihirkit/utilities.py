"""
Utilities module providing essential helper functions for data manipulation and sorting.

This module includes utilities for flattening nested data structures and performing
natural (human-like) sorting on various data types including pandas DataFrames, Series,
and standard Python collections.

Key Features:
    * Universal Flattening: Recursively flatten nested lists, tuples, and NumPy arrays
    * Natural Sorting: Sort strings containing numbers in human-readable order
    * Multi-Type Support: Works with pandas DataFrames, Series, lists, tuples, sets, and NumPy arrays
    * Type Safety: Comprehensive type hints and error handling for robust code
    * Performance Optimized: Efficient algorithms for large datasets

Functions:
    flatten: Recursively flatten nested iterables into a single flat list
    natural_sort_key: Generate sorting key for natural (human-readable) sorting
    natural_sort: Perform natural sorting on various data structures

Example:
    ```python
    # Flatten nested structures
    nested_list = [1, [2, 3], [4, [5, 6]]]
    flat_list = flatten(nested_list)  # [1, 2, 3, 4, 5, 6]

    # Natural sorting
    items = ["item1", "item10", "item2", "item20"]
    sorted_items = natural_sort(items)  # ["item1", "item2", "item10", "item20"]
    ```
"""

import re
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd


def flatten(iterable: list | tuple | np.ndarray) -> list:
    """
    Recursively flatten nested iterables into a single flat list.

    Takes nested structures (lists, tuples, NumPy arrays) and returns a flat list
    containing all elements. Handles arbitrary nesting depth and mixed data types.
    Treats strings and bytes as atomic units to prevent character-level flattening.

    Args:
        iterable: The nested structure to flatten (list, tuple, or NumPy array)

    Returns:
        list: A flat list containing all elements from the nested structure

    Example:
        ```python
        # Simple nested list
        nested = [1, [2, 3], 4]
        result = flatten(nested)  # [1, 2, 3, 4]

        # Deeply nested structure
        deep_nested = [1, [2, [3, [4, 5]], 6], 7]
        result = flatten(deep_nested)  # [1, 2, 3, 4, 5, 6, 7]

        # NumPy array
        arr = np.array([[1, 2], [3, 4]])
        result = flatten(arr)  # [1, 2, 3, 4]

        # Mixed types
        mixed = [1, [2.5, "hello"], [True, [None]]]
        result = flatten(mixed)  # [1, 2.5, 'hello', True, None]
        ```
    """
    if isinstance(iterable, np.ndarray):
        return iterable.flatten().tolist()

    if getattr(iterable, "__iter__", False) and not isinstance(iterable, (str, bytes)):
        # One-liner is equivalent to
        # flattened = []
        # for sub_iterable in iterable:
        #     for elem in flatten(sub_iterable): # this makes it recursive
        #         flattened.append(elem)
        # return flattened
        return [elem for sub_iterable in iterable for elem in flatten(sub_iterable)]

    # If the iterable is a string or a non-iterable object (e.g. an int / float / bool / etc.)
    #   return the same item as a list, since the return value will go to another instance of
    #   flatten() and will be passed to the `for elem in flatten(sub_iterable)` code above
    # Therefore, `for elem in flatten(sub_iterable)` will evaluate to `elem = iterable` which
    # in turn will return `[elem] --> [iterable]` to the previous level (and so on . . .)
    #   is an int / float / bool / etc.
    return [iterable]


def natural_sort_key(item: str) -> Tuple[float | str | Any, ...]:
    """
    Generate a sorting key for natural (human-readable) sorting of strings containing numbers.

    Splits strings into alternating text and numeric segments, converting numeric segments
    to floats for proper numerical comparison. This ensures that "item2" comes before
    "item10" instead of the lexicographic ordering where "item10" would come first.

    Args:
        item: The string to generate a natural sorting key for

    Returns:
        Tuple: A tuple where numeric parts are converted to floats and text parts remain as strings

    Example:
        ```python
        # Understanding the sorting key
        key1 = natural_sort_key("item2")    # ('item', 2.0, '')
        key2 = natural_sort_key("item10")   # ('item', 10.0, '')
        key3 = natural_sort_key("item2a")   # ('item', 2.0, 'a')

        # This ensures "item2" < "item10" (2.0 < 10.0)
        # Rather than "item10" < "item2" (lexicographic string comparison)
        ```
    """
    # regex to split the string into numeric and non-numeric parts
    split_parts = re.split(r"(\d+)", str(item))

    # convert numeric parts to floats and leave non-numeric parts unchanged
    return tuple(float(part) if part.isdigit() else part for part in split_parts)


def natural_sort(
    data, by: Optional[Iterable] = None
) -> pd.DataFrame | pd.Series | list | tuple | set:
    """
    Perform natural sorting on various data structures treating numbers within strings numerically.

    Sorts data in human-readable order where numeric parts within strings are compared
    numerically rather than lexicographically. Supports pandas DataFrames, Series,
    and standard Python collections.

    Args:
        data: The data structure to sort (DataFrame, Series, list, tuple, or set)
        by: For DataFrames only - column name(s) to sort by (required for DataFrames)

    Returns:
        Same type as input (except tuple/set return list), sorted in natural order

    Raises:
        ValueError: If DataFrame is provided without 'by' parameter
        TypeError: If data type is not supported

    Example:
        ```python
        # List sorting
        items = ["file1.txt", "file10.txt", "file2.txt", "file20.txt"]
        sorted_items = natural_sort(items)
        # Result: ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]

        # DataFrame sorting
        df = pd.DataFrame({
            'filename': ['file10.txt', 'file2.txt', 'file1.txt'],
            'size': [1024, 2048, 512]
        })
        sorted_df = natural_sort(df, by=['filename'])

        # Series sorting
        series = pd.Series(["item10", "item2", "item1"])
        sorted_series = natural_sort(series)

        # Version sorting
        versions = ["v1.2", "v1.10", "v1.2.1", "v1.3", "v2.1"]
        sorted_versions = natural_sort(versions)
        # Result: ["v1.2", "v1.2.1", "v1.3", "v1.10", "v2.1"]
        ```
    """
    match data:
        case pd.DataFrame():
            if by is None:
                raise ValueError(
                    "When sorting a pandas DataFrame, the `by` argument must be provided."
                )

            # Create temporary _sorted columns by applying the natural_sort_key to each column in `by`
            for col in by:
                data[f"{col}_sorted"] = data[col].map(natural_sort_key)
            temporary_column_names = [f"{col}_sorted" for col in by]

            # Sort the DataFrame by the temporary `_sorted` columns
            sorted_data = data.sort_values(by=temporary_column_names)

            # Drop the temporary columns
            sorted_data = sorted_data.drop(columns=temporary_column_names)

            return sorted_data

        case pd.Series():
            return data.iloc[data.map(natural_sort_key).argsort().to_numpy()]
        case list() | tuple() | set():
            return sorted(data, key=natural_sort_key)
        case _:
            raise TypeError(
                f"The input must be one of type [pandas.DataFrame, pandas.Series, list, tuple, set]. Found {type(data)}"
            )
