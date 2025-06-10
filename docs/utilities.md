# Utilities Module Documentation

## Overview

The `utilities` module provides essential helper functions for data manipulation and sorting operations. It includes utilities for flattening nested data structures and performing natural (human-like) sorting on various data types including pandas DataFrames, Series, and standard Python collections.

## Features

- **Universal Flattening**: Recursively flatten nested lists, tuples, and NumPy arrays into flat lists
- **Natural Sorting**: Sort strings containing numbers in human-readable order (e.g., "item2" before "item10")
- **Multi-Type Support**: Works with pandas DataFrames, Series, lists, tuples, sets, and NumPy arrays
- **Type Safety**: Comprehensive type hints and error handling for robust code
- **Performance Optimized**: Efficient algorithms for large datasets

## Installation

```bash
pip install pandas numpy
```

## Quick Start

```python
from utilities import flatten, natural_sort, natural_sort_key

# Flatten nested structures
nested_list = [1, [2, 3], [4, [5, 6]]]
flat_list = flatten(nested_list)  # [1, 2, 3, 4, 5, 6]

# Natural sorting
items = ["item1", "item10", "item2", "item20"]
sorted_items = natural_sort(items)  # ["item1", "item2", "item10", "item20"]
```

## Function Reference

### flatten()

Recursively flattens nested iterables into a single flat list.

```python
def flatten(iterable: list | tuple | np.ndarray) -> list
```

**Parameters:**

- `iterable`: The nested structure to flatten (list, tuple, or NumPy array)

**Returns:**

- `list`: A flat list containing all elements from the nested structure

**Supported Input Types:**

- Lists (including nested lists)
- Tuples (including nested tuples)
- NumPy arrays (any dimension)
- Mixed nested structures
- Non-iterable objects (returned as single-element lists)

#### Examples

##### Basic Flattening

```python
# Simple nested list
nested = [1, [2, 3], 4]
result = flatten(nested)
print(result)  # [1, 2, 3, 4]

# Deeply nested structure
deep_nested = [1, [2, [3, [4, 5]], 6], 7]
result = flatten(deep_nested)
print(result)  # [1, 2, 3, 4, 5, 6, 7]

# Mixed types
mixed = [1, [2.5, "hello"], [True, [None]]]
result = flatten(mixed)
print(result)  # [1, 2.5, 'hello', True, None]
```

##### NumPy Array Flattening

```python
import numpy as np

# 2D NumPy array
arr_2d = np.array([[1, 2], [3, 4]])
result = flatten(arr_2d)
print(result)  # [1, 2, 3, 4]

# 3D NumPy array
arr_3d = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
result = flatten(arr_3d)
print(result)  # [1, 2, 3, 4, 5, 6, 7, 8]

# Mixed with lists
mixed_array = [arr_2d, [9, 10]]
result = flatten(mixed_array)
print(result)  # [1, 2, 3, 4, 9, 10]
```

##### Tuple and Mixed Structure Flattening

```python
# Nested tuples
nested_tuple = (1, (2, 3), (4, (5, 6)))
result = flatten(nested_tuple)
print(result)  # [1, 2, 3, 4, 5, 6]

# Mixed lists and tuples
mixed_structure = [1, (2, [3, (4, 5)]), 6]
result = flatten(mixed_structure)
print(result)  # [1, 2, 3, 4, 5, 6]
```

##### Edge Cases

```python
# Single element (non-iterable)
single = 42
result = flatten(single)
print(result)  # [42]

# String (treated as non-iterable to avoid character splitting)
text = "hello"
result = flatten(text)
print(result)  # ["hello"]

# Empty nested structure
empty_nested = [[], [[], []], []]
result = flatten(empty_nested)
print(result)  # []
```

### natural_sort_key()

Generates a sorting key for natural (human-readable) sorting of strings containing numbers.

```python
def natural_sort_key(item: str) -> Tuple[float | str | Any, ...]
```

**Parameters:**

- `item`: The string to generate a natural sorting key for

**Returns:**

- `Tuple`: A tuple where numeric parts are converted to floats and text parts remain as strings

**How It Works:**
The function splits strings into alternating text and numeric segments, converting numeric segments to floats for proper numerical comparison.

#### Examples

```python
# Understanding the sorting key
key1 = natural_sort_key("item2")    # ('item', 2.0, '')
key2 = natural_sort_key("item10")   # ('item', 10.0, '')
key3 = natural_sort_key("item2a")   # ('item', 2.0, 'a')

# This ensures "item2" < "item10" (2.0 < 10.0)
# Rather than "item10" < "item2" (lexicographic string comparison)
```

### natural_sort()

Performs natural sorting on various data structures, treating numbers within strings numerically rather than lexicographically.

```python
def natural_sort(data, by: Optional[Iterable] = None) -> pd.DataFrame | pd.Series | list | tuple | set
```

**Parameters:**

- `data`: The data structure to sort
- `by`: For DataFrames only - column name(s) to sort by

**Returns:**

- Same type as input, sorted in natural order

**Supported Data Types:**

- `pandas.DataFrame`: Requires `by` parameter
- `pandas.Series`: Sorts by values
- `list`: Returns sorted list
- `tuple`: Returns sorted list (tuple → list conversion)
- `set`: Returns sorted list (set → list conversion)

#### Examples

##### List Sorting

```python
# Basic string list with numbers
items = ["file1.txt", "file10.txt", "file2.txt", "file20.txt"]
sorted_items = natural_sort(items)
print(sorted_items)  # ["file1.txt", "file2.txt", "file10.txt", "file20.txt"]

# Mixed alphanumeric
versions = ["v1.2", "v1.10", "v1.2.1", "v1.3", "v2.1"]
sorted_versions = natural_sort(versions)
print(sorted_versions)  # ["v1.2", "v1.2.1", "v1.3", "v1.10", "v2.1"]

# Complex naming patterns
files = ["chapter1", "chapter2", "chapter10", "appendixA", "appendixB"]
sorted_files = natural_sort(files)
print(sorted_files)  # ["appendixA", "appendixB", "chapter1", "chapter2", "chapter10"]
```

##### Pandas Series Sorting

```python
import pandas as pd

# Create a Series with mixed string-number data
series = pd.Series(["item10", "item2", "item1", "item20"])
sorted_series = natural_sort(series)
print(sorted_series)
# Output:
# 2    item1
# 1    item2
# 0    item10
# 3    item20

# Series with complex patterns
versions_series = pd.Series(["v2.1.0", "v1.10.5", "v1.2.3", "v1.3.0"])
sorted_versions = natural_sort(versions_series)
print(sorted_versions)
# Proper version sorting: v1.2.3, v1.3.0, v1.10.5, v2.1.0
```

##### Pandas DataFrame Sorting

```python
import pandas as pd

# Create sample DataFrame
df = pd.DataFrame({
    'filename': ['file10.txt', 'file2.txt', 'file1.txt', 'file20.txt'],
    'size': [1024, 2048, 512, 4096],
    'version': ['v1.10', 'v1.2', 'v1.1', 'v2.0']
})

# Sort by single column
sorted_df = natural_sort(df, by=['filename'])
print(sorted_df)
#    filename  size version
# 2  file1.txt   512    v1.1
# 1  file2.txt  2048    v1.2
# 0 file10.txt  1024   v1.10
# 3 file20.txt  4096    v2.0

# Sort by multiple columns
sorted_df = natural_sort(df, by=['version', 'filename'])
print(sorted_df)
# First by version (v1.1, v1.2, v1.10, v2.0), then by filename within versions
```

##### Advanced DataFrame Sorting

```python
# Complex DataFrame with multiple sortable columns
df = pd.DataFrame({
    'product': ['product1_v2', 'product1_v10', 'product2_v1', 'product1_v1'],
    'category': ['A1', 'A10', 'A2', 'A1'],
    'date': ['2024-01-10', '2024-01-02', '2024-01-15', '2024-01-01']
})

# Multi-column natural sort
sorted_df = natural_sort(df, by=['category', 'product'])
print(sorted_df)
# Results in proper ordering: A1 before A2 before A10,
# and product1_v1 before product1_v2 before product1_v10
```

##### Tuple and Set Sorting

```python
# Tuple sorting (returns list)
file_tuple = ("doc1.pdf", "doc10.pdf", "doc2.pdf")
sorted_files = natural_sort(file_tuple)
print(sorted_files)  # ["doc1.pdf", "doc2.pdf", "doc10.pdf"]
print(type(sorted_files))  # <class 'list'>

# Set sorting (returns list)
file_set = {"page1.html", "page10.html", "page2.html"}
sorted_files = natural_sort(file_set)
print(sorted_files)  # ["page1.html", "page2.html", "page10.html"]
print(type(sorted_files))  # <class 'list'>
```

## Use Cases and Applications

### File and Directory Management

```python
# Sort filenames naturally
files = ["report1.pdf", "report10.pdf", "report2.pdf", "summary.doc"]
sorted_files = natural_sort(files)
# Result: ["report1.pdf", "report2.pdf", "report10.pdf", "summary.doc"]

# Version control and releases
versions = ["v1.0.0", "v1.0.10", "v1.0.2", "v1.1.0", "v2.0.0"]
sorted_versions = natural_sort(versions)
# Result: ["v1.0.0", "v1.0.2", "v1.0.10", "v1.1.0", "v2.0.0"]
```

### Data Processing Pipelines

```python
# Flatten nested configuration data
config_data = [
    ["database", ["host", "localhost"], ["port", 5432]],
    ["cache", ["redis", ["host", "127.0.0.1"], ["port", 6379]]],
    ["logging", ["level", "INFO"]]
]
flat_config = flatten(config_data)
# Convert nested config to flat list for processing

# Process file lists in natural order
import os
files = os.listdir("./data/")
processed_files = natural_sort([f for f in files if f.endswith('.csv')])
for file in processed_files:
    # Process files in logical order: data1.csv, data2.csv, data10.csv
    pass
```

### Scientific Data Analysis

```python
import pandas as pd

# Sort experimental samples naturally
samples_df = pd.DataFrame({
    'sample_id': ['sample1_rep1', 'sample1_rep10', 'sample1_rep2', 'sample2_rep1'],
    'measurement': [1.23, 1.45, 1.34, 2.11],
    'condition': ['ctrl1', 'ctrl10', 'ctrl2', 'test1']
})

sorted_samples = natural_sort(samples_df, by=['condition', 'sample_id'])
# Ensures logical grouping and ordering of experimental data
```

### Log File Analysis

```python
# Flatten nested log structures
log_entries = [
    ["2024-01-01", ["INFO", "System started"]],
    ["2024-01-01", ["ERROR", ["Database", ["Connection failed", "Retry in 5s"]]]],
    ["2024-01-01", ["INFO", "System ready"]]
]
flat_logs = flatten(log_entries)

# Sort log files naturally
log_files = ["app.log.1", "app.log.10", "app.log.2", "error.log.1"]
sorted_logs = natural_sort(log_files)
# Process logs in chronological order
```

## Performance Considerations

### Flattening Performance

- **Time Complexity**: O(n) where n is the total number of elements
- **Space Complexity**: O(n) for the output list
- **Memory Efficient**: Uses generator expressions for lazy evaluation
- **NumPy Optimization**: Direct flattening for NumPy arrays

### Natural Sorting Performance

- **Time Complexity**: O(n log n) for sorting, O(m) per item for key generation (m = string length)
- **Memory Usage**: Creates temporary sorting columns for DataFrames
- **Pandas Optimization**: Leverages pandas' optimized sorting algorithms

## Error Handling

The module includes comprehensive error handling:

```python
# Type validation for natural_sort
try:
    result = natural_sort("invalid_type")
except TypeError as e:
    print(f"Error: {e}")
    # Error: The input must be one of type [pandas.DataFrame, pandas.Series, list, tuple, set]. Found <class 'str'>

# DataFrame without 'by' parameter
try:
    df = pd.DataFrame({'col': [1, 2, 3]})
    result = natural_sort(df)  # Missing 'by' parameter
except ValueError as e:
    print(f"Error: {e}")
    # Error: When sorting a pandas DataFrame, the `by` argument must be provided.
```

## Best Practices

### 1. Choose the Right Function

```python
# For nested structures - use flatten()
nested_data = [1, [2, [3, 4]], 5]
flat_data = flatten(nested_data)

# For natural ordering - use natural_sort()
filenames = ["file1.txt", "file10.txt", "file2.txt"]
ordered_files = natural_sort(filenames)
```

### 2. Handle DataFrame Sorting Properly

```python
# Always specify 'by' parameter for DataFrames
df = pd.DataFrame({'name': ['item1', 'item10', 'item2']})

# Correct usage
sorted_df = natural_sort(df, by=['name'])

# Multiple columns
sorted_df = natural_sort(df, by=['category', 'name'])
```

### 3. Memory Management for Large Datasets

```python
# For very large datasets, consider processing in chunks
def process_large_dataset(large_list):
    chunk_size = 10000
    for i in range(0, len(large_list), chunk_size):
        chunk = large_list[i:i + chunk_size]
        processed_chunk = natural_sort(chunk)
        yield processed_chunk
```

### 4. Type Awareness

```python
# Be aware of return type changes
original_tuple = ("c", "a", "b")
sorted_result = natural_sort(original_tuple)
print(type(sorted_result))  # <class 'list'> - not tuple!

# Convert back if needed
sorted_tuple = tuple(natural_sort(original_tuple))
```

## Integration Examples

### With File Processing

```python
import os
from pathlib import Path

def process_files_naturally(directory):
    """Process files in natural order."""
    files = os.listdir(directory)
    sorted_files = natural_sort(files)

    for filename in sorted_files:
        filepath = Path(directory) / filename
        print(f"Processing: {filename}")
        # Process file...

# Usage
process_files_naturally("./data/")
```

### With Data Analysis Pipelines

```python
def analyze_experiment_data(data_files):
    """Analyze experimental data in logical order."""
    # Flatten any nested file structures
    flat_files = flatten(data_files) if isinstance(data_files[0], (list, tuple)) else data_files

    # Sort files naturally
    ordered_files = natural_sort(flat_files)

    results = []
    for file in ordered_files:
        # Load and analyze each file
        df = pd.read_csv(file)
        analysis_result = df.groupby('condition').mean()
        results.append(analysis_result)

    return results
```

### With Configuration Management

```python
def flatten_config(config_dict):
    """Flatten nested configuration dictionaries."""
    def extract_values(d):
        if isinstance(d, dict):
            return [[k, extract_values(v)] for k, v in d.items()]
        elif isinstance(d, list):
            return [extract_values(item) for item in d]
        else:
            return d

    nested_config = extract_values(config_dict)
    return flatten(nested_config)

# Usage
config = {
    "database": {"host": "localhost", "port": 5432},
    "features": ["auth", "logging", "cache"]
}
flat_config = flatten_config(config)
```

## Limitations and Considerations

1. **String Preservation**: The `flatten()` function treats strings as atomic units to prevent character-level flattening
2. **Type Conversion**: Natural sorting of tuples and sets returns lists, not the original type
3. **DataFrame Columns**: Temporary sorting columns are created and cleaned up automatically
4. **Memory Usage**: Large nested structures may require significant memory during flattening
5. **Unicode Handling**: Natural sorting works with Unicode strings but numeric detection is ASCII-based

## Advanced Usage

### Custom Natural Sorting

```python
def custom_natural_sort_key(item, numeric_priority=True):
    """Custom natural sort with additional options."""
    parts = re.split(r'(\d+)', str(item))
    result = []

    for part in parts:
        if part.isdigit():
            # Convert to int/float based on preference
            result.append(int(part) if numeric_priority else float(part))
        else:
            result.append(part.lower())  # Case-insensitive text sorting

    return tuple(result)

# Use with standard Python sorted()
items = ["File1.TXT", "file10.txt", "File2.TXT"]
custom_sorted = sorted(items, key=custom_natural_sort_key)
```

### Performance Monitoring

```python
import time
from functools import wraps

def time_function(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# Monitor performance
@time_function
def large_data_processing():
    large_nested = [[i, [j, k]] for i in range(1000) for j in range(10) for k in range(5)]
    flattened = flatten(large_nested)
    return natural_sort([f"item{i}" for i in flattened[:100]])

result = large_data_processing()
```

This documentation provides comprehensive coverage of the utilities module, following the structure and style of the Database class documentation while being tailored to the specific functionality of the utility functions.
