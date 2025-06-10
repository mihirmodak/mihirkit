# Filesystem Module

## Overview

The `filesystem.py` module provides high-performance, memory-efficient utilities for filesystem operations using Python's `pathlib` and `os.scandir`. It offers optimized methods for finding files and directories with advanced filtering, sorting, and parallel processing capabilities.

## Features

- **High Performance**: Uses `os.scandir` for optimized directory traversal
- **Memory Efficient**: Generator-based methods for large directory trees
- **Parallel Processing**: Multi-threaded directory scanning for improved performance
- **Advanced Filtering**: Regex pattern matching and file extension filtering
- **Flexible Sorting**: Multiple sorting criteria with ascending/descending options
- **Caching**: LRU cache for frequently accessed patterns and results
- **Context Management**: File handling with automatic resource cleanup
- **Level Control**: Configurable directory traversal depth
- **Early Termination**: Head limits for large result sets

## Installation

```bash
# No additional dependencies required - uses standard library only
```

## Quick Start

```python
from filesystem import Directory, File

# Get all subdirectories
subdirs = Directory.get_subdirs("/path/to/directory")

# Get Python files with sorting
python_files = Directory.get_files("/path/to/project", ext="py", sort_on="modified")

# Work with a specific file
file = File("/path/to/file.txt")
print(file.contents)
```

## Classes

### Directory Class

The `Directory` class provides static methods for directory operations and can be instantiated to work with specific directory paths.

#### Constructor

```python
Directory(path: pathlib.Path | str)
```

**Parameters:**
- `path`: Directory path as string or pathlib.Path object

#### Finding Subdirectories

##### get_subdirs() - Comprehensive Subdirectory Search

```python
@staticmethod
@lru_cache
get_subdirs(
    path: Union[pathlib.Path, str],
    filter: Union[re.Pattern[str], str] = r".*",
    max_levels: int = -1,
    sort_on: Optional[Union[Literal["default", "created", "modified", "name", "size"], Callable]] = None,
    sort_order: Literal["asc", "desc", "ascending", "descending"] = "descending",
    use_parallel: bool = False,
    max_workers: int = 4
) -> List[pathlib.Path]
```

**Parameters:**
- `path`: Starting directory path
- `filter`: Regex pattern to match directory names (default: matches all)
- `max_levels`: Maximum depth to traverse (-1 for unlimited)
- `sort_on`: Sorting criteria ("default", "created", "modified", "name", "size", or custom function)
- `sort_order`: Sort direction ("asc"/"ascending" or "desc"/"descending")
- `use_parallel`: Enable parallel processing for large directory trees
- `max_workers`: Number of parallel workers when `use_parallel=True`

**Returns:**
- List of `pathlib.Path` objects representing matching subdirectories

##### iter_subdirs() - Memory-Efficient Generator

```python
@staticmethod
iter_subdirs(
    path: Union[pathlib.Path, str],
    filter: Union[re.Pattern[str], str] = r".*",
    max_levels: int = -1
) -> Generator[pathlib.Path, None, None]
```

Generator version for memory-efficient iteration over subdirectories without loading all results into memory.

#### Finding Files

##### get_files() - Comprehensive File Search

```python
@staticmethod
@lru_cache
get_files(
    path: Union[pathlib.Path, str],
    filter: Union[re.Pattern[str], str] = r".*",
    ext: str = "",
    sort_on: Optional[Union[Literal["default", "created", "modified", "name", "size"], Callable]] = None,
    sort_order: Literal["asc", "desc", "ascending", "descending"] = "descending",
    traverse_levels: Union[bool, int] = 0,
    head: Optional[int] = None,
    use_parallel: bool = False,
    max_workers: int = 4
) -> List[pathlib.Path]
```

**Parameters:**
- `path`: Starting directory path
- `filter`: Regex pattern to match file names
- `ext`: File extension filter (without leading dot)
- `sort_on`: Sorting criteria
- `sort_order`: Sort direction
- `traverse_levels`: Directory depth (`True`=-1, `False`=0, or specific integer)
- `head`: Limit number of results returned
- `use_parallel`: Enable parallel processing
- `max_workers`: Number of parallel workers

**Returns:**
- List of `pathlib.Path` objects representing matching files

##### iter_files() - Memory-Efficient File Generator

```python
@staticmethod
iter_files(
    path: Union[pathlib.Path, str],
    filter: Union[re.Pattern[str], str] = r".*",
    ext: str = "",
    traverse_levels: Union[bool, int] = 0
) -> Generator[pathlib.Path, None, None]
```

Generator version for memory-efficient file iteration.

### File Class

The `File` class provides utilities for working with individual files.

#### Constructor

```python
File(filepath: pathlib.Path | str)
```

**Parameters:**
- `filepath`: File path as string or pathlib.Path object

#### Properties

- `path`: The `pathlib.Path` object representing the file
- `contents`: Cached property that reads and returns the entire file contents as a string

#### Context Manager Support

```python
with File("/path/to/file.txt") as f:
    content = f.read()
```

## Usage Examples

> [!IMPORTANT]
Always handle filesystem errors appropriately in production code using try-except blocks.

### Basic Directory Operations

#### Finding Subdirectories

```python
from filesystem import Directory

# Get all subdirectories (current level only)
subdirs = Directory.get_subdirs("/home/user/projects", max_levels=0)

# Get all subdirectories recursively
all_subdirs = Directory.get_subdirs("/home/user/projects")

# Find directories matching a pattern
test_dirs = Directory.get_subdirs(
    "/home/user/projects",
    filter=r".*test.*"  # Directories containing "test"
)

# Find directories with regex pattern
python_dirs = Directory.get_subdirs(
    "/home/user",
    filter=r"^python.*$"  # Directories starting with "python"
)

# Limit traversal depth
shallow_subdirs = Directory.get_subdirs(
    "/home/user/projects",
    max_levels=2  # Only go 2 levels deep
)
```

#### Memory-Efficient Directory Iteration

```python
# Use generator for large directory trees
for subdir in Directory.iter_subdirs("/large/directory/tree"):
    print(f"Found directory: {subdir}")
    # Process directories one at a time without loading all into memory
```

### Advanced Directory Search with Sorting

```python
# Sort by creation time (newest first)
recent_dirs = Directory.get_subdirs(
    "/home/user/projects",
    sort_on="created",
    sort_order="descending"
)

# Sort by name (alphabetical)
sorted_dirs = Directory.get_subdirs(
    "/home/user/projects",
    sort_on="name",
    sort_order="ascending"
)

# Sort by size (largest first)
large_dirs = Directory.get_subdirs(
    "/home/user/projects",
    sort_on="size",
    sort_order="descending"
)

# Custom sorting function
def custom_sort(path):
    return len(path.name)  # Sort by directory name length

custom_sorted = Directory.get_subdirs(
    "/home/user/projects",
    sort_on=custom_sort,
    sort_order="ascending"
)
```

### Parallel Processing for Large Directory Trees

```python
# Enable parallel processing for large directories
large_tree_subdirs = Directory.get_subdirs(
    "/very/large/directory/tree",
    use_parallel=True,
    max_workers=8  # Use 8 parallel workers
)

# Parallel file search
large_tree_files = Directory.get_files(
    "/very/large/directory/tree",
    ext="txt",
    traverse_levels=True,  # Recursive
    use_parallel=True,
    max_workers=6
)
```

### File Search Operations

#### Basic File Finding

```python
# Get all files in current directory
all_files = Directory.get_files("/home/user/documents")

# Get Python files only
python_files = Directory.get_files(
    "/home/user/projects",
    ext="py"
)

# Get files matching pattern
config_files = Directory.get_files(
    "/home/user/projects",
    filter=r".*config.*"  # Files containing "config"
)

# Recursive file search
all_txt_files = Directory.get_files(
    "/home/user",
    ext="txt",
    traverse_levels=True  # Search all subdirectories
)

# Limited depth search
shallow_files = Directory.get_files(
    "/home/user/projects",
    ext="py",
    traverse_levels=2  # Only go 2 levels deep
)
```

#### Advanced File Search with Filters

```python
# Complex regex patterns
log_files = Directory.get_files(
    "/var/log",
    filter=r"^.*\.log(\.\d+)?$"  # .log files with optional rotation numbers
)

# Multiple criteria
recent_python_files = Directory.get_files(
    "/home/user/projects",
    ext="py",
    filter=r"^(test_|main|__init__|config).*",  # Specific file patterns
    sort_on="modified",
    sort_order="descending",
    head=20  # Get only the 20 most recently modified
)

# Case-insensitive matching with compiled regex
import re
case_insensitive_pattern = re.compile(r".*README.*", re.IGNORECASE)
readme_files = Directory.get_files(
    "/home/user/projects",
    filter=case_insensitive_pattern,
    traverse_levels=True
)
```

#### Sorted File Results

```python
# Sort by modification time (most recent first)
recent_files = Directory.get_files(
    "/home/user/documents",
    sort_on="modified",
    sort_order="descending"
)

# Sort by file size (largest first)
large_files = Directory.get_files(
    "/home/user/downloads",
    sort_on="size",
    sort_order="descending",
    head=10  # Top 10 largest files
)

# Sort by name with head limit
alpha_files = Directory.get_files(
    "/home/user/documents",
    sort_on="name",
    sort_order="ascending",
    head=50  # First 50 alphabetically
)
```

#### Memory-Efficient File Processing

```python
# Process large numbers of files without loading all into memory
for file_path in Directory.iter_files("/large/directory", ext="txt", traverse_levels=True):
    # Process each file individually
    with open(file_path, 'r') as f:
        # Process file content
        line_count = sum(1 for line in f)
        print(f"{file_path}: {line_count} lines")
```

### File Operations

#### Basic File Usage

```python
from filesystem import File

# Create file object
file = File("/path/to/document.txt")

# Access file path
print(f"File path: {file.path}")
print(f"File name: {file.path.name}")
print(f"File parent: {file.path.parent}")

# Read file contents (cached)
content = file.contents
print(f"File size: {len(content)} characters")

# Contents are cached, subsequent calls are fast
content_again = file.contents  # No file I/O, returns cached content
```

#### Context Manager Usage

```python
# Automatic file handling
with File("/path/to/data.txt") as f:
    lines = f.readlines()
    # File is automatically closed when exiting the context

# Manual resource management
file = File("/path/to/data.txt")
try:
    with file as f:
        content = f.read()
        # Process content
finally:
    # File is automatically closed by context manager
    pass
```

### Performance Optimization Examples

#### Choosing the Right Method

```python
# For small directories (< 1000 items), use regular methods
small_dir_files = Directory.get_files("/small/directory")

# For large directories, use parallel processing
large_dir_files = Directory.get_files(
    "/large/directory",
    use_parallel=True,
    max_workers=4
)

# For very large result sets, use generators
for file_path in Directory.iter_files("/huge/directory/tree", traverse_levels=True):
    # Process one file at a time
    process_file(file_path)

# For sorted results with limits, combine efficiently
top_recent = Directory.get_files(
    "/documents",
    sort_on="modified",
    sort_order="descending",
    head=100  # Only get top 100, more efficient than sorting everything
)
```

#### Caching Benefits

```python
# First call scans filesystem
files1 = Directory.get_files("/home/user/projects", ext="py")

# Second call with same parameters returns cached result
files2 = Directory.get_files("/home/user/projects", ext="py")  # Fast!

# Different parameters trigger new scan
files3 = Directory.get_files("/home/user/projects", ext="txt")  # New scan
```

### Real-World Examples

#### Project Analysis

```python
def analyze_project_structure(project_path):
    """Analyze a software project's structure."""
    project = pathlib.Path(project_path)

    # Get all Python files
    python_files = Directory.get_files(
        project_path,
        ext="py",
        traverse_levels=True,
        sort_on="size",
        sort_order="descending"
    )

    # Get test directories
    test_dirs = Directory.get_subdirs(
        project_path,
        filter=r".*(test|spec).*",
        max_levels=3
    )

    # Get configuration files
    config_files = Directory.get_files(
        project_path,
        filter=r".*(config|settings|\.env|\.ini|\.yaml|\.json).*",
        traverse_levels=True
    )

    return {
        "python_files": len(python_files),
        "largest_file": python_files[0] if python_files else None,
        "test_directories": len(test_dirs),
        "config_files": len(config_files)
    }

# Usage
stats = analyze_project_structure("/home/user/myproject")
print(f"Python files: {stats['python_files']}")
print(f"Test directories: {stats['test_directories']}")
```

#### Log File Management

```python
def find_recent_logs(log_directory, days=7):
    """Find log files modified in the last N days."""
    import time

    cutoff_time = time.time() - (days * 24 * 60 * 60)

    # Get all log files, sorted by modification time
    log_files = Directory.get_files(
        log_directory,
        filter=r".*\.log(\.\d+)?$",
        traverse_levels=True,
        sort_on="modified",
        sort_order="descending"
    )

    # Filter by modification time
    recent_logs = [
        log_file for log_file in log_files
        if log_file.stat().st_mtime > cutoff_time
    ]

    return recent_logs

# Usage
recent_logs = find_recent_logs("/var/log", days=3)
for log in recent_logs:
    print(f"Recent log: {log}")
```

#### Backup File Cleanup

```python
def cleanup_backup_files(directory, keep_count=5):
    """Keep only the N most recent backup files."""

    backup_files = Directory.get_files(
        directory,
        filter=r".*\.bak$|.*\.backup$|.*~$",
        traverse_levels=True,
        sort_on="created",
        sort_order="descending"
    )

    # Keep only the most recent backups
    files_to_delete = backup_files[keep_count:]

    for file_path in files_to_delete:
        print(f"Would delete: {file_path}")
        # file_path.unlink()  # Uncomment to actually delete

    return len(files_to_delete)

# Usage
deleted_count = cleanup_backup_files("/home/user/backups", keep_count=10)
print(f"Would delete {deleted_count} backup files")
```

## API Reference

### Directory Class Methods

#### Static Methods

```python
get_subdirs(path, filter=r".*", max_levels=-1, sort_on=None,
           sort_order="descending", use_parallel=False, max_workers=4) -> List[pathlib.Path]

iter_subdirs(path, filter=r".*", max_levels=-1) -> Generator[pathlib.Path, None, None]

get_files(path, filter=r".*", ext="", sort_on=None, sort_order="descending",
         traverse_levels=0, head=None, use_parallel=False, max_workers=4) -> List[pathlib.Path]

iter_files(path, filter=r".*", ext="", traverse_levels=0) -> Generator[pathlib.Path, None, None]
```

#### Aliases

- `get_subdirectories()` → `get_subdirs()`
- `get_subfolders()` → `get_subdirs()`
- `Folder` → `Directory` (class alias)

### File Class

#### Properties

- `path`: `pathlib.Path` object
- `contents`: `str` (cached property)

#### Methods

- `__init__(filepath)`: Constructor
- `__enter__()`: Context manager entry
- `__exit__()`: Context manager exit
- `__repr__()`: String representation

### Sorting Options

The `sort_on` parameter accepts these values:

- `"default"`: Combines path and creation time
- `"name"`: Sort by file/directory name
- `"size"`: Sort by file/directory size
- `"created"`: Sort by creation time
- `"modified"`: Sort by modification time
- `callable`: Custom sorting function

### Filter Patterns

The `filter` parameter accepts:

- **String**: Interpreted as regex pattern (e.g., `r".*\.py$"`)
- **Compiled regex**: Pre-compiled `re.Pattern` object
- **Common patterns**:
  - `r".*"`: Match all (default)
  - `r"^test_.*"`: Files/dirs starting with "test_"
  - `r".*\.py$"`: Files ending with ".py"
  - `r".*(config|settings).*"`: Contains "config" or "settings"

## Performance Characteristics

### Optimization Features

1. **LRU Caching**: Regex patterns and method results are cached
2. **Early Termination**: `head` parameter stops processing when limit reached
3. **Parallel Processing**: Multi-threaded scanning for large directories
4. **Memory Efficiency**: Generator methods for large result sets
5. **Fast Paths**: Optimized code paths for common operations

### Performance Guidelines

#### When to Use Parallel Processing

```python
# Use parallel for large directory trees (>10,000 items)
large_files = Directory.get_files(
    "/huge/directory",
    use_parallel=True,
    max_workers=4  # Adjust based on CPU cores
)

# Don't use parallel for small directories (overhead not worth it)
small_files = Directory.get_files("/small/dir")  # use_parallel=False (default)
```

#### Memory Usage Optimization

```python
# For large result sets, use generators
for file_path in Directory.iter_files("/large/tree", traverse_levels=True):
    process_file(file_path)  # Process one at a time

# For limited results, use head parameter
top_files = Directory.get_files("/dir", head=100)  # Stop after 100 files

# For sorted limited results, combine efficiently
recent_files = Directory.get_files(
    "/dir",
    sort_on="modified",
    sort_order="desc",
    head=50
)
```

### Caching Behavior

- **Method Results**: Cached based on all parameters
- **Regex Patterns**: Compiled patterns are cached up to 128 entries
- **File Contents**: `File.contents` property caches file content
- **Cache Invalidation**: No automatic invalidation (restart process for fresh results)

## Error Handling

The module handles common filesystem errors gracefully:

- **Permission Errors**: Directories/files without read access are skipped
- **OS Errors**: Invalid paths or filesystem issues are handled
- **Broken Symlinks**: Ignored during traversal

### Exception Handling Examples

```python
try:
    files = Directory.get_files("/restricted/directory")
except PermissionError:
    print("Access denied to directory")
except FileNotFoundError:
    print("Directory does not exist")
except OSError as e:
    print(f"Filesystem error: {e}")

# The module automatically handles errors during traversal
files = Directory.get_files("/mixed/permissions/directory")  # Skips inaccessible subdirs
```

## Best Practices

### 1. Choose the Right Method

```python
# For large directories, use parallel processing
if directory_is_large:
    files = Directory.get_files(path, use_parallel=True)
else:
    files = Directory.get_files(path)

# For memory efficiency with large results, use generators
for file_path in Directory.iter_files(path, traverse_levels=True):
    process_file(file_path)
```

### 2. Use Appropriate Filters

```python
# Pre-compile regex for repeated use
import re
pattern = re.compile(r".*\.py$", re.IGNORECASE)
python_files = Directory.get_files("/project", filter=pattern)

# Use extension filter when appropriate (more efficient)
python_files = Directory.get_files("/project", ext="py")  # Better than regex
```

### 3. Optimize Sorting and Limits

```python
# When you only need top N results, use head parameter
top_files = Directory.get_files(
    "/dir",
    sort_on="size",
    sort_order="desc",
    head=10  # Much more efficient than sorting all then slicing
)
```

### 4. Handle Large Directory Trees

```python
# For very large trees, use generators and process incrementally
def process_large_tree(root_path):
    count = 0
    for file_path in Directory.iter_files(root_path, traverse_levels=True):
        process_file(file_path)
        count += 1
        if count % 1000 == 0:
            print(f"Processed {count} files...")
```

### 5. Resource Management

```python
# Use context managers for file operations
with File("/path/to/file.txt") as f:
    content = f.read()
    # File automatically closed

# For cached content access
file = File("/path/to/file.txt")
content = file.contents  # Cached, efficient for multiple accesses
```

## Limitations

- **No Automatic Cache Invalidation**: Results are cached until process restart
- **Platform Dependencies**: Some features may behave differently on different operating systems
- **Symlink Handling**: Follows symlinks by default (can be changed with `follow_symlinks=False`)
- **Large File Content**: `File.contents` loads entire file into memory
- **Regex Compilation**: String patterns are compiled on each call (use pre-compiled patterns for performance)

## Advanced Usage

### Custom Sorting Functions

```python
def sort_by_extension_then_name(path):
    """Sort by file extension, then by name."""
    return (path.suffix.lower(), path.name.lower())

files = Directory.get_files(
    "/project",
    sort_on=sort_by_extension_then_name,
    sort_order="ascending"
)
```

### Complex Filtering

```python
# Combine regex and extension filtering
import re

# Find Python files but exclude test files
def filter_non_test_python(path):
    return path.suffix == '.py' and not path.name.startswith('test_')

python_files = [
    f for f in Directory.get_files("/project", traverse_levels=True)
    if filter_non_test_python(f)
]
```

### Performance Monitoring

```python
import time

# Measure performance
start_time = time.time()
files = Directory.get_files("/large/directory", use_parallel=True, max_workers=8)
elapsed = time.time() - start_time
print(f"Found {len(files)} files in {elapsed:.2f} seconds")
```

This comprehensive documentation provides detailed guidance for effectively using the filesystem module's capabilities for high-performance file and directory operations.
