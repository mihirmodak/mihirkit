"""
High-performance Python utilities for database operations, filesystem management, and data processing.

This module provides a comprehensive toolkit for common programming tasks:

Database Operations:
    - Database: Secure, multi-engine database interface with CRUD operations,
      transaction management, and SQL injection protection

Filesystem Management:
    - Directory/Folder: Optimized file and directory discovery with regex filtering,
      parallel processing, and memory-efficient iteration
    - File: Simple file handling with caching and context management

Decorators:
    - @disabled: Permanently block method execution for security/compliance
    - @deprecated: Mark methods for migration with informative warnings
    - @timeout: Process-based execution limits with automatic fallback
    - @retry/@selective_retry/@network_retry: Resilient operation handling
    - property: Cached properties with TTL support and setter functionality

Utilities:
    - flatten: Recursively flatten nested lists, tuples, and NumPy arrays
    - natural_sort: Human-readable sorting for strings with embedded numbers
    - natural_sort_key: Generate natural sorting keys for custom sorting

Key Features:
    - Performance optimized with LRU caching and parallel processing
    - Memory efficient with generator-based iteration for large datasets
    - Production ready with comprehensive error handling and security features
    - Type safe with full type hints and validation

Examples:
    >>> from mihirkit import Database, Directory, retry, natural_sort
    >>>
    >>> # Database operations
    >>> with Database("sqlite:///app.db") as db:
    ...     users = db.select("users", where={"active": True})
    >>>
    >>> # File discovery
    >>> python_files = Directory.get_files("/project", ext="py", sort_on="modified")
    >>>
    >>> # Resilient operations
    >>> @retry(max_retries=3)
    >>> def api_call():
    ...     return requests.get("https://api.example.com")
    >>>
    >>> # Natural sorting
    >>> files = natural_sort(["file1.txt", "file10.txt", "file2.txt"])
    >>> # Returns: ["file1.txt", "file2.txt", "file10.txt"]

Dependencies:
    Core: pathlib, os, re, functools, concurrent.futures, multiprocessing
    Optional: pandas, numpy, sqlalchemy, python-dotenv (for specific features)
"""

import importlib

_module_exports = {
    # Import Name: ('.' + File Name, Class / Function Name)
    # Database
    "Database": (".db", "Database"),
    # decorators
    "DisabledMethodError": (".decorators", "DisabledMethodError"),
    "disabled": (".decorators", "disabled"),
    "deprecated": (".decorators", "deprecated"),
    "_is_picklable": (".decorators", "_is_picklable"),
    "_timeout_with_threads": (".decorators", "_timeout_with_threads"),
    "_timeout_with_process": (".decorators", "_timeout_with_process"),
    "timeout": (".decorators", "timeout"),
    "retry": (".decorators", "retry"),
    "selective_retry": (".decorators", "selective_retry"),
    "network_retry": (".decorators", "network_retry"),
    "property": (".decorators", "property"),
    # filesystem
    "Directory": (".filesystem", "Directory"),
    "Folder": (".filesystem", "Folder"),
    "File": (".filesystem", "File"),
    # utilities
    "flatten": (".utilities", "flatten"),
    "natural_sort_key": (".utilities", "natural_sort_key"),
    "natural_sort": (".utilities", "natural_sort"),
}


def __getattr__(name):
    attr = None
    module_path = None
    module = None
    attr_name = ""

    if name in _module_exports:
        module_path, attr_name = _module_exports[name]
        module = importlib.import_module(module_path, __name__)

    # Get the attribute from the module
    try:
        attr = getattr(module, attr_name)
    except AttributeError as err:
        raise ImportError(
            f"Cannot import name '{attr_name} from '{module_path}''"
        ) from err

    # Cache the attribute in this module
    globals()[name] = attr

    if attr is None:
        raise AttributeError(f"Module '{__name__}' has no attribute '{name}'")

    return attr


# Allow for `from package import *`
__all__ = list(_module_exports.keys())  # type: ignore
