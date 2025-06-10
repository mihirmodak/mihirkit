"""
Filesystem utilities module providing high-performance directory and file operations.

This module offers optimized utilities for filesystem operations using Python's pathlib
and os.scandir. It provides memory-efficient methods for finding files and directories
with advanced filtering, sorting, and parallel processing capabilities.

Key Features:
    * High Performance: Uses os.scandir for optimized directory traversal
    * Memory Efficient: Generator-based methods for large directory trees
    * Parallel Processing: Multi-threaded directory scanning for improved performance
    * Advanced Filtering: Regex pattern matching and file extension filtering
    * Flexible Sorting: Multiple sorting criteria with ascending/descending options
    * Caching: LRU cache for frequently accessed patterns and results
    * Context Management: File handling with automatic resource cleanup
    * Level Control: Configurable directory traversal depth
    * Early Termination: Head limits for large result sets

Classes:
    Directory: High-performance directory operations with static methods
    File: File operations with context management and content caching
    Folder: Alias for Directory class

Functions:
    _get_compiled_pattern: Cache compiled regex patterns to avoid recompilation

The module is designed for performance-critical applications that need to process
large directory trees efficiently while providing flexible filtering and sorting options.
"""

import os
import pathlib
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import cached_property, lru_cache
from typing import IO, Callable, Dict, Generator, List, Literal, Optional, Union


# Cache for compiled regex patterns
@lru_cache(maxsize=128)
def _get_compiled_pattern(pattern: str) -> re.Pattern:
    """Cache compiled regex patterns to avoid recompilation."""
    return re.compile(pattern)


# Sorting functions map
__SORTING_NAME_FUNCTION_MAP: Dict[str, Callable] = {
    "default": lambda path: (
        str(path.parent / path.stem),
        os.path.getctime(path),
    ),
    "name": lambda path: str(path.parent / path.stem),
    "size": os.path.getsize,
    "created": os.path.getctime,
    "modified": os.path.getmtime,
}


class Directory:
    """
    High-performance directory operations with advanced filtering and sorting capabilities.

    Provides static methods for directory traversal, file finding, and subdirectory discovery
    using optimized algorithms with optional parallel processing. Supports regex filtering,
    multiple sorting criteria, and memory-efficient iteration for large directory trees.

    Attributes:
        path (pathlib.Path): Directory path when instantiated as an object

    Example:
        ```python
        # Static usage (recommended)
        subdirs = Directory.get_subdirs("/path/to/directory")
        files = Directory.get_files("/path", ext="py", sort_on="modified")

        # Instance usage
        dir_obj = Directory("/path/to/directory")
        print(dir_obj.path)
        ```
    """

    path: pathlib.Path

    def __init__(self, path: pathlib.Path | str):
        """
        Initialize Directory object with specified path.

        Args:
            path: Directory path as string or pathlib.Path object
        """
        self.path = pathlib.Path(path)

    @staticmethod
    @lru_cache
    def get_subdirs(
        path: Union[pathlib.Path, str],
        filter: Union[re.Pattern[str], str] = r".*",
        max_levels: int = -1,
        sort_on: Optional[
            Union[Literal["default", "created", "modified", "name", "size"], Callable]
        ] = None,
        sort_order: Literal["asc", "desc", "ascending", "descending"] = "descending",
        use_parallel: bool = False,
        max_workers: int = 4,
    ) -> List[pathlib.Path]:
        """
        Find subdirectories efficiently using os.scandir with iterative traversal.

        Args:
            path: Starting directory path
            filter: Regex pattern to match directory names
            max_levels: Maximum depth to traverse (-1 for unlimited)
            sort_on: Sorting criteria (None to skip sorting)
            sort_order: Sort order (ascending or descending)
            use_parallel: Use parallel processing for large directory trees
            max_workers: Number of parallel workers if use_parallel is True
        """
        path = pathlib.Path(path)
        filter_pattern = (
            _get_compiled_pattern(filter) if isinstance(filter, str) else filter
        )

        if use_parallel and max_levels != 0:
            return Directory._subdirs_parallel(
                path, filter_pattern, max_levels, sort_on, sort_order, max_workers
            )

        # Fast path for single level
        if max_levels == 0:
            return Directory._subdirs_single_level(
                path, filter_pattern, sort_on, sort_order
            )

        # Iterative traversal using scandir
        matching_subdirs = []
        queue = deque([(path, 0)])

        while queue:
            current_path, level = queue.popleft()

            if max_levels >= 0 and level >= max_levels:
                continue

            try:
                with os.scandir(current_path) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            entry_path: pathlib.Path = pathlib.Path(entry.path)
                            if filter_pattern.match(entry.name):
                                matching_subdirs.append(entry_path)
                            if max_levels < 0 or level + 1 < max_levels:
                                queue.append((entry_path, level + 1))
            except OSError:
                continue

        # Only sort if requested
        if sort_on is not None:
            sorting_function = (
                sort_on
                if callable(sort_on)
                else __SORTING_NAME_FUNCTION_MAP.get(
                    sort_on, __SORTING_NAME_FUNCTION_MAP["default"]
                )
            )
            is_descending = sort_order.lower() in ["descending", "desc"]
            return sorted(matching_subdirs, key=sorting_function, reverse=is_descending)

        return matching_subdirs

    @staticmethod
    def _subdirs_single_level(
        path: pathlib.Path,
        filter_pattern: re.Pattern,
        sort_on: Optional[Union[str, Callable]],
        sort_order: str,
    ) -> List[pathlib.Path]:
        """Optimized method for single directory level."""
        matching_subdirs = []

        try:
            with os.scandir(path) as entries:
                for entry in entries:
                    if entry.is_dir(follow_symlinks=False) and filter_pattern.match(
                        entry.name
                    ):
                        matching_subdirs.append(pathlib.Path(entry.path))
        except OSError:
            pass

        if sort_on is not None:
            sorting_function = (
                sort_on
                if callable(sort_on)
                else __SORTING_NAME_FUNCTION_MAP.get(
                    sort_on, __SORTING_NAME_FUNCTION_MAP["default"]
                )
            )
            is_descending = sort_order.lower() in ["descending", "desc"]
            return sorted(matching_subdirs, key=sorting_function, reverse=is_descending)

        return matching_subdirs

    @staticmethod
    def _subdirs_parallel(
        path: pathlib.Path,
        filter_pattern: re.Pattern,
        max_levels: int,
        sort_on: Optional[Union[str, Callable]],
        sort_order: str,
        max_workers: int,
    ) -> List[pathlib.Path]:
        """Parallel implementation for large directory trees."""
        matching_subdirs = []

        def scan_directory(dir_info):
            dir_path, level = dir_info
            local_matches = []
            subdirs_to_scan = []

            if max_levels >= 0 and level >= max_levels:
                return local_matches, subdirs_to_scan

            try:
                with os.scandir(dir_path) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            if filter_pattern.match(entry.name):
                                local_matches.append(pathlib.Path(entry.path))
                            if max_levels < 0 or level + 1 < max_levels:
                                subdirs_to_scan.append((entry.path, level + 1))
            except OSError:
                pass

            return local_matches, subdirs_to_scan

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scan_directory, (path, 0))}

            while futures:
                done = set()
                for future in as_completed(futures):
                    done.add(future)
                    matches, subdirs = future.result()
                    matching_subdirs.extend(matches)

                    for subdir_info in subdirs:
                        futures.add(executor.submit(scan_directory, subdir_info))

                futures -= done

        if sort_on is not None:
            sorting_function = (
                sort_on
                if callable(sort_on)
                else __SORTING_NAME_FUNCTION_MAP.get(
                    sort_on, __SORTING_NAME_FUNCTION_MAP["default"]
                )
            )
            is_descending = sort_order.lower() in ["descending", "desc"]
            return sorted(matching_subdirs, key=sorting_function, reverse=is_descending)

        return matching_subdirs

    @staticmethod
    def iter_subdirs(
        path: Union[pathlib.Path, str],
        filter: Union[re.Pattern[str], str] = r".*",
        max_levels: int = -1,
    ) -> Generator[pathlib.Path, None, None]:
        """Generate subdirectories for memory-efficient iteration."""
        path = pathlib.Path(path)
        filter_pattern = (
            _get_compiled_pattern(filter) if isinstance(filter, str) else filter
        )

        queue = deque([(path, 0)])

        while queue:
            current_path, level = queue.popleft()

            if max_levels >= 0 and level >= max_levels:
                continue

            try:
                with os.scandir(current_path) as entries:
                    for entry in entries:
                        if entry.is_dir(follow_symlinks=False):
                            if filter_pattern.match(entry.name):
                                yield pathlib.Path(entry.path)
                            if max_levels < 0 or level + 1 < max_levels:
                                queue.append((pathlib.Path(entry.path), level + 1))
            except OSError:
                continue

    @staticmethod
    @lru_cache
    def get_files(
        path: Union[pathlib.Path, str],
        filter: Union[re.Pattern[str], str] = r".*",
        ext: str = "",
        sort_on: Optional[
            Union[Literal["default", "created", "modified", "name", "size"], Callable]
        ] = None,
        sort_order: Literal["asc", "desc", "ascending", "descending"] = "descending",
        traverse_levels: Union[bool, int] = 0,
        head: Optional[int] = None,
        use_parallel: bool = False,
        max_workers: int = 4,
    ) -> List[pathlib.Path]:
        """
        Find files efficiently using optimized traversal methods.

        Args:
            path: Starting directory path
            filter: Regex pattern to match file names
            ext: File extension filter (without dot)
            sort_on: Sorting criteria (None to skip sorting)
            sort_order: Sort order
            traverse_levels: How many levels to traverse (True=-1, False=0)
            head: Limit number of results
            use_parallel: Use parallel processing for large trees
            max_workers: Number of parallel workers
        """
        path = pathlib.Path(path)
        filter_pattern = (
            _get_compiled_pattern(filter) if isinstance(filter, str) else filter
        )

        if isinstance(traverse_levels, bool):
            traverse_levels = -1 if traverse_levels else 0

        ext = ext.lstrip(".") if ext else ""

        # Fast paths for common cases
        if filter == r".*" and ext and traverse_levels == 0:
            files = [p for p in path.glob(f"*.{ext}") if p.is_file()]
            if sort_on is None and head:
                return files[:head]
        elif filter == r".*" and not ext and traverse_levels == 0:
            files = [p for p in path.iterdir() if p.is_file()]
            if sort_on is None and head:
                return files[:head]
        else:
            # Use appropriate method based on requirements
            if use_parallel and traverse_levels != 0:
                files = Directory._files_parallel(
                    path, filter_pattern, ext, traverse_levels, max_workers
                )
            else:
                files = list(
                    Directory._iter_files_scandir(
                        path,
                        filter_pattern,
                        ext,
                        traverse_levels,
                        head if sort_on is None else None,
                    )
                )

        # Apply sorting if requested
        if sort_on is not None:
            sorting_function = (
                sort_on
                if callable(sort_on)
                else __SORTING_NAME_FUNCTION_MAP.get(
                    sort_on, __SORTING_NAME_FUNCTION_MAP["default"]
                )
            )
            is_descending = sort_order.lower() in ["descending", "desc"]
            files = sorted(files, key=sorting_function, reverse=is_descending)

        # Apply head limit after sorting
        if head and len(files) > head:
            files = files[:head]

        return files

    @staticmethod
    def _iter_files_scandir(
        path: pathlib.Path,
        filter_pattern: re.Pattern,
        ext: str,
        traverse_levels: int,
        early_limit: Optional[int] = None,
    ) -> Generator[pathlib.Path, None, None]:
        """Memory-efficient file iteration using scandir."""
        ext_match = f".{ext}" if ext else ""
        count = 0

        queue = deque([(path, 0)])

        while queue:
            current_path, level = queue.popleft()

            if traverse_levels >= 0 and level > traverse_levels:
                continue

            try:
                with os.scandir(current_path) as entries:
                    for entry in entries:
                        if entry.is_file(follow_symlinks=False):
                            if (
                                not ext or entry.name.endswith(ext_match)
                            ) and filter_pattern.match(entry.name):
                                yield pathlib.Path(entry.path)
                                count += 1
                                if early_limit and count >= early_limit:
                                    return
                        elif entry.is_dir(follow_symlinks=False) and (
                            traverse_levels < 0 or level < traverse_levels
                        ):
                            queue.append((pathlib.Path(entry.path), level + 1))
            except OSError:
                continue

    @staticmethod
    def _files_parallel(
        path: pathlib.Path,
        filter_pattern: re.Pattern,
        ext: str,
        traverse_levels: int,
        max_workers: int,
    ) -> List[pathlib.Path]:
        """Parallel file search for large directory trees."""
        matching_files = []
        ext_match = f".{ext}" if ext else ""

        def scan_directory(dir_info):
            dir_path, level = dir_info
            local_files = []
            subdirs_to_scan = []

            if traverse_levels >= 0 and level > traverse_levels:
                return local_files, subdirs_to_scan

            try:
                with os.scandir(dir_path) as entries:
                    for entry in entries:
                        if entry.is_file(follow_symlinks=False):
                            if (
                                not ext or entry.name.endswith(ext_match)
                            ) and filter_pattern.match(entry.name):
                                local_files.append(pathlib.Path(entry.path))
                        elif entry.is_dir(follow_symlinks=False) and (
                            traverse_levels < 0 or level < traverse_levels
                        ):
                            subdirs_to_scan.append((entry.path, level + 1))
            except OSError:
                pass

            return local_files, subdirs_to_scan

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(scan_directory, (path, 0))}

            while futures:
                done = set()
                for future in as_completed(futures):
                    done.add(future)
                    files, subdirs = future.result()
                    matching_files.extend(files)

                    for subdir_info in subdirs:
                        futures.add(executor.submit(scan_directory, subdir_info))

                futures -= done

        return matching_files

    @staticmethod
    def iter_files(
        path: Union[pathlib.Path, str],
        filter: Union[re.Pattern[str], str] = r".*",
        ext: str = "",
        traverse_levels: Union[bool, int] = 0,
    ) -> Generator[pathlib.Path, None, None]:
        """Generate files for memory-efficient iteration."""
        path = pathlib.Path(path)
        filter_pattern = (
            _get_compiled_pattern(filter) if isinstance(filter, str) else filter
        )

        if isinstance(traverse_levels, bool):
            traverse_levels = -1 if traverse_levels else 0

        ext = ext.lstrip(".") if ext else ""

        yield from Directory._iter_files_scandir(
            path, filter_pattern, ext, traverse_levels
        )

    # Aliases
    get_subdirectories = get_subdirs
    get_subfolders = get_subdirs


class File:
    """
    File operations with context management and content caching capabilities.

    Provides utilities for working with individual files including automatic
    resource management through context managers and cached content reading
    for improved performance on repeated access.

    Attributes:
        path (pathlib.Path): File path as pathlib.Path object

    Example:
        ```python
        # Context manager usage
        with File("/path/to/file.txt") as f:
            content = f.read()

        # Cached content access
        file = File("/path/to/file.txt")
        content = file.contents  # Cached after first access
        ```
    """

    path: pathlib.Path

    def __init__(self, filepath: pathlib.Path | str):
        """
        Initialize File object with specified path.

        Args:
            filepath: File path as string or pathlib.Path object
        """
        self.path = pathlib.Path(filepath)

    def __repr__(self) -> str:
        """Return string representation of the file path."""
        return str(self.path)

    def __enter__(self) -> IO:
        """Open the file for reading when entering the context."""
        self._file = self.path.open("r")
        return self._file

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Close the file when exiting the context."""
        if hasattr(self, "_file") and self._file and not self._file.closed:
            self._file.close()

    @cached_property
    def contents(self) -> str:
        """Read and cache file contents."""
        with open(self.path) as f:
            return f.read()


# Alias
Folder = Directory
