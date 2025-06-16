# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).


## [1.0.0] - 2025-06-16

### BREAKING CHANGES

- Database constructor now accepts `cache_size` parameter (default: 128)
- `select()` method signature updated with `ignore_cache` parameter
- `_bulk_insert()` method signature changed to include `if_exists` and `index` parameters

### Added

#### Database Module Enhancements
- QueryCache class for manual query result caching with LRU eviction
- `create_table()` method with support for constraints, foreign keys, and indexes
- Cache management methods (`clear_cache()`, `get_cache_stats()`) to Database class
- **kwargs support to insert methods for improved flexibility
- Algorithms submodule support to __init__.py with lazy loading

### Changed

- Replace @lru_cache decorator with manual caching in select() method using QueryCache
- Enhanced `_bulk_insert()` with better type safety and pandas to_sql parameter support
- Improved type hints throughout database module


## [0.0.1] - 2025-06-09

### Added

#### Core Infrastructure
- Pre-commit configuration (.pre-commit-config.yaml) with comprehensive code quality hooks
  - Black code formatter
  - isort import sorting
  - flake8 linting with additional plugins (docstrings, bugbear, comprehensions)
  - Bandit security scanning
  - mypy type checking
  - Commit message validation with commitizen
- CI settings for automated fixes and weekly dependency updates

#### Database Module Enhancements
- Bulk insert support for pandas DataFrames in Database class
- LRU caching on `select()` method for improved performance
- Stored procedure support with `callproc()` method including input/output parameters
- Comprehensive connection string construction from various configuration sources
- TTL caching support for expensive operations
- Enhanced error handling with security warnings for unsafe operations
- Support for multiple database engines (SQLite, PostgreSQL, MySQL, Oracle)
- Parameterized queries for SQL injection protection
- Transaction management with automatic commit/rollback

#### Filesystem Module (New)
- High-performance `Directory` class with os.scandir optimization
- Parallel processing support for large directory trees (configurable worker count)
- Memory-efficient generators (`iter_files`, `iter_subdirs`) for large result sets
- Advanced filtering with regex patterns and file extension support
- Multiple sorting criteria (name, size, created, modified) with ascending/descending options
- `File` class with context manager and cached property support
- Level-controlled directory traversal with configurable depth limits
- Early termination with head limits for performance optimization

#### Utilities Module (New)
- `flatten()` function for recursive flattening of nested structures
  - Support for lists, tuples, and NumPy arrays
  - Handles arbitrary nesting depth and mixed data types
- `natural_sort()` function with human-readable string sorting
  - Handles numeric sorting within strings ("item2" before "item10")
  - Support for pandas DataFrame/Series sorting with natural ordering
  - Works with lists, tuples, sets, and pandas objects
- `natural_sort_key()` helper function for custom sorting implementations
- Comprehensive type hints and error handling throughout

#### Decorators Module Enhancements
- `@timeout` decorator with multiprocessing and automatic threading fallback
  - Process-based execution limits for reliable operation
  - Automatic fallback to threading for non-picklable objects
  - Configurable timeout periods with proper cleanup
- Enhanced `@retry` decorator with exponential backoff and selective exception handling
  - Configurable retry attempts and delay intervals
  - Support for specific exception types to retry on
  - Detailed warning messages for retry attempts
- Convenience decorators for common use cases:
  - `@selective_retry()` for specific transient exceptions
  - `@network_retry()` optimized for network operations
- Custom `property` class with advanced caching features:
  - TTL (time-to-live) support for automatic cache expiration
  - Setter support with value transformation
  - Manual cache invalidation methods
- Improved error messages and warning clarity across all decorators
- `DisabledMethodError` custom exception for permanently blocked methods

#### Project Configuration
- Enhanced pyproject.toml with comprehensive dependency management
  - Core dependencies: numpy, pandas, openpyxl, python-dotenv, sqlalchemy
  - Proper package discovery configuration
  - GitHub homepage URL and project metadata
- Commitizen configuration for conventional commits and automatic changelog generation
- Dynamic imports in __init__.py for better module organization and lazy loading
- Comprehensive .gitignore covering Python, IDE, and build artifacts

#### Documentation
- Detailed module documentation (db.md, decorators.md, filesystem.md, utilities.md)
- Comprehensive README.md with quick start guide and feature overview
- Usage examples and API reference for all modules
- Performance considerations and optimization guidelines
- Troubleshooting guides and error handling patterns
- Security features documentation and best practices
- Installation instructions and dependency information

### Security
- SQL injection protection through parameterized queries and identifier sanitization
- Security warnings for potentially unsafe operations (raw SQL execution)
- Bandit security scanning in pre-commit hooks
- Input validation and sanitization throughout all modules

### Performance
- LRU caching for frequently accessed database query results
- Parallel processing support for filesystem operations on large directories
- Memory-efficient generators for processing large datasets
- Optimized algorithms using os.scandir for filesystem traversal
- Cached properties for expensive computations with TTL support

---

**Note**: This changelog covers the initial development phase of mihirkit, transforming it from a basic database utility into a comprehensive Python toolkit with production-ready features, extensive documentation, and development best practices.
