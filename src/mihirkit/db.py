"""
Database Utilities Module.

A high-level, secure database interface built on SQLAlchemy that provides simplified
CRUD operations, configuration management, and security features for Python applications.

This module offers a Database class that abstracts database operations while maintaining
security through parameterized queries and input sanitization. It supports multiple
database engines and flexible configuration methods.

Key Features:
    * Multi-database support (SQLite, PostgreSQL, MySQL, Oracle, etc.)
    * Flexible configuration (connection strings, JSON, environment files)
    * SQL injection protection through parameterized queries
    * Simplified CRUD operations with advanced query features
    * Automatic transaction management with context manager support
    * Bulk data operations with pandas DataFrame integration
    * Stored procedure support with input/output parameters
    * Connection management and health monitoring
    * Method result caching for improved performance

Security Features:
    * Identifier sanitization for table/column names
    * Parameterized queries to prevent SQL injection
    * Value validation to detect malicious input
    * Security warnings for potentially unsafe operations

Configuration Methods:
    1. Connection strings (SQLAlchemy format)
    2. Environment files (.env)
    3. JSON configuration files
    4. Dictionary configuration objects

Basic Usage:
    ```python
    from mihirkit.db import Database

    # Using connection string
    db = Database("postgresql://user:password@localhost/mydb")

    # Using environment file
    db = Database("config/database.env")

    # Context manager for automatic cleanup
    with Database("sqlite:///app.db") as db:
        # Select operations
        users = db.select("users", where={"active": True})

        # Insert operations
        db.insert("users", data={"name": "John", "email": "john@example.com"})

        # Update operations
        db.update("users", values={"status": "verified"}, where={"id": 1})

        # Delete operations
        db.delete("users", where={"active": False})
    ```

Advanced Query Features:
    * WHERE clauses with multiple conditions
    * ORDER BY with multiple columns and directions
    * GROUP BY with HAVING clauses
    * LIMIT and OFFSET for pagination
    * Parameterized raw SQL execution
    * Stored procedure calls with I/O parameters

Transaction Management:
    * Automatic commit on successful context exit
    * Automatic rollback on exceptions
    * Manual commit/rollback methods
    * Connection status monitoring

Bulk Operations:
    * Insert from pandas DataFrames
    * Insert from dictionaries and lists
    * Efficient batch processing
    * Memory-optimized operations

Error Handling:
    The module provides comprehensive error handling for:
    * Invalid configurations and connection strings
    * Database connection failures
    * SQL execution errors
    * Parameter validation errors
    * Transaction rollback scenarios

Performance Considerations:
    * LRU caching for SELECT operations
    * Connection pooling through SQLAlchemy
    * Bulk insert optimizations
    * Memory-efficient query processing

Dependencies:
    * sqlalchemy: Core database abstraction layer
    * pandas: DataFrame integration for bulk operations
    * python-dotenv: Environment file parsing
    * Standard library: json, re, warnings, functools

Example Configuration Files:

    Environment file (.env):
    ```
    ENGINE=postgresql
    HOSTNAME=localhost
    PORT=5432
    NAME=mydb
    USERNAME=user
    PASSWORD=password
    ```

    JSON configuration:
    ```json
    {
        "ENGINE": "mysql",
        "HOSTNAME": "localhost",
        "PORT": 3306,
        "NAME": "mydb",
        "USERNAME": "user",
        "PASSWORD": "password"
    }
    ```

Classes:
    Database: Main database interface class providing all functionality

Note:
    This module prioritizes security and ease of use. All database operations
    use parameterized queries to prevent SQL injection attacks. For maximum
    security, avoid using the raw SQL execution methods in production unless
    absolutely necessary.
"""

import json
import re
import warnings
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
import sqlalchemy
from dotenv import dotenv_values
from sqlalchemy import Connection as SQLAlchemyConnection
from sqlalchemy import Engine as SQLAlchemyEngine
from sqlalchemy.orm import Session as SQLAlchemySession
from sqlalchemy.orm import sessionmaker as SQLAlchemySessionMaker


class QueryCache:
    """
    A simple caching system for database query results.

    The cache maps query arguments (converted to hashable form) to query results.
    This is designed to be used manually within database methods rather than as a decorator.

    Features:
        - Converts unhashable arguments (dicts, lists) to hashable tuples for cache keys
        - Configurable cache size with automatic LRU eviction
        - Manual cache management (get, put, clear, stats)
        - Simple argument → result mapping

    Example:
        ```python
        class Database:
            def __init__(self):
                self.cache = QueryCache(maxsize=128)

            def select(self, table, where=None, order_by=None):
                # Check cache first
                cache_key = self.cache.create_key(table=table, where=where, order_by=order_by)
                cached_result = self.cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

                # Execute query
                result = self._execute_query(...)

                # Cache the result
                self.cache.put(cache_key, result)
                return result
        ```
    """

    def __init__(self, maxsize: int = 128):
        """
        Initialize the QueryCache.

        Args:
            maxsize: Maximum number of cached query results
        """
        self.maxsize = maxsize
        self._cache: Dict[Tuple, Any] = {}
        self._access_order: List[Tuple] = []  # Track access order for LRU eviction

    def _make_hashable(self, obj: Any) -> Any:
        """
        Convert unhashable objects to hashable equivalents for caching.

        Args:
            obj: Object to convert (dict, list, or any other type)

        Returns:
            Hashable equivalent of the input object
        """
        if obj is None:
            return None
        elif isinstance(obj, dict):
            # Convert dict to sorted tuple of (key, value) pairs
            return tuple(sorted((k, self._make_hashable(v)) for k, v in obj.items()))
        elif isinstance(obj, list):
            # Convert list to tuple, recursively making each element hashable
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, tuple):
            # Tuple is already hashable, but items might not be
            return tuple(self._make_hashable(item) for item in obj)
        elif isinstance(obj, set):
            # Convert set to sorted tuple
            return tuple(sorted(self._make_hashable(item) for item in obj))
        else:
            # For primitive types (int, str, bool, etc.) and other hashable types
            try:
                # Test if the object is hashable
                hash(obj)
                return obj
            except TypeError:
                # If not hashable, convert to string representation
                return str(obj)

    def create_key(self, **kwargs) -> Tuple:
        """
        Create a hashable cache key from keyword arguments.

        Args:
            **kwargs: The arguments to create a cache key from

        Returns:
            Hashable tuple representing the unique cache key

        Example:
            ```python
            cache = QueryCache()
            key = cache.create_key(table="users", where={"active": True}, limit=10)
            ```
        """
        # Convert kwargs to hashable form, maintaining sorted order for consistency
        hashable_kwargs = tuple(
            sorted((k, self._make_hashable(v)) for k, v in kwargs.items())
        )
        return hashable_kwargs

    def get(self, cache_key: Tuple) -> Optional[Any]:
        """
        Get a cached result by cache key.

        Args:
            cache_key: The cache key (typically created by create_key())

        Returns:
            Cached result if found, None otherwise

        Example:
            ```python
            cache = QueryCache()
            key = cache.create_key(table="users", where={"id": 1})
            result = cache.get(key)  # Returns None if not cached
            ```
        """
        if cache_key in self._cache:
            # Update access order for LRU
            self._update_access_order(cache_key)
            return self._cache[cache_key]
        return None

    def put(self, cache_key: Tuple, result: Any) -> None:
        """
        Store a result in the cache.

        Args:
            cache_key: The cache key (typically created by create_key())
            result: The result to cache

        Example:
            ```python
            cache = QueryCache()
            key = cache.create_key(table="users", where={"id": 1})
            cache.put(key, [{"id": 1, "name": "John"}])
            ```
        """
        self._cache[cache_key] = result
        self._update_access_order(cache_key)

        # Manage cache size with LRU eviction
        self._evict_lru_entries()

    def _update_access_order(self, key: Tuple) -> None:
        """Update the access order for LRU eviction."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def _evict_lru_entries(self) -> None:
        """Remove least recently used entries when cache exceeds maxsize."""
        while len(self._cache) > self.maxsize:
            # Remove the least recently used entry
            lru_key = self._access_order.pop(0)
            if lru_key in self._cache:
                del self._cache[lru_key]

    def contains(self, cache_key: Tuple) -> bool:
        """
        Check if a cache key exists in the cache.

        Args:
            cache_key: The cache key to check

        Returns:
            True if the key exists, False otherwise
        """
        return cache_key in self._cache

    def clear(self) -> None:
        """
        Clear all cached results.

        Example:
            ```python
            cache = QueryCache()
            cache.clear()  # Remove all cached results
            ```
        """
        self._cache.clear()
        self._access_order.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about cached queries.

        Returns:
            Dictionary containing cache statistics

        Example:
            ```python
            cache = QueryCache()
            stats = cache.get_stats()
            print(f"Cache contains {stats['size']} queries")
            print(f"Cache utilization: {stats['utilization']:.1%}")
            ```
        """
        return {
            "size": len(self._cache),
            "maxsize": self.maxsize,
            "utilization": len(self._cache) / self.maxsize if self.maxsize > 0 else 0,
            "hit_ratio": getattr(self, "_hits", 0)
            / max(getattr(self, "_attempts", 1), 1),
            "sample_keys": list(self._cache.keys())[:3],  # First 3 keys for debugging
        }


class Database:
    """
    High-level database interface providing secure CRUD operations and configuration management.

    This class abstracts database operations using SQLAlchemy while maintaining security through
    parameterized queries and input sanitization. Supports multiple database engines and flexible
    configuration methods including connection strings, environment files, and JSON files.

    Attributes:
        config (Dict[str, Any]): Database configuration dictionary
        engine (SQLAlchemyEngine): SQLAlchemy database engine
        session_maker (SQLAlchemySessionMaker): SQLAlchemy session factory
        session (SQLAlchemySession): Active database session
        connection (SQLAlchemyConnection): Active database connection

    Example:
        ```python
        # Using connection string
        db = Database("postgresql://user:password@localhost/mydb")

        # Using context manager for automatic cleanup
        with Database("sqlite:///app.db") as db:
            users = db.select("users", where={"active": True})
            db.insert("users", data={"name": "John", "email": "john@example.com"})
        ```
    """

    config: Dict[str, Any]
    engine: SQLAlchemyEngine
    session_maker: SQLAlchemySessionMaker
    session: SQLAlchemySession
    connection: SQLAlchemyConnection
    cache: QueryCache

    def __init__(
        self,
        config: str | Dict[str, Any] = "sqlite+pysqlite:///:memory:",
        cache_size: int = 128,
    ):
        """
        Initialize database connection with flexible configuration support.

        Automatically detects configuration type and establishes database connection.
        Supports connection strings, file paths (.env, .json), and dictionary configs.

        Args:
            config: Database configuration as connection string, file path, or dictionary.
                   Defaults to in-memory SQLite database.

        Raises:
            ValueError: If configuration format is invalid or cannot be parsed
            SQLAlchemyError: If database connection fails

        Example:
            ```python
            # Connection string
            db = Database("postgresql://user:pass@localhost/mydb")

            # Environment file
            db = Database("config/database.env")

            # Dictionary config
            db = Database({
                "ENGINE": "sqlite",
                "NAME": "app.db"
            })
            ```
        """
        self.config = self._load_config(config)

        self.engine = sqlalchemy.create_engine(self.connection_string)
        self.session_maker = sqlalchemy.orm.sessionmaker(bind=self.engine)
        self.session = self.session_maker()
        self.connection = self.session.connection()
        self.cache = QueryCache(maxsize=cache_size)

    @property
    def cursor(self) -> Any:
        """
        Context manager that yields a raw database cursor for direct database operations.

        Automatically creates and closes the cursor to ensure proper resource management.
        Use for operations that require direct cursor access beyond standard CRUD operations.

        Yields:
            Database cursor object for direct SQL operations

        Example:
            ```python
            with db.cursor as cursor:
                cursor.execute("SHOW TABLES")
                tables = cursor.fetchall()
            ```
        """
        self._cursor = self.connection.connection.cursor()
        yield self._cursor
        self._cursor.close()

    @property
    def connection_string(self) -> str:
        """
        Generate SQLAlchemy-compatible connection string from current configuration.

        Constructs connection string in the format:
        dialect+driver://username:password@hostname:port/dbname?param=value

        Returns:
            str: Complete SQLAlchemy connection string

        Example:
            ```python
            db = Database({
                "ENGINE": "postgresql",
                "HOSTNAME": "localhost",
                "PORT": 5432,
                "NAME": "mydb",
                "USERNAME": "user",
                "PASSWORD": "password"
            })
            print(db.connection_string)
            # Output: postgresql://user:password@localhost:5432/mydb
            ```
        """
        if isinstance(self.config, str):
            return self.config

        # Dialect and optional driver
        engine = self.config["ENGINE"]
        driver = self.config.get("DRIVER")
        dialect_driver = f"{engine}+{driver}" if driver else engine

        # Authentication
        username = self.config.get("USERNAME")
        password = self.config.get("PASSWORD")
        auth = ""
        if username:
            auth = username
            if password:
                auth += f":{password}"
            auth += "@"

        # Host and optional port
        host = self.config["HOSTNAME"]
        port = self.config.get("PORT")
        host_port = f"{host}:{port}" if port else host

        # Database name
        dbname = self.config["NAME"]

        # Query parameters
        query_params = self.config.get("QUERY_PARAMS")
        query_string = ""
        if query_params:
            param_pairs = [f"{k}={v}" for k, v in query_params.items()]
            query_string = "?" + "&".join(param_pairs)

        return f"{dialect_driver}://{auth}{host_port}/{dbname}{query_string}"

    def _load_config(self, config: str | Dict[str, Any]) -> Dict[str, Any]:
        """
        Load and parse database configuration from various sources.

        Automatically detects configuration type and delegates to appropriate parser.
        Supports dictionaries, connection strings, .env files, and JSON files.

        Args:
            config: Configuration source - dictionary, connection string, or file path

        Returns:
            Dict[str, Any]: Parsed configuration dictionary with database parameters

        Raises:
            ValueError: If configuration format is invalid or unsupported
        """
        # If config is already a dictionary, use it directly
        if isinstance(config, dict):
            return config

        # If config is a string, determine the type if not provided
        if isinstance(config, str):
            # Load configuration based on type
            match config:
                case _ if config.endswith(".env"):
                    return self._load_from_env(config)
                case _ if config.endswith(".json"):
                    return self._load_from_json(config)
                case _ if "://" in config:
                    return self._parse_connection_string(config)

        raise ValueError(
            "Invalid configuration format. Must be a dictionary, file path, or connection string."
        )

    def _load_from_env(self, env_path: str) -> Dict[str, Any]:
        """
        Load database configuration from .env file.

        Parses environment file and extracts database configuration variables.
        Filters out None values and returns clean configuration dictionary.

        Args:
            env_path: Path to .env file containing database configuration

        Returns:
            Dict[str, Any]: Configuration dictionary with database parameters

        Raises:
            FileNotFoundError: If .env file doesn't exist
            PermissionError: If .env file cannot be read
        """
        # Load environment variables from .env file as a dictionary
        config = dotenv_values(env_path)

        # Remove None values
        return {k: v for k, v in config.items() if v is not None}

    def _load_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Load database configuration from JSON file.

        Reads and parses JSON configuration file containing database parameters.

        Args:
            json_path: Path to JSON file with database configuration

        Returns:
            Dict[str, Any]: Parsed configuration dictionary

        Raises:
            FileNotFoundError: If JSON file doesn't exist
            json.JSONDecodeError: If JSON file contains invalid JSON
            PermissionError: If JSON file cannot be read
        """
        with open(json_path) as f:
            return json.load(f)

    def _parse_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """
        Parse SQLAlchemy connection string into configuration dictionary.

        Extracts database parameters from connection string format:
        dialect+driver://username:password@host:port/database?param=value

        Args:
            connection_string: SQLAlchemy-format database connection string

        Returns:
            Dict[str, Any]: Configuration dictionary with extracted parameters

        Raises:
            ValueError: If connection string format is invalid

        Example:
            ```python
            config = db._parse_connection_string(
                "postgresql://user:pass@localhost:5432/mydb?sslmode=require"
            )
            # Returns: {
            #     "ENGINE": "postgresql",
            #     "USERNAME": "user",
            #     "PASSWORD": "pass",
            #     "HOSTNAME": "localhost",
            #     "PORT": "5432",
            #     "NAME": "mydb",
            #     "QUERY_PARAMS": {"sslmode": "require"}
            # }
            ```
        """
        # Basic validation
        if "://" not in connection_string:
            raise ValueError(
                "Invalid connection string format. Expected SQLAlchemy format: dialect+driver://username:password@host:port/database"
            )

        # Split connection string into dialect+driver and connection details
        dialect_part, connection_part = connection_string.split("://", 1)

        # Handle dialect and driver
        if "+" in dialect_part:
            dialect, driver = dialect_part.split("+", 1)
        else:
            dialect, driver = dialect_part, None

        # Start building configuration
        config: dict[str, Any] = {
            "ENGINE": dialect,
            "DRIVER": driver,
            "CONNECTION_STRING": connection_string,  # Store the full connection string
        }

        # Parse username, password, host, port, and database if available
        # Username and password
        if "@" in connection_part:
            auth_part, host_part = connection_part.split("@", 1)

            if ":" in auth_part:
                username, password = auth_part.split(":", 1)
                config["USERNAME"] = username
                config["PASSWORD"] = password
            else:
                config["USERNAME"] = auth_part
        else:
            host_part = connection_part

        # Host, port, and database
        if "/" in host_part:
            host_and_port, database_part = host_part.split("/", 1)

            # Handle query parameters if present
            if "?" in database_part:
                database, query_params = database_part.split("?", 1)
                config["NAME"] = database

                # Parse query parameters (for additional connection options)
                params: dict[str, Any] = {}
                for param in query_params.split("&"):
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key] = value

                config["QUERY_PARAMS"] = params
            else:
                config["NAME"] = database_part

            # Parse host and port
            if ":" in host_and_port:
                hostname, port = host_and_port.split(":", 1)
                config["HOSTNAME"] = hostname
                config["PORT"] = port
            else:
                config["HOSTNAME"] = host_and_port
        else:
            # Just host and possibly port
            if ":" in host_part:
                hostname, port = host_part.split(":", 1)
                config["HOSTNAME"] = hostname
                config["PORT"] = port
            else:
                config["HOSTNAME"] = host_part

        return config

    def __enter__(self):
        """
        Enter context manager for automatic transaction and connection management.

        Returns:
            Database: Self reference for use in with statement
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Exit context manager with automatic transaction handling and cleanup.

        Automatically commits transaction on successful exit or rolls back on exception.
        Always closes the database connection for proper resource cleanup.

        Args:
            exc_type: Exception type if exception occurred, None otherwise
            exc_val: Exception value if exception occurred, None otherwise
            exc_tb: Exception traceback if exception occurred, None otherwise

        Returns:
            bool: False to allow exception propagation
        """
        if exc_type is not None:
            # An exception occurred, rollback the transaction
            self.rollback()

        else:
            # No exception, commit the transaction
            self.commit()

        # Always close the connection
        self.close()

        return False  # this allows the original exception to propagate through

    def __del__(self):
        """
        Destructor that ensures proper cleanup of database resources.

        Automatically rolls back any pending transactions and closes connections
        when the Database object is garbage collected.
        """
        self.rollback()
        self.close()

    @property
    def connected(self) -> bool:
        """
        Check if database connection is currently active and available.

        Returns:
            bool: True if connection is active, False otherwise

        Example:
            ```python
            if db.connected:
                result = db.select("users")
            else:
                print("Database connection is closed")
            ```
        """
        # Check if connection object exists
        if not hasattr(self, "connection") or self.connection is None:
            return False

        return not self.connection.closed

    def clear_cache(self) -> None:
        """Clear the query cache for this database instance."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for this database instance."""
        return self.cache.get_stats()

    def _sanitize_identifier(self, identifier: str) -> str:
        """
        Sanitize SQL identifiers (table and column names) to prevent injection attacks.

        Validates that identifiers contain only safe characters (alphanumeric and underscore).
        This prevents SQL injection through malicious table or column names.

        Args:
            identifier: Table or column name to sanitize

        Returns:
            str: Validated identifier if safe

        Raises:
            ValueError: If identifier contains unsafe characters

        Example:
            ```python
            safe_table = db._sanitize_identifier("users")  # ✓ Valid
            # db._sanitize_identifier("users; DROP TABLE users;")  # ✗ Raises ValueError
            ```
        """
        # Check if the identifier is safe using regex
        if not re.match(r"^[a-zA-Z0-9_]+$", identifier):
            raise ValueError(
                f"Invalid identifier: {identifier}. Only alphanumeric and underscore characters allowed."
            )
        return identifier

    def _sanitize_value(self, value: Any) -> Any:
        """
        Sanitize input values to detect potentially malicious SQL injection attempts.

        Performs basic validation to detect obvious SQL injection patterns.
        Note: Primary protection comes from parameterized queries, this is additional validation.

        Args:
            value: Input value to validate

        Returns:
            Any: Original value if validation passes

        Raises:
            ValueError: If value contains potentially malicious SQL patterns
        """
        # For SQLAlchemy, we rely on parameterized queries
        # This method is mainly for validation
        if isinstance(value, str) and (";" in value or "--" in value or "/*" in value):
            raise ValueError(f"Potentially malicious value detected: {value}")
        return value

    def commit(self):
        """
        Commit the current database transaction.

        Saves all pending changes to the database. For in-memory SQLite databases,
        this operation is skipped as commits are not meaningful.

        Example:
            ```python
            db.insert("users", data={"name": "John"})
            db.commit()  # Persist changes to database
            ```
        """
        if "sqlite+pysqlite:///:memory:" in self.connection_string:
            return
        if hasattr(self, "session") and self.session is not None and self.connected:
            self.session.commit()

    def rollback(self):
        """
        Rollback the current database transaction.

        Discards all pending changes since the last commit. Used for error recovery
        and transaction management.

        Example:
            ```python
            try:
                db.insert("users", data={"name": "John"})
                # Some error occurs
            except Exception:
                db.rollback()  # Discard changes
            ```
        """
        if hasattr(self, "session") and self.session is not None and self.connected:
            self.session.rollback()

    def close(self):
        """
        Close the database connection and dispose of the engine.

        Properly closes all database resources including connections and engine.
        Should be called when database operations are complete.

        Example:
            ```python
            db = Database("sqlite:///app.db")
            try:
                # Database operations
                pass
            finally:
                db.close()  # Ensure cleanup
            ```
        """
        if hasattr(self, "connection") and self.connection:
            self.connection.close()
        if hasattr(self, "engine") and self.engine:
            self.engine.dispose()

    def create_table(
        self,
        table: str,
        columns: Dict[str, str],
        if_not_exists: bool = True,
        primary_key: Optional[List[str]] = None,
        unique: Optional[List[str]] = None,
        foreign_keys: Optional[List[Dict[str, str]]] = None,
        indexes: Optional[List[List[str]]] = None,
    ) -> str:
        """
        Create a new table in the database with specified columns and constraints.

        Args:
            table: Name of the table to create
            columns: Dictionary mapping column names to SQL data types (e.g., {"id": "INTEGER", "name": "VARCHAR(100)"})
            if_not_exists: If True, adds IF NOT EXISTS to the CREATE TABLE statement
            primary_key: List of column names to use as primary key
            unique: List of column names to enforce UNIQUE constraint
            foreign_keys: List of foreign key definitions, each as dict with keys: column, ref_table, ref_column, on_delete (optional)
            indexes: List of lists, each inner list is columns to index together

        Returns:
            str: Success message indicating table creation

        Raises:
            ValueError: If columns dict is empty or invalid
            SQLAlchemyError: If table creation fails

        Example:
            ```python
            db.create_table(
                "users",
                columns={"id": "INTEGER", "name": "TEXT", "email": "TEXT"},
                primary_key=["id"],
                unique=["email"]
            )
            ```
        """
        if not columns or not isinstance(columns, dict):
            raise ValueError("The `columns` argument must be a non-empty dictionary.")

        # Sanitize table and column names
        table_name = self._sanitize_identifier(table)
        col_defs = []
        for col, col_type in columns.items():
            col_name = self._sanitize_identifier(col)
            col_defs.append(f"{col_name} {col_type}")

        # Primary key
        if primary_key:
            pk_cols = [self._sanitize_identifier(col) for col in primary_key]
            col_defs.append(f"PRIMARY KEY ({', '.join(pk_cols)})")

        # Unique constraint
        if unique:
            uq_cols = [self._sanitize_identifier(col) for col in unique]
            col_defs.append(f"UNIQUE ({', '.join(uq_cols)})")

        # Foreign keys
        if foreign_keys:
            for fk in foreign_keys:
                col = self._sanitize_identifier(fk["column"])
                ref_table = self._sanitize_identifier(fk["ref_table"])
                ref_col = self._sanitize_identifier(fk["ref_column"])
                fk_clause = f"FOREIGN KEY ({col}) REFERENCES {ref_table}({ref_col})"
                if "on_delete" in fk:
                    fk_clause += f" ON DELETE {fk['on_delete'].upper()}"
                col_defs.append(fk_clause)

        # Compose CREATE TABLE statement
        if_not_exists_sql = "IF NOT EXISTS " if if_not_exists else ""
        create_stmt = f"CREATE TABLE {if_not_exists_sql}{table_name} (\n  {',\n  '.join(col_defs)}\n)"

        # Execute CREATE TABLE
        self.connection.execute(sqlalchemy.text(create_stmt))

        # Create indexes if specified
        if indexes:
            for idx_cols in indexes:
                idx_cols_sanitized = [
                    self._sanitize_identifier(col) for col in idx_cols
                ]
                idx_name = f"idx_{table_name}_{'_'.join(idx_cols_sanitized)}"
                idx_stmt = f"CREATE INDEX IF NOT EXISTS {idx_name} ON {table_name} ({', '.join(idx_cols_sanitized)})"
                self.connection.execute(sqlalchemy.text(idx_stmt))

        self.connection.commit()
        return f"Table `{table_name}` created successfully."

    def select(
        self,
        table: str,
        columns: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
        order_by: Optional[str | List[Dict[str, str]]] = None,
        group_by: Optional[str | List[str]] = None,
        having: Optional[Dict[str, Any]] = None,
        limit: Optional[int] = None,
        offset: Optional[int] = None,
        ignore_cache: bool = False,
    ) -> List[Dict]:
        """
        Execute SELECT query with manual caching support.

        Args:
            table: Target table name for the query
            columns: List of column names to select (None for all columns)
            where: Dictionary of column-value pairs for WHERE clause conditions
            order_by: Sorting specification - string for single column or list of dicts
            group_by: Column name(s) for GROUP BY clause (string or list)
            having: Dictionary of column-value pairs for HAVING clause (requires group_by)
            limit: Maximum number of rows to return
            offset: Number of rows to skip

        Returns:
            List[Dict]: Query results as list of dictionaries with column names as keys
        """
        # Create cache key from all arguments
        cache_key = self.cache.create_key(
            table=table,
            columns=columns,
            where=where,
            order_by=order_by,
            group_by=group_by,
            having=having,
            limit=limit,
            offset=offset,
        )

        # Check if result is already cached
        cached_result = self.cache.get(cache_key)
        if cached_result is not None and not ignore_cache:
            return cached_result

        # Execute the query (your existing implementation)
        # Validate HAVING clause can only be used with GROUP BY
        if having and not group_by:
            raise ValueError("HAVING clause can only be used with GROUP BY")

        # Validate LIMIT and OFFSET
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError(f"LIMIT must be a non-negative integer, got {limit}")

        if offset is not None:
            if not isinstance(offset, int) or offset < 0:
                raise ValueError(f"OFFSET must be a non-negative integer, got {offset}")
            if limit is None and "mysql" in self.config.get("DRIVER", "").lower():
                raise ValueError("OFFSET can only be used with LIMIT in MySQL")

        # Sanitize table name
        table = self._sanitize_identifier(table)

        # Build columns
        column_elements = [sqlalchemy.text("*")]
        if columns is not None and isinstance(columns, (list, tuple, set)):
            sanitized_columns = [self._sanitize_identifier(col) for col in columns]
            column_elements = [sqlalchemy.text(col) for col in sanitized_columns]

        # Start with the basic select statement
        statement = sqlalchemy.select(*column_elements).select_from(
            sqlalchemy.text(table)
        )

        # Prepare parameters with unique naming to avoid conflicts
        params = {}
        param_count = 0

        # Build the WHERE clause
        if where:
            where_conditions = []
            for col, val in where.items():
                col = self._sanitize_identifier(col)
                self._sanitize_value(val)
                param_name = f"where_{param_count}"
                where_conditions.append(sqlalchemy.text(f"{col} = :{param_name}"))
                params[param_name] = val
                param_count += 1

            if where_conditions:
                combined_condition = sqlalchemy.text(
                    " AND ".join([str(condition) for condition in where_conditions])
                )
                statement = statement.where(combined_condition)

        # Add GROUP BY clause
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]

            sanitized_group_by = [self._sanitize_identifier(col) for col in group_by]
            statement = statement.group_by(
                *[sqlalchemy.text(col) for col in sanitized_group_by]
            )

            # Add HAVING clause
            if having:
                having_conditions = []
                for col, val in having.items():
                    col = self._sanitize_identifier(col)
                    self._sanitize_value(val)
                    param_name = f"having_{param_count}"
                    having_conditions.append(sqlalchemy.text(f"{col} = :{param_name}"))
                    params[param_name] = val
                    param_count += 1

                if having_conditions:
                    combined_condition = sqlalchemy.text(
                        " AND ".join(
                            [str(condition) for condition in having_conditions]
                        )
                    )
                    statement = statement.having(combined_condition)

        # Add ORDER BY clause
        if order_by:
            if isinstance(order_by, str):
                sanitized_col = self._sanitize_identifier(order_by)
                statement = statement.order_by(sqlalchemy.text(sanitized_col))
            else:
                order_clauses = []
                for item in order_by:
                    if not isinstance(item, dict) or "column" not in item:
                        raise ValueError(
                            "Each ORDER BY item must be a dict with 'column' key"
                        )

                    col = self._sanitize_identifier(item["column"])
                    direction = item.get("direction", "ASC").upper()
                    if direction not in ["ASC", "DESC"]:
                        raise ValueError(
                            f"ORDER BY direction must be 'ASC' or 'DESC', got '{direction}'"
                        )

                    order_clauses.append(sqlalchemy.text(f"{col} {direction}"))

                statement = statement.order_by(*order_clauses)

        # Add LIMIT and OFFSET clauses
        if limit is not None:
            statement = statement.limit(limit)
        if offset is not None:
            statement = statement.offset(offset)

        # Execute query with parameters
        result = self.connection.execute(statement, params)

        # Convert results to list of dictionaries
        query_results = [dict(row._mapping) for row in result]

        # Cache the results
        self.cache.put(cache_key, query_results)

        return query_results

    def insert(
        self,
        table: str,
        data: Optional[list | dict | pd.DataFrame] = None,
        columns: Optional[List[str]] = None,
        values: Optional[List[List[Any]]] = None,
        **kwargs: Any,
    ) -> str:
        """
        Insert data into database table using traditional or bulk methods.

        Supports two insertion modes: traditional (columns + values) and bulk (data parameter).
        Automatically chooses appropriate method based on provided parameters.

        Args:
            table: Target table name for insertion
            columns: List of column names (for traditional mode)
            values: List of value lists to insert (for traditional mode)
            data: Data to insert as dictionary, list of dicts, or DataFrame (for bulk mode)

        Returns:
            str: Success message with number of rows inserted

        Raises:
            ValueError: If neither data nor columns+values are provided, or if data is empty
            SQLAlchemyError: If insertion fails

        Example:
            ```python
            # Traditional insertion
            db.insert(
                "users",
                columns=["name", "email"],
                values=[["John", "john@example.com"], ["Jane", "jane@example.com"]]
            )

            # Bulk insertion from dictionary
            db.insert("users", data={"name": "John", "email": "john@example.com"})

            # Bulk insertion from DataFrame
            import pandas as pd
            df = pd.DataFrame({
                "name": ["Alice", "Bob"],
                "email": ["alice@example.com", "bob@example.com"]
            })
            db.insert("users", data=df)
            ```
        """
        if data is not None:
            return self._bulk_insert(table, data, **kwargs)
        elif columns and values:
            return self._single_insert(table, columns, values, **kwargs)
        else:
            raise ValueError(
                "Either the `data` argument or the `columns` and `values` arguments must be provdided."
            )

    def _single_insert(
        self, table: str, columns: List[str], values: List[List[Any]], **kwargs: Any
    ) -> str:
        """
        Insert rows using traditional column-value list method.

        Internal method for handling traditional INSERT operations with explicit
        column and value specifications. Uses parameterized queries for security.

        Args:
            table: Target table name
            columns: List of column names
            values: List of value lists, each containing values for one row

        Returns:
            str: Success message with number of rows inserted

        Raises:
            ValueError: If value list length doesn't match column list length
            SQLAlchemyError: If insertion fails
        """
        # Sanitize table and column names
        table_name = self._sanitize_identifier(table)
        sanitized_columns = [self._sanitize_identifier(col) for col in columns]

        # Validate values
        for row in values:
            if len(row) != len(columns):
                raise ValueError(
                    f"Value list length ({len(row)}) doesn't match column list length ({len(columns)})"
                )
            for val in row:
                self._sanitize_value(val)  # Validation

        # Build base insert statement using SQLAlchemy query builder
        table_ref = sqlalchemy.text(table_name)
        insert_stmt = sqlalchemy.insert(table_ref)

        # Execute for each row of values - build parameter dictionary
        row_count = 0
        for row in values:
            # Create parameter dictionary with column names as keys
            row_params = {sanitized_columns[i]: val for i, val in enumerate(row)}

            # Execute insert with parameters
            result = self.connection.execute(insert_stmt, row_params)
            row_count += result.rowcount

        # Commit changes
        self.connection.commit()
        return f"Query Result: {row_count} rows inserted."

    def _bulk_insert(
        self,
        table: str,
        data: List[Dict[str, Any]] | Dict[str, list[Any]] | pd.DataFrame,
        if_exists: Literal["fail", "replace", "append"] = "fail",
        index: bool = False,
        **kwargs: Any,
    ) -> str:
        """
        Insert data using pandas DataFrame bulk operations for improved performance.

        Internal method for handling bulk INSERT operations using pandas to_sql method.
        Efficiently handles large datasets and provides better performance than row-by-row insertion.

        Args:
            table: Target table name
            data: Data to insert as dictionary, list of dictionaries, or DataFrame
            **kwargs: Additional arguments passed to pandas to_sql method

        Returns:
            str: Success message with number of rows inserted

        Raises:
            ValueError: If data is None or empty DataFrame
            AssertionError: If data is not a supported type
            SQLAlchemyError: If insertion fails

        Example:
            ```python
            # Insert DataFrame with additional pandas options
            db._bulk_insert("users", df, if_exists="append", index=False)
            ```
        """
        if data is None or (isinstance(data, pd.DataFrame) and data.empty):
            raise ValueError(
                "The pandas DataFrame in the `data` argument cannot be empty."
            )

        if not isinstance(data, (dict, list, pd.DataFrame)):
            raise TypeError(
                f"The `data` argument must be either a dictionary, a list of dictionaries, or a pandas DataFrame. Found {type(data)} instead."
            )

        match data:
            case pd.DataFrame():
                pass
            case dict():
                data = pd.DataFrame([data])  # Convert single dict to DataFrame
            case list():
                if not all(isinstance(row, dict) for row in data):
                    raise TypeError(
                        "All items in the list must be dictionaries. Found non-dictionary item."
                    )
                data = pd.DataFrame(data)  # Convert list of dicts to DataFrame

        data.to_sql(
            name=table, con=self.engine, if_exists=if_exists, index=index, **kwargs
        )

        return f"Query Result: {len(data)} rows inserted."

    def update(self, table: str, values: Dict[str, Any], where: Dict[str, Any]) -> str:
        """
        Update existing rows in database table with new values.

        Modifies rows matching the WHERE condition with new values. Uses parameterized
        queries for security and requires explicit WHERE clause to prevent accidental
        mass updates.

        Args:
            table: Target table name for update operation
            values: Dictionary of column-value pairs to update
            where: Dictionary of column-value pairs for WHERE clause conditions

        Returns:
            str: Success message with number of rows updated

        Raises:
            ValueError: If no values provided or no WHERE condition specified
            SQLAlchemyError: If update operation fails

        Example:
            ```python
            # Update single user
            db.update(
                "users",
                values={"status": "verified", "last_login": "2024-01-15"},
                where={"id": 123}
            )

            # Update multiple users with complex condition
            db.update(
                "users",
                values={"status": "inactive"},
                where={"last_login": "2023-01-01", "role": "guest"}
            )
            ```
        """
        # Sanitize table name
        table_name = self._sanitize_identifier(table)

        if not values:
            raise ValueError("No values provided to update")

        if not where:
            raise ValueError(
                "No WHERE condition provided. To update all rows, use where={'1': 1}"
            )

        # Validate and sanitize values
        sanitized_values = {}
        for col, val in values.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            sanitized_values[col] = val

        # Validate and sanitize where conditions
        sanitized_where = {}
        for col, val in where.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            sanitized_where[col] = val

        # Build update statement using SQLAlchemy query builder
        table_ref = sqlalchemy.text(table_name)
        update_stmt = sqlalchemy.update(table_ref)

        # Add SET clause using .values()
        update_stmt = update_stmt.values(**sanitized_values)

        # Build WHERE clause using SQLAlchemy text conditions
        where_conditions = []
        params = {}
        param_count = 0

        for col, val in sanitized_where.items():
            param_name = f"where_{param_count}"
            where_conditions.append(sqlalchemy.text(f"{col} = :{param_name}"))
            params[param_name] = val
            param_count += 1

        if where_conditions:
            # Combine conditions with AND
            combined_condition = sqlalchemy.text(
                " AND ".join([str(condition) for condition in where_conditions])
            )
            update_stmt = update_stmt.where(combined_condition)

        # Execute update
        result = self.connection.execute(update_stmt, params)
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows modified."

    def delete(self, table: str, where: Optional[Dict[str, Any]] = None) -> str:
        """
        Delete rows from database table based on WHERE conditions.

        Removes rows matching the WHERE clause conditions. Issues warning if no WHERE
        clause is provided to prevent accidental deletion of all data.

        Args:
            table: Target table name for deletion
            where: Dictionary of column-value pairs for WHERE clause conditions.
                If None, deletes all rows (with warning)

        Returns:
            str: Success message with number of rows deleted

        Raises:
            ValueError: If WHERE conditions are invalid
            SQLAlchemyError: If deletion fails

        Example:
            ```python
            # Delete specific user
            db.delete("users", where={"id": 123})

            # Delete inactive users
            db.delete("users", where={"status": "inactive", "last_login": "2023-01-01"})

            # Delete all users (issues warning)
            db.delete("users")  # Use rollback() if unintended
            ```
        """
        # Sanitize table name
        table_name = self._sanitize_identifier(table)

        if where is None:
            warnings.warn(
                message="Got `None` as the value of the `where` argument. This will result in all the data from the table being deleted. Please roll back the transaction using the `rollback()` method to undo this deletion.",
                stacklevel=2,
            )
            where = {"1": 1}

        # Validate and sanitize where conditions
        sanitized_where = {}
        for col, val in where.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            sanitized_where[col] = val

        if not sanitized_where:
            raise ValueError(
                "No WHERE condition provided. To delete all rows, use where={'1': 1}"
            )

        # Build delete statement using SQLAlchemy query builder
        table_ref = sqlalchemy.text(table_name)
        delete_stmt = sqlalchemy.delete(table_ref)

        # Build WHERE clause using SQLAlchemy text conditions
        where_conditions = []
        params = {}
        param_count = 0

        for col, val in sanitized_where.items():
            # Handle special case for delete all rows
            if col == "1" and val == 1:
                where_conditions.append(sqlalchemy.text("1 = 1"))
            else:
                param_name = f"where_{param_count}"
                where_conditions.append(sqlalchemy.text(f"{col} = :{param_name}"))
                params[param_name] = val
                param_count += 1

        if where_conditions:
            # Combine conditions with AND
            combined_condition = sqlalchemy.text(
                " AND ".join([str(condition) for condition in where_conditions])
            )
            delete_stmt = delete_stmt.where(combined_condition)

        # Execute delete
        result = self.connection.execute(delete_stmt, params)
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows deleted."

    def execute(self, query: str, params: Optional[Dict[str, Any]] = None) -> Any:
        """
        Execute raw SQL query with optional parameters. Use with caution for security.

        Provides direct SQL execution capability for complex queries not covered by
        standard CRUD methods. Issues security warning as this bypasses some safety checks.

        Args:
            query: Raw SQL query string with named parameter placeholders (:param)
            params: Dictionary of parameter values for named placeholders

        Returns:
            List[Dict] for SELECT queries, or success message for other operations

        Raises:
            SQLAlchemyError: If query execution fails

        Warning:
            This method bypasses identifier sanitization and should be used carefully.
            Always use parameterized queries to prevent SQL injection.

        Example:
            ```python
            # Complex SELECT query
            results = db.execute('''
                SELECT u.name, COUNT(o.id) as order_count
                FROM users u
                LEFT JOIN orders o ON u.id = o.user_id
                WHERE u.created_at > :start_date
                GROUP BY u.id, u.name
                HAVING COUNT(o.id) > :min_orders
            ''', {
                "start_date": "2024-01-01",
                "min_orders": 5
            })

            # Custom DDL operation
            db.execute("CREATE INDEX idx_user_email ON users(email)")
            ```
        """
        warnings.warn(
            message="Use caution with the raw SQL execution method. It does not validate the SQL statement and can lead to SQL injection vulnerabilities if the user input is not properly sanitized.",
            stacklevel=2,
        )

        result = self.connection.execute(sqlalchemy.text(query), params or {})

        # For SELECT queries, return results
        if query.strip().upper().startswith("SELECT"):
            return [dict(row._mapping) for row in result]

        # For other queries, commit and return affected row count
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows modified."

    def callproc(
        self,
        procedure: str,
        in_params: Dict[str, Any] | List[Any],
        out_params: Optional[Dict[str, type] | List[type]] = None,
    ) -> Dict[str, Any]:
        """
        Execute stored procedure with input and output parameters.

        Calls database stored procedures and handles both input and output parameters.
        Automatically manages parameter binding and result extraction.

        Args:
            procedure: Name of the stored procedure to execute
            in_params: Input parameters as dictionary (named) or list (positional)
            out_params: Output parameter specifications as dictionary with names and types
                       or list of types for positional parameters

        Returns:
            Dict[str, Any]: Dictionary containing output parameter values.
                           Empty dict if no output parameters specified.

        Raises:
            SQLAlchemyError: If procedure execution fails

        Example:
            ```python
            # Procedure with input parameters only
            db.callproc("update_user_status", {"user_id": 123, "status": "active"})

            # Procedure with input and output parameters
            result = db.callproc(
                "calculate_user_stats",
                in_params={"user_id": 123},
                out_params={"total_orders": int, "total_spent": float}
            )
            print(f"Orders: {result['total_orders']}, Spent: ${result['total_spent']}")

            # Positional parameters
            result = db.callproc(
                "get_summary",
                in_params=[123, "2024-01-01"],
                out_params=[int, str]  # Returns out_param1, out_param2
            )
            ```
        """
        # Get the raw connection
        raw_conn = self.connection.connection

        # Create a cursor
        cursor = raw_conn.cursor()

        try:
            out_params = out_params if out_params is not None else []

            # Build the list of input params
            if isinstance(in_params, dict):
                in_params = list(in_params.values())
            params = in_params.copy()

            # Add output params to the params dict
            out_param_names = [
                f"out_param{counter}" for counter in range(1, len(out_params) + 1)
            ]
            if out_params != []:
                if isinstance(out_params, dict):
                    out_param_names = list(out_params.keys())
                    out_params = list(out_params.values())

                for param_type in out_params:
                    params.append(cursor.var(param_type))

            # Call the procedure
            cursor.callproc(procedure, params)

            raw_conn.commit()

            if out_params == []:
                return {}

            # Get the values of the output parameters
            out_param_values = [
                item.getvalue() for item in params if item not in in_params
            ]

            return dict(zip(out_param_names, out_param_values))

        except Exception:
            raw_conn.rollback()
            raise
        finally:
            cursor.close()
