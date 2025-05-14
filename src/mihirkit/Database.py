from typing import Optional, Any, List, Dict, Union
import re
import json
import logging

from sqlalchemy import create_engine, text
from dotenv import dotenv_values

from src.mihirkit.decorators import disabled

class Database:
    def __init__(self, config: Union[str, Dict[str, Any]], config_type: str = None):
        """
        Initialize database connection.
        
        Args:
            config: Configuration source (connection string, file path, or dict)
            config_type: Type of configuration ('env', 'json', 'connection_string', or None for auto-detect)
        """
        self.config = self._load_config(config, config_type)
        self._establish_connection()
        
    def _load_config(self, config: str | Dict[str, Any], config_type: str = None) -> Dict[str, Any]:
        """
        Load configuration from various sources. The config must have the following keys:
            - ENGINE
            - HOSTNAME
            - PORT
            - NAME
            - USERNAME
            - PASSWORD
            - SID
        
        Args:
            config: Configuration source (connection string, file path, or dict)
            config_type: Type of configuration ('env', 'json', 'connection_string', or None for auto-detect)
            
        Returns:
            Dictionary with database configuration
        """
        # If config is already a dictionary, use it directly
        if isinstance(config, dict):
            return config
            
        # If config is a string, determine the type if not provided
        if isinstance(config, str):
            if config_type is None:
                # Auto-detect config type
                if config.endswith('.env'):
                    config_type = 'env'
                elif config.endswith('.json'):
                    config_type = 'json'
                elif '://' in config:
                    config_type = 'connection_string'
                else:
                    raise ValueError("Could not determine configuration type. Please specify config_type.")
            
            # Load configuration based on type
            match config_type:
                case 'env':
                    return self._load_from_env(config)
                case 'json':
                    return self._load_from_json(config)
                case 'connection_string':
                    return self._parse_connection_string(config)
        
        raise ValueError("Invalid configuration format. Must be a dictionary, file path, or connection string.")
    
    def _load_from_env(self, env_path: str) -> Dict[str, Any]:
        """
        Load configuration from .env file.
        
        Args:
            env_path: Path to .env file
            
        Returns:
            Configuration dictionary
        """
        # Load environment variables from .env file as a dictionary
        config = dotenv_values(env_path)
        
        # Remove None values
        return {k: v for k, v in config.items() if v is not None}
    
    def _load_from_json(self, json_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON file.
        
        Args:
            json_path: Path to JSON file
            
        Returns:
            Configuration dictionary
        """
        with open(json_path, 'r') as f:
            return json.load(f)
    
    def _parse_connection_string(self, connection_string: str) -> Dict[str, Any]:
        """
        Parse database connection string into configuration dictionary.
        Supports all SQLAlchemy connection string formats.
        
        Args:
            connection_string: Database connection string (SQLAlchemy format)
            
        Returns:
            Configuration dictionary
        """
        # Basic validation
        if "://" not in connection_string:
            raise ValueError("Invalid connection string format. Expected SQLAlchemy format: dialect+driver://username:password@host:port/database")
        
        # Split connection string into dialect+driver and connection details
        dialect_part, connection_part = connection_string.split("://", 1)
        
        # Handle dialect and driver
        if "+" in dialect_part:
            dialect, driver = dialect_part.split("+", 1)
        else:
            dialect, driver = dialect_part, None
            
        # Start building configuration
        config = {
            "ENGINE": dialect,
            "DRIVER": driver,
            "CONNECTION_STRING": connection_string  # Store the full connection string
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
                params = {}
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
    
    def _establish_connection(self):
        """Establish database connection based on configuration."""
        # If we have a full connection string, use it directly
        if "CONNECTION_STRING" in self.config:
            self.engine = create_engine(self.config["CONNECTION_STRING"])
            self.connection = self.engine.connect()
            self.db_engine = self.config["ENGINE"]
            return
            
        # Define mapping of engines to their default driver and connection string templates
        engine_configs = {
            "mysql": {
                "default_driver": "pymysql",
                "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}",
                "use_db_command": "USE {database}"
            },
            "oracle": {
                "default_driver": "oracledb",
                "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}/{sid}",
                "use_db_command": None  # Oracle doesn't use USE, database is in connection string
            },
            "postgresql": {
                "default_driver": "psycopg2",
                "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}",
                "use_db_command": "SET search_path TO {database}"
            },
            "postgres": {
                "default_driver": "psycopg2",
                "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}",
                "use_db_command": "SET search_path TO {database}"
            },
            "sqlite": {
                "default_driver": None,
                "template": "sqlite:///{path}",
                "use_db_command": None  # SQLite doesn't use USE, database is path
            },
            "mssql": {
                "default_driver": "pyodbc",
                "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}",
                "use_db_command": "USE {database}"
            },
        }
        
        # Get engine and normalize to lowercase
        engine = self.config.get("ENGINE", "").lower()
        
        # Get engine configuration or use generic fallback
        engine_config = engine_configs.get(engine, {
            "default_driver": None,
            "template": "{engine}+{driver}://{username}:{password}@{hostname}:{port}" if self.config.get("DRIVER") else "{engine}://{username}:{password}@{hostname}:{port}",
            "use_db_command": "USE {database}"  # Try standard MySQL syntax as fallback
        })
        
        # Build connection string parameters
        conn_params = {
            "engine": engine,
            "driver": self.config.get("DRIVER") or engine_config["default_driver"],
            "username": self.config.get("USERNAME", ""),
            "password": self.config.get("PASSWORD", ""),
            "hostname": self.config.get("HOSTNAME", ""),
            "port": self.config.get("PORT", ""),
            "path": self.config.get("PATH", ":memory:"),  # For SQLite
            "sid": self.config.get("SID", "")  # For Oracle
        }
        
        # Skip driver part if None
        if conn_params["driver"] is None:
            conn_params["driver"] = ""
            engine_config["template"] = engine_config["template"].replace("+{driver}", "")
            
        # Create connection string using template
        connection_string = engine_config["template"].format(**conn_params)
        
        # Create engine with the constructed connection string
        self.engine = create_engine(connection_string)
        self.connection = self.engine.connect()
        self.db_engine = engine
        
        # Use the specified database if provided and there's a command for it
        if "NAME" in self.config and self.config["NAME"] and engine_config["use_db_command"]:
            try:
                use_command = engine_config["use_db_command"].format(database=self.config["NAME"])
                self.connection.execute(text(use_command))
            except Exception as e:
                # Log the error but don't raise, as the database might be specified in the connection string
                logging.debug(f"Warning: Could not select database {self.config['NAME']}: {str(e)}")

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        self.close()
    
    @property
    def connected(self) -> bool:
        """Check if database connection is active."""
        # Check if connection object exists
        if not hasattr(self, 'connection') or self.connection is None:
            raise ConnectionError("No connection object")
        
        # Try to execute a simple query
        result = self.connection.execute(text("SELECT 1"))
        if result.scalar():
            return True
        else:
            raise ConnectionError("Unable to execute test query") 

    def _sanitize_identifier(self, identifier: str) -> str:
        """Sanitize SQL identifiers (table/column names)."""
        # Check if the identifier is safe using regex
        if not re.match(r'^[a-zA-Z0-9_]+$', identifier):
            raise ValueError(f"Invalid identifier: {identifier}. Only alphanumeric and underscore characters allowed.")
        return identifier
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize values to prevent SQL injection."""
        # For SQLAlchemy, we rely on parameterized queries
        # This method is mainly for validation
        if isinstance(value, str) and (";" in value or "--" in value or "/*" in value):
            raise ValueError(f"Potentially malicious value detected: {value}")
        return value
    
    def commit(self):
        """Commit current transaction."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.commit()
    
    def rollback(self):
        """Rollback current transaction."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.rollback()

    def close(self):
        """Close database connection."""
        if hasattr(self, 'connection') and self.connection:
            self.connection.close()
        if hasattr(self, 'engine') and self.engine:
            self.engine.dispose()

    def select(self, table: str, columns: List[str] = None, where: Dict[str, Any] = None, 
            order_by: Union[str, List[Dict[str, str]]] = None, 
            group_by: Union[str, List[str]] = None,
            having: Dict[str, Any] = None,
            limit: int = None,
            offset: int = None) -> List[Dict]:
        """
        Perform a SELECT query with enhanced functionality.
        
        Args:
            table: Table name
            columns: List of columns to select. None means all columns (*)
            where: Dictionary of column-value pairs for WHERE clause
            order_by: Can be either:
                    - String with column name (defaults to ASC)
                    - List of dicts like [{"column": "name", "direction": "ASC"}, ...]
            group_by: Column(s) to group by. Can be a string or list of strings
            having: Dictionary of column-value pairs for HAVING clause (used with GROUP BY)
            limit: Maximum number of rows to return
            offset: Number of rows to skip before starting to return rows
            
        Returns:
            List of dictionaries representing selected rows
        """
        # Sanitize table and column names
        table = self._sanitize_identifier(table)
        
        # Build column part of query
        if columns is None:
            column_str = "*"
        else:
            sanitized_columns = [self._sanitize_identifier(col) for col in columns]
            column_str = ", ".join(sanitized_columns)
        
        # Build base query
        query = f"SELECT {column_str} FROM {table}"
        
        # Add WHERE clause if specified
        params = {}
        param_count = 0
        
        if where:
            where_conditions = []
            for col, val in where.items():
                col = self._sanitize_identifier(col)
                self._sanitize_value(val)  # Just for validation
                param_name = f"where_{param_count}"
                where_conditions.append(f"{col} = :{param_name}")
                params[param_name] = val
                param_count += 1
            
            query += " WHERE " + " AND ".join(where_conditions)
        
        # Add GROUP BY clause if specified
        if group_by:
            if isinstance(group_by, str):
                group_by = [group_by]
            
            sanitized_group_by = [self._sanitize_identifier(col) for col in group_by]
            query += " GROUP BY " + ", ".join(sanitized_group_by)
            
            # Add HAVING clause if specified (only valid with GROUP BY)
            if having:
                having_conditions = []
                for col, val in having.items():
                    col = self._sanitize_identifier(col)
                    self._sanitize_value(val)  # Just for validation
                    param_name = f"having_{param_count}"
                    having_conditions.append(f"{col} = :{param_name}")
                    params[param_name] = val
                    param_count += 1
                
                query += " HAVING " + " AND ".join(having_conditions)
        elif having:
            raise ValueError("HAVING clause can only be used with GROUP BY")
        
        # Add ORDER BY clause if specified
        if order_by:
            if isinstance(order_by, str):
                # Simple case: just a column name (default to ASC)
                query += f" ORDER BY {self._sanitize_identifier(order_by)}"
            else:
                # List of dicts with column and direction
                order_parts = []
                for item in order_by:
                    if not isinstance(item, dict) or "column" not in item:
                        raise ValueError("Each ORDER BY item must be a dict with 'column' key")
                    
                    col = self._sanitize_identifier(item["column"])
                    
                    # Validate direction if provided
                    direction = item.get("direction", "ASC").upper()
                    if direction not in ["ASC", "DESC"]:
                        raise ValueError(f"ORDER BY direction must be 'ASC' or 'DESC', got '{direction}'")
                    
                    order_parts.append(f"{col} {direction}")
                
                query += " ORDER BY " + ", ".join(order_parts)
        
        # Add LIMIT and OFFSET clauses if specified
        if limit is not None:
            if not isinstance(limit, int) or limit < 0:
                raise ValueError(f"LIMIT must be a non-negative integer, got {limit}")
            query += f" LIMIT {limit}"
            
            # Add OFFSET only if LIMIT is specified
            if offset is not None:
                if not isinstance(offset, int) or offset < 0:
                    raise ValueError(f"OFFSET must be a non-negative integer, got {offset}")
                query += f" OFFSET {offset}"
        elif offset is not None:
            # OFFSET requires LIMIT in MySQL
            raise ValueError("OFFSET can only be used with LIMIT in MySQL")
        
        # Execute query with parameters
        result = self.connection.execute(text(query), params)
        
        # Return results as list of dictionaries
        return [dict(row._mapping) for row in result]

    def insert(self, table: str, columns: List[str], values: List[List[Any]]) -> str:
        """
        Insert rows into a table.
        
        Args:
            table: Table name
            columns: List of column names
            values: List of value lists to insert
            
        Returns:
            String with number of rows inserted
        """
        # Sanitize table and column names
        table = self._sanitize_identifier(table)
        sanitized_columns = [self._sanitize_identifier(col) for col in columns]
        
        # Validate values
        for row in values:
            if len(row) != len(columns):
                raise ValueError(f"Value list length ({len(row)}) doesn't match column list length ({len(columns)})")
            for val in row:
                self._sanitize_value(val)  # Validation
        
        # Build query with named parameters
        column_str = ", ".join(sanitized_columns)
        placeholders = ", ".join([f":{col}_{i}" for i, col in enumerate(columns)])
        query = f"INSERT INTO {table} ({column_str}) VALUES ({placeholders})"
        
        # Execute for each row of values
        row_count = 0
        for row_idx, row in enumerate(values):
            params = {f"{columns[i]}_{i}": val for i, val in enumerate(row)}
            result = self.connection.execute(text(query), params)
            row_count += result.rowcount
        
        # Commit changes
        self.connection.commit()
        return f"Query Result: {row_count} rows inserted."
    
    def update(self, table: str, values: Dict[str, Any], where: Dict[str, Any]) -> str:
        """
        Update rows in a table.
        
        Args:
            table: Table name
            values: Dictionary of column-value pairs to update
            where: Dictionary of column-value pairs for WHERE clause
            
        Returns:
            String with number of rows updated
        """
        # Sanitize table name
        table = self._sanitize_identifier(table)
        
        # Build SET clause
        set_parts = []
        params = {}
        param_count = 0
        
        for col, val in values.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            param_name = f"set_{param_count}"
            set_parts.append(f"{col} = :{param_name}")
            params[param_name] = val
            param_count += 1
            
        if not set_parts:
            raise ValueError("No values provided to update")
            
        # Build WHERE clause
        where_parts = []
        for col, val in where.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            param_name = f"where_{param_count}"
            where_parts.append(f"{col} = :{param_name}")
            params[param_name] = val
            param_count += 1
            
        if not where_parts:
            raise ValueError("No WHERE condition provided. To update all rows, use where={'1': 1}")
            
        # Build and execute query
        query = f"UPDATE {table} SET {', '.join(set_parts)} WHERE {' AND '.join(where_parts)}"
        
        result = self.connection.execute(text(query), params)
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows modified."
    
    def delete(self, table: str, where: Optional[Dict[str, Any]]) -> str:
        """
        Delete rows from a table.
        
        Args:
            table: Table name
            where: Dictionary of column-value pairs for WHERE clause
            
        Returns:
            String with number of rows deleted
        """
        # Sanitize table name
        table = self._sanitize_identifier(table)

        if where is None:
            where = {'1': 1}
        
        # Build WHERE clause
        where_parts = []
        params = {}
        param_count = 0
        
        for col, val in where.items():
            col = self._sanitize_identifier(col)
            self._sanitize_value(val)  # Validation
            param_name = f"where_{param_count}"
            where_parts.append(f"{col} = :{param_name}")
            params[param_name] = val
            param_count += 1
            
        if not where_parts:
            raise ValueError("No WHERE condition provided. To delete all rows, use where={'1': 1}")
            
        # Build and execute query
        query = f"DELETE FROM {table} WHERE {' AND '.join(where_parts)}"
        
        result = self.connection.execute(text(query), params)
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows deleted."
    
    @disabled("This method has been disabled for security reasons. Please use the specific CRUD methods instead.")
    def execute(self, query: str, params: Dict[str, Any] = None) -> Any:
        """
        Execute a raw SQL query. Use with caution!
        Only available during initialization.
        
        Args:
            query: SQL query string (using :param as the stand-in for parameters)
            params: Parameters for the query (using named parameters)
            
        Returns:
            Query results or affected row count
        """
        result = self.connection.execute(text(query), params or {})
        
        # For SELECT queries, return results
        if query.strip().upper().startswith("SELECT"):
            return [dict(row._mapping) for row in result]
        
        # For other queries, commit and return affected row count
        self.connection.commit()
        return f"Query Result: {result.rowcount} rows modified."