# Database Class Documentation

## Overview

The `Database` class provides a high-level, secure interface for database operations using SQLAlchemy. It supports multiple database engines and offers simplified methods for common database operations while preventing SQL injection attacks through parameterized queries.

## Features

- **Multiple Database Support**: Works with any SQLAlchemy-supported database (SQLite, MySQL, PostgreSQL, Oracle, etc.)
- **Flexible Configuration**: Load database settings from connection strings, JSON files, or environment files
- **SQL Injection Protection**: Built-in sanitization for identifiers and parameterized queries
- **Simplified CRUD Operations**: Easy-to-use methods for SELECT, INSERT, UPDATE, and DELETE
- **Transaction Management**: Automatic commit/rollback with context manager support
- **Advanced Query Features**: Support for GROUP BY, HAVING, ORDER BY, LIMIT, and OFFSET
- **Bulk Insert Support**: Insert data from dictionaries, lists, or pandas DataFrames
- **Stored Procedure Support**: Call stored procedures with input/output parameters
- **Connection Management**: Automatic connection handling and cleanup
- **Method Caching**: LRU cache for SELECT operations to improve performance
- **Raw SQL Access**: Execute custom SQL with security warnings

## Installation

```bash
pip install sqlalchemy python-dotenv pandas
```

## Quick Start

```python
from db import Database

# Using in-memory SQLite (default)
db = Database()

# Using a connection string
db = Database("postgresql://user:password@localhost/mydb")

# Using a configuration file
db = Database("config/database.env")
```

## Configuration Methods

### 1. Connection String (Default Method)

```python
# PostgreSQL
db = Database("postgresql://user:password@localhost:5432/mydb")

# MySQL
db = Database("mysql+pymysql://user:password@localhost:3306/mydb")

# SQLite
db = Database("sqlite:///path/to/database.db")

# In-memory SQLite (default)
db = Database("sqlite+pysqlite:///:memory:")

# Oracle
db = Database("oracle+cx_oracle://user:password@host:1521/?service_name=XE")
```

### 2. Environment File (.env)

Create a `.env` file:
```env
ENGINE=postgresql
DRIVER=psycopg2
HOSTNAME=localhost
PORT=5432
NAME=mydb
USERNAME=user
PASSWORD=password
```

Then load it:
```python
db = Database("config/database.env")
```

### 3. JSON Configuration

Create a `config.json` file:
```json
{
    "ENGINE": "mysql",
    "DRIVER": "pymysql",
    "HOSTNAME": "localhost",
    "PORT": 3306,
    "NAME": "mydb",
    "USERNAME": "user",
    "PASSWORD": "password"
}
```

Then load it:
```python
db = Database("config/database.json")
```

### 4. Dictionary Configuration

```python
config = {
    "ENGINE": "postgresql",
    "HOSTNAME": "localhost",
    "PORT": 5432,
    "NAME": "mydb",
    "USERNAME": "user",
    "PASSWORD": "password"
}
db = Database(config)
```

## Usage Examples

> [!IMPORTANT]
Always wrap database operations in try-except blocks for production code.

### Basic CRUD Operations

#### SELECT Examples

```python
# Select all columns from a table
results = db.select("users")

# Select specific columns
results = db.select("users", columns=["id", "name", "email"])

# Select with WHERE clause
results = db.select("users", where={"status": "active"})

# Select with multiple conditions
results = db.select("users",
    where={"status": "active", "role": "admin"}
)

# Select with ORDER BY
results = db.select("users",
    order_by="created_at"  # Default ASC
)

# Select with complex ORDER BY
results = db.select("users",
    order_by=[
        {"column": "status", "direction": "DESC"},
        {"column": "name", "direction": "ASC"}
    ]
)

# Select with GROUP BY and HAVING
results = db.select("orders",
    columns=["user_id", "COUNT(*) as order_count", "SUM(total) as total_spent"],
    group_by="user_id",
    having={"order_count": 5}  # Users with exactly 5 orders
)

# Select with LIMIT and OFFSET (pagination)
results = db.select("products",
    order_by="price",
    limit=10,
    offset=20  # Skip first 20 results
)
```

> [!NOTE]
The `select()` method uses LRU caching for improved performance on repeated queries.

#### INSERT Examples

##### Traditional Insert (columns + values)
```python
# Insert single row
db.insert("users",
    columns=["name", "email", "status"],
    values=[["John Doe", "john@example.com", "active"]]
)

# Insert multiple rows
db.insert("users",
    columns=["name", "email", "status"],
    values=[
        ["Alice", "alice@example.com", "active"],
        ["Bob", "bob@example.com", "pending"],
        ["Charlie", "charlie@example.com", "active"]
    ]
)
```

##### Bulk Insert (data parameter)
```python
# Insert from dictionary
user_data = {
    "name": "John Doe",
    "email": "john@example.com",
    "status": "active"
}
db.insert("users", data=user_data)

# Insert from list of dictionaries
users_data = [
    {"name": "Alice", "email": "alice@example.com", "status": "active"},
    {"name": "Bob", "email": "bob@example.com", "status": "pending"},
    {"name": "Charlie", "email": "charlie@example.com", "status": "active"}
]
db.insert("users", data=users_data)

# Insert from pandas DataFrame
import pandas as pd
df = pd.DataFrame({
    "name": ["Alice", "Bob", "Charlie"],
    "email": ["alice@example.com", "bob@example.com", "charlie@example.com"],
    "status": ["active", "pending", "active"]
})
db.insert("users", data=df)

# Bulk insert with additional pandas.to_sql parameters
db.insert("users", data=df, if_exists="append", index=False)
```

#### UPDATE Examples

```python
# Update single column
db.update("users",
    values={"status": "inactive"},
    where={"id": 5}
)

# Update multiple columns
db.update("users",
    values={
        "status": "active",
        "last_login": "2024-01-15 10:30:00"
    },
    where={"email": "john@example.com"}
)

# Update with multiple WHERE conditions
db.update("products",
    values={"price": 29.99},
    where={"category": "electronics", "brand": "TechCo"}
)
```

#### DELETE Examples

```python
# Delete specific rows
db.delete("users", where={"status": "inactive"})

# Delete with multiple conditions
db.delete("logs",
    where={"created_at": "2023-01-01", "severity": "debug"}
)

# Delete all rows (issues a warning)
db.delete("temp_data")  # Will issue a warning and delete all rows
```

> [!WARNING]
Calling `delete()` without a `where` parameter will issue a warning and delete all rows. Use `rollback()` to undo if this was unintended.

### Context Manager Usage

```python
# Automatic commit and cleanup on success, rollback on exception
with Database("postgresql://user:pass@localhost/mydb") as db:
    db.insert("users",
        columns=["name", "email"],
        values=[["Jane", "jane@example.com"]]
    )
    # Automatically commits on successful exit
    # Automatically rolls back on exception

# Manual transaction control
db = Database("mysql://user:pass@localhost/mydb")
try:
    db.insert("orders", columns=["user_id", "total"], values=[[1, 99.99]])
    db.update("inventory", values={"quantity": 5}, where={"product_id": 10})
    db.commit()
except Exception as e:
    db.rollback()
    raise
finally:
    db.close()
```

### Raw SQL Execution

```python
# Execute raw SQL (issues security warning)
query = "SELECT * FROM users WHERE created_at > :date AND status = :status"
params = {"date": "2024-01-01", "status": "active"}
results = db.execute(query, params)

# Complex raw query
query = """
    SELECT u.name, COUNT(o.id) as order_count
    FROM users u
    LEFT JOIN orders o ON u.id = o.user_id
    WHERE u.created_at > :start_date
    GROUP BY u.id, u.name
    HAVING COUNT(o.id) > :min_orders
"""
params = {"start_date": "2024-01-01", "min_orders": 5}
results = db.execute(query, params)
```

> [!CAUTION]
The `execute()` method issues a security warning. Use the specific CRUD methods when possible.

### Stored Procedures

```python
# Call procedure with input parameters
result = db.callproc("calculate_discount",
    in_params={"customer_id": 123, "order_total": 500.00}
)

# Call procedure with output parameters
result = db.callproc("get_customer_stats",
    in_params={"customer_id": 123},
    out_params={"total_orders": int, "total_spent": float}
)
print(f"Total orders: {result['total_orders']}")
print(f"Total spent: ${result['total_spent']}")

# Using positional parameters
result = db.callproc("update_inventory",
    in_params=[100, -5],  # product_id, quantity_change
    out_params=[int]  # new_quantity
)
```

### Connection Status and Management

```python
# Check connection status
if db.connected:
    print("Database is connected")

# Get connection string
print(f"Connected to: {db.connection_string}")

# Access underlying SQLAlchemy objects
engine = db.engine
session = db.session
connection = db.connection

# Use cursor directly
with db.cursor as cursor:
    cursor.execute("SHOW TABLES")
    tables = cursor.fetchall()
```

## API Reference

### Constructor

```python
Database(config: str | Dict[str, Any] = "sqlite+pysqlite:///:memory:")
```

**Parameters:**
- `config`: Database configuration (connection string, file path, or dictionary)

### Properties

- `connected`: Returns `True` if database connection is active
- `connection_string`: Returns the SQLAlchemy connection string
- `cursor`: Context manager that yields a database cursor
- `engine`: SQLAlchemy Engine instance
- `session`: SQLAlchemy Session instance
- `session_maker`: SQLAlchemy SessionMaker instance
- `connection`: SQLAlchemy Connection instance

### Methods

#### select()
```python
@lru_cache
select(table: str, columns: Optional[List[str]] = None,
       where: Optional[Dict[str, Any]] = None,
       order_by: Optional[str | List[Dict[str, str]]] = None,
       group_by: Optional[str | List[str]] = None,
       having: Optional[Dict[str, Any]] = None,
       limit: Optional[int] = None,
       offset: Optional[int] = None) -> List[Dict]
```

#### insert()
```python
insert(table: str, columns: Optional[List[str]] = None,
       values: Optional[List[List[Any]]] = None,
       data: Optional[list | dict | pd.DataFrame] = None) -> str
```

#### update()
```python
update(table: str, values: Dict[str, Any], where: Dict[str, Any]) -> str
```

#### delete()
```python
delete(table: str, where: Optional[Dict[str, Any]] = None) -> str
```

#### execute()
```python
execute(query: str, params: Optional[Dict[str, Any]] = None) -> Any
```

#### callproc()
```python
callproc(procedure: str, in_params: Dict[str, Any] | List[Any],
         out_params: Optional[Dict[str, type] | List[type]] = None) -> Dict[str, Any]
```

#### Transaction Methods
- `commit()`: Commit the current transaction
- `rollback()`: Rollback the current transaction
- `close()`: Close the database connection

## Security Features

1. **Identifier Sanitization**: Table and column names are validated to contain only alphanumeric characters and underscores
2. **Parameterized Queries**: All queries use parameterized statements to prevent SQL injection
3. **Value Validation**: Basic validation to detect potentially malicious values
4. **No Direct String Interpolation**: Never directly interpolates user input into SQL strings
5. **Security Warnings**: Issues warnings when using potentially unsafe operations like raw SQL execution

## Best Practices

1. **Use Context Managers**: Ensures proper cleanup and transaction handling
   ```python
   with Database("connection_string") as db:
       # Your database operations
   ```

2. **Prefer Specific Methods**: Use `select()`, `insert()`, `update()`, `delete()` instead of raw SQL

3. **Handle Exceptions**: Always handle database exceptions appropriately
   ```python
   try:
       db.insert("users", columns=["email"], values=[["test@example.com"]])
   except Exception as e:
       print(f"Database error: {e}")
       db.rollback()
   ```

4. **Use Bulk Operations**: For large datasets, use the `data` parameter in `insert()`
   ```python
   # More efficient for large datasets
   db.insert("users", data=large_dataframe)
   ```

5. **Close Connections**: Always close connections when done
   ```python
   db = Database("connection_string")
   try:
       # Your operations
   finally:
       db.close()
   ```

6. **Use Transactions**: Group related operations in transactions
   ```python
   try:
       db.update("accounts", {"balance": 100}, {"id": 1})
       db.update("accounts", {"balance": 200}, {"id": 2})
       db.commit()
   except Exception:
       db.rollback()
   ```

## Error Handling

The class raises various exceptions:
- `ValueError`: For invalid configurations, identifiers, or parameters
- `SQLAlchemyError`: For database-specific errors
- Connection errors from the underlying database driver

## Limitations

- The `having` clause requires a `group_by` clause
- `offset` requires `limit` for MySQL connections
- Raw SQL execution (`execute()`) bypasses some safety checks
- Stored procedure support depends on the database driver's capabilities
- LRU cache may cause memory usage with large result sets

## Advanced Features

### Connection String Construction

The class automatically constructs connection strings from configuration dictionaries:

```python
config = {
    "ENGINE": "postgresql",
    "DRIVER": "psycopg2",
    "HOSTNAME": "localhost",
    "PORT": 5432,
    "NAME": "mydb",
    "USERNAME": "user",
    "PASSWORD": "password",
    "QUERY_PARAMS": {"sslmode": "require"}
}
db = Database(config)
# Constructs: postgresql+psycopg2://user:password@localhost:5432/mydb?sslmode=require
```

### Memory Management

The class automatically handles memory management for in-memory SQLite databases:

```python
# Special handling for in-memory databases
db = Database("sqlite+pysqlite:///:memory:")
# Commit operations are skipped for memory databases
```

### Performance Optimization

- **LRU Caching**: The `select()` method uses `@lru_cache` for repeated queries
- **Bulk Operations**: Use pandas DataFrame integration for large datasets
- **Connection Pooling**: Leverages SQLAlchemy's built-in connection pooling
