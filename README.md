# MihirKit

A high-performance Python toolkit providing secure database operations, optimized filesystem management, powerful decorators, and essential data processing utilities. Designed for production environments with emphasis on performance, security, and developer experience.

## 🚀 Quick Start

```python
from mihirkit import Database, Directory, retry, natural_sort

# Secure database operations
with Database("sqlite:///app.db") as db:
    users = db.select("users", where={"active": True}, limit=10)
    db.insert("logs", data={"action": "login", "user_id": 123})

# High-performance file discovery
python_files = Directory.get_files(
    "/project",
    ext="py",
    sort_on="modified",
    traverse_levels=True
)

# Resilient API calls
@retry(max_retries=3, exponential_backoff=True)
def fetch_data():
    return requests.get("https://api.example.com/data")

# Natural sorting
files = natural_sort(["file1.txt", "file10.txt", "file2.txt"])
# Result: ["file1.txt", "file2.txt", "file10.txt"]
```

## 📦 Installation

```bash
# Basic installation (core features)
pip install mihirkit

# Development installation
git clone https://github.com/yourusername/mihirkit.git
cd mihirkit
pip install -e .
```

### Dependencies

**Core Dependencies:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing

**Optional Dependencies:**
- `sqlalchemy` - Database operations (required for Database class)
- `python-dotenv` - Environment file support (for database configuration)

## 🔋 Features Overview

### 🗄️ Database Operations
Multi-engine database interface with security-first design, supporting SQLite, PostgreSQL, MySQL, Oracle, and more. Features include SQL injection protection, transaction management, and efficient bulk operations.

**[📖 Full Database Documentation](docs/db.md)**

### 📁 Filesystem Management
High-performance file and directory operations with parallel processing, memory-efficient iteration, regex filtering, and advanced sorting capabilities.

**[📖 Full Filesystem Documentation](docs/filesystem.md)**

### 🎯 Decorators
Comprehensive decorator collection for method control, timeout protection, retry logic, deprecation management, and performance optimization with caching.

**[📖 Full Decorators Documentation](docs/decorators.md)**

### 🔧 Utilities
Essential data processing utilities including recursive flattening of nested structures and natural sorting for human-readable ordering.

**[📖 Full Utilities Documentation](docs/utilities.md)**

## 📚 Quick Examples

### Database Operations

```python
from mihirkit import Database

# Simple database operations
with Database("postgresql://user:pass@localhost/mydb") as db:
    # SELECT with conditions
    users = db.select("users", where={"status": "active"}, limit=10)

    # Bulk INSERT from DataFrame
    import pandas as pd
    df = pd.DataFrame({"name": ["Alice", "Bob"], "age": [25, 30]})
    db.insert("users", data=df)

    # UPDATE with conditions
    db.update("users", {"last_login": "2024-01-15"}, {"id": 123})
```

### Filesystem Operations

```python
from mihirkit import Directory, File

# Find files with advanced filtering
python_files = Directory.get_files(
    "/project",
    ext="py",
    filter=r"^(test_|main).*",  # Files starting with "test_" or "main"
    sort_on="modified",
    traverse_levels=True
)

# Memory-efficient iteration
for file_path in Directory.iter_files("/large/dataset", ext="csv"):
    with File(file_path) as f:
        process_data(f.read())
```

### Decorators

```python
from mihirkit import timeout, retry, deprecated, disabled

@timeout(30)
@retry(max_retries=3, exponential_backoff=True)
def robust_api_call():
    return requests.get("https://api.example.com/data")

@deprecated("Use new_method for better performance", "new_method")
def legacy_function():
    return "still works but warns"

@disabled("Security vulnerability found")
def dangerous_operation():
    pass  # This will raise DisabledMethodError when called
```

### Utilities

```python
from mihirkit import flatten, natural_sort

# Flatten nested structures
nested = [1, [2, [3, 4]], np.array([[5, 6]])]
flat = flatten(nested)  # [1, 2, 3, 4, 5, 6]

# Natural sorting
versions = ["v1.10", "v1.2", "v1.1", "v2.0"]
sorted_versions = natural_sort(versions)  # ["v1.1", "v1.2", "v1.10", "v2.0"]
```

## 🔧 Configuration Examples

### Database Configuration

```python
# Connection string (recommended)
db = Database("postgresql://user:password@localhost:5432/mydb")

# Environment file
db = Database("config/database.env")

# Dictionary configuration
config = {
    "ENGINE": "mysql",
    "HOSTNAME": "localhost",
    "PORT": 3306,
    "USERNAME": "user",
    "PASSWORD": "password",
    "NAME": "mydb"
}
db = Database(config)
```

### Performance Tuning

```python
# Enable parallel processing for large directories
large_files = Directory.get_files(
    "/huge/directory",
    use_parallel=True,
    max_workers=8,
    head=1000
)

# Configure retry behavior
@retry(
    max_retries=5,
    delay=2,
    exponential_backoff=True,
    exceptions=(ConnectionError, TimeoutError)
)
def network_operation():
    pass
```

## 🔒 Security Features

- **SQL Injection Prevention**: Parameterized queries and identifier sanitization
- **Path Validation**: Protection against directory traversal attacks
- **Method Control**: Complete blocking of dangerous operations with `@disabled`
- **Connection Security**: SSL/TLS support for database connections
- **Permission Handling**: Graceful handling of filesystem access issues

## 🚀 Performance

### Key Optimizations
- **Parallel Processing**: Multi-threaded filesystem operations
- **Memory Efficiency**: Generator-based iteration for large datasets
- **Caching**: LRU caching for frequently accessed patterns and results
- **Optimized Algorithms**: `os.scandir` for filesystem operations, efficient sorting

### Performance Guidelines
- Use `use_parallel=True` for directories with >10,000 files
- Use generator methods (`iter_files`, `iter_subdirs`) for very large result sets
- Leverage `head` parameter to limit results for better performance
- Use bulk database operations for large datasets

## 📖 Complete Documentation

| Module | Description | Documentation |
|--------|-------------|---------------|
| **Database** | Secure multi-engine database operations | [docs/db.md](docs/db.md) |
| **Filesystem** | High-performance file and directory utilities | [docs/filesystem.md](docs/filesystem.md) |
| **Decorators** | Method control, retry logic, and caching | [docs/decorators.md](docs/decorators.md) |
| **Utilities** | Data processing and sorting utilities | [docs/utilities.md](docs/utilities.md) |

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/

# Run with coverage
python -m pytest tests/ --cov=mihirkit --cov-report=html

# Run specific module tests
python -m pytest tests/test_database.py -v
python -m pytest tests/test_filesystem.py -v
python -m pytest tests/test_decorators.py -v
python -m pytest tests/test_utilities.py -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/yourusername/mihirkit.git
cd mihirkit

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Run tests
pytest
```

### Code Standards
- **Formatting**: Black code formatter
- **Linting**: Flake8 with line length 88
- **Type Hints**: Full type annotation coverage
- **Documentation**: Google-style docstrings

## 📈 Roadmap

### Version 1.0 (Current: v0.0.1)
- [x] Core database operations with security features
- [x] High-performance filesystem utilities
- [x] Comprehensive decorator system
- [x] Data processing utilities
- [x] Complete documentation and examples
- [ ] Comprehensive test suite (in progress)
- [ ] Performance benchmarks
- [ ] API stability guarantees

### Version 1.1 (Planned)
- [ ] Async/await support for database operations
- [ ] Enhanced caching with Redis integration
- [ ] Additional database engines (MongoDB, ClickHouse)
- [ ] Command-line interface for common operations
- [ ] Plugin architecture for custom extensions

### Version 2.0 (Future)
- [ ] Advanced query builder with method chaining
- [ ] Distributed filesystem operations
- [ ] Machine learning utilities integration
- [ ] Enhanced monitoring and logging
- [ ] Cloud storage backends support

## 📄 License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **SQLAlchemy**: Database abstraction and ORM functionality
- **Pandas**: Data manipulation and analysis
- **Python Standard Library**: Core functionality and design patterns

## 📞 Support

- **Documentation**: Complete documentation available in the [docs/](docs/) directory
- **Issues**: [GitHub Issues](https://github.com/yourusername/mihirkit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/mihirkit/discussions)

---

**MihirKit v0.0.1** - Made with ❤️ for the Python community
