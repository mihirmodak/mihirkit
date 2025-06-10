# Decorators Class Documentation

## Overview

The `decorators.py` module provides a comprehensive collection of Python decorators for method control, deprecation management, timeout handling, retry logic, and advanced property caching. These decorators enhance method behavior while maintaining clean, readable code.

## Features

- **Method Control**: Completely disable dangerous or obsolete methods
- **Deprecation Management**: Gradual migration with developer warnings
- **Timeout Protection**: Process-based execution limits for reliable operation
- **Automatic Retry**: Resilient operation handling for transient failures
- **Cached Properties**: Performance optimization with setter support
- **Multiprocessing Support**: True process isolation for timeout operations
- **Comprehensive Warnings**: Detailed feedback for deprecated and retry operations

## Installation

```bash
# No additional dependencies required beyond Python standard library
# All required modules are part of Python's standard library:
# - typing, functools, concurrent.futures, multiprocessing, warnings, time
```

## Quick Start

```python
from decorators import disabled, deprecated, timeout, retry, property

# Disable a dangerous method
@disabled("This method causes data corruption")
def dangerous_operation():
    pass

# Mark method as deprecated
@deprecated("Use new_method instead", "new_method")
def old_method():
    return "legacy behavior"

# Add timeout protection
@timeout(30)
def network_operation():
    # Long-running operation
    pass
```

## Decorator Reference

### 1. @disabled

Completely prevents method execution and raises `DisabledMethodError` when called.

#### Signature

```python
def disabled(reason: Optional[str] = None) -> Callable
```

#### Parameters

- **`reason`** *(Optional[str])*: Explanation for why the method is disabled

#### Raises

- **`DisabledMethodError`**: Custom exception raised when the method is called

#### Usage Examples

```python
class SecurityService:
    @disabled()
    def delete_all_data(self):
        """Completely blocked method"""
        pass

    @disabled("Security vulnerability found - CVE-2024-001")
    def legacy_auth(self, token):
        """Method disabled for security reasons"""
        return self._validate_token(token)

service = SecurityService()

# Attempting to call disabled method
try:
    service.delete_all_data()
except Exception as e:
    print(f"Error: {e}")
# Output: Error: Method delete_all_data is disabled. It is not available for use.

# Disabled method with specific reason
try:
    service.legacy_auth("token123")
except Exception as e:
    print(f"Error: {e}")
# Output: Error: Method legacy_auth is disabled: Security vulnerability found - CVE-2024-001. It is not available for use.
```

> [!WARNING]
> Disabled methods are permanently blocked and cannot be overridden at runtime. Use this decorator only when you need to completely prevent method execution.

### 2. @deprecated

Issues deprecation warnings while still allowing method execution. Essential for gradual API migration.

#### Signature

```python
def deprecated(reason: Optional[str] = None, alternative: Optional[str] = None) -> Callable
```

#### Parameters

- **`reason`** *(Optional[str])*: Explanation for deprecation
- **`alternative`** *(Optional[str])*: Suggested replacement method name

#### Warning Details

- Issues `DeprecationWarning` with `stacklevel=2`
- Method continues to execute normally after warning

#### Usage Examples

```python
import warnings

class DataProcessor:
    @deprecated()
    def old_format(self, data):
        return data.upper()

    @deprecated("Performance issues with large datasets")
    def inefficient_sort(self, items):
        return sorted(items, reverse=True)  # Inefficient implementation

    @deprecated("Use process_data_v3 for better performance", "process_data_v3")
    def process_data_v2(self, data):
        return [item.strip() for item in data]

    def process_data_v3(self, data):
        """New optimized version"""
        return [item.strip() for item in data if item.strip()]

# Enable deprecation warnings
warnings.simplefilter("always")

processor = DataProcessor()

# Basic deprecation warning
result = processor.old_format("hello")
print(f"Result: {result}")
# Warning: Method 'old_format' is deprecated. This method will be removed in a future version.

# Deprecation with specific reason
sorted_data = processor.inefficient_sort([3, 1, 4, 1, 5])
print(f"Sorted: {sorted_data}")
# Warning: Method 'inefficient_sort' is deprecated: Performance issues with large datasets. This method will be removed in a future version.

# Deprecation with alternative suggestion
processed = processor.process_data_v2(["  hello  ", "  world  "])
print(f"Processed: {processed}")
# Warning: Method 'process_data_v2' is deprecated: Use process_data_v3 for better performance. This method will be removed in a future version. Use 'process_data_v3' instead.
```

> [!NOTE]
> Deprecated methods continue to function normally. The decorator only issues warnings to encourage migration to newer alternatives.

### 3. @timeout

Enforces execution time limits using multiprocessing for true process termination, more reliable than thread-based approaches.

#### Signature

```python
def timeout(seconds: int, use_processes: bool = True) -> Callable
```

#### Parameters

- **`seconds`** *(int)*: Maximum execution time in seconds
- **`use_processes`** *(bool)*: Whether to use multiprocessing (default: True). Automatically falls back to threading if arguments aren't picklable

#### Implementation Details

- **Smart Fallback**: Automatically detects if arguments are picklable and falls back to threading if needed
- **Process Isolation**: Uses `multiprocessing.Process` for true process isolation when possible
- **Thread Fallback**: Falls back to threading for non-picklable objects with appropriate warnings
- **Automatic Cleanup**: Properly terminates processes and handles cleanup

#### Raises

- **`TimeoutError`**: When execution exceeds the specified time limit

#### Usage Examples

```python
import time

class NetworkService:
    @timeout(5)
    def quick_api_call(self):
        """API call that completes quickly"""
        time.sleep(2)
        return {"status": "success", "data": "API response"}

    @timeout(3)
    def slow_operation(self):
        """Operation that will timeout"""
        time.sleep(10)  # Exceeds 3-second limit
        return "This will never be returned"

    @timeout(4)
    def failing_operation(self):
        """Operation that fails within timeout"""
        time.sleep(1)
        raise ConnectionError("Network unreachable")

service = NetworkService()

# Successful operation within timeout
try:
    result = service.quick_api_call()
    print(f"Success: {result}")
except TimeoutError as e:
    print(f"Timeout: {e}")
# Output: Success: {'status': 'success', 'data': 'API response'}

# Operation that exceeds timeout
try:
    result = service.slow_operation()
    print(f"Success: {result}")
except TimeoutError as e:
    print(f"Timeout: {e}")
# Output: Timeout: Function slow_operation timed out after 3 seconds

# Exception handling within timeout period
try:
    result = service.failing_operation()
    print(f"Success: {result}")
except TimeoutError as e:
    print(f"Timeout: {e}")
except ConnectionError as e:
    print(f"Network Error: {e}")
# Output: Network Error: Network unreachable
```

> [!IMPORTANT]
> The timeout decorator uses multiprocessing, which requires:
> - Function arguments must be picklable
> - No shared global state between processes
> - Higher resource overhead than threading

> [!CAUTION]
> Process creation has significant overhead. Avoid using timeout on frequently called methods or those with very short execution times.

### 4. @retry

Automatically retries failed method calls with configurable attempts and delays. Issues detailed warnings for each retry attempt.

#### Signature

```python
def retry(max_retries: int = 3, delay: int = 1,
          exceptions: Union[Tuple[type, ...], type] = Exception,
          exponential_backoff: bool = False) -> Callable
```

#### Parameters

- **`max_retries`** *(int)*: Maximum number of retry attempts (default: 3)
- **`delay`** *(int)*: Base delay in seconds between retries (default: 1)
- **`exceptions`** *(Union[Tuple[type, ...], type])*: Exception type(s) to retry on (default: Exception for all exceptions)
- **`exponential_backoff`** *(bool)*: Whether to use exponential backoff for delays (default: False)

#### Behavior

- **Selective Exception Handling**: Only retries specified exception types
- **Exponential Backoff**: Optional exponential delay increase between attempts
- **Proper Error Reporting**: Correct attempt numbering in warning messages
- **Non-Retryable Exceptions**: Immediately re-raises exceptions not in the retry list

#### Usage Examples

```python
import random
import warnings

class APIClient:
    def __init__(self):
        self.request_count = 0

    @retry()  # Default: retries all exceptions
    def unreliable_request(self):
        self.request_count += 1
        if random.random() < 0.6:  # 60% failure rate
            raise ConnectionError(f"Request {self.request_count} failed")
        return f"Success on request {self.request_count}"

    @retry(max_retries=5, delay=2, exceptions=(ConnectionError, TimeoutError))
    def selective_operation(self):
        """Only retries network-related errors"""
        if random.random() < 0.3:
            raise ConnectionError("Network error")
        elif random.random() < 0.1:
            raise ValueError("This won't be retried")
        return "Operation completed"

    @retry(max_retries=3, delay=1, exponential_backoff=True)
    def backoff_operation(self):
        """Uses exponential backoff: 1s, 2s, 4s delays"""
        if random.random() < 0.8:
            raise ConnectionError("Service unavailable")
        return "Success with backoff"

# Enable all warnings to see retry attempts
warnings.simplefilter("always")

client = APIClient()

# Example with eventual success after retries
random.seed(42)  # For reproducible example
try:
    result = client.unreliable_request()
    print(f"Final result: {result}")
except Exception as e:
    print(f"Failed after all retries: {e}")

# Example with selective exception handling
try:
    result = client.selective_operation()
    print(f"Selective result: {result}")
except ValueError as e:
    print(f"ValueError not retried: {e}")
except Exception as e:
    print(f"Failed after retries: {e}")

# Convenience decorators for common patterns
@selective_retry(max_retries=2, retry_on=(ConnectionError, TimeoutError))
def database_operation():
    """Only retries common transient errors"""
    if random.random() < 0.5:
        raise ConnectionError("Database connection lost")
    return "Database operation successful"

@network_retry(max_retries=3, exponential_backoff=True)
def api_call():
    """Optimized for network operations with exponential backoff"""
    if random.random() < 0.7:
        raise ConnectionError("API server unavailable")
    return "API call successful"

# Test convenience decorators
try:
    result = database_operation()
    print(f"Database result: {result}")
except Exception as e:
    print(f"Database operation failed: {e}")

try:
    result = api_call()
    print(f"API result: {result}")
except Exception as e:
    print(f"API call failed: {e}")
```

> [!NOTE]
> The improved retry decorator now properly handles selective exception retrying and includes convenient presets for common scenarios.

### 5. property

A custom property descriptor that provides caching behavior with setter support. This is **not** Python's built-in `@property`.

#### Features

- **Automatic Caching**: Values are computed once and cached with optional TTL
- **Setter Support**: Allows value modification through `@property.setter` syntax
- **Deletion Support**: Cache can be cleared to force recomputation
- **TTL Support**: Optional time-to-live for automatic cache expiration
- **Memory Efficient**: Uses instance `__dict__` for storage with automatic cleanup

#### Implementation Details

- Caches values using the pattern: `_cached_{function_name}`
- Optional TTL with automatic expiration checking
- Setter functions can transform values before caching
- Manual cache invalidation methods available

#### Usage Examples

```python
import time
import math

class ScientificCalculator:
    def __init__(self, dataset):
        self._dataset = dataset
        self._computation_count = 0

    @property
    def expensive_analysis(self):
        """Performs expensive statistical analysis with caching"""
        print("Performing complex statistical analysis...")
        self._computation_count += 1
        time.sleep(2)  # Simulate expensive computation

        # Complex statistical calculations
        mean = sum(self._dataset) / len(self._dataset)
        variance = sum((x - mean) ** 2 for x in self._dataset) / len(self._dataset)
        return {
            "mean": mean,
            "variance": variance,
            "std_dev": math.sqrt(variance),
            "computation_id": self._computation_count
        }

    @expensive_analysis.setter
    def expensive_analysis(self, result_dict):
        """Custom setter with validation"""
        if not isinstance(result_dict, dict):
            raise TypeError("Analysis result must be a dictionary")
        if "mean" not in result_dict:
            raise ValueError("Analysis result must contain 'mean' key")
        return result_dict  # Return validated value for caching

class TemperatureConverter:
    def __init__(self, celsius):
        self._celsius = celsius

    @property(ttl=30)  # Cache expires after 30 seconds
    def fahrenheit(self):
        """Convert celsius to fahrenheit with TTL caching"""
        print(f"Converting {self._celsius}°C to Fahrenheit...")
        return (self._celsius * 9 / 5) + 32

    @fahrenheit.setter
    def fahrenheit(self, fahrenheit_value):
        """Set temperature via fahrenheit, automatically update celsius"""
        print(f"Setting temperature to {fahrenheit_value}°F")
        self._celsius = (fahrenheit_value - 32) * 5 / 9
        return fahrenheit_value  # Cache the fahrenheit value

class DataAnalyzer:
    def __init__(self):
        self._raw_data = [1, 2, 3]
        self._data_version = 0

    @property
    def analysis_result(self):
        """Analysis with automatic cache invalidation"""
        print("Computing analysis...")
        return sum(self._raw_data)

    def add_data(self, value):
        """Add data and automatically invalidate cache"""
        self._raw_data.append(value)
        # Manually invalidate cache when data changes
        self.analysis_result.invalidate_cache(self)

# Demonstration of caching behavior
calculator = ScientificCalculator([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("=== Basic Caching Demonstration ===")

# First access - triggers computation
result1 = calculator.expensive_analysis
print(f"First access result: {result1}")
print(f"Computation count: {calculator._computation_count}")

# Second access - uses cached value (no computation)
result2 = calculator.expensive_analysis
print(f"Second access result: {result2}")
print(f"Computation count: {calculator._computation_count}")

print("\n=== TTL Caching Demonstration ===")

# TTL caching example
temp = TemperatureConverter(25)
print(f"25°C = {temp.fahrenheit}°F")  # Triggers conversion
print(f"Cached value: {temp.fahrenheit}°F")  # Uses cached result

# Wait for cache to expire (in real usage)
# time.sleep(31)  # Would cause recomputation
# print(f"After TTL expiry: {temp.fahrenheit}°F")  # Would trigger new conversion

print("\n=== Manual Cache Invalidation ===")

# Cache invalidation example
analyzer = DataAnalyzer()
print(f"Initial analysis: {analyzer.analysis_result}")  # Computes: sum([1,2,3]) = 6
print(f"Cached analysis: {analyzer.analysis_result}")   # Uses cached value

analyzer.add_data(10)  # This invalidates the cache
print(f"After adding data: {analyzer.analysis_result}")  # Recomputes: sum([1,2,3,10]) = 16
```

> [!TIP]
> The enhanced property class now supports TTL (time-to-live) for automatic cache expiration and manual cache invalidation methods for better memory management.'mean' key")
        return result_dict  # Return validated value for caching

class TemperatureConverter:
    def __init__(self, celsius):
        self._celsius = celsius

    @property
    def fahrenheit(self):
        """Convert celsius to fahrenheit with caching"""
        print(f"Converting {self._celsius}°C to Fahrenheit...")
        return (self._celsius * 9 / 5) + 32

    @fahrenheit.setter
    def fahrenheit(self, fahrenheit_value):
        """Set temperature via fahrenheit, automatically update celsius"""
        print(f"Setting temperature to {fahrenheit_value}°F")
        self._celsius = (fahrenheit_value - 32) * 5 / 9
        return fahrenheit_value  # Cache the fahrenheit value

# Demonstration of caching behavior
calculator = ScientificCalculator([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

print("=== Caching Demonstration ===")

# First access - triggers computation
result1 = calculator.expensive_analysis
print(f"First access result: {result1}")
print(f"Computation count: {calculator._computation_count}")

# Second access - uses cached value (no computation)
result2 = calculator.expensive_analysis
print(f"Second access result: {result2}")
print(f"Computation count: {calculator._computation_count}")

# Manual override using setter
new_result = {"mean": 100, "variance": 0, "std_dev": 0, "computation_id": 999}
calculator.expensive_analysis = new_result
print(f"After manual set: {calculator.expensive_analysis}")
print(f"Computation count: {calculator._computation_count}")

# Delete cache to force recomputation
del calculator.expensive_analysis
result3 = calculator.expensive_analysis
print(f"After cache deletion: {result3}")
print(f"Final computation count: {calculator._computation_count}")

print("\n=== Temperature Converter ===")

# Temperature conversion with caching
temp = TemperatureConverter(25)
print(f"25°C = {temp.fahrenheit}°F")  # Triggers conversion
print(f"Cached value: {temp.fahrenheit}°F")  # Uses cached result

# Set via fahrenheit
temp.fahrenheit = 100
print(f"After setting 100°F: Celsius = {temp._celsius}°C")
print(f"Fahrenheit (cached): {temp.fahrenheit}°F")
```

> [!TIP]
> This custom property class is ideal for expensive computations that don't change frequently but might need occasional manual updates.

> [!NOTE]
> Unlike Python's built-in `@property`, this implementation automatically caches computed values. Use the standard `@property` if you need fresh computation on every access.

## Advanced Usage Examples

### Combining Decorators

Decorators can be combined for sophisticated behavior control. The order of application matters:

```python
import warnings

class RobustAPIClient:
    """Example of combining multiple decorators effectively"""

    @deprecated("Use enhanced_api_call for better reliability", "enhanced_api_call")
    @retry(max_retries=3, delay=2)
    @timeout(30)
    def legacy_api_call(self, endpoint, data=None):
        """Legacy API call with full protection stack"""
        import random
        import time

        # Simulate API behavior
        if random.random() < 0.3:  # 30% chance of failure
            raise ConnectionError("API server temporarily unavailable")

        time.sleep(5)  # Simulate network delay
        return {
            "endpoint": endpoint,
            "data": data,
            "timestamp": time.time(),
            "status": "success"
        }

    @retry(max_retries=2, delay=1)
    @timeout(15)
    def enhanced_api_call(self, endpoint, data=None):
        """Modern API call with improved error handling"""
        time.sleep(1)  # Faster operation
        return {
            "endpoint": endpoint,
            "data": data,
            "version": "2.0",
            "status": "success"
        }

    @disabled("This method has critical security vulnerabilities")
    def unsafe_operation(self):
        """Method disabled for security reasons"""
        pass

# Usage demonstration
warnings.simplefilter("always")
client = RobustAPIClient()

try:
    # This will show deprecation warning, might retry on failure, and has timeout protection
    result = client.legacy_api_call("/users", {"filter": "active"})
    print(f"Legacy API result: {result}")
except (TimeoutError, ConnectionError) as e:
    print(f"Legacy API failed: {e}")

try:
    # Modern API call
    result = client.enhanced_api_call("/users", {"filter": "active"})
    print(f"Enhanced API result: {result}")
except Exception as e:
    print(f"Enhanced API failed: {e}")

try:
    # Attempt to use disabled method
    client.unsafe_operation()
except Exception as e:
    print(f"Security blocked: {e}")
```

### Real-World Example: Database Connection Manager

```python
import time
import random

class DatabaseConnectionManager:
    """Production-ready database manager with comprehensive decorator usage"""

    def __init__(self):
        self._connection = None
        self._last_health_check = None
        self._query_count = 0

    @retry(max_retries=5, delay=3)
    @timeout(60)
    def connect(self, host, port, database):
        """Establish database connection with retry and timeout protection"""
        print(f"Attempting connection to {host}:{port}/{database}")

        # Simulate connection process that might fail
        if random.random() < 0.2:  # 20% chance of connection failure
            raise ConnectionError(f"Failed to connect to {host}:{port}")

        time.sleep(3)  # Simulate connection establishment time
        self._connection = {
            "host": host,
            "port": port,
            "database": database,
            "connected_at": time.time()
        }
        print("Database connection established successfully")
        return self._connection

    @property
    def connection_health(self):
        """Expensive health check with caching"""
        if not self._connection:
            return {"status": "disconnected", "health": "unknown"}

        print("Performing comprehensive health check...")
        time.sleep(2)  # Simulate expensive health check

        # Simulate health check results
        health_score = random.uniform(0.7, 1.0)
        return {
            "status": "connected",
            "health": "good" if health_score > 0.8 else "degraded",
            "score": health_score,
            "checked_at": time.time()
        }

    @deprecated("Use execute_parameterized_query for better security", "execute_parameterized_query")
    @retry(max_retries=3, delay=1)
    @timeout(30)
    def execute_query(self, sql):
        """Legacy query execution method"""
        if not self._connection:
            raise RuntimeError("Database not connected")

        self._query_count += 1

        # Simulate query execution that might fail
        if random.random() < 0.1:  # 10% chance of query failure
            raise RuntimeError("Query execution failed - deadlock detected")

        time.sleep(1)  # Simulate query execution time
        return {
            "sql": sql,
            "rows_affected": random.randint(1, 100),
            "execution_time": 1.0,
            "query_id": self._query_count
        }

    @retry(max_retries=2, delay=1)
    @timeout(20)
    def execute_parameterized_query(self, sql, params=None):
        """Modern, secure query execution"""
        if not self._connection:
            raise RuntimeError("Database not connected")

        time.sleep(0.5)  # Faster execution
        return {
            "sql": sql,
            "params": params,
            "rows_affected": random.randint(1, 50),
            "execution_time": 0.5,
            "secure": True
        }

    @disabled("Direct SQL execution disabled for security compliance")
    def execute_raw_sql(self, sql):
        """Dangerous method that bypasses all safety checks"""
        pass

    def disconnect(self):
        """Clean disconnection"""
        if self._connection:
            print("Disconnecting from database...")
            self._connection = None
            # Clear cached health status
            if hasattr(self, '_cached_connection_health'):
                del self._cached_connection_health

# Comprehensive demonstration
warnings.simplefilter("always")
db_manager = DatabaseConnectionManager()

print("=== Database Connection Example ===")

# Establish connection with retry and timeout protection
try:
    connection = db_manager.connect("prod-db-01", 5432, "analytics")
    print(f"Connected: {connection}")
except Exception as e:
    print(f"Connection failed after all retries: {e}")

# Check connection health (cached property)
print(f"\nHealth Check 1: {db_manager.connection_health}")
print(f"Health Check 2: {db_manager.connection_health}")  # Uses cached result

# Execute queries using deprecated method
try:
    result = db_manager.execute_query("SELECT * FROM users WHERE active = 1")
    print(f"\nLegacy query result: {result}")
except Exception as e:
    print(f"Legacy query failed: {e}")

# Execute queries using modern method
try:
    result = db_manager.execute_parameterized_query(
        "SELECT * FROM users WHERE active = ?",
        params=[1]
    )
    print(f"Modern query result: {result}")
except Exception as e:
    print(f"Modern query failed: {e}")

# Attempt to use disabled method
try:
    db_manager.execute_raw_sql("DROP TABLE users;")
except Exception as e:
    print(f"Security prevention: {e}")

# Clean shutdown
db_manager.disconnect()
```

## Best Practices

### Decorator Selection Guide

| Use Case | Recommended Decorator | Configuration |
|----------|----------------------|---------------|
| Security-critical methods | `@disabled` | Provide clear reason |
| API migration | `@deprecated` | Include alternative method |
| Network operations | `@timeout` + `@retry` | 30s timeout, 3 retries |
| Database operations | `@retry` + `@timeout` | 2 retries, 60s timeout |
| Expensive computations | `property` | Cache results, manual clearing |

### Configuration Recommendations

```python
# Production-ready configurations

# Conservative retry for critical operations
@retry(max_retries=3, delay=2)

# Aggressive timeout for user-facing operations
@timeout(10)

# Informative deprecation messages
@deprecated("Deprecated in v2.1.0 - security improvements", "secure_method_v2")

# Disable dangerous operations
@disabled("Method removed due to CVE-2024-001 security vulnerability")
```

### Optimal Decorator Ordering

When combining decorators, apply them in this order (innermost to outermost):

```python
# Recommended stacking order:
@deprecated("Use new_method", "new_method")  # 4. Warn about usage (outermost)
@retry(max_retries=3, delay=1)              # 3. Handle failures
@timeout(30)                                # 2. Control execution time
def legacy_operation():                     # 1. Core functionality (innermost)
    pass
```

> [!TIP]
> **Rationale**: Timeout controls the maximum time per attempt, retry handles multiple attempts, and deprecation warns about the entire operation regardless of success/failure.

## Error Handling

### Exception Hierarchy

The decorators raise specific exceptions that should be handled appropriately:

```python
# Exception types raised by each decorator
try:
    decorated_method()
except DisabledMethodError:
    # From @disabled - method permanently blocked
    logger.error("Attempted to call disabled method")
except TimeoutError:
    # From @timeout - execution exceeded time limit
    logger.warning("Operation timed out - may need retry")
except Exception as e:
    # From @retry - original exception after all retries exhausted
    # From property - any exception from underlying computation
    logger.error(f"Operation failed: {e}")
```

### Comprehensive Error Handling Pattern

```python
def robust_operation_handler(operation_func, *args, **kwargs):
    """Template for handling all decorator exceptions"""
    try:
        return operation_func(*args, **kwargs)
    except DisabledMethodError as e:
        # Method is permanently disabled
        logger.error(f"Disabled method called: {e}")
        return {"error": "operation_disabled", "message": str(e)}
    except TimeoutError as e:
        # Operation took too long
        logger.warning(f"Operation timeout: {e}")
        return {"error": "timeout", "message": str(e)}
    except ConnectionError as e:
        # Network/connection issues (often from @retry exhaustion)
        logger.error(f"Connection failed: {e}")
        return {"error": "connection_failed", "message": str(e)}
    except Exception as e:
        # Other errors
        logger.error(f"Unexpected error: {e}")
        return {"error": "unknown", "message": str(e)}
```

## Performance Considerations

### Resource Usage Analysis

| Decorator | CPU Overhead | Memory Usage | I/O Impact | Best Use Case |
|-----------|-------------|--------------|------------|---------------|
| `@disabled` | Minimal | Minimal | None | Security, compliance |
| `@deprecated` | Low | Low | None | API migration |
| `@timeout` | High* | Medium* | None | Critical operations |
| `@retry` | Variable** | Low | Variable** | Unreliable services |
| `property` | Low*** | Medium*** | None | Expensive computations |

*\* Process creation overhead*
*\** Depends on failure rate and retry configuration*
*\*** Caching reduces CPU, increases memory*

### Performance Optimization Guidelines

> [!TIP]
> **Timeout Optimization**: Reserve `@timeout` for operations that truly need process isolation. For simple time limits, consider using `signal.alarm()` or async timeouts.

> [!TIP]
> **Retry Optimization**: Implement exponential backoff for better performance:
> ```python
> # Instead of fixed delays, consider exponential backoff
> import time
>
> def exponential_retry(max_retries=3, base_delay=1):
>     def decorator(func):
>         @wraps(func)
>         def wrapper(*args, **kwargs):
>             for attempt in range(max_retries):
>                 try:
>                     return func(*args, **kwargs)
>                 except Exception as e:
>                     if attempt < max_retries - 1:
>                         delay = base_delay * (2 ** attempt)  # Exponential backoff
>                         time.sleep(delay)
>                     else:
>                         raise
>         return wrapper
>     return decorator
> ```

> [!TIP]
> **Property Cache Management**: For long-running applications, implement periodic cache clearing:
> ```python
> class LongRunningService:
>     @property
>     def expensive_data(self):
>         return self._compute_expensive_data()
>
>     def clear_caches(self):
>         """Call periodically to prevent memory bloat"""
>         attrs_to_clear = [attr for attr in self.__dict__
>                          if attr.startswith('_cached_')]
>         for attr in attrs_to_clear:
>             delattr(self, attr)
> ```

### Memory Management

```python
class MemoryEfficientProcessor:
    """Example of memory-conscious decorator usage"""

    @property
    def large_dataset_analysis(self):
        """Cache management for large results"""
        # For very large results, consider size limits
        result = self._analyze_large_dataset()

        # Optional: Implement size-based cache eviction
        if hasattr(self, '_cache_size_limit'):
            self._check_cache_size()

        return result

    def _check_cache_size(self):
        """Implement cache size monitoring"""
        import sys
        cached_attrs = {k: v for k, v in self.__dict__.items()
                       if k.startswith('_cached_')}
        total_size = sum(sys.getsizeof(v) for v in cached_attrs.values())

        if total_size > self._cache_size_limit:
            # Clear oldest cache entries
            self._evict_cache_entries()
```

## Troubleshooting

### Common Issues and Solutions

#### Issue: @timeout not interrupting CPU-bound loops

```python
# Problem: Infinite CPU loops cannot be interrupted
@timeout(5)
def cpu_intensive_loop():
    while True:  # This cannot be forcibly terminated
        pass

# Solution: Use cooperative interruption
@timeout(5)
def cooperative_cpu_work():
    import signal
    import time

    def timeout_handler(signum, frame):
        raise TimeoutError("Operation interrupted")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)

    try:
        # Your CPU-intensive work here
        for i in range(1000000):
            if i % 10000 == 0:  # Check periodically
                time.sleep(0.001)  # Allow signal handling
    finally:
        signal.alarm(0)  # Clear alarm
```

#### Issue: Pickle errors with @timeout

```python
# Problem: Complex objects can't be pickled for multiprocessing
class ComplexObject:
    def __init__(self):
        self.lock = threading.Lock()  # Cannot be pickled

@timeout(10)
def method_with_complex_args(self, complex_obj):
    return complex_obj.process()

# Solution: Use simpler interfaces
@timeout(10)
def method_with_simple_args(self, data_dict):
    # Reconstruct complex object inside the method
    complex_obj = ComplexObject()
    complex_obj.load_from_dict(data_dict)
    return complex_obj.process()
```

#### Issue: @retry masking critical errors

```python
# Problem: Retrying all exceptions, including critical ones
@retry(max_retries=5)
def critical_operation():
    raise SystemExit("Critical system error")  # This shouldn't be retried!

# Solution: Create selective retry decorator
def selective_retry(max_retries=3, delay=1, retry_exceptions=(ConnectionError, TimeoutError)):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except retry_exceptions as e:
                    if attempt < max_retries - 1:
                        time.sleep(delay)
                    else:
                        raise
                except Exception:
                    # Don't retry other exceptions
                    raise
        return wrapper
    return decorator
```

#### Issue: Property cache not updating when underlying data changes

```python
# Problem: Cached property doesn't reflect data changes
class DataAnalyzer:
    def __init__(self):
        self._raw_data = [1, 2, 3]

    @property
    def analysis_result(self):
        return sum(self._raw_data)  # Cached result won't update

    def add_data(self, value):
        self._raw_data.append(value)
### Convenience Decorators

The improved implementation includes specialized decorators for common use cases:

#### selective_retry()

```python
def selective_retry(max_retries: int = 3, delay: int = 1,
                   retry_on: Union[Tuple[type, ...], type] = (ConnectionError, TimeoutError),
                   exponential_backoff: bool = False) -> Callable
```

Preconfigured for common transient exceptions.

#### network_retry()

```python
def network_retry(max_retries: int = 3, delay: int = 2, exponential_backoff: bool = True) -> Callable
```

Optimized for network operations with exponential backoff enabled by default.

#### Usage Examples

```python
# For database operations - only retry connection issues
@selective_retry(max_retries=3, retry_on=(ConnectionError, TimeoutError))
def database_query():
    pass

# For API calls - optimized network retry with exponential backoff
@network_retry(max_retries=5)
def api_request():
    pass

# Custom selective retry for specific exceptions
@retry(max_retries=2, exceptions=(ValueError, KeyError), exponential_backoff=True)
def data_processing():
    pass
```

## Advanced Usage Examples

### Combining Decorators

Decorators can be combined for sophisticated behavior control. The order of application matters:

```python
import warnings

class RobustAPIClient:
    """Example of combining multiple decorators effectively"""

    @deprecated("Use enhanced_api_call for better reliability", "enhanced_api_call")
    @network_retry(max_retries=3)  # Uses exponential backoff by default
    @timeout(30, use_processes=True)  # Auto-fallback if needed
    def legacy_api_call(self, endpoint, data=None):
        """Legacy API call with full protection stack"""
        import random
        import time

        # Simulate API behavior
        if random.random() < 0.3:  # 30% chance of failure
            raise ConnectionError("API server temporarily unavailable")

        time.sleep(5)  # Simulate network delay
        return {
            "endpoint": endpoint,
            "data": data,
            "timestamp": time.time(),
            "status": "success"
        }

    @selective_retry(max_retries=2, retry_on=(ConnectionError, TimeoutError))
    @timeout(15)
    def enhanced_api_call(self, endpoint, data=None):
        """Modern API call with improved error handling"""
        time.sleep(1)  # Faster operation
        return {
            "endpoint": endpoint,
            "data": data,
            "version": "2.0",
            "status": "success"
        }

    @disabled("This method has critical security vulnerabilities")
    def unsafe_operation(self):
        """Method disabled for security reasons"""
        pass

# Usage demonstration
warnings.simplefilter("always")
client = RobustAPIClient()

try:
    # This will show deprecation warning, use network retry, and has timeout protection
    result = client.legacy_api_call("/users", {"filter": "active"})
    print(f"Legacy API result: {result}")
except (TimeoutError, ConnectionError) as e:
    print(f"Legacy API failed: {e}")

try:
    # Modern API call with selective retry
    result = client.enhanced_api_call("/users", {"filter": "active"})
    print(f"Enhanced API result: {result}")
except Exception as e:
    print(f"Enhanced API failed: {e}")

try:
    # Attempt to use disabled method
    client.unsafe_operation()
except DisabledMethodError as e:
    print(f"Security blocked: {e}")
```

### Real-World Example: Database Connection Manager

```python
import time
import random

class DatabaseConnectionManager:
    """Production-ready database manager with comprehensive decorator usage"""

    def __init__(self):
        self._connection = None
        self._last_health_check = None
        self._query_count = 0

    @network_retry(max_retries=5, delay=3)
    @timeout(60)
    def connect(self, host, port, database):
        """Establish database connection with retry and timeout protection"""
        print(f"Attempting connection to {host}:{port}/{database}")

        # Simulate connection process that might fail
        if random.random() < 0.2:  # 20% chance of connection failure
            raise ConnectionError(f"Failed to connect to {host}:{port}")

        time.sleep(3)  # Simulate connection establishment time
        self._connection = {
            "host": host,
            "port": port,
            "database": database,
            "connected_at": time.time()
        }
        print("Database connection established successfully")
        return self._connection

    @property(ttl=30)  # Cache expires after 30 seconds
    def connection_health(self):
        """Expensive health check with TTL caching"""
        if not self._connection:
            return {"status": "disconnected", "health": "unknown"}

        print("Performing comprehensive health check...")
        time.sleep(2)  # Simulate expensive health check

        # Simulate health check results
        health_score = random.uniform(0.7, 1.0)
        return {
            "status": "connected",
            "health": "good" if health_score > 0.8 else "degraded",
            "score": health_score,
            "checked_at": time.time()
        }

    @deprecated("Use execute_parameterized_query for better security", "execute_parameterized_query")
    @selective_retry(max_retries=3, retry_on=(ConnectionError, TimeoutError))
    @timeout(30)
    def execute_query(self, sql):
        """Legacy query execution method"""
        if not self._connection:
            raise RuntimeError("Database not connected")

        self._query_count += 1

        # Simulate query execution that might fail
        if random.random() < 0.1:  # 10% chance of query failure
            raise ConnectionError("Query execution failed - connection lost")

        time.sleep(1)  # Simulate query execution time
        return {
            "sql": sql,
            "rows_affected": random.randint(1, 100),
            "execution_time": 1.0,
            "query_id": self._query_count
        }

    @retry(max_retries=2, exceptions=(ConnectionError, TimeoutError), exponential_backoff=True)
    @timeout(20)
    def execute_parameterized_query(self, sql, params=None):
        """Modern, secure query execution"""
        if not self._connection:
            raise RuntimeError("Database not connected")

        time.sleep(0.5)  # Faster execution
        return {
            "sql": sql,
            "params": params,
            "rows_affected": random.randint(1, 50),
            "execution_time": 0.5,
            "secure": True
        }

    @disabled("Direct SQL execution disabled for security compliance")
    def execute_raw_sql(self, sql):
        """Dangerous method that bypasses all safety checks"""
        pass

    def disconnect(self):
        """Clean disconnection"""
        if self._connection:
            print("Disconnecting from database...")
            self._connection = None
            # Clear cached health status
            self.connection_health.invalidate_cache(self)

# Comprehensive demonstration
warnings.simplefilter("always")
db_manager = DatabaseConnectionManager()

print("=== Database Connection Example ===")

# Establish connection with retry and timeout protection
try:
    connection = db_manager.connect("prod-db-01", 5432, "analytics")
    print(f"Connected: {connection}")
except Exception as e:
    print(f"Connection failed after all retries: {e}")

# Check connection health (TTL cached property)
print(f"\nHealth Check 1: {db_manager.connection_health}")
print(f"Health Check 2: {db_manager.connection_health}")  # Uses cached result

# Execute queries using deprecated method
try:
    result = db_manager.execute_query("SELECT * FROM users WHERE active = 1")
    print(f"\nLegacy query result: {result}")
except Exception as e:
    print(f"Legacy query failed: {e}")

# Execute queries using modern method
try:
    result = db_manager.execute_parameterized_query(
        "SELECT * FROM users WHERE active = ?",
        params=[1]
    )
    print(f"Modern query result: {result}")
except Exception as e:
    print(f"Modern query failed: {e}")

# Attempt to use disabled method
try:
    db_manager.execute_raw_sql("DROP TABLE users;")
except DisabledMethodError as e:
    print(f"Security prevention: {e}")

# Clean shutdown
db_manager.disconnect()
```

### Enhanced Testing for Improved Features

```python
import unittest
import warnings
from unittest.mock import patch, MagicMock
import time
import threading

class TestImprovedDecorators(unittest.TestCase):
    """Test the improved decorator implementations"""

    def test_timeout_automatic_fallback(self):
        """Test timeout decorator's automatic fallback to threading"""

        class UnpicklableObject:
            def __init__(self):
                self.lock = threading.Lock()

        @timeout(2)
        def method_with_unpicklable_arg(obj):
            time.sleep(0.5)
            return "success"

        unpicklable_obj = UnpicklableObject()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = method_with_unpicklable_arg(unpicklable_obj)

            # Should succeed despite unpicklable argument
            self.assertEqual(result, "success")

            # Should issue fallback warning
            fallback_warnings = [warning for warning in w
                                if "falling back" in str(warning.message)]
            self.assertEqual(len(fallback_warnings), 1)

    def test_retry_selective_exceptions(self):
        """Test retry decorator's selective exception handling"""

        call_count = [0]

        @retry(max_retries=2, exceptions=(ConnectionError,), delay=0)
        def selective_method():
            call_count[0] += 1
            if call_count[0] == 1:
                raise ConnectionError("Will be retried")
            elif call_count[0] == 2:
                raise ValueError("Won't be retried")

        with self.assertRaises(ValueError):
            selective_method()

        # Should have attempted twice (initial + 1 retry for ConnectionError)
        self.assertEqual(call_count[0], 2)

    def test_retry_exponential_backoff(self):
        """Test retry decorator's exponential backoff"""

        delays = []

        def mock_sleep(delay):
            delays.append(delay)

        with patch('time.sleep', side_effect=mock_sleep):
            @retry(max_retries=3, delay=1, exponential_backoff=True)
            def always_fails():
                raise ConnectionError("Test error")

            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")
                with self.assertRaises(ConnectionError):
                    always_fails()

        # Should use exponential backoff: 1s, 2s delays
        self.assertEqual(delays, [1, 2])

    def test_property_ttl_expiration(self):
        """Test property TTL functionality"""

        class TestClass:
            def __init__(self):
                self.computation_count = 0

            @property(ttl=1)  # 1 second TTL
            def computed_value(self):
                self.computation_count += 1
                return f"computed_{self.computation_count}"

        obj = TestClass()

        # First access
        result1 = obj.computed_value
        self.assertEqual(result1, "computed_1")
        self.assertEqual(obj.computation_count, 1)

        # Second access within TTL
        result2 = obj.computed_value
        self.assertEqual(result2, "computed_1")  # Cached
        self.assertEqual(obj.computation_count, 1)

        # Wait for TTL to expire
        time.sleep(1.1)

        # Access after TTL expiry
        result3 = obj.computed_value
        self.assertEqual(result3, "computed_2")  # Recomputed
        self.assertEqual(obj.computation_count, 2)

    def test_property_manual_invalidation(self):
        """Test property manual cache invalidation"""

        class TestClass:
            def __init__(self):
                self.computation_count = 0

            @property
            def computed_value(self):
                self.computation_count += 1
                return f"computed_{self.computation_count}"

        obj = TestClass()

        # First access
        result1 = obj.computed_value
        self.assertEqual(result1, "computed_1")

        # Manual invalidation
        obj.computed_value.invalidate_cache(obj)

        # Next access should recompute
        result2 = obj.computed_value
        self.assertEqual(result2, "computed_2")

    def test_convenience_decorators(self):
        """Test convenience decorator functions"""

        call_count = [0]

        @network_retry(max_retries=2)
        def network_operation():
            call_count[0] += 1
            if call_count[0] < 2:
                raise ConnectionError("Network error")
            return "success"

        with warnings.catch_warnings(record=True):
            warnings.simplefilter("always")
            result = network_operation()

            self.assertEqual(result, "success")
            self.assertEqual(call_count[0], 2)

    def test_disabled_method_custom_exception(self):
        """Test that disabled methods raise the custom DisabledMethodError"""

        @disabled("Security reasons")
        def dangerous_method():
            return "should not execute"

        with self.assertRaises(DisabledMethodError) as context:
            dangerous_method()

        # Should be our custom exception type
        self.assertIsInstance(context.exception, DisabledMethodError)
        self.assertIn("Security reasons", str(context.exception))

if __name__ == '__main__':
    unittest.main(verbosity=2)
```
import time
from typing import Dict, Any

# Setup structured logging for decorator events
decorator_logger = logging.getLogger('decorators')
decorator_logger.setLevel(logging.INFO)

# Custom handler for decorator metrics
class DecoratorMetricsHandler(logging.Handler):
    """Custom handler to collect decorator metrics"""

    def __init__(self):
        super().__init__()
        self.metrics = {
            'timeout_errors': 0,
            'retry_attempts': 0,
            'deprecated_calls': 0,
            'disabled_attempts': 0
        }

    def emit(self, record):
        if 'timeout' in record.getMessage().lower():
            self.metrics['timeout_errors'] += 1
        elif 'retrying' in record.getMessage().lower():
            self.metrics['retry_attempts'] += 1
        elif 'deprecated' in record.getMessage().lower():
            self.metrics['deprecated_calls'] += 1
        elif 'disabled' in record.getMessage().lower():
            self.metrics['disabled_attempts'] += 1

metrics_handler = DecoratorMetricsHandler()
decorator_logger.addHandler(metrics_handler)

# Enhanced decorators with monitoring
def monitored_retry(max_retries: int = 3, delay: int = 1):
    """Enhanced retry decorator with monitoring"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            for attempt in range(max_retries):
                try:
                    result = func(*args, **kwargs)
                    # Log successful completion
                    execution_time = time.time() - start_time
                    decorator_logger.info(
                        f"Function {func.__name__} succeeded on attempt {attempt + 1} "
                        f"in {execution_time:.2f}s"
                    )
                    return result
                except Exception as e:
                    if attempt < max_retries - 1:
                        decorator_logger.warning(
                            f"Retrying {func.__name__} due to {type(e).__name__}: {e}. "
                            f"Attempt {attempt + 1}/{max_retries}"
                        )
                        time.sleep(delay)
                    else:
                        execution_time = time.time() - start_time
                        decorator_logger.error(
                            f"Function {func.__name__} failed after {max_retries} attempts "
                            f"in {execution_time:.2f}s. Final error: {e}"
                        )
                        raise
        return wrapper
    return decorator

def monitored_timeout(seconds: int):
    """Enhanced timeout decorator with monitoring"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                # Use original timeout implementation
                result = timeout(seconds)(func)(*args, **kwargs)
                execution_time = time.time() - start_time
                decorator_logger.info(
                    f"Function {func.__name__} completed in {execution_time:.2f}s "
                    f"(timeout: {seconds}s)"
                )
                return result
            except TimeoutError as e:
                decorator_logger.error(
                    f"Function {func.__name__} timed out after {seconds}s"
                )
                raise
        return wrapper
    return decorator
```

### Configuration Management

```python
import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class DecoratorConfig:
    """Centralized configuration for decorator behavior"""

    # Retry settings
    default_max_retries: int = 3
    default_retry_delay: int = 1
    retry_exponential_backoff: bool = False

    # Timeout settings
    default_timeout_seconds: int = 30
    timeout_warning_threshold: float = 0.8  # Warn at 80% of timeout

    # Deprecation settings
    deprecation_warnings_enabled: bool = True
    deprecation_log_level: str = "WARNING"

    # Property caching
    property_cache_max_size: Optional[int] = None
    property_cache_ttl: Optional[int] = None  # Time to live in seconds

    @classmethod
    def from_environment(cls) -> 'DecoratorConfig':
        """Load configuration from environment variables"""
        return cls(
            default_max_retries=int(os.getenv('DECORATOR_MAX_RETRIES', 3)),
            default_retry_delay=int(os.getenv('DECORATOR_RETRY_DELAY', 1)),
            retry_exponential_backoff=os.getenv('DECORATOR_EXPONENTIAL_BACKOFF', 'false').lower() == 'true',
            default_timeout_seconds=int(os.getenv('DECORATOR_TIMEOUT_SECONDS', 30)),
            timeout_warning_threshold=float(os.getenv('DECORATOR_TIMEOUT_WARNING', 0.8)),
            deprecation_warnings_enabled=os.getenv('DECORATOR_DEPRECATION_WARNINGS', 'true').lower() == 'true',
            deprecation_log_level=os.getenv('DECORATOR_DEPRECATION_LOG_LEVEL', 'WARNING'),
            property_cache_max_size=int(os.getenv('DECORATOR_CACHE_MAX_SIZE')) if os.getenv('DECORATOR_CACHE_MAX_SIZE') else None,
            property_cache_ttl=int(os.getenv('DECORATOR_CACHE_TTL')) if os.getenv('DECORATOR_CACHE_TTL') else None
        )

# Global configuration instance
decorator_config = DecoratorConfig.from_environment()

# Environment-aware decorators
def production_retry(max_retries: Optional[int] = None, delay: Optional[int] = None):
    """Production-ready retry decorator with configuration support"""
    effective_retries = max_retries or decorator_config.default_max_retries
    effective_delay = delay or decorator_config.default_retry_delay

    if decorator_config.retry_exponential_backoff:
        return monitored_retry(effective_retries, effective_delay)  # With exponential backoff
    else:
        return monitored_retry(effective_retries, effective_delay)

def production_timeout(seconds: Optional[int] = None):
    """Production-ready timeout decorator with configuration support"""
    effective_timeout = seconds or decorator_config.default_timeout_seconds
    return monitored_timeout(effective_timeout)
```

### Health Checks and Metrics

```python
def get_decorator_health_status() -> Dict[str, Any]:
    """Get health status and metrics for decorator usage"""
    return {
        "metrics": metrics_handler.metrics,
        "configuration": {
            "max_retries": decorator_config.default_max_retries,
            "timeout_seconds": decorator_config.default_timeout_seconds,
            "deprecation_warnings": decorator_config.deprecation_warnings_enabled
        },
        "recommendations": _get_performance_recommendations()
    }

def _get_performance_recommendations() -> List[str]:
    """Analyze metrics and provide performance recommendations"""
    recommendations = []
    metrics = metrics_handler.metrics

    if metrics['timeout_errors'] > 10:
        recommendations.append("High timeout error rate - consider increasing timeout values")

    if metrics['retry_attempts'] > 50:
        recommendations.append("High retry rate - investigate underlying service reliability")

    if metrics['deprecated_calls'] > 0:
        recommendations.append(f"{metrics['deprecated_calls']} deprecated method calls detected - plan migration")

    return recommendations

# Export health check endpoint for monitoring systems
def decorator_health_check():
    """Health check endpoint for monitoring systems"""
    status = get_decorator_health_status()

    # Determine overall health
    metrics = status['metrics']
    is_healthy = (
        metrics['timeout_errors'] < 5 and
        metrics['retry_attempts'] < 20 and
        len(status['recommendations']) == 0
    )

    return {
        "status": "healthy" if is_healthy else "degraded",
        "details": status
    }
```

---

## Conclusion

The `decorators.py` module provides a robust foundation for production Python applications requiring reliable method control, timeout protection, retry logic, and performance optimization. Each decorator has been designed with production use in mind, offering comprehensive error handling, monitoring capabilities, and flexible configuration options.

### Key Takeaways

- **Security First**: Use `@disabled` for permanent method blocking when security is paramount
- **Gradual Migration**: Leverage `@deprecated` for smooth API transitions with clear guidance
- **Reliability**: Combine `@timeout` and `@retry` for resilient external service interactions
- **Performance**: Utilize the custom `property` class for expensive computation caching
- **Production Ready**: Implement monitoring, configuration management, and health checks

### Next Steps

1. **Customize for Your Environment**: Adapt the configuration system to match your deployment needs
2. **Integrate Monitoring**: Connect decorator metrics to your observability platform
3. **Establish Policies**: Define organization-wide standards for decorator usage
4. **Regular Review**: Periodically audit decorator usage and performance metrics

> [!NOTE]
> This documentation represents the complete functionality available in the `decorators.py` module. For the most current implementation details, always refer to the source code and maintain comprehensive test coverage for any modifications.
        # Cache is now stale!

# Solution: Invalidate cache when data changes
class SmartDataAnalyzer:
    def __init__(self):
        self._raw_data = [1, 2, 3]

    @property
    def analysis_result(self):
        return sum(self._raw_data)

    def add_data(self, value):
        self._raw_data.append(value)
