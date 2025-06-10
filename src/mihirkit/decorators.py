"""
Decorators module providing method control, timeout handling, retry logic, and caching.

This module offers a comprehensive collection of Python decorators for enhancing method
behavior while maintaining clean, readable code. It includes decorators for method control,
deprecation management, timeout protection, automatic retry logic, and advanced property
caching with TTL support.

Key Features:
    * Method Control: Completely disable dangerous or obsolete methods
    * Deprecation Management: Gradual migration with developer warnings
    * Timeout Protection: Process-based execution limits for reliable operation
    * Automatic Retry: Resilient operation handling for transient failures
    * Cached Properties: Performance optimization with setter support and TTL
    * Multiprocessing Support: True process isolation for timeout operations
    * Comprehensive Warnings: Detailed feedback for deprecated and retry operations

Classes:
    DisabledMethodError: Custom exception for disabled methods
    property: Enhanced cached property descriptor with TTL and setter support

Functions:
    disabled: Decorator to permanently disable methods
    deprecated: Decorator to mark methods as deprecated with warnings
    timeout: Decorator to enforce execution time limits
    retry: Decorator to automatically retry failed operations
    selective_retry: Convenience retry decorator for specific exceptions
    network_retry: Retry decorator optimized for network operations
"""

import multiprocessing
import pickle  # nosec B403 - Only used for internal caching, never with untrusted data
import threading
import time
import warnings
from functools import wraps
from typing import Any, Callable, NoReturn, Optional, Tuple, Union


class DisabledMethodError(Exception):
    """Exception raised when a deliberately disabled method is called."""

    pass


def disabled(reason: Optional[str] = None):
    """
    Disable a method and raise an error when it's called.

    Args:
        reason: Message explaining why the method is disabled

    Returns:
        Decorated function that raises DisabledMethodError when called
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> NoReturn:
            method_name = func.__name__
            message: str = f"Method '{method_name}' is disabled"
            if reason:
                message += f": {reason}."
            message += ". It is not available for use."
            raise DisabledMethodError(message)

        return wrapper

    return decorator


def deprecated(reason: Optional[str] = None, alternative: Optional[str] = None):
    """
    Mark a method as deprecated and issue warnings while still executing.

    Raises a DeprecationWarning but still executes the method.

    Args:
        reason: Message explaining why the method is deprecated
        alternative: Suggested alternative method to use instead

    Returns:
        Decorated function that issues a warning before execution
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            message = f"Method '{method_name}' is deprecated"
            if reason:
                message += f": {reason}"

            message += ". This method will be removed in a future version."
            if alternative:
                message += f" Use '{alternative}' instead."

            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def _is_picklable(obj) -> bool:
    """Check if an object can be pickled for multiprocessing."""
    try:
        pickle.dumps(obj)
        return True
    except (pickle.PicklingError, TypeError, AttributeError):
        return False


def _timeout_with_threads(
    seconds: int, func: Callable, args: tuple, kwargs: dict
) -> Any:
    """Implement thread-based timeout for non-picklable objects."""
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(seconds)

    if thread.is_alive():
        # Thread is still running, but we can't forcibly terminate it
        # This is a limitation of thread-based timeouts
        raise TimeoutError(
            f"Function {func.__name__} timed out after {seconds} seconds (thread-based timeout)"
        )

    if exception[0]:
        raise exception[0]

    return result[0]


def _timeout_with_process(
    seconds: int, func: Callable, args: tuple, kwargs: dict
) -> Any:
    """Implement process-based timeout for picklable objects."""

    def target(queue: multiprocessing.Queue, *args, **kwargs):
        try:
            result = func(*args, **kwargs)
            queue.put((True, result))
        except Exception as e:
            queue.put((False, e))

    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=target, args=(queue, *args), kwargs=kwargs)
    process.start()
    process.join(seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(
            f"Function {func.__name__} timed out after {seconds} seconds"
        )

    if queue.empty():
        raise TimeoutError(
            f"Function {func.__name__} timed out after {seconds} seconds"
        )

    success, result = queue.get()
    if success:
        return result
    else:
        raise result


def timeout(seconds: int, use_processes: bool = True) -> Callable:
    """
    Raise TimeoutError if function execution exceeds specified seconds.

    Args:
        seconds: Maximum execution time in seconds
        use_processes: If True, use multiprocessing (default). If False, use threading.
                      Automatically falls back to threading if arguments aren't picklable.
    """

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Check if we should use processes and if arguments are picklable
            if use_processes:
                # Check if all arguments are picklable
                try:
                    pickle.dumps((args, kwargs))
                    pickle.dumps(func)
                    return _timeout_with_process(seconds, func, args, kwargs)
                except (pickle.PicklingError, TypeError, AttributeError):
                    # Fall back to thread-based timeout
                    warnings.warn(
                        f"Arguments for {func.__name__} are not picklable, "
                        f"falling back to thread-based timeout (less reliable)",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    return _timeout_with_threads(seconds, func, args, kwargs)
            else:
                return _timeout_with_threads(seconds, func, args, kwargs)

        return wrapper

    return decorator


def retry(
    max_retries: int = 3,
    delay: int = 1,
    exceptions: Union[Tuple[type, ...], type] = Exception,
    exponential_backoff: bool = False,
) -> Callable:
    """
    Retry a function if it raises specified exceptions.

    Args:
        max_retries: Maximum number of retries before raising the original error
        delay: Base delay in seconds between retries
        exceptions: Exception type(s) to retry on. Can be a single exception or tuple of exceptions.
                   Defaults to Exception (all exceptions)
        exponential_backoff: If True, use exponential backoff (delay * 2^attempt)

    Returns:
        Decorated function that retries on failure
    """
    # Ensure exceptions is a tuple
    if not isinstance(exceptions, tuple):
        exceptions = (exceptions,)

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        # Calculate delay (exponential backoff if enabled)
                        current_delay = (
                            delay * (2**attempt) if exponential_backoff else delay
                        )

                        warnings.warn(
                            f"Retrying {func.__name__} due to error: {e}. "
                            f"Attempt {attempt + 1}/{max_retries}",
                            RuntimeWarning,
                            stacklevel=2,
                        )
                        time.sleep(current_delay)
                    else:
                        # All retries exhausted
                        break
                except Exception:
                    # Don't retry exceptions not in the specified list
                    raise

            # Re-raise the last exception if all retries failed
            raise last_exception

        return wrapper

    return decorator


def selective_retry(
    max_retries: int = 3,
    delay: int = 1,
    retry_on: Union[Tuple[type, ...], type] = (ConnectionError, TimeoutError),
    exponential_backoff: bool = False,
) -> Callable:
    """
    Retry only on specific common transient exceptions.

    Convenience decorator that only retries on specific common transient exceptions.

    Args:
        max_retries: Maximum number of retries
        delay: Base delay between retries
        retry_on: Exception types to retry on (default: ConnectionError, TimeoutError)
        exponential_backoff: Whether to use exponential backoff
    """
    return retry(
        max_retries=max_retries,
        delay=delay,
        exceptions=retry_on,
        exponential_backoff=exponential_backoff,
    )


def network_retry(
    max_retries: int = 3, delay: int = 2, exponential_backoff: bool = True
) -> Callable:
    """
    Retry network operations with optimized settings.

    Convenience decorator optimized for network operations.
    Retries on common network exceptions with exponential backoff.
    """
    network_exceptions = (
        ConnectionError,
        TimeoutError,
        OSError,  # Can include network-related OS errors
    )

    return retry(
        max_retries=max_retries,
        delay=delay,
        exceptions=network_exceptions,
        exponential_backoff=exponential_backoff,
    )


class property:
    """Cached property descriptor with setter support and TTL functionality."""

    def __init__(self, func: Callable, ttl: Optional[int] = None):
        """
        Initialize the cached property descriptor.

        Args:
            func: The function to cache
            ttl: Time to live in seconds for cache expiration (optional)
        """
        self.func = func
        self._setter: Callable = lambda: None
        self._deleter: Callable = lambda: None
        self.attr_name = f"_cached_{func.__name__}"
        self.timestamp_attr = f"_cached_{func.__name__}_timestamp"
        self.ttl = ttl  # Time to live in seconds
        self.__doc__ = func.__doc__

    def __get__(self, instance, owner=None):
        """Get the cached value or compute and cache if not available."""
        if instance is None:
            return self

        # Check if we have a cached value
        if self.attr_name in instance.__dict__:
            # Check TTL if specified
            if self.ttl is not None:
                timestamp = getattr(instance, self.timestamp_attr, 0)
                if time.time() - timestamp > self.ttl:
                    # Cache expired, remove it
                    delattr(instance, self.attr_name)
                    if hasattr(instance, self.timestamp_attr):
                        delattr(instance, self.timestamp_attr)
                else:
                    # Cache is still valid
                    return instance.__dict__[self.attr_name]
            else:
                # No TTL, return cached value
                return instance.__dict__[self.attr_name]

        # Compute and cache the value
        value = self.func(instance)
        instance.__dict__[self.attr_name] = value
        if self.ttl is not None:
            setattr(instance, self.timestamp_attr, time.time())
        return value

    def __set__(self, instance, value):
        """Set the cached value, optionally applying setter transformation."""
        if self._setter:
            value = self._setter(instance, value)
        instance.__dict__[self.attr_name] = value
        if self.ttl is not None:
            setattr(instance, self.timestamp_attr, time.time())

    def __delete__(self, instance):
        """Delete the cached value and timestamp."""
        instance.__dict__.pop(self.attr_name, None)
        if hasattr(instance, self.timestamp_attr):
            delattr(instance, self.timestamp_attr)

    def setter(self, func: Callable):
        """Enable the use of `@<property>.setter` syntax."""
        self._setter = func
        return self

    def deleter(self, func: Callable):
        """Enable the use of `@<property>.deleter` syntax."""
        self._deleter = func
        return self

    def invalidate_cache(self, instance):
        """Manually invalidate the cache for this property."""
        if hasattr(instance, self.attr_name):
            delattr(instance, self.attr_name)
        if hasattr(instance, self.timestamp_attr):
            delattr(instance, self.timestamp_attr)
