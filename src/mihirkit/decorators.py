from typing import Callable, Any
from functools import wraps
import concurrent.futures
import warnings

# ----- DISABLED -----
class DisabledMethodError(Exception):
    """Exception raised when a deliberately disabled method is called."""
    pass

def disabled(reason: str = "This method has been disabled"):
    """
    Decorator to disable a method and raise an error when it's called.
    
    Args:
        reason: Message explaining why the method is disabled
        
    Returns:
        Decorated function that raises DisabledMethodError when called
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            method_name = func.__name__
            raise DisabledMethodError(f"Method '{method_name}' is disabled: {reason}")
        return wrapper
    return decorator


# ----- DEPRECATED -----
def deprecated(reason: str = "This method is deprecated", alternative: str = None):
    """
    Decorator to mark a method as deprecated.
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
            message = f"Method '{method_name}' is deprecated: {reason}"
            
            if alternative:
                message += f". Use '{alternative}' instead"
                
            warnings.warn(message, DeprecationWarning, stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


# ----- TIMEOUT -----
def timeout(seconds: int) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(func, *args, **kwargs)
                try:
                    return future.result(timeout=seconds)
                except concurrent.futures.TimeoutError:
                    raise TimeoutError(f"Function {func.__name__} timed out after {seconds} seconds")
        return wrapper
    return decorator