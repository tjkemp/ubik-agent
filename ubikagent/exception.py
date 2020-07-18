
class UbikAgentError(Exception):
    """Base class for exceptions in this project."""
    pass

class UbikTypeError(UbikAgentError, TypeError):
    """Error raised when passing arguments of the wrong type."""
    pass
