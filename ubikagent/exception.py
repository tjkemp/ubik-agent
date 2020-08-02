
class UbikAgentError(Exception):
    """Base class for exceptions in this project."""
    pass

class UbikTypeError(UbikAgentError, TypeError):
    """Error raised when passing arguments of the wrong type."""
    pass

class UbikFileExistsError(UbikAgentError, FileExistsError):
    """Error raised when a directory to be created exists already."""
    pass
