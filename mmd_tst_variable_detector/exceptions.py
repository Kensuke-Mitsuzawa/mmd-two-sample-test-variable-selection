class OptimizationException(Exception):
    # exception when the optimization is failed.
    pass


class SameDataException(Exception):
    """Exception raised when x and y data are the same."""
    pass


class ParameterSearchException(Exception):
    """Exception raised when the parameter search is failed."""
    pass