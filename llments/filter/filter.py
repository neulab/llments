"""A module for filters that can be applied to the input."""
import abc


class Filter:
    """A filter that can be applied to the input."""

    @abc.abstractmethod
    def check_input(self, input: str) -> bool:
        """Check if the input passes the filter.

        Args:
            input: The input to be checked.

        Returns:
            True if the input passes the filter, False otherwise.
        """
        ...
