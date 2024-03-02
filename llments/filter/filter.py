import abc


class Filter:
    @abc.abstractmethod
    def check_input(self, input: str) -> bool:
        """Check if the input passes the filter.

        Args:
            input: The input to be checked.

        Returns:
            True if the input passes the filter, False otherwise.
        """
        ...
