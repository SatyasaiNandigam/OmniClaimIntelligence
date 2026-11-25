"""Shared exception types."""


class ConfigError(Exception):
    """One or more config/pack files are missing or invalid. Message lists every problem found."""


class RegistryError(KeyError):
    """Unknown or duplicate registry entry."""

    def __str__(self) -> str:  # KeyError would otherwise repr() the message
        return str(self.args[0])
