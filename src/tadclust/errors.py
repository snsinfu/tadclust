"""Exception types for user-facing failures.

Errors are dataclasses. Catch a specific class for normal handling, or catch
TadclustError and match on the class and its payload fields.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(eq=False)
class TadclustError(Exception):
    message: str
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        super().__init__(self.message)
        for note in self.notes:
            self.add_note(note)


@dataclass(eq=False)
class ConfigError(TadclustError):
    field: str | None = None


@dataclass(eq=False)
class InputError(TadclustError):
    path: str | None = None
    row: int | None = None


@dataclass(eq=False)
class McoolError(TadclustError):
    path: str | None = None
    resolution: int | None = None


@dataclass(eq=False)
class ClusteringError(TadclustError):
    """Failure while selecting units, building features, or clustering."""

    chrom: str | None = None
    matrix: str | None = None


@dataclass(eq=False)
class TrackError(TadclustError):
    path: str | None = None
    chrom: str | None = None


@dataclass(eq=False)
class OutputError(TadclustError):
    path: str | None = None
