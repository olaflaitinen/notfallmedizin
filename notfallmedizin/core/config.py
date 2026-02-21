# Copyright 2026 Gustav Olaf Yunus Laitinen-Fredriksson Lundström-Imanov.
# SPDX-License-Identifier: Apache-2.0

"""Global configuration management for the notfallmedizin library.

Provides a thread-safe, centralized configuration system for
library-wide behaviour: random state, parallelism, verbosity,
numerical precision, and caching. Use get_config/set_config
or config_context for temporary overrides.

References:
    Context manager pattern: PEP 343.
"""

from __future__ import annotations

import copy
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Generator, Literal, Optional, Union

from notfallmedizin.core.exceptions import ConfigurationError

_lock = threading.Lock()


@dataclass
class NotfallmedizinConfig:
    """Library-wide configuration container.

    Parameters
    ----------
    random_state : int or None, optional
        Seed for reproducible results. ``None`` means non-deterministic
        behaviour. Default is ``None``.
    n_jobs : int, optional
        Number of parallel jobs. ``1`` disables parallelism; ``-1`` uses
        all available cores. Default is ``1``.
    verbose : bool, optional
        If ``True``, enable informational log output. Default is ``False``.
    precision : {"float32", "float64"}, optional
        Default floating-point precision for numerical computations.
        Default is ``"float64"``.
    cache_dir : str or Path or None, optional
        Directory for caching intermediate results. ``None`` disables
        caching. Default is ``None``.
    """

    random_state: Optional[int] = None
    n_jobs: int = 1
    verbose: bool = False
    precision: Literal["float32", "float64"] = "float64"
    cache_dir: Optional[Union[str, Path]] = None

    def __post_init__(self) -> None:
        """Validate configuration values after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Check that all configuration values are within acceptable bounds.

        Raises
        ------
        ConfigurationError
            If any configuration value is invalid.
        """
        if self.random_state is not None:
            if not isinstance(self.random_state, int) or self.random_state < 0:
                raise ConfigurationError(
                    f"'random_state' must be a non-negative integer or None, "
                    f"got {self.random_state!r}."
                )

        if not isinstance(self.n_jobs, int) or self.n_jobs == 0:
            raise ConfigurationError(
                f"'n_jobs' must be a non-zero integer, got {self.n_jobs!r}."
            )

        if not isinstance(self.verbose, bool):
            raise ConfigurationError(
                f"'verbose' must be a boolean, got {type(self.verbose).__name__}."
            )

        if self.precision not in ("float32", "float64"):
            raise ConfigurationError(
                f"'precision' must be 'float32' or 'float64', got {self.precision!r}."
            )

        if self.cache_dir is not None:
            self.cache_dir = Path(self.cache_dir)


_global_config = NotfallmedizinConfig()


def get_config() -> NotfallmedizinConfig:
    """Return a deep copy of the current global configuration.

    Returns
    -------
    NotfallmedizinConfig
        A copy of the active configuration. Mutating the returned object
        does not affect the global state; use :func:`set_config` instead.

    Examples
    --------
    >>> from notfallmedizin.core.config import get_config
    >>> cfg = get_config()
    >>> cfg.n_jobs
    1
    """
    with _lock:
        return copy.deepcopy(_global_config)


def set_config(**kwargs: Any) -> None:
    """Update the global configuration with the provided keyword arguments.

    Only recognized configuration fields are accepted. Unknown keys raise
    :class:`ConfigurationError`.

    Parameters
    ----------
    **kwargs
        Keyword arguments corresponding to fields of
        :class:`NotfallmedizinConfig`.

    Raises
    ------
    ConfigurationError
        If an unrecognized configuration key is provided.

    Examples
    --------
    >>> from notfallmedizin.core.config import set_config, get_config
    >>> set_config(n_jobs=4, verbose=True)
    >>> get_config().n_jobs
    4
    """
    global _global_config
    valid_keys = {f.name for f in fields(NotfallmedizinConfig)}
    invalid_keys = set(kwargs.keys()) - valid_keys
    if invalid_keys:
        raise ConfigurationError(
            f"Unrecognized configuration keys: {sorted(invalid_keys)}. "
            f"Valid keys are: {sorted(valid_keys)}."
        )

    with _lock:
        current = copy.deepcopy(_global_config)
        for key, value in kwargs.items():
            setattr(current, key, value)
        current._validate()
        _global_config = current


def reset_config() -> None:
    """Reset the global configuration to default values.

    Examples
    --------
    >>> from notfallmedizin.core.config import reset_config, get_config
    >>> reset_config()
    >>> get_config().n_jobs
    1
    """
    global _global_config
    with _lock:
        _global_config = NotfallmedizinConfig()


@contextmanager
def config_context(**kwargs: Any) -> Generator[NotfallmedizinConfig, None, None]:
    """Context manager for temporary configuration overrides.

    The configuration is restored to its previous state when the context
    manager exits, even if an exception is raised.

    Parameters
    ----------
    **kwargs
        Keyword arguments corresponding to fields of
        :class:`NotfallmedizinConfig`.

    Yields
    ------
    NotfallmedizinConfig
        The temporarily modified configuration object.

    Raises
    ------
    ConfigurationError
        If an unrecognized configuration key is provided.

    Examples
    --------
    >>> from notfallmedizin.core.config import config_context, get_config
    >>> with config_context(n_jobs=-1, verbose=True) as cfg:
    ...     assert cfg.n_jobs == -1
    >>> get_config().n_jobs  # restored to previous value
    1
    """
    global _global_config
    with _lock:
        saved = copy.deepcopy(_global_config)
    try:
        set_config(**kwargs)
        yield get_config()
    finally:
        with _lock:
            _global_config = saved
