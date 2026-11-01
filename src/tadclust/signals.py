"""Signal handling shared by the CLI and worker processes."""

from __future__ import annotations

import os
import signal
from contextlib import suppress
from typing import NoReturn

INTERRUPT_EXIT_CODE = 128 + signal.SIGINT


def install_interrupt_handler() -> None:
    """Reinstall CPython's SIGINT handler.

    Importing polars installs a native SIGINT handler from
    _polars_runtime.abi3.so at the OS level, replacing CPython's handler.
    signal.getsignal() still reports default_int_handler, so the change is
    hidden. The native handler does not raise KeyboardInterrupt, and blocking
    waits therefore never abort. Reinstalling the Python handler fixes this.

    See: https://github.com/pola-rs/polars/issues/21739
    """
    signal.signal(signal.SIGINT, signal.default_int_handler)


def ignore_sigint() -> None:
    """Let the parent process own Ctrl-C shutdown.

    Workers must not raise KeyboardInterrupt: multiprocessing would print
    tracebacks and may race with pool teardown.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def exit_by_sigint() -> NoReturn:
    """Terminate by SIGINT so the parent observes WIFSIGNALED.

    Call only after exception unwinding has completed all cleanup. This runs
    multiprocessing finalizers so the resource tracker does not warn about
    leaked semaphores when the process dies by signal.
    """
    if os.name == "posix":
        # Run multiprocessing finalizers before dying by signal. A signal
        # death skips interpreter shutdown, so the resource tracker would
        # otherwise report semaphores as leaked.
        import multiprocessing.util as mp_util

        exit_function = getattr(mp_util, "_exit_function", None)
        if callable(exit_function):
            with suppress(BaseException):
                exit_function()
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    raise SystemExit(INTERRUPT_EXIT_CODE)
