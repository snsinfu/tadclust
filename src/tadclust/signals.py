"""Signal handling shared by the CLI and worker processes."""

from __future__ import annotations

import os
import signal
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import NoReturn

INTERRUPT_EXIT_CODE = 128 + signal.SIGINT


@contextmanager
def block_sigint_during_spawn() -> Iterator[None]:
    """Block SIGINT while spawn children start.

    POSIX inherits the calling thread's signal mask across exec. Spawn children
    import the main module and dependencies before their initializer runs. If
    SIGINT arrived during those imports, the child would print a traceback.
    Blocking it here keeps the signal pending in the parent and inherited as
    blocked by the children. The parent's mask is restored afterwards, so a
    pending interrupt is raised then.
    """
    if hasattr(signal, "pthread_sigmask"):
        previous = signal.pthread_sigmask(signal.SIG_BLOCK, {signal.SIGINT})
        try:
            yield
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous)
    else:
        previous_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            yield
        finally:
            signal.signal(signal.SIGINT, previous_handler)


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
