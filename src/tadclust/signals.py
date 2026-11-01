"""Signal handling shared by the CLI and worker processes."""

from __future__ import annotations

import os
import signal
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

    Call only after exception unwinding has completed all cleanup.
    """
    if os.name == "posix":
        signal.signal(signal.SIGINT, signal.SIG_DFL)
        os.kill(os.getpid(), signal.SIGINT)
    raise SystemExit(INTERRUPT_EXIT_CODE)
