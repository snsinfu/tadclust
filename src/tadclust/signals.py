"""Signal handling shared by the CLI and worker processes."""

from __future__ import annotations

import signal


def ignore_sigint() -> None:
    """Let the parent process own Ctrl-C shutdown.

    Workers must not raise KeyboardInterrupt: multiprocessing would print
    tracebacks and may race with pool teardown.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)
