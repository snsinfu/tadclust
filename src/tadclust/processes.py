"""Process start-method selection for parallel work."""

from __future__ import annotations

import multiprocessing as mp
import signal
from multiprocessing import forkserver
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from multiprocessing.context import ForkServerContext, SpawnContext

_PREPARED = False


def prepare_process_context() -> None:
    """Start the forkserver with SIGINT ignored and `__main__` preloaded.

    The server is a clean, exec'd process, so it does not inherit the parent's
    open HDF5 handles. Starting it while SIGINT is ignored means its inherited
    disposition is SIG_IGN: the server can preload `__main__` without an
    import-time KeyboardInterrupt, and workers reset SIGINT to SIG_IGN before
    importing anything. The parent restores its handler immediately after the
    server is launched; preloading happens asynchronously in the server.
    """
    global _PREPARED

    if _PREPARED or "forkserver" not in mp.get_all_start_methods():
        return

    mp.set_forkserver_preload(["__main__"])
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        forkserver.ensure_running()
    finally:
        signal.signal(signal.SIGINT, previous)
    _PREPARED = True


def process_context() -> ForkServerContext | SpawnContext:
    """Return the process context used for CPU-bound workers.

    Prefer forkserver over plain fork: the server is exec'd, so it does not
    inherit the parent's open HDF5 handles. Prefer it over spawn: workers are
    forked from the preloaded server, so they do not re-import the main module
    and dependencies. Fall back to spawn where forkserver is unavailable.
    """
    if "forkserver" in mp.get_all_start_methods():
        prepare_process_context()
        return cast("ForkServerContext", mp.get_context("forkserver"))
    return cast("SpawnContext", mp.get_context("spawn"))
