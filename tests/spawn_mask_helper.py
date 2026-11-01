"""Test helper: report the SIGINT mask inherited by a spawn child."""

from __future__ import annotations

import multiprocessing as mp
import signal

from tadclust.signals import block_sigint_during_spawn


def child() -> None:
    mask = signal.pthread_sigmask(signal.SIG_BLOCK, [])
    print("blocked:", sorted(int(sig) for sig in mask), flush=True)


def main() -> None:
    context = mp.get_context("spawn")
    with block_sigint_during_spawn():
        process = context.Process(target=child)
        process.start()
    process.join()


if __name__ == "__main__":
    main()
