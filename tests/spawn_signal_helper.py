"""Test helper: report the SIGINT disposition inherited by a spawn child."""

from __future__ import annotations

import multiprocessing as mp
import signal

from tadclust.signals import ignore_sigint_during_spawn


def child() -> None:
    print("ignored:", signal.getsignal(signal.SIGINT) is signal.SIG_IGN, flush=True)


def main() -> None:
    context = mp.get_context("spawn")
    with ignore_sigint_during_spawn():
        process = context.Process(target=child)
        process.start()
    process.join()


if __name__ == "__main__":
    main()
