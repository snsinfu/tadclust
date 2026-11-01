"""Test helper: report the SIGINT disposition in a worker process."""

from __future__ import annotations

import signal

from tadclust.processes import process_context


def child() -> None:
    print("ignored:", signal.getsignal(signal.SIGINT) is signal.SIG_IGN, flush=True)


def main() -> None:
    context = process_context()
    process = context.Process(target=child)
    process.start()
    process.join()


if __name__ == "__main__":
    main()
