"""One process at a time may open the USB camera.

The web body, the see pipeline, and the ear all want ``/dev/video0``.
The lock is a file so it works across those processes.
"""

from __future__ import annotations

import fcntl
import os
from contextlib import contextmanager
from typing import Iterator

LOCK_PATH = "/tmp/bruhos-camera.lock"


@contextmanager
def camera_lock(*, blocking: bool = True) -> Iterator[None]:
    fd = os.open(LOCK_PATH, os.O_CREAT | os.O_RDWR, 0o644)
    flags = fcntl.LOCK_EX if blocking else fcntl.LOCK_EX | fcntl.LOCK_NB
    try:
        fcntl.flock(fd, flags)
    except BlockingIOError:
        os.close(fd)
        raise
    try:
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
