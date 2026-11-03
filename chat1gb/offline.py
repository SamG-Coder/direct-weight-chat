"""Reject outgoing Python socket connections; no downloads or network clients."""
import os
import sys

_installed = False


def install():
    global _installed
    if _installed:
        return
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["WANDB_MODE"] = "disabled"

    def guard(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr"):
            raise PermissionError("This experiment is offline; network access is disabled")

    sys.addaudithook(guard)
    _installed = True


install()
