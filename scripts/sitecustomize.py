"""Fail-closed network guard loaded by every release subprocess."""
import json, os, pathlib, socket
def _blocked(*args, **kwargs):
    path = os.environ.get("OFFLINE_NETWORK_LOG")
    if path:
        with pathlib.Path(path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"operation":"socket","args":repr(args[:2])}) + "\n")
    raise RuntimeError("network access is disabled by offline release")
socket.create_connection = _blocked
socket.getaddrinfo = _blocked
socket.socket.connect = _blocked
socket.socket.connect_ex = _blocked
