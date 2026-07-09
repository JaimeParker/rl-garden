"""Actor<->learner HTTP sync layer for real-robot training.

Self-written, stdlib-only (``http.server`` / ``urllib.request``) -- SERL uses
an external ``agentlace`` package for this, but it is a pip dependency not
vendored anywhere in this repo, so it isn't reused here. Transitions flow
actor -> learner; policy parameters flow learner -> actor. Neither side may
block the other's real-time loop: ``LearnerSyncServer`` accepts a transition
POST and hands it off to a callback immediately, and ``ActorSyncClient``
queues transitions and polls for params on a background thread so the
actor's fixed-frequency control loop never waits on network I/O.
"""
from __future__ import annotations

import io
import queue
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

import torch


class LearnerSyncServer:
    """Runs on the learner process.

    Receives transitions via POST /transition (each call synchronously
    invokes ``on_transition`` on the server's request-handling thread --
    callers that mutate shared state from it are responsible for their own
    locking). Serves the latest published policy params via
    GET /policy_params, versioned so a client only downloads a payload when
    the version differs from the one it already has cached.
    """

    def __init__(
        self,
        host: str,
        port: int,
        on_transition: Callable[[dict[str, Any]], None],
    ) -> None:
        self._on_transition = on_transition
        self._params_lock = threading.Lock()
        self._params_bytes: Optional[bytes] = None
        self._params_version = 0

        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def server_address(self) -> tuple[str, int]:
        return self._httpd.server_address

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=5.0)

    def publish_params(self, state_dict: dict[str, Any]) -> None:
        buffer = io.BytesIO()
        torch.save(state_dict, buffer)
        with self._params_lock:
            self._params_bytes = buffer.getvalue()
            self._params_version += 1

    def _latest_params(self) -> tuple[int, Optional[bytes]]:
        with self._params_lock:
            return self._params_version, self._params_bytes


def _make_handler(server: LearnerSyncServer):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            del format, args  # silence stdlib's default stderr access log

        def do_POST(self) -> None:
            if self.path != "/transition":
                self.send_response(404)
                self.end_headers()
                return
            length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(length)
            transition = torch.load(io.BytesIO(body), weights_only=False)
            server._on_transition(transition)
            self.send_response(204)
            self.end_headers()

        def do_GET(self) -> None:
            if not self.path.startswith("/policy_params"):
                self.send_response(404)
                self.end_headers()
                return
            client_version = -1
            if "?" in self.path:
                query = self.path.split("?", 1)[1]
                params = dict(p.split("=", 1) for p in query.split("&") if "=" in p)
                if "version" in params:
                    client_version = int(params["version"])
            version, payload = server._latest_params()
            if payload is None or version == client_version:
                self.send_response(204)
                self.send_header("X-Params-Version", str(version))
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("X-Params-Version", str(version))
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    return Handler


class ActorSyncClient:
    """Runs on the actor process.

    ``push_transition`` and ``latest_policy_params`` are both non-blocking:
    transitions are queued and drained by a background thread that owns the
    actual HTTP round trip, and policy params are polled on a fixed interval
    by that same background thread and cached locally -- the actor's control
    loop only ever reads the cache, it never makes a network call itself.
    This is required for the actor to hold its fixed control frequency: a
    slow learner (e.g. mid gradient-step) must never stall robot control.
    """

    def __init__(
        self,
        learner_url: str,
        poll_interval: float = 1.0,
        timeout: float = 5.0,
    ) -> None:
        self._base_url = learner_url.rstrip("/")
        self._timeout = timeout
        self._poll_interval = poll_interval

        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._params_lock = threading.Lock()
        self._cached_version = -1
        self._cached_params: Optional[dict[str, Any]] = None

        self._stop_event = threading.Event()
        self._push_thread = threading.Thread(target=self._push_worker, daemon=True)
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)

    def start(self) -> None:
        self._push_thread.start()
        self._poll_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)  # unblock the push worker's queue.get()
        self._push_thread.join(timeout=5.0)
        self._poll_thread.join(timeout=5.0)

    def push_transition(self, transition: dict[str, Any]) -> None:
        self._queue.put(transition)

    def latest_policy_params(self) -> Optional[dict[str, Any]]:
        """Returns the most recently cached params, or ``None`` before the
        first successful poll. Never blocks or makes a network call."""
        with self._params_lock:
            return self._cached_params

    def _push_worker(self) -> None:
        while not self._stop_event.is_set():
            transition = self._queue.get()
            if transition is None:
                continue
            try:
                buffer = io.BytesIO()
                torch.save(transition, buffer)
                req = urllib.request.Request(
                    f"{self._base_url}/transition",
                    data=buffer.getvalue(),
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                    resp.read()
            except urllib.error.URLError:
                pass  # dropped transition; robot control must not stall on this

    def _poll_worker(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            self._poll_once()

    def _poll_once(self) -> None:
        try:
            req = urllib.request.Request(
                f"{self._base_url}/policy_params?version={self._cached_version}"
            )
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                version = int(resp.headers.get("X-Params-Version", self._cached_version))
                if resp.status == 204:
                    self._cached_version = version
                    return
                body = resp.read()
                params = torch.load(io.BytesIO(body), weights_only=False)
            with self._params_lock:
                self._cached_version = version
                self._cached_params = params
        except urllib.error.URLError:
            pass  # keep the previously cached params
