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
import time
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
        self._stats_lock = threading.Lock()
        self._transition_posts = 0
        self._policy_param_gets = 0
        self._last_transition_post_ts: Optional[float] = None
        self._last_policy_param_get_ts: Optional[float] = None

        handler = _make_handler(self)
        self._httpd = ThreadingHTTPServer((host, port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)

    @property
    def server_address(self) -> tuple[str, int]:
        return self._httpd.server_address

    @property
    def published_version(self) -> int:
        with self._params_lock:
            return self._params_version

    def connection_stats(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "transition_posts": self._transition_posts,
                "policy_param_gets": self._policy_param_gets,
                "last_transition_post_ts": self._last_transition_post_ts,
                "last_policy_param_get_ts": self._last_policy_param_get_ts,
            }

    def start(self) -> None:
        self._thread.start()
        host, port = self.server_address
        print(f"[sync] learner listening on {host}:{port}", flush=True)

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

    def _record_transition_post(self) -> None:
        with self._stats_lock:
            self._transition_posts += 1
            self._last_transition_post_ts = time.monotonic()

    def _record_policy_param_get(self) -> None:
        with self._stats_lock:
            self._policy_param_gets += 1
            self._last_policy_param_get_ts = time.monotonic()


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
            server._record_transition_post()
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
            server._record_policy_param_get()
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
        monitor_interval: float = 5.0,
    ) -> None:
        self._base_url = learner_url.rstrip("/")
        self._timeout = timeout
        self._poll_interval = poll_interval
        self._monitor_interval = monitor_interval

        self._queue: "queue.Queue[dict[str, Any]]" = queue.Queue()
        self._params_lock = threading.Lock()
        self._cached_version = -1
        self._cached_params: Optional[dict[str, Any]] = None
        self._stats_lock = threading.Lock()
        self._posted_transitions = 0
        self._failed_transition_posts = 0
        self._policy_param_updates = 0
        self._policy_poll_noops = 0
        self._failed_policy_polls = 0
        self._last_transition_post_ts: Optional[float] = None
        self._last_policy_poll_ts: Optional[float] = None
        self._last_policy_update_ts: Optional[float] = None

        self._stop_event = threading.Event()
        self._push_thread = threading.Thread(target=self._push_worker, daemon=True)
        self._poll_thread = threading.Thread(target=self._poll_worker, daemon=True)
        self._monitor_thread = (
            threading.Thread(target=self._monitor_worker, daemon=True)
            if self._monitor_interval > 0
            else None
        )
        self._logged_first_transition_post = False
        self._logged_first_policy_params = False

    def start(self) -> None:
        self._push_thread.start()
        self._poll_thread.start()
        if self._monitor_thread is not None:
            self._monitor_thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._queue.put(None)  # unblock the push worker's queue.get()
        self._push_thread.join(timeout=5.0)
        self._poll_thread.join(timeout=5.0)
        if self._monitor_thread is not None:
            self._monitor_thread.join(timeout=5.0)

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
                with self._stats_lock:
                    self._posted_transitions += 1
                    self._last_transition_post_ts = time.monotonic()
                if not self._logged_first_transition_post:
                    print(
                        f"[sync] actor posted first transition to {self._base_url}",
                        flush=True,
                    )
                    self._logged_first_transition_post = True
            except urllib.error.URLError:
                with self._stats_lock:
                    self._failed_transition_posts += 1
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
                    with self._stats_lock:
                        self._policy_poll_noops += 1
                        self._last_policy_poll_ts = time.monotonic()
                    return
                body = resp.read()
                params = torch.load(io.BytesIO(body), weights_only=False)
            with self._params_lock:
                self._cached_version = version
                self._cached_params = params
            with self._stats_lock:
                self._policy_param_updates += 1
                self._last_policy_poll_ts = time.monotonic()
                self._last_policy_update_ts = self._last_policy_poll_ts
            if not self._logged_first_policy_params:
                print(
                    f"[sync] actor received first policy params from {self._base_url} "
                    f"(version={version})",
                    flush=True,
                )
                self._logged_first_policy_params = True
        except urllib.error.URLError:
            with self._stats_lock:
                self._failed_policy_polls += 1
            pass  # keep the previously cached params

    def _monitor_worker(self) -> None:
        while not self._stop_event.wait(self._monitor_interval):
            now = time.monotonic()
            with self._stats_lock:
                posted = self._posted_transitions
                post_failures = self._failed_transition_posts
                policy_updates = self._policy_param_updates
                policy_noops = self._policy_poll_noops
                policy_failures = self._failed_policy_polls
                last_post = self._last_transition_post_ts
                last_poll = self._last_policy_poll_ts
                last_update = self._last_policy_update_ts
            print(
                "[sync] actor link "
                f"url={self._base_url} "
                f"queue={self._queue.qsize()} "
                f"posted={posted} post_failures={post_failures} "
                f"policy_updates={policy_updates} policy_noops={policy_noops} "
                f"policy_failures={policy_failures} "
                f"last_post={self._format_age(last_post, now)} "
                f"last_policy_poll={self._format_age(last_poll, now)} "
                f"last_policy_update={self._format_age(last_update, now)}",
                flush=True,
            )

    @staticmethod
    def _format_age(timestamp: Optional[float], now: float) -> str:
        if timestamp is None:
            return "never"
        return f"{max(0.0, now - timestamp):.1f}s_ago"
