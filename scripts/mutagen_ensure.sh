#!/usr/bin/env bash
set -euo pipefail

SESSION_NAME="${1:-rl-garden-code}"
TIMEOUT_SEC="${TIMEOUT_SEC:-20}"
SLEEP_SEC=2

is_healthy() {
    local out
    out="$(mutagen sync list "${SESSION_NAME}" 2>/dev/null || true)"
    [[ -n "${out}" ]] \
        && grep -q "Alpha:" <<<"${out}" \
        && grep -q "Beta:" <<<"${out}" \
        && grep -q "Connected: Yes" <<<"${out}" \
        && grep -q "Status: Watching for changes" <<<"${out}"
}

echo "[mutagen] ensuring daemon is running"
mutagen daemon start >/dev/null || true

echo "[mutagen] starting project sessions"
mutagen project start >/dev/null || true

echo "[mutagen] waiting for ${SESSION_NAME} to become healthy"
deadline=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < deadline )); do
    if is_healthy; then
        echo "[mutagen] healthy: ${SESSION_NAME}"
        mutagen sync list "${SESSION_NAME}"
        exit 0
    fi
    sleep "${SLEEP_SEC}"
done

echo "[mutagen] unhealthy after ${TIMEOUT_SEC}s, forcing session cycle"
mutagen sync terminate "${SESSION_NAME}" >/dev/null 2>&1 || true
mutagen project start >/dev/null || true

deadline=$((SECONDS + TIMEOUT_SEC))
while (( SECONDS < deadline )); do
    if is_healthy; then
        echo "[mutagen] recovered: ${SESSION_NAME}"
        mutagen sync list "${SESSION_NAME}"
        exit 0
    fi
    sleep "${SLEEP_SEC}"
done

echo "[mutagen] recovery failed for ${SESSION_NAME}" >&2
mutagen sync list "${SESSION_NAME}" || true
exit 1
