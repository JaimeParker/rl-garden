#!/usr/bin/env bash
# Usage: scripts/test.sh <fast|gpu|sim|slow|docs|linkcheck|all> [extra pytest args]
# Caller sets CUDA_VISIBLE_DEVICES / MUJOCO_GL as needed.
set -euo pipefail
cd "$(dirname "$0")/.."

lane="${1:-}"
[[ -n "$lane" ]] || { echo "usage: $0 <fast|gpu|sim|slow|docs|linkcheck|all> [pytest args]" >&2; exit 2; }
shift

run_lane() {
  local name="$1"; shift
  case "$name" in
    docs) sphinx-build -W -n --keep-going -b html docs docs/_build/html; return $? ;;
    linkcheck) sphinx-build -b linkcheck docs docs/_build/linkcheck; return $? ;;
  esac
  local args=(-q -o cache_dir="${PYTEST_CACHE_DIR:-/tmp/rl_garden_pytest_cache}"
              --timeout="${PYTEST_TIMEOUT:-600}" --durations=25 --maxfail="${PYTEST_MAXFAIL:-20}")
  if [[ -n "${PYTEST_JUNIT_DIR:-}" ]]; then
    mkdir -p "$PYTEST_JUNIT_DIR"
    args+=(--junitxml="$PYTEST_JUNIT_DIR/$name.xml")
  fi
  case "$name" in
    fast) args+=(-m "not gpu and not sim and not slow" -n "${PYTEST_WORKERS:-8}" --dist loadfile) ;;
    gpu)  args+=(-m "gpu and not slow") ;;
    sim)  args+=(-m "sim and not gpu and not slow") ;;
    slow) args+=(-m slow -n "${PYTEST_SLOW_WORKERS:-4}" --dist loadgroup) ;;
  esac
  python -m pytest "${args[@]}" "$@" tests
}

if [[ "$lane" != "all" ]]; then
  case "$lane" in fast|gpu|sim|slow|docs|linkcheck) ;; *) echo "unknown lane: $lane" >&2; exit 2 ;; esac
  run_lane "$lane" "$@"
  exit $?
fi

status=0
summaries=()
for name in fast sim gpu slow docs; do
  log="$(mktemp)"
  set +e
  run_lane "$name" "$@" 2>&1 | tee "$log"
  rc=${PIPESTATUS[0]}
  set -e
  summary="$(grep -E '^(=+ )?[0-9]+ (passed|failed|skipped|error|deselected)|passed|failed|build succeeded|build finished' "$log" | tail -1 || true)"
  summaries+=("[$name] rc=$rc $summary")
  rm -f "$log"
  [[ $rc -eq 0 ]] || status=1
done
echo "==== lane summary ===="
printf '%s\n' "${summaries[@]}"
exit $status
