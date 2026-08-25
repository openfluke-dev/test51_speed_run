#!/usr/bin/env bash
# Launch test51 speed-run in Docker with restart: always (resume on crash).
#
# Usage:
#   ./run-docker.sh           # build + up -d
#   ./run-docker.sh --logs    # follow logs
#   ./run-docker.sh --stop
#   ./run-docker.sh --status
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"
PROJECT=test51-speed

# Build context must see sibling tide/ + webgpu/ next to welvet/
ROOT="$(cd "$DIR/../../../../.." && pwd)"
if [[ ! -d "$ROOT/welvet" || ! -d "$ROOT/tide" || ! -d "$ROOT/webgpu" ]]; then
  echo "error: need siblings welvet/ tide/ webgpu/ under $ROOT" >&2
  echo "  (docker build context = parent of welvet)" >&2
  exit 1
fi

cmd="${1:-up}"
case "$cmd" in
  up|"")
    if [[ ! -f .env ]]; then
      cp .env.example .env
      echo "wrote .env from .env.example (SPEED_LAYER=all, SPEED_WORKERS=4)"
    fi
    # Free host ports if a native binary is still holding them.
    if command -v lsof >/dev/null 2>&1; then
      for p in 5151 8080; do
        pids=$(lsof -tiTCP:"$p" -sTCP:LISTEN 2>/dev/null || true)
        if [[ -n "${pids:-}" ]]; then
          echo "killing host listeners on :$p → $pids"
          # shellcheck disable=SC2086
          kill $pids 2>/dev/null || true
          sleep 1
        fi
      done
    fi
    docker compose -p "$PROJECT" up --build -d
    echo
    echo "up · project=$PROJECT  restart=always  resume=true"
    echo "  dash  http://localhost:${SPEED_PORT:-5151}"
    echo "  tide  http://localhost:${TIDE_PORT:-8080}"
    echo "  ckpt  $DIR/speed_ckpt/<layer>/"
    echo "  logs  ./run-docker.sh --logs"
    ;;
  --logs|logs)
    docker compose -p "$PROJECT" logs -f --tail=80
    ;;
  --stop|stop)
    docker compose -p "$PROJECT" down
    ;;
  --status|status)
    docker compose -p "$PROJECT" ps
    echo
    docker compose -p "$PROJECT" logs --tail=30
    ;;
  --restart|restart)
    docker compose -p "$PROJECT" restart
    ;;
  *)
    echo "Usage: $0 [up|--logs|--stop|--status|--restart]" >&2
    exit 2
    ;;
esac
