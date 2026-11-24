#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
if [ ! -f web/dist/index.html ]; then
  printf '%s\n' '先构建界面：cd web && npm ci && npm run build'
  exit 1
fi
if [ ! -f data/manifest.json ] || [ ! -f results/run_001/selection.json ]; then
  printf '%s\n' '请先执行 .venv/bin/python -m demo.bootstrap_public 初始化本地演示数据。'
  exit 1
fi
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m demo.server --port "${PORT:-8766}" "$@"
fi
exec python3 -m demo.server --port "${PORT:-8766}" "$@"
