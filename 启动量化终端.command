#!/bin/sh
set -eu
cd "$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
url="http://127.0.0.1:${PORT:-8766}"
printf '%s\n' 'QuantFlow 量化研究终端' "浏览器地址：$url" ''
health=$(curl -fsS --max-time 1 "$url/api/health" 2>/dev/null || true)
case "$health" in
  *'"status": "ok"'*'"mode": "local_research"'*) exec open "$url" ;;
esac
exec ./start_demo.sh --open-browser
