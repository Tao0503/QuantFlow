"""Local interview demo: JSON API and built React frontend on one origin."""
from __future__ import annotations

import argparse
import csv
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import math
import mimetypes
from pathlib import Path
import re
import sys
import traceback
from urllib.parse import unquote, urlsplit, parse_qs


ROOT = Path(__file__).resolve().parents[1]
MAX_BODY = 1024 * 1024


class DemoHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    server_version = "QuantWorkbench/1.0"

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.log_date_time_string()}] {fmt % args}\n")

    def _headers(self, status, content_type, length):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(length))
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Cache-Control", "no-store" if self.path.startswith("/api/") else "no-cache")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; font-src 'self'; connect-src 'self'; frame-ancestors 'none'")

    def _json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str).encode("utf-8")
        self._headers(status, "application/json; charset=utf-8", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _check_origin(self):
        host = self.headers.get("Host", "")
        allowed_hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if host not in allowed_hosts:
            self._json(403, {"error": "Only this local application host is allowed"})
            return False
        if self.command not in ("GET", "HEAD"):
            origin = self.headers.get("Origin")
            if origin:
                parsed = urlsplit(origin)
                if parsed.scheme != "http" or parsed.netloc not in allowed_hosts:
                    self._json(403, {"error": "Cross-origin writes are not allowed"})
                    return False
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                self._json(403, {"error": "Cross-site writes are not allowed"})
                return False
        return True

    def _body(self):
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError:
            raise ValueError("Invalid Content-Length")
        if length < 0 or length > MAX_BODY:
            raise OverflowError("JSON request body exceeds 1 MiB")
        if not length:
            return {}
        if self.headers.get("Content-Type", "").split(";")[0].strip() != "application/json":
            raise TypeError("Content-Type must be application/json")
        raw = self.rfile.read(length)

        def invalid_constant(value):
            raise ValueError(f"Nonfinite JSON number: {value}")

        def finite_float(value):
            result = float(value)
            if not math.isfinite(result):
                raise ValueError("JSON numbers must be finite")
            return result

        try:
            value = json.loads(raw, parse_constant=invalid_constant, parse_float=finite_float)
        except (UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid JSON request") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._json(405, {"error": "Deletion is not part of this demo; pause instances instead"})

    def _handle(self):
        if not self._check_origin():
            return
        try:
            path = urlsplit(self.path).path
            if path.startswith("/api/"):
                self._api(path)
            elif self.command == "GET":
                if path.startswith("/artifacts/"):
                    self._static(path.removeprefix("/artifacts/"), ROOT / "results/run_001", spa=False)
                else:
                    self._static(path.removeprefix("/"), self.server.static_dir, spa=True)
            else:
                self._json(405, {"error": "Method not allowed"})
        except OverflowError as exc:
            self._json(413, {"error": str(exc)})
        except TypeError as exc:
            self._json(415, {"error": str(exc)})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except KeyError as exc:
            self._json(404, {"error": str(exc).strip("'")})
        except RuntimeError as exc:
            self._json(409, {"error": str(exc)})
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception:
            traceback.print_exc()
            self._json(500, {"error": "Local service error; see the server log"})

    def _api(self, path):
        service = self.server.service
        method = self.command
        if method == "GET" and path == "/api/health":
            return self._json(200, {"status": "ok", "mode": "local_research", "version": "1.0"})
        if method == "GET" and path == "/api/bootstrap":
            return self._json(200, service.bootstrap())
        if method == "POST" and path == "/api/research-agent":
            from .research_agent import run_research_agent
            return self._json(200, run_research_agent(service, self._body()))
        lists = {"/api/instances": "instances", "/api/optimization/tasks": "tasks",
                 "/api/genomes": "genomes", "/api/backtests": "backtests"}
        if method == "GET" and path in lists:
            return self._json(200, {"items": service.bootstrap()[lists[path]]})
        if method == "POST" and path == "/api/instances":
            return self._json(201, service.create_instance(self._body()))
        match = re.fullmatch(r"/api/instances/([^/]+)", path)
        if method == "PATCH" and match:
            return self._json(200, service.update_instance(unquote(match[1]), self._body()))
        if method == "POST" and path == "/api/optimization/tasks":
            return self._json(202, service.start_optimization(self._body()))
        if method == "POST" and path == "/api/optimization/tuning":
            return self._json(202, service.start_joint_tuning(self._body()))
        match = re.fullmatch(r"/api/optimization/tasks/([^/]+)(/cancel)?", path)
        if match:
            if method == "GET" and not match[2]:
                return self._json(200, service.get_task(unquote(match[1])))
            if method == "POST" and match[2]:
                self._body()
                return self._json(200, service.cancel_task(unquote(match[1])))
        match = re.fullmatch(r"/api/genomes/([^/]+)/promote", path)
        if method == "POST" and match:
            body = self._body()
            return self._json(200, service.promote_genome(unquote(match[1]), body.get("instance_id")))
        if method == "POST" and path == "/api/backtests":
            return self._json(202, service.start_backtest(self._body()))
        match = re.fullmatch(r"/api/backtests/([^/]+)(/export)?", path)
        if method == "GET" and match:
            result = service.get_backtest(unquote(match[1]))
            if match[2]:
                if result.get("status") != "completed":
                    raise RuntimeError("Backtest must finish before export")
                return self._export(result)
            return self._json(200, result)
        match = re.fullmatch(r"/api/replay/([^/]+)", path)
        if method == "GET" and match:
            return self._json(200, service.get_replay(unquote(match[1])))
        self._json(404, {"error": "API endpoint not found"})

    def _export(self, result):
        rows = result.get("trades", [])
        keys = ["date", "signal_date", "quantity", "price", "open_price", "notional", "commission", "slippage_cost", "target_weight", "signal"]
        if result.get("allocations"):
            keys.insert(0, "symbol")
        text = io.StringIO()
        writer = csv.DictWriter(text, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
        data = ("\ufeff" + text.getvalue()).encode("utf-8")
        self._headers(200, "text/csv; charset=utf-8", len(data))
        identifier = re.sub(r"[^A-Za-z0-9_-]", "", str(result.get("id", "backtest")))
        self.send_header("Content-Disposition", f'attachment; filename="trades-{identifier}.csv"')
        self.end_headers()
        self.wfile.write(data)

    def _static(self, relative, root, spa=False):
        root = Path(root).resolve()
        relative = unquote(relative)
        if "\x00" in relative or "\\" in relative or any(x == ".." for x in relative.split("/")):
            return self._json(403, {"error": "Invalid file path"})
        target = (root / (relative or "index.html")).resolve()
        if not target.is_relative_to(root):
            return self._json(403, {"error": "File lies outside the application"})
        if target.is_dir():
            target = (target / "index.html").resolve()
            if not target.is_relative_to(root):
                return self._json(403, {"error": "File lies outside the application"})
        if not target.is_file() and spa and not Path(relative).suffix:
            target = (root / "index.html").resolve()
            if not target.is_relative_to(root):
                return self._json(403, {"error": "File lies outside the application"})
        if not target.is_file():
            return self._json(404, {"error": "File not found; build the frontend before starting the app"})
        data = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self._headers(200, content_type, len(data))
        self.end_headers()
        self.wfile.write(data)


def make_server(host="127.0.0.1", port=0, service=None, static_dir=None):
    if host not in ("127.0.0.1", "localhost"):
        raise ValueError("This local demonstration server binds to loopback only")
    if service is None:
        from .service import DemoService
        service = DemoService(ROOT / "demo/state/workbench.sqlite3", ROOT)
    server = DemoHTTPServer((host, port), Handler)
    server.service = service
    server.static_dir = Path(static_dir) if static_dir else ROOT / "web/dist"
    return server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--db", type=Path, default=ROOT / "demo/state/workbench.sqlite3")
    parser.add_argument("--open-browser", action="store_true", help="Open the local workspace in the default browser")
    args = parser.parse_args()
    from .service import DemoService
    service = DemoService(args.db, ROOT)
    server = make_server(port=args.port, service=service)
    url = f"http://127.0.0.1:{server.server_port}"
    print(f"Quant workbench: {url}\n"
          f"State: {args.db.resolve()}\nMode: historical research and replay", flush=True)
    try:
        if args.open_browser:
            import webbrowser
            if not webbrowser.open(url):
                print(f"在浏览器中打开 {url}", flush=True)
        server.serve_forever(poll_interval=.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        service.close(wait=True)


if __name__ == "__main__":
    main()
