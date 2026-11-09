"""HTTP layer only: routing, auth, JSON in/out. All logic lives in Session."""
import json
import secrets
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import collage
from gui import schema
from gui.session import MAX_UPLOAD, Busy, Session

STATIC = Path(__file__).parent / "static"
FILES = {  # explicit whitelist: nothing else under static/ is ever served
    "/": ("index.html", "text/html; charset=utf-8"),
    "/app.css": ("app.css", "text/css; charset=utf-8"),
    "/js/app.js": ("js/app.js", "text/javascript; charset=utf-8"),
    "/js/controls.js": ("js/controls.js", "text/javascript; charset=utf-8"),
    "/js/stage.js": ("js/stage.js", "text/javascript; charset=utf-8"),
}
FILES.update({f"/fonts/{f.name}": (f"fonts/{f.name}", "font/woff2") for f in (STATIC / "fonts").glob("*.woff2")})


def make_server(port, out_dir):
    session = Session(out_dir)
    token = secrets.token_urlsafe(16)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *a):  # quiet; the CLI prints its own progress
            pass

        # -- helpers --
        def _send(self, code, body, ctype, extra=None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            if self.close_connection:
                self.send_header("Connection", "close")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._send(code, json.dumps(obj).encode(), "application/json")

        def _err(self, code, msg):
            self.close_connection = True   # the request body may be unread; don't reuse the socket
            self._json({"error": msg}, code)

        def _host_ok(self):
            return self.headers.get("Host", "") in (f"127.0.0.1:{port}", f"localhost:{port}")

        def _body(self, limit=1 << 20):
            n = int(self.headers.get("Content-Length") or 0)
            if n > limit:
                raise ValueError("Request too large.")
            return self.rfile.read(n) if n else b""

        def _args(self):
            return json.loads(self._raw) if self._raw else {}

        def _dispatch(self, method):
            if not self._host_ok():
                return self._err(403, "Bad host.")
            if method != "GET" and self.headers.get("X-Token") != token:
                return self._err(403, "Missing or wrong token.")
            url = urlparse(self.path)
            try:
                # Read the body up front for every route except the upload (which streams its own),
                # so a route that ignores its body can never leave bytes that desync keep-alive.
                self._raw = b"" if url.path == "/api/photos" else self._body()
                return self._route(method, url.path, parse_qs(url.query))
            except Busy:
                self._err(409, "A collage is already being generated.")
            except KeyError:
                self._err(404, "Not found.")
            except (ValueError, json.JSONDecodeError) as e:
                self._err(400, str(e))
            except Exception as e:  # last resort: report instead of dropping the connection
                self._err(500, f"{type(e).__name__}: {e}")

        def do_GET(self):
            self._dispatch("GET")

        def do_POST(self):
            self._dispatch("POST")

        def do_DELETE(self):
            self._dispatch("DELETE")

        # -- routes --
        def _route(self, m, path, q):
            s = session
            if m == "GET" and path in FILES:
                name, ctype = FILES[path]
                data = (STATIC / name).read_bytes()
                if path == "/":
                    data = data.replace(b"__TOKEN__", token.encode())
                return self._send(200, data, ctype)
            if m == "GET" and path == "/api/schema":
                return self._json({"settings": schema.SETTINGS, "max_photos": collage.MAX_PHOTOS})
            if m == "GET" and path == "/api/state":
                return self._json({"photos": s.photo_view(), "status": s.status, "plan": s.plan_view(),
                                   "settings": s.settings})
            if m == "GET" and path == "/api/status":
                return self._json({"status": s.status, "rev": s.rev})
            if m == "GET" and path.startswith("/api/photos/") and path.endswith("/thumb.jpg"):
                return self._send(200, s.thumbs[path.split("/")[3]], "image/jpeg",
                                  {"Cache-Control": "max-age=3600"})
            if m == "GET" and path == "/api/preview.jpg":
                scale = max(0.1, min(1.0, float(q.get("scale", ["1"])[0])))
                ov = None
                if "id" in q:
                    ov = (int(q["id"][0]),) + tuple(float(q[k][0]) for k in ("dx", "dy", "panx", "pany"))
                return self._send(200, s.jpeg(scale, ov), "image/jpeg")
            if m == "GET" and path.startswith("/api/download/"):
                p = s.exports[unquote(path.split("/")[3])]
                return self._send(200, p.read_bytes(), "image/jpeg",
                                  {"Content-Disposition": f'attachment; filename="{p.name}"'})

            if m == "POST" and path == "/api/photos":
                n = int(self.headers.get("Content-Length") or 0)
                if n > MAX_UPLOAD:
                    raise ValueError("That file is over 40 MB.")
                name = unquote(self.headers.get("X-Filename", "photo"))
                s.add_upload(name, self.rfile.read(n), self.headers.get("X-Modified"))
                return self._json({"photos": s.photo_view()})
            if m == "POST" and path == "/api/photos/folder":
                s.add_folder(self._args().get("path", ""))
                return self._json({"photos": s.photo_view()})
            if m == "DELETE" and path.startswith("/api/photos/"):
                s.remove(path.split("/")[3])
                return self._json({"photos": s.photo_view()})
            if m == "POST" and path == "/api/generate":
                s.start_generate(self._args())
                return self._json({"ok": True})
            if m == "POST" and path == "/api/placement":
                a = self._args()
                s.set_offset(int(a["id"]), float(a["dx"]), float(a["dy"]),
                             float(a.get("panx", 0)), float(a.get("pany", 0)), bool(a.get("merge")))
                return self._json({"plan": s.plan_view()})
            if m == "POST" and path in ("/api/undo", "/api/redo"):
                s.step(-1 if path.endswith("undo") else 1)
                return self._json({"plan": s.plan_view()})
            if m == "POST" and path == "/api/reset":
                a = self._args()
                s.reset(None if a.get("id") is None else int(a["id"]))
                return self._json({"plan": s.plan_view()})
            if m == "POST" and path == "/api/export":
                p = s.export(self._args().get("quality", 95))
                return self._json({"name": p.name, "path": str(p), "url": f"/api/download/{p.name}"})
            raise KeyError(path)

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    return srv, session
