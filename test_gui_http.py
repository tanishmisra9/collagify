"""HTTP-level checks for the GUI server. Run: python test_gui_http.py

Self-contained: synthetic photos, no AI models, no pics/ folder, temp dirs cleaned up.
It drives the real server over a real keep-alive connection, which is the layer
test_gui_core.py never touches.
"""
import http.client
import io
import json
import os
import re
import socket
import tempfile
import threading
import time
from pathlib import Path

from PIL import Image

import collage
from gui.server import make_server


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def photo_bytes(w, h, n, fmt="JPEG"):
    noise = Image.effect_noise((w, h), 70).convert("RGB")
    tint = Image.new("RGB", (w, h), ((n * 53) % 200 + 30, (n * 97) % 200 + 30, (n * 31) % 200 + 30))
    buf = io.BytesIO()
    Image.blend(noise, tint, 0.6).save(buf, fmt)
    return buf.getvalue()


class Client:
    def __init__(self, port, token=None):
        self.port, self.token = port, token
        self.c = http.client.HTTPConnection("127.0.0.1", port, timeout=90)   # ONE connection, reused

    def req(self, method, path, body=None, headers=None, token=True):
        h = dict(headers or {})
        if token and self.token:
            h["X-Token"] = self.token
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        self.c.request(method, path, body=body, headers=h)
        r = self.c.getresponse()
        data = r.read()
        ctype = r.getheader("Content-Type", "")
        return r.status, (json.loads(data) if ctype.startswith("application/json") else data), r

    def upload(self, name, data, modified_ms=None):
        h = {"X-Filename": name, "Content-Type": "application/octet-stream"}
        if modified_ms is not None:
            h["X-Modified"] = str(modified_ms)
        return self.req("POST", "/api/photos", data, h)[:2]


def wait_done(cl, timeout=90):
    end = time.time() + timeout
    while time.time() < end:
        st = cl.req("GET", "/api/status")[1]["status"]
        if st["state"] != "running":
            return st
        time.sleep(0.1)
    raise AssertionError("generate did not finish")


def main():
    port, out = free_port(), tempfile.mkdtemp(prefix="collagify-out-")
    srv, session = make_server(port, out)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        cl = Client(port)
        page = cl.req("GET", "/", token=False)[1].decode()
        cl.token = re.search(r'collagify-token" content="([^"]+)"', page).group(1)
        run(port, cl, session, Path(out))
    finally:
        srv.shutdown()
        session.close()
    print("all http checks passed")


def run(port, cl, session, out):
    # -- auth, host check, static whitelist -------------------------------
    assert cl.req("POST", "/api/reset", {}, token=False)[0] == 403, "POST without token must be refused"
    assert cl.req("POST", "/api/reset", {}, {"X-Token": "nope"}, token=False)[0] == 403
    assert Client(port).req("GET", "/api/state", headers={"Host": "evil.test"})[0] == 403, "bad Host must be refused"
    assert cl.req("GET", "/gui/server.py")[0] == 404 and cl.req("GET", "/fonts/../server.py")[0] == 404
    st, data, r = cl.req("GET", "/fonts/instrument-sans.woff2")
    assert st == 200 and r.getheader("Content-Type") == "font/woff2" and data[:4] == b"wOF2", "self-hosted font served"
    assert b"fonts.googleapis.com" not in cl.req("GET", "/")[1], "no third-party font requests"
    assert cl.req("GET", "/api/schema")[1]["max_photos"] == collage.MAX_PHOTOS

    # -- uploads: real filenames, dates, duplicates, rejects -----------------
    day = 86400 * 1000
    t0 = 1_600_000_000_000
    assert cl.upload("b.jpg", photo_bytes(400, 300, 1), t0 + 2 * day)[0] == 200
    assert cl.upload("a.jpg", photo_bytes(300, 400, 2), t0 + 1 * day)[0] == 200
    st, js = cl.upload("a.jpg", photo_bytes(500, 500, 3), t0 + 3 * day)    # same filename, different photo
    assert st == 200 and len(js["photos"]) == 3 and len({p["id"] for p in js["photos"]}) == 3
    for p in session.photos:                      # name/date orders sort on these paths
        assert Path(p["path"]).name == p["name"], "upload path must keep the real filename"
    mt = {Path(p["path"]).stat().st_mtime for p in session.photos}
    assert mt == {(t0 + k * day) / 1000 for k in (1, 2, 3)}, "browser lastModified must become the file date"
    st, js = cl.upload("x.heic", b"junk")
    assert st == 400 and "Export it as JPEG" in js["error"]
    st, js = cl.upload("fake.jpg", b"not an image")
    assert st == 400 and len(session.photos) == 3, "an unreadable upload must not register"
    assert cl.req("GET", "/api/state")[0] == 200, "connection survives rejected uploads"

    # -- generate, then every body-ignoring POST on ONE keep-alive connection -
    settings = {"width": 360, "ratio": "9:16", "color_strength": 0, "cutouts": False, "faces": False, "seed": 5}
    assert cl.req("POST", "/api/generate", settings)[0] == 200
    st = wait_done(cl)
    assert st["state"] == "done" and not st.get("warning"), st
    plan = cl.req("GET", "/api/state")[1]["plan"]
    ids = {p["id"] for p in session.photos}
    assert {it["photo"] for it in plan["items"]} == ids, "each plan item maps to its own photo, even with equal filenames"

    steps = [
        ("POST", "/api/placement", {"id": 0, "dx": 6, "dy": 4, "panx": 0, "pany": 0}),
        ("POST", "/api/undo", {}),                # regression: body {} was left unread and broke the next request
        ("POST", "/api/redo", {}),
        ("POST", "/api/reset", {}),
        ("GET", "/api/preview.jpg?scale=0.4&id=0&dx=3&dy=3&panx=0&pany=0", None),
        ("POST", "/api/export", {"quality": 80}),
    ]
    for method, path, body in steps:
        st, data, _ = cl.req(method, path, body)
        assert st == 200, f"{method} {path} -> {st} {data!r}"
        nxt = cl.req("GET", "/api/state")           # same socket: fails with 501 if the body desynced it
        assert nxt[0] == 200, f"request after {path} -> {nxt[0]}"
    name = json.loads(json.dumps(cl.req("POST", "/api/export", {})[1]))["name"]
    assert cl.req("GET", f"/api/download/{name}")[1][:2] == b"\xff\xd8" and (out / name).exists()

    # -- error paths keep the connection usable ---------------------------------
    assert cl.req("POST", "/api/placement", {"id": 99, "dx": 0, "dy": 0})[0] == 404
    assert cl.req("POST", "/api/generate", {"ratio": "banana"})[0] == 400
    assert cl.req("GET", "/api/state")[0] == 200

    # -- the photo cap holds under concurrent uploads ---------------------------
    for n in (4, 5):
        assert cl.upload(f"c{n}.jpg", photo_bytes(300, 300, n))[0] == 200
    assert len(session.photos) == 5
    results = []

    def racer(i):
        c = Client(port, cl.token)
        results.append(c.upload(f"r{i}.jpg", photo_bytes(300, 300, 10 + i))[0])

    threads = [threading.Thread(target=racer, args=(i,)) for i in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sorted(results) == [200, 400, 400, 400], results
    assert len(session.photos) == collage.MAX_PHOTOS, "cap must hold when uploads race"
    st, js = cl.upload("one-more.jpg", photo_bytes(300, 300, 99))
    assert st == 400 and "at most" in js["error"]

    # -- a degraded analysis warns and is never cached --------------------------
    real = (collage.detect_faces, collage.face_detector_failed, collage.retry_face_detector)
    collage.detect_faces = lambda img, *a, **k: []
    collage.retry_face_detector = lambda: None
    try:
        faces_on = dict(settings, faces=True, seed=6)
        collage.face_detector_failed = lambda: True
        assert cl.req("POST", "/api/generate", faces_on)[0] == 200
        st = wait_done(cl)
        assert st["state"] == "done" and "face detection" in st["warning"].lower(), st
        assert session._analysis is None, "a degraded analysis must not be cached"
        collage.face_detector_failed = lambda: False
        assert cl.req("POST", "/api/generate", faces_on)[0] == 200
        st = wait_done(cl)
        assert st["state"] == "done" and not st.get("warning") and session._analysis is not None, st
    finally:
        collage.detect_faces, collage.face_detector_failed, collage.retry_face_detector = real


if __name__ == "__main__":
    main()
