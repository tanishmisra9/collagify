#!/usr/bin/env python3
"""Collagify GUI: a local web app for building and fine-tuning collages.

    python gui.py              # opens http://127.0.0.1:<port> in your browser
    python gui.py --port 8765 --no-browser

Binds to localhost only. Nothing leaves your machine.
"""
import argparse
import threading
import webbrowser

from gui.server import make_server


def main():
    p = argparse.ArgumentParser(description="Collagify GUI")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--out-dir", default="output", help="Where Export saves (default: output)")
    p.add_argument("--no-browser", action="store_true")
    a = p.parse_args()
    srv, session = make_server(a.port, a.out_dir)
    url = f"http://127.0.0.1:{a.port}/"
    print(f"Collagify GUI running at {url}  (Ctrl-C to stop)")
    if not a.no_browser:
        threading.Timer(0.4, webbrowser.open, args=(url,)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
        session.close()


if __name__ == "__main__":
    main()
