"""可选的本地静态服务器（只用标准库）。

双击 index.html 直接打开即可（数据通过 <script> 加载，没有 fetch 请求），
这个脚本只是为了让地址栏看起来更像一个网站，或在某些浏览器策略下更稳定：

    python webapp/serve.py            # 然后访问 http://127.0.0.1:8000/
    python webapp/serve.py --port 8123 --no-browser
"""
from __future__ import annotations

import argparse
import http.server
import socketserver
import threading
import webbrowser
from functools import partial
from pathlib import Path

HERE = Path(__file__).resolve().parent


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".js": "text/javascript",
        ".css": "text/css",
        ".json": "application/json",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):  # 保持输出安静
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Serve the static recommendation demo.")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    handler = partial(Handler, directory=str(HERE))
    with socketserver.TCPServer((args.host, args.port), handler) as server:
        url = f"http://{args.host}:{args.port}/"
        print(f"Serving {HERE} at {url}  (Ctrl+C to stop)")
        if not args.no_browser:
            threading.Timer(0.4, webbrowser.open, [url]).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nstopped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
