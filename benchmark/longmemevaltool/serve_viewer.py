#!/usr/bin/env python3
"""启动 LongMemEval 查看器的本地服务器。

用法:
    cd /mnt/petrelfs/leihaodong/ICML/locomo/benchmark/LongMemEval
    python serve_viewer.py

然后打开浏览器访问: http://localhost:8765
"""

import http.server
import socketserver
import os

PORT = 8765
PORT = 65534
DIR = os.path.dirname(os.path.abspath(__file__))

os.chdir(DIR)

class Handler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        super().end_headers()

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"LongMemEval 查看器: http://localhost:{PORT}/viewer.html")
    print("按 Ctrl+C 停止")
    httpd.serve_forever()
