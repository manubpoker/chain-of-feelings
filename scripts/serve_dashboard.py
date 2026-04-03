"""Simple HTTP server for the Chain of Feelings dashboard.

Serves dashboard.html and results/ directory with CORS headers.
No dependencies beyond Python stdlib.

Usage:
    uv run scripts/serve_dashboard.py
    # Then open http://localhost:8420
"""

import http.server
import os
import sys
from functools import partial


class CORSHandler(http.server.SimpleHTTPRequestHandler):
    """HTTP handler with CORS headers for local development."""

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        # Colour-coded logging
        status = args[1] if len(args) > 1 else ""
        if str(status).startswith("2"):
            colour = "\033[32m"  # green
        elif str(status).startswith("3"):
            colour = "\033[33m"  # yellow
        elif str(status).startswith("4"):
            colour = "\033[31m"  # red
        else:
            colour = "\033[0m"
        reset = "\033[0m"
        sys.stderr.write(f"  {colour}{format % args}{reset}\n")


def main():
    port = 8420

    # Serve from project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)

    handler = partial(CORSHandler, directory=project_root)

    with http.server.HTTPServer(("", port), handler) as server:
        url = f"http://localhost:{port}/dashboard.html"
        print("=" * 60)
        print("Chain of Feelings -- Dashboard Server")
        print("=" * 60)
        print(f"\n  Serving from: {project_root}")
        print(f"  Dashboard:    {url}")
        print(f"  Data:         http://localhost:{port}/results/viz_data.json")
        print(f"\n  Press Ctrl+C to stop.\n")

        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\n  Server stopped.")


if __name__ == "__main__":
    main()
