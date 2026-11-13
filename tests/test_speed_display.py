"""Optional interactive spinner test; set NEMO_TEST_CLI to the built binary."""
import json
import os
from pathlib import Path
import select
import subprocess
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


CLI = os.environ.get("NEMO_TEST_CLI")


@unittest.skipIf(os.name == "nt" or not CLI, "set NEMO_TEST_CLI on a POSIX system")
class SpeedDisplayTests(unittest.TestCase):
    def test_thinking_shows_previous_generation_speed(self):
        import pty

        class MockServer(BaseHTTPRequestHandler):
            calls = 0

            def log_message(self, *_args):
                pass

            def do_GET(self):
                body = json.dumps({"default_generation_settings": {"n_ctx": 8192},
                                   "chat_template": "mock"}).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path != "/v1/chat/completions":
                    self.send_response(404)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                type(self).calls += 1
                content = "first reply" if self.calls == 1 else "second reply"
                chunk = {"choices": [{"delta": {"content": content}, "finish_reason": "stop"}],
                         "usage": {"completion_tokens": 20}, "timings": {"predicted_per_second": 8.36}}
                body = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                time.sleep(0.5)  # Allow the interactive spinner to render.
                self.wfile.write(body)

        server = ThreadingHTTPServer(("127.0.0.1", 0), MockServer)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                workspace = Path(directory) / "workspace"
                workspace.mkdir()
                env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
                env.update(NEMO_BASE_URL=f"http://127.0.0.1:{server.server_port}/v1",
                           NEMO_CTX="8192", NEMO_MAX_TOKENS="512",
                           NEMO_SESSION_DIR=str(Path(directory) / "session"), NEMO_PYTHON_LSP="off")
                master, slave = pty.openpty()
                process = subprocess.Popen([CLI, "--workspace", str(workspace)], stdin=slave,
                                           stdout=slave, stderr=slave, env=env)
                os.close(slave)

                def read_until(marker):
                    output = b""
                    deadline = time.monotonic() + 20
                    while marker not in output and time.monotonic() < deadline:
                        if select.select([master], [], [], 0.25)[0]:
                            output += os.read(master, 65536)
                        if process.poll() is not None:
                            break
                    self.assertIn(marker, output, output.decode(errors="replace"))
                    return output

                prompt = b"You [workspace]> "
                try:
                    read_until(prompt)
                    os.write(master, b"first\n")
                    first = read_until(prompt)
                    self.assertIn(b"last: -- Tp/s", first)
                    self.assertIn(b"first reply", first)
                    os.write(master, b"second\n")
                    second = read_until(prompt)
                    self.assertIn(b"last: 8.4 Tp/s", second)
                    self.assertIn(b"second reply", second)
                    os.write(master, b"exit\n")
                    self.assertEqual(process.wait(timeout=10), 0)
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=10)
                    os.close(master)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
