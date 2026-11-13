"""Optional integration test against a real llama-server and a local GGUF.

Set NEMO_TEST_LLAMA_SERVER and NEMO_TEST_GGUF to run this test. The GGUF is
used only to initialize the server; /apply-template does not run inference.
"""
import json
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import unittest
import urllib.error
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
SERVER = os.environ.get("NEMO_TEST_LLAMA_SERVER")
MODEL = os.environ.get("NEMO_TEST_GGUF")


@unittest.skipUnless(SERVER and MODEL, "set NEMO_TEST_LLAMA_SERVER and NEMO_TEST_GGUF")
class Gemma4TemplateTests(unittest.TestCase):
    def test_tool_declaration_and_result_round_trip(self):
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        base = f"http://127.0.0.1:{port}"
        with tempfile.TemporaryFile(mode="w+") as log:
            server = subprocess.Popen([
                SERVER, "--model", MODEL, "--chat-template-file", str(ROOT / "templates/gemma4.jinja"),
                "--jinja", "--host", "127.0.0.1", "--port", str(port), "--ctx-size", "2048",
                "--n-gpu-layers", "0", "--no-warmup", "--parallel", "1",
            ], stdout=log, stderr=log)
            try:
                deadline = time.monotonic() + 90
                while time.monotonic() < deadline:
                    if server.poll() is not None:
                        log.seek(0)
                        self.fail("llama-server exited: " + log.read()[-4000:])
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except urllib.error.HTTPError as error:
                        error.close()
                        time.sleep(0.25)
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.25)
                else:
                    self.fail("llama-server did not become ready")

                tool = {"type": "function", "function": {
                    "name": "read_file", "description": "Read a file",
                    "parameters": {"type": "object", "properties": {
                        "file_path": {"type": "string", "description": "Path"}},
                        "required": ["file_path"]}}}
                messages = [
                    {"role": "user", "content": "Read sample.txt"},
                    {"role": "assistant", "tool_calls": [{"id": "call_1", "type": "function",
                     "function": {"name": "read_file", "arguments": '{"file_path":"sample.txt"}'}}]},
                    {"role": "tool", "tool_call_id": "call_1", "content": "sample content"},
                ]
                body = json.dumps({"messages": messages, "tools": [tool], "tool_choice": "auto",
                                   "parallel_tool_calls": False, "add_generation_prompt": True,
                                   "chat_template_kwargs": {"enable_thinking": False}}).encode()
                request = urllib.request.Request(base + "/apply-template", data=body,
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    prompt = json.load(response)["prompt"]
                self.assertIn("declaration:read_file", prompt)
                self.assertIn("<|tool_call>call:read_file{", prompt)
                self.assertIn("sample.txt", prompt)
                self.assertIn("<|tool_response>response:read_file{", prompt)
                self.assertIn("sample content", prompt)
            finally:
                server.terminate()
                try:
                    server.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    server.kill()
                    server.wait()


if __name__ == "__main__":
    unittest.main()
