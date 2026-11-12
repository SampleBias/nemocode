import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
# Resolve bash through PATH. On Windows a bare "bash" is looked up in System32
# first, which finds the WSL launcher instead of Git Bash.
BASH = shutil.which("bash") or "bash"


class LauncherTests(unittest.TestCase):
    def run_shell(self, code, directory, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
        env.update(NEMO_WORKSPACE=directory, **extra)
        return subprocess.run([BASH, "-c", 'source "$1"; ' + code, "test", str(ROOT / "start-nemo.sh")],
                              env=env, text=True, capture_output=True, timeout=5)

    def test_literal_config_and_environment_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "must-not-exist"
            (Path(directory) / ".env").write_text(
                f'NEMO_THREADS=3\nNEMO_CTX=8192\nNEMO_MODEL="$(touch {marker})"\nUNRELATED_APP_KEY=value\n'
            )
            result = self.run_shell('printf "%s|%s|%s" "$THREADS" "$CTX" "$NEMO_MODEL"', directory, NEMO_THREADS="7")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(result.stdout.startswith("7|8192|$(touch"), result.stdout)
            self.assertFalse(marker.exists())

    def test_release_pin_cannot_escape_vendor_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell("true", directory, NEMO_LLAMA_RELEASE="../../elsewhere")
            self.assertNotEqual(result.returncode, 0)

    def test_model_checksum_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "test.gguf"
            path.write_bytes(b"test artifact")
            result = self.run_shell("verify_model", directory, NEMO_MODEL_PATH=str(path), NEMO_MODEL_SHA256="0" * 64)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("checksum mismatch", result.stderr)

    def test_model_presets_pin_revision_and_checksum(self):
        with tempfile.TemporaryDirectory() as directory:
            show = 'printf "%s|%s|%s|%s" "$MODEL_PRESET" "$MODEL_FILE" "${#MODEL_SHA256}" "$HF_URL"'
            for preset in ("mini", "small", "medium", "large", "nemotron", "gemma4"):
                result = self.run_shell(show, directory, NEMO_MODEL_PRESET=preset)
                self.assertEqual(result.returncode, 0, result.stderr)
                name, file, checksum_length, url = result.stdout.split("|")
                self.assertEqual(name, preset)
                self.assertEqual(checksum_length, "64")
                self.assertTrue(url.endswith("/" + file), url)
                self.assertNotIn("/resolve/main/", url)
            # A model the user chose is never checked against a preset's checksum.
            own = self.run_shell(show, directory, NEMO_MODEL_PATH=str(Path(directory) / "own.gguf"))
            self.assertTrue(own.stdout.startswith("custom|own.gguf|0|"), own.stdout)
            moved = self.run_shell(show, directory, NEMO_MODEL_PRESET="small", NEMO_MODEL_REVISION="main")
            self.assertIn("|0|", moved.stdout)
            self.assertIn("/resolve/main/", moved.stdout)
            self.assertNotEqual(self.run_shell("true", directory, NEMO_MODEL_PRESET="huge").returncode, 0)

    def test_cpu_only_request_never_selects_a_gpu_sized_model(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell('printf "%s" "$MODEL_PRESET"', directory, NEMO_GPU_LAYERS="0")
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn(result.stdout, ("mini", "small"))

    def test_gemma4_preset_uses_pinned_text_model(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell(
                'printf "%s|%s|%s|%s|%s" "$MODEL_REPO" "$MODEL_FILE" "$MODEL_REVISION" "$MODEL_SHA256" "$MODEL_BYTES"',
                directory, NEMO_MODEL_PRESET="gemma4",
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.split("|"), [
                "S4MPL3BI4S/gemma4-coding-agent",
                "gemma-4-E4B-it.Q4_K_M.gguf",
                "659ee84acf247b242897af9a7cb646339ebeca24",
                "59fcabeca6ade1b5a90b2230e7e174e7c6f7e9c8b7a79f0b4db3cc23326e3e32",
                "5335274944",
            ])

    @unittest.skipIf(os.name == "nt", "PTY-based launcher interaction test")
    def test_interactive_model_choice_stops_or_continues_download(self):
        import pty

        for answer, expected in (("\n", "default.gguf"), ("y\n1\n", "Nemotron-3-Nano-4B-Coding-Agent-Q4_K_M.gguf"),
                                 ("y\nwrong\n2\n", "gemma-4-E4B-it.Q4_K_M.gguf"), ("y\n0\n", "default.gguf")):
            with self.subTest(answer=answer), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fake_curl = root / "curl"
                fake_curl.write_text(
                    f"#!{sys.executable}\n"
                    "import os, pathlib, signal, sys, time\n"
                    "output = pathlib.Path(sys.argv[sys.argv.index('-o') + 1])\n"
                    "output.write_bytes(b'partial')\n"
                    "if output.name == 'default.gguf.partial':\n"
                    "    def stop(*args):\n"
                    "        pathlib.Path(os.environ['FAKE_STOPPED']).write_text('stopped')\n"
                    "        sys.exit(143)\n"
                    "    signal.signal(signal.SIGTERM, stop)\n"
                    "    pathlib.Path(os.environ['FAKE_READY']).write_text('ready')\n"
                    "    while not pathlib.Path(os.environ['FAKE_RELEASE']).exists(): time.sleep(0.01)\n"
                )
                fake_curl.chmod(0o755)
                env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
                env.update(PATH=f"{root}{os.pathsep}{env.get('PATH', '')}", NEMO_WORKSPACE=directory,
                           NEMO_MODELS_DIR=directory, FAKE_READY=str(root / "ready"),
                           FAKE_RELEASE=str(root / "release"), FAKE_STOPPED=str(root / "stopped"))
                script = ('source "$1"; MODEL_FILE=default.gguf; MODEL_PATH="$NEMO_MODELS_DIR/default.gguf"; '
                          'MODEL_BYTES=0; HF_URL=https://example.invalid/default; '
                          'verify_model() { return 0; }; download_model_with_choice; '
                          'printf "CHOSEN:%s\\n" "$MODEL_FILE"')
                master, slave = pty.openpty()
                process = subprocess.Popen([BASH, "-c", script, "test", str(ROOT / "start-nemo.sh")],
                                           stdin=slave, stdout=slave, stderr=slave, env=env)
                os.close(slave)
                try:
                    deadline = time.monotonic() + 5
                    while not (root / "ready").exists() and time.monotonic() < deadline:
                        self.assertIsNone(process.poll(), "launcher exited before the download started")
                        time.sleep(0.01)
                    self.assertTrue((root / "ready").exists(), "download did not start")
                    os.write(master, answer.encode())
                    if expected == "default.gguf":
                        (root / "release").touch()
                    self.assertEqual(process.wait(timeout=5), 0)
                    output = os.read(master, 16384).decode(errors="replace")
                    self.assertIn(f"CHOSEN:{expected}", output)
                    if expected != "default.gguf":
                        self.assertTrue((root / "stopped").exists())
                        self.assertEqual((root / "default.gguf.partial").read_bytes(), b"partial")
                        self.assertTrue((root / expected).exists())
                    else:
                        self.assertTrue((root / "default.gguf").exists())
                finally:
                    if process.poll() is None:
                        process.terminate()
                        process.wait(timeout=5)
                    os.close(master)

    def test_explicit_and_noninteractive_model_choices_do_not_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            script = 'download_model() { printf "download:%s" "$MODEL_PRESET"; }; download_model_with_choice'
            for extra in ({}, {"NEMO_MODEL_PRESET": "gemma4"}, {"NEMO_MODEL_PATH": str(Path(directory) / "model.gguf")}):
                result = self.run_shell(script, directory, **extra)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue(result.stdout.startswith("download:"))
                self.assertNotIn("Try an experimental model", result.stderr)

    @unittest.skipIf(os.name == "nt", "PTY-based launcher interaction test")
    def test_explicit_model_preset_bypasses_prompt_in_terminal(self):
        import pty

        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
            env.update(NEMO_WORKSPACE=directory, NEMO_MODEL_PRESET="gemma4")
            master, slave = pty.openpty()
            try:
                process = subprocess.Popen(
                    [BASH, "-c", 'source "$1"; download_model() { printf "selected:%s" "$MODEL_PRESET"; }; download_model_with_choice',
                     "test", str(ROOT / "start-nemo.sh")],
                    stdin=slave, stdout=slave, stderr=slave, env=env,
                )
                os.close(slave)
                slave = None
                self.assertEqual(process.wait(timeout=5), 0)
                output = os.read(master, 16384).decode(errors="replace")
                self.assertIn("selected:gemma4", output)
                self.assertNotIn("Try an experimental model", output)
            finally:
                if slave is not None:
                    os.close(slave)
                os.close(master)

    def test_failed_default_download_propagates_after_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell(
                'model_prompt_allowed() { return 0; }; download_model() { return 7; }; download_model_with_choice',
                directory,
            )
            self.assertEqual(result.returncode, 7)

    def test_model_loading_message_in_noninteractive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_shell(
                'SERVER_PID=$$; probes=0; curl() { probes=$((probes + 1)); (( probes >= 2 )); }; '
                'sleep() { :; }; wait_for_server; SERVER_PID=""', directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("Loading model; waiting for llama-server", result.stderr)
            self.assertIn("Local model server is ready", result.stderr)
            self.assertNotIn("\x1b", result.stderr)

    @unittest.skipIf(os.name == "nt", "PTY-based launcher interaction test")
    def test_model_loading_spinner_is_cleared_when_ready(self):
        import pty

        with tempfile.TemporaryDirectory() as directory:
            env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
            env["NEMO_WORKSPACE"] = directory
            script = ('source "$1"; SERVER_PID=$$; probes=0; '
                      'curl() { probes=$((probes + 1)); (( probes >= 3 )); }; '
                      'sleep() { :; }; wait_for_server; SERVER_PID=""')
            master, slave = pty.openpty()
            try:
                process = subprocess.Popen([BASH, "-c", script, "test", str(ROOT / "start-nemo.sh")],
                                           stdin=slave, stdout=slave, stderr=slave, env=env)
                os.close(slave)
                slave = None
                self.assertEqual(process.wait(timeout=5), 0)
                output = os.read(master, 16384).decode(errors="replace")
                self.assertIn("Loading model... |", output)
                self.assertIn("Loading model... /", output)
                self.assertIn("\x1b[2KLocal model server is ready", output)
            finally:
                if slave is not None:
                    os.close(slave)
                os.close(master)

    def test_foreign_workspace_config_cannot_grant_privileges(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / ".env"
            config.write_text("NEMO_PERMISSION=trusted\nNEMO_LLAMA_SERVER=/somewhere/server\nNEMO_CTX=4096\n")
            show = 'printf "%s|%s|%s" "${NEMO_PERMISSION:-}" "${NEMO_LLAMA_SERVER:-}" "$CTX"'
            implicit = self.run_shell(show, directory)
            self.assertEqual(implicit.returncode, 0, implicit.stderr)
            self.assertEqual(implicit.stdout, "||4096")
            self.assertIn("Ignoring NEMO_PERMISSION", implicit.stderr)
            explicit = self.run_shell(show, directory, NEMO_CONFIG=str(config))
            self.assertEqual(explicit.stdout, "trusted|/somewhere/server|4096")

    def test_source_launch_rebuilds_existing_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").touch()
            binary = root / "target/release/nemocode"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nprintf 'agent\\n'\n")
            binary.chmod(0o755)
            result = self.run_shell(
                'ROOT="$WORKSPACE"; cargo() { printf "build\\n"; }; run_agent --version',
                directory,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "build\nagent\n")

    def test_source_launch_does_not_run_stale_binary_after_build_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "Cargo.toml").touch()
            binary = root / "target/release/nemocode"
            binary.parent.mkdir(parents=True)
            binary.write_text("#!/bin/sh\nprintf 'stale\\n'\n")
            binary.chmod(0o755)
            result = self.run_shell(
                'ROOT="$WORKSPACE"; cargo() { return 1; }; run_agent --version', directory,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertNotIn("stale", result.stdout)

    def test_packaged_launcher_finds_sibling_binary_without_path_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "libexec/nemocode").mkdir(parents=True)
            binary = root / "bin/nemocode"
            binary.parent.mkdir()
            binary.write_text("#!/bin/sh\nprintf 'packaged\\n'\n")
            binary.chmod(0o755)
            result = self.run_shell('ROOT="$WORKSPACE/libexec/nemocode"; run_agent --version', directory)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout, "packaged\n")


if __name__ == "__main__":
    unittest.main()
