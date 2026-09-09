import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class LauncherTests(unittest.TestCase):
    def run_shell(self, code, directory, **extra):
        env = {k: v for k, v in os.environ.items() if not k.startswith("NEMO_")}
        env.update(NEMO_WORKSPACE=directory, **extra)
        return subprocess.run(["bash", "-c", 'source "$1"; ' + code, "test", str(ROOT / "start-nemo.sh")],
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
