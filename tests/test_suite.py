import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


bench = module("bench")
report = module("report")


def gpus():
    return [{"index": str(i), "uuid": f"GPU-{i}", "name": "NVIDIA GeForce RTX 3090",
             "memory.total": 24576., "memory.used": 1000., "memory.free": 23576.,
             "temperature.gpu": 50., "utilization.gpu": 0., "power.limit": "210.00"} for i in range(2)]


def options(output, **kwargs):
    result = dict(output=output, mode="inference", max_temp=85, api_base="http://unused",
                  expected_cpu="5700X", requests=4, output_tokens=8, timeout=3,
                  min_output_tps=0, contexts=[16], concurrencies=[1, 2], soak_seconds=0)
    return argparse.Namespace(**(result | kwargs))


class Handler(BaseHTTPRequestHandler):
    wrong_usage = False

    def log_message(self, *args):
        pass

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        usage = {"prompt_tokens": len(payload["prompt"]), "completion_tokens": payload["max_tokens"]}
        if self.wrong_usage:
            usage["prompt_tokens"] += 1
        events = [{"choices": [{"text": "hello"}]}, {"choices": [], "usage": usage}]
        for event in events:
            self.wfile.write(("data: " + json.dumps(event) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")


class SuiteTests(unittest.TestCase):
    def test_percentile(self):
        self.assertEqual(bench.percentile([4, 1, 3, 2], .5), 2)
        self.assertEqual(bench.percentile([4, 1, 3, 2], .95), 4)
        self.assertIsNone(bench.percentile([], .5))

    def test_completion_validation(self):
        with self.assertRaises(RuntimeError):
            bench.validate_completion({"choices": [{"text": ""}]})

    def test_invalid_lists(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            bench.positive_list("1,0")

    def test_run_records_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = bench.Run(options(Path(tmp) / "out"))
            run.check("bad", lambda: 1 / 0)
            run.finish()
            self.assertEqual(json.loads((run.directory / "report.json").read_text())["status"], "FAIL")

    def test_telemetry_safety_fails_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = bench.Run(options(Path(tmp) / "out"))
            rows = gpus()
            rows[0]["temperature.gpu"] = 90
            with patch.object(bench, "gpu_snapshot", return_value=rows), \
                    patch.object(bench, "server_metrics", return_value={}):
                run.start_monitor()
                self.assertTrue(run.stop.wait(3))
                run.finish()
            self.assertEqual(run.data["status"], "FAIL")

    def test_real_http_streams_and_token_accounting(self):
        Handler.wrong_usage = False
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run = bench.Run(options(Path(tmp) / "out", api_base=f"http://127.0.0.1:{server.server_port}"))
                with patch.object(bench, "gpu_snapshot", side_effect=lambda: gpus()):
                    result = bench.inference_case(run, 16, 2)
                self.assertEqual(len(result["requests"]), 4)
                self.assertEqual(sum(r["output_tokens"] for r in result["requests"]), 32)
                self.assertGreater(result["aggregate_output_tokens_s"], 0)
                self.assertGreaterEqual(result["ttft_p95_s"], 0)
                Handler.wrong_usage = True
                with self.assertRaisesRegex(RuntimeError, "input token count mismatch"):
                    bench.stream_request(run, 16, 1)
        finally:
            Handler.wrong_usage = False
            server.shutdown()
            server.server_close()
            thread.join()

    def test_preflight_cpu_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = bench.Run(options(Path(tmp) / "out"))
            with patch.object(bench, "command", return_value={"returncode": 0, "stdout": "AMD 5700G", "stderr": ""}), \
                    patch.object(bench, "gpu_snapshot", return_value=gpus()):
                entry = run.check("preflight", lambda: bench.inventory(run))
            self.assertEqual(entry["status"], "FAIL")

    def test_safety_guard_blocks_new_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = bench.Run(options(Path(tmp) / "out"))
            run.stop.set()
            with patch.object(bench, "http") as request:
                with self.assertRaises(RuntimeError):
                    bench.stream_request(run, 16, 0)
                request.assert_not_called()

    def test_missing_selected_stages_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "configuration.json").write_text(json.dumps({"mode": "quick"}))
            self.assertEqual(report.summarize(directory, 0), 1)
            summary = json.loads((directory / "summary.json").read_text())
            self.assertTrue(any("tp2" in item for item in summary["failures"]))

    def test_new_kernel_xid_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            (directory / "configuration.json").write_text(json.dumps({"mode": "preflight"}))
            stage = directory / "preflight"
            stage.mkdir()
            (stage / "report.json").write_text(json.dumps({"status": "PASS_SELECTED_CHECKS", "checks": []}))
            (directory / "kernel.log").write_text("NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus\n")
            self.assertEqual(report.summarize(directory, 0), 1)

    def test_comparison_regression_and_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ("old", "new")]
            for path in paths:
                path.mkdir()
                (path / "summary.json").write_text(json.dumps({"status": "PASS_SELECTED_CHECKS"}))
            key = ("tp2", 256, 2)
            old = {key: [{"aggregate_output_tokens_s": 100., "ttft_p95_s": 1.}]}
            new = {key: [{"aggregate_output_tokens_s": 60., "ttft_p95_s": 1.}]}
            with patch.object(report, "fingerprint", return_value={"same": True}), \
                    patch.object(report, "case_metrics", side_effect=[old, new]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(report.compare(*paths, .2), 1)
            with patch.object(report, "fingerprint", side_effect=[{"a": 1}, {"a": 2}]):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    report.compare(*paths, .2)

    def test_interrupted_run_not_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "out"
            with patch("sys.argv", ["bench.py", "inference", "--output", str(output)]), \
                    patch.object(bench, "inventory", return_value=[]), \
                    patch.object(bench.Run, "start_monitor"), \
                    patch.object(bench, "inference", side_effect=KeyboardInterrupt):
                with self.assertRaises(KeyboardInterrupt):
                    bench.main()
            self.assertEqual(json.loads((output / "report.json").read_text())["status"], "FAIL")

    @unittest.skipUnless(shutil.which("bash") and shutil.which("flock"), "requires bash/flock")
    def test_shell_orchestrator_quick_success_and_startup_failure(self):
        fake_docker = '''#!/usr/bin/env python3
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path.cwd()
if args[:2] == ['image', 'inspect']:
    print(json.dumps([{'Id': 'sha256:fake', 'RepoDigests': []}]))
elif args and args[0] == 'inspect':
    print('exited unhealthy' if os.environ.get('FAKE_STARTUP_FAIL') else 'running healthy')
elif args and args[0] == 'compose':
    if 'ps' in args and '--status' not in args:
        print('fake-container')
    elif 'run' in args:
        output = args[args.index('--output') + 1]
        path = root / output.lstrip('/')
        path.mkdir(parents=True)
        (path / 'report.json').write_text(json.dumps({'status': 'PASS_SELECTED_CHECKS', 'checks': []}))
'''
        for fail in (False, True):
            with self.subTest(startup_failure=fail), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for name in ("run.sh", "report.py", "compose.yaml"):
                    shutil.copy(ROOT / name, root / name)
                tools = root / "bin"
                tools.mkdir()
                (tools / "docker").write_text(fake_docker)
                (tools / "docker").chmod(0o755)
                (tools / "journalctl").write_text("#!/bin/sh\necho 'No journal permission' >&2\nexit 1\n")
                (tools / "journalctl").chmod(0o755)
                env = os.environ | {"PATH": str(tools) + os.pathsep + os.environ["PATH"]}
                if fail:
                    env["FAKE_STARTUP_FAIL"] = "1"
                result = subprocess.run(["bash", "run.sh", "quick"], cwd=root, env=env,
                                        capture_output=True, text=True, timeout=30)
                self.assertEqual(result.returncode, 1 if fail else 0, result.stdout + result.stderr)
                runs = list((root / "results").iterdir())
                self.assertEqual(len(runs), 1)
                summary = json.loads((runs[0] / "summary.json").read_text())
                self.assertEqual(summary["status"], "FAIL" if fail else "PASS_SELECTED_CHECKS")
                if not fail:
                    self.assertEqual(set(summary["stages"]), {"preflight", "hardware", "tp1-gpu0", "tp1-gpu1", "tp2"})


if __name__ == "__main__":
    unittest.main()
