import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import textwrap
import tomllib
import unittest
import uuid


ROOT = Path(__file__).resolve().parents[1]


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.root = ROOT / "tests" / (".install-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.addCleanup(shutil.rmtree, self.root)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.data = self.root / "data" / "omawhisper"
        self.log = self.root / "commands.jsonl"
        self.python = self.bin / "python"
        self.python.write_text(f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            from pathlib import Path
            import shutil
            import sys

            program = Path(sys.argv[0])
            args = sys.argv[1:]
            with open(os.environ["INSTALL_TEST_LOG"], "a") as stream:
                stream.write(json.dumps({"program": str(program), "args": args}) + "\\n")
            if args[:1] == ["-"]:
                sys.argv = args
                exec(compile(sys.stdin.read(), "<install-engines>", "exec"), {"__name__": "__main__"})
            elif args[:1] == ["-c"]:
                from importlib.metadata import PackageNotFoundError
                from unittest.mock import patch
                state = program.parent.parent / "packages.json"
                packages = set(json.loads(state.read_text())) if state.exists() else set()
                def distribution(name):
                    if os.environ.get("INSTALL_TEST_METADATA_FAILURE"):
                        raise RuntimeError("Cannot inspect metadata")
                    if name not in packages:
                        raise PackageNotFoundError(name)
                    return object()
                def import_module(name):
                    if os.environ.get("INSTALL_TEST_SMOKE_FAILURE") == program.parent.parent.name:
                        raise ImportError("Smoke import failed")
                    return object()
                sys.argv = ["-c", *args[2:]]
                with patch("importlib.metadata.distribution", side_effect=distribution), \\
                     patch("importlib.import_module", side_effect=import_module):
                    exec(compile(args[1], "<environment-probe>", "exec"), {"__name__": "__main__"})
            elif args[:2] == ["-m", "venv"]:
                target = Path(args[2]) / "bin/python"
                target.parent.mkdir(parents=True)
                shutil.copy2(program, target)
            elif args[:2] == ["-m", "pip"]:
                state = program.parent.parent / "packages.json"
                packages = set(json.loads(state.read_text())) if state.exists() else set()
                if args[2] == "show":
                    sys.exit(0 if args[3] in packages else 1)
                if args[2] != "install":
                    sys.exit("Unexpected pip command")
                if os.environ.get("INSTALL_TEST_PIP_FAILURE") == program.parent.parent.name:
                    sys.exit("Pip install failed")
                extras = args[-1].rsplit("[", 1)[1].rstrip("]").split(",")
                if "whisper" in extras:
                    packages.update(["faster-whisper", "onnxruntime"])
                if "parakeet" in extras:
                    packages.update(["onnx-asr", "onnxruntime"])
                if "parakeet-cuda" in extras:
                    packages.update(["onnx-asr", "onnxruntime-gpu"])
                state.write_text(json.dumps(sorted(packages)))
            elif not args or not args[0].endswith("/scripts/install.py"):
                sys.exit("Unexpected Python command")
            """))
        self.python.chmod(0o700)
        tool_source = f"#!{sys.executable}\n" + textwrap.dedent("""\
            import json
            import os
            import sys
            with open(os.environ["INSTALL_TEST_LOG"], "a") as stream:
                stream.write(json.dumps({"program": sys.argv[0], "args": sys.argv[1:]}) + "\\n")
            """)
        for name in ("omarchy", "hyprctl", "pw-record", "pw-dump", "wl-copy",
                     "wl-paste", "systemctl", "omarchy-shell"):
            tool = self.bin / name
            tool.write_text(tool_source)
            tool.chmod(0o700)
        self.environment = {
            **os.environ,
            "HOME": str(self.root / "home"),
            "XDG_DATA_HOME": str(self.data.parent),
            "XDG_CONFIG_HOME": str(self.root / "config"),
            "XDG_STATE_HOME": str(self.root / "state"),
            "XDG_RUNTIME_DIR": str(self.root / "runtime"),
            "OMAWHISPER_PYTHON": str(self.python),
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "INSTALL_TEST_LOG": str(self.log),
        }

    def run_installer(self, *args, succeeds=True, engines=False):
        script = "install-engines.sh" if engines else "install.sh"
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / script), *args],
            cwd=ROOT, env=self.environment, capture_output=True, text=True,
        )
        if succeeds:
            self.assertEqual(result.returncode, 0, result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def check(self, *args, status):
        result = self.run_installer("--check", *args, engines=True, succeeds=status == 0)
        self.assertEqual(result.returncode, status, result.stderr)
        return result

    def commands(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()]

    def installs(self):
        return [
            (Path(row["program"]).parent.parent.name, row["args"][-1])
            for row in self.commands() if row["args"][:3] == ["-m", "pip", "install"]
        ]

    def provision(self, name, packages):
        directory = self.data / name
        (directory / "bin").mkdir(parents=True)
        shutil.copy2(self.python, directory / "bin/python")
        (directory / "packages.json").write_text(json.dumps(packages))
        return directory

    def test_cpu_install_creates_separate_whisper_and_parakeet_environments(self):
        self.run_installer()
        self.assertEqual(self.installs(), [
            ("venv", f"{ROOT}[whisper]"),
            ("venv-onnx-cpu", f"{ROOT}[parakeet]"),
        ])
        self.assertFalse((self.data / "venv-onnx-cuda").exists())
        created = [row["args"][2] for row in self.commands()
                   if row["args"][:2] == ["-m", "venv"]]
        self.assertEqual(created, [str(self.data / name) for name in ("venv", "venv-onnx-cpu")])
        self.assertTrue(all(row["program"] == str(self.python) for row in self.commands()
                            if row["args"][:2] == ["-m", "venv"]))

    def test_cuda_install_separates_package_namespaces(self):
        self.run_installer("--cuda")
        self.assertEqual(self.installs(), [
            ("venv", f"{ROOT}[whisper,cuda]"),
            ("venv-onnx-cpu", f"{ROOT}[parakeet]"),
            ("venv-onnx-cuda", f"{ROOT}[parakeet-cuda]"),
        ])
        for name in ("venv", "venv-onnx-cpu", "venv-onnx-cuda"):
            packages = json.loads((self.data / name / "packages.json").read_text())
            if name == "venv-onnx-cuda":
                self.assertEqual(packages, ["onnx-asr", "onnxruntime-gpu"])
            else:
                self.assertIn("onnxruntime", packages)
                self.assertNotIn("onnxruntime-gpu", packages)

    def test_new_runtimes_reuse_the_existing_interpreter_without_an_override(self):
        main = self.provision("venv", ["faster-whisper", "onnxruntime"])
        shutil.copy2(self.python, self.bin / "python3")
        self.environment.pop("OMAWHISPER_PYTHON")
        self.run_installer("--cuda", engines=True)
        creation = [row for row in self.commands() if row["args"][:2] == ["-m", "venv"]]
        self.assertEqual(len(creation), 2)
        self.assertTrue(all(row["program"] == str(main / "bin/python") for row in creation))
    def test_cpu_update_retains_cuda_environment_and_user_settings(self):
        config = self.root / "config/omawhisper/config.json"
        config.parent.mkdir(parents=True)
        saved = '{"acceleration":"gpu","model":"parakeet-v3-fp32"}'
        config.write_text(saved)
        self.run_installer("--cuda")
        cuda = self.data / "venv-onnx-cuda"
        before = {str(path.relative_to(cuda)): path.read_bytes()
                  for path in cuda.rglob("*") if path.is_file()}
        self.log.write_text("")
        self.run_installer()
        self.assertEqual(self.installs(), [])
        self.assertFalse(any(row["args"][:2] == ["-m", "venv"] for row in self.commands()))
        self.assertFalse(any("venv-onnx-cuda" in row["program"] for row in self.commands()))
        self.assertEqual(before, {str(path.relative_to(cuda)): path.read_bytes()
                                  for path in cuda.rglob("*") if path.is_file()})
        self.assertEqual(config.read_text(), saved)

    def test_existing_main_cpu_onnx_packages_are_safe(self):
        self.provision("venv", ["onnx-asr", "onnxruntime", "faster-whisper"])
        self.run_installer("--cuda")
        self.assertEqual(len(self.installs()), 3)

    def test_cuda_flag_is_removed_and_section_is_forwarded(self):
        for args in (("--section", "left"), ("--cuda", "--section", "center"),
                     ("--section", "right", "--cuda", "--cuda")):
            with self.subTest(args=args):
                self.log.write_text("")
                self.run_installer(*args)
                calls = [row for row in self.commands() if row["args"]
                         and row["args"][0].endswith("/scripts/install.py")]
                self.assertEqual(len(calls), 1)
                self.assertEqual(calls[0]["program"], str(self.data / "venv/bin/python"))
                self.assertEqual(calls[0]["args"], [
                    str(ROOT / "scripts/install.py"), *[arg for arg in args if arg != "--cuda"],
                ])
                self.assertIn({"program": str(self.bin / "omarchy"),
                               "args": ["plugin", "validate", str(ROOT)]}, self.commands())

    def test_wrong_or_coinstalled_onnx_distributions_abort_before_any_install(self):
        for name, wrong in (("venv", "onnxruntime-gpu"),
                            ("venv-onnx-cpu", "onnxruntime-gpu"),
                            ("venv-onnx-cuda", "onnxruntime")):
            for packages in ([wrong], ["onnxruntime", "onnxruntime-gpu"]):
                with self.subTest(name=name, packages=packages):
                    self.log.write_text("")
                    directory = self.provision(name, packages)
                    result = self.run_installer("--cuda", succeeds=False)
                    self.assertIn(str(directory), result.stderr)
                    self.assertIn("Refusing", result.stderr)
                    self.assertIn("Recreate only this virtual environment", result.stderr)
                    self.assertEqual(self.installs(), [])
                    self.assertFalse(any(row["args"] and row["args"][0].endswith("install.py")
                                         for row in self.commands()))
                    self.assertEqual(json.loads((directory / "packages.json").read_text()), packages)
                    shutil.rmtree(self.data)

    def test_cuda_environment_cannot_contain_whisper_even_without_cpu_ort(self):
        self.provision("venv-onnx-cuda", ["faster-whisper", "onnxruntime-gpu"])
        result = self.run_installer("--cuda", succeeds=False)
        self.assertIn("faster-whisper", result.stderr)
        self.assertIn("--cuda", result.stderr)
        self.assertEqual(self.installs(), [])

    def test_dependency_extras_keep_cpu_and_cuda_packages_separate(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
        extras = project["optional-dependencies"]
        self.assertEqual(project["dependencies"], [])
        self.assertEqual(extras["parakeet"], ["onnx-asr[cpu,hub]>=0.12,<0.13"])
        self.assertEqual(extras["parakeet-cuda"], [
            "onnx-asr[gpu,hub]>=0.12,<0.13",
            "onnxruntime-gpu[cuda,cudnn]>=1.24.2,<1.25",
        ])
        for name in ("all", "engines"):
            self.assertIn("onnx-asr[cpu,hub]>=0.12,<0.13", extras[name])
        self.assertEqual(extras["cuda"], [
            "ctranslate2>=4.6.3,<5", "nvidia-cublas-cu12>=12.4,<13",
        ])

    def test_dependency_only_install_never_touches_desktop_or_service(self):
        self.run_installer("--cuda", engines=True)
        self.assertEqual(len(self.installs()), 3)
        self.assertFalse(any(Path(row["program"]).name != "python" for row in self.commands()))
        self.assertFalse(any(row["args"] and row["args"][0].endswith("install.py")
                             for row in self.commands()))
        self.assertFalse((self.root / "config").exists())

    def test_check_missing_environments_is_offline_and_read_only(self):
        self.check(status=1)
        self.check("--cuda", status=1)
        self.assertFalse(self.data.exists())
        self.assertFalse(any(row["args"][:1] != ["-"] for row in self.commands()))

    def test_current_cpu_markers_are_ready_but_cuda_requires_more(self):
        import hashlib
        self.run_installer(engines=True)
        fingerprint = hashlib.sha256((ROOT / "pyproject.toml").read_bytes()).hexdigest()
        for name, extras in (("venv", ["whisper"]), ("venv-onnx-cpu", ["parakeet"])):
            self.assertEqual(json.loads((self.data / name / ".omawhisper-ready").read_text()),
                             {"manifest_sha256": fingerprint, "extras": extras})
        self.log.write_text("")
        self.check(status=0)
        self.check("--cuda", status=1)
        self.assertFalse(any(row["args"][:2] == ["-m", "pip"] for row in self.commands()))
        self.assertFalse((self.data / "venv-onnx-cuda").exists())

    def test_cuda_marker_is_a_cpu_superset_and_check_does_not_write(self):
        self.run_installer("--cuda", engines=True)
        before = {str(path.relative_to(self.data)): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in self.data.rglob("*") if path.is_file()}
        self.log.write_text("")
        self.check(status=0)
        self.check("--cuda", status=0)
        self.assertEqual(before, {
            str(path.relative_to(self.data)): (path.read_bytes(), path.stat().st_mtime_ns)
            for path in self.data.rglob("*") if path.is_file()
        })
        self.assertFalse(any(row["args"][:2] == ["-m", "pip"] for row in self.commands()))

    def test_stale_missing_corrupt_or_wrong_extra_markers_need_provisioning(self):
        self.run_installer(engines=True)
        marker = self.data / "venv/.omawhisper-ready"
        original = json.loads(marker.read_text())
        for value in (None, "broken", "[]",
                      json.dumps({**original, "manifest_sha256": "old"}),
                      json.dumps({**original, "extras": ["parakeet"]}),
                      json.dumps({**original, "extras": [None]})):
            with self.subTest(value=value):
                if value is None:
                    marker.unlink()
                else:
                    marker.write_text(value)
                self.check(status=1)
        self.log.write_text("")
        self.run_installer(engines=True)
        self.assertEqual(self.installs(), [("venv", f"{ROOT}[whisper]")])
        self.check(status=0)

    def test_failed_pip_or_smoke_does_not_leave_a_ready_marker(self):
        for failure in ("INSTALL_TEST_PIP_FAILURE", "INSTALL_TEST_SMOKE_FAILURE"):
            with self.subTest(failure=failure):
                self.run_installer(engines=True)
                marker = self.data / "venv/.omawhisper-ready"
                marker.write_text('{"manifest_sha256": "old", "extras": ["whisper"]}')
                self.environment[failure] = "venv"
                result = self.run_installer(engines=True, succeeds=False)
                self.assertEqual(result.returncode, 2)
                self.assertFalse(marker.exists())
                self.check(status=1)
                self.environment.pop(failure)
                shutil.rmtree(self.data)

    def test_check_reports_collision_and_metadata_errors_as_real_errors(self):
        self.provision("venv", ["onnxruntime", "onnxruntime-gpu"])
        result = self.check(status=2)
        self.assertIn("Refusing", result.stderr)
        self.assertEqual(self.installs(), [])
        self.environment["INSTALL_TEST_METADATA_FAILURE"] = "1"
        result = self.check(status=2)
        self.assertIn("Cannot inspect metadata", result.stderr)

    def test_invalid_dependency_arguments_do_not_install(self):
        for args in (("--section", "left"), ("--unknown",), ("--check", "--unknown")):
            with self.subTest(args=args):
                result = self.run_installer(*args, engines=True, succeeds=False)
                self.assertEqual(result.returncode, 2)
                self.assertEqual(self.installs(), [])
                self.assertFalse(self.data.exists())
