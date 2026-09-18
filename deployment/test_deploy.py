import hashlib
import importlib.util
import io
import json
import os
import socket
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("deploy.py")
SPEC = importlib.util.spec_from_file_location("external_engram_deploy", MODULE_PATH)
deploy = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = deploy
SPEC.loader.exec_module(deploy)
X_SHA = hashlib.sha256(b"x").hexdigest()


def release_fixture():
    assets = [
        {"path": "core.gguf", "size": 1, "sha256": X_SHA, "role": "core"},
        {"path": "expertpack/expertpack.v2.manifest", "size": 1, "sha256": X_SHA, "role": "expert_manifest"},
        {"path": "engram/v1/engram.v1.manifest.json", "size": 1, "sha256": X_SHA, "role": "engram_manifest"},
        {"path": "runtime.tar", "size": 1, "sha256": X_SHA, "role": "source"},
        {"path": "runtime.hashes.json", "size": 1, "sha256": X_SHA, "role": "source_hashes"},
    ]
    assets.extend({"path": f"expertpack/pack-{i:02}.bin", "size": 1, "sha256": X_SHA, "role": "expert"} for i in range(49))
    rows = []
    for i in range(52):
        weight, scale = f"engram/w-{i:02}.bin", f"engram/s-{i:02}.bin"
        assets.extend(({"path": weight, "size": 1, "sha256": X_SHA, "role": "engram"},
                       {"path": scale, "size": 1, "sha256": X_SHA, "role": "engram"}))
        rows.append({"layer": 1, "row_start": i, "row_count": 1, "weight_path": weight,
                     "weight_row_bytes": 1, "scale_path": scale, "scale_row_bytes": 1})
    return {"id": deploy.RELEASE_ID, "repo": deploy.REPO, "revision": deploy.REVISION,
            "assets": assets, "engram_rows": rows,
            "defaults": {"gpu_cache_gib": 172, "host_cache_gib": 64, "io_threads": 4,
                         "threads": 20, "batch": 128, "context": 4096, "port": 48241}}


class SecurityTests(unittest.TestCase):
    def test_rejects_traversal_and_whitespace(self):
        for value in ("../core.gguf", "/core.gguf", "a/../b", "a b"):
            with self.assertRaises(deploy.DeploymentError):
                deploy.safe_rel(value, "test")

    def test_safe_archive_requires_runtime_root_and_regular_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive_path = Path(temporary) / "bad.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("../outside")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            with tarfile.open(archive_path) as archive:
                with self.assertRaises(deploy.DeploymentError):
                    deploy.safe_tar_members(archive)

    def test_archive_extract_checks_exact_source_hashes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root, archive_path = Path(temporary), Path(temporary) / "runtime.tar"
            with tarfile.open(archive_path, "w") as archive:
                info = tarfile.TarInfo("runtime/src/file.cpp")
                info.size = 1
                archive.addfile(info, io.BytesIO(b"x"))
            deploy.extract_source(root, archive_path, {"src/file.cpp": X_SHA}, X_SHA, X_SHA)
            self.assertEqual((root / "runtime/src/file.cpp").read_bytes(), b"x")
            with self.assertRaises(deploy.DeploymentError):
                deploy.extract_source(root, archive_path, {"src/file.cpp": "0" * 64}, X_SHA, "0" * 64)


class ReceiptAndPlanningTests(unittest.TestCase):
    def test_port_probe_refuses_listener_and_allows_its_immediate_release(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        port = listener.getsockname()[1]
        try:
            self.assertFalse(deploy.port_available(port))
        finally:
            listener.close()
        self.assertTrue(deploy.port_available(port))

    def test_receipt_reuse_requires_unchanged_stat(self):
        with tempfile.TemporaryDirectory() as temporary:
            data = Path(temporary)
            path = data / "core.gguf"
            path.write_bytes(b"x")
            asset = deploy.Asset("core.gguf", 1, X_SHA, "core")
            record = deploy.verify_asset(data, asset, None, False)
            old_mtime = record["before"]["mtime_ns"]
            os.utime(path, ns=(path.stat().st_atime_ns, old_mtime + 2_000_000_000))
            self.assertNotEqual(deploy.stat_record(path)["mtime_ns"], old_mtime)
            rebuilt = deploy.verify_asset(data, asset, record, False)
            self.assertEqual(rebuilt["sha256"], X_SHA)
            self.assertNotEqual(rebuilt["before"], record["before"])

    def test_release_fixture_and_offline_command_plan(self):
        release = deploy.validate_release(release_fixture())
        command = deploy.runtime_command(Path("/root/dsv41/build-engram/bin/llama-server"),
                                         Path("/snapshot/core.gguf"), 172, 64, 4, 20,
                                         128, 4096, 3, 48241)
        self.assertEqual(command[1:5], ["-m", str(Path("/snapshot/core.gguf")),
                                        "--moe-stream-cache", "172"])
        self.assertEqual(command[5:9], ["--moe-stream-l2", "64",
                                        "--moe-stream-io-threads", "4"])
        self.assertIn("--moe-stream-direct", command)
        self.assertIn("--no-warmup", command)
        self.assertEqual(command[command.index("-np") + 1], "3")
        self.assertEqual(command[command.index("--reasoning") + 1], "auto")
        self.assertEqual(len([a for a in release["assets"] if a.role == "expert"]), 49)
        self.assertNotIn("parallel", release["defaults"])

    def test_optional_parallel_default_rejects_non_positive_values(self):
        invalid = release_fixture()
        invalid["defaults"]["parallel"] = 0
        with self.assertRaises(deploy.DeploymentError):
            deploy.validate_release(invalid)
        self.assertEqual(deploy.make_parser().parse_args(["start", "--parallel", "2"]).parallel, 2)

    def test_engram_manifest_checks_declared_sidecar_hashes(self):
        release = deploy.validate_release(release_fixture())
        chunks = []
        for i in range(52):
            chunks.append({"row_start": i, "row_count": 1,
                           "weight": {"remote_path": f"engram/w-{i:02}.bin", "row_bytes": 1,
                                      "size": 1, "sha256": X_SHA},
                           "scale": {"remote_path": f"engram/s-{i:02}.bin", "row_bytes": 1,
                                     "size": 1, "sha256": X_SHA}})
        manifest = {"layers": [{"layer": 1, "rows": 52, "chunks": chunks}]}
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            deploy.check_engram_manifest(path, release)
            manifest["layers"][0]["chunks"][0]["weight"]["sha256"] = "0" * 64
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(deploy.DeploymentError):
                deploy.check_engram_manifest(path, release)

    def test_dsexp2_v2_header_and_manifest_relative_pack_names(self):
        release = deploy.validate_release(release_fixture())
        lines = ["DSEXP2 2"]
        lines.extend(f"PACK {i} pack-{i:02}.bin 1 {X_SHA}" for i in range(49))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "expertpack.v2.manifest"
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            # The asset list's manifest is in expertpack/, matching real PACK-relative names.
            original = next(a for a in release["assets"] if a.role == "expert_manifest")
            self.assertEqual(original.path, "expertpack/expertpack.v2.manifest")
            deploy.parse_expert_manifest(path, release["assets"])


if __name__ == "__main__":
    unittest.main()
