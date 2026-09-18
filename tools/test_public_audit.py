import tempfile
import unittest
from pathlib import Path

from public_audit import audit


class PublicAuditTests(unittest.TestCase):
    def test_ignores_bytecode_cache_but_rejects_transfer_payload(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "module.pyc").write_bytes(b"cache")
            (root / "payload.b64").write_text("opaque", encoding="utf-8")
            self.assertIn("forbidden artifact: payload.b64", audit(root))

    def test_rejects_private_key_and_openssh_key_material(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "notes.txt").write_text(
                "-----BEGIN " + "OPENSSH " + "PRIVATE " + "KEY-----\n" + "ssh-" + "ed25519 AAAATEST\n",
                encoding="utf-8",
            )
            findings = audit(root)
            self.assertTrue(any("private key marker" in finding for finding in findings))
            self.assertTrue(any("embedded OpenSSH key" in finding for finding in findings))

    def test_clean_tree_passes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "README.md").write_text("loopback 127.0.0.1 only", encoding="utf-8")
            self.assertEqual(audit(root), [])


if __name__ == "__main__":
    unittest.main()
