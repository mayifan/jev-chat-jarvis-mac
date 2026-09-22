"""Offline unit tests for env-file write / preserve / mask (no AppKit, no network)."""

from __future__ import annotations

import os
import stat
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import config_io  # noqa: E402


class MaskTests(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(config_io.mask_key(""), "（未配置）")

    def test_short(self):
        self.assertEqual(config_io.mask_key("short"), "•••••")

    def test_long(self):
        masked = config_io.mask_key("sk-ABCDEFGHIJKLMNOPQRSTUV")
        self.assertTrue(masked.startswith("sk-ABC"))
        self.assertIn("…", masked)
        self.assertIn("STUV", masked)


class EnvWriteTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "env"

    def tearDown(self):
        self.tmp.cleanup()

    def test_create_new_file_mode_600(self):
        config_io.update_env_file(self.path, {
            "OPENAI_API_KEY": "sk-test-key-value-1234",
            "OPENAI_BASE_URL": "https://api.deepseek.com",
            "OPENAI_MODEL": "deepseek-chat",
        })
        self.assertTrue(self.path.exists())
        mode = stat.S_IMODE(self.path.stat().st_mode)
        self.assertEqual(mode, 0o600)
        text = self.path.read_text()
        self.assertIn("OPENAI_API_KEY=sk-test-key-value-1234", text)
        self.assertIn("OPENAI_BASE_URL=https://api.deepseek.com", text)
        self.assertIn("OPENAI_MODEL=deepseek-chat", text)

    def test_preserve_comments_and_unknown_keys(self):
        self.path.write_text(
            "# my header comment\n"
            "OPENAI_API_KEY=old-key-aaaaaaaa\n"
            "# keep this note\n"
            "CUSTOM_FOO=bar\n"
            "OPENAI_MODEL=old-model\n"
            "export TYPESAFE_API_KEY=jev-old  # inline note\n",
            encoding="utf-8",
        )
        os.chmod(self.path, 0o600)
        config_io.update_env_file(self.path, {
            "OPENAI_API_KEY": "new-key-bbbbbbbb",
            "OPENAI_MODEL": "new-model",
            "TYPESAFE_API_KEY": "jev-new",
        })
        text = self.path.read_text()
        self.assertIn("# my header comment", text)
        self.assertIn("# keep this note", text)
        self.assertIn("CUSTOM_FOO=bar", text)
        self.assertIn("OPENAI_API_KEY=new-key-bbbbbbbb", text)
        self.assertIn("OPENAI_MODEL=new-model", text)
        self.assertIn("export TYPESAFE_API_KEY=jev-new", text)
        self.assertIn("# inline note", text)
        self.assertNotIn("old-key-aaaaaaaa", text)
        self.assertNotIn("jev-old", text)
        mode = stat.S_IMODE(self.path.stat().st_mode)
        self.assertEqual(mode, 0o600)

    def test_quote_values_with_spaces(self):
        config_io.update_env_file(self.path, {
            "JEV_TONES": "摸鱼=like a pro|兵法=use metaphors",
        })
        text = self.path.read_text()
        self.assertIn('JEV_TONES="摸鱼=like a pro|兵法=use metaphors"', text)

    def test_remove_key_with_none(self):
        self.path.write_text(
            "OPENAI_API_KEY=abc\nOPENAI_MODEL=m\n", encoding="utf-8")
        config_io.update_env_file(self.path, {"OPENAI_API_KEY": None})
        text = self.path.read_text()
        self.assertNotIn("OPENAI_API_KEY", text)
        self.assertIn("OPENAI_MODEL=m", text)

    def test_apply_panel_updates_leaves_key_when_none(self):
        self.path.write_text(
            "OPENAI_API_KEY=keep-me-xxxxxxxxxxxx\n"
            "OPENAI_BASE_URL=https://example.com\n",
            encoding="utf-8",
        )
        config_io.apply_panel_updates(
            openai_key=None,  # unchanged
            openai_base="https://api.deepseek.com",
            openai_model="deepseek-chat",
            path=self.path,
        )
        text = self.path.read_text()
        self.assertIn("OPENAI_API_KEY=keep-me-xxxxxxxxxxxx", text)
        self.assertIn("OPENAI_BASE_URL=https://api.deepseek.com", text)
        self.assertIn("OPENAI_MODEL=deepseek-chat", text)


class ModelsUrlTests(unittest.TestCase):
    def test_appends_v1(self):
        self.assertEqual(
            config_io.models_url("https://api.deepseek.com"),
            "https://api.deepseek.com/v1/models")

    def test_respects_existing_version(self):
        self.assertEqual(
            config_io.models_url("https://api.openai.com/v1"),
            "https://api.openai.com/v1/models")
        self.assertEqual(
            config_io.models_url("https://open.bigmodel.cn/api/paas/v4"),
            "https://open.bigmodel.cn/api/paas/v4/models")


class ProbeOfflineTests(unittest.TestCase):
    def test_probe_missing_key(self):
        ids, err = config_io.probe_models("https://example.com", "")
        self.assertEqual(ids, [])
        self.assertIn("Key", err)

    def test_probe_parses_openai_shape(self):
        payload = b'{"data":[{"id":"a"},{"id":"b"}]}'

        class FakeResp:
            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def read(self):
                return payload

        import json
        # urlopen returns a context manager; json.load reads from it — patch at json.load
        # is awkward; instead patch urlopen to yield a BytesIO-like via a simple stub.

        class CM:
            def __enter__(self):
                import io
                return io.BytesIO(payload)

            def __exit__(self, *a):
                return False

        with patch("urllib.request.urlopen", return_value=CM()):
            ids, err = config_io.probe_models(
                "https://example.com/v1", "sk-test", timeout=1)
        self.assertEqual(err, "")
        self.assertEqual(ids, ["a", "b"])


if __name__ == "__main__":
    unittest.main()
