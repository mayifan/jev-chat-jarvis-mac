"""Offline unit tests for model download progress state (no AppKit, no network)."""

from __future__ import annotations

import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import model_download as md  # noqa: E402


class FormatTests(unittest.TestCase):
    def test_format_bytes(self):
        self.assertEqual(md.format_bytes(0), "0 B")
        self.assertEqual(md.format_bytes(512), "512 B")
        self.assertEqual(md.format_bytes(1024), "1.0 KB")
        self.assertEqual(md.format_bytes(1536), "1.5 KB")
        self.assertEqual(md.format_bytes(1024 ** 3), "1.0 GB")
        self.assertEqual(md.format_bytes(None), "—")

    def test_format_speed_eta(self):
        self.assertEqual(md.format_speed(0), "—")
        self.assertEqual(md.format_speed(None), "—")
        self.assertIn("/s", md.format_speed(2048))
        self.assertEqual(md.format_eta(None), "—")
        self.assertEqual(md.format_eta(45), "45s")
        self.assertEqual(md.format_eta(125), "2m05s")
        self.assertEqual(md.format_eta(3725), "1h02m")


class CachePathTests(unittest.TestCase):
    def test_respects_huggingface_hub_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"HUGGINGFACE_HUB_CACHE": tmp}, clear=False):
                # clear HF_HOME influence
                env = {"HUGGINGFACE_HUB_CACHE": tmp}
                with patch.dict("os.environ", env, clear=False):
                    # unset HF_HOME for this check
                    with patch.dict("os.environ", {"HF_HOME": ""}, clear=False):
                        root = md.hub_cache_root()
                        self.assertEqual(root, Path(tmp))
                        self.assertEqual(
                            md.model_cache_dir("Mapika/decider-2b"),
                            Path(tmp) / "models--Mapika--decider-2b",
                        )

    def test_respects_hf_home(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict(
                "os.environ",
                {"HF_HOME": tmp, "HUGGINGFACE_HUB_CACHE": ""},
                clear=False,
            ):
                # Force empty HUGGINGFACE_HUB_CACHE
                import os
                os.environ.pop("HUGGINGFACE_HUB_CACHE", None)
                os.environ["HF_HOME"] = tmp
                self.assertEqual(md.hub_cache_root(), Path(tmp) / "hub")


class ReadinessTests(unittest.TestCase):
    def test_is_ready_false_empty_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.dict("os.environ", {"HUGGINGFACE_HUB_CACHE": tmp}, clear=False):
                import os
                os.environ["HUGGINGFACE_HUB_CACHE"] = tmp
                os.environ.pop("HF_HOME", None)
                self.assertFalse(md.is_ready("Mapika/decider-2b"))

    def test_is_ready_false_tokenizer_only_snapshot(self):
        """Config+tokenizer without weights must NOT count as ready (real HF partial cache)."""
        with tempfile.TemporaryDirectory() as tmp:
            import os
            os.environ["HUGGINGFACE_HUB_CACHE"] = tmp
            os.environ.pop("HF_HOME", None)
            snap = (
                Path(tmp) / "models--Mapika--decider-2b" / "snapshots" / "abcd"
            )
            snap.mkdir(parents=True)
            (snap / "config.json").write_text("{}")
            (snap / "tokenizer.json").write_text("{}")
            (snap / "tokenizer_config.json").write_text("{}")
            self.assertFalse(md.is_ready("Mapika/decider-2b"))

    def test_is_ready_true_with_weight_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            os.environ["HUGGINGFACE_HUB_CACHE"] = tmp
            os.environ.pop("HF_HOME", None)
            snap = (
                Path(tmp) / "models--Mapika--decider-2b" / "snapshots" / "abcd"
            )
            snap.mkdir(parents=True)
            (snap / "model.safetensors").write_bytes(b"fake")
            (snap / "config.json").write_text("{}")
            self.assertTrue(md.is_ready("Mapika/decider-2b"))


class StateMachineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env_patch = patch.dict(
            "os.environ",
            {"HUGGINGFACE_HUB_CACHE": self.tmp.name},
            clear=False,
        )
        self.env_patch.start()
        import os
        os.environ["HUGGINGFACE_HUB_CACHE"] = self.tmp.name
        os.environ.pop("HF_HOME", None)
        # Isolate singleton
        md._default = None
        self.dl = md.ModelDownloader("Mapika/decider-2b")

    def tearDown(self):
        self.dl.cancel()
        # wait briefly for worker to notice
        t = self.dl._thread
        if t is not None and t.is_alive():
            t.join(timeout=2)
        self.env_patch.stop()
        self.tmp.cleanup()
        md._default = None

    def test_initial_not_downloaded(self):
        snap = self.dl.snapshot()
        self.assertEqual(snap.state, md.STATE_NOT_DOWNLOADED)

    def test_pause_cancel_flags(self):
        # Simulate downloading state without network
        with self.dl._lock:
            self.dl._state = md.STATE_DOWNLOADING
        self.dl.pause()
        self.assertEqual(self.dl.snapshot().state, md.STATE_PAUSED)
        self.assertFalse(self.dl._pause_gate.is_set())

        # Continue via start while "thread alive" — spawn a fake alive waiter
        gate = self.dl._pause_gate

        def fake_worker():
            # hold the thread slot
            while not self.dl._cancel.is_set():
                try:
                    self.dl._cooperative_gate()
                except md.DownloadCancelled:
                    break
                time.sleep(0.05)

        self.dl._cancel.clear()
        self.dl._thread = threading.Thread(target=fake_worker, daemon=True)
        self.dl._thread.start()
        self.assertTrue(self.dl.start())  # resume from paused
        self.assertEqual(self.dl.snapshot().state, md.STATE_DOWNLOADING)
        self.assertTrue(gate.is_set())

        self.dl.cancel()
        self.dl._thread.join(timeout=2)
        self.assertEqual(self.dl.snapshot().state, md.STATE_NOT_DOWNLOADED)

    def test_cooperative_gate_raises_on_cancel(self):
        self.dl._cancel.set()
        with self.assertRaises(md.DownloadCancelled):
            self.dl._cooperative_gate()

    def test_progress_compose_base_and_file(self):
        with self.dl._lock:
            self.dl._total = 1000
            self.dl._base_done = 400
            self.dl._file_done = 0
            self.dl._recompute_downloaded_locked()
        self.dl._set_file_progress(150)
        snap = self.dl.snapshot()
        self.assertEqual(snap.downloaded, 550)
        self.assertAlmostEqual(snap.percent or 0, 55.0, places=0)

    def test_clear_cache_blocked_while_downloading(self):
        with self.dl._lock:
            self.dl._state = md.STATE_DOWNLOADING
        ok, msg = self.dl.clear_cache()
        self.assertFalse(ok)
        self.assertIn("取消", msg)

    def test_clear_cache_removes_dir(self):
        root = md.model_cache_dir(self.dl.repo_id)
        root.mkdir(parents=True)
        (root / "dummy").write_text("x")
        with self.dl._lock:
            self.dl._state = md.STATE_NOT_DOWNLOADED
            self.dl._thread = None
        ok, msg = self.dl.clear_cache()
        self.assertTrue(ok)
        self.assertFalse(root.exists())
        self.assertEqual(self.dl.snapshot().state, md.STATE_NOT_DOWNLOADED)

    def test_status_line_ready_and_failed(self):
        snap = md.ProgressSnapshot(state=md.STATE_READY)
        self.assertEqual(snap.status_line(), "已就绪")
        snap = md.ProgressSnapshot(state=md.STATE_FAILED, error="boom")
        self.assertEqual(snap.status_line(), "失败（boom）")

    def test_tqdm_honours_cancel(self):
        self.dl._cancel.set()
        bar = md._ProgressTqdm(total=100, owner=self.dl, filename="f.bin")
        with self.assertRaises(md.DownloadCancelled):
            bar.update(10)


class SnapshotStatusTests(unittest.TestCase):
    def test_downloading_status_includes_file(self):
        snap = md.ProgressSnapshot(
            state=md.STATE_DOWNLOADING,
            downloaded=100,
            total=200,
            speed_bps=1024,
            eta_s=100,
            current_file="model.safetensors",
        )
        line = snap.status_line()
        self.assertIn("下载中", line)
        self.assertIn("model.safetensors", line)
        self.assertIn("%", line)


if __name__ == "__main__":
    unittest.main()
