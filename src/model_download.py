"""Controllable Hugging Face snapshot download for the local judge model.

Thread-safe progress state for AppKit polling. Pause/cancel are cooperative:
flags are checked between files and inside a custom tqdm stand-in during byte
transfer. Cancel aborts the network transfer but leaves the HF hub cache
intact so a later start resumes incomplete blobs.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DEFAULT_REPO = "Mapika/decider-2b"

STATE_NOT_DOWNLOADED = "未下载"
STATE_DOWNLOADING = "下载中"
STATE_PAUSED = "已暂停"
STATE_READY = "已就绪"
STATE_FAILED = "失败"


class DownloadCancelled(Exception):
    """Raised inside the download thread when the user cancels."""


def format_bytes(n: float | int | None) -> str:
    """Human-readable byte size (base 1024)."""
    if n is None:
        return "—"
    try:
        n = float(n)
    except (TypeError, ValueError):
        return "—"
    if n < 0:
        n = 0.0
    units = ("B", "KB", "MB", "GB", "TB")
    v = float(n)
    for u in units:
        if v < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(v)} {u}"
            return f"{v:.1f} {u}"
        v /= 1024.0
    return f"{v:.1f} TB"


def format_speed(bps: float | None) -> str:
    if not bps or bps <= 0:
        return "—"
    return format_bytes(bps) + "/s"


def format_eta(seconds: float | None) -> str:
    if seconds is None or seconds < 0 or seconds == float("inf"):
        return "—"
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def hub_cache_root() -> Path:
    """Resolve HF hub cache, respecting HF_HOME / HUGGINGFACE_HUB_CACHE."""
    env = os.environ.get("HUGGINGFACE_HUB_CACHE")
    if env:
        return Path(env).expanduser()
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
        return Path(HF_HUB_CACHE)
    except Exception:
        return Path.home() / ".cache" / "huggingface" / "hub"


def model_cache_dir(repo_id: str = DEFAULT_REPO) -> Path:
    return hub_cache_root() / ("models--" + repo_id.replace("/", "--"))


def _has_weight_files(snap_dir: Path) -> bool:
    if not snap_dir.is_dir():
        return False
    for pat in ("*.safetensors", "pytorch_model.bin", "model.bin", "*.gguf"):
        if any(snap_dir.glob(pat)):
            return True
    if (snap_dir / "model.safetensors.index.json").exists():
        return any(snap_dir.glob("*.safetensors"))
    return False


def is_ready(repo_id: str = DEFAULT_REPO) -> bool:
    """True when a complete local snapshot with weights is available (no network)."""
    try:
        from huggingface_hub import snapshot_download

        snapshot_download(repo_id, local_files_only=True)
        return True
    except Exception:
        pass
    root = model_cache_dir(repo_id)
    snaps = root / "snapshots"
    if not snaps.is_dir():
        return False
    return any(_has_weight_files(snap) for snap in snaps.iterdir())


def estimate_partial_bytes(repo_id: str = DEFAULT_REPO) -> int:
    """Sum sizes of incomplete blobs (best-effort resume hint)."""
    root = model_cache_dir(repo_id)
    total = 0
    if not root.is_dir():
        return 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if ".incomplete" in p.name:
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _cached_path(repo_id: str, filename: str) -> Path | None:
    try:
        from huggingface_hub import try_to_load_from_cache
        from huggingface_hub.file_download import _CACHED_NO_EXIST

        cached = try_to_load_from_cache(repo_id, filename)
        if cached is None or cached is _CACHED_NO_EXIST:
            return None
        path = Path(str(cached))
        return path if path.is_file() else None
    except Exception:
        return None


@dataclass
class ProgressSnapshot:
    state: str = STATE_NOT_DOWNLOADED
    downloaded: int = 0
    total: int = 0
    speed_bps: float = 0.0
    eta_s: float | None = None
    current_file: str = ""
    error: str = ""
    repo_id: str = DEFAULT_REPO
    cache_dir: str = ""

    @property
    def percent(self) -> float | None:
        if self.total <= 0:
            return None
        return min(100.0, 100.0 * self.downloaded / self.total)

    def status_line(self) -> str:
        if self.state == STATE_READY:
            return "已就绪"
        if self.state == STATE_FAILED:
            return f"失败（{self.error or '未知错误'}）"
        if self.state == STATE_NOT_DOWNLOADED:
            return "未下载"
        pct = self.percent
        pct_s = f"{pct:.0f}%" if pct is not None else "—"
        bits = [
            self.state,
            f"{format_bytes(self.downloaded)}/"
            f"{format_bytes(self.total) if self.total else '—'}",
            pct_s,
            format_speed(self.speed_bps),
            f"ETA {format_eta(self.eta_s)}",
        ]
        if self.current_file:
            bits.append(self.current_file)
        return " · ".join(bits)


class _ProgressTqdm:
    """Minimal tqdm stand-in: reports absolute file progress to ModelDownloader."""

    def __init__(self, *args, **kwargs):
        self._owner: ModelDownloader | None = kwargs.pop("owner", None)
        self._filename: str = kwargs.pop("filename", "") or ""
        total = kwargs.get("total")
        self.total = int(total or 0)
        self.n = int(kwargs.get("initial") or 0)
        self.desc = kwargs.get("desc") or self._filename
        self.disable = kwargs.get("disable", False)
        if self._owner is not None:
            if self._filename:
                self._owner._set_current_file(self._filename)
            # HF may pass initial= already-downloaded for resume
            self._owner._set_file_progress(self.n)

    def update(self, n: float | int | None = 1):
        if self._owner is None:
            return
        self._owner._cooperative_gate()
        delta = int(n or 0)
        if delta:
            self.n += delta
            self._owner._set_file_progress(self.n)

    def close(self):
        pass

    def clear(self):
        pass

    def refresh(self):
        pass

    def reset(self, total=None):
        if total is not None:
            self.total = int(total)
        self.n = 0

    def set_description(self, desc=None, refresh=True):
        if desc is not None:
            self.desc = desc

    def set_description_str(self, desc=None, refresh=True):
        self.set_description(desc)

    def set_postfix(self, *args, **kwargs):
        pass

    def set_postfix_str(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __iter__(self):
        return iter(())


class ModelDownloader:
    """Download controller for one repo; thread-safe snapshot() for UI polling."""

    def __init__(self, repo_id: str = DEFAULT_REPO):
        self.repo_id = repo_id
        self._lock = threading.RLock()
        self._state = STATE_NOT_DOWNLOADED
        self._downloaded = 0
        self._total = 0
        self._speed_bps = 0.0
        self._eta_s: float | None = None
        self._current_file = ""
        self._error = ""
        self._pause_gate = threading.Event()
        self._pause_gate.set()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[ProgressSnapshot], None]] = []
        self._speed_origin_t = 0.0
        self._speed_origin_bytes = 0
        # Progress composition: completed files + current file bytes
        self._base_done = 0
        self._file_done = 0
        self._refresh_initial_state()

    def _refresh_initial_state(self) -> None:
        if is_ready(self.repo_id):
            self._state = STATE_READY
            self._error = ""
        else:
            self._state = STATE_NOT_DOWNLOADED
            self._downloaded = estimate_partial_bytes(self.repo_id)

    def snapshot(self) -> ProgressSnapshot:
        with self._lock:
            return ProgressSnapshot(
                state=self._state,
                downloaded=self._downloaded,
                total=self._total,
                speed_bps=self._speed_bps,
                eta_s=self._eta_s,
                current_file=self._current_file,
                error=self._error,
                repo_id=self.repo_id,
                cache_dir=str(model_cache_dir(self.repo_id)),
            )

    def add_listener(self, cb: Callable[[ProgressSnapshot], None]) -> None:
        self._listeners.append(cb)

    def _emit(self) -> None:
        snap = self.snapshot()
        for cb in list(self._listeners):
            try:
                cb(snap)
            except Exception:
                pass

    def _set_state(self, state: str, *, error: str = "") -> None:
        with self._lock:
            self._state = state
            if error:
                self._error = error
            elif state != STATE_FAILED:
                self._error = ""
        self._emit()

    def _set_current_file(self, name: str) -> None:
        with self._lock:
            self._current_file = name
            self._file_done = 0
            self._recompute_downloaded_locked()

    def _set_file_progress(self, n: int) -> None:
        now = time.monotonic()
        with self._lock:
            self._file_done = max(0, int(n))
            self._recompute_downloaded_locked()
            if self._speed_origin_t <= 0:
                self._speed_origin_t = now
                self._speed_origin_bytes = self._downloaded
            elapsed = now - self._speed_origin_t
            if elapsed >= 0.4:
                gained = self._downloaded - self._speed_origin_bytes
                self._speed_bps = gained / elapsed if elapsed > 0 else 0.0
                self._speed_origin_t = now
                self._speed_origin_bytes = self._downloaded
            if self._total > 0 and self._speed_bps > 0:
                remain = max(0, self._total - self._downloaded)
                self._eta_s = remain / self._speed_bps
            else:
                self._eta_s = None
        self._emit()

    def _recompute_downloaded_locked(self) -> None:
        self._downloaded = self._base_done + self._file_done

    def _mark_file_complete(self, size: int) -> None:
        with self._lock:
            self._base_done += max(0, size)
            self._file_done = 0
            self._recompute_downloaded_locked()
        self._emit()

    def _cooperative_gate(self) -> None:
        if self._cancel.is_set():
            raise DownloadCancelled("用户取消")
        while not self._pause_gate.is_set():
            if self._cancel.is_set():
                raise DownloadCancelled("用户取消")
            self._pause_gate.wait(timeout=0.2)

    def start(self) -> bool:
        """Start or continue download. False if already running or ready."""
        with self._lock:
            if self._state == STATE_READY and is_ready(self.repo_id):
                return False
            if self._thread is not None and self._thread.is_alive():
                if self._state == STATE_PAUSED:
                    self._pause_gate.set()
                    self._state = STATE_DOWNLOADING
                    self._emit()
                    return True
                return False
            self._cancel.clear()
            self._pause_gate.set()
            self._error = ""
            self._speed_bps = 0.0
            self._eta_s = None
            self._speed_origin_t = 0.0
            self._speed_origin_bytes = self._downloaded
            self._state = STATE_DOWNLOADING
            self._thread = threading.Thread(
                target=self._run, name="model-download", daemon=True)
            self._thread.start()
        self._emit()
        return True

    def pause(self) -> None:
        with self._lock:
            if self._state != STATE_DOWNLOADING:
                return
            self._pause_gate.clear()
            self._state = STATE_PAUSED
            self._speed_bps = 0.0
            self._eta_s = None
        self._emit()

    def cancel(self) -> None:
        """Stop transfer; leave partial HF cache for resume."""
        self._cancel.set()
        self._pause_gate.set()
        with self._lock:
            if self._state in (STATE_DOWNLOADING, STATE_PAUSED):
                self._state = STATE_NOT_DOWNLOADED
                self._current_file = ""
                self._speed_bps = 0.0
                self._eta_s = None
                self._error = ""
        self._emit()

    def refresh_ready(self) -> bool:
        """Re-probe cache; flip to READY / 未下载 when idle."""
        with self._lock:
            if self._state in (STATE_DOWNLOADING, STATE_PAUSED):
                return self._state == STATE_READY
            if self._thread is not None and self._thread.is_alive():
                return False
        ready = is_ready(self.repo_id)
        if ready:
            self._set_state(STATE_READY)
        elif self.snapshot().state == STATE_READY:
            self._set_state(STATE_NOT_DOWNLOADED)
        return ready

    def clear_cache(self) -> tuple[bool, str]:

        with self._lock:
            if self._state in (STATE_DOWNLOADING, STATE_PAUSED):
                return False, "下载进行中，请先取消再清除缓存"
            if self._thread is not None and self._thread.is_alive():
                return False, "下载线程尚未结束，请稍后再试"
        root = model_cache_dir(self.repo_id)
        if not root.exists():
            with self._lock:
                self._downloaded = 0
                self._total = 0
                self._base_done = 0
                self._file_done = 0
                self._state = STATE_NOT_DOWNLOADED
                self._error = ""
            self._emit()
            return True, "缓存目录不存在（已是干净状态）"
        try:
            shutil.rmtree(root)
        except OSError as e:
            return False, f"清除失败：{e}"
        with self._lock:
            self._downloaded = 0
            self._total = 0
            self._base_done = 0
            self._file_done = 0
            self._current_file = ""
            self._error = ""
            self._state = STATE_NOT_DOWNLOADED
        self._emit()
        return True, f"已清除 {root}"

    def _tqdm_factory(self, filename: str):
        owner = self

        class _Bound(_ProgressTqdm):
            def __init__(self, *a, **kw):
                kw.setdefault("owner", owner)
                kw.setdefault("filename", filename)
                super().__init__(*a, **kw)

        return _Bound

    def _list_files(self) -> list[tuple[str, int]]:
        from huggingface_hub import HfApi

        info = HfApi().model_info(self.repo_id, files_metadata=True)
        out: list[tuple[str, int]] = []
        for s in info.siblings or []:
            out.append((s.rfilename, int(getattr(s, "size", None) or 0)))
        return out

    def _run(self) -> None:
        try:
            files = self._list_files()
            total = sum(sz for _, sz in files)
            already = 0
            pending: list[tuple[str, int]] = []
            for name, size in files:
                if _cached_path(self.repo_id, name) is not None:
                    already += size
                else:
                    pending.append((name, size))

            with self._lock:
                self._total = total
                self._base_done = already
                self._file_done = 0
                # incomplete blobs for the first pending file are tracked via tqdm initial
                self._recompute_downloaded_locked()
                self._speed_origin_t = time.monotonic()
                self._speed_origin_bytes = self._downloaded
            self._emit()

            from huggingface_hub import hf_hub_download

            for name, size in pending:
                self._cooperative_gate()
                self._set_current_file(name)
                hf_hub_download(
                    self.repo_id,
                    filename=name,
                    tqdm_class=self._tqdm_factory(name),
                )
                self._mark_file_complete(size if size > 0 else self._file_done)

            if is_ready(self.repo_id):
                with self._lock:
                    self._state = STATE_READY
                    self._current_file = ""
                    self._base_done = max(self._base_done, self._total)
                    self._file_done = 0
                    self._recompute_downloaded_locked()
                    self._speed_bps = 0.0
                    self._eta_s = None
                    self._error = ""
                self._emit()
            else:
                self._set_state(STATE_FAILED, error="下载结束但权重未就绪")
        except DownloadCancelled:
            with self._lock:
                self._state = STATE_NOT_DOWNLOADED
                self._current_file = ""
                self._speed_bps = 0.0
                self._eta_s = None
            self._emit()
        except Exception as e:
            self._set_state(STATE_FAILED, error=f"{type(e).__name__}: {str(e)[:120]}")
        finally:
            with self._lock:
                self._thread = None


_default: ModelDownloader | None = None
_default_lock = threading.Lock()


def get_downloader(repo_id: str = DEFAULT_REPO) -> ModelDownloader:
    global _default
    with _default_lock:
        if _default is None or _default.repo_id != repo_id:
            _default = ModelDownloader(repo_id)
        return _default
