"""AppKit panel: visual download controls for the local judge model (decider-2b).

Opened from the menu-bar item 「本地判断模型…」. All network work runs off the
main thread; UI is refreshed via NSTimer + performSelectorOnMainThread.
"""

from __future__ import annotations

import threading

import objc
import AppKit
from AppKit import (
    NSAppearance,
    NSBackingStoreBuffered,
    NSBezelStyleRounded,
    NSButton,
    NSColor,
    NSFont,
    NSMakeRect,
    NSTextField,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject, NSTimer

import model_download as md

PANEL_W, PANEL_H = 460, 320


def _rgb(hex_code: int, alpha: float = 1.0) -> NSColor:
    return NSColor.colorWithCalibratedRed_green_blue_alpha_(
        ((hex_code >> 16) & 0xFF) / 255.0,
        ((hex_code >> 8) & 0xFF) / 255.0,
        (hex_code & 0xFF) / 255.0,
        alpha,
    )


PALETTE = {
    "bg": _rgb(0xF7F7F7),
    "text": _rgb(0x191919),
    "muted": _rgb(0x888888),
    "green": _rgb(0x07C160),
    "amber": _rgb(0xFA9D3B),
    "red": _rgb(0xFA5151),
}


class DownloadPanelController(NSObject):
    """Floating window for 本地判断模型 download state + controls."""

    def init(self):
        self = objc.super(DownloadPanelController, self).init()
        if self is None:
            return None
        self.window = None
        self._downloader = md.get_downloader()
        self._poll: NSTimer | None = None
        self._state_label = None
        self._progress_label = None
        self._detail_label = None
        self._cache_label = None
        self._hint_label = None
        self._btn_start = None
        self._btn_pause = None
        self._btn_cancel = None
        self._btn_clear = None
        self._build()
        return self

    @objc.python_method
    def _build(self):
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PANEL_W, PANEL_H), style, NSBackingStoreBuffered, False)
        self.window.setTitle_("本地判断模型")
        self.window.setAppearance_(
            NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameAqua))
        self.window.setBackgroundColor_(PALETTE["bg"])
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)
        self.window.setReleasedWhenClosed_(False)

        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PANEL_H))
        view.setWantsLayer_(True)
        view.layer().setBackgroundColor_(PALETTE["bg"].CGColor())
        self.window.setContentView_(view)

        def place(ctrl, x, dy, w, h):
            ctrl.setFrame_(NSMakeRect(x, PANEL_H - dy - h, w, h))
            view.addSubview_(ctrl)

        title = self._label(14, 0, PANEL_W - 28, 20, size=14, bold=True)
        title.setStringValue_(f"本地判断模型 · {md.DEFAULT_REPO}")
        place(title, 14, 16, PANEL_W - 28, 20)

        hint = self._label(14, 0, PANEL_W - 28, 36, size=11, color=PALETTE["muted"])
        hint.setStringValue_(
            "首次使用需下载权重（约 3.5–4 GB）。暂停/取消会保留 Hugging Face "
            "缓存中的未完成文件，下次「开始/继续」可续传。「清除缓存」会删掉整份模型目录。"
        )
        hint.cell().setWraps_(True)
        place(hint, 14, 42, PANEL_W - 28, 36)
        self._hint_label = hint

        self._state_label = self._label(14, 0, PANEL_W - 28, 22, size=13, bold=True)
        self._state_label.setStringValue_("状态：…")
        place(self._state_label, 14, 88, PANEL_W - 28, 22)

        self._progress_label = self._label(14, 0, PANEL_W - 28, 18, size=12)
        self._progress_label.setStringValue_("")
        place(self._progress_label, 14, 114, PANEL_W - 28, 18)

        self._detail_label = self._label(14, 0, PANEL_W - 28, 18, size=11,
                                         color=PALETTE["muted"])
        self._detail_label.setStringValue_("")
        place(self._detail_label, 14, 136, PANEL_W - 28, 18)

        self._cache_label = self._label(14, 0, PANEL_W - 28, 32, size=10,
                                        color=PALETTE["muted"])
        self._cache_label.cell().setWraps_(True)
        self._cache_label.setStringValue_("")
        place(self._cache_label, 14, 158, PANEL_W - 28, 32)

        self._btn_start = self._button(14, 0, 100, 28, "开始/继续", "startDownload:")
        place(self._btn_start, 14, 210, 100, 28)
        self._btn_pause = self._button(124, 0, 80, 28, "暂停", "pauseDownload:")
        place(self._btn_pause, 124, 210, 80, 28)
        self._btn_cancel = self._button(214, 0, 80, 28, "取消", "cancelDownload:")
        place(self._btn_cancel, 214, 210, 80, 28)
        self._btn_clear = self._button(304, 0, 100, 28, "清除缓存", "clearCache:")
        place(self._btn_clear, 304, 210, 100, 28)

        close_btn = self._button(14, 0, 80, 28, "关闭", "closePanel:")
        place(close_btn, 14, 250, 80, 28)

        note = self._label(104, 0, PANEL_W - 118, 40, size=10, color=PALETTE["muted"])
        note.setStringValue_(
            "下载在后台线程进行，不会卡住悬浮窗。"
            "模型就绪后，启动预热会加载权重并做一次空跑前向。"
        )
        note.cell().setWraps_(True)
        place(note, 104, 248, PANEL_W - 118, 40)

    @objc.python_method
    def _label(self, x, y, w, h, size=13, color=None, bold=False):
        tf = NSTextField.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        tf.setStringValue_("")
        tf.setBezeled_(False)
        tf.setDrawsBackground_(False)
        tf.setEditable_(False)
        tf.setSelectable_(True)
        tf.setTextColor_(PALETTE["text"] if color is None else color)
        tf.setFont_(NSFont.boldSystemFontOfSize_(size) if bold
                    else NSFont.systemFontOfSize_(size))
        return tf

    @objc.python_method
    def _button(self, x, y, w, h, title, action):
        btn = NSButton.alloc().initWithFrame_(NSMakeRect(x, y, w, h))
        btn.setTitle_(title)
        btn.setBezelStyle_(NSBezelStyleRounded)
        btn.setFont_(NSFont.systemFontOfSize_(12))
        btn.setContentTintColor_(PALETTE["green"])
        btn.setTarget_(self)
        btn.setAction_(action)
        return btn

    @objc.python_method
    def open(self):
        self._downloader = md.get_downloader()
        self._downloader.refresh_ready()
        self._apply_snapshot(self._downloader.snapshot())
        screen = AppKit.NSScreen.mainScreen().visibleFrame()
        frame = self.window.frame()
        self.window.setFrameOrigin_((
            screen.origin.x + (screen.size.width - frame.size.width) / 2,
            screen.origin.y + (screen.size.height - frame.size.height) / 2,
        ))
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)
        self._start_poll()

    @objc.python_method
    def _start_poll(self):
        if self._poll is not None:
            self._poll.invalidate()
        self._poll = NSTimer.scheduledTimerWithTimeInterval_target_selector_userInfo_repeats_(
            0.35, self, "pollTick:", None, True)
        AppKit.NSRunLoop.currentRunLoop().addTimer_forMode_(
            self._poll, AppKit.NSDefaultRunLoopMode)

    def pollTick_(self, _timer):
        if self.window is None or not self.window.isVisible():
            if self._poll is not None:
                self._poll.invalidate()
                self._poll = None
            return
        self._apply_snapshot(self._downloader.snapshot())

    @objc.python_method
    def _apply_snapshot(self, snap: md.ProgressSnapshot):
        color = PALETTE["muted"]
        if snap.state == md.STATE_READY:
            color = PALETTE["green"]
        elif snap.state == md.STATE_FAILED:
            color = PALETTE["red"]
        elif snap.state in (md.STATE_DOWNLOADING, md.STATE_PAUSED):
            color = PALETTE["amber"]
        self._state_label.setTextColor_(color)
        self._state_label.setStringValue_(f"状态：{snap.state}"
                                          + (f"（{snap.error}）" if snap.error else ""))

        if snap.total > 0 or snap.downloaded > 0:
            pct = snap.percent
            pct_s = f"{pct:.0f}%" if pct is not None else "—"
            self._progress_label.setStringValue_(
                f"{md.format_bytes(snap.downloaded)} / "
                f"{md.format_bytes(snap.total) if snap.total else '—'}  ·  {pct_s}"
            )
        else:
            self._progress_label.setStringValue_("进度：—")

        parts = [
            f"速度 {md.format_speed(snap.speed_bps)}",
            f"ETA {md.format_eta(snap.eta_s)}",
        ]
        if snap.current_file:
            parts.append(f"文件 {snap.current_file}")
        self._detail_label.setStringValue_(" · ".join(parts))

        cache = snap.cache_dir.replace(str(__import__("pathlib").Path.home()), "~")
        self._cache_label.setStringValue_(f"缓存：{cache}")

        busy = snap.state in (md.STATE_DOWNLOADING, md.STATE_PAUSED)
        ready = snap.state == md.STATE_READY
        self._btn_start.setEnabled_(not ready and snap.state != md.STATE_DOWNLOADING)
        # When paused, start is the continue path
        if snap.state == md.STATE_PAUSED:
            self._btn_start.setEnabled_(True)
            self._btn_start.setTitle_("继续")
        elif ready:
            self._btn_start.setTitle_("开始/继续")
            self._btn_start.setEnabled_(False)
        else:
            self._btn_start.setTitle_("开始/继续")
        self._btn_pause.setEnabled_(snap.state == md.STATE_DOWNLOADING)
        self._btn_cancel.setEnabled_(busy)
        self._btn_clear.setEnabled_(not busy)

    def startDownload_(self, _sender):
        self._downloader.start()
        self._apply_snapshot(self._downloader.snapshot())

    def pauseDownload_(self, _sender):
        self._downloader.pause()
        self._apply_snapshot(self._downloader.snapshot())

    def cancelDownload_(self, _sender):
        self._downloader.cancel()
        self._apply_snapshot(self._downloader.snapshot())

    def clearCache_(self, _sender):
        def work():
            ok, msg = self._downloader.clear_cache()
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "clearDone:", (ok, msg), False)

        threading.Thread(target=work, daemon=True).start()

    def clearDone_(self, payload):
        ok, msg = payload
        self._apply_snapshot(self._downloader.snapshot())
        self._detail_label.setTextColor_(PALETTE["green"] if ok else PALETTE["red"])
        self._detail_label.setStringValue_(msg)

    def closePanel_(self, _sender):
        if self._poll is not None:
            self._poll.invalidate()
            self._poll = None
        self.window.orderOut_(None)


def open_download_panel(owner) -> DownloadPanelController:
    """Create (once) and show the download panel; stash on ``owner._download_panel``."""
    panel = getattr(owner, "_download_panel", None)
    if panel is None:
        panel = DownloadPanelController.alloc().init()
        owner._download_panel = panel
    panel.open()
    return panel
