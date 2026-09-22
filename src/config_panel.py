"""In-app configuration panel — an AppKit editor for ~/.config/jev-jarvis/env.

Opened from the menu-bar item. Saves write the shell-style env file (no config.json);
changes apply after the user restarts the app (explicit first-version choice).
"""

from __future__ import annotations

import threading
from pathlib import Path

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
    NSPopUpButton,
    NSSecureTextField,
    NSTextField,
    NSView,
    NSWindow,
    NSWindowStyleMaskClosable,
    NSWindowStyleMaskTitled,
)
from Foundation import NSObject

import config_io

PANEL_W, PANEL_H = 520, 720
SECTION_GAP = 8
ROW_H = 26
LABEL_W = 88
FIELD_X = 100
FIELD_W = 390


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


class ConfigPanelController(NSObject):
    """Floating config window. One instance is kept on HudController."""

    def init(self):
        self = objc.super(ConfigPanelController, self).init()
        if self is None:
            return None
        self.window = None
        self._status = None
        self._source_label = None
        # Per-section widgets: dict of name -> control
        self._fields: dict = {}
        self._model_pops: dict = {}
        self._model_manual: dict = {}
        self._key_existing: dict[str, str] = {}  # prefix -> raw key currently on disk
        self._probe_cache: dict[str, list[str]] = {}
        self._build()
        return self

    # ------------------------------------------------------------------ build
    @objc.python_method
    def _build(self):
        style = NSWindowStyleMaskTitled | NSWindowStyleMaskClosable
        self.window = NSWindow.alloc().initWithContentRect_styleMask_backing_defer_(
            NSMakeRect(0, 0, PANEL_W, PANEL_H), style, NSBackingStoreBuffered, False)
        self.window.setTitle_("jev-jarvis 配置")
        self.window.setAppearance_(
            NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameAqua))
        self.window.setBackgroundColor_(PALETTE["bg"])
        self.window.setLevel_(AppKit.NSFloatingWindowLevel)
        self.window.setReleasedWhenClosed_(False)

        view = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, PANEL_W, PANEL_H))
        view.setWantsLayer_(True)
        view.layer().setBackgroundColor_(PALETTE["bg"].CGColor())
        self.window.setContentView_(view)

        # Layout from top (Cocoa y is bottom-up; we track dy_from_top)
        y_top = PANEL_H - 16

        def place(ctrl, x, dy, w, h):
            nonlocal y_top
            # dy is offset from current y_top cursor when used via advance; here absolute
            ctrl.setFrame_(NSMakeRect(x, PANEL_H - dy - h, w, h))
            view.addSubview_(ctrl)

        # Header
        title = self._label(14, 0, PANEL_W - 28, 20, size=14, bold=True)
        title.setStringValue_("配置 Key 与模型（保存到 env 文件）")
        place(title, 14, 16, PANEL_W - 28, 20)

        hint = self._label(14, 0, PANEL_W - 28, 32, size=11, color=PALETTE["muted"])
        hint.setStringValue_(
            "保存后需重启应用才生效。Key 掩码显示；留空 Key 输入框表示保留原值。"
            "模型列表优先从端点探测，失败时回退手填。")
        hint.cell().setWraps_(True)
        place(hint, 14, 40, PANEL_W - 28, 32)

        self._source_label = self._label(14, 0, PANEL_W - 28, 18, size=11,
                                         color=PALETTE["green"])
        place(self._source_label, 14, 76, PANEL_W - 28, 18)

        # Sections stacked
        cursor = 100
        cursor = self._build_section(
            view, cursor, "typesafe", "判断层 · TypeSafe Jev",
            base_presets=config_io.ENDPOINT_PRESETS["typesafe"],
            api_format="typesafe")
        cursor = self._build_section(
            view, cursor, "openai", "生成层 · OpenAI 兼容（优先）",
            base_presets=config_io.ENDPOINT_PRESETS["openai"],
            api_format="openai")
        cursor = self._build_section(
            view, cursor, "anthropic", "生成层 · Anthropic 兼容",
            base_presets=config_io.ENDPOINT_PRESETS["anthropic"],
            api_format="anthropic")

        # Stretch: optional extras (collapsed into one row pair)
        cursor += 6
        extra_title = self._label(14, 0, PANEL_W - 28, 16, size=12, bold=True)
        extra_title.setStringValue_("可选")
        place(extra_title, 14, cursor, PANEL_W - 28, 16)
        cursor += 22
        self._add_row(view, cursor, "extra_boxes", "JEV_BOXES",
                      placeholder="1 开 / 空关")
        cursor += ROW_H + 6
        self._add_row(view, cursor, "extra_body", "EXTRA_BODY",
                      placeholder='{"enable_thinking": false}')
        cursor += ROW_H + 10

        # Status + buttons
        self._status = self._label(14, 0, PANEL_W - 28, 36, size=11,
                                   color=PALETTE["muted"])
        self._status.cell().setWraps_(True)
        self._status.setStringValue_("")
        place(self._status, 14, cursor, PANEL_W - 28, 36)
        cursor += 44

        save_btn = self._button(14, 0, 120, 28, "保存", "saveConfig:")
        place(save_btn, 14, cursor, 120, 28)
        close_btn = self._button(144, 0, 80, 28, "关闭", "closeConfig:")
        place(close_btn, 144, cursor, 80, 28)
        restart_hint = self._label(240, 0, 260, 28, size=11, color=PALETTE["amber"])
        restart_hint.setStringValue_("保存后请重启应用使配置生效")
        place(restart_hint, 240, cursor, 260, 28)

        path_label = self._label(14, 0, PANEL_W - 28, 16, size=10, color=PALETTE["muted"])
        path_label.setStringValue_(
            f"写入：{str(config_io.preferred_write_path()).replace(str(Path.home()), '~')}"
            "  · 权限 600  · 尽量保留注释")
        place(path_label, 14, cursor + 34, PANEL_W - 28, 16)

    @objc.python_method
    def _build_section(self, view, cursor, name, title, *, base_presets, api_format):
        """Build one provider block; returns next cursor (dy from top)."""
        def place(ctrl, x, dy, w, h):
            ctrl.setFrame_(NSMakeRect(x, PANEL_H - dy - h, w, h))
            view.addSubview_(ctrl)

        hdr = self._label(14, 0, PANEL_W - 28, 18, size=12, bold=True)
        hdr.setStringValue_(title)
        place(hdr, 14, cursor, PANEL_W - 28, 18)
        cursor += 22

        # source line under section title
        src = self._label(14, 0, PANEL_W - 28, 14, size=10, color=PALETTE["muted"])
        src.setStringValue_("")
        place(src, 14, cursor, PANEL_W - 28, 14)
        self._fields[f"{name}_source"] = src
        cursor += 18

        self._add_row(view, cursor, f"{name}_base", "端点",
                      placeholder="https://…")
        # preset popup tucked to the right of base field — rebuild as shortcut menu
        preset = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(0, 0, 110, ROW_H), True)
        preset.addItemWithTitle_("快捷端点")
        for label, url in base_presets:
            preset.addItemWithTitle_(label)
            preset.itemAtIndex_(preset.numberOfItems() - 1).setRepresentedObject_(url)
        preset.setTarget_(self)
        preset.setAction_("presetChosen:")
        preset.setTag_({"typesafe": 1, "openai": 2, "anthropic": 3}[name])
        preset.setToolTip_("选一个常见端点填入 base_url（仍可手改）")
        place(preset, PANEL_W - 124, cursor, 110, ROW_H)
        # shrink base field so it doesn't collide
        base_field = self._fields[f"{name}_base"]
        base_field.setFrame_(NSMakeRect(
            FIELD_X, PANEL_H - cursor - ROW_H, FIELD_W - 120, ROW_H))
        cursor += ROW_H + 6

        self._add_secure_row(view, cursor, f"{name}_key", "API Key")
        cursor += ROW_H + 6

        # model: popup + manual fallback field
        lab = self._label(14, 0, LABEL_W, ROW_H, size=12, color=PALETTE["muted"])
        lab.setStringValue_("模型")
        place(lab, 14, cursor, LABEL_W, ROW_H)

        pop = NSPopUpButton.alloc().initWithFrame_pullsDown_(
            NSMakeRect(0, 0, 220, ROW_H), False)
        pop.addItemWithTitle_("（先探测模型列表）")
        pop.setTarget_(self)
        pop.setAction_("modelChosen:")
        place(pop, FIELD_X, cursor, 220, ROW_H)
        self._model_pops[name] = pop
        pop.setTag_({"typesafe": 1, "openai": 2, "anthropic": 3}[name])

        manual = NSTextField.alloc().initWithFrame_(NSMakeRect(0, 0, 160, ROW_H))
        manual.setFont_(NSFont.systemFontOfSize_(12))
        manual.setPlaceholderString_("或手填模型名")
        place(manual, FIELD_X + 228, cursor, 160, ROW_H)
        self._model_manual[name] = manual
        self._fields[f"{name}_model"] = manual
        cursor += ROW_H + 6

        probe_btn = self._button(FIELD_X, 0, 90, 24, "探测模型", "probeModels:")
        probe_btn.setTag_({"typesafe": 1, "openai": 2, "anthropic": 3}[name])
        place(probe_btn, FIELD_X, cursor, 90, 24)
        test_btn = self._button(FIELD_X + 100, 0, 90, 24, "测试连接", "testConnection:")
        test_btn.setTag_({"typesafe": 1, "openai": 2, "anthropic": 3}[name])
        place(test_btn, FIELD_X + 100, cursor, 90, 24)
        cursor += 28 + SECTION_GAP

        # stash api_format for actions
        self._fields[f"{name}_api"] = api_format
        return cursor

    @objc.python_method
    def _add_row(self, view, cursor, key, label, *, placeholder=""):
        lab = self._label(14, 0, LABEL_W, ROW_H, size=12, color=PALETTE["muted"])
        lab.setStringValue_(label)
        lab.setFrame_(NSMakeRect(14, PANEL_H - cursor - ROW_H, LABEL_W, ROW_H))
        view.addSubview_(lab)
        tf = NSTextField.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, PANEL_H - cursor - ROW_H, FIELD_W, ROW_H))
        tf.setFont_(NSFont.systemFontOfSize_(12))
        if placeholder:
            tf.setPlaceholderString_(placeholder)
        view.addSubview_(tf)
        self._fields[key] = tf

    @objc.python_method
    def _add_secure_row(self, view, cursor, key, label):
        lab = self._label(14, 0, LABEL_W, ROW_H, size=12, color=PALETTE["muted"])
        lab.setStringValue_(label)
        lab.setFrame_(NSMakeRect(14, PANEL_H - cursor - ROW_H, LABEL_W, ROW_H))
        view.addSubview_(lab)
        tf = NSSecureTextField.alloc().initWithFrame_(
            NSMakeRect(FIELD_X, PANEL_H - cursor - ROW_H, FIELD_W - 120, ROW_H))
        tf.setFont_(NSFont.systemFontOfSize_(12))
        tf.setPlaceholderString_("留空 = 保留原值")
        view.addSubview_(tf)
        self._fields[key] = tf
        mask = self._label(FIELD_X + FIELD_W - 110, 0, 110, ROW_H,
                           size=10, color=PALETTE["muted"])
        mask.setStringValue_("")
        mask.setFrame_(NSMakeRect(
            FIELD_X + FIELD_W - 110, PANEL_H - cursor - ROW_H, 110, ROW_H))
        view.addSubview_(mask)
        self._fields[f"{key}_mask"] = mask

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

    # ------------------------------------------------------------------ show / load
    @objc.python_method
    def open(self):
        self.reload_from_disk()
        screen = AppKit.NSScreen.mainScreen().visibleFrame()
        frame = self.window.frame()
        self.window.setFrameOrigin_((
            screen.origin.x + (screen.size.width - frame.size.width) / 2,
            screen.origin.y + (screen.size.height - frame.size.height) / 2,
        ))
        AppKit.NSApp.activateIgnoringOtherApps_(True)
        self.window.makeKeyAndOrderFront_(None)

    @objc.python_method
    def reload_from_disk(self):
        snap = config_io.read_snapshot()
        if snap.generation_using_builtin:
            self._source_label.setStringValue_(
                f"生成层当前：内置共享 key（{config_io.builtin_key_fingerprint()}）"
                " — 填入自己的 OPENAI/ANTHROPIC key 后优先生效")
            self._source_label.setTextColor_(PALETTE["amber"])
        else:
            self._source_label.setStringValue_(
                f"生成层当前：自己的 key · 来源 {snap.generation_source}")
            self._source_label.setTextColor_(PALETTE["green"])

        for name, prov in (
            ("typesafe", snap.typesafe),
            ("openai", snap.openai),
            ("anthropic", snap.anthropic),
        ):
            self._key_existing[name] = prov.key
            self._fields[f"{name}_base"].setStringValue_(prov.base or "")
            self._fields[f"{name}_key"].setStringValue_("")  # never echo plaintext
            mask = self._fields[f"{name}_key_mask"]
            if prov.key_configured:
                mask.setStringValue_(config_io.mask_key(prov.key))
            else:
                mask.setStringValue_("（未配置）")
            src = self._fields[f"{name}_source"]
            if prov.key_configured:
                src.setStringValue_(f"来源：{prov.source} · {config_io.mask_key(prov.key)}")
            else:
                src.setStringValue_("来源：未配置"
                                    + ("（判断将回退本地 decider-2b）" if name == "typesafe"
                                       else ""))
            manual = self._model_manual[name]
            manual.setStringValue_(prov.model or "")
            pop = self._model_pops[name]
            pop.removeAllItems()
            pop.addItemWithTitle_(prov.model or "（先探测模型列表）")

        self._fields["extra_boxes"].setStringValue_(snap.extra.get("JEV_BOXES") or "")
        self._fields["extra_body"].setStringValue_(
            snap.extra.get("OPENAI_EXTRA_BODY") or "")
        self._set_status(
            f"已加载 · 写入目标 {str(snap.write_path).replace(str(Path.home()), '~')}",
            PALETTE["muted"])

    # ------------------------------------------------------------------ helpers
    @objc.python_method
    def _set_status(self, text: str, color=None):
        self._status.setStringValue_(text)
        self._status.setTextColor_(color or PALETTE["muted"])

    @objc.python_method
    def _name_for_tag(self, tag: int) -> str:
        return {1: "typesafe", 2: "openai", 3: "anthropic"}.get(int(tag), "openai")

    @objc.python_method
    def _resolve_key(self, name: str) -> str:
        typed = self._fields[f"{name}_key"].stringValue().strip()
        if typed:
            return typed
        return self._key_existing.get(name) or ""

    @objc.python_method
    def _resolve_model(self, name: str) -> str:
        manual = self._model_manual[name].stringValue().strip()
        if manual:
            return manual
        pop = self._model_pops[name]
        title = pop.titleOfSelectedItem() or ""
        if title.startswith("（"):
            return ""
        return title

    # ------------------------------------------------------------------ actions
    def presetChosen_(self, sender):
        name = self._name_for_tag(sender.tag())
        item = sender.selectedItem()
        url = item.representedObject() if item else None
        if url:
            self._fields[f"{name}_base"].setStringValue_(str(url))
        sender.selectItemAtIndex_(0)

    def modelChosen_(self, sender):
        name = self._name_for_tag(sender.tag())
        title = sender.titleOfSelectedItem() or ""
        if title and not title.startswith("（"):
            self._model_manual[name].setStringValue_(title)

    def probeModels_(self, sender):
        name = self._name_for_tag(sender.tag())
        base = self._fields[f"{name}_base"].stringValue().strip()
        key = self._resolve_key(name)
        api = "anthropic" if name == "anthropic" else "openai"
        self._set_status(f"正在探测 {name} 模型列表…", PALETTE["muted"])

        def work():
            if name == "typesafe":
                # TypeSafe has no public models list comparable to OpenAI; use fallbacks
                ids, err = list(config_io.MODEL_FALLBACKS["typesafe"]), ""
            else:
                ids, err = config_io.probe_models(base, key, api_format=api)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "probeDone:", (name, ids, err), False)

        threading.Thread(target=work, daemon=True).start()

    def probeDone_(self, payload):
        name, ids, err = payload
        pop = self._model_pops[name]
        pop.removeAllItems()
        if ids:
            self._probe_cache[name] = list(ids)
            for mid in ids:
                pop.addItemWithTitle_(mid)
            current = self._model_manual[name].stringValue().strip()
            if current and current in ids:
                pop.selectItemWithTitle_(current)
            elif ids:
                pop.selectItemAtIndex_(0)
                if not current:
                    self._model_manual[name].setStringValue_(ids[0])
            self._set_status(f"{name}：探测到 {len(ids)} 个模型", PALETTE["green"])
        else:
            for mid in config_io.MODEL_FALLBACKS.get(
                    "typesafe" if name == "typesafe" else name, []):
                pop.addItemWithTitle_(mid)
            pop.addItemWithTitle_("（探测失败 — 请手填）")
            self._set_status(
                f"{name} 探测失败：{err or 'unknown'}（已提供常用 fallback，可手填）",
                PALETTE["amber"])

    def testConnection_(self, sender):
        name = self._name_for_tag(sender.tag())
        base = self._fields[f"{name}_base"].stringValue().strip()
        key = self._resolve_key(name)
        model = self._resolve_model(name)
        self._set_status(f"正在测试 {name} 连接…", PALETTE["muted"])

        def work():
            if name == "typesafe":
                ok, msg = config_io.test_typesafe_connection(base, key, model)
            elif name == "anthropic":
                ok, msg = config_io.test_anthropic_connection(base, key, model)
            else:
                ok, msg = config_io.test_openai_connection(base, key, model)
            self.performSelectorOnMainThread_withObject_waitUntilDone_(
                "testDone:", (ok, msg), False)

        threading.Thread(target=work, daemon=True).start()

    def testDone_(self, payload):
        ok, msg = payload
        self._set_status(msg, PALETTE["green"] if ok else PALETTE["red"])

    def saveConfig_(self, sender):
        def key_update(name):
            typed = self._fields[f"{name}_key"].stringValue().strip()
            if typed:
                return typed
            return None  # leave unchanged

        try:
            path = config_io.apply_panel_updates(
                typesafe_key=key_update("typesafe"),
                typesafe_base=self._fields["typesafe_base"].stringValue().strip() or None,
                typesafe_model=self._resolve_model("typesafe") or None,
                openai_key=key_update("openai"),
                openai_base=self._fields["openai_base"].stringValue().strip() or None,
                openai_model=self._resolve_model("openai") or None,
                anthropic_key=key_update("anthropic"),
                anthropic_base=self._fields["anthropic_base"].stringValue().strip() or None,
                anthropic_model=self._resolve_model("anthropic") or None,
                extra={
                    k: v for k, v in {
                        "JEV_BOXES": self._fields["extra_boxes"].stringValue().strip(),
                        "OPENAI_EXTRA_BODY": self._fields["extra_body"].stringValue().strip(),
                    }.items() if v  # only write optional stretch fields when set
                } or None,
            )
        except OSError as e:
            self._set_status(f"保存失败：{e}", PALETTE["red"])
            return

        # Refresh mask labels from what we just wrote (re-read file via userconfig parse)
        # Do NOT mutate os.environ — restart-to-apply is intentional.
        shown = str(path).replace(str(Path.home()), "~")
        self._set_status(
            f"已保存到 {shown}（权限 600）。请重启应用使配置生效。",
            PALETTE["green"])
        # Update masks for keys the user just typed
        for name in ("typesafe", "openai", "anthropic"):
            typed = self._fields[f"{name}_key"].stringValue().strip()
            if typed:
                self._key_existing[name] = typed
                self._fields[f"{name}_key"].setStringValue_("")
                self._fields[f"{name}_key_mask"].setStringValue_(
                    config_io.mask_key(typed))

    def closeConfig_(self, sender):
        self.window.orderOut_(None)


def open_config_panel(owner) -> ConfigPanelController:
    """Create (once) and show the config panel; stash on ``owner._config_panel``."""
    panel = getattr(owner, "_config_panel", None)
    if panel is None:
        panel = ConfigPanelController.alloc().init()
        owner._config_panel = panel
    panel.open()
    return panel
