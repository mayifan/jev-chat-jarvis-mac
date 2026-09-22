"""Env-file editor helpers for the in-app config panel.

The panel is only an editor for the shell-style env file — never a second config
format. Writes target ``~/.config/jev-jarvis/env`` (create dir if needed), keep
mode 600, and best-effort preserve comments / unrecognized lines.
"""

from __future__ import annotations

import json
import os
import re
import stat
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

import builtin
import userconfig

# Keys the config UI knows how to edit. Anything else in the file is left alone.
MANAGED_KEYS = (
    "TYPESAFE_API_KEY",
    "TYPESAFE_BASE_URL",
    "TYPESAFE_MODEL",
    "OPENAI_API_KEY",
    "OPENAI_BASE_URL",
    "OPENAI_MODEL",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_BASE_URL",
    "ANTHROPIC_MODEL",
    "LLM_MODEL",
    "OPENAI_EXTRA_BODY",
    "JEV_BOXES",
    "JEV_TONES",
)

# Common endpoint shortcuts shown when model probe fails / as quick picks.
ENDPOINT_PRESETS = {
    "typesafe": [
        ("TypeSafe 官方", "https://api.typesafe.ai"),
    ],
    "openai": [
        ("DeepSeek", "https://api.deepseek.com"),
        ("OpenAI", "https://api.openai.com/v1"),
        ("智谱 OpenAI 兼容", "https://open.bigmodel.cn/api/paas/v4"),
        ("SiliconFlow", "https://api.siliconflow.cn/v1"),
        ("Ollama 本地", "http://localhost:11434/v1"),
        ("OpenRouter", "https://openrouter.ai/api/v1"),
    ],
    "anthropic": [
        ("Anthropic 官方", "https://api.anthropic.com"),
        ("智谱 Anthropic 兼容", "https://open.bigmodel.cn/api/anthropic"),
    ],
}

# Fallback model names only used when the models API fails.
MODEL_FALLBACKS = {
    "typesafe": ["jev-latest"],
    "openai": ["deepseek-chat", "deepseek-v4-flash", "glm-4-flash", "gpt-4o-mini",
               "qwen2.5:7b"],
    "anthropic": ["claude-3-5-haiku-latest", "glm-4-flash"],
}

_KEY_LINE = re.compile(
    r"^(\s*(?:export\s+)?)([A-Za-z_][A-Za-z0-9_]*)(\s*=\s*)(.*)$"
)


@dataclass
class ProviderSnapshot:
    """One provider's effective settings as shown in the config UI."""

    prefix: str                 # TYPESAFE / OPENAI / ANTHROPIC
    key: str                    # raw key (never echo full in UI)
    base: str
    model: str
    source: str                 # file path label / 环境变量 / 内置默认 / none
    using_builtin: bool = False
    key_configured: bool = False


@dataclass
class ConfigSnapshot:
    typesafe: ProviderSnapshot
    openai: ProviderSnapshot
    anthropic: ProviderSnapshot
    generation_source: str      # which path load_credentials would use
    generation_using_builtin: bool
    write_path: Path
    extra: dict[str, str] = field(default_factory=dict)  # stretch fields


def preferred_write_path() -> Path:
    """Canonical write target: ~/.config/jev-jarvis/env (dev-tool convention)."""
    return userconfig.ENV_FILE


def mask_key(key: str) -> str:
    """Mask an API key the same way generate.credential_status does."""
    if not key:
        return "（未配置）"
    if len(key) <= 10:
        return "•" * len(key)
    return f"{key[:6]}…{key[-4:]}  ({len(key)} chars)"


def source_label(source: str) -> str:
    """Human-readable source, with home collapsed to ~."""
    if not source or source == "none":
        return "未配置"
    home = str(Path.home())
    return source.replace(home, "~")


def _typesafe_snapshot() -> ProviderSnapshot:
    key = userconfig.get("TYPESAFE_API_KEY", "JEV_API_KEY")
    base = userconfig.get("TYPESAFE_BASE_URL") or "https://api.typesafe.ai"
    model = userconfig.get("TYPESAFE_MODEL") or "jev-latest"
    src = userconfig.source_of("TYPESAFE_API_KEY", "JEV_API_KEY")
    return ProviderSnapshot(
        prefix="TYPESAFE",
        key=key,
        base=base,
        model=model,
        source=source_label(src),
        using_builtin=False,
        key_configured=bool(key),
    )


def _gen_provider_snapshot(prefix: str) -> ProviderSnapshot:
    info = userconfig.provider(prefix)
    return ProviderSnapshot(
        prefix=prefix,
        key=info["key"],
        base=info["base"],
        model=info["model"],
        source=source_label(info["source"]),
        using_builtin=False,
        key_configured=bool(info["key"]),
    )


def read_snapshot() -> ConfigSnapshot:
    """Effective settings the UI should display (after userconfig resolution)."""
    # Import here so config_io stays usable without the generation stack's side effects
    # in unit tests that only touch write/parse. generate is still the source of truth
    # for builtin fallback labelling.
    from generate import BUILTIN_SOURCE, load_credentials

    typesafe = _typesafe_snapshot()
    openai = _gen_provider_snapshot("OPENAI")
    anthropic = _gen_provider_snapshot("ANTHROPIC")

    _base, _key, _model, gen_src, _api = load_credentials()
    using_builtin = gen_src == BUILTIN_SOURCE

    extra = {
        "LLM_MODEL": userconfig.get("LLM_MODEL"),
        "OPENAI_EXTRA_BODY": userconfig.get("OPENAI_EXTRA_BODY"),
        "JEV_BOXES": userconfig.get("JEV_BOXES"),
        "JEV_TONES": userconfig.get("JEV_TONES"),
    }
    return ConfigSnapshot(
        typesafe=typesafe,
        openai=openai,
        anthropic=anthropic,
        generation_source=source_label(gen_src),
        generation_using_builtin=using_builtin,
        write_path=preferred_write_path(),
        extra=extra,
    )


def _quote_env_value(val: str) -> str:
    """Quote when the value needs it; leave simple tokens bare."""
    if val == "":
        return '""'
    if re.search(r'[\s#"\'\\$`]', val):
        escaped = val.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return val


def _replace_assignment(line: str, new_value: str) -> str | None:
    """Rewrite KEY=… on a physical line, keeping export prefix and trailing comment."""
    m = _KEY_LINE.match(line.rstrip("\n"))
    if not m:
        return None
    prefix, _key, eq, rest = m.groups()
    # Split trailing comment outside quotes (same idea as parse_env_file)
    quote = None
    cut = len(rest)
    for i, ch in enumerate(rest):
        if quote:
            if ch == quote:
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "#" and i > 0 and rest[i - 1] in " \t":
            cut = i
            break
    trailing = rest[cut:]  # includes leading whitespace + comment, or empty
    return f"{prefix}{_key}{eq}{_quote_env_value(new_value)}{trailing}"


def update_env_file(path: Path, updates: dict[str, str | None],
                    *, mode: int = 0o600) -> Path:
    """Update keys in ``path``, preserving comments and unknown lines best-effort.

    ``updates`` maps KEY -> new value. A value of ``None`` removes that key's
    assignment line (comments above it are kept). Empty string writes KEY=\"\".

    Creates the parent directory if needed. Always sets file mode to ``mode``
    (default 600). Returns the path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    try:
        original = path.read_text(encoding="utf-8")
        lines = original.splitlines(keepends=True)
    except FileNotFoundError:
        lines = []
    except OSError:
        lines = []

    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        raw = line.rstrip("\n\r")
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            out.append(line if line.endswith("\n") else line + "\n")
            continue
        check = stripped[len("export "):].strip() if stripped.startswith("export ") else stripped
        if "=" not in check:
            out.append(line if line.endswith("\n") else line + "\n")
            continue
        key = check.split("=", 1)[0].strip()
        if key not in pending:
            out.append(line if line.endswith("\n") else line + "\n")
            continue
        new_val = pending.pop(key)
        if new_val is None:
            continue  # drop the assignment line
        rewritten = _replace_assignment(raw, new_val)
        if rewritten is None:
            rewritten = f"{key}={_quote_env_value(new_val)}"
        out.append(rewritten + "\n")

    # Append keys that were not present in the file
    if pending:
        if out and not out[-1].endswith("\n"):
            out[-1] = out[-1] + "\n"
        if out and out[-1].strip() and not out[-1].strip().startswith("#"):
            # small spacer before newly appended managed keys
            pass
        if any(v is not None for v in pending.values()):
            if out and out[-1].strip():
                out.append("\n")
            out.append("# --- updated by jev-jarvis config panel ---\n")
        for key in MANAGED_KEYS:
            if key not in pending:
                continue
            new_val = pending.pop(key)
            if new_val is None:
                continue
            out.append(f"{key}={_quote_env_value(new_val)}\n")
        # any leftover non-managed keys (shouldn't happen, but be complete)
        for key, new_val in pending.items():
            if new_val is None:
                continue
            out.append(f"{key}={_quote_env_value(new_val)}\n")

    text = "".join(out)
    # Atomic-ish write: write temp then replace, then chmod
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    os.chmod(path, mode)
    return path


def models_url(base: str) -> str:
    """Build the OpenAI-compatible models list URL from a base (tolerates /v1)."""
    b = (base or "").rstrip("/")
    last = b.rsplit("/", 1)[-1].lower()
    if re.fullmatch(r"v\d+[a-z]*", last):
        return b + "/models"
    return b + "/v1/models"


def probe_models(base: str, key: str, *,
                 api_format: str = "openai",
                 timeout: float = 8.0) -> tuple[list[str], str]:
    """GET the endpoint's model list. Returns (ids, error). Never raises.

    OpenAI-compatible: ``GET {base}/v1/models`` with Bearer key.
    Anthropic-compatible: same path shape with ``x-api-key`` (many gateways
    expose /v1/models for both); on failure returns empty + error so the UI
    falls back to a free-text field.
    """
    if not base:
        return [], "未填写端点 base_url"
    if not key:
        return [], "未填写 API Key"
    url = models_url(base)
    if api_format == "anthropic":
        headers = {
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "authorization": f"Bearer {key}",
        }
    else:
        headers = {"authorization": f"Bearer {key}"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        detail = e.read()[:160].decode(errors="replace")
        return [], f"HTTP {e.code}: {detail}"
    except Exception as e:
        return [], f"{type(e).__name__}: {e}"

    ids: list[str] = []
    raw = data.get("data") if isinstance(data, dict) else None
    if isinstance(raw, list):
        for m in raw:
            if isinstance(m, dict) and m.get("id"):
                ids.append(str(m["id"]))
            elif isinstance(m, str):
                ids.append(m)
    elif isinstance(data, dict) and isinstance(data.get("models"), list):
        for m in data["models"]:
            if isinstance(m, dict) and m.get("id"):
                ids.append(str(m["id"]))
            elif isinstance(m, str):
                ids.append(m)
    if not ids:
        return [], "端点未返回模型列表（可手填模型名）"
    return ids, ""


def test_typesafe_connection(base: str, key: str, model: str,
                             *, timeout: float = 15.0) -> tuple[bool, str]:
    """Hit TypeSafe systemone with a tiny payload — same shape as judge_jev."""
    if not key:
        return False, "未填写 TYPESAFE_API_KEY"
    b = (base or "https://api.typesafe.ai").rstrip("/")
    url = f"{b}/v1/systemone"
    body = {
        "model": model or "jev-latest",
        "state": "ping",
        "questions": {
            "ok": {"type": "choice",
                   "instructions": "连接测试",
                   "criteria": {"yes": None, "no": None}},
        },
    }
    try:
        from generate import http_post_json
        http_post_json(
            url,
            {"content-type": "application/json",
             "authorization": f"Bearer {key}"},
            body, timeout)
        return True, f"连接成功 · {url}"
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        return False, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_openai_connection(base: str, key: str, model: str,
                           *, timeout: float = 15.0) -> tuple[bool, str]:
    """Validate OpenAI-compatible credentials via models list, then a tiny chat."""
    ids, err = probe_models(base, key, api_format="openai", timeout=timeout)
    if err and not ids:
        # Still try chat — some relays hide /models
        pass
    else:
        if model and ids and model not in ids:
            return False, (f"模型 {model} 不在端点返回的列表中"
                           f"（共 {len(ids)} 个）；可改选或手填后重试")
        if ids:
            # models list succeeded — credentials are valid enough
            return True, f"连接成功 · 探测到 {len(ids)} 个模型"

    # Fallback: minimal chat/completions
    from generate import _endpoint, http_post_json
    url = _endpoint(base, "openai")
    body = {
        "model": model or (ids[0] if ids else "gpt-4o-mini"),
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {"content-type": "application/json",
               "authorization": f"Bearer {key}"}
    try:
        http_post_json(url, headers, body, timeout)
        return True, f"连接成功 · {url}"
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        return False, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def test_anthropic_connection(base: str, key: str, model: str,
                              *, timeout: float = 15.0) -> tuple[bool, str]:
    """Validate Anthropic-compatible credentials with a tiny /messages call."""
    if not key:
        return False, "未填写 ANTHROPIC_API_KEY"
    from generate import _endpoint, http_post_json
    url = _endpoint(base or "https://api.anthropic.com", "anthropic")
    body = {
        "model": model or "claude-3-5-haiku-latest",
        "max_tokens": 1,
        "messages": [{"role": "user", "content": "ping"}],
    }
    headers = {
        "content-type": "application/json",
        "x-api-key": key,
        "anthropic-version": "2023-06-01",
    }
    try:
        http_post_json(url, headers, body, timeout)
        return True, f"连接成功 · {url}"
    except urllib.error.HTTPError as e:
        detail = e.read()[:200].decode(errors="replace")
        return False, f"HTTP {e.code}: {detail}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


def builtin_key_fingerprint() -> str:
    """Masked view of the bundled shared key — for the 'using builtin' label."""
    return mask_key(builtin.API_KEY) if builtin.API_KEY else "（无内置 key）"


def apply_panel_updates(
    *,
    typesafe_key: str | None = None,
    typesafe_base: str | None = None,
    typesafe_model: str | None = None,
    openai_key: str | None = None,
    openai_base: str | None = None,
    openai_model: str | None = None,
    anthropic_key: str | None = None,
    anthropic_base: str | None = None,
    anthropic_model: str | None = None,
    extra: dict[str, str | None] | None = None,
    path: Path | None = None,
) -> Path:
    """Translate panel field values into an env-file write.

    ``None`` for a key field means "leave unchanged". Empty string clears it.
    Non-key fields (base/model) always write when provided as a string (including "").
    """
    updates: dict[str, str | None] = {}

    def put(name: str, value: str | None, *, is_key: bool = False) -> None:
        if value is None:
            return
        if is_key and value == "":
            # explicit clear
            updates[name] = ""
            return
        updates[name] = value

    put("TYPESAFE_API_KEY", typesafe_key, is_key=True)
    put("TYPESAFE_BASE_URL", typesafe_base)
    put("TYPESAFE_MODEL", typesafe_model)
    put("OPENAI_API_KEY", openai_key, is_key=True)
    put("OPENAI_BASE_URL", openai_base)
    put("OPENAI_MODEL", openai_model)
    put("ANTHROPIC_API_KEY", anthropic_key, is_key=True)
    put("ANTHROPIC_BASE_URL", anthropic_base)
    put("ANTHROPIC_MODEL", anthropic_model)
    if extra:
        for k, v in extra.items():
            if k in MANAGED_KEYS and v is not None:
                updates[k] = v

    return update_env_file(path or preferred_write_path(), updates)


def parse_env_lines_for_test(text: str) -> list[str]:
    """Expose line splitting for unit tests (no I/O)."""
    return text.splitlines(keepends=True)


__all__ = [
    "MANAGED_KEYS",
    "ENDPOINT_PRESETS",
    "MODEL_FALLBACKS",
    "ProviderSnapshot",
    "ConfigSnapshot",
    "preferred_write_path",
    "mask_key",
    "source_label",
    "read_snapshot",
    "update_env_file",
    "models_url",
    "probe_models",
    "test_typesafe_connection",
    "test_openai_connection",
    "test_anthropic_connection",
    "builtin_key_fingerprint",
    "apply_panel_updates",
]
