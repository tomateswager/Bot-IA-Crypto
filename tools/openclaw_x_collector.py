#!/usr/bin/env python3
"""
Read-only OpenClaw bridge for X/Twitter crypto signals.

This process never imports the trading bot and never reads Binance/API secrets.
It only calls an OpenClaw command, normalizes the results, and writes
external_x_feed.json in the schema consumed by Bot_Trade_v2.py.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


DEFAULT_CONFIG = {
    "allow_non_x_results": False,
    "accept_ai_summaries": False,
    "extract_symbols_from_text": False,
    "max_posts": 30,
    "max_queries_per_run": 2,
    "max_query_seconds": 30,
    "queries": [
        {"name": "btc", "symbols": ["BTC"], "query": 'site:x.com/x.com/status (BTC OR bitcoin) latest crypto posts'},
        {"name": "eth", "symbols": ["ETH"], "query": 'site:x.com/x.com/status (ETH OR ethereum) latest crypto posts'},
        {"name": "sol", "symbols": ["SOL"], "query": 'site:x.com/x.com/status (SOL OR solana) latest crypto posts'},
        {"name": "zec", "symbols": ["ZEC"], "query": 'site:x.com/x.com/status (ZEC OR zcash) latest crypto posts'},
        {"name": "xrp", "symbols": ["XRP"], "query": 'site:x.com/x.com/status (XRP OR ripple) latest crypto posts'},
        {"name": "market", "symbols": ["BTC", "ETH"], "query": 'site:x.com/x.com/status (crypto hack OR exploit OR ETF OR SEC OR Binance listing) latest posts'},
    ],
}


X_URL_RE = re.compile(r"https?://(?:www\.)?(?:x|twitter)\.com/[A-Za-z0-9_./?=&%-]+", re.I)
X_STATUS_URL_RE = re.compile(
    r"https?://(?:www\.)?(?:x|twitter)\.com/([A-Za-z0-9_]{1,32})/status/(\d+)",
    re.I,
)
AI_SUMMARY_RE = re.compile(
    r"\b("
    r"latest|recent|roundup|summary|summarizes|buzz|as of|here'?s|"
    r"highlights?|discussions?|news from x|twitter buzz|based on"
    r")\b",
    re.I,
)
SYMBOL_RE = re.compile(
    r"\$([A-Z0-9]{2,12})(?![A-Z0-9])|"
    r"(?<![A-Z0-9])([A-Z0-9]{2,12})/(?:USDT|USD)(?![A-Z0-9])|"
    r"(?<![A-Z0-9])([A-Z0-9]{2,12})(?:USDT|USD|PERP)(?![A-Z0-9])"
)
STABLES = {"USDT", "USDC", "USD", "BUSD", "DAI", "FDUSD", "TUSD", "EUR", "EURC"}


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def iso_now() -> str:
    return utc_now().isoformat().replace("+00:00", "Z")


def load_json(path: Path, default):
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[openclaw-x] config read failed: {exc}", file=sys.stderr)
    return default


def atomic_write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def command_template() -> list[str]:
    import shlex

    raw = os.getenv("OPENCLAW_WEB_SEARCH_CMD", "").strip()
    if raw:
        # Lightweight shell-like split without invoking a shell. Quoted queries
        # should be supplied through the {query} placeholder.
        return shlex.split(raw)

    argv = [
        "openclaw",
        "infer",
        "web",
        "search",
        "--json",
        "--limit",
        os.getenv("OPENCLAW_WEB_SEARCH_LIMIT", "10"),
    ]
    provider = os.getenv("OPENCLAW_WEB_SEARCH_PROVIDER", "").strip()
    if provider:
        argv.extend(["--provider", provider])
    argv.extend(["--query", "{query}"])
    return argv


def openclaw_available(template: list[str]) -> bool:
    if not template:
        return False
    exe = template[0]
    return shutil.which(exe) is not None


def run_openclaw_query(template: list[str], query: str, timeout: int) -> tuple[str, str | None]:
    argv = [part.replace("{query}", query) for part in template]
    try:
        proc = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except Exception as exc:
        return "", str(exc)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip()
        return proc.stdout or "", f"openclaw rc={proc.returncode}: {err[:240]}"
    err = payload_error(proc.stdout or "")
    if err:
        return proc.stdout or "", err
    return proc.stdout or "", None


def rotated_queries(config: dict) -> tuple[list[dict], int | None]:
    queries = [q for q in config.get("queries", []) if isinstance(q, dict)]
    if not queries:
        return [], None

    try:
        per_run = int(config.get("max_queries_per_run", len(queries)))
    except Exception:
        per_run = len(queries)
    per_run = max(1, min(per_run, len(queries)))
    if per_run >= len(queries):
        return queries, None

    cursor_path = config.get("_cursor_path")
    cursor = 0
    if cursor_path:
        state = load_json(Path(cursor_path), {"index": 0})
        try:
            cursor = int(state.get("index", 0)) % len(queries)
        except Exception:
            cursor = 0

    selected = [queries[(cursor + offset) % len(queries)] for offset in range(per_run)]
    return selected, (cursor + per_run) % len(queries)


def persist_cursor(config: dict, index: int | None) -> None:
    if index is None or not config.get("_cursor_path"):
        return
    atomic_write_json(Path(config["_cursor_path"]), {"index": int(index), "updated_at": iso_now()})


def items_from_payload(payload):
    if isinstance(payload, list):
        rows = []
        for item in payload:
            rows.extend(items_from_payload(item))
        return rows
    if isinstance(payload, dict):
        if isinstance(payload.get("result"), dict):
            return items_from_payload(payload["result"])
        if isinstance(payload.get("outputs"), list):
            rows = []
            for output in payload["outputs"]:
                rows.extend(items_from_payload(output))
            return rows
        for key in ("results", "items", "data", "posts", "tweets", "organic"):
            value = payload.get(key)
            if isinstance(value, list):
                return items_from_payload(value)
        return [payload]
    return []


def payload_error(text: str) -> str | None:
    try:
        payload = json.loads((text or "").strip())
    except Exception:
        return None

    def walk(value):
        if isinstance(value, dict):
            error = value.get("error")
            message = value.get("message")
            if error:
                return f"openclaw {error}: {str(message or '')[:200]}".strip()
            for nested in value.values():
                found = walk(nested)
                if found:
                    return found
        if isinstance(value, list):
            for nested in value:
                found = walk(nested)
                if found:
                    return found
        return None

    return walk(payload)


def parse_openclaw_output(text: str) -> list[dict]:
    text = (text or "").strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
        return [item for item in items_from_payload(payload) if isinstance(item, dict)]
    except Exception:
        pass

    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        url_match = X_URL_RE.search(line)
        rows.append({
            "title": line,
            "snippet": "",
            "url": url_match.group(0) if url_match else "",
        })
    return rows


def normalize_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_openclaw_text(value) -> str:
    text = str(value or "")
    text = re.sub(r"<<<[^>]+>>>", " ", text)
    text = re.sub(r"\bSource:\s*Web Search\s*-+", " ", text, flags=re.I)
    text = re.sub(r"\[\[\d+\]\]\([^)]*\)", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" -")


def sanitize_url(value: str) -> str:
    return str(value or "").strip().rstrip(").,;]>\"'")


def result_url(item: dict) -> str:
    for key in ("url", "link", "href", "tweet_url"):
        value = sanitize_url(item.get(key) or "")
        if value:
            return value
    text = " ".join(str(item.get(k, "")) for k in ("title", "snippet", "text", "content"))
    match = X_URL_RE.search(text)
    return sanitize_url(match.group(0)) if match else ""


def is_x_result(item: dict) -> bool:
    url = result_url(item)
    if X_URL_RE.search(url):
        return True
    source = str(item.get("source") or item.get("provider") or "").lower()
    return source in {"x", "twitter", "openclaw-x"}


def x_status_parts(url: str) -> tuple[str, str] | None:
    match = X_STATUS_URL_RE.search(url or "")
    if not match:
        return None
    return match.group(1), match.group(2)


def looks_like_ai_summary(text: str, item: dict) -> bool:
    clean = normalize_text(text)
    lower = clean.lower()
    if "external_untrusted_content" in lower:
        return True
    if AI_SUMMARY_RE.search(clean[:260]):
        return True
    if clean.count(" - ") >= 2 or clean.count("**") >= 2:
        return True
    if len(clean) > 520 and not (item.get("author") or item.get("username") or item.get("account")):
        return True
    return False


def extract_symbols(text: str, seed_symbols: list[str], extract_from_text: bool = False) -> list[str]:
    symbols = []
    for sym in seed_symbols or []:
        base = str(sym or "").upper().strip().lstrip("$#@")
        if "/" in base:
            base = base.split("/", 1)[0]
        if 2 <= len(base) <= 12 and base not in STABLES and base not in symbols:
            symbols.append(base)
    if not extract_from_text:
        return symbols[:8]
    for match in SYMBOL_RE.finditer(text.upper()):
        base = next((g for g in match.groups() if g), "")
        if base not in STABLES and base not in symbols:
            symbols.append(base)
    return symbols[:8]


def metric(item: dict, *keys, default=0):
    for key in keys:
        try:
            value = item.get(key)
            if value not in (None, ""):
                return int(float(value))
        except Exception:
            continue
    return default


def to_post(item: dict, query_cfg: dict, accept_ai_summaries: bool = False) -> tuple[dict | None, str | None]:
    title = clean_openclaw_text(item.get("title") or item.get("text") or item.get("content"))
    snippet = clean_openclaw_text(item.get("snippet") or item.get("summary") or item.get("description"))
    text = normalize_text(f"{title} {snippet}")
    if len(text) < 8:
        return None, "too_short"
    url = result_url(item)
    status_parts = x_status_parts(url)
    is_summary = looks_like_ai_summary(text, item)
    if not status_parts and not accept_ai_summaries:
        return None, "not_x_status_url"
    if is_summary and not accept_ai_summaries:
        return None, "ai_summary_rejected"
    symbols = extract_symbols(
        text,
        query_cfg.get("symbols", []),
        bool(query_cfg.get("extract_symbols_from_text", query_cfg.get("_extract_symbols_from_text", False))),
    )
    author = normalize_text(item.get("author") or item.get("username") or item.get("account"))
    if not author and status_parts:
        author = f"@{status_parts[0]}"
    quality = "summary" if is_summary else "tweet"
    return {
        "source": "openclaw-x",
        "author": author,
        "published_at": normalize_text(item.get("published_at") or item.get("created_at") or item.get("timestamp")) or iso_now(),
        "symbols": symbols,
        "text": text[:900] if accept_ai_summaries else text[:520],
        "url": url,
        "quality": quality,
        "tweet_id": status_parts[1] if status_parts else "",
        "likes": metric(item, "likes", "like_count", "favorite_count"),
        "retweets": metric(item, "retweets", "reposts", "retweet_count", "repost_count"),
        "views": metric(item, "views", "view_count", "impressions"),
    }, None


def collect(config: dict) -> tuple[list[dict], list[str], list[str], bool]:
    template = command_template()
    if not openclaw_available(template):
        return [], [f"openclaw_not_found: {template[0] if template else 'empty command'}"], [], False

    allow_non_x = bool(config.get("allow_non_x_results", False))
    accept_ai_summaries = bool(config.get("accept_ai_summaries", False))
    max_posts = int(config.get("max_posts", 80))
    timeout = int(config.get("max_query_seconds", 45))
    posts = []
    errors = []
    warnings = []
    skip_counts: dict[str, int] = {}
    seen = set()
    queries, next_cursor = rotated_queries(config)
    for query_cfg in queries:
        query_cfg["_extract_symbols_from_text"] = bool(config.get("extract_symbols_from_text", False))
        query = str(query_cfg.get("query") or "").strip()
        if not query:
            continue
        stdout, err = run_openclaw_query(template, query, timeout)
        if err:
            errors.append(f"{query_cfg.get('name', query)[:40]}: {err}")
            if "missing_" in err or "credentials" in err.lower():
                break
        for item in parse_openclaw_output(stdout):
            if not allow_non_x and not is_x_result(item):
                skip_counts["non_x_result"] = skip_counts.get("non_x_result", 0) + 1
                continue
            post, skip_reason = to_post(item, query_cfg, accept_ai_summaries)
            if not post:
                if skip_reason:
                    skip_counts[skip_reason] = skip_counts.get(skip_reason, 0) + 1
                continue
            key = post.get("url") or post.get("text", "")[:160]
            if key in seen:
                continue
            seen.add(key)
            posts.append(post)
            if len(posts) >= max_posts:
                break
        if len(posts) >= max_posts:
            break
    persist_cursor(config, next_cursor)
    if skip_counts:
        warnings.extend(f"{key}={value}" for key, value in sorted(skip_counts.items()))
    return posts, errors, warnings, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.getenv("OPENCLAW_X_CONFIG", "openclaw_x_sources.json"))
    parser.add_argument("--output", default=os.getenv("EXTERNAL_X_FEED_FILE", "external_x_feed.json"))
    args = parser.parse_args()

    config = load_json(Path(args.config), DEFAULT_CONFIG)
    config["_cursor_path"] = os.getenv("OPENCLAW_X_CURSOR_FILE") or str(Path(args.output).with_name("openclaw_x_cursor.json"))
    posts, errors, warnings, provider_found = collect(config)
    payload = {
        "updated_at": iso_now(),
        "collector": "openclaw_x_collector",
        "status": {
            "provider": "openclaw",
            "provider_found": provider_found,
            "ok": provider_found and bool(posts) and not errors,
            "partial": bool(posts and (errors or warnings)),
            "errors": errors[:8],
            "warnings": warnings[:8],
        },
        "posts": posts,
    }
    atomic_write_json(Path(args.output), payload)
    print(f"[openclaw-x] wrote {len(posts)} posts to {args.output}; provider_found={provider_found}")
    for err in errors[:3]:
        print(f"[openclaw-x] warning: {err}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
