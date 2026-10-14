#!/usr/bin/env python3
"""Lightweight training/log health monitor for Bot_Trade_v2.

Runs as a oneshot from systemd timer. It reads recent logs, detects ML/training
pathologies, writes bot_training_health.json, and optionally sends throttled
Telegram alerts using the same .env credentials as the bot.
"""

from __future__ import annotations

import json
import os
import re
import socket
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path


BOT_DIR = Path(os.getenv("BOT_DIR", "/home/ubuntu/bot"))
LOG_PATH = Path(os.getenv("BOT_LOG_PATH", str(BOT_DIR / "bot_trade.log")))
ERR_PATH = Path(os.getenv("BOT_ERR_PATH", str(BOT_DIR / "bot_trade_errors.log")))
OUT_PATH = Path(os.getenv("BOT_TRAINING_HEALTH_FILE", str(BOT_DIR / "bot_training_health.json")))
STATE_PATH = Path(os.getenv("BOT_TRAINING_HEALTH_STATE", str(BOT_DIR / "bot_training_health_state.json")))
MODEL_PATH = Path(os.getenv("BOT_MODEL_FILE", str(BOT_DIR / "ml_model_v2.pkl")))
ENV_PATH = Path(os.getenv("BOT_ENV_FILE", str(BOT_DIR / ".env")))

LOOKBACK_MIN = int(os.getenv("BOT_TRAINING_MONITOR_LOOKBACK_MIN", "30"))
TAIL_MAX_BYTES = int(os.getenv("BOT_TRAINING_MONITOR_TAIL_BYTES", str(6 * 1024 * 1024)))
ML_NEUTRAL_WARN_COUNT = int(os.getenv("BOT_TRAINING_MONITOR_ML_NEUTRAL_WARN_COUNT", "20"))
RETRAIN_STUCK_MIN = int(os.getenv("BOT_TRAINING_MONITOR_RETRAIN_STUCK_MIN", "90"))
ALERT_THROTTLE_MIN = int(os.getenv("BOT_TRAINING_MONITOR_ALERT_THROTTLE_MIN", "60"))
DEEPSEEK_HTTP_WARN_COUNT = int(os.getenv("BOT_TRAINING_MONITOR_DEEPSEEK_WARN_COUNT", "120"))
TELEGRAM_ENABLED = os.getenv("BOT_TRAINING_MONITOR_TELEGRAM", "1").strip().lower() not in {"0", "false", "no", "off"}

TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})")
AUC_RE = re.compile(r"\[ML-GATE\]\s+(LONG|SHORT)\s+AUC_ref=([0-9.]+)\s+<\s+([0-9.]+)")
SAMPLES_RE = re.compile(r"([A-Z0-9]+/USDT):\s+(\d+)\s+amostras")
MODEL_FINAL_RE = re.compile(r"Modelo final\s+-\s+AUC=([0-9.]+)")
META_AUC_RE = re.compile(r"Meta-modelo\s+(LONG|SHORT)\s+-\s+AUC=([0-9.]+)")
BOOK_RE = re.compile(r"book L/S=([0-9.]+)/([0-9.]+)")
DIR_MINSCORE_RE = re.compile(r"(?:minScore|diagMin|score)_([LS])=([0-9.]+)")


def load_env(path: Path) -> dict[str, str]:
    env = {}
    try:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
    except Exception:
        pass
    return env


def read_tail(path: Path, max_bytes: int = TAIL_MAX_BYTES) -> list[str]:
    try:
        size = path.stat().st_size
        with path.open("rb") as f:
            if size > max_bytes:
                f.seek(-max_bytes, os.SEEK_END)
                f.readline()
            data = f.read()
        return data.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return []


def parse_ts(line: str):
    m = TS_RE.match(line or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    except Exception:
        return None


def recent_lines(lines: list[str], now: datetime, lookback_min: int) -> list[str]:
    cutoff = now - timedelta(minutes=lookback_min)
    out = []
    include_continuation = False
    for line in lines:
        ts = parse_ts(line)
        if ts is None:
            if include_continuation:
                out.append(line)
            continue
        include_continuation = ts >= cutoff
        if include_continuation:
            out.append(line)
    return out


def latest_ts(lines: list[str]):
    for line in reversed(lines):
        ts = parse_ts(line)
        if ts is not None:
            return ts
    return None


def mem_available_mb():
    try:
        vals = {}
        for raw in Path("/proc/meminfo").read_text().splitlines():
            k, rest = raw.split(":", 1)
            vals[k] = int(rest.strip().split()[0])
        return round(vals.get("MemAvailable", 0) / 1024.0, 1)
    except Exception:
        return None


def file_meta(path: Path):
    try:
        st = path.stat()
        age_min = (time.time() - st.st_mtime) / 60.0
        return {
            "exists": True,
            "size_bytes": st.st_size,
            "mtime_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(),
            "age_minutes": round(age_min, 1),
        }
    except Exception:
        return {"exists": False, "size_bytes": 0, "mtime_utc": None, "age_minutes": None}


def count_contains(lines: list[str], text: str) -> int:
    return sum(1 for line in lines if text in line)


def analyze() -> dict:
    now = datetime.now(timezone.utc)
    all_lines = read_tail(LOG_PATH)
    recent = recent_lines(all_lines, now, LOOKBACK_MIN)
    err_recent = recent_lines(read_tail(ERR_PATH, max_bytes=1024 * 1024), now, LOOKBACK_MIN)

    auc_values = {"LONG": [], "SHORT": []}
    auc_min_values = {"LONG": [], "SHORT": []}
    for line in recent:
        m = AUC_RE.search(line)
        if m:
            auc_values[m.group(1)].append(float(m.group(2)))
            auc_min_values[m.group(1)].append(float(m.group(3)))

    latest_auc = {
        side: (vals[-1] if vals else None)
        for side, vals in auc_values.items()
    }
    latest_auc_min = {
        side: (vals[-1] if vals else None)
        for side, vals in auc_min_values.items()
    }
    ml_gate_low = 0
    for side, vals in auc_values.items():
        mins = auc_min_values.get(side, [])
        for i, v in enumerate(vals):
            thr = mins[i] if i < len(mins) else 0.53
            if 0.0 < v < thr:
                ml_gate_low += 1
    ml_vote_skip = count_contains(recent, "ML-VOTE-SKIP")

    min_score_values = []
    latest_book = {"LONG": None, "SHORT": None}
    for line in recent:
        m = BOOK_RE.search(line)
        if m:
            latest_book = {"LONG": float(m.group(1)), "SHORT": float(m.group(2))}
            min_score_values.extend([latest_book["LONG"], latest_book["SHORT"]])
        m2 = DIR_MINSCORE_RE.search(line)
        if m2:
            side = "LONG" if m2.group(1) == "L" else "SHORT"
            val = float(m2.group(2))
            latest_book[side] = val
            min_score_values.append(val)

    retrain_start_lines = [
        line for line in recent
        if "A treinar modelo ML" in line or "retrain em background" in line or "[Retrain]" in line
    ]
    retrain_done_lines = [
        line for line in recent
        if "Modelo ML actualizado" in line or "Modelo final - AUC" in line or "Modelo guardado" in line
        or "retrain conclu" in line.lower()
    ]
    last_retrain_start = latest_ts(retrain_start_lines)
    last_retrain_done = latest_ts(retrain_done_lines)
    retrain_running = bool(last_retrain_start and (not last_retrain_done or last_retrain_done < last_retrain_start))
    retrain_age_min = round((now - last_retrain_start).total_seconds() / 60.0, 1) if retrain_running else 0.0

    samples = {}
    for line in recent:
        m = SAMPLES_RE.search(line)
        if m:
            samples[m.group(1)] = int(m.group(2))

    final_auc = []
    meta_auc = {}
    for line in recent:
        m = MODEL_FINAL_RE.search(line)
        if m:
            final_auc.append(float(m.group(1)))
        m2 = META_AUC_RE.search(line)
        if m2:
            meta_auc[m2.group(1)] = float(m2.group(2))

    counts = {
        "ml_gate_low": ml_gate_low,
        "ml_vote_skip": ml_vote_skip,
        "rl_min_score_high": sum(1 for v in min_score_values if v >= 13.8),
        "catastrophic_technical_named": count_contains(recent, "CATASTRÓFICA TÉCNICA"),
        "catastrophic_macro_named": count_contains(recent, "CATASTRÓFICA MACRO"),
        "cat_fields": sum(1 for line in recent if " cat=" in line),
        "deepseek_http": count_contains(recent, "HTTP Request: POST https://api.deepseek.com/chat/completions"),
        "warnings": count_contains(recent, "] [WARNING]"),
        "errors": count_contains(recent, "] [ERROR]") + len(err_recent),
        "l1_ok": count_contains(recent, "[L1 OK]"),
        "qf_ok": count_contains(recent, "[QF-OK]"),
        "qf_reject": count_contains(recent, "[QF-REJECT]"),
        "feature_audit": count_contains(recent, "FEATURE-AUDIT"),
    }

    model = file_meta(MODEL_PATH)
    mem_mb = mem_available_mb()
    findings = []
    status = "ok"

    if counts["errors"] > 0:
        status = "alert"
        findings.append(f"Erros recentes detectados: {counts['errors']}.")

    if counts["catastrophic_technical_named"] > 20 or counts["catastrophic_macro_named"] > 5:
        status = "alert"
        findings.append("Logs CATASTROFICA voltaram em excesso; investigar treino/DeepSeek.")

    if counts["deepseek_http"] > DEEPSEEK_HTTP_WARN_COUNT:
        status = "alert"
        findings.append(
            f"Chamadas DeepSeek excessivas: {counts['deepseek_http']} em {LOOKBACK_MIN} min "
            f"(provavel treino pesado no processo live)."
        )

    if ml_gate_low >= ML_NEUTRAL_WARN_COUNT:
        if status != "alert":
            status = "watch"
        findings.append(
            f"ML neutro recorrente: {ml_gate_low} gates abaixo do limite adaptativo "
            f"em {LOOKBACK_MIN} min (LONG={latest_auc['LONG']}/{latest_auc_min['LONG']}, "
            f"SHORT={latest_auc['SHORT']}/{latest_auc_min['SHORT']})."
        )

    if counts["rl_min_score_high"] >= 10:
        if status == "ok":
            status = "watch"
        findings.append(
            f"RL min_score alto recorrente: {counts['rl_min_score_high']} leituras >= 13.8 "
            f"(ultimo L/S={latest_book['LONG']}/{latest_book['SHORT']})."
        )

    if retrain_running:
        if retrain_age_min >= RETRAIN_STUCK_MIN:
            status = "alert"
            findings.append(f"Retrain possivelmente preso ha {retrain_age_min} min.")
        else:
            if status == "ok":
                status = "watch"
            findings.append(f"Retrain em execucao ha {retrain_age_min} min; aguardar novo modelo.")

    if model["exists"] and model["age_minutes"] is not None and model["age_minutes"] > 12 * 60:
        if status == "ok":
            status = "watch"
        findings.append(f"Modelo ML tem {model['age_minutes']} min; confirmar se retrain gravou modelo.")

    if mem_mb is not None and mem_mb < 250:
        if status != "alert":
            status = "watch"
        findings.append(f"Memoria disponivel baixa: {mem_mb} MB.")

    if not findings:
        findings.append("Sem anomalias criticas no periodo analisado.")

    relevant_tail = [
        line for line in recent
        if any(key in line for key in (
            "ML-GATE", "ML-VOTE-SKIP", "Retrain", "A treinar modelo", "Modelo final",
            "FEATURE-AUDIT", "CATASTRÓFICA", "ERROR", "WARNING", "[L1 OK]", "[QF-OK]"
        ))
    ][-80:]

    return {
        "status": status,
        "generated_at_utc": now.isoformat(),
        "host": socket.gethostname(),
        "lookback_minutes": LOOKBACK_MIN,
        "log_path": str(LOG_PATH),
        "last_log_ts_utc": latest_ts(all_lines).isoformat() if latest_ts(all_lines) else None,
        "latest_auc": latest_auc,
        "latest_auc_min": latest_auc_min,
        "latest_min_score_book": latest_book,
        "ml_neutral": bool(ml_gate_low >= ML_NEUTRAL_WARN_COUNT),
        "retrain": {
            "running": retrain_running,
            "age_minutes": retrain_age_min,
            "samples_by_symbol": samples,
            "final_auc": final_auc[-3:],
            "meta_auc": meta_auc,
        },
        "counts": counts,
        "model_file": model,
        "memory_available_mb": mem_mb,
        "findings": findings,
        "tail": relevant_tail,
    }


def alert_key(report: dict) -> str:
    return "|".join([report["status"]] + report.get("findings", [])[:3])


def send_telegram(report: dict):
    if not TELEGRAM_ENABLED or report["status"] == "ok":
        return
    env = load_env(ENV_PATH)
    token = env.get("TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = env.get("TELEGRAM_CHAT_ID") or os.getenv("TELEGRAM_CHAT_ID", "")
    if not token or not chat_id:
        return

    state = {}
    try:
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        pass

    key = alert_key(report)
    now_ts = time.time()
    last_key = state.get("last_alert_key")
    last_ts = float(state.get("last_alert_ts", 0) or 0)
    if key == last_key and now_ts - last_ts < ALERT_THROTTLE_MIN * 60:
        return

    title = "ALERTA" if report["status"] == "alert" else "AVISO"
    findings = "\n".join(f"- {x}" for x in report.get("findings", [])[:5])
    msg = (
        f"{title} Bot Training Monitor\n"
        f"Status: {report['status']}\n"
        f"Lookback: {report['lookback_minutes']} min\n"
        f"ML AUC: LONG={report['latest_auc'].get('LONG')} SHORT={report['latest_auc'].get('SHORT')}\n"
        f"ML neutro gates: {report['counts'].get('ml_gate_low')}\n"
        f"Retrain running: {report['retrain'].get('running')} ({report['retrain'].get('age_minutes')} min)\n"
        f"Memoria livre: {report.get('memory_available_mb')} MB\n"
        f"{findings}"
    )
    data = urllib.parse.urlencode({"chat_id": chat_id, "text": msg}).encode()
    req = urllib.request.Request(f"https://api.telegram.org/bot{token}/sendMessage", data=data)
    try:
        urllib.request.urlopen(req, timeout=8).read()
        state.update({"last_alert_key": key, "last_alert_ts": now_ts})
        STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    except Exception:
        pass


def main():
    report = analyze()
    OUT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    send_telegram(report)
    print(f"{report['status']} | " + " | ".join(report.get("findings", [])[:3]))


if __name__ == "__main__":
    main()
