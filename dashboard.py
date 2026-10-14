"""
dashboard.py — Bot Trade v2 Web Dashboard (v2)
Tabs com lazy-load: cada aba só carrega quando está activa.
Logs em tempo real via SSE.

Uso no AWS:
    cd ~
    python3 dashboard.py

Acesso remoto via SSH tunnel (sem abrir a porta do dashboard):
    ssh -i "<caminho-da-chave.pem>" -L 5001:localhost:5001 <utilizador>@<servidor>
    Browser: http://localhost:5001
"""

import csv, json, os, time, threading, re
from datetime import datetime, timezone, timedelta
from collections import defaultdict
from zoneinfo import ZoneInfo

try:
    from flask import Flask, jsonify, Response, render_template_string
except ImportError:
    os.system("pip install flask -q")
    from flask import Flask, jsonify, Response, render_template_string

HOME      = os.path.expanduser("~")
BOT_DIR   = os.path.join(HOME, "bot")

def find_file(*names):
    """Procura ficheiro em ~/bot/ primeiro, depois em ~/.

    Nota: para ficheiros de estado do bot devemos preferir SEMPRE ~/bot/,
    mesmo que o conteúdo seja "{}" (2 bytes). Caso contrário, um ficheiro
    antigo em ~/ pode contaminar o dashboard com posições fantasma.
    """
    for name in names:
        # 1) Preferência absoluta por ~/bot (se existir, usar já)
        p_bot = os.path.join(BOT_DIR, name)
        if os.path.exists(p_bot):
            return p_bot
        # 2) Só se não existir no BOT_DIR, tentar ~/ como fallback legado
        for base in [HOME]:
            p = os.path.join(base, name)
            if os.path.exists(p) and os.path.getsize(p) > 2:
                return p
    # fallback: ~/bot/
    return os.path.join(BOT_DIR, names[0])

BANK_FILE    = find_file("virtual_bank.json")
HISTORY_FILE = find_file("trade_history.json")
MONITORED    = find_file("monitored_coins.json")
SHORT_FILE   = find_file("short_trades.json")
ALPHA_FILE   = find_file("alpha_efficiency.json")
RL_FILE      = find_file("learning_state.json")
RUNTIME_FILE = find_file("runtime_state.json")
TRAINING_HEALTH_FILE = find_file("bot_training_health.json")
LOG_FILE     = os.path.join(BOT_DIR, "bot_trade.log")
ERR_FILE     = os.path.join(BOT_DIR, "bot_trade_errors.log")

V13_DIR         = os.path.join(BOT_DIR, "v13_state")
V13_BANK_FILE   = os.path.join(V13_DIR, "paper_bank_v13.json")
V13_TRADES_FILE = os.path.join(V13_DIR, "paper_trades_v13.csv")
V13_RUNNER_FILE = os.path.join(V13_DIR, "runner_state_v13.json")
V13_MODEL_FILE  = os.path.join(V13_DIR, "model_v13.pkl")
V13_LOG_FILE    = os.path.join(V13_DIR, "runner_v13.log")
V13_ERR_FILE    = os.path.join(V13_DIR, "runner_v13.err")

app = Flask(__name__)
DISPLAY_TZ = ZoneInfo("Europe/Lisbon")
LOG_TS_RE = re.compile(r"^\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")

# ── Helpers ────────────────────────────────────────────────────────────
def rj(path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}


def normalize_bot(bot):
    return "new" if str(bot).lower() in ("new", "v13", "novo") else "old"


def bot_label(bot):
    return "Bot Novo V13" if normalize_bot(bot) == "new" else "Bot Antigo v2"


def safe_float(value, default=0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def trade_partial_pnl(t):
    """PnL ja realizado em saidas parciais, sobretudo TP1."""
    return (
        safe_float(t.get("tp1_pnl"), 0.0) +
        safe_float(t.get("tp2_pnl"), 0.0) +
        safe_float(t.get("partial_pnl"), 0.0)
    )


def trade_closed_pnl(t):
    """PnL da saida final/restante da posicao."""
    return safe_float(t.get("actual_pnl"), 0.0)


def trade_realized_pnl(t):
    """PnL total realizado da trade: parcial + fecho final."""
    return trade_closed_pnl(t) + trade_partial_pnl(t)


def trade_sort_ts(t):
    return t.get("exit_time") or t.get("close_ts") or t.get("timestamp", "")


def read_v13_trades():
    if not os.path.exists(V13_TRADES_FILE):
        return []
    try:
        with open(V13_TRADES_FILE, "r", encoding="utf-8", errors="replace", newline="") as f:
            return list(csv.DictReader(f))
    except Exception:
        return []


def log_paths_for_bot(bot):
    if normalize_bot(bot) == "new":
        return V13_LOG_FILE, V13_ERR_FILE
    return LOG_FILE, ERR_FILE


def now_local():
    return datetime.now(timezone.utc).astimezone(DISPLAY_TZ)


def now_local_ts():
    return now_local().strftime("%H:%M:%S %Z")


def today_str():
    return now_local().strftime("%Y-%m-%d")


def convert_log_line_to_local(line):
    if not line:
        return line
    m = LOG_TS_RE.match(line)
    if not m:
        return line
    try:
        dt_utc = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        dt_local = dt_utc.astimezone(DISPLAY_TZ)
        return f"[{dt_local.strftime('%Y-%m-%d %H:%M:%S %Z')}]" + line[m.end():]
    except Exception:
        return line


def parse_log_line_local_dt(line):
    m = LOG_TS_RE.match(line or "")
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=timezone.utc
        ).astimezone(DISPLAY_TZ)
    except Exception:
        return None


def fmt_iso_local(ts):
    if not ts:
        return ""
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
    except Exception:
        return str(ts)[:16].replace("T", " ")


def read_log_lines(path, n=200):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return [convert_log_line_to_local(l.rstrip()) for l in f.readlines()[-n:]]
    except Exception:
        return []


def read_log_today(path):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            today = today_str()
            out = []
            for raw in f:
                line = raw.rstrip()
                dt_local = parse_log_line_local_dt(line)
                if dt_local is not None:
                    if dt_local.strftime("%Y-%m-%d") == today:
                        out.append(convert_log_line_to_local(line))
                elif today in line:
                    out.append(line)
            return out
    except Exception:
        return []


def get_live_prices(symbols):
    prices = {}
    try:
        import urllib.request
        with urllib.request.urlopen(
            "https://api.binance.com/api/v3/ticker/price", timeout=4
        ) as r:
            data = json.loads(r.read())
        for item in data:
            sym = item["symbol"]
            for s in symbols:
                base = s.replace("/USDT", "").replace("/BTC", "")
                if sym == base + "USDT":
                    prices[s] = float(item["price"])
    except Exception:
        pass
    return prices


def build_v13_portfolio_payload():
    bank = rj(V13_BANK_FILE, {})
    closed = sorted(read_v13_trades(), key=lambda x: x.get("exit_ts") or x.get("entry_ts", ""))

    initial = 10000.0
    balance = safe_float(bank.get("balance"), initial)
    total_pnl = safe_float(bank.get("total_pnl"), sum(safe_float(t.get("pnl")) for t in closed))
    total_trades = int(bank.get("total_trades") or len(closed))
    winning_trades = sum(1 for t in closed if safe_float(t.get("pnl")) > 0)
    win_rate = (winning_trades / total_trades * 100) if total_trades else 0.0
    open_positions = bank.get("open_positions", {}) if isinstance(bank.get("open_positions"), dict) else {}
    peak = safe_float(bank.get("peak_balance"), max(initial, balance))
    drawdown = ((peak - balance) / peak * 100) if peak > 0 else 0.0

    pnl_curve, daily = [], defaultdict(float)
    cum = 0.0
    for t in closed:
        pnl = safe_float(t.get("pnl"))
        cum += pnl
        ts = t.get("exit_ts") or t.get("entry_ts") or ""
        day = str(ts)[:10]
        pnl_curve.append({
            "ts": day, "cum": round(cum, 2), "pnl": round(pnl, 2),
            "sym": t.get("symbol", "?"), "dir": t.get("side", "?")
        })
        if day:
            daily[day] += pnl
    daily_data = [{"date": d, "pnl": round(v, 2)} for d, v in sorted(daily.items())]

    best_day = max(daily_data, key=lambda x: x["pnl"], default=None)
    worst_day = min(daily_data, key=lambda x: x["pnl"], default=None)

    streak = 0
    streak_type = None
    for t in reversed(closed):
        is_win = safe_float(t.get("pnl")) > 0
        if streak == 0:
            streak_type = is_win
        if is_win == streak_type:
            streak += 1
        else:
            break

    total_fees = round(sum(safe_float(p.get("fee_paid")) for p in open_positions.values()), 4)

    return {
        "bot": "new",
        "bot_label": bot_label("new"),
        "summary": {
            "balance": round(balance, 2), "initial": initial,
            "total_pnl": round(total_pnl, 2), "pnl_pct": round(total_pnl / initial * 100, 2),
            "win_rate": round(win_rate, 1), "sharpe": 0.0,
            "total_trades": total_trades, "winning_trades": winning_trades,
            "open_positions": len(open_positions), "peak": round(peak, 2),
            "drawdown_pct": round(drawdown, 2), "total_fees": total_fees,
            "streak": streak, "streak_type": "win" if streak_type else "loss",
            "best_day": best_day, "worst_day": worst_day,
        },
        "pnl_curve": pnl_curve,
        "daily_pnl": daily_data,
        "ts": now_local_ts(),
    }


def build_v13_positions_payload():
    bank = rj(V13_BANK_FILE, {})
    open_positions = bank.get("open_positions", {}) if isinstance(bank.get("open_positions"), dict) else {}
    live_prices = get_live_prices(open_positions.keys()) if open_positions else {}

    positions = []
    for sym, pos in open_positions.items():
        direc = pos.get("side", "long")
        entry = safe_float(pos.get("entry_price"))
        sl = safe_float(pos.get("sl_price"))
        tp = safe_float(pos.get("tp_price"))
        size = safe_float(pos.get("size_usdt"))
        px = live_prices.get(sym)
        if px and entry > 0:
            upnl = size * (entry - px) / entry if direc == "short" else size * (px - entry) / entry
            upnl_pct = (px - entry) / entry * 100 * (-1 if direc == "short" else 1)
            sl_dist = abs(sl - px) / px * 100 if sl > 0 else None
            tp_dist = abs(tp - px) / px * 100 if tp > 0 else None
        else:
            upnl = upnl_pct = sl_dist = tp_dist = None
        positions.append({
            "symbol": sym, "direction": direc,
            "entry": entry, "sl": sl, "tp": tp,
            "size": round(size, 2), "score": round(safe_float(pos.get("confidence")) * 20, 1),
            "px": round(px, 6) if px else None,
            "upnl": round(upnl, 2) if upnl is not None else None,
            "upnl_pct": round(upnl_pct, 2) if upnl_pct is not None else None,
            "sl_dist": round(sl_dist, 2) if sl_dist is not None else None,
            "tp_dist": round(tp_dist, 2) if tp_dist is not None else None,
            "mae": 0.0, "mfe": 0.0,
            "opened": fmt_iso_local(pos.get("entry_ts", "")),
            "status": "open",
        })

    closed = sorted(read_v13_trades(), key=lambda x: x.get("exit_ts") or x.get("entry_ts", ""))
    recent = [{
        "ts": fmt_iso_local(t.get("exit_ts") or t.get("entry_ts", "")),
        "sym": t.get("symbol", "?"),
        "dir": t.get("side", "?"),
        "entry": safe_float(t.get("entry_price")),
        "exit": safe_float(t.get("exit_price")),
        "pnl": round(safe_float(t.get("pnl")), 2),
        "out": t.get("outcome", "?"),
    } for t in reversed(closed[-30:])]

    return {"positions": positions, "recent": recent, "ts": now_local_ts(), "bot": "new"}


def build_v13_ml_payload():
    runner = rj(V13_RUNNER_FILE, {})
    bank = rj(V13_BANK_FILE, {})
    open_positions = bank.get("open_positions", {}) if isinstance(bank.get("open_positions"), dict) else {}
    model_mtime = ""
    if os.path.exists(V13_MODEL_FILE):
        try:
            model_mtime = datetime.fromtimestamp(
                os.path.getmtime(V13_MODEL_FILE), timezone.utc
            ).astimezone(DISPLAY_TZ).strftime("%Y-%m-%d %H:%M")
        except Exception:
            model_mtime = ""
    return {
        "bot": "new",
        "alpha": {"l1": 0, "l2": 0, "l3": 0, "signals": int(bank.get("total_trades", 0)),
                  "total": int(runner.get("iterations", 0) or 0), "l1_pct": 0, "l2_pct": 0,
                  "pass_rate": 0},
        "hash": {"hits": 0, "misses": 0, "savings_usd": 0, "hit_rate": 0},
        "rl": {"coin_states": [], "global_rules": [], "sim_stats": {}, "n_coins": 0,
               "sim_running": False},
        "v13": {
            "iterations": int(runner.get("iterations", 0) or 0),
            "last_check": fmt_iso_local(runner.get("last_check", "")),
            "last_train": fmt_iso_local(runner.get("last_train", "")),
            "model_mtime": model_mtime,
            "balance": round(safe_float(bank.get("balance"), 10000.0), 2),
            "open_positions": len(open_positions),
            "total_trades": int(bank.get("total_trades", 0) or 0),
            "total_pnl": round(safe_float(bank.get("total_pnl")), 2),
        },
        "ts": now_local_ts(),
    }


# ══════════════════════════════════════════════════════════════════════
# API  —  cada aba tem o seu endpoint
# ══════════════════════════════════════════════════════════════════════

@app.route("/api/portfolio")
@app.route("/api/<bot>/portfolio")
def api_portfolio(bot="old"):
    """Aba 1 — Resumo financeiro + gráficos."""
    bot = normalize_bot(bot)
    if bot == "new":
        return jsonify(build_v13_portfolio_payload())

    bank    = rj(BANK_FILE, {})
    history = rj(HISTORY_FILE, [])

    balance        = float(bank.get("balance", 10000))
    initial        = 10000.0
    total_pnl      = float(bank.get("total_pnl", 0))
    vtrades_all    = bank.get("virtual_trades", [])
    total_fees     = round(sum(float(t.get("fee_paid", 0)) for t in vtrades_all), 4)
    win_rate       = float(bank.get("win_rate", 0))
    sharpe         = float(bank.get("sharpe_ratio", 0))
    total_trades   = int(bank.get("total_trades", 0))
    winning_trades = int(bank.get("winning_trades", 0))
    open_pos       = int(bank.get("open_virtual_positions", 0))
    peak           = float(bank.get("peak_balance", initial))
    drawdown       = ((peak - balance) / peak * 100) if peak > 0 else 0
    pnl_pct        = total_pnl / initial * 100
    open_partial_pnl = round(sum(
        trade_partial_pnl(t)
        for t in vtrades_all
        if t.get("status") == "open" and t.get("tp1_hit")
    ), 2)

    # Curva PnL — prioriza virtual_trades fechados, fallback trade_history
    vtrades = bank.get("virtual_trades", [])
    CLOSED_ST = {"closed", "sl_hit", "tp_hit", "tp1_hit", "tp2_hit", "manual_close", "panic_exit"}
    closed  = sorted(
        [t for t in vtrades if t.get("status") in CLOSED_ST and t.get("actual_pnl") is not None],
        key=trade_sort_ts
    )
    if closed:
        pnl_curve, daily = [], defaultdict(float)
        cum = 0.0
        for t in closed:
            v   = trade_realized_pnl(t)
            cum += v
            day  = trade_sort_ts(t)[:10]
            pnl_curve.append({"ts": day, "cum": round(cum, 2), "pnl": round(v, 2),
                               "sym": t.get("symbol","?"), "dir": t.get("direction","?")})
            daily[day] += v
    elif history:
        pnl_curve, daily = [], defaultdict(float)
        cum = 0.0
        for t in sorted(history, key=lambda x: x.get("timestamp", "")):
            v   = float(t.get("profit_net", 0))
            cum += v
            day  = t.get("timestamp", "")[:10]
            pnl_curve.append({"ts": day, "cum": round(cum, 2), "pnl": round(v, 2),
                               "sym": t.get("symbol","?"),
                               "dir": "short" if t.get("short_trade") else "long"})
            daily[day] += v
    else:
        pnl_curve, daily = [], defaultdict(float)

    daily_data = [{"date": d, "pnl": round(v, 2)} for d, v in sorted(daily.items())]

    # Best / worst day
    best_day  = max(daily_data, key=lambda x: x["pnl"], default=None)
    worst_day = min(daily_data, key=lambda x: x["pnl"], default=None)

    # Consecutive wins/losses streak
    streak = 0; streak_type = None
    for t in reversed(closed):
        outcome = t.get("outcome", "")
        is_win  = trade_realized_pnl(t) > 0
        if streak == 0:
            streak_type = is_win
        if is_win == streak_type:
            streak += 1
        else:
            break

    return jsonify({
        "bot": "old",
        "bot_label": bot_label("old"),
        "summary": {
            "balance": round(balance, 2), "initial": initial,
            "total_pnl": round(total_pnl, 2), "pnl_pct": round(pnl_pct, 2),
            "win_rate": round(win_rate, 1), "sharpe": round(sharpe, 3),
            "total_trades": total_trades, "winning_trades": winning_trades,
            "open_positions": open_pos, "peak": round(peak, 2),
            "drawdown_pct": round(drawdown, 2),
            "total_fees": total_fees,
            "open_partial_pnl": open_partial_pnl,
            "streak": streak, "streak_type": "win" if streak_type else "loss",
            "best_day": best_day, "worst_day": worst_day,
        },
        "pnl_curve": pnl_curve,
        "daily_pnl": daily_data,
        "ts": now_local_ts(),
    })


@app.route("/api/positions")
@app.route("/api/<bot>/positions")
def api_positions(bot="old"):
    """Aba 2 — Posições abertas + últimas trades fechadas.

    Posições abertas: usa virtual_bank.json como fonte principal.
    monitored_coins.json/short_trades.json ficam apenas como fallback legado.
    """
    bot = normalize_bot(bot)
    if bot == "new":
        return jsonify(build_v13_positions_payload())

    bank    = rj(BANK_FILE, {})
    history = rj(HISTORY_FILE, [])

    # Fonte canonica: virtual_bank.json, que tambem alimenta saldo/PnL.
    vtrades = bank.get("virtual_trades", [])
    open_vtrades = {}
    for t in vtrades:
        if t.get("status") == "open":
            open_vtrades[(t.get("symbol"), t.get("direction", "long"))] = t

    raw_positions = []
    for t in open_vtrades.values():
        sym   = t.get("symbol", "?")
        direc = t.get("direction", "long")
        raw_positions.append((sym, direc, {
            "entry_price":    t.get("entry_price", 0),
            "stop_loss":      t.get("sl_price", 0),
            "take_profit":    t.get("tp_price", 0),
            "avg_score":      t.get("ensemble_score", 0),
            "kelly_fraction": t.get("kelly_frac", 0),
            "timestamp":      t.get("timestamp", ""),
            "status":         "open",
            "mae_pct":        t.get("mae_pct", 0),
            "mfe_pct":        t.get("mfe_pct", 0),
        }))

    # Fallback legado: usar ficheiros auxiliares so se o banco nao tiver abertos.
    if not raw_positions:
        longs  = rj(MONITORED, {})   # {symbol: record}
        shorts = rj(SHORT_FILE, {})  # {symbol: record}
        OPEN_STATUSES = ("open", "pending_signal", "open_paper")
        for sym, rec in longs.items():
            if rec.get("status") in OPEN_STATUSES:
                raw_positions.append((sym, "long", rec))
        for sym, rec in shorts.items():
            if rec.get("status") in OPEN_STATUSES:
                raw_positions.append((sym, "short", rec))

    live_prices = get_live_prices([sym for sym, _, _ in raw_positions]) if raw_positions else {}

    positions = []
    for sym, direc, rec in raw_positions:
        vt = open_vtrades.get((sym, direc), {})
        entry = safe_float(vt.get("entry_price"), safe_float(rec.get("entry_price"), 0))
        sl    = safe_float(vt.get("sl_price"), safe_float(rec.get("stop_loss"), 0))
        tp    = safe_float(vt.get("tp_price"), safe_float(rec.get("take_profit"), 0))
        score = safe_float(vt.get("ensemble_score"), safe_float(rec.get("avg_score"), 0))
        kelly = safe_float(vt.get("kelly_frac"), safe_float(rec.get("kelly_fraction"), 0))
        realized_pnl = trade_partial_pnl(vt) if vt else 0.0
        tp1_hit = bool(vt.get("tp1_hit")) if vt else bool(rec.get("tp1_hit"))
        px    = live_prices.get(sym)
        if px and entry > 0:
            upnl     = entry * (entry - px) / entry * (kelly * float(bank.get("balance", 10000))) if False else None
            # Calcular PnL em % e USDT a partir do tamanho Kelly × saldo
            bal      = float(bank.get("balance", 10000))
            size_est = safe_float(vt.get("position_size"), safe_float(rec.get("position_size"), 0.0))
            if size_est <= 0:
                size_est = bal * kelly if kelly > 0 else 0
            upnl     = size_est * (entry - px) / entry if direc == "short" else size_est * (px - entry) / entry
            upnl_pct = (px - entry) / entry * 100 * (-1 if direc == "short" else 1)
            sl_dist  = abs(sl - px) / px * 100 if sl > 0 else None
            tp_dist  = abs(tp - px) / px * 100 if tp > 0 else None
        else:
            upnl = upnl_pct = sl_dist = tp_dist = size_est = None

        positions.append({
            "symbol": sym, "direction": direc,
            "entry": entry, "sl": sl, "tp": tp,
            "size": round(size_est, 2) if size_est else None,
            "score": round(score, 1), "px": round(px, 6) if px else None,
            "upnl": round(upnl, 2) if upnl is not None else None,
            "upnl_pct": round(upnl_pct, 2) if upnl_pct is not None else None,
            "realized_pnl": round(realized_pnl, 2),
            "total_live_pnl": round((upnl or 0.0) + realized_pnl, 2) if upnl is not None else round(realized_pnl, 2),
            "tp1_hit": tp1_hit,
            "sl_dist": round(sl_dist, 2) if sl_dist is not None else None,
            "tp_dist": round(tp_dist, 2) if tp_dist is not None else None,
            "mae": round(float(rec.get("mae_pct", 0)), 2),
            "mfe": round(float(rec.get("mfe_pct", 0)), 2),
            "opened": fmt_iso_local(rec.get("timestamp", "")),
            "status": rec.get("status", "?"),
        })

    # ── Últimas 30 fechadas ─────────────────────────────────────────
    vtrades = bank.get("virtual_trades", [])
    CLOSED_ST = {"closed", "sl_hit", "tp_hit", "tp1_hit", "tp2_hit", "manual_close", "panic_exit"}
    closed  = sorted(
        [t for t in vtrades if t.get("status") in CLOSED_ST],
        key=trade_sort_ts
    )
    if not closed and history:
        recent = [{
            "ts":    fmt_iso_local(t.get("timestamp", "")),
            "sym":   t.get("symbol", "?"),
            "dir":   "short" if t.get("short_trade") else "long",
            "entry": t.get("entry_price"), "exit": t.get("exit_price"),
            "pnl":   round(float(t.get("profit_net", 0)), 2),
            "final_pnl": round(float(t.get("profit_net", 0)), 2),
            "partial_pnl": 0.0,
            "out":   t.get("result", "?"),
        } for t in reversed(history[-30:])]
    else:
        recent = [{
            "ts":    fmt_iso_local(trade_sort_ts(t)),
            "sym":   t.get("symbol", "?"),
            "dir":   t.get("direction", "?"),
            "entry": t.get("entry_price"), "exit": t.get("exit_price"),
            "pnl":   round(trade_realized_pnl(t), 2),
            "final_pnl": round(trade_closed_pnl(t), 2),
            "partial_pnl": round(trade_partial_pnl(t), 2),
            "out":   t.get("outcome", "?"),
        } for t in reversed(closed[-30:])]

    return jsonify({
        "positions": positions,
        "recent": recent,
        "ts": now_local_ts(),
        "bot": "old",
    })


@app.route("/api/logs/today")
@app.route("/api/<bot>/logs/today")
def api_logs_today(bot="old"):
    """Aba 3 — Logs de hoje (bot_trade.log + erros)."""
    log_file, err_file = log_paths_for_bot(bot)
    lines  = read_log_today(log_file)
    errors = read_log_today(err_file)
    return jsonify({
        "lines": lines[-500:],
        "errors": errors[-100:],
        "count": len(lines),
        "error_count": len(errors),
        "date": today_str(),
        "ts": now_local_ts(),
        "bot": normalize_bot(bot),
    })


@app.route("/api/logs/stream")
@app.route("/api/<bot>/logs/stream")
def api_logs_stream(bot="old"):
    """Aba 4 — SSE stream em tempo real do log."""
    log_file, _ = log_paths_for_bot(bot)
    def generate():
        try:
            with open(log_file, "r", encoding="utf-8", errors="replace") as f:
                f.seek(0, 2)  # vai para o fim
                while True:
                    line = f.readline()
                    if line:
                        yield f"data: {json.dumps(convert_log_line_to_local(line.rstrip()))}\n\n"
                    else:
                        time.sleep(0.4)
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps('[ERRO: ' + str(e) + ']')}\n\n"
    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.route("/api/ml")
@app.route("/api/<bot>/ml")
def api_ml(bot="old"):
    """Aba 5 — ML model info + RL rules + Alpha efficiency."""
    bot = normalize_bot(bot)
    if bot == "new":
        return jsonify(build_v13_ml_payload())

    alpha = rj(ALPHA_FILE, {})
    rl    = rj(RL_FILE, {})

    # Alpha efficiency
    l1  = alpha.get("layer1_rejected", 0)
    l2  = alpha.get("layer2_rejected", 0)
    l3  = alpha.get("layer3_called", 0)
    sig = alpha.get("signals_sent", 0)
    total = l1 + l2 + l3 + sig or 1

    # RL coin states — min_score por moeda
    # Estrutura real: {symbol: {min_score, avg_reward, final_win_rate, total_tp, total_sl, n_trades}}
    coin_states = rl.get("coin_states", {})
    coins_sorted = sorted(
        [{"symbol": k,
          "min_score":   round(float(v.get("min_score", 13)), 2),
          "avg_reward":  round(float(v.get("avg_reward", 0)), 3),
          "win_rate":    round(float(v.get("final_win_rate", 0)) * 100, 1),
          "total_tp":    int(v.get("total_tp", 0)),
          "total_sl":    int(v.get("total_sl", 0)),
          "n_trades":    int(v.get("n_trades", 0)),
        } for k, v in coin_states.items()],
        key=lambda x: x["min_score"]
    )

    # RL global rules
    # Estrutura real: {rule_name, condition, hit_count, last_reward, expires_at, active}
    global_rules = rl.get("global_rules", [])
    sim_running  = len(coin_states) == 0  # simulação ainda a correr se não há coin_states
    rules_out = []
    for r in global_rules:
        rules_out.append({
            "condition": r.get("condition", "?"),
            "name":      r.get("rule_name", "?"),
            "active":    r.get("active", False),
            "hits":      int(r.get("hit_count", 0)),
            "reward":    round(float(r.get("last_reward", 0)), 3),
            "expires":   r.get("expires_at", ""),
        })

    # RL sim global stats
    sim_stats = rl.get("sim_stats", {})

    # Hash stats from alpha
    hash_hits   = alpha.get("hash_hits", 0)
    hash_misses = alpha.get("hash_misses", 0)
    hash_savings = alpha.get("hash_savings_usd", 0)

    return jsonify({
        "bot": "old",
        "alpha": {
            "l1": l1, "l2": l2, "l3": l3, "signals": sig,
            "total": total,
            "l1_pct": round(l1/total*100, 1),
            "l2_pct": round(l2/total*100, 1),
            "pass_rate": round((l3+sig)/total*100, 1),
        },
        "hash": {
            "hits": hash_hits, "misses": hash_misses,
            "savings_usd": round(float(hash_savings), 3),
            "hit_rate": round(hash_hits/(hash_hits+hash_misses+1)*100, 1),
        },
        "rl": {
            "coin_states": coins_sorted,
            "global_rules": rules_out,
            "sim_stats": sim_stats,
            "n_coins": len(coin_states),
            "sim_running": sim_running,
        },
        "ts": now_local_ts(),
    })


@app.route("/api/training-health")
@app.route("/api/<bot>/training-health")
def api_training_health(bot="old"):
    """Training/log monitor generated by tools/bot_training_monitor.py."""
    bot = normalize_bot(bot)
    if bot == "new":
        return jsonify({
            "status": "n/a",
            "findings": ["Monitor automatico ativo apenas no bot v2 atual."],
            "counts": {},
            "latest_auc": {},
            "retrain": {},
            "tail": [],
            "ts": now_local_ts(),
        })
    data = rj(TRAINING_HEALTH_FILE, {})
    if not data:
        data = {
            "status": "unknown",
            "findings": ["Ainda sem relatorio. O timer corre de 5 em 5 minutos."],
            "counts": {},
            "latest_auc": {},
            "retrain": {},
            "tail": [],
        }
    data["ts"] = now_local_ts()
    return jsonify(data)


# ══════════════════════════════════════════════════════════════════════
# HTML
# ══════════════════════════════════════════════════════════════════════
HTML = r"""<!DOCTYPE html>
<html lang="pt">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Bot Trade v2</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
:root{--bg:#0d1117;--card:#161b22;--border:#30363d;--text:#e6edf3;--muted:#8b949e;
--green:#3fb950;--red:#f85149;--yellow:#d29922;--blue:#58a6ff;--purple:#bc8cff;}
*{box-sizing:border-box;margin:0;padding:0;}
body{background:var(--bg);color:var(--text);font-family:'Segoe UI',system-ui,sans-serif;font-size:14px;}
/* ── header ── */
header{background:var(--card);border-bottom:1px solid var(--border);padding:12px 24px;
display:flex;align-items:center;justify-content:space-between;}
header h1{font-size:17px;font-weight:700;color:var(--blue);}
.header-left{display:flex;align-items:center;gap:14px;flex-wrap:wrap;}
.bot-switch{display:flex;gap:4px;background:#0d1117;border:1px solid var(--border);border-radius:7px;padding:3px;}
.bot-switch button{background:none;border:none;color:var(--muted);padding:5px 10px;border-radius:5px;cursor:pointer;font-size:12px;}
.bot-switch button.active{background:var(--blue);color:#07111f;font-weight:700;}
#dot{width:8px;height:8px;border-radius:50%;background:var(--green);display:inline-block;
margin-right:6px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
/* ── tabs ── */
nav{background:var(--card);border-bottom:1px solid var(--border);display:flex;padding:0 20px;}
nav button{background:none;border:none;color:var(--muted);padding:10px 16px;font-size:13px;
cursor:pointer;border-bottom:2px solid transparent;transition:.15s;}
nav button:hover{color:var(--text);}
nav button.active{color:var(--blue);border-bottom-color:var(--blue);font-weight:600;}
/* ── layout ── */
.pane{display:none;padding:20px 24px;animation:fadeIn .15s;}
.pane.active{display:block;}
@keyframes fadeIn{from{opacity:0}to{opacity:1}}
.grid{display:grid;gap:14px;}
.cards{grid-template-columns:repeat(auto-fit,minmax(155px,1fr));}
.two{grid-template-columns:1fr 1fr;}
.one{grid-template-columns:1fr;}
@media(max-width:680px){.two{grid-template-columns:1fr;}}
/* ── card ── */
.card{background:var(--card);border:1px solid var(--border);border-radius:10px;padding:16px;}
.card h3{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:10px;}
.big{font-size:26px;font-weight:700;}
.sub{font-size:12px;color:var(--muted);margin-top:4px;}
.green{color:var(--green);}.red{color:var(--red);}.yellow{color:var(--yellow);}.blue{color:var(--blue);}
canvas{max-height:230px;}
/* ── table ── */
table{width:100%;border-collapse:collapse;font-size:13px;}
th{padding:7px 10px;color:var(--muted);font-size:11px;font-weight:500;text-transform:uppercase;
letter-spacing:.4px;border-bottom:1px solid var(--border);text-align:left;}
td{padding:7px 10px;border-bottom:1px solid #1d2228;}
tr:last-child td{border-bottom:none;}
tr:hover td{background:#1a1f27;}
/* ── badge ── */
.b{display:inline-block;padding:2px 8px;border-radius:10px;font-size:11px;font-weight:600;}
.b-short{background:#3d1a1a;color:var(--red);}
.b-long{background:#1a3d2a;color:var(--green);}
.b-tp,.b-profit{background:#1a3d2a;color:var(--green);}
.b-sl,.b-loss{background:#3d1a1a;color:var(--red);}
.b-open{background:#1a2d4a;color:var(--blue);}
/* ── log boxes ── */
.logbox{background:#0a0d11;border:1px solid var(--border);border-radius:6px;padding:12px;
font-family:'Cascadia Code','Consolas',monospace;font-size:11.5px;line-height:1.6;
overflow-y:auto;white-space:pre-wrap;word-break:break-all;}
.lg-info{color:#4a90d9;}.lg-warn{color:var(--yellow);}.lg-err{color:var(--red);}
.lg-signal{color:var(--green);font-weight:600;}.lg-rej{color:#6e7681;}
/* ── funnel ── */
.funnel{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:8px;}
.fstep{text-align:center;padding:12px 14px;border-radius:8px;flex:1;min-width:80px;}
.fstep .n{font-size:24px;font-weight:700;}.fstep .l{font-size:10px;color:var(--muted);text-transform:uppercase;}
.farrow{color:var(--muted);font-size:20px;}
/* ── misc ── */
.ts{font-size:11px;color:var(--muted);}
.empty{color:var(--muted);font-size:13px;padding:12px 0;}
.spin{display:inline-block;animation:spin 1s linear infinite;}
@keyframes spin{to{transform:rotate(360deg)}}
.tag-rule{display:inline-block;padding:2px 6px;border-radius:4px;font-size:11px;
background:#1a2d4a;color:var(--blue);margin:2px;}
</style>
</head>
<body>

<header>
  <div class="header-left">
    <h1 id="bot-title">Bot Antigo v2</h1>
    <div class="bot-switch">
      <button class="active" onclick="switchBot('old',this)">Bot Antigo v2</button>
    </div>
  </div>
  <div style="display:flex;align-items:center;gap:14px">
    <span><span id="dot"></span><span class="ts" id="ts-hdr">--</span></span>
    <button onclick="refreshActive()" style="background:var(--border);border:none;color:var(--text);
      padding:5px 12px;border-radius:6px;cursor:pointer;font-size:12px">↻</button>
  </div>
</header>

<nav>
  <button class="active" onclick="switchTab('portfolio',this)">📊 Portfolio</button>
  <button onclick="switchTab('positions',this)">🔓 Posições</button>
  <button onclick="switchTab('logs-today',this)">📅 Logs Hoje</button>
  <button onclick="switchTab('logs-live',this)">⚡ Logs Live</button>
  <button onclick="switchTab('ml',this)">🤖 ML & RL</button>
</nav>

<!-- ═══════════ TAB 1 — PORTFOLIO ═══════════ -->
<div id="pane-portfolio" class="pane active">
  <div class="grid cards" style="margin-bottom:14px">
    <div class="card"><h3>Saldo</h3><div class="big blue" id="p-bal">--</div><div class="sub" id="p-ini">--</div></div>
    <div class="card"><h3>PnL Total</h3><div class="big" id="p-pnl">--</div><div class="sub" id="p-pnlp">--</div></div>
    <div class="card"><h3>Win Rate</h3><div class="big" id="p-wr">--</div><div class="sub" id="p-wrsub">--</div></div>
    <div class="card"><h3>Sharpe Ratio</h3><div class="big" id="p-sh">--</div><div class="sub">Risco/Retorno</div></div>
    <div class="card"><h3>Drawdown</h3><div class="big red" id="p-dd">--</div><div class="sub" id="p-peak">--</div></div>
    <div class="card"><h3>Streak</h3><div class="big" id="p-str">--</div><div class="sub" id="p-strsub">--</div></div>
    <div class="card"><h3>Taxas Pagas</h3><div class="big red" id="p-fees">--</div><div class="sub">Total acumulado</div></div>
  </div>
  <div class="grid two" style="margin-bottom:14px">
    <div class="card">
      <h3>📈 PnL Cumulativo</h3>
      <canvas id="cPnl"></canvas>
      <div id="no-pnl" class="empty" style="display:none">Sem trades fechados ainda.</div>
    </div>
    <div class="card">
      <h3>📊 PnL por Dia</h3>
      <canvas id="cDaily"></canvas>
      <div id="no-daily" class="empty" style="display:none">Sem dados diários.</div>
    </div>
  </div>
  <div class="grid two">
    <div class="card" id="best-worst-card">
      <h3>📌 Melhor / Pior Dia</h3>
      <div id="bw-content" class="empty">--</div>
    </div>
    <div class="card">
      <h3>⏰ Posições Abertas</h3>
      <div class="big blue" id="p-open">--</div>
      <div class="sub" id="p-open-sub">--</div>
    </div>
  </div>
</div>

<!-- ═══════════ TAB 2 — POSIÇÕES ═══════════ -->
<div id="pane-positions" class="pane">
  <div class="card" style="margin-bottom:14px">
    <h3>🔓 Posições Abertas <span class="ts" id="pos-ts"></span></h3>
    <div id="pos-empty" class="empty">Nenhuma posição aberta.</div>
    <div id="pos-wrap" style="overflow-x:auto;display:none">
      <table><thead><tr>
        <th>Par</th><th>Dir</th><th>Entrada</th><th>Preço</th>
        <th>PnL Live</th><th>SL dist</th><th>TP dist</th>
        <th>Size</th><th>Score</th><th>MAE</th><th>MFE</th><th>Aberta em</th>
      </tr></thead><tbody id="tbody-pos"></tbody></table>
    </div>
  </div>
  <div class="card">
    <h3>📋 Últimas 30 Trades Fechadas</h3>
    <div id="trades-empty" class="empty">Sem histórico.</div>
    <div id="trades-wrap" style="overflow-x:auto;display:none">
      <table><thead><tr>
        <th>Data</th><th>Par</th><th>Dir</th><th>Entrada</th><th>Saída</th><th>PnL</th><th>Resultado</th>
      </tr></thead><tbody id="tbody-trades"></tbody></table>
    </div>
  </div>
</div>

<!-- ═══════════ TAB 3 — LOGS HOJE ═══════════ -->
<div id="pane-logs-today" class="pane">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <span style="font-weight:600" id="lt-title">Logs de hoje</span>
    <div style="display:flex;gap:8px">
      <input id="lt-filter" placeholder="Filtrar... ex: SIGNAL, L1, ERROR"
        style="background:var(--card);border:1px solid var(--border);color:var(--text);
        padding:5px 10px;border-radius:6px;font-size:12px;width:220px"
        oninput="filterLogs()">
      <button onclick="loadLogsToday()"
        style="background:var(--border);border:none;color:var(--text);padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px">↻</button>
    </div>
  </div>
  <!-- Erros de hoje -->
  <div class="card" style="margin-bottom:12px" id="err-card">
    <h3>🔴 Erros de Hoje <span id="err-count" class="ts"></span></h3>
    <div class="logbox" id="err-box" style="max-height:120px;display:none"></div>
    <div id="err-none" class="empty">Sem erros hoje. ✅</div>
  </div>
  <!-- Todos os logs -->
  <div class="card">
    <h3>Todos os logs de hoje <span id="lt-count" class="ts"></span></h3>
    <div class="logbox" id="lt-box" style="max-height:600px"></div>
  </div>
</div>

<!-- ═══════════ TAB 4 — LOGS LIVE ═══════════ -->
<div id="pane-logs-live" class="pane">
  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
    <span style="font-weight:600">⚡ Logs em Tempo Real (SSE)</span>
    <div style="display:flex;gap:8px;align-items:center">
      <span id="live-status" class="ts">Desligado</span>
      <label style="display:flex;align-items:center;gap:5px;font-size:12px;cursor:pointer">
        <input type="checkbox" id="live-scroll" checked> Auto-scroll
      </label>
      <input id="live-filter" placeholder="Filtrar..."
        style="background:var(--card);border:1px solid var(--border);color:var(--text);
        padding:5px 10px;border-radius:6px;font-size:12px;width:200px">
      <button id="live-clear"
        style="background:var(--border);border:none;color:var(--text);padding:5px 10px;border-radius:6px;cursor:pointer;font-size:12px"
        onclick="document.getElementById('live-box').innerHTML=''">🗑 Limpar</button>
    </div>
  </div>
  <div class="card" style="padding:0">
    <div class="logbox" id="live-box" style="max-height:70vh;padding:14px"></div>
  </div>
</div>

<!-- ═══════════ TAB 5 — ML & RL ═══════════ -->
<div id="pane-ml" class="pane">
  <div class="grid two" style="margin-bottom:14px">
    <!-- Funil de sinais -->
    <div class="card">
      <h3>🔬 Funil de Sinais</h3>
      <div class="funnel" id="funnel"></div>
      <div style="margin-top:12px;font-size:12px;color:var(--muted)" id="funnel-rate"></div>
    </div>
    <!-- Hash cache -->
    <div class="card">
      <h3>💾 Decision Hash Cache</h3>
      <div id="hash-info" class="empty">--</div>
    </div>
  </div>
  <div class="card" style="margin-bottom:14px">
    <h3>Monitor Treino / Logs <span class="ts" id="train-ts"></span></h3>
    <div class="grid cards" style="margin-bottom:12px">
      <div><div class="big" id="train-status">--</div><div class="sub">Estado</div></div>
      <div><div class="big blue" id="train-auc">--</div><div class="sub">AUC ML</div></div>
      <div><div class="big yellow" id="train-minscore">--</div><div class="sub">RL minScore L/S</div></div>
      <div><div class="big" id="train-mem">--</div><div class="sub">Memoria livre</div></div>
    </div>
    <div id="train-findings" class="empty">Ainda sem relatorio automatico.</div>
    <div class="logbox" id="train-tail" style="display:none;max-height:170px;margin-top:10px"></div>
  </div>
  <!-- RL Global Rules -->
  <div class="card" style="margin-bottom:14px">
    <h3>🧠 RL Global Rules <span class="ts" id="rl-ts"></span></h3>
    <div id="rules-empty" class="empty">Sem regras globais ainda.</div>
    <div id="rules-wrap" style="overflow-x:auto;display:none">
      <table><thead><tr>
        <th>Condição</th><th>Estado</th><th>Hits</th><th>Avg Reward</th><th>Expira em</th>
      </tr></thead><tbody id="tbody-rules"></tbody></table>
    </div>
  </div>
  <!-- Min score por moeda -->
  <div class="card">
    <h3>📊 Min Score por Moeda (RL aprendido)</h3>
    <div id="coins-empty" class="empty">Sem dados de moedas.</div>
    <div id="coins-wrap" style="overflow-x:auto;display:none">
      <table><thead><tr>
        <th>Moeda</th><th>Min Score</th><th>Avg Reward</th><th>Trades</th><th>TP</th><th>Win Rate</th>
      </tr></thead><tbody id="tbody-coins"></tbody></table>
    </div>
  </div>
</div>

<script>
// ── state ─────────────────────────────────────────────────
let currentTab = 'portfolio';
let currentBot = 'old';
let sseSource  = null;
let charts     = {};
let refreshTimer = null;

function api(path) {
  return `/api/${currentBot}/${path}`;
}

function switchBot(bot, btn) {
  if (bot === currentBot) return;
  stopSSE();
  clearTimeout(refreshTimer);
  currentBot = bot;
  document.querySelectorAll('.bot-switch button').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('bot-title').textContent = bot === 'new' ? 'Bot Novo V13' : 'Bot Antigo v2';
  Object.values(charts).forEach(ch => { if (ch) ch.destroy(); });
  charts = {};
  loadTab(currentTab);
}

// ── tab switching ──────────────────────────────────────────
function switchTab(name, btn) {
  document.querySelectorAll('.pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav button').forEach(b => b.classList.remove('active'));
  document.getElementById('pane-' + name).classList.add('active');
  btn.classList.add('active');
  const prev = currentTab;
  currentTab = name;

  // Stop SSE if leaving live logs
  if (prev === 'logs-live' && name !== 'logs-live') stopSSE();

  // Load data for new tab
  clearTimeout(refreshTimer);
  loadTab(name);
}

function loadTab(name) {
  if      (name === 'portfolio')  loadPortfolio();
  else if (name === 'positions')  loadPositions();
  else if (name === 'logs-today') loadLogsToday();
  else if (name === 'logs-live')  startSSE();
  else if (name === 'ml')         loadML();
}

function refreshActive() { loadTab(currentTab); }

// Auto-refresh every 30s for active tab (except live which is push)
function scheduleRefresh() {
  clearTimeout(refreshTimer);
  if (currentTab !== 'logs-live') {
    refreshTimer = setTimeout(() => { loadTab(currentTab); scheduleRefresh(); }, 30000);
  }
}

// ── helpers ───────────────────────────────────────────────
const f = (n, d=2) => n == null ? '—' : Number(n).toFixed(d);
const s = n => n >= 0 ? '+' : '';
const c = n => n >= 0 ? 'green' : 'red';

function colorLine(line) {
  if (line.includes('[ERROR]') || line.includes('ERROR'))   return `<span class="lg-err">${esc(line)}</span>`;
  if (line.includes('[WARNING]') || line.includes('WARNING')) return `<span class="lg-warn">${esc(line)}</span>`;
  if (line.includes('[SIGNAL]') || line.includes('APROVADO')) return `<span class="lg-signal">${esc(line)}</span>`;
  if (line.includes('[L1]') || line.includes('[L2]') || line.includes('VETO') || line.includes('CORR') || line.includes('REJECT'))
    return `<span class="lg-rej">${esc(line)}</span>`;
  if (line.includes('[INFO]')) return `<span class="lg-info">${esc(line)}</span>`;
  return esc(line);
}

function esc(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}

// ── portfolio ─────────────────────────────────────────────
function loadPortfolio() {
  fetch(api('portfolio')).then(r=>r.json()).then(d => {
    const s2 = d.summary;
    document.getElementById('ts-hdr').textContent = d.ts;
    if (d.bot_label) document.getElementById('bot-title').textContent = d.bot_label;
    document.getElementById('p-bal').textContent  = f(s2.balance) + ' USDT';
    document.getElementById('p-ini').textContent  = 'Inicial: ' + f(s2.initial) + ' USDT';

    const pnlEl = document.getElementById('p-pnl');
    pnlEl.textContent = (s2.total_pnl >= 0 ? '+' : '') + f(s2.total_pnl) + ' USDT';
    pnlEl.className = 'big ' + c(s2.total_pnl);
    document.getElementById('p-pnlp').textContent = (s2.pnl_pct >= 0 ? '+' : '') + f(s2.pnl_pct) + '%';

    const wrEl = document.getElementById('p-wr');
    wrEl.textContent = f(s2.win_rate, 1) + '%';
    wrEl.className = 'big ' + (s2.win_rate >= 50 ? 'green' : 'yellow');
    document.getElementById('p-wrsub').textContent =
      s2.winning_trades + 'W / ' + (s2.total_trades - s2.winning_trades) + 'L (' + s2.total_trades + ' total)';

    const shEl = document.getElementById('p-sh');
    shEl.textContent = f(s2.sharpe, 3);
    shEl.className = 'big ' + (s2.sharpe >= 1 ? 'green' : s2.sharpe >= 0 ? 'yellow' : 'red');

    document.getElementById('p-dd').textContent   = '-' + f(s2.drawdown_pct) + '%';
    document.getElementById('p-peak').textContent = 'Pico: ' + f(s2.peak) + ' USDT';

    const strEl = document.getElementById('p-str');
    strEl.textContent = s2.streak + 'x';
    strEl.className = 'big ' + (s2.streak_type === 'win' ? 'green' : 'red');
    document.getElementById('p-strsub').textContent =
      s2.streak > 0 ? (s2.streak_type === 'win' ? '✅ Vitórias consecutivas' : '❌ Perdas consecutivas') : '—';

    document.getElementById('p-fees').textContent = '-' + f(s2.total_fees, 4) + ' USDT';

    document.getElementById('p-open').textContent    = s2.open_positions;
    const openPartial = s2.open_partial_pnl && Math.abs(s2.open_partial_pnl) > 0.004
      ? ` | TP1 realizado aberto: ${s(s2.open_partial_pnl)}${f(s2.open_partial_pnl)} USDT`
      : '';
    document.getElementById('p-open-sub').textContent = 'posicoes em ' + (d.bot_label || 'paper trading') + openPartial;

    // Melhor/Pior dia
    const bw = document.getElementById('bw-content');
    if (s2.best_day || s2.worst_day) {
      bw.innerHTML = (s2.best_day  ? `<div style="margin-bottom:6px">🏆 <b>Melhor:</b> ${s2.best_day.date} → <span class="green">+${f(s2.best_day.pnl)} USDT</span></div>` : '') +
                     (s2.worst_day ? `<div>💣 <b>Pior:</b> ${s2.worst_day.date} → <span class="red">${f(s2.worst_day.pnl)} USDT</span></div>` : '');
      bw.className = '';
    }

    // Charts
    buildPnlChart(d.pnl_curve);
    buildDailyChart(d.daily_pnl);
    scheduleRefresh();
  }).catch(() => scheduleRefresh());
}

function buildPnlChart(data) {
  const el = document.getElementById('cPnl');
  const noEl = document.getElementById('no-pnl');
  if (!data || data.length === 0) { el.style.display='none'; noEl.style.display='block'; return; }
  el.style.display=''; noEl.style.display='none';
  const last = data[data.length-1].cum;
  if (charts.pnl) charts.pnl.destroy();
  charts.pnl = new Chart(el.getContext('2d'), {
    type: 'line',
    data: {
      labels: data.map(d=>d.ts),
      datasets: [{
        data: data.map(d=>d.cum),
        borderColor: last >= 0 ? '#3fb950' : '#f85149',
        backgroundColor: last >= 0 ? 'rgba(63,185,80,.07)' : 'rgba(248,81,73,.07)',
        borderWidth: 2, pointRadius: 1, fill: true, tension: 0.35,
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false},tooltip:{callbacks:{
        label: ctx => {
          const d = data[ctx.dataIndex];
          return [`Cum: ${ctx.parsed.y >= 0 ? '+' : ''}${ctx.parsed.y.toFixed(2)} USDT`,
                  `Trade: ${d.pnl >= 0 ? '+' : ''}${d.pnl.toFixed(2)} (${d.sym} ${d.dir})`];
        }
      }}},
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:6},grid:{color:'#21262d'}},
        y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}
      }
    }
  });
}

function buildDailyChart(data) {
  const el = document.getElementById('cDaily');
  const noEl = document.getElementById('no-daily');
  if (!data || data.length === 0) { el.style.display='none'; noEl.style.display='block'; return; }
  el.style.display=''; noEl.style.display='none';
  if (charts.daily) charts.daily.destroy();
  charts.daily = new Chart(el.getContext('2d'), {
    type: 'bar',
    data: {
      labels: data.map(d=>d.date),
      datasets: [{
        data: data.map(d=>d.pnl),
        backgroundColor: data.map(d => d.pnl >= 0 ? 'rgba(63,185,80,.75)' : 'rgba(248,81,73,.75)'),
        borderRadius: 4,
      }]
    },
    options: {
      responsive:true, maintainAspectRatio:true,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:'#8b949e',maxTicksLimit:8},grid:{display:false}},
        y:{ticks:{color:'#8b949e'},grid:{color:'#21262d'}}
      }
    }
  });
}

// ── positions ─────────────────────────────────────────────
function loadPositions() {
  fetch(api('positions')).then(r=>r.json()).then(d => {
    document.getElementById('ts-hdr').textContent = d.ts;
    document.getElementById('pos-ts').textContent = '— ' + d.ts;

    const tbody = document.getElementById('tbody-pos');
    tbody.innerHTML = '';
    if (d.positions && d.positions.length > 0) {
      document.getElementById('pos-empty').style.display = 'none';
      document.getElementById('pos-wrap').style.display = '';
      d.positions.forEach(p => {
        let pnlHtml = p.upnl != null
          ? `<span class="${c(p.total_live_pnl)}">${s(p.total_live_pnl)}${f(p.total_live_pnl)}</span> <span class="${c(p.upnl)}">(${s(p.upnl_pct)}${f(p.upnl_pct)}%)</span>`
          : '—';
        if (p.realized_pnl && Math.abs(p.realized_pnl) > 0.004) {
          pnlHtml += `<div class="sub">TP1 realizado: ${s(p.realized_pnl)}${f(p.realized_pnl)} USDT</div>`;
        }
        tbody.innerHTML += `<tr>
          <td><b>${p.symbol}</b></td>
          <td><span class="b b-${p.direction}">${p.direction.toUpperCase()}</span></td>
          <td>${f(p.entry,6)}</td>
          <td>${p.px != null ? f(p.px,6) : '—'}</td>
          <td>${pnlHtml}</td>
          <td>${p.sl_dist != null ? f(p.sl_dist)+'%' : '—'}</td>
          <td>${p.tp_dist != null ? f(p.tp_dist)+'%' : '—'}</td>
          <td>${f(p.size)} USDT</td>
          <td>${f(p.score,1)}/20</td>
          <td class="red">${f(p.mae)}%</td>
          <td class="green">${f(p.mfe)}%</td>
          <td class="ts">${p.opened}</td>
        </tr>`;
      });
    } else {
      document.getElementById('pos-empty').style.display = '';
      document.getElementById('pos-wrap').style.display = 'none';
    }

    const tbody2 = document.getElementById('tbody-trades');
    tbody2.innerHTML = '';
    if (d.recent && d.recent.length > 0) {
      document.getElementById('trades-empty').style.display = 'none';
      document.getElementById('trades-wrap').style.display = '';
      d.recent.forEach(t => {
        const oc = t.out || '';
        const bcls = (oc.includes('tp') || oc==='profit') ? 'b-tp'
                   : (oc.includes('sl') || oc==='loss')   ? 'b-sl' : 'b-open';
        const partialHtml = t.partial_pnl && Math.abs(t.partial_pnl) > 0.004
          ? `<div class="sub">incl. TP1 ${s(t.partial_pnl)}${f(t.partial_pnl)} | fecho ${s(t.final_pnl)}${f(t.final_pnl)}</div>`
          : '';
        tbody2.innerHTML += `<tr>
          <td class="ts">${t.ts}</td>
          <td><b>${t.sym}</b></td>
          <td><span class="b b-${t.dir}">${t.dir.toUpperCase()}</span></td>
          <td>${t.entry != null ? f(t.entry,6) : '—'}</td>
          <td>${t.exit  != null ? f(t.exit, 6) : '—'}</td>
          <td class="${c(t.pnl)}">${s(t.pnl)}${f(t.pnl)} USDT${partialHtml}</td>
          <td><span class="b ${bcls}">${oc}</span></td>
        </tr>`;
      });
    } else {
      document.getElementById('trades-empty').style.display = '';
      document.getElementById('trades-wrap').style.display = 'none';
    }
    scheduleRefresh();
  }).catch(() => scheduleRefresh());
}

// ── logs hoje ─────────────────────────────────────────────
let allLogLines = [];
function loadLogsToday() {
  fetch(api('logs/today')).then(r=>r.json()).then(d => {
    document.getElementById('ts-hdr').textContent = d.ts;
    document.getElementById('lt-title').textContent = 'Logs de ' + d.date + ' (' + d.count + ' linhas)';
    document.getElementById('lt-count').textContent = '— ' + d.count + ' linhas';
    allLogLines = d.lines;

    // Erros
    if (d.errors && d.errors.length > 0) {
      document.getElementById('err-count').textContent = '— ' + d.error_count + ' erros';
      document.getElementById('err-box').style.display = '';
      document.getElementById('err-none').style.display = 'none';
      document.getElementById('err-box').innerHTML = d.errors.map(colorLine).join('\n');
    } else {
      document.getElementById('err-box').style.display = 'none';
      document.getElementById('err-none').style.display = '';
    }

    filterLogs();
    scheduleRefresh();
  }).catch(() => scheduleRefresh());
}

function filterLogs() {
  const q = document.getElementById('lt-filter').value.trim().toLowerCase();
  const filtered = q ? allLogLines.filter(l => l.toLowerCase().includes(q)) : allLogLines;
  const box = document.getElementById('lt-box');
  box.innerHTML = filtered.map(colorLine).join('\n');
  box.scrollTop = box.scrollHeight;
}

// ── logs live (SSE) ───────────────────────────────────────
function startSSE() {
  if (sseSource) return;
  document.getElementById('live-status').textContent = '🟢 Ligado';
  const box = document.getElementById('live-box');
  sseSource = new EventSource(api('logs/stream'));
  sseSource.onmessage = e => {
    const line = JSON.parse(e.data);
    const q = document.getElementById('live-filter').value.trim().toLowerCase();
    if (q && !line.toLowerCase().includes(q)) return;
    const span = document.createElement('span');
    span.innerHTML = colorLine(line) + '\n';
    box.appendChild(span);
    // Limitar a 1000 linhas
    while (box.children.length > 1000) box.removeChild(box.firstChild);
    if (document.getElementById('live-scroll').checked)
      box.scrollTop = box.scrollHeight;
  };
  sseSource.onerror = () => {
    document.getElementById('live-status').textContent = '🔴 Erro (a reconectar...)';
  };
}

function stopSSE() {
  if (sseSource) { sseSource.close(); sseSource = null; }
  document.getElementById('live-status').textContent = 'Desligado';
}

// ── ML & RL ───────────────────────────────────────────────
function loadML() {
  fetch(api('ml')).then(r=>r.json()).then(d => {
    document.getElementById('ts-hdr').textContent = d.ts;
    document.getElementById('rl-ts').textContent  = '— ' + d.ts;

    loadTrainingHealth();
    if (d.bot === 'new') {
      const v = d.v13;
      document.getElementById('funnel').innerHTML = `
        <div class="fstep" style="background:#1a2d4a"><div class="n blue">${v.iterations}</div><div class="l">Iteracoes</div></div>
        <div class="farrow">|</div>
        <div class="fstep" style="background:#1a3d2a"><div class="n green">${v.open_positions}</div><div class="l">Abertas</div></div>
        <div class="farrow">|</div>
        <div class="fstep" style="background:#3d2a1a"><div class="n yellow">${v.total_trades}</div><div class="l">Trades</div></div>
        <div class="farrow">|</div>
        <div class="fstep" style="background:#1a2d4a"><div class="n ${v.total_pnl >= 0 ? 'green' : 'red'}">${s(v.total_pnl)}${f(v.total_pnl)}</div><div class="l">PnL</div></div>
      `;
      document.getElementById('funnel-rate').textContent =
        `Ultimo check: ${v.last_check || '--'} | Treino: ${v.last_train || '--'} | Modelo: ${v.model_mtime || '--'}`;
      document.getElementById('hash-info').innerHTML = `
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px">
          <div><div class="big blue">${f(v.balance)}</div><div class="sub">Saldo V13</div></div>
          <div><div class="big ${v.total_pnl >= 0 ? 'green' : 'red'}">${s(v.total_pnl)}${f(v.total_pnl)}</div><div class="sub">PnL V13</div></div>
          <div><div style="font-size:18px;font-weight:700">${v.open_positions}</div><div class="sub">Posicoes abertas</div></div>
          <div><div style="font-size:18px;font-weight:700">${v.total_trades}</div><div class="sub">Trades fechadas</div></div>
        </div>`;
      document.getElementById('rules-empty').textContent = 'O V13 nao usa o motor RL/global-rules do bot antigo.';
      document.getElementById('rules-empty').style.display = '';
      document.getElementById('rules-wrap').style.display = 'none';
      document.getElementById('coins-empty').textContent = 'O V13 le sinais do DecisionEngine + MLFilter em strategy_v2.';
      document.getElementById('coins-empty').style.display = '';
      document.getElementById('coins-wrap').style.display = 'none';
      scheduleRefresh();
      return;
    }

    // Funnel
    const a = d.alpha;
    document.getElementById('funnel').innerHTML = `
      <div class="fstep" style="background:#1a2d4a"><div class="n blue">${a.total}</div><div class="l">Analisados</div></div>
      <div class="farrow">→</div>
      <div class="fstep" style="background:#3d1a1a"><div class="n red">${a.l1}</div><div class="l">L1 rejeit.<br><small>${a.l1_pct}%</small></div></div>
      <div class="farrow">→</div>
      <div class="fstep" style="background:#3d2a1a"><div class="n yellow">${a.l2}</div><div class="l">L2 rejeit.<br><small>${a.l2_pct}%</small></div></div>
      <div class="farrow">→</div>
      <div class="fstep" style="background:#1a3d1a"><div class="n" style="color:#4dbb5f">${a.l3}</div><div class="l">L3 conselho</div></div>
      <div class="farrow">→</div>
      <div class="fstep" style="background:#1a3d2a"><div class="n green">${a.signals}</div><div class="l">✅ Sinais</div></div>
    `;
    document.getElementById('funnel-rate').textContent =
      `Taxa de passagem: ${a.pass_rate}% | L1 bloqueou ${a.l1_pct}% | L2 bloqueou ${a.l2_pct}%`;

    // Hash
    const h = d.hash;
    document.getElementById('hash-info').innerHTML = `
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:4px">
        <div><div class="big blue">${h.hit_rate}%</div><div class="sub">Hit Rate</div></div>
        <div><div class="big green">$${h.savings_usd}</div><div class="sub">$ Poupado</div></div>
        <div><div style="font-size:18px;font-weight:700">${h.hits}</div><div class="sub">Cache Hits</div></div>
        <div><div style="font-size:18px;font-weight:700">${h.misses}</div><div class="sub">Cache Misses</div></div>
      </div>`;

    // Global rules
    const rules = d.rl.global_rules;
    if (rules && rules.length > 0) {
      document.getElementById('rules-empty').style.display = 'none';
      document.getElementById('rules-wrap').style.display = '';
      const tb = document.getElementById('tbody-rules');
      tb.innerHTML = '';
      rules.forEach(r => {
        const rwCls = r.reward >= 0 ? 'green' : 'red';
        const exp = r.expires ? r.expires.slice(0,16) : 'Permanente';
        tb.innerHTML += `<tr>
          <td><code style="background:#161b22;padding:2px 6px;border-radius:4px;font-size:12px">${esc(r.condition)}</code></td>
          <td><span style="color:${r.active ? '#f85149' : '#3fb950'}">${r.active ? '🔴 Bloqueio ativo' : '🟢 Inativo'}</span></td>
          <td>${r.hits}</td>
          <td class="${rwCls}">${r.reward >= 0 ? '+' : ''}${r.reward}</td>
          <td class="ts">${exp}</td>
        </tr>`;
      });
    } else {
      document.getElementById('rules-empty').style.display = '';
      document.getElementById('rules-wrap').style.display = 'none';
    }

    // Coin states
    const coins = d.rl.coin_states;
    if (d.rl.sim_running) {
      document.getElementById('coins-empty').textContent = '⏳ Simulação RL ainda a correr em background... (4-8h após arranque)';
      document.getElementById('coins-empty').style.display = '';
      document.getElementById('coins-wrap').style.display = 'none';
    } else if (coins && coins.length > 0) {
      document.getElementById('coins-empty').style.display = 'none';
      document.getElementById('coins-wrap').style.display = '';
      const tb2 = document.getElementById('tbody-coins');
      tb2.innerHTML = '';
      coins.forEach(c2 => {
        const wr    = c2.win_rate != null ? c2.win_rate.toFixed(0) + '%' : '—';
        const rwCls = c2.avg_reward >= 0 ? 'green' : 'red';
        const msCls = c2.min_score <= 13 ? 'green' : c2.min_score <= 15 ? 'yellow' : 'red';
        tb2.innerHTML += `<tr>
          <td><b>${c2.symbol}</b></td>
          <td class="${msCls}">${c2.min_score}</td>
          <td class="${rwCls}">${c2.avg_reward >= 0 ? '+' : ''}${c2.avg_reward}</td>
          <td>${c2.n_trades}</td>
          <td class="green">${c2.total_tp}</td>
          <td>${wr}</td>
        </tr>`;
      });
    } else {
      document.getElementById('coins-empty').style.display = '';
      document.getElementById('coins-wrap').style.display = 'none';
    }
    scheduleRefresh();
  }).catch(() => scheduleRefresh());
}

// ── init ──────────────────────────────────────────────────
function loadTrainingHealth() {
  fetch(api('training-health')).then(r=>r.json()).then(h => {
    const status = (h.status || 'unknown').toLowerCase();
    const cls = status === 'alert' ? 'red' : (status === 'watch' ? 'yellow' : 'green');
    const auc = h.latest_auc || {};
    const book = h.latest_min_score_book || {};
    const counts = h.counts || {};
    document.getElementById('train-ts').textContent = h.ts ? '— ' + h.ts : '';
    document.getElementById('train-status').className = 'big ' + cls;
    document.getElementById('train-status').textContent = status.toUpperCase();
    document.getElementById('train-auc').textContent = `L ${auc.LONG ?? '--'} / S ${auc.SHORT ?? '--'}`;
    document.getElementById('train-minscore').textContent = `L ${book.LONG ?? '--'} / S ${book.SHORT ?? '--'}`;
    document.getElementById('train-mem').textContent = h.memory_available_mb != null ? `${h.memory_available_mb} MB` : '--';
    const findings = h.findings || [];
    document.getElementById('train-findings').innerHTML =
      findings.length ? findings.map(x => `<div>• ${esc(x)}</div>`).join('') : 'Sem anomalias.';
    const tailBox = document.getElementById('train-tail');
    const tail = h.tail || [];
    if (tail.length || counts.rl_min_score_high) {
      tailBox.style.display = '';
      tailBox.innerHTML = esc(tail.slice(-40).join('\n'));
    } else {
      tailBox.style.display = 'none';
      tailBox.innerHTML = '';
    }
  }).catch(() => {});
}

loadPortfolio();
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


if __name__ == "__main__":
    print("=" * 55)
    print("  Bot Trade v2 — Dashboard v2")
    print("  http://localhost:5001")
    print()
    print("  SSH tunnel:")
    print('  ssh -i "<caminho-da-chave.pem>" -L 5001:localhost:5001 <utilizador>@<servidor>')
    print("=" * 55)
    app.run(host="127.0.0.1", port=5001, debug=False, threaded=True)
