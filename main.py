import os
import json
import time
import tempfile
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
import numpy as np
import pandas as pd
from google import genai

# =====================================================================
# 1. KONFIGURASI KEAMANAN & PARAMETER MODAL (DARI GITHUB SECRETS)
# =====================================================================
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

STATE_FILE = "trade_state.json"
HISTORY_FILE = "trade_history.json"
TIMEZONE = ZoneInfo("Asia/Jakarta")

# Parameter Modal & Risiko sesuai Audit Brief
DEFAULT_EQUITY_USDT = 10.00
MIN_MARGIN_RATIO = 0.20           # Minimal 20% ekuitas (2.00 USDT pada 10 USDT)
MAX_MARGIN_RATIO_STRICT = 0.40    # Strictly < 40% ekuitas (< 4.00 USDT pada 10 USDT)
TARGET_MARGIN_RATIO = 0.25        # Target margin 25% ekuitas (2.50 USDT)
MAX_STOP_LOSS_EQUITY_RISK = 0.045 # Maksimal risiko SL bersih 4.5% ekuitas (~0.45 USDT)
MAX_UNIQUE_COINS_PER_DAY = 3

# Biaya Transaksi & Slippage Bitget USDT-Futures
TAKER_FEE_RATE = 0.0006
SLIPPAGE_BUFFER_RATE = 0.0005
ROUND_TRIP_COST_RATE = (2 * TAKER_FEE_RATE) + SLIPPAGE_BUFFER_RATE

EXCLUDED_COINS = {"BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "USDCUSDT"}

GRANULARITY_MS = {
    "15m": 15 * 60 * 1000,
    "1H": 60 * 60 * 1000,
    "4H": 4 * 60 * 60 * 1000,
}

# =====================================================================
# 2. MANAJEMEN STATE ATOMIK & KUOTA 3 KOIN PER HARI (ASIA/JAKARTA)
# =====================================================================
def get_trading_day() -> str:
    return datetime.now(TIMEZONE).date().isoformat()

def atomic_write_json(filepath: str, data: dict | list) -> None:
    dir_name = os.path.dirname(os.path.abspath(filepath)) or "."
    with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, encoding="utf-8") as tmp_file:
        json.dump(data, tmp_file, indent=2)
        temp_name = tmp_file.name
    os.replace(temp_name, filepath)

def ensure_persistence_files() -> None:
    if not os.path.exists(HISTORY_FILE):
        atomic_write_json(HISTORY_FILE, [])

def load_trade_state() -> dict:
    today = get_trading_day()
    default_state = {
        "equity_usdt": DEFAULT_EQUITY_USDT,
        "active": False,
        "coin": None,
        "bias": None,
        "entry": None,
        "sl": None,
        "original_sl": None,
        "tp1": None,
        "tp2": None,
        "tp3": None,
        "margin_usdt": None,
        "leverage": None,
        "notional_usdt": None,
        "tp1_hit": False,
        "tp2_hit": False,
        "last_checked_candle_ts": 0,
        "daily_tracker": {
            "date": today,
            "symbols": []
        }
    }
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    loaded = json.loads(content)
                    default_state.update(loaded)
        except Exception as e:
            print(f"⚠️ Gagal membaca {STATE_FILE}, membuat state baru: {e}")

    tracker = default_state.get("daily_tracker", {})
    if tracker.get("date") != today:
        default_state["daily_tracker"] = {"date": today, "symbols": []}
    return default_state

def save_trade_state(state: dict) -> None:
    atomic_write_json(STATE_FILE, state)

def can_scan_symbol(symbol: str, state: dict) -> bool:
    today = get_trading_day()
    tracker = state.setdefault("daily_tracker", {"date": today, "symbols": []})
    if tracker.get("date") != today:
        tracker["date"] = today
        tracker["symbols"] = []
    symbols = set(tracker.get("symbols", []))
    if symbol in symbols:
        return True
    return len(symbols) < MAX_UNIQUE_COINS_PER_DAY

def register_accepted_signal(symbol: str, state: dict) -> bool:
    if not can_scan_symbol(symbol, state):
        return False
    symbols = set(state["daily_tracker"].get("symbols", []))
    symbols.add(symbol)
    state["daily_tracker"]["symbols"] = sorted(symbols)
    return True

def load_trade_history() -> list:
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
                if content:
                    return json.loads(content)
        except Exception:
            pass
    return []

def save_trade_lesson(lesson_text: str) -> None:
    history = load_trade_history()
    now_jkt = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M WIB")
    history.append({"timestamp": now_jkt, "lesson": lesson_text})
    atomic_write_json(HISTORY_FILE, history[-5:])

# =====================================================================
# 3. PENGAMBILAN DATA CANDLE TERTUTUP (ANTI LOOK-AHEAD BIAS)
# =====================================================================
def get_top_volume_coins(limit: int = 15) -> list[str]:
    url = "https://api.bitget.com/api/v2/mix/market/tickers"
    params = {"productType": "USDT-FUTURES"}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("code") == "00000" and "data" in res:
            sorted_coins = sorted(res["data"], key=lambda x: float(x.get("usdtVolume", 0)), reverse=True)
            coins = [
                item["symbol"] for item in sorted_coins
                if item["symbol"].endswith("USDT") and item["symbol"] not in EXCLUDED_COINS
            ][:limit]
            print(f"✅ Berhasil memuat {len(coins)} koin volume tertinggi dari Bitget.")
            return coins
    except Exception as e:
        print(f"⚠️ Gagal mengambil daftar ticker Bitget: {e}")
    return ["XRPUSDT", "ADAUSDT", "DOGEUSDT", "TRXUSDT", "LINKUSDT", "AVAXUSDT", "NEARUSDT"]

def get_closed_candles(symbol: str, granularity: str, limit: int = 120) -> pd.DataFrame | None:
    url = "https://api.bitget.com/api/v2/mix/market/candles"
    params = {"symbol": symbol, "productType": "USDT-FUTURES", "granularity": granularity, "limit": limit}
    try:
        now_ms = int(time.time() * 1000)
        gran_ms = GRANULARITY_MS[granularity]
        res = requests.get(url, params=params, timeout=10).json()
        if res.get("code") == "00000" and "data" in res and res["data"]:
            df = pd.DataFrame(res["data"], columns=["ts", "open", "high", "low", "close", "volume", "quote_vol"])
            df["ts"] = df["ts"].astype(np.int64)
            for col in ["open", "high", "low", "close", "volume"]:
                df[col] = df[col].astype(float)
            df = df.sort_values("ts", ascending=True).reset_index(drop=True)
            df["close_ts"] = df["ts"] + gran_ms
            df = df[df["close_ts"] <= now_ms].reset_index(drop=True)
            return df
    except Exception as e:
        print(f"❌ Error mengambil candle {symbol} ({granularity}): {e}")
    return None

# =====================================================================
# 4. INDIKATOR, FILTER STRUKTUR & KALKULATOR PNL DETERMINISTIK
# =====================================================================
def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
    df["ema50"] = df["close"].ewm(span=50, adjust=False).mean()

    delta = df["close"].diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df["rsi"] = 100.0 - (100.0 / (1.0 + rs))

    ema12 = df["close"].ewm(span=12, adjust=False).mean()
    ema26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema12 - ema26
    df["signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]

    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()

    df["vol_sma20"] = df["volume"].rolling(window=20).mean()
    return df

def evaluate_snapshot_at_cutoff(df_4h: pd.DataFrame, df_1h: pd.DataFrame, df_15m_slice: pd.DataFrame) -> dict | None:
    if len(df_15m_slice) < 40:
        return None
    cutoff_ts = int(df_15m_slice["close_ts"].iloc[-1])

    sub_4h = df_4h[df_4h["close_ts"] <= cutoff_ts]
    sub_1h = df_1h[df_1h["close_ts"] <= cutoff_ts]
    if len(sub_4h) < 55 or len(sub_1h) < 55:
        return None

    ind_4h = compute_indicators(sub_4h)
    ind_1h = compute_indicators(sub_1h)
    ind_15m = compute_indicators(df_15m_slice)

    c4 = ind_4h.iloc[-1]
    macro_trend = "BULLISH" if (c4["close"] > c4["ema20"] > c4["ema50"]) else (
                  "BEARISH" if (c4["close"] < c4["ema20"] < c4["ema50"]) else "SIDEWAYS")

    c1 = ind_1h.iloc[-1]
    med_trend = "BULLISH" if (c1["close"] > c1["ema20"] > c1["ema50"]) else (
                "BEARISH" if (c1["close"] < c1["ema20"] < c1["ema50"]) else "SIDEWAYS")

    last = ind_15m.iloc[-1]
    prev = ind_15m.iloc[-2]
    recent_3 = ind_15m.tail(3)
    recent_12 = ind_15m.tail(12)
    prior_window = ind_15m.iloc[-25:-5]

    prior_high = float(prior_window["high"].max())
    prior_low = float(prior_window["low"].min())
    atr14 = float(last["atr"])
    candle_range = max(float(last["high"] - last["low"]), 1e-10)
    body_ratio = abs(float(last["close"] - last["open"])) / candle_range

    vol_ratio = float(last["volume"] / (last["vol_sma20"] + 1e-10))
    volume_status = "VALID_VOL" if vol_ratio >= 1.15 else "LOW_VOLUME_FAKE"

    trigger_type = "NONE"
    sweep_level = None

    if macro_trend == "BULLISH" and med_trend == "BULLISH":
        if last["close"] > prior_high and last["close"] > last["open"] and body_ratio >= 0.50:
            trigger_type = "BULLISH_CHOCH_BREAKOUT"
            sweep_level = float(recent_12["low"].min())
        else:
            touched_support = bool((recent_3["low"] <= (recent_3["ema20"] + 0.25 * atr14)).any())
            held_structure = bool((recent_3["low"] >= (recent_3["ema50"] - 0.35 * atr14)).all())
            reclaimed_ema20 = bool(last["close"] > last["ema20"] and last["close"] > last["open"] and body_ratio >= 0.40)
            if touched_support and held_structure and reclaimed_ema20:
                trigger_type = "BULLISH_PRO_TREND_RETEST"
                sweep_level = float(recent_3["low"].min())

    elif macro_trend == "BEARISH" and med_trend == "BEARISH":
        if last["close"] < prior_low and last["close"] < last["open"] and body_ratio >= 0.50:
            trigger_type = "BEARISH_CHOCH_BREAKOUT"
            sweep_level = float(recent_12["high"].max())
        else:
            touched_resistance = bool((recent_3["high"] >= (recent_3["ema20"] - 0.25 * atr14)).any())
            held_structure = bool((recent_3["high"] <= (recent_3["ema50"] + 0.35 * atr14)).all())
            rejected_ema20 = bool(last["close"] < last["ema20"] and last["close"] < last["open"] and body_ratio >= 0.40)
            if touched_resistance and held_structure and rejected_ema20:
                trigger_type = "BEARISH_PRO_TREND_RETEST"
                sweep_level = float(recent_3["high"].max())

    return {
        "close_ts": cutoff_ts,
        "price": float(last["close"]),
        "atr_15m": round(atr14, 6),
        "macro_trend_4h": macro_trend,
        "medium_trend_1h": med_trend,
        "micro_trigger_type": trigger_type,
        "micro_sweep_protection_level": sweep_level,
        "micro_rsi_15m": round(float(last["rsi"]), 2),
        "micro_macd_hist_15m": round(float(last["hist"]), 6),
        "micro_macd_hist_rising": bool(last["hist"] > prev["hist"]),
        "micro_volume_ratio": round(vol_ratio, 2),
        "micro_volume_status": volume_status,
    }

def build_deterministic_trade_plan(coin: str, snap: dict, equity_usdt: float) -> dict | None:
    if snap["micro_volume_status"] != "VALID_VOL" or snap["micro_trigger_type"] == "NONE":
        return None

    trig = snap["micro_trigger_type"]
    rsi = snap["micro_rsi_15m"]
    hist = snap["micro_macd_hist_15m"]
    hist_rising = snap["micro_macd_hist_rising"]
    entry = snap["price"]
    atr = snap["atr_15m"]
    sweep = snap["micro_sweep_protection_level"]

    if trig.startswith("BULLISH"):
        if not (45.0 <= rsi <= 66.0 and hist > 0 and hist_rising):
            return None
        bias = "LONG"
        sl = float(sweep - (0.25 * atr))
        risk_dist = entry - sl
    elif trig.startswith("BEARISH"):
        if not (34.0 <= rsi <= 55.0 and hist < 0 and not hist_rising):
            return None
        bias = "SHORT"
        sl = float(sweep + (0.25 * atr))
        risk_dist = sl - entry
    else:
        return None

    if risk_dist <= 0:
        return None

    sl_pct = risk_dist / entry
    if not (0.0035 <= sl_pct <= 0.0120):
        return None

    min_margin = round(equity_usdt * MIN_MARGIN_RATIO, 2)
    max_margin_cap = round((equity_usdt * MAX_MARGIN_RATIO_STRICT) - 0.01, 2)
    margin = round(equity_usdt * TARGET_MARGIN_RATIO, 2)
    margin = max(min_margin, min(margin, max_margin_cap))

    tp2_pct = 3.0 * sl_pct
    required_lev = int(np.ceil(0.20 / max(tp2_pct - ROUND_TRIP_COST_RATE, 1e-6)))
    leverage = int(np.clip(required_lev, 10, 20))

    max_allowed_loss_usdt = equity_usdt * MAX_STOP_LOSS_EQUITY_RISK
    while leverage >= 10:
        notional = margin * leverage
        est_sl_loss = notional * (sl_pct + ROUND_TRIP_COST_RATE)
        if est_sl_loss <= max_allowed_loss_usdt:
            break
        leverage -= 1

    if leverage < 10:
        return None

    notional = round(margin * leverage, 2)
    if bias == "LONG":
        tp1 = entry + (1.5 * risk_dist)
        tp2 = entry + (3.0 * risk_dist)
        tp3 = entry + (5.0 * risk_dist)
    else:
        tp1 = entry - (1.5 * risk_dist)
        tp2 = entry - (3.0 * risk_dist)
        tp3 = entry - (5.0 * risk_dist)

    est_cost_usdt = notional * ROUND_TRIP_COST_RATE
    net_loss_sl = (notional * sl_pct) + est_cost_usdt
    net_gain_tp1 = (notional * (1.5 * sl_pct)) - est_cost_usdt
    net_gain_tp2 = (notional * (3.0 * sl_pct)) - est_cost_usdt
    net_gain_tp3 = (notional * (5.0 * sl_pct)) - est_cost_usdt
    blended_net_gain = (0.40 * net_gain_tp1) + (0.40 * net_gain_tp2) + (0.20 * net_gain_tp3)

    # Kalkulasi khusus simulasi jika memakai tepat 2.00 USDT
    notional_2usd = 2.00 * leverage
    cost_2usd = notional_2usd * ROUND_TRIP_COST_RATE
    loss_2usd = (notional_2usd * sl_pct) + cost_2usd
    gain_tp1_2usd = (notional_2usd * (1.5 * sl_pct)) - cost_2usd
    gain_tp2_2usd = (notional_2usd * (3.0 * sl_pct)) - cost_2usd
    gain_tp3_2usd = (notional_2usd * (5.0 * sl_pct)) - cost_2usd
    blended_2usd = (0.40 * gain_tp1_2usd) + (0.40 * gain_tp2_2usd) + (0.20 * gain_tp3_2usd)

    return {
        "coin": coin,
        "bias": bias,
        "trigger_type": trig,
        "entry": round(entry, 6),
        "sl": round(sl, 6),
        "tp1": round(tp1, 6),
        "tp2": round(tp2, 6),
        "tp3": round(tp3, 6),
        "margin_usdt": margin,
        "leverage": leverage,
        "notional_usdt": notional,
        "sl_pct": round(sl_pct * 100, 2),
        "est_cost_usdt": round(est_cost_usdt, 3),
        "net_loss_sl_usdt": round(net_loss_sl, 2),
        "net_gain_tp1_usdt": round(net_gain_tp1, 2),
        "net_gain_tp2_usdt": round(net_gain_tp2, 2),
        "net_gain_tp3_usdt": round(net_gain_tp3, 2),
        "blended_net_gain_usdt": round(blended_net_gain, 2),
        "loss_2usd": round(loss_2usd, 2),
        "gain_tp1_2usd": round(gain_tp1_2usd, 2),
        "gain_tp2_2usd": round(gain_tp2_2usd, 2),
        "gain_tp3_2usd": round(gain_tp3_2usd, 2),
        "blended_2usd": round(blended_2usd, 2),
        "volume_ratio": snap["micro_volume_ratio"],
        "rsi_15m": snap["micro_rsi_15m"],
        "macd_hist_15m": snap["micro_macd_hist_15m"],
        "close_ts": snap["close_ts"]
    }

def format_pnl_appendix(plan: dict, equity: float, daily_symbols: list[str]) -> str:
    m = plan["margin_usdt"]
    lev = plan["leverage"]
    return (
        f"\n\n📊 SIMULASI GAIN & LOSS (JIKA PAKAI MARGIN 2.00 USDT | LEV {lev}x)\n"
        f"• Nilai Posisi (Notional): {2.0 * lev:.2f} USDT\n"
        f"• Jika Kena SL: -{plan['loss_2usd']:.2f} USDT (Sisa Saldo: {equity - plan['loss_2usd']:.2f} USDT)\n"
        f"• Jika Kena TP1 (1.5R): +{plan['gain_tp1_2usd']:.2f} USDT\n"
        f"• Jika Kena TP2 (3.0R): +{plan['gain_tp2_2usd']:.2f} USDT\n"
        f"• Jika Kena TP3 (5.0R): +{plan['gain_tp3_2usd']:.2f} USDT\n"
        f"• Profit Cicil (40% TP1 + 40% TP2 + 20% TP3): +{plan['blended_2usd']:.2f} USDT\n"
        f"----------------------------------------\n"
        f"🛡️ REKOMENDASI AUDIT (MARGIN 25% EKUITAS = {m:.2f} USDT | LEV {lev}x)\n"
        f"• Kuota Koin Hari Ini: {len(daily_symbols)}/{MAX_UNIQUE_COINS_PER_DAY} ({', '.join(daily_symbols)})\n"
        f"• Risiko Bersih SL: -{plan['net_loss_sl_usdt']:.2f} USDT\n"
        f"• Target Bersih Cicil 3-Tier: +{plan['blended_net_gain_usdt']:.2f} USDT"
    )

# =====================================================================
# 5. INTEGRASI TELEGRAM & VERIFIKASI MODEL AKTIF GOOGLE AI STUDIO
# =====================================================================
def send_telegram_alert(message: str) -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID kosong! Pastikan sudah diisi di Settings -> Secrets -> Actions.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        res = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        print(f"📩 Telegram API Response: {res.status_code}")
    except Exception as e:
        print(f"❌ Telegram Network Error: {e}")

def get_verified_ai_studio_models(client: genai.Client) -> list[str]:
    """
    Mengambil daftar model resmi yang benar-benar ada di akun Google AI Studio pengguna,
    lalu mengurutkannya berdasarkan prioritas: 4 -> 3.8 -> 3.5 -> 3.1 Pro -> Fallback.
    """
    excluded_keywords = (
        "embedding", "imagen", "veo", "tts", "audio", "image",
        "live", "aqa", "learnlm", "robotics", "computer-use"
    )
    valid_models = []
    try:
        for m in client.models.list():
            raw_name = getattr(m, "name", "")
            short_name = raw_name.replace("models/", "")
            low_name = short_name.lower()

            if "gemini" not in low_name:
                continue
            if any(k in low_name for k in excluded_keywords):
                continue

            supported = getattr(m, "supported_actions", None)
            if supported and "generateContent" not in supported:
                continue

            valid_models.append(short_name)
    except Exception as e:
        print(f"⚠️ Gagal mengambil daftar model dinamis dari AI Studio: {e}")

    if not valid_models:
        # Fallback jika list() gagal karena pembatasan jaringan sementara
        return [
            "gemini-4.0-flash", "gemini-3.8-flash", "gemini-3.5-flash",
            "gemini-3.1-pro-preview", "gemini-3.1-pro", "gemini-3-flash-preview",
            "gemini-2.5-flash"
        ]

    print(f"📋 Daftar Model Teks Resmi di AI Studio Anda: {valid_models}")

    ordered_models = []
    priority_tiers = [
        "gemini-4",
        "gemini-3.8",
        "gemini-3.5",
        "gemini-3.1-pro",
        "gemini-3.1",
        "gemini-3",
        "gemini-2.5"
    ]

    for tier in priority_tiers:
        tier_matches = [m for m in valid_models if m.lower().startswith(tier)]
        # Utamakan model non-lite dan nama paling ringkas/stabil di tiap tier
        tier_matches.sort(key=lambda x: (1 if "lite" in x.lower() else 0, len(x)))
        for match in tier_matches:
            if match not in ordered_models:
                ordered_models.append(match)

    # Tambahkan sisa model valid lainnya sebagai cadangan terakhir
    for m in valid_models:
        if m not in ordered_models:
            ordered_models.append(m)

    print(f"🎯 Urutan Prioritas Eksekusi Model: {ordered_models[:6]}")
    return ordered_models

def call_gemini(prompt: str) -> str | None:
    if not GEMINI_API_KEY:
        print("⚠️ GEMINI_API_KEY kosong! Pastikan sudah diisi di Settings -> Secrets -> Actions.")
        return None

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        print(f"❌ Gagal menginisialisasi klien Gemini: {e}")
        return None

    candidate_models = get_verified_ai_studio_models(client)

    for model_name in candidate_models:
        for _ in range(2):
            try:
                print(f"🤖 Menghubungi Model AI Studio: {model_name}...")
                response = client.models.generate_content(model=model_name, contents=prompt)
                if response and response.text:
                    print(f"✅ Respons diterima dari {model_name}")
                    return response.text.strip()
            except Exception as e:
                print(f"⚠️ Model {model_name} melewati giliran: {e}")
                time.sleep(2)
    return None

# =====================================================================
# 6. ORKESTRATOR UTAMA
# =====================================================================
def execute_serverless_quant_analysis() -> None:
    print("🔧 Memeriksa konfigurasi Environment & File State...")
    print(f"🔑 Status GEMINI_API_KEY: {'TERDETEKSI' if GEMINI_API_KEY else 'KOSONG'}")
    print(f"🔑 Status TELEGRAM_BOT_TOKEN: {'TERDETEKSI' if TELEGRAM_BOT_TOKEN else 'KOSONG'}")
    print(f"🔑 Status TELEGRAM_CHAT_ID: {'TERDETEKSI' if TELEGRAM_CHAT_ID else 'KOSONG'}")

    ensure_persistence_files()
    state = load_trade_state()
    equity = float(state.get("equity_usdt", DEFAULT_EQUITY_USDT))
    history_lessons = load_trade_history()
    now_jkt_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M WIB")

    # MODE A: MEMANTAU POSISI AKTIF
    if state["active"]:
        coin = state["coin"]
        print(f"🔒 MODE MANAGING: Memantau posisi aktif {coin}...")
        df_15m = get_closed_candles(coin, "15m", limit=25)
        if df_15m is None or df_15m.empty:
            return

        last_ts = int(state.get("last_checked_candle_ts", 0))
        new_candles = df_15m[df_15m["close_ts"] > last_ts].copy()
        if new_candles.empty:
            current_price = float(df_15m["close"].iloc[-1])
            send_telegram_alert(
                f"⏱️ MONITORING UPDATE ({now_jkt_str})\n"
                f"Koin: {coin} ({state['bias']} {state['leverage']}x)\n"
                f"Harga Terakhir: {current_price} | Entry: {state['entry']} | SL: {state['sl']}\n"
                f"Status: Posisi masih berjalan aman."
            )
            return

        bias = state["bias"]
        entry = float(state["entry"])
        tp1 = float(state["tp1"])
        tp2 = float(state["tp2"])
        tp3 = float(state["tp3"])
        notional = float(state.get("notional_usdt", 30.0))

        sl_triggered = False
        tp3_triggered = False
        exit_price = float(new_candles["close"].iloc[-1])

        for _, row in new_candles.iterrows():
            c_high = float(row["high"])
            c_low = float(row["low"])
            current_sl = float(state["sl"])
            state["last_checked_candle_ts"] = int(row["close_ts"])

            if bias == "LONG":
                if c_low <= current_sl:
                    sl_triggered = True
                    exit_price = current_sl
                    break
                if c_high >= tp1 and not state["tp1_hit"]:
                    state["tp1_hit"] = True
                    state["sl"] = entry
                if c_high >= tp2 and not state["tp2_hit"]:
                    state["tp2_hit"] = True
                if c_high >= tp3:
                    tp3_triggered = True
                    exit_price = tp3
                    break
            elif bias == "SHORT":
                if c_high >= current_sl:
                    sl_triggered = True
                    exit_price = current_sl
                    break
                if c_low <= tp1 and not state["tp1_hit"]:
                    state["tp1_hit"] = True
                    state["sl"] = entry
                if c_low <= tp2 and not state["tp2_hit"]:
                    state["tp2_hit"] = True
                if c_low <= tp3:
                    tp3_triggered = True
                    exit_price = tp3
                    break

        if sl_triggered:
            if state["tp1_hit"]:
                tp1_part_gain = 0.40 * ((notional * (abs(tp1 - entry) / entry)) - (notional * ROUND_TRIP_COST_RATE))
                state["equity_usdt"] = round(equity + tp1_part_gain, 2)
                send_telegram_alert(
                    f"🛡️ SMC BREAK-EVEN EXIT\nKoin: {coin} ({bias})\nStatus: BE HIT pada {exit_price}\n"
                    f"Profit TP1 (40%) Tersimpan: +{tp1_part_gain:.2f} USDT\n"
                    f"Estimasi Saldo: {state['equity_usdt']:.2f} USDT"
                )
            else:
                sl_loss = (notional * (abs(entry - exit_price) / entry)) + (notional * ROUND_TRIP_COST_RATE)
                state["equity_usdt"] = round(max(0.0, equity - sl_loss), 2)
                prompt = (
                    f"Posisi {coin} ({bias}) terkena Stop Loss di {exit_price} (Entry: {entry}). "
                    f"Kerugian: -{sl_loss:.2f} USDT. Klasifikasikan KESALAHAN_STRUKTUR atau NOISE_VOLATILITAS. "
                    f"Akhiri dengan 1 kalimat setelah kata 'LESSON:'. Tanpa bold."
                )
                ai_resp = call_gemini(prompt) or f"SL Hit pada {exit_price}. LESSON: Perketat validasi volume."
                if "LESSON:" in ai_resp:
                    save_trade_lesson(ai_resp.split("LESSON:")[1].strip())
                send_telegram_alert(
                    f"🚨 SMC STOP LOSS CLOSED\nKoin: {coin} ({bias})\nExit: {exit_price} (-{sl_loss:.2f} USDT)\n"
                    f"Estimasi Saldo: {state['equity_usdt']:.2f} USDT\nEvaluasi: {ai_resp}"
                )

            state.update({
                "active": False, "coin": None, "bias": None, "entry": None,
                "sl": None, "original_sl": None, "tp1": None, "tp2": None, "tp3": None,
                "margin_usdt": None, "leverage": None, "notional_usdt": None,
                "tp1_hit": False, "tp2_hit": False, "last_checked_candle_ts": 0
            })
            save_trade_state(state)
            return

        if tp3_triggered:
            sl_pct = abs(entry - float(state["original_sl"])) / entry
            cost = notional * ROUND_TRIP_COST_RATE
            blended_gain = (0.40 * (notional * 1.5 * sl_pct - cost)) + (0.40 * (notional * 3.0 * sl_pct - cost)) + (0.20 * (notional * 5.0 * sl_pct - cost))
            state["equity_usdt"] = round(equity + blended_gain, 2)
            send_telegram_alert(
                f"🎯 SMC FULL TAKE PROFIT (TP3) HIT!\nKoin: {coin} ({bias})\nExit TP3: {exit_price}\n"
                f"Total Profit Bersih: +{blended_gain:.2f} USDT\n"
                f"Estimasi Saldo: {state['equity_usdt']:.2f} USDT"
            )
            state.update({
                "active": False, "coin": None, "bias": None, "entry": None,
                "sl": None, "original_sl": None, "tp1": None, "tp2": None, "tp3": None,
                "margin_usdt": None, "leverage": None, "notional_usdt": None,
                "tp1_hit": False, "tp2_hit": False, "last_checked_candle_ts": 0
            })
            save_trade_state(state)
            return

        save_trade_state(state)
        send_telegram_alert(
            f"⏱️ MONITORING POSISI ({now_jkt_str})\n"
            f"Koin: {coin} ({bias} {state['leverage']}x)\n"
            f"Harga Penutupan 15m: {exit_price} | SL Aktif: {state['sl']}\n"
            f"TP1 Hit: {state['tp1_hit']} | TP2 Hit: {state['tp2_hit']}"
        )
        return

    # MODE B: SCANNING KANDIDAT BARU
    print(f"🕵️ MODE SCANNING ({now_jkt_str}) | Simbol Hari Ini: {state['daily_tracker']['symbols']}")
    monitored_coins = get_top_volume_coins(limit=15)
    valid_candidates = []
    aligned_trends = []

    for coin in monitored_coins:
        if not can_scan_symbol(coin, state):
            continue

        df_4h = get_closed_candles(coin, "4H", limit=100)
        df_1h = get_closed_candles(coin, "1H", limit=100)
        df_15m = get_closed_candles(coin, "15m", limit=70)
        if df_4h is None or df_1h is None or df_15m is None or len(df_15m) < 50:
            continue

        trajectory = []
        for offset in [-3, -2, -1, 0]:
            sliced_15m = df_15m.iloc[:len(df_15m) + offset].reset_index(drop=True)
            snap = evaluate_snapshot_at_cutoff(df_4h, df_1h, sliced_15m)
            if snap:
                trajectory.append(snap)

        if len(trajectory) == 4:
            latest_snap = trajectory[-1]
            if latest_snap["macro_trend_4h"] == latest_snap["medium_trend_1h"] and latest_snap["macro_trend_4h"] != "SIDEWAYS":
                aligned_trends.append(f"{coin} ({latest_snap['macro_trend_4h']})")
            trade_plan = build_deterministic_trade_plan(coin, latest_snap, equity)
            if trade_plan:
                trade_plan["trajectory"] = trajectory
                valid_candidates.append(trade_plan)

    if not valid_candidates:
        save_trade_state(state)
        aligned_str = ", ".join(aligned_trends[:6]) if aligned_trends else "Tidak ada"
        send_telegram_alert(
            f"ℹ️ SCAN REPORT ({now_jkt_str})\n"
            f"Status: NO_TRADE (Belum ada setup 15m yang lolos filter ketat Volume + ATR + RSI/MACD)\n"
            f"Koin Tren 4H/1H Searah: {aligned_str}\n"
            f"Kuota Koin Hari Ini: {len(state['daily_tracker']['symbols'])}/{MAX_UNIQUE_COINS_PER_DAY} | Saldo: {equity:.2f} USDT"
        )
        return

    valid_candidates.sort(key=lambda x: x["volume_ratio"], reverse=True)
    best_plan = valid_candidates[0]

    lessons_str = "\n".join([f"- {item['lesson']}" for item in history_lessons]) if history_lessons else "Belum ada catatan."
    prompt = (
        f"Anda adalah Senior Quantitative Risk Auditor. Evaluasi kandidat sinyal berikut:\n"
        f"Koin: {best_plan['coin']} | Bias: {best_plan['bias']} | Trigger 15m: {best_plan['trigger_type']}\n"
        f"Entry: {best_plan['entry']} | SL: {best_plan['sl']} | TP1: {best_plan['tp1']} | TP2: {best_plan['tp2']} | TP3: {best_plan['tp3']}\n"
        f"Trajektori 4 Candle 15m Terakhir: {json.dumps(best_plan['trajectory'])}\n"
        f"Hipotesis Historis: {lessons_str}\n\n"
        f"TUGAS:\n"
        f"1. Jangan ubah angka harga atau leverage.\n"
        f"2. Awali baris pertama dengan 'DECISION: APPROVE' atau 'DECISION: REJECT'.\n"
        f"3. Berikan narasi singkat tanpa format bold."
    )

    ai_review = call_gemini(prompt) or "DECISION: APPROVE\nNarasi: Konfirmasi struktur dan volume valid."
    if "DECISION: REJECT" in ai_review.upper():
        save_trade_state(state)
        send_telegram_alert(
            f"🛑 KANDIDAT DITOLAK AI ({now_jkt_str})\n"
            f"Koin: {best_plan['coin']} ({best_plan['bias']})\n"
            f"Alasan: {ai_review}"
        )
        return

    if not register_accepted_signal(best_plan["coin"], state):
        save_trade_state(state)
        return

    state.update({
        "active": True,
        "coin": best_plan["coin"],
        "bias": best_plan["bias"],
        "entry": best_plan["entry"],
        "sl": best_plan["sl"],
        "original_sl": best_plan["sl"],
        "tp1": best_plan["tp1"],
        "tp2": best_plan["tp2"],
        "tp3": best_plan["tp3"],
        "margin_usdt": best_plan["margin_usdt"],
        "leverage": best_plan["leverage"],
        "notional_usdt": best_plan["notional_usdt"],
        "tp1_hit": False,
        "tp2_hit": False,
        "last_checked_candle_ts": best_plan["close_ts"]
    })
    save_trade_state(state)

    clean_narrative = ai_review.replace("DECISION: APPROVE", "").strip()
    signal_msg = (
        f"💎 AUDITED QUANT SIGNAL ({now_jkt_str})\n"
        f"Koin: {best_plan['coin']} ({best_plan['bias']})\n"
        f"Trigger 15m: {best_plan['trigger_type']}\n"
        f"Volume Ratio: {best_plan['volume_ratio']}x SMA20 | RSI(14): {best_plan['rsi_15m']}\n"
        f"----------------------------------------\n"
        f"🎯 Level Eksekusi:\n"
        f"• Entry: {best_plan['entry']}\n"
        f"• Stop Loss (SL): {best_plan['sl']} (Jarak Spot {best_plan['sl_pct']:.2f}%)\n"
        f"• TP1 (1.5R - Jual 40% & Geser SL ke BE): {best_plan['tp1']}\n"
        f"• TP2 (3.0R - Jual 40%): {best_plan['tp2']}\n"
        f"• TP3 (5.0R - Jual 20%): {best_plan['tp3']}\n"
        f"----------------------------------------\n"
        f"🧠 Analisis AI: {clean_narrative}"
        + format_pnl_appendix(best_plan, equity, state["daily_tracker"]["symbols"])
    )
    send_telegram_alert(signal_msg)

if __name__ == "__main__":
    execute_serverless_quant_analysis()
