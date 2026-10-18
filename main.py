import os
import io
import math
import wave
import struct
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
MAX_PENDING_CANDLES = 4           # Auto-Cancel jika dalam 4 candle (1 jam) limit belum kejemput

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
# 2. GENERATOR EFEK SUARA & SISTEM NOTIFIKASI PINTAR TELEGRAM
# =====================================================================
SOUND_PROFILES = {
    "SIGNAL": [(523.25, 0.12), (659.25, 0.12), (783.99, 0.22)],                 # Nada Naik Cepat (C5 -> E5 -> G5)
    "TP": [(523.25, 0.10), (659.25, 0.10), (783.99, 0.10), (1046.50, 0.30)],    # Fanfare Kemenangan (C5 -> E5 -> G5 -> C6)
    "SL": [(440.00, 0.18), (329.63, 0.18), (220.00, 0.35)],                     # Nada Turun Berat (A4 -> E4 -> A3)
    "CANCEL": [(698.46, 0.12), (0.0, 0.06), (698.46, 0.20)],                    # Double Beep Peringatan (F5 -> F5)
}

def generate_tone_wav_bytes(pattern_name: str) -> io.BytesIO | None:
    notes = SOUND_PROFILES.get(pattern_name)
    if not notes:
        return None

    sample_rate = 22050
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)

        frames = bytearray()
        for freq, duration in notes:
            num_samples = int(sample_rate * duration)
            for i in range(num_samples):
                if freq <= 0:
                    sample = 0
                else:
                    fade_len = max(1, int(0.015 * sample_rate))
                    env = 1.0
                    if i < fade_len:
                        env = i / fade_len
                    elif i > num_samples - fade_len:
                        env = (num_samples - i) / fade_len
                    val = math.sin(2.0 * math.pi * freq * (i / sample_rate))
                    sample = int(val * env * 24000)
                frames.extend(struct.pack("<h", sample))
        wav_file.writeframes(bytes(frames))

    buf.seek(0)
    buf.name = f"alert_{pattern_name.lower()}.wav"
    return buf

def send_telegram_alert(message: str, alert_type: str = "SILENT") -> None:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID kosong! Pastikan sudah diisi di Settings -> Secrets -> Actions.")
        return

    is_silent = alert_type in ("SILENT", "NO_TRADE", "STAY")
    base_url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}"

    if not is_silent and alert_type in SOUND_PROFILES:
        try:
            audio_buf = generate_tone_wav_bytes(alert_type)
            if audio_buf:
                caption_map = {
                    "SIGNAL": "🔔 AUDIO CUE: Sinyal Koin Baru Ditemukan!",
                    "TP": "🏆 AUDIO CUE: Target Take Profit Tercapai!",
                    "SL": "🚨 AUDIO CUE: Posisi Terkena Stop Loss!",
                    "CANCEL": "⚠️ AUDIO CUE: Sinyal / Posisi Dibatalkan!"
                }
                requests.post(
                    f"{base_url}/sendVoice",
                    data={
                        "chat_id": TELEGRAM_CHAT_ID,
                        "caption": caption_map.get(alert_type, "🔔 Notifikasi Penting"),
                        "disable_notification": False
                    },
                    files={"voice": (audio_buf.name, audio_buf, "audio/wav")},
                    timeout=10
                )
        except Exception as e:
            print(f"⚠️ Gagal mengirim audio cue: {e}")

    try:
        res = requests.post(
            f"{base_url}/sendMessage",
            json={
                "chat_id": TELEGRAM_CHAT_ID,
                "text": message,
                "disable_notification": is_silent
            },
            timeout=10
        )
        mode_label = "SILENT MODE 🔕" if is_silent else f"SOUND ALERT ({alert_type}) 🔔"
        print(f"📩 Telegram API Response: {res.status_code} [{mode_label}]")
    except Exception as e:
        print(f"❌ Telegram Network Error: {e}")

# =====================================================================
# 3. MANAJEMEN STATE ATOMIK & KUOTA 3 KOIN PER HARI (ASIA/JAKARTA)
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

def get_empty_position_fields() -> dict:
    return {
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
        "entry_filled": False,
        "pending_candles_count": 0,
        "tp1_hit": False,
        "tp2_hit": False,
        "last_checked_candle_ts": 0,
    }

def load_trade_state() -> dict:
    today = get_trading_day()
    default_state = {
        "equity_usdt": DEFAULT_EQUITY_USDT,
        **get_empty_position_fields(),
        "last_telegram_update_id": 0,
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

def refund_cancelled_symbol_slot(symbol: str, state: dict) -> None:
    tracker = state.get("daily_tracker", {})
    symbols = set(tracker.get("symbols", []))
    if symbol in symbols:
        symbols.remove(symbol)
        tracker["symbols"] = sorted(symbols)

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
# 4. PENGAMBILAN DATA BITGET & PEMBACA PERINTAH MANUAL TELEGRAM
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

def get_latest_market_price(symbol: str, fallback_price: float) -> float:
    """Mengambil harga pasar terkini dari Bitget saat pengguna melakukan /cancel manual."""
    url = "https://api.bitget.com/api/v2/mix/market/ticker"
    params = {"symbol": symbol, "productType": "USDT-FUTURES"}
    try:
        res = requests.get(url, params=params, timeout=8).json()
        if res.get("code") == "00000" and res.get("data"):
            data = res["data"]
            row = data[0] if isinstance(data, list) else data
            return float(row.get("lastPr", fallback_price))
    except Exception:
        pass
    return fallback_price

def process_telegram_commands(state: dict) -> dict:
    """
    Membaca pesan perintah dari pengguna di Telegram:
    - /cancel [harga_opsional] : Membatalkan posisi/order aktif secara manual
    - /filled                  : Mengonfirmasi Limit Order sudah terjemput
    - /balance [angka]         : Mengatur ulang saldo ekuitas USDT
    - /status                  : Mengecek status posisi & kuota koin harian
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return state

    last_update_id = int(state.get("last_telegram_update_id", 0))
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"offset": last_update_id + 1, "timeout": 2}

    try:
        res = requests.get(url, params=params, timeout=8).json()
        if not res.get("ok"):
            return state
        updates = res.get("result", [])
    except Exception as e:
        print(f"⚠️ Gagal memeriksa command Telegram: {e}")
        return state

    if not updates:
        return state

    now_jkt_str = datetime.now(TIMEZONE).strftime("%Y-%m-%d %H:%M WIB")
    state_modified = False

    for upd in updates:
        upd_id = int(upd.get("update_id", 0))
        if upd_id > last_update_id:
            last_update_id = upd_id
            state["last_telegram_update_id"] = last_update_id
            state_modified = True

        msg = upd.get("message") or upd.get("edited_message")
        if not msg:
            continue

        chat_id_str = str(msg.get("chat", {}).get("id", ""))
        if chat_id_str != str(TELEGRAM_CHAT_ID).strip():
            continue

        raw_text = (msg.get("text") or "").strip()
        if not raw_text:
            continue

        parts = raw_text.split()
        cmd = parts[0].lower().split("@")[0]

        # -------------------------------------------------------------
        # COMMAND 1: /cancel atau cancel (Batal Posisi / Pending Order)
        # -------------------------------------------------------------
        if cmd in ("/cancel", "cancel", "/batal", "batal", "/close", "close"):
            if not state.get("active"):
                send_telegram_alert(
                    f"ℹ️ INFO COMMAND ({now_jkt_str})\n"
                    f"Tidak ada posisi atau order aktif untuk dibatalkan. Bot sedang dalam mode Scanning.",
                    alert_type="SILENT"
                )
            else:
                coin = state["coin"]
                bias = state["bias"]
                entry = float(state["entry"])
                notional = float(state.get("notional_usdt", 30.0))
                equity = float(state.get("equity_usdt", DEFAULT_EQUITY_USDT))
                was_filled = bool(state.get("entry_filled", False))

                # Jika masih Pending (belum kejemput), kembalikan kuota koin harian tanpa potong saldo
                if not was_filled and len(parts) == 1:
                    refund_cancelled_symbol_slot(coin, state)
                    state.update(get_empty_position_fields())
                    save_trade_state(state)
                    send_telegram_alert(
                        f"🟡⚠️ [MANUAL CANCEL — ORDER PENDING DIHAPUS] ⚠️🟡\n"
                        f"Waktu: {now_jkt_str}\n"
                        f"Koin: {coin} ({bias})\n"
                        f"Status: Limit Order dibatalkan manual oleh Anda sebelum terjemput.\n"
                        f"Kuota Koin Hari Ini Dikembalikan: {len(state['daily_tracker']['symbols'])}/{MAX_UNIQUE_COINS_PER_DAY} | Saldo Tetap: {equity:.2f} USDT\n"
                        f"Bot sekarang langsung kembali memindai koin lain!",
                        alert_type="CANCEL"
                    )
                else:
                    # Jika posisi sudah berjalan (atau user menyertakan harga exit manual: /cancel 0.543)
                    custom_exit = None
                    if len(parts) > 1:
                        try:
                            custom_exit = float(parts[1].replace(",", "."))
                        except ValueError:
                            custom_exit = None

                    exit_px = custom_exit if custom_exit is not None else get_latest_market_price(coin, entry)
                    price_diff_pct = ((exit_px - entry) / entry) if bias == "LONG" else ((entry - exit_px) / entry)
                    realized_pnl = (notional * price_diff_pct) - (notional * ROUND_TRIP_COST_RATE)
                    state["equity_usdt"] = round(max(0.0, equity + realized_pnl), 2)
                    state.update(get_empty_position_fields())
                    save_trade_state(state)

                    send_telegram_alert(
                        f"🟡⚠️ [MANUAL CANCEL / CLOSE POSISI AKTIF] ⚠️🟡\n"
                        f"Waktu: {now_jkt_str}\n"
                        f"Koin: {coin} ({bias})\n"
                        f"Harga Entry: {entry} | Harga Exit Manual: {exit_px}\n"
                        f"Estimasi PnL Bersih (Setelah Fee): {realized_pnl:+.2f} USDT\n"
                        f"Estimasi Saldo Baru: {state['equity_usdt']:.2f} USDT\n"
                        f"Status: Posisi direset. Bot langsung lanjut memindai peluang baru!",
                        alert_type="CANCEL"
                    )

        # -------------------------------------------------------------
        # COMMAND 2: /filled atau filled (Paksa Tandai Order Terjemput)
        # -------------------------------------------------------------
        elif cmd in ("/filled", "filled", "/masuk", "masuk"):
            if state.get("active"):
                state["entry_filled"] = True
                save_trade_state(state)
                send_telegram_alert(
                    f"✅ KONFIRMASI MANUAL ({now_jkt_str})\n"
                    f"Posisi {state['coin']} ({state['bias']}) telah ditandai AKTIF TERJEMPUT (Filled).",
                    alert_type="SILENT"
                )
            else:
                send_telegram_alert(
                    f"ℹ️ Tidak ada order pending yang sedang menunggu konfirmasi /filled.",
                    alert_type="SILENT"
                )

        # -------------------------------------------------------------
        # COMMAND 3: /balance [nominal] (Sinkronisasi Saldo Manual)
        # -------------------------------------------------------------
        elif cmd in ("/balance", "/saldo", "saldo") and len(parts) > 1:
            try:
                new_bal = float(parts[1].replace(",", "."))
                if new_bal > 0:
                    state["equity_usdt"] = round(new_bal, 2)
                    save_trade_state(state)
                    send_telegram_alert(
                        f"💰 SINKRONISASI SALDO ({now_jkt_str})\n"
                        f"Saldo ekuitas bot berhasil diperbarui menjadi: {state['equity_usdt']:.2f} USDT.",
                        alert_type="SILENT"
                    )
            except ValueError:
                pass

        # -------------------------------------------------------------
        # COMMAND 4: /status atau status (Cek Status Bot)
        # -------------------------------------------------------------
        elif cmd in ("/status", "status"):
            pos_info = (
                f"{state['coin']} ({state['bias']} {state['leverage']}x) | Entry: {state['entry']} | SL: {state['sl']} | Filled: {state['entry_filled']}"
                if state.get("active") else "Tidak ada (Mode Scanning)"
            )
            send_telegram_alert(
                f"📋 STATUS BOT SAAT INI ({now_jkt_str})\n"
                f"• Saldo Akun: {state.get('equity_usdt', DEFAULT_EQUITY_USDT):.2f} USDT\n"
                f"• Posisi Aktif: {pos_info}\n"
                f"• Kuota Koin Hari Ini: {len(state['daily_tracker']['symbols'])}/{MAX_UNIQUE_COINS_PER_DAY} ({', '.join(state['daily_tracker']['symbols']) or 'Kosong'})\n"
                f"• Daftar Perintah: /cancel, /cancel [harga], /filled, /balance [angka], /status",
                alert_type="SILENT"
            )

    if state_modified:
        save_trade_state(state)
    return state

# =====================================================================
# 5. INDIKATOR, FILTER STRUKTUR & DETEKSI AUTO-CANCEL
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

def check_structural_invalidation_cancel(df_1h: pd.DataFrame, df_15m: pd.DataFrame, bias: str) -> tuple[bool, str]:
    if df_1h is None or len(df_1h) < 55 or df_15m is None or len(df_15m) < 30:
        return False, ""

    ind_1h = compute_indicators(df_1h)
    ind_15m = compute_indicators(df_15m)

    c1 = ind_1h.iloc[-1]
    last_15m = ind_15m.iloc[-1]
    prior_10_low = float(ind_15m["low"].iloc[-12:-2].min())
    prior_10_high = float(ind_15m["high"].iloc[-12:-2].max())
    vol_spike = float(last_15m["volume"]) > (1.35 * float(last_15m["vol_sma20"]))

    if bias == "LONG":
        if c1["close"] < c1["ema20"] < c1["ema50"]:
            return True, "Struktur Tren 1H patah dan berbalik menjadi BEARISH (Close < EMA20 < EMA50)."
        if last_15m["close"] < prior_10_low and last_15m["close"] < last_15m["ema50"] and vol_spike:
            return True, "Terjadi Bearish CHOCH 15m dengan lonjakan volume distribusi di bawah EMA50."

    elif bias == "SHORT":
        if c1["close"] > c1["ema20"] > c1["ema50"]:
            return True, "Struktur Tren 1H patah dan berbalik menjadi BULLISH (Close > EMA20 > EMA50)."
        if last_15m["close"] > prior_10_high and last_15m["close"] > last_15m["ema50"] and vol_spike:
            return True, "Terjadi Bullish CHOCH 15m dengan lonjakan volume akumulasi di atas EMA50."

    return False, ""

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

    c1 = ind_
