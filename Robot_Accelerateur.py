#!/usr/bin/env python3
import time
import warnings
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
warnings.filterwarnings("ignore")

CONFIG = {
    "min_beta": 1.5,
    "min_avg_volume": 500000,
    "min_rel_volume": 1.2,
    "min_price": 5.0,
    "stoch_low": 15,
    "stoch_high": 85,
    "lookback_days": 120,
    "sr_lookback": 40,
    "sr_tolerance_pct": 0.018,
    "sleep_between_tickers": 0.35,
}

UNIVERSE = list(dict.fromkeys([
    "NVDA","AMD","MU","INTC","SMCI","MRVL","WDC","STX","ON","VRT",
    "AVGO","LRCX","AMAT","KLAC","QCOM","TXN","ADI","NXPI","MCHP",
    "ANET","CRWD","PANW","FTNT","ZS","DDOG","NET","SNOW","MDB",
    "PLTR","APP","TSLA","COIN","HOOD","SOFI","AFRM","UPST","SHOP",
    "SNAP","ROKU","UBER","ABNB","DKNG","CVNA","CCL","NCLH","RCL",
    "DAL","UAL","AAL","MARA","RIOT","CLSK","HUT","CIFR","WULF",
    "IREN","ENPH","FSLR","PLUG","BE","JOBY","ACHR","RKLB","ASTS",
    "RGTI","IONQ","QBTS","SMR","OPEN","COMP","Z","RKT",
]))

def calculate_indicators(df):
    if len(df) < 60:
        return df
    ema_fast = df["Close"].ewm(span=12, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    low_min = df["Low"].rolling(14).min()
    high_max = df["High"].rolling(14).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(3).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["RSI"] = 100 - (100 / (1 + gain / (loss + 1e-9)))
    df["Avg_Volume_20"] = df["Volume"].rolling(20).mean()
    df["Rel_Volume"] = df["Volume"] / (df["Avg_Volume_20"] + 1e-9)
    recent = df.tail(CONFIG["sr_lookback"])
    df["Resistance"] = recent["High"].max()
    df["Support"] = recent["Low"].min()
    return df

def detect_candlestick(row):
    o, h, l, c = float(row["Open"]), float(row["High"]), float(row["Low"]), float(row["Close"])
    body = abs(c - o)
    full_range = h - l + 1e-9
    upper = h - max(o, c)
    lower = min(o, c) - l
    body_ratio = body / full_range
    if body_ratio > 0.85:
        return ("Marubozu haussier", "LONG", 2) if c > o else ("Marubozu baissier", "SHORT", 2)
    is_doji = body_ratio < 0.12
    if is_doji and lower > body * 2.5 and upper < body * 0.8:
        return ("Doji Dragon / Porteur d'eau", "LONG", 3)
    if is_doji and upper > body * 2.5 and lower < body * 0.8:
        return ("Doji Pierre tombale", "SHORT", 3)
    if lower > body * 2.0 and upper < body * 0.6 and body_ratio < 0.35:
        return ("Marteau", "LONG", 3) if c >= o else ("Le Pendu", "SHORT", 2)
    if upper > body * 2.0 and lower < body * 0.6 and body_ratio < 0.35:
        return ("Marteau inverse", "LONG", 2) if c >= o else ("Etoile filante", "SHORT", 3)
    if is_doji and lower > full_range * 0.55:
        return ("Marteau Doji", "LONG", 3)
    return (None, None, 0)

def detect_chart_patterns(df):
    last = df.iloc[-1]
    price = float(last["Close"])
    support = float(last["Support"])
    resistance = float(last["Resistance"])
    tol = CONFIG["sr_tolerance_pct"]
    near_s = abs(price - support) / price <= tol or price <= support * 1.025
    near_r = abs(price - resistance) / price <= tol or price >= resistance * 0.975
    candle, direction, strength = detect_candlestick(last)
    if near_s and direction == "LONG" and strength >= 2:
        return f"{candle} sur Support"
    if near_r and direction == "SHORT" and strength >= 2:
        return f"{candle} sur Resistance"
    if price > resistance * 1.008 and last["Rel_Volume"] >= 1.2 and last["Close"] > last["Open"]:
        return "Breakout Resistance"
    if price < support * 0.992 and last["Rel_Volume"] >= 1.2 and last["Close"] < last["Open"]:
        return "Breakdown Support"
    if strength >= 3:
        return candle
    return "3 indicateurs verts"

def check_long(row):
    return (row["Rel_Volume"] >= CONFIG["min_rel_volume"] and row["MACD"] > row["MACD_Signal"]
            and CONFIG["stoch_low"] <= row["Stoch_K"] <= CONFIG["stoch_high"] and row["Stoch_K"] >= row["Stoch_D"])

def check_short(row):
    return (row["Rel_Volume"] >= CONFIG["min_rel_volume"] and row["MACD"] < row["MACD_Signal"]
            and CONFIG["stoch_low"] <= row["Stoch_K"] <= CONFIG["stoch_high"] and row["Stoch_K"] <= row["Stoch_D"])

def get_stock_info(ticker):
    try:
        info = yf.Ticker(ticker).info
        beta = info.get("beta")
        avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
        qt = str(info.get("quoteType", "")).upper()
        return {"beta": beta, "avg_volume": avg_vol, "sector": info.get("sector", "N/A"), "is_etf": "ETF" in qt or "ETP" in qt}
    except Exception:
        return {"beta": None, "avg_volume": 0, "sector": "N/A", "is_etf": False}

def scan_ticker(ticker):
    try:
        end = datetime.now()
        start = end - timedelta(days=CONFIG["lookback_days"])
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True, threads=False)
        if df.empty or len(df) < 60:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = calculate_indicators(df).dropna(subset=["MACD", "Stoch_K", "Rel_Volume"])
        if len(df) < 5:
            return None
        last = df.iloc[-1]
        info = get_stock_info(ticker)
        if info["is_etf"] or float(last["Close"]) < CONFIG["min_price"]:
            return None
        if info["beta"] is not None and info["beta"] < CONFIG["min_beta"]:
            return None
        if info["avg_volume"] < CONFIG["min_avg_volume"]:
            return None
        is_long = check_long(last)
        is_short = check_short(last)
        if not (is_long or is_short):
            return None
        return {
            "Ticker": ticker,
            "Direction": "LONG" if is_long else "SHORT",
            "Price": round(float(last["Close"]), 2),
            "Rel_Volume": round(float(last["Rel_Volume"]), 2),
            "Stoch_K": round(float(last["Stoch_K"]), 1),
            "Pattern": detect_chart_patterns(df),
        }
    except Exception:
        return None

def write_html(results, total):
    lines = ["<html><head><meta charset='utf-8'><title>Robot Accelerateur</title></head><body>",
             "<h1>Robot Accelerateur</h1>",
             "<p>%s</p>" % datetime.now().strftime("%Y-%m-%d %H:%M"),
             "<p>Titres scannes : %s | Setups : %s</p>" % (total, len(results))]
    if not results:
        lines.append("<p>Aucun setup aujourd'hui.</p>")
    else:
        lines.append("<ul>")
        for r in results:
            lines.append("<li>%s | %s | %s | RelVol %s | Stoch %s | %s</li>" % (
                r["Ticker"], r["Direction"], r["Price"], r["Rel_Volume"], r["Stoch_K"], r["Pattern"]))
        lines.append("</ul>")
    lines.append("</body></html>")
    open("rapport.html", "w", encoding="utf-8").write("\n".join(lines))

def run_robot():
    print("ROBOT ACCELERATEUR v2.2")
    results = []
    total = len(UNIVERSE)
    for i, ticker in enumerate(UNIVERSE, 1):
        if i % 15 == 0:
            print("Progression : %s/%s..." % (i, total))
        res = scan_ticker(ticker)
        if res:
            results.append(res)
            print("OK %s -> %s | %s | %s" % (res["Ticker"], res["Direction"], res["Price"], res["Pattern"]))
        time.sleep(CONFIG["sleep_between_tickers"])
    print("Scan termine : %s | Setups : %s" % (total, len(results)))
    write_html(results, total)
    pre, close_moves = scan_big_moves()
    write_moves_html(pre, close_moves)
    print("Mouvements pre-market :", len(pre))
    print("Mouvements cloture :", len(close_moves))
    pd.DataFrame(results).to_csv("report_latest.csv", index=False)
    if not results:
        print("Aucun setup aujourd'hui.")



MOVE_THRESHOLD = 3.0

def scan_big_moves():
    pre, close_moves = [], []
    for ticker in UNIVERSE:
        try:
            info = yf.Ticker(ticker).info
            prev = info.get("previousClose") or info.get("regularMarketPreviousClose")
            if not prev:
                continue
            pre_px = info.get("preMarketPrice")
            post_px = info.get("postMarketPrice")
            last_px = info.get("regularMarketPrice") or info.get("currentPrice")
            if pre_px and prev:
                pct = (float(pre_px) - float(prev)) / float(prev) * 100
                if abs(pct) >= MOVE_THRESHOLD:
                    pre.append({"Ticker": ticker, "Sens": "HAUSSE" if pct > 0 else "BAISSE",
                                "Pct": round(pct, 2), "Prix": round(float(pre_px), 2), "Ref": round(float(prev), 2)})
            if last_px and prev:
                pct = (float(last_px) - float(prev)) / float(prev) * 100
                if abs(pct) >= MOVE_THRESHOLD:
                    close_moves.append({"Ticker": ticker, "Sens": "HAUSSE" if pct > 0 else "BAISSE",
                                        "Pct": round(pct, 2), "Prix": round(float(last_px), 2), "Ref": round(float(prev), 2),
                                        "Session": "Cloture"})
            if post_px and prev:
                pct = (float(post_px) - float(prev)) / float(prev) * 100
                if abs(pct) >= MOVE_THRESHOLD:
                    close_moves.append({"Ticker": ticker, "Sens": "HAUSSE" if pct > 0 else "BAISSE",
                                        "Pct": round(pct, 2), "Prix": round(float(post_px), 2), "Ref": round(float(prev), 2),
                                        "Session": "After hours"})
        except Exception:
            pass
        time.sleep(0.15)
    pre.sort(key=lambda x: abs(x["Pct"]), reverse=True)
    close_moves.sort(key=lambda x: abs(x["Pct"]), reverse=True)
    return pre, close_moves

def write_moves_html(pre, close_moves):
    lines = ["<html><head><meta charset='utf-8'><title>Gros mouvements</title></head><body>",
             "<h1>Robot Accelerateur - Gros mouvements</h1>",
             "<p>%s</p>" % datetime.now().strftime("%Y-%m-%d %H:%M"),
             "<p>Seuil : variation d au moins 3%% vs veille</p>",
             "<h2>Avant ouverture (pre-market)</h2>"]
    if not pre:
        lines.append("<p>Aucun gros mouvement pre-market.</p>")
    else:
        lines.append("<ul>")
        for r in pre:
            lines.append("<li>%s | %s | %s%% | Prix %s | Veille %s</li>" % (r["Ticker"], r["Sens"], r["Pct"], r["Prix"], r["Ref"]))
        lines.append("</ul>")
    lines.append("<h2>Cloture / after hours</h2>")
    if not close_moves:
        lines.append("<p>Aucun gros mouvement a la fermeture.</p>")
    else:
        lines.append("<ul>")
        for r in close_moves:
            lines.append("<li>%s | %s | %s%% | %s | Prix %s | Veille %s</li>" % (r["Ticker"], r["Sens"], r["Pct"], r.get("Session",""), r["Prix"], r["Ref"]))
        lines.append("</ul>")
    lines.append("</body></html>")
    open("rapport_mouvements.html", "w", encoding="utf-8").write("\n".join(lines))


if __name__ == "__main__":
    run_robot()
