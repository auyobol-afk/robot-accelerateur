#!/usr/bin/env python3
"""
=============================================================================
 ROBOT ACCÉLÉRATEUR v2.2 - Scanner quotidien US Options
=============================================================================
 Critères STRICTS :
 1. VOLUME       : Relative Volume >= 1.5 + expansion
 2. MACD         : Ligne au-dessus (Long) / en-dessous (Short) du Signal + histogramme en expansion
 3. STOCHASTIQUE : Zone 20-80 + orientation directionnelle
 4. FIGURES + BOUGIES sur Support ou Résistance
    (Marteau, Marteau inversé, Marteau Doji, Étoile filante, Pendu,
     Marubozu, Doji Dragon, Doji Pierre tombale, Doji porteur d'eau)
=============================================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings("ignore")

CONFIG = {
    "min_beta": 1.5,
    "min_avg_volume": 500_000,
    "min_rel_volume": 1.2,
    "min_price": 5.0,
    "stoch_low": 15,
    "stoch_high": 85,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "stoch_k": 14,
    "stoch_d": 3,
    "lookback_days": 120,
    "max_stocks_to_scan": 180,
    "sr_lookback": 40,
    "sr_tolerance_pct": 0.018,
    "output_csv": True,
    "output_console": True,
    "sleep_between_tickers": 0.40,
}

UNIVERSE = [
    "NVDA", "AMD", "MU", "INTC", "SMCI", "MRVL", "WDC", "STX", "ON", "VRT",
    "AVGO", "LRCX", "AMAT", "KLAC", "QCOM", "TXN", "ADI", "NXPI", "MCHP",
    "MPWR", "SWKS", "ANET", "CRWD", "PANW", "FTNT", "ZS", "DDOG", "NET",
    "SNOW", "MDB", "TEAM", "OKTA", "APP", "PLTR",
    "TSLA", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "SHOP", "MELI", "SE",
    "SNAP", "PINS", "ROKU", "SPOT", "UBER", "ABNB", "DKNG", "CVNA",
    "CCL", "NCLH", "RCL", "DAL", "UAL", "AAL", "LUV", "JBLU",
    "MARA", "RIOT", "CLSK", "HUT", "CIFR", "WULF", "IREN", "BTBT",
    "ENPH", "SEDG", "FSLR", "RUN", "PLUG", "BE", "FCEL",
    "JOBY", "ACHR", "RKLB", "ASTS",
    "RGTI", "IONQ", "QBTS", "SMR", "OKLO",
    "OPEN", "COMP", "Z", "ZG", "RKT",
]
UNIVERSE = list(dict.fromkeys(UNIVERSE))

def calculate_indicators(df):
    if len(df) < 60:
        return df
    ema_fast = df["Close"].ewm(span=CONFIG["macd_fast"], adjust=False).mean()
    ema_slow = df["Close"].ewm(span=CONFIG["macd_slow"], adjust=False).mean()
    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=CONFIG["macd_signal"], adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]
    low_min = df["Low"].rolling(CONFIG["stoch_k"]).min()
    high_max = df["High"].rolling(CONFIG["stoch_k"]).max()
    df["Stoch_K"] = 100 * (df["Close"] - low_min) / (high_max - low_min + 1e-9)
    df["Stoch_D"] = df["Stoch_K"].rolling(CONFIG["stoch_d"]).mean()
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["Avg_Volume_20"] = df["Volume"].rolling(20).mean()
    df["Rel_Volume"] = df["Volume"] / (df["Avg_Volume_20"] + 1e-9)
    df["Volume_Expansion"] = df["Volume"] > df["Volume"].shift(1)
    look = CONFIG["sr_lookback"]
    recent = df.tail(look)
    df["Resistance"] = recent["High"].max()
    df["Support"] = recent["Low"].min()
    return df

def detect_candlestick(row):
    o = float(row["Open"])
    h = float(row["High"])
    l = float(row["Low"])
    c = float(row["Close"])
    body = abs(c - o)
    full_range = h - l + 1e-9
    upper_shadow = h - max(o, c)
    lower_shadow = min(o, c) - l
    body_ratio = body / full_range
    if body_ratio > 0.85:
        if c > o:
            return {"candle": "Marubozu haussier", "direction": "LONG", "strength": 2}
        else:
            return {"candle": "Marubozu baissier", "direction": "SHORT", "strength": 2}
    is_doji = body_ratio < 0.12
    if is_doji and lower_shadow > body * 2.5 and upper_shadow < body * 0.8:
        return {"candle": "Doji Dragon / Porteur d'eau", "direction": "LONG", "strength": 3}
    if is_doji and upper_shadow > body * 2.5 and lower_shadow < body * 0.8:
        return {"candle": "Doji Pierre tombale", "direction": "SHORT", "strength": 3}
    if (lower_shadow > body * 2.0 and upper_shadow < body * 0.6 and body_ratio < 0.35):
        return {"candle": "Marteau", "direction": "LONG", "strength": 3}
    if (upper_shadow > body * 2.0 and lower_shadow < body * 0.6 and body_ratio < 0.35):
        return {"candle": "Marteau inversé", "direction": "LONG", "strength": 2}
    if is_doji and lower_shadow > full_range * 0.55:
        return {"candle": "Marteau Doji", "direction": "LONG", "strength": 3}
    if (upper_shadow > body * 2.0 and lower_shadow < body * 0.5 and body_ratio < 0.35 and c < o):
        return {"candle": "Étoile filante", "direction": "SHORT", "strength": 3}
    if (lower_shadow > body * 2.0 and upper_shadow < body * 0.6 and body_ratio < 0.35 and c < o):
        return {"candle": "Le Pendu", "direction": "SHORT", "strength": 2}
    return {"candle": None, "direction": None, "strength": 0}

def detect_chart_patterns(df):
    if len(df) < 40:
        return {"pattern": None, "direction": None, "strength": 0, "candle": None}
    last = df.iloc[-1]
    price = float(last["Close"])
    support = float(last["Support"])
    resistance = float(last["Resistance"])
    tol = CONFIG["sr_tolerance_pct"]
    near_support = abs(price - support) / price <= tol or price <= support * 1.025
    near_resistance = abs(price - resistance) / price <= tol or price >= resistance * 0.975
    candle_info = detect_candlestick(last)
    if near_support and candle_info["direction"] == "LONG" and candle_info["strength"] >= 2:
        return {"pattern": f"{candle_info['candle']} sur Support", "direction": "LONG", "strength": candle_info["strength"] + 1, "candle": candle_info["candle"]}
    if near_resistance and candle_info["direction"] == "SHORT" and candle_info["strength"] >= 2:
        return {"pattern": f"{candle_info['candle']} sur Résistance", "direction": "SHORT", "strength": candle_info["strength"] + 1, "candle": candle_info["candle"]}
    if price > resistance * 1.008 and last["Rel_Volume"] >= 1.4 and last["Close"] > last["Open"]:
        return {"pattern": "Breakout Résistance", "direction": "LONG", "strength": 3, "candle": candle_info["candle"]}
    if price < support * 0.992 and last["Rel_Volume"] >= 1.4 and last["Close"] < last["Open"]:
        return {"pattern": "Breakdown Support", "direction": "SHORT", "strength": 3, "candle": candle_info["candle"]}
    if near_support:
        lows = df["Low"].tail(30)
        min1 = lows.min()
        candidates = lows[lows > min1 * 1.008]
        if len(candidates) > 0:
            min2 = candidates.min()
            if abs(min1 - min2) / min1 < 0.025:
                return {"pattern": "Double Bottom sur Support", "direction": "LONG", "strength": 3, "candle": candle_info["candle"]}
    if near_resistance:
        highs = df["High"].tail(30)
        max1 = highs.max()
        candidates = highs[highs < max1 * 0.992]
        if len(candidates) > 0:
            max2 = candidates.max()
            if abs(max1 - max2) / max1 < 0.025:
                return {"pattern": "Double Top sur Résistance", "direction": "SHORT", "strength": 3, "candle": candle_info["candle"]}
    if candle_info["strength"] >= 3:
        return {"pattern": candle_info["candle"], "direction": candle_info["direction"], "strength": 2, "candle": candle_info["candle"]}
    return {"pattern": None, "direction": None, "strength": 0, "candle": None}

def check_long_setup(row, prev_row, pattern_info):
    vol_ok = row["Rel_Volume"] >= CONFIG["min_rel_volume"]
    macd_ok = row["MACD"] > row["MACD_Signal"]
    stoch_ok = (CONFIG["stoch_low"] <= row["Stoch_K"] <= CONFIG["stoch_high"]) and (row["Stoch_K"] >= row["Stoch_D"])
    return vol_ok and macd_ok and stoch_ok

def check_short_setup(row, prev_row, pattern_info):
    vol_ok = row["Rel_Volume"] >= CONFIG["min_rel_volume"]
    macd_ok = row["MACD"] < row["MACD_Signal"]
    stoch_ok = (CONFIG["stoch_low"] <= row["Stoch_K"] <= CONFIG["stoch_high"]) and (row["Stoch_K"] <= row["Stoch_D"])
    return vol_ok and macd_ok and stoch_ok

def get_stock_info(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        beta = info.get("beta")
        avg_vol = info.get("averageVolume") or info.get("averageVolume10days") or 0
        sector = info.get("sector", "N/A")
        quote_type = str(info.get("quoteType", "")).upper()
        is_etf = "ETF" in quote_type or "ETP" in quote_type
        return {"beta": beta, "avg_volume": avg_vol, "sector": sector, "is_etf": is_etf}
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
        df = calculate_indicators(df)
        df = df.dropna(subset=["MACD", "Stoch_K", "Rel_Volume"])
        if len(df) < 5:
            return None
        last = df.iloc[-1]
        prev = df.iloc[-2]
        info = get_stock_info(ticker)
        if info["is_etf"]:
            return None
        if info["beta"] is not None and info["beta"] < CONFIG["min_beta"]:
            return None
        if info["avg_volume"] < CONFIG["min_avg_volume"]:
            return None
        if float(last["Close"]) < CONFIG["min_price"]:
            return None
        pattern_info = detect_chart_patterns(df)
        is_long = check_long_setup(last, prev, pattern_info)
        is_short = check_short_setup(last, prev, pattern_info)
        if not (is_long or is_short):
            return None
        return {
            "Ticker": ticker,
            "Direction": "LONG" if is_long else "SHORT",
            "Price": round(float(last["Close"]), 2),
            "Beta": round(float(info["beta"]), 2) if info["beta"] else "N/A",
            "Rel_Volume": round(float(last["Rel_Volume"]), 2),
            "MACD": round(float(last["MACD"]), 3),
            "MACD_Signal": round(float(last["MACD_Signal"]), 3),
            "MACD_Hist": round(float(last["MACD_Hist"]), 3),
            "Stoch_K": round(float(last["Stoch_K"]), 1),
            "Stoch_D": round(float(last["Stoch_D"]), 1),
            "RSI": round(float(last["RSI"]), 1),
            "Pattern": pattern_info["pattern"],
            "Candle": pattern_info.get("candle"),
            "Pattern_Strength": pattern_info["strength"],
            "Support": round(float(last["Support"]), 2),
            "Resistance": round(float(last["Resistance"]), 2),
            "Sector": info["sector"],
            "Date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
    except Exception:
        return None

def run_robot():
    print("=" * 75)
    print(" ROBOT ACCÉLÉRATEUR v2.2 – Scan quotidien US Options")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 75)
    print(f" Univers : {len(UNIVERSE)} tickers high-beta")
    print(" Critères : Volume + MACD + Stochastique + Bougies/Figures sur S/R")
    print("-" * 75)
    results = []
    total = min(len(UNIVERSE), CONFIG["max_stocks_to_scan"])
    for i, ticker in enumerate(UNIVERSE[:total], 1):
        if i % 15 == 0:
            print(f"  Progression : {i}/{total}...")
        res = scan_ticker(ticker)
        if res:
            results.append(res)
            print(f"  ✅ {res['Ticker']:6s} → {res['Direction']:5s} | {res['Price']:8.2f} | RelVol {res['Rel_Volume']:.2f} | {res['Pattern']}")
        time.sleep(CONFIG["sleep_between_tickers"])
    print("-" * 75)
    print(f" Scan terminé : {total} titres | Setups validés : {len(results)}")
    print("=" * 75)

    with open("report_body.txt", "w", encoding="utf-8") as f:
        f.write("Robot Accelerateur - Rapport quotidien\n")
        f.write("Setups valides : %s\n\n" % len(results))
        if not results:
            f.write("Aucun setup aujourd'hui.\n")
        else:
            for r in results:
                f.write("- %s | %s | Prix %s | RelVol %s | %s\n" % (r["Ticker"], r["Direction"], r["Price"], r["Rel_Volume"], r.get("Pattern")))

    if not results:
        print("\n❌ Aucun setup ne valide les 4 conditions aujourd'hui.\n")
        pd.DataFrame().to_csv("report_latest.csv", index=False)
        return None
    df_res = pd.DataFrame(results)
    df_res = df_res.sort_values(by=["Direction", "Pattern_Strength", "Rel_Volume"], ascending=[True, False, False])
    if CONFIG["output_console"]:
        print("\n📊 RAPPORT QUOTIDIEN – SETUPS VALIDÉS\n")
        cols = ["Ticker", "Direction", "Price", "Beta", "Rel_Volume", "Stoch_K", "RSI", "Pattern", "Candle", "Support", "Resistance"]
        print(df_res[cols].to_string(index=False))
        print()
        print(f" LONG  : {len(df_res[df_res['Direction']=='LONG'])}")
        print(f" SHORT : {len(df_res[df_res['Direction']=='SHORT'])}")
    if CONFIG["output_csv"]:
        filename = f"Robot_Accelerateur_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df_res.to_csv(filename, index=False)
        df_res.to_csv("report_latest.csv", index=False)
        print(f"\n📁 Rapport sauvegardé : {filename}")
    return df_res

if __name__ == "__main__":
    run_robot()
