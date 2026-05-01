“””
╔══════════════════════════════════════════════════════════════════╗
║        美股清晨備課選股雷達  |  Swing Trading Setup Scanner       ║
║        風格：Minervini 強勢股 + 均線回踩 / 緊湊盤整 VCP           ║
╚══════════════════════════════════════════════════════════════════╝
“””

import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, date
import time
import warnings
warnings.filterwarnings(“ignore”)

# ══════════════════════════════════════════════

# 設定（Webhook 從環境變數讀取，安全）

# ══════════════════════════════════════════════

DISCORD_WEBHOOK_URL = os.environ.get(“DISCORD_WEBHOOK_URL”, “”)

MIN_PRICE           = 10.0
MIN_AVG_VOL_50      = 500_000
MAX_FROM_52W_HIGH   = 0.25
EMA_TOUCH_PCT       = 0.015
TIGHT_RANGE_PCT     = 0.05
VCP_LOOKBACK        = 15

# ══════════════════════════════════════════════

# Ticker 清單

# ══════════════════════════════════════════════

NASDAQ_100 = [
“AAPL”,“MSFT”,“NVDA”,“AMZN”,“META”,“GOOGL”,“GOOG”,“TSLA”,“AVGO”,“COST”,
“NFLX”,“AMD”,“ADBE”,“QCOM”,“INTC”,“INTU”,“AMAT”,“AMGN”,“BKNG”,“TXN”,
“MU”,“LRCX”,“PANW”,“KLAC”,“SNPS”,“CDNS”,“MELI”,“ABNB”,“REGN”,“CRWD”,
“FTNT”,“MNST”,“ORLY”,“CTAS”,“PCAR”,“ROST”,“PAYX”,“DXCM”,“ADP”,“CPRT”,
“FAST”,“MRVL”,“KDP”,“IDXX”,“ODFL”,“BIIB”,“CTSH”,“VRSK”,“ANSS”,“DLTR”,
“GEHC”,“ON”,“FANG”,“TEAM”,“WDAY”,“ZS”,“DDOG”,“ALGN”,“EBAY”,“NXPI”,
“CSCO”,“TMUS”,“HON”,“PEP”,“SBUX”,“PYPL”,“ISRG”,“VRTX”,“GILD”,“CSX”,
“MRNA”,“ADSK”,“CHTR”,“LULU”,“MAR”,“MDLZ”,“MCHP”,“XLNX”,“SWKS”,
]

SP500_SELECT = [
“JPM”,“BAC”,“WFC”,“GS”,“MS”,“BLK”,“SCHW”,“AXP”,“V”,“MA”,
“UNH”,“JNJ”,“LLY”,“PFE”,“ABBV”,“MRK”,“BMY”,“TMO”,“DHR”,“ABT”,
“XOM”,“CVX”,“COP”,“SLB”,“EOG”,“OXY”,“PSX”,“VLO”,“MPC”,“HAL”,
“HD”,“LOW”,“TGT”,“WMT”,“TJX”,“NKE”,“MCD”,“YUM”,
“BA”,“CAT”,“DE”,“MMM”,“GE”,“RTX”,“LMT”,“NOC”,“GD”,
“NEE”,“DUK”,“SO”,“AMT”,“PLD”,“CCI”,“EQIX”,“SPG”,“O”,“WELL”,
“LIN”,“APD”,“SHW”,“ECL”,“PPG”,“UPS”,“FDX”,“ODFL”,“NSC”,
“DIS”,“CMCSA”,“T”,“VZ”,
# ── 擴展區：加入更多 Ticker ──
]

ALL_TICKERS = list(set(NASDAQ_100 + SP500_SELECT))

# ══════════════════════════════════════════════

# 工具函數

# ══════════════════════════════════════════════

def calc_ema(series, span):
return series.ewm(span=span, adjust=False).mean()

def calc_rs(stock_close, spy_close):
aligned_spy = spy_close.reindex(stock_close.index, method=“ffill”)
return stock_close / aligned_spy

def check_minervini(hist, spy_hist):
close = hist[“Close”]
if len(close) < 210:
return False, {}

```
ema10  = calc_ema(close, 10)
ema20  = calc_ema(close, 20)
ema50  = calc_ema(close, 50)
ema150 = calc_ema(close, 150)
ema200 = calc_ema(close, 200)

lc, e10, e20, e50, e150, e200 = (
    float(close.iloc[-1]), float(ema10.iloc[-1]), float(ema20.iloc[-1]),
    float(ema50.iloc[-1]), float(ema150.iloc[-1]), float(ema200.iloc[-1])
)

trend_ok    = (lc > e150) and (lc > e200) and (e150 > e200)
momentum_ok = (e10 > e20) and (e20 > e50)
high_52w    = float(close.tail(252).max())
from_high   = (lc - high_52w) / high_52w
position_ok = from_high >= -MAX_FROM_52W_HIGH

rs_series   = calc_rs(close, spy_hist["Close"])
rs_ma50     = rs_series.rolling(50).mean()
rs_ok       = float(rs_series.iloc[-1]) > float(rs_ma50.iloc[-1])
rs_vs_ma    = (float(rs_series.iloc[-1]) / float(rs_ma50.iloc[-1]) - 1) * 100

passed = trend_ok and momentum_ok and position_ok and rs_ok
detail = {
    "price": round(lc, 2), "ema10": round(e10, 2),
    "ema20": round(e20, 2), "ema50": round(e50, 2),
    "ema150": round(e150, 2), "ema200": round(e200, 2),
    "high_52w": round(high_52w, 2), "from_high": round(from_high * 100, 1),
    "rs_vs_ma": round(rs_vs_ma, 1),
}
return passed, detail
```

def check_setups(hist, detail):
close, low, high, volume = hist[“Close”], hist[“Low”], hist[“High”], hist[“Volume”]
ema20      = calc_ema(close, 20)
ema50      = calc_ema(close, 50)
avg_vol_50 = volume.rolling(50).mean()
setups     = []

```
# Setup 1：均線縮量回踩
for ema_series, ema_name in [(ema20, "EMA 20"), (ema50, "EMA 50")]:
    ema_val     = float(ema_series.iloc[-1])
    recent_lows = [float(low.iloc[-1]), float(low.iloc[-2])]
    touched     = any(abs(l - ema_val) / ema_val <= EMA_TOUCH_PCT for l in recent_lows)
    above       = float(close.iloc[-1]) > ema_val
    low_vol     = float(volume.iloc[-1]) < float(avg_vol_50.iloc[-1])

    if touched and above and low_vol:
        setups.append({
            "type": "Setup 1", "ema_name": ema_name,
            "ema_val": round(ema_val, 2),
            "vol_ratio": round(float(volume.iloc[-1]) / float(avg_vol_50.iloc[-1]), 2),
            "vcp": False,
        })

# Setup 2：緊湊盤整
if len(high) >= 5:
    h5, l5 = float(high.iloc[-5:].max()), float(low.iloc[-5:].min())
    rng5   = (h5 - l5) / l5

    if rng5 < TIGHT_RANGE_PCT:
        vcp_flag = False
        if len(high) >= VCP_LOOKBACK:
            mid       = VCP_LOOKBACK // 2
            h_f = float(high.iloc[-VCP_LOOKBACK:-mid].max())
            l_f = float(low.iloc[-VCP_LOOKBACK:-mid].min())
            h_l = float(high.iloc[-mid:].max())
            l_l = float(low.iloc[-mid:].min())
            rf  = (h_f - l_f) / l_f if l_f > 0 else 0
            rl  = (h_l - l_l) / l_l if l_l > 0 else 0
            vcp_flag = (rl < rf * 0.75) and rf > 0

        setups.append({
            "type": "Setup 2",
            "range_pct": round(rng5 * 100, 1),
            "vcp": vcp_flag,
        })

return setups
```

# ══════════════════════════════════════════════

# 主掃描邏輯

# ══════════════════════════════════════════════

def run_scan():
print(f”\n{’=’*55}”)
print(f”  美股選股雷達  {datetime.now().strftime(’%Y-%m-%d %H:%M UTC’)}”)
print(f”  掃描：{len(ALL_TICKERS)} 隻”)
print(f”{’=’*55}\n”)

```
# 下載 SPY
print("▶ 下載 SPY...")
spy_hist = yf.download("SPY", period="2y", auto_adjust=True, progress=False)
if spy_hist.empty:
    print("❌ SPY 數據失敗")
    return []

# 批量下載
print("▶ 批量下載股票...")
all_hist  = {}
batch_size = 50

for i in range(0, len(ALL_TICKERS), batch_size):
    batch = ALL_TICKERS[i:i+batch_size]
    try:
        data = yf.download(
            " ".join(batch), period="2y",
            auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
        if len(batch) == 1:
            all_hist[batch[0]] = data
        else:
            for t in batch:
                try:
                    if t in data.columns.get_level_values(0):
                        td = data[t].dropna()
                        if not td.empty:
                            all_hist[t] = td
                except Exception:
                    pass
    except Exception as e:
        print(f"  ⚠ 批次錯誤：{e}")
    time.sleep(0.5)

print(f"  ✓ 載入 {len(all_hist)} 隻\n")

# 第一層
print("▶ 第一層：Minervini 過濾...")
passed_l1 = []
for ticker in ALL_TICKERS:
    hist = all_hist.get(ticker)
    if hist is None or len(hist) < 210:
        continue
    if float(hist["Close"].iloc[-1]) < MIN_PRICE:
        continue
    if float(hist["Volume"].tail(50).mean()) < MIN_AVG_VOL_50:
        continue
    passed, detail = check_minervini(hist, spy_hist)
    if not passed:
        continue
    detail["ticker"] = ticker
    detail["hist"]   = hist
    passed_l1.append(detail)

print(f"  → 通過：{len(passed_l1)} 隻\n")

# 第二層
print("▶ 第二層：買點捕捉...")
results = []
for detail in passed_l1:
    ticker = detail["ticker"]
    hist   = detail.pop("hist")
    for setup in check_setups(hist, detail):
        results.append({**detail, **setup})

print(f"  → Setup：{len(results)} 個\n")
return results
```

# ══════════════════════════════════════════════

# Discord 通知

# ══════════════════════════════════════════════

def send_discord(results):
if not DISCORD_WEBHOOK_URL:
print(“⚠ 未設定 DISCORD_WEBHOOK_URL”)
print_terminal(results)
return

```
today  = date.today().strftime("%Y-%m-%d")
setup1 = [r for r in results if r["type"] == "Setup 1"]
setup2 = [r for r in results if r["type"] == "Setup 2"]

embed = {
    "title":       f"📅 {today}  盤後備課名單",
    "description": f"掃描 **{len(ALL_TICKERS)}** 隻  ｜  發現 **{len(results)}** 個 Setup",
    "color":       0x00d9ff if results else 0xff5252,
    "timestamp":   datetime.utcnow().isoformat(),
    "footer":      {"text": "美股清晨備課選股雷達 · Swing Trading"},
    "fields":      [],
}

if setup1:
    lines = [
        f"**${r['ticker']}** `${r['price']:.2f}`  "
        f"靠近 {r.get('ema_name','EMA')} ({r.get('ema_val',0):.2f})  "
        f"量比 `{r.get('vol_ratio',0):.2f}x`"
        for r in setup1
    ]
    embed["fields"].append({
        "name":   f"🎯 Setup 1 ── 均線縮量回踩（{len(setup1)} 隻）",
        "value":  "\n".join(lines), "inline": False,
    })

if setup2:
    lines = [
        f"**${r['ticker']}** `${r['price']:.2f}`  "
        f"波幅 `{r.get('range_pct',0):.1f}%`"
        + ("  🔥 **疑似 VCP**" if r.get("vcp") else "")
        for r in setup2
    ]
    embed["fields"].append({
        "name":   f"🎯 Setup 2 ── 緊湊盤整（{len(setup2)} 隻）",
        "value":  "\n".join(lines), "inline": False,
    })

if results:
    rows = [
        f"`{r['ticker']:5s}` ${r['price']:.2f} "
        f"E20:{r['ema20']:.2f} E50:{r['ema50']:.2f} "
        f"距高:{r['from_high']:.1f}% RS:{r['rs_vs_ma']:+.1f}%"
        for r in results[:12]
    ]
    embed["fields"].append({
        "name": "📊 技術詳情", "value": "\n".join(rows), "inline": False,
    })

if not results:
    embed["fields"].append({
        "name": "今日結果", "inline": False,
        "value": "無符合條件股票，市場整固期，耐心等待。",
    })

try:
    resp = requests.post(
        DISCORD_WEBHOOK_URL,
        data=json.dumps({"embeds": [embed]}),
        headers={"Content-Type": "application/json"},
        timeout=10,
    )
    print("✅ Discord 已發送！" if resp.status_code in (200, 204)
          else f"⚠ Discord 錯誤：{resp.status_code}")
except Exception as e:
    print(f"❌ 發送失敗：{e}")
```

def print_terminal(results):
today  = date.today().strftime(”%Y-%m-%d”)
setup1 = [r for r in results if r[“type”] == “Setup 1”]
setup2 = [r for r in results if r[“type”] == “Setup 2”]
print(f”\n{‘═’*55}”)
print(f”  📅 {today}  盤後備課名單”)
print(f”{‘═’*55}”)
if setup1:
print(”\n  🎯 Setup 1：均線縮量回踩”)
for r in setup1:
print(f”     ${r[‘ticker’]:6s} ${r[‘price’]:.2f}  “
f”{r.get(‘ema_name’,‘EMA’)} ${r.get(‘ema_val’,0):.2f}  “
f”量比 {r.get(‘vol_ratio’,0):.2f}x”)
if setup2:
print(”\n  🎯 Setup 2：緊湊盤整”)
for r in setup2:
vcp = “  [🔥 疑似 VCP]” if r.get(“vcp”) else “”
print(f”     ${r[‘ticker’]:6s} ${r[‘price’]:.2f}  “
f”波幅 {r.get(‘range_pct’,0):.1f}%{vcp}”)
if not results:
print(”\n  今日無符合條件股票”)
print(f”\n{‘═’*55}\n”)

if **name** == “**main**”:
results = run_scan()
send_discord(results)
print(“✅ 完成！”)
