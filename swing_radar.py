import os
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, date
import time
import warnings
warnings.filterwarnings('ignore')

DISCORD_WEBHOOK_URL = os.environ.get('DISCORD_WEBHOOK_URL') or ''
MIN_PRICE = 10.0
MIN_AVG_VOL_50 = 500000
MAX_FROM_52W_HIGH = 0.25
EMA_TOUCH_PCT = 0.015
TIGHT_RANGE_PCT = 0.05
VCP_LOOKBACK = 15

NASDAQ_100 = [
    'AAPL','MSFT','NVDA','AMZN','META','GOOGL','GOOG','TSLA','AVGO','COST',
    'NFLX','AMD','ADBE','QCOM','INTC','INTU','AMAT','AMGN','BKNG','TXN',
    'MU','LRCX','PANW','KLAC','SNPS','CDNS','MELI','ABNB','REGN','CRWD',
    'FTNT','MNST','ORLY','CTAS','PCAR','ROST','PAYX','DXCM','ADP','CPRT',
    'FAST','MRVL','KDP','IDXX','ODFL','BIIB','CTSH','VRSK','ANSS','DLTR',
    'GEHC','ON','FANG','TEAM','WDAY','ZS','DDOG','ALGN','EBAY','NXPI',
    'CSCO','TMUS','HON','PEP','SBUX','PYPL','ISRG','VRTX','GILD','CSX',
    'MRNA','ADSK','CHTR','LULU','MAR','MDLZ','MCHP',
]

SP500_SELECT = [
    'JPM','BAC','WFC','GS','MS','BLK','SCHW','AXP','V','MA',
    'UNH','JNJ','LLY','PFE','ABBV','MRK','BMY','TMO','DHR','ABT',
    'XOM','CVX','COP','SLB','EOG','OXY','PSX','VLO','MPC','HAL',
    'HD','LOW','TGT','WMT','TJX','NKE','MCD','YUM',
    'BA','CAT','DE','MMM','GE','RTX','LMT','NOC','GD',
    'NEE','DUK','SO','AMT','PLD','CCI','EQIX','SPG','O','WELL',
    'LIN','APD','SHW','ECL','PPG','UPS','FDX','NSC',
    'DIS','CMCSA','T','VZ',
]

ALL_TICKERS = list(set(NASDAQ_100 + SP500_SELECT))


def calc_ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def calc_rs(stock_close, spy_close):
    aligned_spy = spy_close.reindex(stock_close.index, method='ffill')
    return stock_close / aligned_spy


def check_minervini(hist, spy_hist):
    close = hist['Close']
    if len(close) < 210:
        return False, {}
    ema10 = calc_ema(close, 10)
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    ema150 = calc_ema(close, 150)
    ema200 = calc_ema(close, 200)
    lc = float(close.iloc[-1])
    e10 = float(ema10.iloc[-1])
    e20 = float(ema20.iloc[-1])
    e50 = float(ema50.iloc[-1])
    e150 = float(ema150.iloc[-1])
    e200 = float(ema200.iloc[-1])
    trend_ok = (lc > e150) and (lc > e200) and (e150 > e200)
    momentum_ok = (e10 > e20) and (e20 > e50)
    high_52w = float(close.tail(252).max())
    from_high = (lc - high_52w) / high_52w
    position_ok = from_high >= -MAX_FROM_52W_HIGH
    rs_series = calc_rs(close, spy_hist['Close'])
    rs_ma50 = rs_series.rolling(50).mean()
    rs_ok = float(rs_series.iloc[-1]) > float(rs_ma50.iloc[-1])
    rs_vs_ma = (float(rs_series.iloc[-1]) / float(rs_ma50.iloc[-1]) - 1) * 100
    passed = trend_ok and momentum_ok and position_ok and rs_ok
    detail = {
        'price': round(lc, 2),
        'ema10': round(e10, 2),
        'ema20': round(e20, 2),
        'ema50': round(e50, 2),
        'high_52w': round(high_52w, 2),
        'from_high': round(from_high * 100, 1),
        'rs_vs_ma': round(rs_vs_ma, 1),
    }
    return passed, detail


def check_setups(hist, detail):
    close = hist['Close']
    low = hist['Low']
    high = hist['High']
    volume = hist['Volume']
    ema20 = calc_ema(close, 20)
    ema50 = calc_ema(close, 50)
    avg_vol_50 = volume.rolling(50).mean()
    setups = []
    for ema_series, ema_name in [(ema20, 'EMA20'), (ema50, 'EMA50')]:
        ema_val = float(ema_series.iloc[-1])
        recent_lows = [float(low.iloc[-1]), float(low.iloc[-2])]
        touched = any(abs(l - ema_val) / ema_val <= EMA_TOUCH_PCT for l in recent_lows)
        above = float(close.iloc[-1]) > ema_val
        low_vol = float(volume.iloc[-1]) < float(avg_vol_50.iloc[-1])
        if touched and above and low_vol:
            vol_ratio = float(volume.iloc[-1]) / float(avg_vol_50.iloc[-1])
            setups.append({
                'type': 'Setup1',
                'ema_name': ema_name,
                'ema_val': round(ema_val, 2),
                'vol_ratio': round(vol_ratio, 2),
                'vcp': False,
            })
    if len(high) >= 5:
        h5 = float(high.iloc[-5:].max())
        l5 = float(low.iloc[-5:].min())
        rng5 = (h5 - l5) / l5
        if rng5 < TIGHT_RANGE_PCT:
            vcp_flag = False
            if len(high) >= VCP_LOOKBACK:
                mid = VCP_LOOKBACK // 2
                h_f = float(high.iloc[-VCP_LOOKBACK:-mid].max())
                l_f = float(low.iloc[-VCP_LOOKBACK:-mid].min())
                h_l = float(high.iloc[-mid:].max())
                l_l = float(low.iloc[-mid:].min())
                rf = (h_f - l_f) / l_f if l_f > 0 else 0
                rl = (h_l - l_l) / l_l if l_l > 0 else 0
                vcp_flag = (rl < rf * 0.75) and rf > 0
            setups.append({
                'type': 'Setup2',
                'range_pct': round(rng5 * 100, 1),
                'vcp': vcp_flag,
            })
    return setups


def run_scan​​​​​​​​​​​​​​​​
