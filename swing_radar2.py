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

NASDAQ_100 = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','COST','NFLX','AMD','ADBE','QCOM','INTU','AMAT','AMGN','TXN','MU','LRCX','PANW','CRWD','FTNT','ORLY','CTAS','ROST','PAYX','DXCM','ADP','MRVL','IDXX','BIIB','VRSK','DLTR','TEAM','WDAY','ZS','DDOG','EBAY','NXPI','CSCO','TMUS','PEP','SBUX','ISRG','VRTX','GILD','CSX','ADSK','CHTR','LULU','MAR','MCHP']
SP500_SELECT = ['JPM','BAC','WFC','GS','MS','V','MA','UNH','JNJ','LLY','PFE','ABBV','MRK','TMO','DHR','XOM','CVX','COP','HD','LOW','TGT','WMT','NKE','MCD','BA','CAT','GE','RTX','LMT','NEE','AMT','PLD','LIN','SHW','UPS','FDX','DIS','CMCSA','VZ']
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
    rs_last = float(rs_series.dropna().values[-1])
    rs_ma_last = float(rs_ma50.dropna().values[-1])
    rs_ma_last = float(rs_ma50.dropna().iloc[-1]) if not hasattr(rs_ma50.dropna().iloc[-1], "iloc") else float(rs_ma50.dropna().iloc[-1].iloc[0])
    rs_ok = rs_last > rs_ma_last
    rs_vs_ma = (rs_last / rs_ma_last - 1) * 100
    passed = trend_ok and momentum_ok and position_ok and rs_ok
    detail = {'price': round(lc,2), 'ema20': round(e20,2), 'ema50': round(e50,2), 'high_52w': round(high_52w,2), 'from_high': round(from_high*100,1), 'rs_vs_ma': round(rs_vs_ma,1)}
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
            setups.append({'type': 'Setup1', 'ema_name': ema_name, 'ema_val': round(ema_val,2), 'vol_ratio': round(float(volume.iloc[-1])/float(avg_vol_50.iloc[-1]),2), 'vcp': False})
    if len(high) >= 5:
        h5 = float(high.iloc[-5:].max())
        l5 = float(low.iloc[-5:].min())
        rng5 = (h5 - l5) / l5
        if rng5 < TIGHT_RANGE_PCT:
            vcp_flag = False
            if len(high) >= VCP_LOOKBACK:
                mid = VCP_LOOKBACK // 2
                rf = (float(high.iloc[-VCP_LOOKBACK:-mid].max()) - float(low.iloc[-VCP_LOOKBACK:-mid].min())) / float(low.iloc[-VCP_LOOKBACK:-mid].min()) if float(low.iloc[-VCP_LOOKBACK:-mid].min()) > 0 else 0
                rl = (float(high.iloc[-mid:].max()) - float(low.iloc[-mid:].min())) / float(low.iloc[-mid:].min()) if float(low.iloc[-mid:].min()) > 0 else 0
                vcp_flag = (rl < rf * 0.75) and rf > 0
            setups.append({'type': 'Setup2', 'range_pct': round(rng5*100,1), 'vcp': vcp_flag})
    return setups

def run_scan():
    print('===== Swing Radar =====')
    spy_hist = yf.download('SPY', period='2y', auto_adjust=True, progress=False)
    if spy_hist.empty:
        print('SPY failed')
        return []
    all_hist = {}
    for i in range(0, len(ALL_TICKERS), 50):
        batch = ALL_TICKERS[i:i+50]
        try:
            data = yf.download(' '.join(batch), period='2y', auto_adjust=True, progress=False, group_by='ticker', threads=True)
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
            print('err:' + str(e))
        time.sleep(0.5)
    results = []
    for ticker in ALL_TICKERS:
        hist = all_hist.get(ticker)
        if hist is None or len(hist) < 210:
            continue
        if float(hist['Close'].iloc[-1]) < MIN_PRICE:
            continue
        if float(hist['Volume'].tail(50).mean()) < MIN_AVG_VOL_50:
            continue
        passed, detail = check_minervini(hist, spy_hist)
        if not passed:
            continue
        detail['ticker'] = ticker
        for setup in check_setups(hist, detail):
            results.append({**detail, **setup})
    print('Setups: ' + str(len(results)))
    return results

def send_discord(results):
    if not DISCORD_WEBHOOK_URL:
        print('No webhook')
        return
    today = date.today().strftime('%Y-%m-%d')
    s1 = [r for r in results if r['type'] == 'Setup1']
    s2 = [r for r in results if r['type'] == 'Setup2']
    embed = {'title': today + ' Swing Radar', 'description': 'Found ' + str(len(results)), 'color': 56575, 'fields': []}
    if s1:
        embed['fields'].append({'name': 'Setup1 EMA Pullback', 'value': '\n'.join(['$'+r['ticker']+' $'+str(r['price'])+' '+r.get('ema_name','')+' vol '+str(r.get('vol_ratio',0))+'x' for r in s1]), 'inline': False})
    if s2:
        embed['fields'].append({'name': 'Setup2 Tight Base', 'value': '\n'.join(['$'+r['ticker']+' $'+str(r['price'])+' '+str(r.get('range_pct',0))+'%'+(' VCP' if r.get('vcp') else '') for r in s2]), 'inline': False})
    if not results:
        embed['fields'].append({'name': 'Result', 'value': 'No setups today.', 'inline': False})
    try:
        resp = requests.post(DISCORD_WEBHOOK_URL, data=json.dumps({'embeds': [embed]}), headers={'Content-Type': 'application/json'}, timeout=10)
        print('Discord: ' + str(resp.status_code))
    except Exception as e:
        print('err:' + str(e))

if __name__ == '__main__':
    results = run_scan()
    send_discord(results)
    print('Done!')

