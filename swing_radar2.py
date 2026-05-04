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
TIGHT_RANGE_PCT = 0.05
VCP_LOOKBACK = 15

NASDAQ_100 = ['AAPL','MSFT','NVDA','AMZN','META','GOOGL','TSLA','AVGO','COST','NFLX','AMD','ADBE','QCOM','INTU','AMAT','AMGN','TXN','MU','LRCX','PANW','CRWD','FTNT','ORLY','CTAS','ROST','PAYX','DXCM','ADP','MRVL','IDXX','BIIB','VRSK','DLTR','TEAM','WDAY','ZS','DDOG','EBAY','NXPI','CSCO','TMUS','PEP','SBUX','ISRG','VRTX','GILD','CSX','ADSK','CHTR','LULU','MAR','MCHP']
SP500_SELECT = ['JPM','BAC','WFC','GS','MS','V','MA','UNH','JNJ','LLY','PFE','ABBV','MRK','TMO','DHR','XOM','CVX','COP','HD','LOW','TGT','WMT','NKE','MCD','BA','CAT','GE','RTX','LMT','NEE','AMT','PLD','LIN','SHW','UPS','FDX','DIS','CMCSA','VZ']

def get_tickers_from_tv():
    url = 'https://scanner.tradingview.com/america/scan'
    payload = {
        'filter': [
            {'left': 'exchange', 'operation': 'in_range', 'right': ['NASDAQ', 'NYSE', 'AMEX']},
            {'left': 'market_cap_basic', 'operation': 'greater', 'right': 2000000000},
            {'left': 'EMA50', 'operation': 'greater', 'right': 'EMA200'},
            {'left': 'close', 'operation': 'greater', 'right': 'EMA50'},
            {'left': 'average_volume_30d_calc', 'operation': 'greater', 'right': 1000000}
        ],
        'columns': ['name'],
        'range': [0, 500]
    }
    try:
        r = requests.post(url, json=payload, timeout=10)
        tickers = [x['d'][0] for x in r.json()['data']]
        print('TV Screener: ' + str(len(tickers)) + ' tickers')
        return tickers
    except Exception as e:
        print('TV Screener failed: ' + str(e))
        return list(set(NASDAQ_100 + SP500_SELECT))

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
    rs_clean = rs_series.dropna()
    rs_ma_clean = rs_ma50.dropna()
    if len(rs_clean) < 10 or len(rs_ma_clean) < 10:
        return False, {}
    rs_last = float(rs_clean.values[-1])
    rs_ma_last = float(rs_ma_clean.values[-1])
    rs_ok = rs_last > rs_ma_last
    rs_vs_ma = (rs_last / rs_ma_last - 1) * 100
    passed = trend_ok and momentum_ok and position_ok and rs_ok
    detail = {
        'price': round(lc, 2),
        'ema20': round(e20, 2),
        'ema50': round(e50, 2),
        'high_52w': round(high_52w, 2),
        'from_high': round(from_high*100, 1),
        'rs_vs_ma': round(rs_vs_ma, 1)
    }
    return passed, detail

def count_consolidation_days(high, low, max_range_pct=0.08):
    """
    從今日往前數，連續幾日都喺 max_range_pct 波幅內
    用嚟顯示「整固 X 日」
    """
    if len(high) < 2:
        return 1
    days = 1
    base_high = float(high.iloc[-1])
    base_low = float(low.iloc[-1])
    for i in range(2, min(len(high), 31)):
        h = float(high.iloc[-i])
        l = float(low.iloc[-i])
        base_high = max(base_high, h)
        base_low = min(base_low, l)
        if base_low > 0 and (base_high - base_low) / base_low < max_range_pct:
            days = i
        else:
            break
    return days

def check_setups(hist, detail):
    """
    Setup 邏輯（v4）：
    -------------------------------------------------------
    Setup0  預備觀察  整固中，距突破 <3%，縮量  → 加 Watchlist
    Setup2  VCP突破   整固後放量突破             → 買入訊號
    Setup3  20日新高  創新高放量                 → 買入訊號
    -------------------------------------------------------
    優先級：若同一隻股票同時符合 Setup0 + Setup2/3
            只顯示 Setup2/3，唔重複顯示 Setup0
    Setup1 已移除：回測勝率只有 41.9%
    """
    close = hist['Close']
    high = hist['High']
    low = hist['Low']
    volume = hist['Volume']
    avg_vol_50 = volume.rolling(50).mean()
    setups = []
    has_buy_signal = False  # 標記有冇正式買入訊號

    # ==============================
    # Setup2：VCP 緊密整固 + 放量突破
    # ==============================
    if len(high) >= 6:
        h5 = float(high.iloc[-6:-1].max())
        l5 = float(low.iloc[-6:-1].min())
        today_close = float(close.iloc[-1])
        today_vol = float(volume.iloc[-1])
        avg_vol = float(avg_vol_50.iloc[-1]) if not pd.isna(avg_vol_50.iloc[-1]) else 0

        if l5 > 0 and avg_vol > 0:
            tight_range = (h5 - l5) / l5 < TIGHT_RANGE_PCT
            breakout = today_close > h5
            volume_ok = today_vol > avg_vol

            if tight_range and breakout and volume_ok:
                has_buy_signal = True
                vcp_flag = False
                if len(high) >= VCP_LOOKBACK:
                    mid = VCP_LOOKBACK // 2
                    rf = (float(high.iloc[-VCP_LOOKBACK:-mid].max()) - float(low.iloc[-VCP_LOOKBACK:-mid].min())) / float(low.iloc[-VCP_LOOKBACK:-mid].min()) if float(low.iloc[-VCP_LOOKBACK:-mid].min()) > 0 else 0
                    rl = (float(high.iloc[-mid:].max()) - float(low.iloc[-mid:].min())) / float(low.iloc[-mid:].min()) if float(low.iloc[-mid:].min()) > 0 else 0
                    vcp_flag = (rl < rf * 0.75) and rf > 0
                setups.append({
                    'type': 'Setup2',
                    'range_pct': round((h5-l5)/l5*100, 1),
                    'vcp': vcp_flag,
                    'vol_ratio': round(today_vol / avg_vol, 2),
                    'breakout_price': round(h5, 2)
                })

    # ==============================
    # Setup3：20 日新高 + 成交量 >= 1.5x 均量
    # ==============================
    if len(close) >= 20 and len(volume) >= 50:
        high20 = float(close.iloc[-20:-1].max())
        today_close = float(close.iloc[-1])
        today_vol = float(volume.iloc[-1])
        avg_vol = float(avg_vol_50.iloc[-1]) if not pd.isna(avg_vol_50.iloc[-1]) else 0
        if today_close > high20 and avg_vol > 0 and today_vol >= avg_vol * 1.5:
            has_buy_signal = True
            setups.append({
                'type': 'Setup3',
                'vol_ratio': round(today_vol / avg_vol, 2),
                'high20': round(high20, 2)
            })

    # ==============================
    # Setup0：預備觀察（只喺冇買入訊號時才加）
    # 條件：
    #   1. 10日波幅 < 8%
    #   2. 今日收市 >= 10日高位 97%（距突破 < 3%）
    #   3. 今日未突破 10日高位（否則係 Setup2）
    #   4. 近5日均量 < 50日均量 80%（縮量）
    # ==============================
    if not has_buy_signal and len(high) >= 11 and len(volume) >= 50:
        h10 = float(high.iloc[-11:-1].max())   # 前10日最高（唔包今日）
        l10 = float(low.iloc[-11:-1].min())    # 前10日最低（唔包今日）
        today_close = float(close.iloc[-1])
        avg_vol = float(avg_vol_50.iloc[-1]) if not pd.isna(avg_vol_50.iloc[-1]) else 0
        recent_5d_vol = float(volume.iloc[-5:].mean()) if len(volume) >= 5 else 0

        if l10 > 0 and avg_vol > 0 and h10 > 0:
            range10 = (h10 - l10) / l10
            near_breakout = today_close >= h10 * 0.97   # 距突破 < 3%
            not_broken = today_close < h10               # 未突破
            low_vol = recent_5d_vol < avg_vol * 0.80    # 縮量 < 80%

            if range10 < 0.08 and near_breakout and not_broken and low_vol:
                consol_days = count_consolidation_days(high, low, max_range_pct=0.08)
                dist_pct = round((h10 - today_close) / today_close * 100, 1)
                vol_ratio = round(recent_5d_vol / avg_vol, 2)
                setups.append({
                    'type': 'Setup0',
                    'range_pct': round(range10*100, 1),
                    'breakout_price': round(h10, 2),
                    'dist_pct': dist_pct,
                    'vol_ratio': vol_ratio,
                    'consol_days': consol_days
                })

    return setups

def run_scan():
    ALL_TICKERS = get_tickers_from_tv()
    scanned_count = len(ALL_TICKERS)
    print('===== Swing Radar (v4) =====')
    spy_hist = yf.download('SPY', period='2y', auto_adjust=True, progress=False)
    if spy_hist.empty:
        print('SPY failed')
        return [], scanned_count
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
    s0 = len([r for r in results if r['type'] == 'Setup0'])
    s2 = len([r for r in results if r['type'] == 'Setup2'])
    s3 = len([r for r in results if r['type'] == 'Setup3'])
    print('Setup0 (候選): ' + str(s0) + ' | Setup2 (VCP突破): ' + str(s2) + ' | Setup3 (新高): ' + str(s3))
    return results, scanned_count

def send_discord(results, scanned_count=0):
    if not DISCORD_WEBHOOK_URL:
        print('No webhook')
        return
    today = date.today().strftime('%Y-%m-%d')
    s0 = [r for r in results if r['type'] == 'Setup0']
    s2 = [r for r in results if r['type'] == 'Setup2']
    s3 = [r for r in results if r['type'] == 'Setup3']
    buy_count = len(s2) + len(s3)

    embeds = []

    # ---- Embed 1：買入訊號（綠色）----
    buy_embed = {
        'title': today + ' Swing Radar (v4)',
        'description': 'Scanned ' + str(scanned_count) + ' tickers | 買入訊號: ' + str(buy_count) + ' | 候選觀察: ' + str(len(s0)),
        'color': 3066993,  # 綠色
        'fields': []
    }
    if s2:
        buy_embed['fields'].append({
            'name': '🚀 Setup2 VCP突破 (勝率53%) — 買入訊號',
            'value': '\n'.join([
                '$'+r['ticker']+
                ' | 現價 $'+str(r['price'])+
                ' | 整固 波幅'+str(r.get('range_pct',0))+'%'+
                (' VCP' if r.get('vcp') else '')+
                ' | 量'+str(r.get('vol_ratio',0))+'x'+
                ' | 突破$'+str(r.get('breakout_price',0))
                for r in s2
            ]),
            'inline': False
        })
    if s3:
        buy_embed['fields'].append({
            'name': '📈 Setup3 20日新高放量 — 買入訊號',
            'value': '\n'.join([
                '$'+r['ticker']+
                ' | 現價 $'+str(r['price'])+
                ' | 量'+str(r.get('vol_ratio',0))+'x'+
                ' | 前高$'+str(r.get('high20',0))
                for r in s3
            ]),
            'inline': False
        })
    if not s2 and not s3:
        buy_embed['fields'].append({
            'name': '買入訊號',
            'value': 'No buy signals today.',
            'inline': False
        })
    embeds.append(buy_embed)

    # ---- Embed 2：候選觀察（藍色，獨立）----
    if s0:
        display = s0[:8]
        extra = len(s0) - 8
        watchlist_lines = []
        for r in display:
            line = (
                '$'+r['ticker']+
                ' | 現價 $'+str(r['price'])+
                ' | 整固 '+str(r.get('consol_days',10))+'日 波幅'+str(r.get('range_pct',0))+'%'+
                ' | 距突破 $'+str(r.get('breakout_price',0))+' 僅'+str(r.get('dist_pct',0))+'%'+
                ' | 縮量 '+str(r.get('vol_ratio',0))+'x'
            )
            watchlist_lines.append(line)
        if extra > 0:
            watchlist_lines.append('...另有 ' + str(extra) + ' 隻候選')

        watchlist_embed = {
            'title': '👀 Setup0 候選觀察 (未突破，請先加入 Watchlist)',
            'description': '以下股票整固中，距突破位 <3%，等放量突破確認先行動',
            'color': 3447003,  # 藍色
            'fields': [{
                'name': '候選清單',
                'value': '\n'.join(watchlist_lines),
                'inline': False
            }]
        }
        embeds.append(watchlist_embed)

    try:
        resp = requests.post(
            DISCORD_WEBHOOK_URL,
            data=json.dumps({'embeds': embeds}),
            headers={'Content-Type': 'application/json'},
            timeout=10
        )
        print('Discord: ' + str(resp.status_code))
    except Exception as e:
        print('err:' + str(e))

if __name__ == '__main__':
    results, scanned_count = run_scan()
    send_discord(results, scanned_count)
    print('Done!')
