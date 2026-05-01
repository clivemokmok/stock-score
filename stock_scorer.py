import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="美股強勢股技術評分", page_icon="📈", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; background-color: #0a0e17; color: #c8d6e5; }
.stApp { background-color: #0a0e17; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1100px; }
.stTextInput > div > div > input { background: #0f1724; border: 1px solid #1e2d40; border-radius: 4px; color: #00d9ff; font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; padding: 0.6rem 1rem; }
.stTextInput > div > div > input:focus { border-color: #00d9ff; box-shadow: 0 0 0 2px rgba(0,217,255,0.15); }
.stButton > button { background: transparent; border: 1px solid #00d9ff; color: #00d9ff; font-family: 'IBM Plex Mono', monospace; font-size: 0.85rem; letter-spacing: 2px; text-transform: uppercase; padding: 0.5rem 2rem; border-radius: 3px; }
.score-card { background: #0f1724; border: 1px solid #1e2d40; border-radius: 6px; padding: 1.1rem 1.4rem; margin-bottom: 0.8rem; display: flex; justify-content: space-between; align-items: center; }
.pass { color: #00e676; } .fail { color: #ff5252; } .partial { color: #ffab40; }
.badge { font-size: 0.68rem; font-family: 'IBM Plex Mono', monospace; padding: 2px 8px; border-radius: 2px; text-transform: uppercase; letter-spacing: 1px; margin-left: 8px; }
.badge-pass { background: rgba(0,230,118,0.12); color: #00e676; }
.badge-fail { background: rgba(255,82,82,0.12); color: #ff5252; }
.badge-partial { background: rgba(255,171,64,0.12); color: #ffab40; }
.rec-box { margin-top: 1.5rem; padding: 1.2rem 1.8rem; border-radius: 6px; border-left: 4px solid; text-align: center; }
.rec-strong-buy { background: rgba(0,230,118,0.08); border-color: #00e676; }
.rec-pullback { background: rgba(255,171,64,0.08); border-color: #ffab40; }
.rec-avoid { background: rgba(255,82,82,0.08); border-color: #ff5252; }
.rec-label { font-family: 'IBM Plex Mono', monospace; font-size: 1.4rem; font-weight: 600; letter-spacing: 3px; text-transform: uppercase; }
.rec-sub { font-size: 0.78rem; color: #4a6580; margin-top: 4px; }
.metric-strip { background: #0f1724; border: 1px solid #1e2d40; border-radius: 6px; padding: 1rem 1.5rem; margin-bottom: 1.5rem; display: flex; gap: 2.5rem; flex-wrap: wrap; }
.metric-label { font-size: 0.68rem; color: #3a5570; text-transform: uppercase; letter-spacing: 1px; font-family: 'IBM Plex Mono', monospace; }
.metric-value { font-size: 1rem; font-family: 'IBM Plex Mono', monospace; font-weight: 600; color: #c8d6e5; margin-top: 2px; }
.section-head { font-size: 0.7rem; letter-spacing: 2px; text-transform: uppercase; color: #3a5570; font-family: 'IBM Plex Mono', monospace; border-bottom: 1px solid #1e2d40; padding-bottom: 6px; margin-bottom: 1rem; margin-top: 1.5rem; }
</style>
""", unsafe_allow_html=True)

def fetch_data(ticker):
    tk = yf.Ticker(ticker)
    df = tk.history(period="1y", auto_adjust=True)
    return (None, None) if df.empty else (df, tk.fast_info)

def score_ema(df):
    c = df["Close"]
    e10, e20, e50, e200 = [c.ewm(span=s, adjust=False).mean().iloc[-1] for s in [10,20,50,200]]
    lc = c.iloc[-1]
    checks = {"EMA10>20": e10>e20, "EMA20>50": e20>e50, "EMA50>200": e50>e200, "Price>EMA200": lc>e200}
    score = (sum(checks.values())/4)*40
    return score, {"score":round(score,1),"max":40,"checks":checks,"values":{"EMA10":round(e10,2),"EMA20":round(e20,2),"EMA50":round(e50,2),"EMA200":round(e200,2),"Price":round(lc,2)}}

def score_rs(df):
    spy = yf.Ticker("SPY").history(period="1y", auto_adjust=True)
    end = df.index[-1]
    start = end - timedelta(days=90)
    df3 = df[df.index>=start]["Close"]
    s3 = spy[spy.index>=start]["Close"]
    common = df3.index.intersection(s3.index)
    if len(common)<20: return 0, {"score":0,"max":30,"note":"數據不足"}
    df3, s3 = df3.loc[common], s3.loc[common]
    rs = (df3/df3.iloc[0])/(s3/s3.iloc[0])
    slope,_ = np.polyfit(np.arange(len(rs)), rs.values, 1)
    rs_ret = rs.iloc[-1]-1.0
    if slope>0:
        score = min(slope*500,15)+min(max(rs_ret*100,0),15)
    else:
        score = max(0, 10+slope*300)
    score = max(0,min(score,30))
    return score, {"score":round(score,1),"max":30,"rs_return_pct":round(rs_ret*100,2),"slope":round(slope,6),"slope_positive":slope>0,"rs_series":rs}

def score_vcp(df):
    recent = df.tail(60)
    if len(recent)<40: return 0, {"score":0,"max":20,"note":"數據不足"}
    def vol(w): return ((w["High"]-w["Low"])/w["Close"]).mean()*100
    v1,v2,v3 = vol(recent.iloc[:20]),vol(recent.iloc[20:40]),vol(recent.iloc[40:])
    c12,c23 = v2<v1, v3<v2
    cp = (v1-v3)/v1*100 if v1>0 else 0
    if c12 and c23: score = min(15+cp*0.5,20)
    elif v3<v1: score = min(10+cp*0.3,16)
    elif c12 or c23: score = 8
    else: score = max(0,5-abs(cp)*0.2)
    last10,prev10 = recent.tail(10),recent.iloc[30:40]
    r10 = (last10["High"].max()-last10["Low"].min())/last10["Close"].mean()*100
    rp = (prev10["High"].max()-prev10["Low"].min())/prev10["Close"].mean()*100
    score = min(score+(2 if r10<rp*0.7 else 0),20)
    return score, {"score":round(score,1),"max":20,"v1":round(v1,2),"v2":round(v2,2),"v3":round(v3,2),"contracting_1_2":c12,"contracting_2_3":c23,"contraction_pct":round(cp,1)}

def score_volume(df):
    vol,close = df["Volume"],df["Close"]
    avg50 = vol.rolling(50).mean().iloc[-1]
    lv,lc = vol.iloc[-1],close.iloc[-1]
    r5v,r5c = vol.tail(5),close.tail(5)
    up = r5c.diff()>0; down = r5c.diff()<=0
    up_vol = r5v[up].mean() if up.any() else 0
    ratio = up_vol/avg50 if avg50>0 else 1
    hb = (lv>avg50*1.5)and(lc>close.iloc[-2])
    lb = (lv<avg50*0.8)and(lc<close.iloc[-2])
    if hb: score,pat = 10,"帶量突破"
    elif lb: score,pat = 8,"縮量回踩"
    elif ratio>1.2: score,pat = 7,"上升放量"
    elif ratio>0.9: score,pat = 5,"成交量正常"
    else: score,pat = 3,"成交量偏弱"
    return score, {"score":round(score,1),"max":10,"last_vol":int(lv),"avg_vol_50":int(avg50) if not np.isnan(avg50) else 0,"vol_ratio":round(lv/avg50,2) if avg50>0 else 0,"pattern":pat}

def get_rec(t):
    if t>=75: return "Strong Buy","技術面強勢，適合進場或加碼","rec-strong-buy","#00e676"
    elif t>=50: return "Pullback Watch","等待回踩確認後進場","rec-pullback","#ffab40"
    else: return "Avoid","技術面偏弱，暫時觀望","rec-avoid","#ff5252"

def make_gauge(score):
    c = "#00e676" if score>=75 else "#ffab40" if score>=50 else "#ff5252"
    fig = go.Figure(go.Indicator(mode="gauge+number",value=score,
        number={"font":{"family":"IBM Plex Mono","size":52,"color":c}},
        gauge={"axis":{"range":[0,100],"tickfont":{"family":"IBM Plex Mono","size":11,"color":"#3a5570"},"dtick":25},
               "bar":{"color":c,"thickness":0.25},"bgcolor":"#0f1724","borderwidth":0,
               "steps":[{"range":[0,100],"color":"#0f1724"}]}))
    fig.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0a0e17",margin=dict(t=20,b=0,l=20,r=20),height=240)
    return fig

def make_rs_chart(rs):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=rs.index,y=rs.values,mode="lines",line=dict(color="#00d9ff",width=1.5),fill="tozeroy",fillcolor="rgba(0,217,255,0.05)"))
    fig.add_hline(y=1.0,line_dash="dash",line_color="#1e2d40",line_width=1)
    fig.update_layout(paper_bgcolor="#0a0e17",plot_bgcolor="#0f1724",margin=dict(t=10,b=10,l=10,r=10),height=120,
        xaxis=dict(showgrid=False,showticklabels=False),yaxis=dict(showgrid=True,gridcolor="#1e2d40",tickfont=dict(family="IBM Plex Mono",size=10,color="#3a5570")),showlegend=False)
    return fig

def pct_class(s,m):
    p=s/m
    if p>=0.75: return "pass","badge-pass","PASS"
    elif p>=0.40: return "partial","badge-partial","PARTIAL"
    else: return "fail","badge-fail","FAIL"

def fmt_vol(n):
    if n>=1_000_000: return f"{n/1_000_000:.1f}M"
    elif n>=1_000: return f"{n/1_000:.0f}K"
    return str(n)

st.markdown('<div style="border-bottom:1px solid #1e2d40;padding-bottom:1rem;margin-bottom:2rem"><span style="font-family:IBM Plex Mono,monospace;font-size:1.6rem;font-weight:600;color:#00d9ff">📈 美股強勢股</span> <span style="font-size:0.8rem;color:#4a6580;font-family:IBM Plex Mono,monospace;text-transform:uppercase;letter-spacing:2px">Technical Scoring System</span></div>', unsafe_allow_html=True)

c1,c2,c3 = st.columns([3,1.2,4])
with c1: ticker_input = st.text_input("TICKER", value="", placeholder="e.g. NVDA, TSLA, AAPL")
with c2:
    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    btn = st.button("▶  分析", use_container_width=True)

st.markdown("<hr style='border-color:#1e2d40'>", unsafe_allow_html=True)

if btn and ticker_input.strip():
    ticker = ticker_input.strip().upper()
    with st.spinner(f"獲取 {ticker} 數據..."):
        df, info = fetch_data(ticker)
    if df is None or len(df)<60:
        st.error(f"❌ 無法獲取 {ticker} 數據，請確認代號。")
        st.stop()
    with st.spinner("計算評分..."):
        es,ed = score_ema(df); rs,rd = score_rs(df); vs,vd = score_vcp(df); vols,vold = score_volume(df)
    total = es+rs+vs+vols
    rl,rsub,rcls,rc = get_rec(total)

    lp,pp = df["Close"].iloc[-1],df["Close"].iloc[-2]
    chg = lp-pp; chgp = chg/pp*100
    h52 = df["Close"].tail(252).max(); l52 = df["Close"].tail(252).min()
    dist = (lp-h52)/h52*100

    left,right = st.columns([1.1,1],gap="large")
    with left:
        chg_cls = "color:#00e676" if chg>=0 else "color:#ff5252"
        sign = "+" if chg>=0 else ""
        st.markdown(f'<div class="metric-strip"><div><div class="metric-label">Ticker</div><div class="metric-value" style="color:#00d9ff;letter-spacing:3px">{ticker}</div></div><div><div class="metric-label">Last Price</div><div class="metric-value">${lp:.2f} <span style="{chg_cls};font-size:0.8rem">{sign}{chg:.2f} ({sign}{chgp:.1f}%)</span></div></div><div><div class="metric-label">52W High</div><div class="metric-value">${h52:.2f}</div></div><div><div class="metric-label">From High</div><div class="metric-value" style="{"color:#ff5252" if dist<-15 else "color:#00e676"}">{dist:+.1f}%</div></div></div>', unsafe_allow_html=True)
        st.markdown('<div class="section-head">技術評分</div>', unsafe_allow_html=True)
        st.plotly_chart(make_gauge(round(total,1)), use_container_width=True, config={"displayModeBar":False})
        st.markdown(f'<div class="rec-box {rcls}"><div class="rec-label" style="color:{rc}">{rl}</div><div class="rec-sub">{rsub}</div></div>', unsafe_allow_html=True)

    with right:
        st.markdown('<div class="section-head">細項評分明細</div>', unsafe_allow_html=True)
        ec,eb,et = pct_class(es,40)
        checks_html = " · ".join([f'<span class="{"pass" if v else "fail"}">{k}</span>' for k,v in ed["checks"].items()])
        st.markdown(f'<div class="score-card"><div><div style="font-size:0.85rem;color:#7a9ab5;text-transform:uppercase;letter-spacing:1px">📐 EMA 排列</div><div style="font-size:0.72rem;color:#3a5570;font-family:IBM Plex Mono,monospace;margin-top:3px">{checks_html}</div></div><div style="text-align:right"><span class="pts {ec}" style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600">{es:.0f}<span style="font-size:0.7rem;color:#3a5570"> / 40</span></span><span class="badge {eb}">{et}</span></div></div>', unsafe_allow_html=True)
        rc2,rb2,rt2 = pct_class(rs,30)
        st.markdown(f'<div class="score-card"><div><div style="font-size:0.85rem;color:#7a9ab5;text-transform:uppercase;letter-spacing:1px">📊 RS 相對強度</div><div style="font-size:0.72rem;color:#3a5570;font-family:IBM Plex Mono,monospace;margin-top:3px">斜率 {"↑" if rd.get("slope_positive") else "↓"} {rd.get("slope",0):.5f} · 相對回報 {rd.get("rs_return_pct",0):+.1f}%</div></div><div style="text-align:right"><span style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600" class="{rc2}">{rs:.0f}<span style="font-size:0.7rem;color:#3a5570"> / 30</span></span><span class="badge {rb2}">{rt2}</span></div></div>', unsafe_allow_html=True)
        if "rs_series" in rd: st.plotly_chart(make_rs_chart(rd["rs_series"]), use_container_width=True, config={"displayModeBar":False})
        vc,vb,vt = pct_class(vs,20)
        st.markdown(f'<div class="score-card"><div><div style="font-size:0.85rem;color:#7a9ab5;text-transform:uppercase;letter-spacing:1px">🔺 VCP 波幅收窄</div><div style="font-size:0.72rem;color:#3a5570;font-family:IBM Plex Mono,monospace;margin-top:3px">W1 {vd.get("v1",0):.1f}% → W2 {vd.get("v2",0):.1f}% → W3 {vd.get("v3",0):.1f}% · 收窄 {vd.get("contraction_pct",0):.1f}%</div></div><div style="text-align:right"><span style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600" class="{vc}">{vs:.0f}<span style="font-size:0.7rem;color:#3a5570"> / 20</span></span><span class="badge {vb}">{vt}</span></div></div>', unsafe_allow_html=True)
        vlc,vlb,vlt = pct_class(vols,10)
        st.markdown(f'<div class="score-card"><div><div style="font-size:0.85rem;color:#7a9ab5;text-transform:uppercase;letter-spacing:1px">📦 成交量形態</div><div style="font-size:0.72rem;color:#3a5570;font-family:IBM Plex Mono,monospace;margin-top:3px">{vold.get("pattern","--")} · {fmt_vol(vold.get("last_vol",0))} · 比率 {vold.get("vol_ratio",0):.2f}x</div></div><div style="text-align:right"><span style="font-family:IBM Plex Mono,monospace;font-size:1rem;font-weight:600" class="{vlc}">{vols:.0f}<span style="font-size:0.7rem;color:#3a5570"> / 10</span></span><span class="badge {vlb}">{vlt}</span></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="background:#0f1724;border:1px solid #2a3f58;border-radius:6px;padding:0.9rem 1.4rem;margin-top:0.5rem;display:flex;justify-content:space-between;align-items:center"><div style="font-size:0.8rem;color:#7a9ab5;text-transform:uppercase;letter-spacing:2px;font-family:IBM Plex Mono,monospace">總分 Total Score</div><div style="font-family:IBM Plex Mono,monospace;font-size:1.6rem;font-weight:700;color:{"#00e676" if total>=75 else "#ffab40" if total>=50 else "#ff5252"}">{total:.1f} <span style="font-size:0.9rem;color:#3a5570">/ 100</span></div></div>', unsafe_allow_html=True)

elif btn:
    st.warning("請輸入股票代號再按分析。")
else:
    st.markdown('<div style="text-align:center;padding:4rem 2rem;color:#2a3f58"><div style="font-size:3rem">📈</div><div style="font-family:IBM Plex Mono,monospace;font-size:1rem;letter-spacing:2px;text-transform:uppercase;margin-top:1rem">輸入美股代號開始分析</div><div style="font-size:0.78rem;margin-top:0.8rem">EMA排列 · RS強度 · VCP形態 · 成交量 · 100分制評分</div></div>', unsafe_allow_html=True)
