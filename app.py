import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

st.title("ETF 回測（台股＋美股 → 全部以 TWD 計）")

def backtest_portfolio(
    tickers, weights, start_date, end_date,
    one_off=0, dca=0, dca_freq='M', rebalance_freq='M'
):
    # 1) 下載價格，只要 Close
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)
    prices = raw["Close"].copy()  # DataFrame: index=日期, columns=tickers

    # 2) 美股換算成 TWD
    usd_cols = [t for t in prices.columns if not t.endswith(".TW")]
    if usd_cols:
        # 下載美元兌台幣匯率，取 Close → 一維 Series
        fx = yf.download("TWD=X", start=start_date, end=end_date)["Close"]
        # 跟 prices 同步 index，向前填值
        fx = fx.reindex(prices.index).ffill()
        # 用 pandas multiply（axis=0 表示以 index 對齊）
        prices[usd_cols] = prices[usd_cols].multiply(fx, axis="index")

    # 3) 日報酬
    returns = prices.pct_change().fillna(0)
    dates = prices.index

    # 4) 現金流：一次性 + 定期定額
    cash_flow = pd.Series(0.0, index=dates)
    cash_flow.iloc[0] = one_off
    dca_dates = prices.resample(dca_freq).first().index
    for d in dca_dates:
        if d in cash_flow.index:
            cash_flow.loc[d] += dca

    # 5) 回測主程式
    w = np.array(weights)
    pf = pd.Series(index=dates, dtype=float)
    pf.iloc[0] = cash_flow.iloc[0]
    rebalance_dates = [
        grp.index[-1]
        for _, grp in prices.groupby(pd.Grouper(freq=rebalance_freq))
    ]

    for i in range(1, len(dates)):
        pf.iloc[i] = (
            pf.iloc[i-1] * (1 + returns.iloc[i].dot(w))
            + cash_flow.iloc[i]
        )
        if dates[i] in rebalance_dates:
            total = pf.iloc[i]
            w = np.array(weights)
            pf.iloc[i] = total

    return pf, prices, returns

# ───────────────────────────────────────────────────
# 參數輸入
tickers_txt = st.text_input("標的（逗號分隔）", "2330.TW,00850.TW,VOO,AAPL")
weights_txt = st.text_input("權重（加總=1）",       "0.4,0.3,0.2,0.1")
start = st.date_input("開始日", pd.to_datetime("2015-01-01"))
end   = st.date_input("結束日",   pd.to_datetime("2025-06-07"))
one_off = st.number_input("一次性投入（TWD）", 10000.0, step=1000.0)
dca     = st.number_input("定期定額（TWD）",  1000.0,  step=100.0)
dca_freq       = st.selectbox("DCA 週期", ["D","W","M","Q","A"], index=2)
rebalance_freq = st.selectbox("重平衡週期", ["D","W","M","Q","A"], index=2)

if st.button("執行回測"):
    ts = [t.strip() for t in tickers_txt.split(",")]
    ws = list(map(float, weights_txt.split(",")))

    pf, prices, returns = backtest_portfolio(
        ts, ws,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        one_off=one_off,
        dca=dca,
        dca_freq=dca_freq,
        rebalance_freq=rebalance_freq
    )

    # 資產配置圓餅圖
    fig1, ax1 = plt.subplots()
    ax1.pie(ws, labels=ts, autopct="%.1f%%")
    ax1.set_title("資產配置（TWD）")
    st.pyplot(fig1)

    # 累積績效 + 關鍵指標
    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(pf.index, pf.values, label="Portfolio (TWD)")
    ax2.set_title("累積績效")
    ax2.set_xlabel("日期"); ax2.set_ylabel("台幣價值")
    st.pyplot(fig2)

    total_ret = pf.iloc[-1] / pf.iloc[0] - 1
    ann_ret   = (1 + total_ret) ** (252 / len(pf)) - 1
    max_dd    = (pf / pf.cummax() - 1).min()
    st.write(f"**期間累積報酬**：{total_ret*100:.2f}%")
    st.write(f"**年化報酬率**：{ann_ret*100:.2f}%")
    st.write(f"**最大回撤**：{max_dd*100:.2f}%")

    # 月度報酬熱力圖
    monthly = prices.resample("M").last().pct_change().fillna(0)
    dfm = monthly.copy()
    dfm.index = pd.to_datetime(dfm.index)
    dfm["Year"], dfm["Month"] = dfm.index.year, dfm.index.month
    heat = dfm.pivot("Year", "Month", ts)

    fig3, ax3 = plt.subplots(figsize=(10,6))
    sns.heatmap(heat, center=0, cmap="RdYlGn", ax=ax3,
                cbar_kws={"format":"%.0f%%"})
    ax3.set_title("月度報酬熱力圖")
    st.pyplot(fig3)
