import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def backtest_portfolio(tickers, weights, start_date, end_date,
                       one_off=0, dca=0, dca_freq='M', rebalance_freq='M'):
    prices = yf.download(tickers, start=start_date, end=end_date,
                         auto_adjust=True)['Close']
    returns = prices.pct_change().fillna(0)
    dates = prices.index

    cash_flow = pd.Series(0, index=dates)
    cash_flow.iloc[0] += one_off
    for d in prices.resample(dca_freq).first().index:
        cash_flow.loc[d] += dca

    w = np.array(weights)
    portfolio = pd.Series(index=dates, dtype=float)
    portfolio.iloc[0] = cash_flow.iloc[0]
    rebalance_dates = [grp.index[-1]
        for _, grp in prices.groupby(pd.Grouper(freq=rebalance_freq))]

    for i in range(1, len(dates)):
        portfolio.iloc[i] = (portfolio.iloc[i-1] * (1 + np.dot(returns.iloc[i], w))
                             + cash_flow.iloc[i])
        if dates[i] in rebalance_dates:
            total = portfolio.iloc[i]
            w = np.array(weights)
            portfolio.iloc[i] = total
    return portfolio

st.title("ETF 資產回測工具")

tickers = st.text_input("標的 (逗號分隔)", "2330.TW,00850.TW,VOO,AAPL")
weights = st.text_input("權重 (加總=1)", "0.4,0.3,0.2,0.1")

start = st.date_input("開始日", pd.to_datetime("2015-01-01"))
end   = st.date_input("結束日",   pd.to_datetime("2025-06-07"))
one_off = st.number_input("一次性投入", 10000.0)
dca     = st.number_input("定期定額", 1000.0)
dca_freq = st.selectbox("DCA 週期", ["D","W","M","Q","A"])
rebalance_freq = st.selectbox("重平衡週期", ["D","W","M","Q","A"])

if st.button("執行回測"):
    ts = [t.strip() for t in tickers.split(',')]
    ws = list(map(float, weights.split(',')))
    pf = backtest_portfolio(
        ts, ws,
        start.strftime('%Y-%m-%d'),
        end.strftime('%Y-%m-%d'),
        one_off=one_off,
        dca=dca,
        dca_freq=dca_freq,
        rebalance_freq=rebalance_freq
    )

    fig, ax = plt.subplots()
    ax.plot(pf.index, pf.values, label="Portfolio")
    ax.set_title("累積績效")
    ax.set_xlabel("日期"); ax.set_ylabel("價值")
    st.pyplot(fig)

    total_ret = pf.iloc[-1]/pf.iloc[0] - 1
    ann_ret   = (1+total_ret)**(252/len(pf)) - 1
    max_dd    = (pf/pf.cummax() - 1).min()
    st.write("**期間累積報酬**：", f"{total_ret*100:.2f}%")
    st.write("**年化報酬率**：", f"{ann_ret*100:.2f}%")
    st.write("**最大回撤**：",   f"{max_dd*100:.2f}%")
