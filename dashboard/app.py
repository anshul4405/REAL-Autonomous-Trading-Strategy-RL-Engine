import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    st.set_page_config(page_title="RL Trading Dashboard", layout="wide")
    st.title("Autonomous Streamlit Trading Dashboard")
    
    st.sidebar.header("Agent Configuration")
    ticker = st.sidebar.text_input("Ticker Symbol", value="AAPL")
    mode = st.sidebar.radio("Mode", ["Paper Trading", "Backtest Viewer"])
    
    st.markdown(f"**Tracking Asset:** {ticker} | **Current Status:** Actively Monitoring")

    # Top Level Metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(label="Portfolio Value", value="$10,500.20", delta="+5.0%")
    with col2:
        st.metric(label="Current Position", value="Long 50 Shares")
    with col3:
        st.metric(label="Latest Agent Action", value="HOLD", delta_color="off")
    with col4:
        st.metric(label="Risk Trades Today", value="2 / 5")
        
    st.subheader("Performance Chart")
    
    # Placeholder for dynamic plotting
    dates = pd.date_range(start="2023-01-01", periods=100)
    values = np.cumsum(np.random.randn(100) * 100) + 10000
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dates, values, label="Net Worth", color="green")
    ax.set_title("Simulated Portfolio Growth")
    ax.set_ylabel("USD")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    st.pyplot(fig)
    
    st.subheader("Recent Trades Log")
    trade_data = pd.DataFrame({
        "Date": ["2023-04-01", "2023-04-05", "2023-04-10"],
        "Action": ["BUY", "SELL", "BUY"],
        "Price": ["$150.20", "$155.00", "$153.10"],
        "Shares": [10, 10, 15],
        "Reason": ["Agent Policy", "Take Profit Hit", "Agent Policy"]
    })
    st.dataframe(trade_data, use_container_width=True)

if __name__ == "__main__":
    main()
