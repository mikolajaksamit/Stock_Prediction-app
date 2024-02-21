import streamlit as st
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import ta
from plotly import graph_objects as go

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App")

stocks = ("AAPL", "META", "GOOG", "MSFT", "BTC-USD", "ETH-USD")

selected_stock = st.selectbox("Select dataset for prediction", stocks)

n_years = st.slider("Year of prediction", 1, 9)
period = n_years * 365

@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data
data_load_state = st.text("Load data...")
data = load_data(selected_stock)
data_load_state.text("Loading data complete!")

st.subheader("Raw data")
st.write(data.tail())

# Oblicz wskaźniki techniczne: MACD i RSI
data['MACD'] = ta.trend.macd_diff(data['Close'])
data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
data['SMA'] = ta.trend.sma_indicator(data['Close'], window=20)  # SMA 20

fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], name="SMA", line=dict(color="yellow")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["MACD"], name="MACD", line=dict(color="red")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="green")))
fig.layout.update(title_text="Technical Indicators", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="green")))
fig_rsi.layout.update(title_text="RSI (14)", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_rsi)
#forecasting

df_train = data[["Date", "Close"]]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

m = Prophet()
m.fit(df_train)

future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Forecast data")
st.write(forecast.tail())

st.write('Forecast data')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write('Forecast components')
fig2 = m.plot_components(forecast)
st.write(fig2)
