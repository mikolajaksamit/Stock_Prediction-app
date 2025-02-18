import streamlit as st
import pandas as pd
from datetime import date
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

# Ustawienia początkowe
START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title("Stock Prediction App 📈")

stocks = ("AAPL", "META", "GOOG", "MSFT", "BTC-USD", "ETH-USD")
selected_stock = st.selectbox("Wybierz akcje do analizy", stocks)

n_years = st.slider("Ilość lat do prognozy", 1, 5)
period = n_years * 365

# Pobieranie danych
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start=START, end=TODAY)
    data.reset_index(inplace=True)
    return data

data_load_state = st.text("Ładowanie danych...")
data = load_data(selected_stock)
data_load_state.text("Dane załadowane!")

st.subheader("Dane historyczne")
st.write(data.tail())

# Obliczanie wskaźników technicznych
def calculate_sma(data, window):
    return data['Close'].rolling(window=window).mean()

def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['SMA'] = calculate_sma(data, window=20)
data['RSI'] = calculate_rsi(data, window=14)

st.subheader("Dane z wskaźnikami technicznymi")
st.write(data.tail())

# Wizualizacja
fig = go.Figure()
fig.add_trace(go.Scatter(x=data["Date"], y=data["Close"], name="Close", line=dict(color="blue")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["SMA"], name="SMA", line=dict(color="yellow")))
fig.add_trace(go.Scatter(x=data["Date"], y=data["RSI"], name="RSI", line=dict(color="green")))
fig.layout.update(title_text="Wskaźniki techniczne", xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

# Przygotowanie danych do Prophet
df_train = data[["Date", "Close"]].rename(columns={"Date": "ds", "Close": "y"})

# Naprawa błędu: konwersja danych
df_train.dropna(inplace=True)  # Usuwanie pustych wartości
df_train["ds"] = pd.to_datetime(df_train["ds"])  # Konwersja dat
df_train["y"] = pd.to_numeric(df_train["y"], errors="coerce")  # Konwersja liczb

# Trenowanie modelu
m = Prophet()
m.fit(df_train)

# Przewidywanie przyszłych wartości
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)

st.subheader("Prognoza")
st.write(forecast.tail())

# Wizualizacja prognozy
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)

st.write("Składowe prognozy")
fig2 = m.plot_components(forecast)
st.write(fig2)
