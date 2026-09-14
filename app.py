import streamlit as st
import yfinance as yf
import pandas as pd

# Konfigurasi Halaman
st.set_page_config(page_title="Scalping Dashboard IDX Real-Time", layout="wide")

st.title("⚡ Dashboard Scalping Saham Indonesia (IDX)")
st.markdown("Analisis teknikal cepat (*MACD, RSI, Bollinger Bands*) untuk strategi *scalping* saham harian.")

# Daftar Emiten Utama IDX
@st.cache_data(ttl=3600)
def get_idx_universe():
    return [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", 
        "ASII.JK", "GOTO.JK", "ADRO.JK", "PTBA.JK", "ANTM.JK", 
        "MDKA.JK", "UNTR.JK", "ARTO.JK", "BRIS.JK", "PGAS.JK"
    ]

all_tickers = get_idx_universe()

# Sidebar Input
st.sidebar.header("Pengaturan Scalping")
selected_ticker = st.sidebar.selectbox("Pilih Emiten:", all_tickers)
custom_ticker = st.sidebar.text_input("Atau Ketik Kode Saham:", value="")

target_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_ticker
if not target_ticker.endswith(".JK") and target_ticker:
    target_ticker += ".JK"

if st.sidebar.button("🔄 Perbarui Data"):
    st.rerun()

# Fungsi Ambil Data Intraday (Interval 5 Menit untuk Scalping)
@st.cache_data(ttl=30)
def fetch_scalping_data(ticker):
    stock = yf.Ticker(ticker)
    # Mengambil data interval 5 menit untuk rentang waktu 5 hari terakhir
    df = stock.history(period="5d", interval="5m")
    info = stock.info
    return df, info

try:
    with st.spinner(f"Memproses indikator teknikal untuk {target_ticker}..."):
        df, info = fetch_scalping_data(target_ticker)
        
    if not df.empty and len(df) > 26:
        # Kalkulasi Indikator Teknikal
        # 1. RSI 14
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))

        # 2. MACD (12, 26, 9)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        # 3. Bollinger Bands (20, 2)
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        std = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (std * 2)

        current_price = df['Close'].iloc[-1]
        prev_close = info.get('previousClose', df['Close'].iloc[-2])
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close else 0
        current_rsi = df['RSI'].iloc[-1]
        current_macd = df['MACD'].iloc[-1]
        current_signal = df['Signal_Line'].iloc[-1]

        # Metrik Atas
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Harga Real-Time", f"Rp {current_price:,.2f}", f"{pct_change:.2f}%")
        c2.metric("RSI (14)", f"{current_rsi:.2f}", "Overbought >70 | Oversold <30" if current_rsi > 70 or current_rsi < 30 else "Normal")
        
        signal_status = "BULLISH 🟢" if current_macd > current_signal else "BEARISH 🔴"
        c3.metric("Sinyal MACD", signal_status)
        c4.metric("Volume Terakhir", f"{int(df['Volume'].iloc[-1]):,}")

        # Rekomendasi Aksi Scalping Cepat
        st.subheader("💡 Indikasi Sinyal Scalper Cepat")
        if current_rsi < 35 and current_macd > current_signal:
            st.success("🟢 **Peluang BUY (Scalping):** RSI berada di area *oversold* dan momentum MACD mulai berbalik naik.")
        elif current_rsi > 68:
            st.warning("⚠️ **Peluang SELL / Take Profit:** RSI mendekati area *overbought*, waspadai potensi koreksi cepat.")
        else:
            st.info("ℹ️ **Konsolidasi / Wait and See:** Belum ada sinyal ekstrem yang valid untuk eksekusi kilat.")

        # Grafik Harga & Bollinger Bands
        st.subheader("📈 Grafik Harga & Bollinger Bands (Interval 5 Menit)")
        st.line_chart(df[['Close', 'BB_Upper', 'BB_Middle', 'BB_Lower']])

        # Grafik RSI
        st.subheader("📉 Indikator RSI (14)")
        st.line_chart(df[['RSI']])

    else:
        st.warning("Data intraday tidak mencukupi untuk menghitung indikator teknikal.")

except Exception as e:
    st.error(f"Terjadi kesalahan teknis: {e}")

# Tautan Platform Eksternal
st.markdown("---")
st.subheader("🔗 Cek Grafik Lanjutan di Platform Utama")
clean_sym = target_ticker.replace(".JK", "")

col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.markdown(f"[TradingView](https://www.tradingview.com/chart/?symbol=IDX:{clean_sym})")
with col_b:
    st.markdown(f"[Yahoo Finance](https://finance.yahoo.com/quote/{target_ticker})")
with col_c:
    st.markdown(f"[Investing.com](https://www.investing.com/equities/{clean_sym.lower()}-indonesia)")
with col_d:
    st.markdown(f"[IDX Resmi](https://www.idx.co.id/)")
