import streamlit as st
import yfinance as yf
import pandas as pd

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Swing Trading IDX", 
    layout="wide"
)

st.title("📈 Dashboard Swing Trading & Indikator Teknikal Saham IDX")
st.markdown(
    "Analisis teknikal *real-time* khusus strategi *swing trading* saham Indonesia terintegrasi dengan "
    "**Yahoo Finance**, **IDX**, **TradingView**, dan **Investing.com**."
)

# Daftar emiten utama IDX
@st.cache_data(ttl=3600)
def get_idx_universe():
    return [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", 
        "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "GOTO.JK", 
        "ADRO.JK", "PTBA.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", 
        "KLBF.JK", "SMGR.JK", "CPIN.JK", "INKP.JK", "MEDC.JK",
        "ARTO.JK", "BRIS.JK", "PGAS.JK", "BUKA.JK", "JSMR.JK"
    ]

all_tickers = get_idx_universe()

# Sidebar Navigasi dan Pencarian
st.sidebar.header("🔍 Pengaturan Analisis")
selected_ticker = st.sidebar.selectbox("Pilih Emiten Populer:", all_tickers)
custom_ticker = st.sidebar.text_input("Atau Ketik Kode Saham (contoh: BBCA):", value="")

target_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_ticker
if not target_ticker.endswith(".JK") and target_ticker:
    target_ticker += ".JK"

if st.sidebar.button("🔄 Perbarui Data"):
    st.rerun()

# Fungsi Mengambil Data Historis (Periode 6 Bulan untuk Analisis Swing)
@st.cache_data(ttl=60)
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo", interval="1d")
    info = stock.info
    return df, info

try:
    with st.spinner(f"Menghitung indikator teknikal untuk {target_ticker}..."):
        df, info = fetch_stock_data(target_ticker)
        
    if not df.empty:
        # Perhitungan Indikator Teknikal (MA, RSI, MACD)
        df['MA20'] = df['Close'].rolling(window=20).mean()
        df['MA50'] = df['Close'].rolling(window=50).mean()
        
        # RSI 14 Periode
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()

        current_price = df['Close'].iloc[-1]
        prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else current_price)
        change = current_price - prev_close
        pct_change = (change / prev_close) * 100 if prev_close else 0

        # Metrik Atas
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Harga Terakhir", f"Rp {current_price:,.2f}", f"{pct_change:.2f}%")
        col2.metric("RSI (14)", f"{df['RSI'].iloc[-1]:.2f}")
        col3.metric("MA 20", f"Rp {df['MA20'].iloc[-1]:,.2f}")
        col4.metric("MA 50", f"Rp {df['MA50'].iloc[-1]:,.2f}")

        # Kotak Analisis Sinyal Swing Trading Otomatis
        st.subheader("💡 Sinyal & Analisis Swing Trading")
        rsi_val = df['RSI'].iloc[-1]
        ma20_val = df['MA20'].iloc[-1]
        ma50_val = df['MA50'].iloc[-1]
        
        signal_box = st.container()
        with signal_box:
            if current_price > ma20_val and ma20_val > ma50_val and rsi_val < 70:
                st.success("**Sinyal: POTENSI BUY (Uptrend / Golden Cross Trend)** - Harga berada di atas MA20 & MA50 dengan RSI belum jenuh beli (Overbought).")
            elif rsi_val > 70:
                st.warning("**Sinyal: OVERBOUGHT (Waspada Koreksi)** - RSI di atas 70, indikasi harga sudah naik terlalu tinggi dalam jangka pendek.")
            elif rsi_val < 30:
                st.info("**Sinyal: OVERSOLD (Potensi Rebound)** - RSI di bawah 30, perhatikan peluang pantulan harga (*rebound*).")
            else:
                st.info("**Sinyal: NEUTRAL / WAIT & SEE** - Tren harga berkonsolidasi, tunggu konfirmasi volume atau perlintasan indikator.")

        # Grafik Harga dan Moving Average
        st.subheader(f"📊 Grafik Harga & Tren MA (Moving Average): {target_ticker}")
        chart_data = df[['Close', 'MA20', 'MA50']]
        st.line_chart(chart_data)

        # Grafik RSI Terpisah untuk Cek Kejenuhan Pasar
        st.subheader("📉 Indikator RSI (Relative Strength Index)")
        st.line_chart(df[['RSI']])
        st.caption("Catatan: RSI di atas 70 mengindikasikan jenuh beli (rawan turun), di bawah 30 mengindikasikan jenuh jual (potensi naik).")

    else:
        st.warning("Data saham tidak ditemukan.")

except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data teknikal: {e}")

# Tautan Cek Platform Eksternal
st.markdown("---")
st.subheader("🔗 Akses Cepat Grafik Lanjutan")
clean_sym = target_ticker.replace(".JK", "")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown("**TradingView**")
    st.markdown(f"[Buka Chart TA](https://www.tradingview.com/chart/?symbol=IDX:{clean_sym})")
with c2:
    st.markdown("**Yahoo Finance**")
    st.markdown(f"[Cek Market Info](https://finance.yahoo.com/quote/{target_ticker})")
with c3:
    st.markdown("**Investing.com**")
    st.markdown(f"[Analisis & Berita](https://www.investing.com/equities/{clean_sym.lower()}-indonesia)")
with c4:
    st.markdown("**IDX (Bursa Efek)**")
    st.markdown("[Situs Resmi BEI](https://www.idx.co.id/)")
