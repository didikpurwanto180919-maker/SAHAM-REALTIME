import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_percentage_error
import streamlit.components.v1 as components
import plotly.graph_objects as go

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Dashboard Prediksi ML Swing Trading IDX Pro", 
    layout="wide"
)

# Script Auto-Refresh & Notifikasi Alarm Suara Otomatis
components.html(
    """
    <script>
        function playAlertTone() {
            try {
                let ctx = new (window.AudioContext || window.webkitAudioContext)();
                let osc = ctx.createOscillator();
                let gain = ctx.createGain();
                osc.type = 'sine';
                osc.frequency.value = 587.33; // Nada D5
                gain.gain.setValueAtTime(0.1, ctx.currentTime);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 0.3);
            } catch(e) {
                console.log("Audio context blocked by browser policy");
            }
        }
        setTimeout(playAlertTone, 1000);

        setTimeout(function(){
            window.location.reload();
        }, 60000); // Refresh setiap 60 detik
    </script>
    """,
    height=0,
)

st.title("🤖 AI & Machine Learning Presisi Tinggi: Prediksi Saham IDX & Sinyal Aksi")
st.markdown(
    "Dashboard analisis prediktif berbasis *Random Forest Machine Learning* untuk rekomendasi waktu **BELI (BUY)** dan **JUAL (SELL)** "
    "secara presisi, dilengkapi sistem alarm notifikasi real-time."
)

# Memuat Daftar Seluruh Emiten IDX Secara Otomatis
@st.cache_data(ttl=86400)
def get_idx_universe():
    try:
        url = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all_emiten.csv"
        df_emiten = pd.read_csv(url)
        tickers = [str(code).strip().upper() + ".JK" for code in df_emiten['Code'].dropna().unique()]
        return sorted(tickers)
    except Exception:
        return [
            "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", 
            "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "GOTO.JK", 
            "ADRO.JK", "PTBA.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", 
            "KLBF.JK", "SMGR.JK", "CPIN.JK", "INKP.JK", "MEDC.JK",
            "ARTO.JK", "BRIS.JK", "PGAS.JK", "BUKA.JK", "JSMR.JK", "CUAN.JK"
        ]

all_tickers = get_idx_universe()

# Sidebar Navigasi dan Pengaturan Model ML
st.sidebar.header("🔍 Pengaturan Model ML & Data")
selected_target = st.sidebar.selectbox("Pilih Emiten:", all_tickers)
custom_ticker = st.sidebar.text_input("Atau Ketik Kode Saham (contoh: CUAN):", value="")

timeframe_option = st.sidebar.selectbox(
    "Pilih Interval Grafik:", 
    ["1 Hari (Daily - 60 Hari)", "1 Jam (Hourly - 1 Bulan)"]
)

if "1 Hari" in timeframe_option:
    interval_val = "1d"
    period_val = "60d"
else:
    interval_val = "1h"
    period_val = "1mo"

target_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_target
if not target_ticker.endswith(".JK") and target_ticker:
    target_ticker += ".JK"

if st.sidebar.button("🔄 Perbarui & Prediksi Ulang Sekarang"):
    st.cache_data.clear()
    st.rerun()

current_date_str = str(datetime.date.today())
current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

@st.cache_data(ttl=15)
def fetch_stock_data(ticker, period, interval):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)
    info = stock.info
    df = df.dropna().sort_index()
    return df, info

try:
    with st.spinner(f"Menarik data real-time & kalkulasi presisi tinggi untuk {target_ticker}..."):
        df, info = fetch_stock_data(target_ticker, period_val, interval_val)
        
    if not df.empty and len(df) > 35:
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # --- FEATURE ENGINEERING YANG DISINKRONKAN AGAR TIDAK ERROR ---
        df_ml = pd.DataFrame(index=df.index)
        df_ml['Close'] = df['Close']
        df_ml['Volume'] = df['Volume']
        df_ml['MA5'] = df['Close'].rolling(window=5).mean()
        df_ml['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Indikator RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ml['RSI'] = 100 - (100 / (1 + rs))
        
        # Indikator MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df_ml['MACD'] = exp1 - exp2

        # Target Prediksi 3 Hari ke Depan
        df_ml['Prediction_Target'] = df['Close'].shift(-3)

        # Buang semua baris NaN secara serentak untuk memastikan ukuran X dan y persis sama
        df_ml = df_ml.dropna()

        if len(df_ml) < 10:
            st.warning("Data historis bersih setelah kalkulasi indikator terlalu sedikit untuk melatih model ML.")
        else:
            X = df_ml[['MA5', 'MA20', 'Volume', 'RSI', 'MACD']]
            y = df_ml['Prediction_Target']
            
            train_size = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            # Random Forest Regressor untuk presisi non-linear
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred_test = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred_test)
            accuracy_percentage = max(0, 100 - (mape * 100))

            current_price = df['Close'].iloc[-1]
            prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else current_price)
            change = current_price - prev_close
            pct_change = (change / prev_close) * 100 if prev_close else 0

            latest_features = pd.DataFrame({
                'MA5': [df_ml['MA5'].iloc[-1]],
                'MA20': [df_ml['MA20'].iloc[-1]],
                'Volume': [df_ml['Volume'].iloc[-1]],
                'RSI': [df_ml['RSI'].iloc[-1]],
                'MACD': [df_ml['MACD'].iloc[-1]]
            })
            three_day_pred = model.predict(latest_features)[0]
            pred_change = ((three_day_pred - current_price) / current_price) * 100
            current_rsi = df_ml['RSI'].iloc[-1]

            # --- LOGIKA PENENTUAN WAKTU BELI & JUAL PRESISI ---
            if three_day_pred > current_price and current_rsi < 60:
                action_signal = "STRONG BUY (WAKTU BELI UTAMA)"
                target_sell = three_day_pred * 1.025 
                stop_loss = current_price * 0.975   
            elif three_day_pred > current_price:
                action_signal = "HOLD / CAUTION BUY"
                target_sell = three_day_pred
                stop_loss = current_price * 0.97
            else:
                action_signal = "SELL / TAKE PROFIT (WAKTU JUAL)"
                target_sell = current_price
                stop_loss = current_price * 0.97

            # Metrik Atas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harga Real-Time", f"Rp {current_price:,.2f}", f"{pct_change:.2f}%")
            col2.metric("Prediksi ML (3 Hari)", f"Rp {three_day_pred:,.2f}", f"{pred_change:.2f}%")
            col3.metric("RSI (14) Indicator", f"{current_rsi:.2f}")
            col4.metric("Akurasi Model", f"{accuracy_percentage:.2f}% (Valid)")

            # Kotak Peringatan Sinyal Beli & Jual Presisi
            st.subheader("🚨 Alarm Sinyal Eksekusi Trading (Beli & Jual)")
            if "STRONG BUY" in action_signal:
                st.success(f"""
                🔔 **ALARM NOTIFIKASI: SAATNYA BELI (BUY)**  
                - **Rekomendasi Aksi:** Segera lakukan akumulasi pembelian untuk target swing 3 hari ke depan.  
                - **Target Harga Jual (Take Profit):** Rp {target_sell:,.2f}  
                - **Batas Risiko (Stop Loss):** Rp {stop_loss:,.2f}  
                - **Proyeksi Keuntungan:** +{pred_change:.2f}% (Akurasi Model: {accuracy_percentage:.2f}%)
                """)
            elif "SELL" in action_signal:
                st.warning(f"""
                🔔 **ALARM NOTIFIKASI: SAATNYA JUAL / TAKE PROFIT (SELL)**  
                - **Rekomendasi Aksi:** Amankan keuntungan atau keluar pasar untuk menghindari potensi koreksi.  
                - **Target Koreksi ML:** Rp {three_day_pred:,.2f}  
                - **Tingkat Keyakinan Model:** {accuracy_percentage:.2f}%
                """)
            else:
                st.info(f"""
                🔔 **ALARM NOTIFIKASI: WAIT & SEE (KONSOLIDASI)**  
                - **Rekomendasi Aksi:** Tahan posisi atau tunggu konfirmasi volume lonjakan berikutnya.  
                - **Proyeksi Harga 3 Hari:** Rp {three_day_pred:,.2f}
                """)

            # --- GRAFIK PLOTLY DENGAN GARIS PROYEKSI 3 HARI ---
            st.subheader(f"📊 Grafik Perbandingan & Proyeksi Harga 3 Hari Kedepan ({target_ticker})")
            
            last_date = df.index[-1]
            if interval_val == "1d":
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=3)
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=3, freq='h')

            step_diff = (three_day_pred - current_price) / 3
            future_prices = [current_price + step_diff * i for i in range(1, 4)]
            
            projection_x = [last_date] + list(future_dates)
            projection_y = [current_price] + future_prices

            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Close'], 
                mode='lines', 
                name='Harga Aktual (Real-Time)',
                line=dict(color='#1f77b4', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=projection_x, 
                y=projection_y, 
                mode='lines+markers', 
                name='Proyeksi AI Presisi (3 Hari Kedepan)',
                line=dict(color='#2ca02c', width=3, dash='dash'),
                marker=dict(size=9, color='#2ca02c')
            ))
            
            fig.update_layout(
                xaxis=dict(
                    title="Tanggal Perdagangan",
                    range=[df.index[0], future_dates[-1] + pd.Timedelta(days=1 if interval_val=="1d" else 2)]
                ),
                yaxis_title="Harga (IDR)",
                hovermode="x unified",
                margin=dict(l=20, r=20, t=20, b=20),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"🔄 Data & Sinyal Alarm diperbarui secara real-time pada tanggal {current_date_str} pukul {current_time_str} WIB.")

    else:
        st.warning("Data historis tidak mencukupi atau emiten tidak aktif.")

except Exception as e:
    st.error(f"Terjadi kesalahan saat memproses model Machine Learning: {e}")

# Tautan Cek Platform Eksternal
st.markdown("---")
st.subheader("🔗 Akses Cepat Grafik & Sumber Data Lanjutan")
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
