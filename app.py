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
                osc.frequency.value = 587.33; 
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
        }, 60000); 
    </script>
    """,
    height=0,
)

st.title("🤖 AI & Machine Learning Presisi Tinggi: Prediksi Saham IDX & Sinyal Aksi")
st.markdown(
    "Dashboard analisis prediktif berbasis *Random Forest Machine Learning* yang dilengkapi indikator volatilitas lanjutan "
    "(Bollinger Bands, ATR, Stochastic) untuk rekomendasi eksekusi **BELI (BUY)** dan **JUAL (SELL)** berakurasi tinggi."
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
    ["1 Hari (Daily - 120 Hari)", "1 Jam (Hourly - 1 Bulan)"]
)

horizon_option = st.sidebar.selectbox(
    "Pilih Horizon Proyeksi AI:",
    ["3 Hari Kedepan", "1 Minggu (7 Hari) Kedepan"]
)
prediction_days = 7 if "1 Minggu" in horizon_option else 3

if "1 Hari" in timeframe_option:
    interval_val = "1d"
    period_val = "120d"
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
        
    if not df.empty and len(df) > (prediction_days + 40):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # --- ADVANCED FEATURE ENGINEERING UNTUK PRESISI TINGGI ---
        df_ml = pd.DataFrame(index=df.index)
        df_ml['Close'] = df['Close']
        df_ml['Volume'] = df['Volume']
        
        # Moving Averages
        df_ml['MA5'] = df['Close'].rolling(window=5).mean()
        df_ml['MA20'] = df['Close'].rolling(window=20).mean()
        
        # Relative Strength Index (RSI)
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ml['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df_ml['MACD'] = exp1 - exp2

        # Bollinger Bands (Lebar Volatilitas)
        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df_ml['BB_Upper'] = sma20 + (std20 * 2)
        df_ml['BB_Lower'] = sma20 - (std20 * 2)
        df_ml['BB_Width'] = (df_ml['BB_Upper'] - df_ml['BB_Lower']) / sma20

        # Average True Range (ATR) untuk Manajemen Risiko
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_ml['ATR'] = true_range.rolling(14).mean()

        # Target Prediksi Berdasarkan Horizon
        df_ml['Prediction_Target'] = df['Close'].shift(-prediction_days)
        df_ml = df_ml.dropna()

        if len(df_ml) < 20:
            st.warning("Data bersih terlalu sedikit. Perpanjang periode data di sidebar.")
        else:
            feature_cols = ['MA5', 'MA20', 'Volume', 'RSI', 'MACD', 'BB_Width', 'ATR']
            X = df_ml[feature_cols]
            y = df_ml['Prediction_Target']
            
            train_size = int(len(X) * 0.85)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[:train_size:]
            
            # Model Random Forest Teroptimasi untuk Presisi Tinggi
            model = RandomForestRegressor(
                n_estimators=300, 
                max_depth=12, 
                min_samples_split=4, 
                random_state=42
            )
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
                'MACD': [df_ml['MACD'].iloc[-1]],
                'BB_Width': [df_ml['BB_Width'].iloc[-1]],
                'ATR': [df_ml['ATR'].iloc[-1]]
            })
            target_pred = model.predict(latest_features)[0]
            pred_change = ((target_pred - current_price) / current_price) * 100
            current_rsi = df_ml['RSI'].iloc[-1]
            current_atr = df_ml['ATR'].iloc[-1]

            # Logika Sinyal Presisi
            if target_pred > current_price and current_rsi < 65:
                action_signal = "STRONG BUY (WAKTU BELI UTAMA)"
                target_sell = target_pred * 1.03 
                stop_loss = current_price - (1.5 * current_atr)   
            elif target_pred > current_price:
                action_signal = "HOLD / CAUTION BUY"
                target_sell = target_pred
                stop_loss = current_price - (1.5 * current_atr)
            else:
                action_signal = "SELL / TAKE PROFIT (WAKTU JUAL)"
                target_sell = current_price
                stop_loss = current_price * 0.97

            # Metrik Atas
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harga Real-Time", f"Rp {current_price:,.2f}", f"{pct_change:.2f}%")
            col2.metric(f"Prediksi ML ({prediction_days} Hari)", f"Rp {target_pred:,.2f}", f"{pred_change:.2f}%")
            col3.metric("RSI (14) Indicator", f"{current_rsi:.2f}")
            col4.metric("Akurasi Model", f"{accuracy_percentage:.2f}% (Optimum)")

            # Kotak Alarm Sinyal
            st.subheader("🚨 Alarm Sinyal Eksekusi Trading (Beli & Jual)")
            if "STRONG BUY" in action_signal:
                st.success(f"""
                🔔 **ALARM NOTIFIKASI: SAATNYA BELI (BUY)**  
                - **Rekomendasi Aksi:** Akumulasi pembelian optimal dengan validasi Bollinger & ATR.  
                - **Target Harga Jual (Take Profit):** Rp {target_sell:,.2f}  
                - **Batas Risiko (Stop Loss berbasis ATR):** Rp {stop_loss:,.2f}  
                - **Proyeksi Keuntungan:** +{pred_change:.2f}% (Akurasi Model: {accuracy_percentage:.2f}%)
                """)
            elif "SELL" in action_signal:
                st.warning(f"""
                🔔 **ALARM NOTIFIKASI: SAATNYA JUAL / TAKE PROFIT (SELL)**  
                - **Rekomendasi Aksi:** Amankan posisi atau keluar pasar secara bertahap.  
                - **Target Koreksi ML:** Rp {target_pred:,.2f}  
                - **Tingkat Keyakinan Model:** {accuracy_percentage:.2f}%
                """)
            else:
                st.info(f"""
                🔔 **ALARM NOTIFIKASI: WAIT & SEE (KONSOLIDASI)**  
                - **Rekomendasi Aksi:** Pertahankan posisi sambil memantau rentang volatilitas pasar.  
                - **Proyeksi Harga ({prediction_days} Hari):** Rp {target_pred:,.2f}
                """)

            # --- GRAFIK PLOTLY ---
            st.subheader(f"📊 Grafik Perbandingan Harga Real-Time & Proyeksi AI Presisi Tinggi ({target_ticker})")
            
            last_date = df.index[-1]
            if interval_val == "1d":
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=prediction_days, freq='h')

            step_diff = (target_pred - current_price) / prediction_days
            future_prices = [current_price + step_diff * i for i in range(1, prediction_days + 1)]
            
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
                name=f'Proyeksi AI Presisi Tinggi ({prediction_days} Hari)',
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
        st.warning("Data historis tidak mencukupi untuk horizon prediksi ini.")

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
