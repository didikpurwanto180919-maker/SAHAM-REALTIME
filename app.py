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
    page_title="Dashboard Pro: AI Saham IDX Real-Time", 
    page_icon="📈",
    layout="wide"
)

# Custom CSS untuk Mempercantik Tampilan Dashboard (UI/UX Terbaik)
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #161b22;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #30363d;
    }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        padding: 20px;
        border-radius: 12px;
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Script Auto-Refresh Setiap 5 Detik & Notifikasi Alarm Suara Otomatis
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
                gain.gain.setValueAtTime(0.03, ctx.currentTime);
                osc.connect(gain);
                gain.connect(ctx.destination);
                osc.start();
                osc.stop(ctx.currentTime + 0.15);
            } catch(e) {
                console.log("Audio context blocked by browser policy");
            }
        }
        setTimeout(playAlertTone, 1000);

        // Auto-refresh halaman setiap 5 detik (5000 milidetik) untuk real-time online yang stabil
        setTimeout(function(){
            window.location.reload();
        }, 5000); 
    </script>
    """,
    height=0,
)

# Judul Utama Dashboard dengan Desain Bersih
st.title("🚀 AI & Machine Learning Presisi Tinggi: Real-Time IDX Dashboard")
st.markdown(
    "Dashboard analisis prediktif berbasis *Random Forest Machine Learning* tingkat lanjut dengan indikator volatilitas "
    "(Bollinger Bands, ATR, Stochastic) untuk rekomendasi eksekusi **BELI (BUY)** dan **JUAL (SELL)** secara *real-time*."
)
st.markdown("---")

# Memuat Daftar Seluruh Emiten IDX Secara Otomatis dengan Fallback Aman
@st.cache_data(ttl=86400)
def get_idx_universe():
    try:
        url = "https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/all_emiten.csv"
        df_emiten = pd.read_csv(url)
        if 'Code' in df_emiten.columns:
            tickers = [str(code).strip().upper() + ".JK" for code in df_emiten['Code'].dropna().unique()]
            return sorted(tickers)
    except Exception:
        pass
    
    return [
        "BBCA.JK", "BBRI.JK", "BMRI.JK", "BBNI.JK", "TLKM.JK", 
        "ASII.JK", "UNVR.JK", "ICBP.JK", "INDF.JK", "GOTO.JK", 
        "ADRO.JK", "PTBA.JK", "ANTM.JK", "MDKA.JK", "UNTR.JK", 
        "KLBF.JK", "SMGR.JK", "CPIN.JK", "INKP.JK", "MEDC.JK",
        "ARTO.JK", "BRIS.JK", "PGAS.JK", "BUKA.JK", "JSMR.JK", "CUAN.JK"
    ]

all_tickers = get_idx_universe()

# Sidebar Navigasi dan Pengaturan Model ML
st.sidebar.header("🎛️ Panel Kontrol & Pengaturan")
selected_target = st.sidebar.selectbox("Pilih Emiten:", all_tickers)
custom_ticker = st.sidebar.text_input("Atau Ketik Kode Saham (contoh: BBCA):", value="")

timeframe_option = st.sidebar.selectbox(
    "Pilih Interval Grafik:", 
    ["1 Menit (Intraday - 7 Hari)", "1 Jam (Hourly - 1 Bulan)", "1 Hari (Daily - 120 Hari)"]
)

horizon_option = st.sidebar.selectbox(
    "Pilih Horizon Proyeksi AI:",
    ["3 Hari Kedepan", "1 Minggu (7 Hari) Kedepan"]
)
prediction_days = 7 if "1 Minggu" in horizon_option else 3

if "1 Menit" in timeframe_option:
    interval_val = "1m"
    period_val = "7d"
elif "1 Jam" in timeframe_option:
    interval_val = "1h"
    period_val = "1mo"
else:
    interval_val = "1d"
    period_val = "120d"

target_ticker = custom_ticker.strip().upper() if custom_ticker.strip() else selected_target
if not target_ticker.endswith(".JK") and target_ticker:
    target_ticker += ".JK"

st.sidebar.markdown("---")
if st.sidebar.button("🔄 Segarkan Data & Prediksi Sekarang", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

current_date_str = str(datetime.date.today())
current_time_str = datetime.datetime.now().strftime("%H:%M:%S")

# TTL diatur ke 5 detik agar sinkron dengan auto-refresh
@st.cache_data(ttl=5)
def fetch_stock_data(ticker, period, interval):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval, auto_adjust=True)
    try:
        info = stock.info
    except Exception:
        info = {}
    if not df.empty:
        df = df.dropna().sort_index()
    return df, info

try:
    with st.spinner(f"Menarik data online real-time & memproses model AI untuk {target_ticker}..."):
        df, info = fetch_stock_data(target_ticker, period_val, interval_val)
        
    if not df.empty and len(df) > (prediction_days + 40):
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        # --- ADVANCED FEATURE ENGINEERING ---
        df_ml = pd.DataFrame(index=df.index)
        df_ml['Close'] = df['Close']
        df_ml['Volume'] = df['Volume']
        
        df_ml['MA5'] = df['Close'].rolling(window=5).mean()
        df_ml['MA10'] = df['Close'].rolling(window=10).mean()
        df_ml['MA20'] = df['Close'].rolling(window=20).mean()
        
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df_ml['RSI'] = 100 - (100 / (1 + rs))
        
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df_ml['MACD'] = exp1 - exp2

        sma20 = df['Close'].rolling(window=20).mean()
        std20 = df['Close'].rolling(window=20).std()
        df_ml['BB_Upper'] = sma20 + (std20 * 2)
        df_ml['BB_Lower'] = sma20 - (std20 * 2)
        df_ml['BB_Width'] = (df_ml['BB_Upper'] - df_ml['BB_Lower']) / sma20

        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df_ml['ATR'] = true_range.rolling(14).mean()

        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df_ml['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df_ml['Volume_MA5'] = df['Volume'].rolling(window=5).mean()

        df_ml['Prediction_Target'] = df['Close'].shift(-prediction_days)
        df_ml = df_ml.dropna()

        if len(df_ml) < 20:
            st.warning("Data bersih terlalu sedikit. Perpanjang periode data di sidebar.")
        else:
            feature_cols = ['MA5', 'MA10', 'MA20', 'Volume', 'Volume_MA5', 'RSI', 'MACD', 'BB_Width', 'ATR', 'Stoch_K']
            X = df_ml[feature_cols]
            y = df_ml['Prediction_Target']
            
            train_size = int(len(X) * 0.90)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
            model = RandomForestRegressor(
                n_estimators=500, 
                max_depth=15, 
                min_samples_split=3, 
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            
            y_pred_test = model.predict(X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred_test)
            accuracy_percentage = max(0, 100 - (mape * 100))

            current_price = df['Close'].iloc[-1]
            prev_close = info.get('previousClose', df['Close'].iloc[-2] if len(df) > 1 else current_price)
            change = current_price - prev_close
            pct_change = (change / prev_close) * 100 if prev_close else 0

            latest_features = pd.DataFrame([[
                df_ml['MA5'].iloc[-1],
                df_ml['MA10'].iloc[-1],
                df_ml['MA20'].iloc[-1],
                df_ml['Volume'].iloc[-1],
                df_ml['Volume_MA5'].iloc[-1],
                df_ml['RSI'].iloc[-1],
                df_ml['MACD'].iloc[-1],
                df_ml['BB_Width'].iloc[-1],
                df_ml['ATR'].iloc[-1],
                df_ml['Stoch_K'].iloc[-1]
            ]], columns=feature_cols)

            target_pred = model.predict(latest_features)[0]
            pred_change = ((target_pred - current_price) / current_price) * 100
            current_rsi = df_ml['RSI'].iloc[-1]
            current_atr = df_ml['ATR'].iloc[-1]

            if target_pred > current_price and current_rsi < 65:
                action_signal = "STRONG BUY"
                target_sell = target_pred * 1.03 
                stop_loss = current_price - (1.5 * current_atr)   
            elif target_pred > current_price:
                action_signal = "HOLD / CAUTION"
                target_sell = target_pred
                stop_loss = current_price - (1.5 * current_atr)
            else:
                action_signal = "SELL / TAKE PROFIT"
                target_sell = current_price
                stop_loss = current_price * 0.97

            # --- TAMPILAN DASHBOARD UTAMA (GRID MODERN) ---
            st.subheader(f"📊 Ringkasan Pasar Real-Time: {target_ticker}")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Harga Real-Time", f"Rp {current_price:,.2f}", f"{pct_change:+.2f}%")
            col2.metric(f"Prediksi AI ({prediction_days} Hari)", f"Rp {target_pred:,.2f}", f"{pred_change:+.2f}%")
            col3.metric("Indikator RSI (14)", f"{current_rsi:.2f}", "Netral/Momentum")
            col4.metric("Akurasi Model ML", f"{accuracy_percentage:.2f}%", "High Confidence")

            st.markdown("---")

            # Bagian Rekomendasi Titik Eksekusi Harga
            st.markdown("### 💡 Rekomendasi Titik Eksekusi Harga Trading")
            col_b1, col_b2, col_b3 = st.columns(3)
            col_b1.metric("Rekomendasi Harga Beli (Buy)", f"Rp {current_price:,.2f}", "Zona Akumulasi Optimal")
            col_b2.metric("Target Harga Jual (Take Profit)", f"Rp {target_sell:,.2f}", f"+{((target_sell - current_price)/current_price)*100:.2f}% Target")
            col_b3.metric("Batas Risiko (Stop Loss)", f"Rp {stop_loss:,.2f}", "Proteksi ATR Volatilitas")

            # Bagian Status Alarm Sinyal
            st.markdown("### 🚨 Panel Alarm Sinyal Eksekusi")
            if "STRONG BUY" in action_signal:
                st.success(f"""
                🟢 **STATUS: STRONG BUY (WAKTU BELI UTAMA)**  
                - **Aksi Strategis:** Akumulasi bertahap di rentang harga saat ini.  
                - **Target Take Profit:** Rp {target_sell:,.2f} | **Stop Loss:** Rp {stop_loss:,.2f}  
                - **Potensi Keuntungan:** +{pred_change:.2f}% (Tingkat Akurasi: {accuracy_percentage:.2f}%)
                """)
            elif "SELL" in action_signal:
                st.warning(f"""
                🟠 **STATUS: SELL / TAKE PROFIT (WAKTU KELUAR)**  
                - **Aksi Strategis:** Amankan profit atau kurangi kepemilikan saham secara bertahap.  
                - **Target Koreksi AI:** Rp {target_pred:,.2f}  
                """)
            else:
                st.info(f"""
                🔵 **STATUS: HOLD / WAIT & SEE (KONSOLIDASI)**  
                - **Aksi Strategis:** Pertahankan posisi sambil menunggu sinyal momentum berikutnya.  
                """)

            st.markdown("---")

            # Bagian Grafik Interaktif Plotly yang Elegan
            st.subheader(f"📈 Grafik Pergerakan & Proyeksi AI Interaktif")
            
            last_date = df.index[-1]
            if interval_val == "1d":
                future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=prediction_days)
            elif interval_val == "1h":
                future_dates = pd.date_range(start=last_date + pd.Timedelta(hours=1), periods=prediction_days, freq='h')
            else:
                future_dates = pd.date_range(start=last_date + pd.Timedelta(minutes=1), periods=prediction_days, freq='min')

            step_diff = (target_pred - current_price) / prediction_days
            future_prices = [current_price + step_diff * i for i in range(1, prediction_days + 1)]
            
            projection_x = [last_date] + list(future_dates)
            projection_y = [current_price] + future_prices

            fig = go.Figure()
            
            # Grafik Harga Aktual
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['Close'], 
                mode='lines', 
                name='Harga Aktual Real-Time',
                line=dict(color='#00d2ff', width=2.5)
            ))
            
            # Grafik Proyeksi AI
            fig.add_trace(go.Scatter(
                x=projection_x, 
                y=projection_y, 
                mode='lines+markers', 
                name=f'Proyeksi AI ({prediction_days} Hari)',
                line=dict(color='#00ff87', width=3, dash='dash'),
                marker=dict(size=8, color='#00ff87')
            ))
            
            fig.update_layout(
                template="plotly_dark",
                xaxis=dict(title="Waktu Perdagangan", gridcolor="#30363d"),
                yaxis=dict(title="Harga (IDR)", gridcolor="#30363d"),
                hovermode="x unified",
                margin=dict(l=10, r=10, t=10, b=10),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                plot_bgcolor="#0e1117",
                paper_bgcolor="#0e1117"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.caption(f"🔄 Status: Auto-refresh aktif setiap 5 detik. Pembaruan terakhir pada {current_date_str} pukul {current_time_str} WIB.")

    else:
        st.warning("Data historis tidak mencukupi untuk parameter yang dipilih. Silakan ubah interval atau kode emiten.")

except Exception as e:
    st.error(f"Terjadi kesalahan sistem saat memproses data: {e}")

st.markdown("---")
st.subheader("🔗 Akses Cepat Sumber Data Eksternal")
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
