# =========================
# ENTERPRISE PRO MAX STOCK AI APP
# FINAL ACADEMIC SUBMISSION VERSION
# =========================

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import sqlite3
import hashlib
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import plotly.graph_objects as go
import plotly.express as px
from fpdf import FPDF

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Enterprise Pro Max Stock AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# CUSTOM CSS
# =========================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #f8fbff 0%, #eef5ff 100%);
    color: #1f2937;
}
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.85);
    backdrop-filter: blur(18px);
    border-right: 1px solid rgba(255,255,255,0.4);
}
.main-title {
    font-size: 2.3rem;
    font-weight: 800;
    color: #0f172a;
    margin-bottom: 0.3rem;
}
.sub-title {
    font-size: 1rem;
    color: #475569;
    margin-bottom: 1.2rem;
}
.glass-card {
    background: rgba(255, 255, 255, 0.72);
    backdrop-filter: blur(16px);
    border: 1px solid rgba(255,255,255,0.35);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 10px 35px rgba(15, 23, 42, 0.08);
    margin-bottom: 16px;
}
.metric-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.9), rgba(245,249,255,0.75));
    border: 1px solid rgba(255,255,255,0.45);
    border-radius: 20px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(15,23,42,0.08);
    text-align: center;
}
.metric-title {
    font-size: 0.95rem;
    color: #64748b;
    font-weight: 600;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #0f172a;
    margin-top: 6px;
}
.stButton>button {
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    border-radius: 14px;
    padding: 0.6rem 1.1rem;
    font-weight: 700;
    box-shadow: 0 8px 20px rgba(59,130,246,0.25);
}
.stButton>button:hover {
    transform: translateY(-1px);
    transition: 0.2s ease;
}
.stTextInput>div>div>input,
.stNumberInput input,
.stSelectbox div[data-baseweb="select"] > div,
.stTextArea textarea {
    border-radius: 14px !important;
    background: rgba(255,255,255,0.8) !important;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 800;
    color: #0f172a;
    margin: 0.4rem 0 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# =========================
# DATABASE
# =========================
conn = sqlite3.connect("stock_app.db", check_same_thread=False)
c = conn.cursor()

def init_db():
    c.execute("""
    CREATE TABLE IF NOT EXISTS users(
        email TEXT PRIMARY KEY,
        password TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        stock TEXT,
        prediction REAL,
        current_price REAL,
        predicted_change REAL,
        recommendation TEXT,
        confidence REAL,
        model_used TEXT,
        date TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS watchlist(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        stock TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS portfolio(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT,
        stock TEXT,
        quantity REAL,
        buy_price REAL,
        buy_date TEXT
    )
    """)

    conn.commit()

init_db()

# =========================
# HELPERS
# =========================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    return "@" in email and "." in email and len(email) >= 6

def validate_password(password):
    return len(password) >= 6

def metric_card(title, value):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def safe_float(x, default=0.0):
    try:
        return float(x)
    except:
        return default

# =========================
# AUTH
# =========================
def register_user(email, password):
    try:
        c.execute("INSERT INTO users VALUES (?,?)", (email.strip().lower(), hash_password(password)))
        conn.commit()
        return True
    except:
        return False

def login_user(email, password):
    c.execute("SELECT * FROM users WHERE email=? AND password=?", (email.strip().lower(), hash_password(password)))
    return c.fetchone()

# =========================
# STOCK MAP
# =========================
STOCKS = {
    "Reliance": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "Infosys": "INFY.NS",
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "ITC": "ITC.NS",
    "SBI": "SBIN.NS",
    "Wipro": "WIPRO.NS",
    "HUL": "HINDUNILVR.NS",
    "Adani Enterprises": "ADANIENT.NS",
    "Tata Motors": "TATAMOTORS.NS",
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
    "Tesla": "TSLA",
    "NVIDIA": "NVDA"
}

# =========================
# DATA FETCH
# =========================
@st.cache_data(ttl=300)
def get_stock_data(symbol, period="1y", interval="1d"):
    try:
        symbol = str(symbol).strip().upper()

        df = yf.download(
            tickers=symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

        if df is None or df.empty:
            return pd.DataFrame()

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

        df = df.reset_index()

        if "Date" not in df.columns and "Datetime" in df.columns:
            df.rename(columns={"Datetime": "Date"}, inplace=True)

        required_cols = ["Date", "Open", "High", "Low", "Close", "Volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(subset=["Date", "Close"])
        df = df.sort_values("Date").reset_index(drop=True)

        return df

    except:
        return pd.DataFrame()

# =========================
# INDICATORS
# =========================
def add_indicators(df):
    if df.empty or len(df) < 30:
        return pd.DataFrame()

    df = df.copy()

    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["EMA_20"] = df["Close"].ewm(span=20, adjust=False).mean()

    delta = df["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, 1e-10)

    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_SIGNAL"] = df["MACD"].ewm(span=9, adjust=False).mean()

    rolling_std = df["Close"].rolling(20).std()
    df["BB_HIGH"] = df["SMA_20"] + (2 * rolling_std)
    df["BB_LOW"] = df["SMA_20"] - (2 * rolling_std)

    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility_10"] = df["Daily_Return"].rolling(10).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.ffill().bfill()
    df = df.dropna().reset_index(drop=True)

    return df

# =========================
# AI MODEL
# =========================
def train_model(df):
    try:
        if df.empty or len(df) < 50:
            return None, None, None, None, None

        df = df.copy()
        required_cols = ["Close", "SMA_20", "EMA_20", "RSI", "MACD", "MACD_SIGNAL", "Volume"]

        for col in required_cols:
            if col not in df.columns:
                return None, None, None, None, None

        df["Target"] = df["Close"].shift(-1)
        df = df.dropna()

        X = df[required_cols].copy()
        y = df["Target"].copy()

        X = X.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()
        y = pd.to_numeric(y, errors="coerce").replace([np.inf, -np.inf], np.nan).ffill().bfill()

        valid_idx = X.dropna().index.intersection(y.dropna().index)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]

        if len(X) < 30:
            return None, None, None, None, None

        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        model = RandomForestRegressor(n_estimators=150, random_state=42)
        model.fit(X_train, y_train)

        test_pred = model.predict(X_test) if len(X_test) > 0 else []
        mae = mean_absolute_error(y_test, test_pred) if len(X_test) > 0 else 0
        r2 = r2_score(y_test, test_pred) if len(X_test) > 0 else 0

        latest_features = X.iloc[[-1]]
        prediction = float(model.predict(latest_features)[0])

        return model, latest_features, prediction, mae, r2

    except:
        return None, None, None, None, None

# =========================
# RECOMMENDATION ENGINE
# =========================
def get_recommendation(latest_price, predicted_price, rsi, macd, macd_signal):
    score = 0

    if predicted_price > latest_price:
        score += 2
    else:
        score -= 2

    if rsi < 30:
        score += 2
    elif rsi > 70:
        score -= 2

    if macd > macd_signal:
        score += 1
    else:
        score -= 1

    if score >= 2:
        return "BUY"
    elif score <= -2:
        return "SELL"
    else:
        return "HOLD"

# =========================
# DATABASE HELPERS
# =========================
def save_prediction(email, stock, prediction, current_price, predicted_change, recommendation, confidence, model_used="RandomForest"):
    try:
        c.execute("""
        INSERT INTO history(email, stock, prediction, current_price, predicted_change, recommendation, confidence, model_used, date)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            email,
            stock,
            float(prediction),
            float(current_price),
            float(predicted_change),
            recommendation,
            float(confidence),
            model_used,
            str(datetime.now())
        ))
        conn.commit()
    except Exception as e:
        st.error(f"Error saving prediction: {e}")

def get_prediction_history(email):
    try:
        query = """
        SELECT stock, prediction, current_price, predicted_change, recommendation, confidence, model_used, date
        FROM history
        WHERE email=?
        ORDER BY id DESC
        """
        df = pd.read_sql_query(query, conn, params=(email,))
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        return df
    except:
        return pd.DataFrame()

def add_to_watchlist(email, stock):
    c.execute("SELECT * FROM watchlist WHERE email=? AND stock=?", (email, stock))
    exists = c.fetchone()
    if not exists:
        c.execute("INSERT INTO watchlist(email, stock) VALUES (?,?)", (email, stock))
        conn.commit()

def remove_from_watchlist(email, stock):
    c.execute("DELETE FROM watchlist WHERE email=? AND stock=?", (email, stock))
    conn.commit()

def get_watchlist(email):
    return pd.read_sql_query("SELECT stock FROM watchlist WHERE email=?", conn, params=(email,))

def add_portfolio_stock(email, stock, quantity, buy_price):
    c.execute("""
    INSERT INTO portfolio(email, stock, quantity, buy_price, buy_date)
    VALUES (?,?,?,?,?)
    """, (email, stock, quantity, buy_price, str(datetime.now().date())))
    conn.commit()

def get_portfolio(email):
    return pd.read_sql_query("""
    SELECT stock, quantity, buy_price, buy_date
    FROM portfolio
    WHERE email=?
    """, conn, params=(email,))

# =========================
# PDF REPORT
# =========================
def generate_pdf_report(stock_name, symbol, current_price, prediction, change_pct, recommendation, confidence):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Enterprise Pro Max Stock AI Report", ln=True, align="C")

    pdf.ln(8)
    pdf.set_font("Arial", "", 12)
    pdf.cell(0, 10, f"Stock: {stock_name} ({symbol})", ln=True)
    pdf.cell(0, 10, f"Current Price: {current_price:.2f}", ln=True)
    pdf.cell(0, 10, f"Predicted Next Price: {prediction:.2f}", ln=True)
    pdf.cell(0, 10, f"Expected Change: {change_pct:.2f}%", ln=True)
    pdf.cell(0, 10, f"Recommendation: {recommendation}", ln=True)
    pdf.cell(0, 10, f"Model Confidence: {confidence:.2f}%", ln=True)
    pdf.cell(0, 10, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)

    file_path = "stock_report.pdf"
    pdf.output(file_path)
    return file_path

# =========================
# CHARTS
# =========================
def plot_line_chart(df, y_col, title, name=None):
    if df.empty or y_col not in df.columns:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df[y_col], mode="lines", name=name if name else y_col))
    fig.update_layout(title=title, template="plotly_white", height=420)
    st.plotly_chart(fig, use_container_width=True)

def plot_multi_chart(df):
    if df.empty:
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["SMA_20"], mode="lines", name="SMA 20"))
    fig.add_trace(go.Scatter(x=df["Date"], y=df["EMA_20"], mode="lines", name="EMA 20"))
    fig.update_layout(title="Close vs Moving Averages", template="plotly_white", height=450)
    st.plotly_chart(fig, use_container_width=True)

# =========================
# SESSION
# =========================
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None

# =========================
# LOGIN / REGISTER
# =========================
if not st.session_state.logged_in:
    st.markdown('<div class="main-title">📈 Enterprise Pro Max Stock AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Professional AI Stock Analysis & Prediction Platform</div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])

    with tab1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        email = st.text_input("Email", key="login_email")
        password = st.text_input("Password", type="password", key="login_password")

        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.error("Please fill all fields.")
            else:
                user = login_user(email, password)
                if user:
                    st.session_state.logged_in = True
                    st.session_state.user_email = email.strip().lower()
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        reg_email = st.text_input("Email", key="reg_email")
        reg_password = st.text_input("Password", type="password", key="reg_password")
        reg_confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")

        if st.button("Register", use_container_width=True):
            if not reg_email or not reg_password or not reg_confirm:
                st.error("Please fill all fields.")
            elif not validate_email(reg_email):
                st.error("Enter a valid email.")
            elif not validate_password(reg_password):
                st.error("Password must be at least 6 characters.")
            elif reg_password != reg_confirm:
                st.error("Passwords do not match.")
            else:
                if register_user(reg_email, reg_password):
                    st.success("Registration successful. Please login.")
                else:
                    st.error("User already exists.")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================
# MAIN APP
# =========================
else:
    st.sidebar.markdown("## 📊 Enterprise Pro Max")
    st.sidebar.caption(f"Logged in as: {st.session_state.user_email}")

    page = st.sidebar.radio("Navigation", [
        "🏠 Dashboard",
        "📊 Stock Analysis",
        "🔮 AI Predictions",
        "📋 Stock Screener",
        "⭐ Watchlist",
        "💼 Portfolio Tracker",
        "📊 Stock Comparison",
        "🤖 AI Chatbot",
        "🕘 Prediction History",
        "📄 Export Report"
    ])

    stock_name = st.sidebar.selectbox("Select Company", list(STOCKS.keys()))
    manual_symbol = st.sidebar.text_input("Or Enter Custom Symbol", "")
    period = st.sidebar.selectbox("Time Period", ["6mo", "1y", "2y", "5y"], index=1)

    stock_symbol = manual_symbol.strip().upper() if manual_symbol.strip() else STOCKS[stock_name].strip().upper()

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.user_email = None
        st.rerun()

    df_raw = get_stock_data(stock_symbol, period=period)
    df = add_indicators(df_raw) if not df_raw.empty else pd.DataFrame()

    latest_price = float(df["Close"].iloc[-1]) if not df.empty else 0.0
    previous_price = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_price
    day_change = latest_price - previous_price
    day_change_pct = (day_change / previous_price * 100) if previous_price != 0 else 0

    # =========================
    # DASHBOARD
    # =========================
    if page == "🏠 Dashboard":
        st.markdown(f'<div class="main-title">📈 {stock_symbol} Dashboard</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-title">Professional overview of selected stock</div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            metric_card("Current Price", f"{latest_price:.2f}")
        with c2:
            metric_card("Day Change", f"{day_change:.2f}")
        with c3:
            metric_card("Day Change %", f"{day_change_pct:.2f}%")
        with c4:
            metric_card("Volume", f"{int(df['Volume'].iloc[-1]):,}" if not df.empty else "N/A")

        st.markdown("### 📉 Stock Trend")
        plot_line_chart(df, "Close", "Stock Price Trend", "Close Price")

        if not df.empty:
            st.markdown("### 📋 Latest Data")
            st.dataframe(df.tail(10), use_container_width=True)

    # =========================
    # STOCK ANALYSIS
    # =========================
    elif page == "📊 Stock Analysis":
        st.markdown(f'<div class="main-title">📊 Stock Analysis — {stock_symbol}</div>', unsafe_allow_html=True)

        if df.empty:
            st.error("Could not load stock data.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                metric_card("Current Price", f"{latest_price:.2f}")
            with c2:
                metric_card("RSI", f"{df['RSI'].iloc[-1]:.2f}")
            with c3:
                metric_card("MACD", f"{df['MACD'].iloc[-1]:.2f}")
            with c4:
                metric_card("Volatility", f"{df['Volatility_10'].iloc[-1]:.4f}")

            plot_multi_chart(df)
            plot_line_chart(df, "RSI", "RSI Indicator", "RSI")
            plot_line_chart(df, "MACD", "MACD Indicator", "MACD")
            plot_line_chart(df, "Volume", "Volume Analysis", "Volume")

            temp = df.copy()
            temp["Daily Return %"] = temp["Close"].pct_change() * 100
            plot_line_chart(temp, "Daily Return %", "Daily Return Analysis", "Return %")

            st.markdown("### 📌 Technical Summary")
            summary = []

            if df["RSI"].iloc[-1] > 70:
                summary.append("Stock may be **overbought** based on RSI.")
            elif df["RSI"].iloc[-1] < 30:
                summary.append("Stock may be **oversold** based on RSI.")
            else:
                summary.append("RSI is in a **neutral range**.")

            if df["MACD"].iloc[-1] > df["MACD_SIGNAL"].iloc[-1]:
                summary.append("MACD shows **bullish momentum**.")
            else:
                summary.append("MACD shows **bearish momentum**.")

            if latest_price > df["SMA_20"].iloc[-1]:
                summary.append("Price is trading **above SMA 20**.")
            else:
                summary.append("Price is trading **below SMA 20**.")

            st.success("\n\n".join([f"- {s}" for s in summary]))

    # =========================
    # AI PREDICTIONS
    # =========================
    elif page == "🔮 AI Predictions":
        st.markdown(f'<div class="main-title">🔮 AI Prediction — {stock_symbol}</div>', unsafe_allow_html=True)

        if df.empty:
            st.error("Not enough stock data available.")
        else:
            if st.button("🚀 Run AI Prediction", use_container_width=True):
                model, latest_features, pred, mae, r2 = train_model(df)

                if model is None or pred is None:
                    st.error("Prediction failed. Try another stock or longer time period.")
                else:
                    predicted_change = ((pred - latest_price) / latest_price) * 100 if latest_price != 0 else 0
                    latest_rsi = df["RSI"].iloc[-1]
                    latest_macd = df["MACD"].iloc[-1]
                    latest_macd_signal = df["MACD_SIGNAL"].iloc[-1]

                    recommendation = get_recommendation(latest_price, pred, latest_rsi, latest_macd, latest_macd_signal)
                    confidence = max(0, min(100, (r2 if r2 is not None else 0) * 100))

                    c1, c2, c3, c4 = st.columns(4)
                    with c1:
                        metric_card("Current Price", f"{latest_price:.2f}")
                    with c2:
                        metric_card("Predicted Price", f"{pred:.2f}")
                    with c3:
                        metric_card("Expected Change", f"{predicted_change:.2f}%")
                    with c4:
                        metric_card("Confidence", f"{confidence:.2f}%")

                    if recommendation == "BUY":
                        st.success(f"### 📈 Recommendation: {recommendation}")
                    elif recommendation == "SELL":
                        st.error(f"### 📉 Recommendation: {recommendation}")
                    else:
                        st.warning(f"### ⏸ Recommendation: {recommendation}")

                    st.info(f"Model MAE: {mae:.2f} | Model R² Score: {r2:.4f}")

                    save_prediction(
                        st.session_state.user_email,
                        stock_symbol,
                        pred,
                        latest_price,
                        predicted_change,
                        recommendation,
                        confidence
                    )

                    st.success("Prediction saved successfully!")

                    chart_df = pd.DataFrame({
                        "Type": ["Current Price", "Predicted Price"],
                        "Price": [latest_price, pred]
                    })

                    fig = px.bar(chart_df, x="Type", y="Price", title="Current vs Predicted Price")
                    fig.update_layout(template="plotly_white", height=400)
                    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # STOCK SCREENER
    # =========================
    elif page == "📋 Stock Screener":
        st.markdown('<div class="main-title">📋 Stock Screener</div>', unsafe_allow_html=True)

        screener_data = []
        for name, sym in STOCKS.items():
            temp = get_stock_data(sym, period="6mo")
            temp = add_indicators(temp) if not temp.empty else pd.DataFrame()
            if not temp.empty:
                current = float(temp["Close"].iloc[-1])
                start = float(temp["Close"].iloc[0])
                ret = ((current - start) / start) * 100 if start != 0 else 0
                rsi = float(temp["RSI"].iloc[-1])
                screener_data.append([name, sym, round(current, 2), round(ret, 2), round(rsi, 2)])

        if screener_data:
            screener_df = pd.DataFrame(screener_data, columns=["Stock", "Symbol", "Current Price", "6M Return %", "RSI"])
            st.dataframe(screener_df.sort_values("6M Return %", ascending=False), use_container_width=True)
        else:
            st.warning("Could not load screener data.")

    # =========================
    # WATCHLIST
    # =========================
    elif page == "⭐ Watchlist":
        st.markdown('<div class="main-title">⭐ Watchlist</div>', unsafe_allow_html=True)

        if st.button("➕ Add Selected Stock", use_container_width=True):
            add_to_watchlist(st.session_state.user_email, stock_symbol)
            st.success(f"{stock_symbol} added to watchlist.")

        watchlist_df = get_watchlist(st.session_state.user_email)

        if not watchlist_df.empty:
            st.dataframe(watchlist_df, use_container_width=True)
            remove_stock = st.selectbox("Remove Stock", watchlist_df["stock"].tolist())
            if st.button("❌ Remove Stock", use_container_width=True):
                remove_from_watchlist(st.session_state.user_email, remove_stock)
                st.success(f"{remove_stock} removed.")
                st.rerun()
        else:
            st.info("Your watchlist is empty.")

    # =========================
    # PORTFOLIO TRACKER
    # =========================
    elif page == "💼 Portfolio Tracker":
        st.markdown('<div class="main-title">💼 Portfolio Tracker</div>', unsafe_allow_html=True)

        with st.form("portfolio_form"):
            p_stock = st.text_input("Stock Symbol", stock_symbol)
            quantity = st.number_input("Quantity", min_value=1.0, step=1.0)
            buy_price = st.number_input("Buy Price", min_value=0.0, step=0.1)
            submitted = st.form_submit_button("Add to Portfolio")

            if submitted:
                add_portfolio_stock(st.session_state.user_email, p_stock.strip().upper(), quantity, buy_price)
                st.success(f"{p_stock} added to portfolio.")

        portfolio_df = get_portfolio(st.session_state.user_email)

        if not portfolio_df.empty:
            portfolio_rows = []
            total_invested = 0
            total_current = 0

            for _, row in portfolio_df.iterrows():
                sym = row["stock"]
                temp = get_stock_data(sym, period="1mo")
                temp = add_indicators(temp) if not temp.empty else pd.DataFrame()

                current = float(temp["Close"].iloc[-1]) if not temp.empty else row["buy_price"]
                invested = row["quantity"] * row["buy_price"]
                current_value = row["quantity"] * current
                pnl = current_value - invested

                total_invested += invested
                total_current += current_value

                portfolio_rows.append([
                    sym, row["quantity"], row["buy_price"], round(current, 2),
                    round(invested, 2), round(current_value, 2), round(pnl, 2)
                ])

            out_df = pd.DataFrame(portfolio_rows, columns=[
                "Stock", "Qty", "Buy Price", "Current Price", "Invested", "Current Value", "PnL"
            ])

            c1, c2, c3 = st.columns(3)
            with c1:
                metric_card("Total Invested", f"{total_invested:.2f}")
            with c2:
                metric_card("Current Value", f"{total_current:.2f}")
            with c3:
                metric_card("Net PnL", f"{(total_current-total_invested):.2f}")

            st.dataframe(out_df, use_container_width=True)

            pie_df = out_df.groupby("Stock")["Current Value"].sum().reset_index()
            fig = px.pie(pie_df, names="Stock", values="Current Value", title="Portfolio Allocation")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No portfolio stocks added yet.")

    # =========================
    # STOCK COMPARISON
    # =========================
    elif page == "📊 Stock Comparison":
        st.markdown('<div class="main-title">📊 Stock Comparison</div>', unsafe_allow_html=True)

        compare_stocks = st.multiselect("Select Stocks to Compare", list(STOCKS.keys()), default=["Reliance", "TCS"])

        if compare_stocks:
            fig = go.Figure()
            for s in compare_stocks:
                sym = STOCKS[s]
                temp = get_stock_data(sym, period="1y")
                temp = add_indicators(temp) if not temp.empty else pd.DataFrame()
                if not temp.empty:
                    base = temp["Close"].iloc[0]
                    temp["Normalized"] = (temp["Close"] / base) * 100
                    fig.add_trace(go.Scatter(x=temp["Date"], y=temp["Normalized"], mode="lines", name=s))

            fig.update_layout(title="Normalized Price Comparison", template="plotly_white", height=500)
            st.plotly_chart(fig, use_container_width=True)

    # =========================
    # AI CHATBOT
    # =========================
    elif page == "🤖 AI Chatbot":
        st.markdown('<div class="main-title">🤖 AI Stock Chatbot</div>', unsafe_allow_html=True)

        user_q = st.text_input("Ask something about the selected stock...")

        if user_q:
            q = user_q.lower()

            if "price" in q:
                st.success(f"The current price of {stock_symbol} is approximately {latest_price:.2f}.")
            elif "rsi" in q and not df.empty:
                st.success(f"The latest RSI for {stock_symbol} is {df['RSI'].iloc[-1]:.2f}.")
            elif "trend" in q and not df.empty:
                trend = "bullish" if latest_price > df["SMA_20"].iloc[-1] else "bearish"
                st.success(f"The short-term trend appears {trend}.")
            elif "macd" in q and not df.empty:
                st.success(f"The latest MACD is {df['MACD'].iloc[-1]:.2f}.")
            elif "recommend" in q and not df.empty:
                model, latest_features, pred, mae, r2 = train_model(df)
                if pred:
                    rec = get_recommendation(latest_price, pred, df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["MACD_SIGNAL"].iloc[-1])
                    st.success(f"Current AI-based recommendation for {stock_symbol}: {rec}")
                else:
                    st.info("Prediction data not available for recommendation.")
            else:
                st.info("Try asking: current price, RSI, MACD, trend, recommendation")

    # =========================
    # PREDICTION HISTORY
    # =========================
    elif page == "🕘 Prediction History":
        st.markdown('<div class="main-title">🕘 Prediction History</div>', unsafe_allow_html=True)

        history = get_prediction_history(st.session_state.user_email)

        if not history.empty:
            st.dataframe(history, use_container_width=True)

            fig = px.line(history, x="date", y="prediction", color="stock", markers=True, title="Prediction History Trend")
            fig.update_layout(template="plotly_white", height=450)
            st.plotly_chart(fig, use_container_width=True)

            csv = history.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇ Download History CSV",
                csv,
                file_name="prediction_history.csv",
                mime="text/csv",
                use_container_width=True
            )
        else:
            st.info("No prediction history available yet.")

    # =========================
    # EXPORT REPORT
    # =========================
    elif page == "📄 Export Report":
        st.markdown('<div class="main-title">📄 Export Report</div>', unsafe_allow_html=True)

        if not df.empty:
            model, latest_features, pred, mae, r2 = train_model(df)
        else:
            model, latest_features, pred, mae, r2 = None, None, None, None, None

        if model is not None and pred is not None:
            predicted_change = ((pred - latest_price) / latest_price) * 100 if latest_price != 0 else 0
            recommendation = get_recommendation(latest_price, pred, df["RSI"].iloc[-1], df["MACD"].iloc[-1], df["MACD_SIGNAL"].iloc[-1])
            confidence = max(0, min(100, (r2 if r2 is not None else 0) * 100))

            if st.button("📥 Generate PDF Report", use_container_width=True):
                pdf_file = generate_pdf_report(
                    stock_symbol, stock_symbol, latest_price, pred, predicted_change, recommendation, confidence
                )

                with open(pdf_file, "rb") as f:
                    st.download_button(
                        "⬇ Download Report",
                        f,
                        file_name=f"{stock_symbol}_stock_report.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            st.warning("Not enough data to generate report.")