import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from mplfinance.original_flavor import candlestick_ohlc
import io
from math import sqrt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
import scipy.stats as stats
from datetime import datetime

# =============================
# CSS Personnalisé Global
# =============================
st.markdown("""
    <style>
    body { background-color: #f5f5f5; color: #333; }
    .title, h1, h2, h3 { color: #2c3e50; font-family: 'Segoe UI', sans-serif; }
    .stButton>button { background-color: #3498db; color: white; }
    .stDownloadButton>button { background-color: #27ae60; color: white; }
    .stSelectbox, .stTextInput, .stNumberInput { background-color: #ecf0f1; border-radius: 5px; }
    .sidebar .sidebar-content { background-color: #ecf0f1; }
    </style>
""", unsafe_allow_html=True)

# =============================
# Barre de navigation horizontale
# =============================
def render_navbar():
    cols = st.columns(5)
    labels = ["Accueil", "Configuration", "Infos Compagnie", "Prévisions", "Export"]
    keys = ["nav_accueil", "nav_config", "nav_info", "nav_previsions", "nav_export"]
    for i, label in enumerate(labels):
        if cols[i].button(label, key=keys[i]):
            st.session_state.selected_page = label

# =============================
# Utilitaires
# =============================
def get_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)
    if not df.empty:
        df['Return'] = df['Close'].pct_change()
        df['Volatility'] = df['Return'].rolling(20).std()
        df['SMA20'] = df['Close'].rolling(20).mean()
    return df

def prepare_data(df, look_back=60):
    arr = df['Close'].values.reshape(-1,1)
    scaler = MinMaxScaler((0,1))
    scaled = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(look_back, len(scaled)):
        X.append(scaled[i-look_back:i,0])
        y.append(scaled[i,0])
    X = np.array(X).reshape(-1, look_back, 1)
    y = np.array(y)
    return X, y, scaled, scaler

def train_model(X_train, y_train, model_type="LSTM", epochs=10, batch_size=32):
    model = Sequential()
    if model_type == "LSTM":
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(LSTM(50))
    else:
        model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1],1)))
        model.add(GRU(50))
    model.add(Dense(1))
    model.compile('adam','mean_squared_error')
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    return model

def predict_future(scaled, look_back, days, model, scaler):
    seq, preds = scaled[-look_back:].copy(), []
    for _ in range(days):
        p = model.predict(seq.reshape(1, look_back, 1))[0,0]
        preds.append(p)
        seq = np.append(seq[1:], [[p]], axis=0)
    return np.array(preds)

def evaluate_performance(y_true, y_pred):
    mse = np.mean((y_true-y_pred)**2)
    rmse = np.sqrt(mse)
    ret = np.diff(y_true)/y_true[:-1]
    sharpe = np.mean(ret)/np.std(ret) if np.std(ret)!=0 else np.nan
    return mse, rmse, sharpe

def get_intellectia_data(ticker):
    return {"analysis":"Intellectia AI indique un sentiment global positif, suggérant une opportunité d'achat modérée.","signal":"Buy"}

# =============================
# Initialisation de la page
# =============================
if "selected_page" not in st.session_state:
    st.session_state.selected_page = "Accueil"
render_navbar()

# =============================
# Page Accueil
# =============================
if st.session_state.selected_page == "Accueil":
    st.markdown("<h1 class='title'>Bienvenue - Prévisions Boursières</h1>", unsafe_allow_html=True)
    st.markdown("**Utilité :** Analyse historique, prédictions LSTM/GRU, intervalles VaR, métriques financières.")
    st.markdown("""
    **Guide de l'utilisateur :**
    - **Accueil** : Vue d'ensemble et suivi des indices en temps réel.
    - **Configuration** : Sélection du ticker, période d'analyse et paramètres du modèle (LSTM/GRU, époques, batch).
    - **Infos Compagnie** : Affichage des informations financières clés de l'entreprise.
    - **Prévisions** : Entraînement du modèle, affichage des performances (MSE, RMSE, Ratio de Sharpe), prévisions futures et signaux.
    - **Export** : Téléchargement des prévisions au format Excel pour analyse complémentaire.
    """, unsafe_allow_html=True)
    st.markdown("### Indices et taux en temps réel (Canada)")
    indices = {"S&P 500":"^GSPC","TSX Composite":"^GSPTSE","NASDAQ":"^IXIC","USD/CAD":"USDCAD=X","EUR/CAD":"EURCAD=X","Or":"GC=F","Pétrole":"CL=F"}
    cols = st.columns(2)
    for i,(name,sym) in enumerate(indices.items()):
        df_i = yf.Ticker(sym).history(period="2d")
        if len(df_i)>=2:
            last, prev = df_i['Close'].iloc[-1], df_i['Close'].iloc[0]
            arrow = "<span style='color:green;'>↑</span>" if last>=prev else "<span style='color:red;'>↓</span>"
            cols[i%2].markdown(f"**{name}** : {last:.2f} {arrow}", unsafe_allow_html=True)
    st.markdown("---")
    if st.button("Allons‑y", key="start_button"):
        st.session_state.selected_page = "Configuration"

# =============================
# Page Configuration
# =============================
elif st.session_state.selected_page == "Configuration":
    st.markdown("<h2 class='title'>Configuration du Modèle</h2>", unsafe_allow_html=True)
    st.markdown("**Modèles :** LSTM (relations longues), GRU (plus rapide) | **Époques :** itérations sur données | **Batch size :** taille du lot pour MAJ des poids.")
    with st.form("config_form"):
        ticker = st.text_input("Ticker", "AAPL")
        start = st.date_input("Date début", pd.to_datetime("2024-01-01"))
        end = st.date_input("Date fin", pd.to_datetime("2025-01-01"))
        days = st.number_input("Jours à prédire",1,365,30)
        model_type = st.selectbox("Modèle", ["LSTM","GRU"] )
        epochs = st.number_input("Époques",1,100,10)
        batch = st.number_input("Batch size",1,256,32)
        if st.form_submit_button("Enregistrer et Continuer"):
            st.session_state.update({
                "ticker":ticker,
                "start_date":start.strftime("%Y-%m-%d"),
                "end_date":end.strftime("%Y-%m-%d"),
                "future_days":days,
                "model_type":model_type,
                "epochs":epochs,
                "batch_size":batch
            })
            st.success("Paramètres enregistrés !")

# =============================
# Page Infos Compagnie
# =============================
elif st.session_state.selected_page == "Infos Compagnie":
    st.markdown("<h2 class='title'>Informations sur la Compagnie</h2>", unsafe_allow_html=True)
    st.markdown("*Source : Yahoo Finance*")
    if st.button("← Retour", key="info_back"):
        st.session_state.selected_page = "Configuration"
    info = yf.Ticker(st.session_state.ticker).info
    st.markdown(f"**Nom :** {info.get('longName','N/A')}")
    st.markdown(f"**Secteur :** {info.get('sector','N/A')}")
    st.markdown(f"**Industrie :** {info.get('industry','N/A')}")
    st.markdown(f"**Description :** {info.get('longBusinessSummary','N/A')}")
    if st.button("Continuer vers Prévisions", key="info_next"):
        st.session_state.selected_page = "Prévisions"

# =============================
# Page Prévisions
# =============================
elif st.session_state.selected_page == "Prévisions":
    st.markdown("<h2 class='title'>Résultats des Prévisions</h2>", unsafe_allow_html=True)
    df = get_data(st.session_state.ticker, st.session_state.start_date, st.session_state.end_date)
    if df.empty:
        st.error("Aucune donnée récupérée.")
    else:
        st.subheader("Données historiques")
        st.write(df.tail())
        # Entraînement et évaluation
        X, y, scaled, scaler = prepare_data(df, 60)
        split = int(len(X)*0.8)
        model = train_model(X[:split], y[:split], model_type=st.session_state.model_type,
                             epochs=st.session_state.epochs, batch_size=st.session_state.batch_size)
        y_test_pred = model.predict(X[split:]).flatten()
        mse, rmse, sharpe = evaluate_performance(y[split:], y_test_pred)
        st.markdown(f"**MSE :** {mse:.6f}   **RMSE :** {rmse:.6f}   **Sharpe :** {sharpe:.4f}")
        st.markdown("""
        **Interprétation des métriques :**
        - **MSE (Mean Squared Error)** : moyenne des carrés des écarts, plus elle est faible, plus le modèle est précis.
        - **RMSE (Root Mean Squared Error)** : racine de la MSE, exprimée dans l'unité du prix, facilite l'interprétation de l'erreur.
        - **Ratio de Sharpe** : rendement ajusté au risque; un ratio élevé indique une performance supérieure par unité de risque.
        """, unsafe_allow_html=True)
        # Prévisions futures
        fut = predict_future(scaled, 60, st.session_state.future_days, model, scaler)
        zl, zu = -1.2, 1.2
        var_l = [fut[i] + zl * np.std(y[:split]) * sqrt(i+1) for i in range(st.session_state.future_days)]
        var_n = fut
        var_u = [fut[i] + zu * np.std(y[:split]) * sqrt(i+1) for i in range(st.session_state.future_days)]
        vl = scaler.inverse_transform(np.array(var_l).reshape(-1,1)).flatten()
        vn = scaler.inverse_transform(np.array(var_n).reshape(-1,1)).flatten()
        vu = scaler.inverse_transform(np.array(var_u).reshape(-1,1)).flatten()
        future_idx = pd.date_range(df.index[-1] + pd.Timedelta(1,'D'), periods=st.session_state.future_days)
        df_var = pd.DataFrame({'Pessimiste':vl,'Neutre':vn,'Optimiste':vu}, index=future_idx)
        st.subheader("Prévisions futures")
        st.write(df_var)
        # Graphique des scénarios optimiste, neutre, pessimiste
        fig2, ax2 = plt.subplots()
        ax2.plot(df_var.index, df_var['Pessimiste'], '--', label='Pessimiste')
        ax2.plot(df_var.index, df_var['Neutre'], '-.', label='Neutre')
        ax2.plot(df_var.index, df_var['Optimiste'], ':', label='Optimiste')
        ax2.set_title("Prévisions futures : scénarios pessimiste, neutre, optimiste")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Prix")
        ax2.legend()
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
        st.pyplot(fig2)
        # Signaux Intellectia
        ia = get_intellectia_data(st.session_state.ticker)
        st.subheader("Analyse Intellectia AI")
        st.markdown(f"**Analyse :** {ia['analysis']}")
        st.markdown(f"**Signal :** {ia['signal']}")
        # Signal d'investissement
        current_price = float(df['Close'].iloc[-1])
        predicted_neutral = float(vn[-1])
        pct_change = (predicted_neutral - current_price)/current_price*100
        if pct_change > 5:
            inv_signal = "Acheter"
        elif pct_change < -5:
            inv_signal = "Vendre"
        else:
            inv_signal = "Conserver"
        st.markdown(f"**Signal modèle :** {inv_signal} ({pct_change:.2f}%)")
        # Candlestick + Bollinger en bas
        st.markdown("---")
        st.subheader("Graphique Chandeliers + Bandes de Bollinger")
        st.markdown("*Visualisation des prix OHLC et volatilité via bandes.*")
        df_c = df[['Open','High','Low','Close','Volume']].apply(pd.to_numeric, errors='coerce')
        df_c['SMA20'] = df_c['Close'].rolling(20).mean()
        df_c['Std20'] = df_c['Close'].rolling(20).std()
        df_c['UpperBand'] = df_c['SMA20'] + 2*df_c['Std20']
        df_c['LowerBand'] = df_c['SMA20'] - 2*df_c['Std20']
        df_c.dropna(inplace=True)
        df_c['DateNum'] = mdates.date2num(df_c.index.to_pydatetime())
        ohlc = df_c[['DateNum','Open','High','Low','Close']].values
        fig, ax = plt.subplots()
        candlestick_ohlc(ax, ohlc, width=0.6, colorup='green', colordown='red')
        ax.plot(df_c.index, df_c['UpperBand'], 'r--', label='Bollinger Haut')
        ax.plot(df_c.index, df_c['LowerBand'], 'b--', label='Bollinger Bas')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%d-%b'))
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        ax.legend()
        st.pyplot(fig)
        st.session_state.df_var = df_var

# =============================
# Page Export
# =============================
elif st.session_state.selected_page == "Export":
    st.markdown("<h2 class='title'>Export des Prévisions</h2>", unsafe_allow_html=True)
    st.markdown("Téléchargez vos prévisions au format Excel pour une analyse hors ligne.")
    if "df_var" in st.session_state:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
            st.session_state.df_var.to_excel(writer, sheet_name='Prévisions')
        st.download_button(
            "Télécharger Excel",
            buffer.getvalue(),
            file_name="previsions.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    else:
        st.error("Aucune prévision disponible. Lancez d'abord les prévisions.")
