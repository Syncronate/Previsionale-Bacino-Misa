import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
import math  # Importa il modulo math

# Ripristino delle classi e funzioni dal modello originale

class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=2, dropout=0.2):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Fully connected layer per la previsione dei livelli idrometrici
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Inizializzazione dello stato nascosto
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Prendiamo solo l'output dell'ultimo timestep
        out = out[:, -1, :]

        # Fully connected layer
        # out shape: (batch_size, output_size * output_window)
        out = self.fc(out)

        # Reshaping per ottenere la sequenza di output
        # out shape: (batch_size, output_window, output_size)
        out = out.view(out.size(0), self.output_window, self.output_size)

        return out

# Funzione per caricare il modello addestrato
@st.cache_resource
def load_model(model_path, input_size, output_size, output_window):
    # Definizione delle costanti
    HIDDEN_SIZE = 128
    NUM_LAYERS = 2

    # Impostazione del device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Creazione del modello
    model = HydroLSTM(input_size, HIDDEN_SIZE, output_size, output_window, NUM_LAYERS).to(device)

    # Caricamento dei pesi del modello
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    return model, device

# Funzione per caricare gli scaler
@st.cache_resource
def load_scalers(scaler_features_path, scaler_targets_path):
    scaler_features = joblib.load(scaler_features_path)
    scaler_targets = joblib.load(scaler_targets_path)
    return scaler_features, scaler_targets

# Funzione per fare previsioni
def predict(model, input_data, scaler_features, scaler_targets, hydro_features, device, output_window):
    """
    Funzione per fare previsioni con il modello addestrato.

    Args:
        model: Il modello addestrato
        input_data: Dati di input non normalizzati (array di forma [input_window, num_features])
        scaler_features: Scaler per normalizzare i dati di input
        scaler_targets: Scaler per denormalizzare le previsioni
        hydro_features: Nomi dei sensori idrometrici
        device: Dispositivo (CPU/GPU)

    Returns:
        Previsioni denormalizzate
    """
    model.eval()

    # Normalizzazione dei dati di input
    input_normalized = scaler_features.transform(input_data)

    # Conversione in tensore PyTorch
    input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)

    # Previsione
    with torch.no_grad():
        output = model(input_tensor)

    # Conversione in numpy
    output_np = output.cpu().numpy().reshape(-1, len(hydro_features))

    # Denormalizzazione
    predictions = scaler_targets.inverse_transform(output_np)

    # Reshape per ottenere [output_window, num_hydro_features]
    predictions = predictions.reshape(output_window, len(hydro_features))

    return predictions

# Funzione per plot dei risultati
def plot_predictions(predictions, hydro_features, output_window, start_time=None):
    figures = []

    # Per ogni sensore idrometrico
    for i, sensor_name in enumerate(hydro_features):
        fig, ax = plt.subplots(figsize=(10, 5))

        # Creazione dell'asse x per le ore future
        if start_time:
            hours = [start_time + timedelta(hours=h) for h in range(output_window)]
            ax.plot(hours, predictions[:, i], marker='o', linestyle='-', label=f'Previsione {sensor_name}')
            plt.gcf().autofmt_xdate()
        else:
            hours = np.arange(output_window)
            ax.plot(hours, predictions[:, i], marker='o', linestyle='-', label=f'Previsione {sensor_name}')
            ax.set_xlabel('Ore future')

        ax.set_title(f'Previsione - {sensor_name}')
        ax.set_ylabel('Livello idrometrico [m]')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()

        figures.append(fig)

    return figures

# Funzione per ottenere un link di download per un file
def get_table_download_link(df):
    """Genera un link per scaricare il dataframe come file CSV"""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f'<a href="data:file/csv;base64,{b64}" download="previsioni.csv">Scarica i dati CSV</a>'

# Funzione per scaricare grafici
def get_image_download_link(fig, filename, text):
    """Genera un link per scaricare il grafico come immagine"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f'<a href="data:image/png;base64,{b64}" download="{filename}">Scarica {text}</a>'

# Titolo dell'app
st.title('Dashboard Modello Predittivo Idrologico')

# Sidebar per le opzioni
st.sidebar.header('Impostazioni')

# Opzione per caricare i propri file o usare quelli demo
use_demo_files = st.sidebar.checkbox('Usa file di esempio', value=True)

if use_demo_files:
    # Qui dovresti fornire percorsi ai file di esempio
    DATA_PATH = 'dati_idro.csv'  # Sostituisci con il percorso corretto
    MODEL_PATH = 'best_hydro_model.pth'  # Sostituisci con il percorso corretto
    SCALER_FEATURES_PATH = 'scaler_features.joblib'  # Sostituisci con il percorso corretto
    SCALER_TARGETS_PATH = 'scaler_targets.joblib'  # Sostituisci con il percorso corretto
else:
    # Caricamento dei file dall'utente
    st.sidebar.subheader('Carica i tuoi file')
    data_file = st.sidebar.file_uploader('File CSV con i dati storici', type=['csv'])
    model_file = st.sidebar.file_uploader('File del modello (.pth)', type=['pth'])
    scaler_features_file = st.sidebar.file_uploader('File scaler features (.joblib)', type=['joblib'])
    scaler_targets_file = st.sidebar.file_uploader('File scaler targets (.joblib)', type=['joblib'])

    # Controllo se tutti i file sono stati caricati
    if not (data_file and model_file and scaler_features_file and scaler_targets_file):
        st.sidebar.warning('Carica tutti i file necessari per procedere')
    else:
        pass

# Definizione delle costanti
INPUT_WINDOW = 6 # MODIFICATO INPUT_WINDOW A 6 ORE
OUTPUT_WINDOW = 6 # MODIFICATO OUTPUT_WINDOW A 6 ORE

# Caricamento dei dati storici
try:
    if use_demo_files:
        df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    else:
        df = pd.read_csv(data_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    st.sidebar.success(f'Dati caricati: {len(df)} righe')
except Exception as e:
    st.sidebar.error(f'Errore nel caricamento dei dati: {e}')
    df = None

# Estrazione delle caratteristiche
rain_features = [
    'Cumulata Sensore 1295 (Arcevia)',
    'Cumulata Sensore 2637 (Bettolelle)',
    'Cumulata Sensore 2858 (Barbara)',
    'Cumulata Sensore 2964 (Corinaldo)'
]

humidity_feature = ['Umidita\' Sensore 3452 (Montemurello)']

hydro_features = [
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
]

feature_columns = rain_features + humidity_feature + hydro_features

model = None
scaler_features = None
scaler_targets = None

# Caricamento del modello e degli scaler SOLO se i file sono stati caricati o si usano quelli demo
if use_demo_files or (data_file and model_file and scaler_features_file and scaler_targets_file):
    try:
        if use_demo_files:
            model, device = load_model(MODEL_PATH, len(feature_columns), len(hydro_features), OUTPUT_WINDOW)
            scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
        else:
            model_bytes = io.BytesIO(model_file.read())
            model, device = load_model(model_bytes, len(feature_columns), len(hydro_features), OUTPUT_WINDOW)
            scaler_features, scaler_targets = load_scalers(scaler_features_file, scaler_targets_file)

        st.sidebar.success('Modello e scaler caricati con successo')
    except Exception as e:
        st.sidebar.error(f'Errore nel caricamento del modello o degli scaler: {e}')

# Menu principale
st.sidebar.header('Menu')
page = st.sidebar.radio('Scegli una funzionalità',
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'])

if page == 'Dashboard':
    st.header('Dashboard Idrologica')

    if df is None or model is None or scaler_features is None or scaler_targets is None:
        st.warning("Attenzione: Alcuni file necessari non sono stati caricati correttamente. Alcune funzionalità potrebbero non essere disponibili.")
    else:
        # Mostra ultimi dati disponibili
        st.subheader('Ultimi dati disponibili')
        last_data = df.iloc[-1]
        last_date = last_data['Data e Ora']

        # Formattazione dei dati per la visualizzazione
        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**Data e ora ultimo rilevamento:** {last_date}")

            # Ultimi dati idrologici
            st.subheader('Livelli idrometrici attuali')
            hydro_data = []
            for feature in hydro_features:
                hydro_data.append({'Sensore': feature, 'Valore [m]': last_data[feature]})
            st.dataframe(pd.DataFrame(hydro_data).round(3))

        with col2:
            # Ultimi dati di pioggia
            st.subheader('Precipitazioni cumulate attuali')
            rain_data = []
            for feature in rain_features:
                rain_data.append({'Sensore': feature, 'Valore [mm]': last_data[feature]})
            st.dataframe(pd.DataFrame(rain_data).round(2))

            # Umidità
            st.subheader('Umidità attuale')
            st.write(f"{humidity_feature[0]}: {last_data[humidity_feature[0]]:.1f}%")

        # Previsione basata sugli ultimi dati disponibili
        st.header('Previsione in base agli ultimi dati')

        if st.button('Genera previsione'):
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile generare la previsione.")
            else:
                with st.spinner('Generazione previsione in corso...'):
                    # Preparazione dei dati di input (ultime INPUT_WINDOW ore)
                    latest_data = df.iloc[-INPUT_WINDOW:][feature_columns].values

                    # Previsione
                    predictions = predict(model, latest_data, scaler_features, scaler_targets, hydro_features, device, OUTPUT_WINDOW)

                    # Visualizzazione dei risultati
                    st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore')

                    # Creazione dataframe risultati
                    start_time = last_date
                    prediction_times = [start_time + timedelta(hours=i) for i in range(OUTPUT_WINDOW)]
                    results_df = pd.DataFrame(predictions, columns=hydro_features)
                    results_df['Ora previsione'] = prediction_times
                    results_df = results_df[['Ora previsione'] + hydro_features]

                    # Visualizzazione tabella risultati
                    st.dataframe(results_df.round(3))

                    # Download dei risultati
                    st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

                    # Grafici per ogni sensore
                    st.subheader('Grafici delle previsioni')
                    figs = plot_predictions(predictions, hydro_features, OUTPUT_WINDOW, start_time)

                    # Visualizzazione grafici
                    for i, fig in enumerate(figs):
                        st.pyplot(fig)
                        sensor_name = hydro_features[i].replace(' ', '_').replace('/', '_')
                        st.markdown(get_image_download_link(fig, f"{sensor_name}.png", f"il grafico di {hydro_features[i]}"), unsafe_allow_html=True)

elif page == 'Simulazione':
    st.header('Simulazione Idrologica')
    st.write('Inserisci i valori per simulare uno scenario idrologico')

    if df is None or model is None or scaler_features is None or scaler_targets is None:
        st.warning("Attenzione: Alcuni file necessari non sono stati caricati correttamente. La simulazione potrebbe non funzionare.")
    else:
        # Opzioni per la simulazione
        sim_method = st.radio(
            "Metodo di simulazione",
            ['Modifica dati recenti', 'Inserisci dati orari', 'Inserisci manualmente tutti i valori']
        )

        if sim_method == 'Modifica dati recenti':
            # Prendiamo i dati recenti come base
            recent_data = df.iloc[-INPUT_WINDOW:][feature_columns].copy() # USA INPUT_WINDOW MODIFICATO

            # Permettiamo all'utente di modificare la pioggia
            st.subheader('Modifica valori di pioggia')
            rain_multiplier = st.slider('Fattore moltiplicativo pioggia', 0.0, 5.0, 1.0, 0.1)

            # Modifichiamo i valori di pioggia
            for col in rain_features:
                recent_data[col] = recent_data[col] * rain_multiplier

            # Permettiamo all'utente di modificare l'umidità
            st.subheader('Modifica valori di umidità')
            humidity_value = st.slider('Umidità (%)', 0.0, 100.0, float(recent_data[humidity_feature[0]].mean()), 0.5)
            recent_data[humidity_feature[0]] = humidity_value

            # Prendiamo i valori modificati
            sim_data = recent_data.values

        elif sim_method == 'Inserisci dati orari':
            st.subheader('Inserisci dati per ogni ora (6 ore precedenti)') # AGGIORNATO TESTO ORE

            # Creiamo un dataframe vuoto per i dati della simulazione
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns)))

            # Opzioni per la compilazione rapida
            st.subheader("Strumenti di compilazione rapida")
            quick_fill_col1, quick_fill_col2 = st.columns(2)

            with quick_fill_col1:
                # Opzioni per compilare rapidamente pioggia
                st.write("Scenario di pioggia")
                rain_scenario = st.selectbox(
                    "Seleziona uno scenario predefinito",
                    ["Nessuna pioggia", "Pioggia leggera", "Pioggia moderata", "Pioggia intensa", "Evento estremo"]
                )

                rain_values = {
                    "Nessuna pioggia": 0.0,
                    "Pioggia leggera": 2.0,
                    "Pioggia moderata": 5.0,
                    "Pioggia intensa": 15.0,
                    "Evento estremo": 30.0
                }

                rain_duration = st.slider("Durata pioggia (ore)", 0, 6, 3) # AGGIORNATO RANGE ORE
                rain_start = st.slider("Ora di inizio pioggia", 0, 5, 0) # AGGIORNATO RANGE ORE

                apply_rain = st.button("Applica scenario di pioggia")

            with quick_fill_col2:
                # Opzioni per compilare rapidamente umidità
                st.write("Umidità del terreno")
                humidity_preset = st.selectbox(
                    "Seleziona condizione di umidità",
                    ["Molto secco", "Secco", "Normale", "Umido", "Saturo"]
                )

                humidity_values = {
                    "Molto secco": 20.0,
                    "Secco": 40.0,
                    "Normale": 60.0,
                    "Umido": 80.0,
                    "Saturo": 95.0
                }

                apply_humidity = st.button("Applica umidità")

            # Creiamo tabs per separare i diversi tipi di dati
            data_tabs = st.tabs(["Pioggia", "Umidità", "Idrometri"])

            # Utilizziamo session_state per mantenere i valori tra le interazioni
            if 'rain_data' not in st.session_state:
                st.session_state.rain_data = np.zeros((INPUT_WINDOW, len(rain_features)))
            if 'humidity_data' not in st.session_state:
                st.session_state.humidity_data = np.zeros((INPUT_WINDOW, len(humidity_feature)))
            if 'hydro_data' not in st.session_state:
                st.session_state.hydro_data = np.zeros((INPUT_WINDOW, len(hydro_features)))

            # Riferimenti più corti per maggiore leggibilità
            rain_data = st.session_state.rain_data
            humidity_data = st.session_state.humidity_data
            hydro_data = st.session_state.hydro_data

            # Se l'utente ha cliccato su applica scenario di pioggia
            if apply_rain:
                for h in range(rain_duration):
                    hour_idx = (rain_start + h) % 6 # AGGIORNATO MODULO
                    if hour_idx < INPUT_WINDOW:
                        for i in range(len(rain_features)):
                            rain_data[hour_idx, i] = rain_values[rain_scenario]

            # Se l'utente ha cliccato su applica umidità
            if apply_humidity:
                for h in range(INPUT_WINDOW):
                    humidity_data[h, 0] = humidity_values[humidity_preset]

            # Tab per la pioggia
            with data_tabs[0]:
                st.write("Inserisci i valori di pioggia per ogni ora (mm)")

                # Creiamo un layout a griglia per l'inserimento dei dati orari
                num_cols = 3  # AGGIORNATO NUM COLONNE GRIGLIA
                num_rows = math.ceil(INPUT_WINDOW / num_cols)

                for feature_idx, feature in enumerate(rain_features):
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"rain_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = rain_data[hour_idx, feature_idx]

                                    # Aggiorniamo il valore nel session_state se è stato modificato dallo scenario
                                    if apply_rain and hour_idx >= rain_start and hour_idx < rain_start + rain_duration:
                                        st.session_state[input_key] = rain_values[rain_scenario]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        0.0, 100.0,
                                        st.session_state[input_key], 0.5,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    rain_data[hour_idx, feature_idx] = value

            # Tab per l'umidità
            with data_tabs[1]:
                st.write("Inserisci i valori di umidità per ogni ora (%)")

                for feature_idx, feature in enumerate(humidity_feature):
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"humidity_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = humidity_data[hour_idx, feature_idx]

                                    # Aggiorniamo il valore nel session_state se è stato modificato dallo scenario
                                    if apply_humidity:
                                        st.session_state[input_key] = humidity_values[humidity_preset]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        0.0, 100.0,
                                        st.session_state[input_key], 0.5,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    humidity_data[hour_idx, feature_idx] = value

            # Tab per gli idrometri
            with data_tabs[2]:
                st.write("Inserisci i livelli idrometrici per ogni ora (m)")

                for feature_idx, feature in enumerate(hydro_features):
                    st.write(f"### {feature}")

                    # Aggiungiamo un modo per impostare un valore costante
                    const_col1, const_col2 = st.columns([3, 1])

                    # Chiave univoca per ogni valore costante
                    const_key = f"const_{feature_idx}"
                    if const_key not in st.session_state:
                        st.session_state[const_key] = 0.0

                    with const_col1:
                        const_value = st.number_input(
                            f"Valore costante per {feature}",
                            -1.0, 10.0,
                            st.session_state[const_key],
                            0.01,
                            key=const_key
                        )

                    # Creiamo una funzione di callback per applicare il valore costante
                    def apply_constant_value(feature_idx=feature_idx, value=const_value):
                        for h in range(INPUT_WINDOW):
                            st.session_state.hydro_data[h, feature_idx] = value
                            input_key = f"hydro_{feature_idx}_{h}" # Generate input key
                            st.session_state[input_key] = value # Update input widget session state

                    with const_col2:
                        if st.button(f"Applica a tutte le ore", key=f"apply_const_{feature_idx}", on_click=apply_constant_value):
                            pass # Callback già gestisce l'aggiornamento

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"hydro_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = hydro_data[hour_idx, feature_idx]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        -1.0, 10.0,
                                        st.session_state[input_key], 0.01,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    hydro_data[hour_idx, feature_idx] = value

            # Componiamo i dati per la simulazione
            rain_offset = 0
            humidity_offset = len(rain_features)
            hydro_offset = humidity_offset + len(humidity_feature)

            for h in range(INPUT_WINDOW):
                # Copiamo i dati di pioggia
                for i in range(len(rain_features)):
                    sim_data[h, rain_offset + i] = rain_data[h, i]

                # Copiamo i dati di umidità
                for i in range(len(humidity_feature)):
                    sim_data[h, humidity_offset + i] = humidity_data[h, i]

                # Copiamo i dati degli idrometri
                for i in range(len(hydro_features)):
                    sim_data[h, hydro_offset + i] = hydro_data[h, i]

            # Visualizziamo un'anteprima dei dati
            st.subheader("Anteprima dei dati inseriti")
            preview_df = pd.DataFrame(sim_data, columns=feature_columns)
            preview_df.index = [f"Ora {i}" for i in range(INPUT_WINDOW)]
            st.dataframe(preview_df.round(2))

        else:  # Inserimento manuale completo
            st.subheader('Inserisci valori per ogni parametro')

            # Creiamo un dataframe vuoto per i dati della simulazione
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns)))

            # Raggruppiamo i controlli per tipo di sensore
            with st.expander("Imposta valori di pioggia"):
                for i, feature in enumerate(rain_features):
                    value = st.number_input(f'{feature} (mm)', 0.0, 100.0, 0.0, 0.5)
                    sim_data[:, i] = value

            with st.expander("Imposta valore di umidità"):
                value = st.number_input(f'{humidity_feature[0]} (%)', 0.0, 100.0, 50.0, 0.5)
                sim_data[:, len(rain_features)] = value

            with st.expander("Imposta livelli idrometrici"):
                offset = len(rain_features) + len(humidity_feature)
                for i, feature in enumerate(hydro_features):
                    value = st.number_input(f'{feature} (m)', -1.0, 10.0, 0.0, 0.01)
                    sim_data[:, offset + i] = value

        # Bottone per eseguire la simulazione
        if st.button('Esegui simulazione', type="primary"):
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile eseguire la simulazione.")
            else:
                with st.spinner('Simulazione in corso...'):
                    # Previsione
                    predictions = predict(model, sim_data, scaler_features, scaler_targets, hydro_features, device, OUTPUT_WINDOW)

                    # Visualizzazione dei risultati
                    st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore')

                    # Creazione dataframe risultati
                    current_time = datetime.now()
                    prediction_times = [current_time + timedelta(hours=i) for i in range(OUTPUT_WINDOW)]
                    results_df = pd.DataFrame(predictions, columns=hydro_features)
                    results_df['Ora previsione'] = prediction_times
                    results_df = results_df[['Ora previsione'] + hydro_features]

                    # Visualizzazione tabella risultati
                    st.dataframe(results_df.round(3))

                    # Download dei risultati
                    st.markdown(get_table_download_link(results_df), unsafe_allow_html=True)

                    # Grafici per ogni sensore
                    st.subheader('Grafici delle previsioni')

                    # Creiamo una visualizzazione che mostri sia i dati inseriti che le previsioni
                    for i, feature in enumerate(hydro_features):
                        fig, ax = plt.subplots(figsize=(10, 6))

                        # Indice per i dati degli idrometri nel sim_data
                        hydro_idx = len(rain_features) + len(humidity_feature) + i

                        # Dati storici (input)
                        input_times = [current_time - timedelta(hours=INPUT_WINDOW - h) for h in range(INPUT_WINDOW)]
                        ax.plot(input_times, sim_data[:, hydro_idx], 'b-', label='Dati inseriti')

                        # Dati previsti (output)
                        ax.plot(prediction_times, predictions[:, i], 'r-', label='Previsione')

                        # Linea verticale per separare dati storici e previsione
                        ax.axvline(x=current_time, color='black', linestyle='--')
                        ax.annotate('Ora attuale', (current_time, ax.get_ylim()[0]),
                                    xytext=(10, 10), textcoords='offset points')

                        ax.set_title(f'Idrometro: {feature}')
                        ax.set_xlabel('Data/Ora')
                        ax.set_ylabel('Livello (m)')
                        ax.legend()
                        ax.grid(True)

                        # Formattazione delle date sull'asse x
                        plt.xticks(rotation=45)
                        plt.tight_layout()

                        st.pyplot(fig)
                        sensor_name = feature.replace(' ', '_').replace('/', '_')
                        st.markdown(get_image_download_link(fig, f"sim_{sensor_name}.png", f"il grafico di {feature}"), unsafe_allow_html=True)

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')

    if df is None:
        st.warning("Attenzione: File dati non caricato correttamente. L'analisi dati storici non è disponibile.")
    else:
        # Selezione dell'intervallo di date
        st.subheader('Seleziona l\'intervallo di date')
        min_date_analisi = df['Data e Ora'].min().date()
        max_date_analisi = df['Data e Ora'].max().date()

        col1, col2 = st.columns(2)
        with col1:
            start_date_analisi = st.date_input('Data inizio', min_date_analisi, min_value=min_date_analisi, max_value=max_date_analisi)
        with col2:
            end_date_analisi = st.date_input('Data fine', max_date_analisi, min_value=min_date_analisi, max_value=max_date_analisi)

        # Filtraggio dei dati
        mask = (df['Data e Ora'].dt.date >= start_date_analisi) & (df['Data e Ora'].dt.date <= end_date_analisi)
        filtered_df = df.loc[mask]

        if len(filtered_df) > 0:
            st.success(f'Trovate {len(filtered_df)} righe nel periodo selezionato')

            # Tipo di analisi
            analysis_type = st.selectbox(
                'Seleziona il tipo di analisi',
                ['Andamento temporale', 'Correlazione tra sensori', 'Statistiche descrittive']
            )

            if analysis_type == 'Andamento temporale':
                st.subheader('Andamento temporale dei dati')

                # Selezione dei sensori
                sensor_type = st.radio('Tipo di sensore', ['Idrometri', 'Pluviometri', 'Umidità'])

                if sensor_type == 'Idrometri':
                    sensors = st.multiselect('Seleziona i sensori', hydro_features, default=[hydro_features[0]])
                elif sensor_type == 'Pluviometri':
                    sensors = st.multiselect('Seleziona i sensori', rain_features, default=[rain_features[0]])
                else:  # Umidità
                    sensors = humidity_feature

                if sensors:
                    # Creazione del grafico
                    fig, ax = plt.subplots(figsize=(12, 6))

                    for sensor in sensors:
                        ax.plot(filtered_df['Data e Ora'], filtered_df[sensor], label=sensor)

                    ax.set_xlabel('Data e Ora')
                    ax.set_ylabel('Valore')
                    ax.set_title(f'Andamento temporale {"".join(sensors)} - {start_date_analisi} a {end_date_analisi}')
                    ax.legend()
                    ax.grid(True)
                    plt.xticks(rotation=45)
                    plt.tight_layout()

                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig, "andamento_temporale.png", "questo grafico"), unsafe_allow_html=True)

            elif analysis_type == 'Correlazione tra sensori':
                st.subheader('Analisi di correlazione tra sensori')

                # Selezione delle variabili
                corr_features = st.multiselect(
                    'Seleziona le variabili da analizzare',
                    feature_columns,
                    default=[hydro_features[0], rain_features[0]]
                )

                if len(corr_features) > 1:
                    # Matrice di correlazione
                    corr_matrix = filtered_df[corr_features].corr()

                    # Heatmap
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax, fmt='.2f')
                    plt.title('Matrice di correlazione')
                    plt.tight_layout()

                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig, "correlazione.png", "questa matrice di correlazione"), unsafe_allow_html=True)

                    # Scatterplot
                    if len(corr_features) == 2:
                        fig, ax = plt.subplots(figsize=(10, 6))
                        sns.scatterplot(data=filtered_df, x=corr_features[0], y=corr_features[1], ax=ax)
                        plt.title(f'Scatterplot {corr_features[0]} vs {corr_features[1]}')
                        plt.tight_layout()

                        st.pyplot(fig)
                        st.markdown(get_image_download_link(fig, "scatterplot.png", "questo scatterplot"), unsafe_allow_html=True)

            else:  # Statistiche descrittive
                st.subheader('Statistiche descrittive')

                # Selezione delle variabili
                stat_features = st.multiselect(
                    'Seleziona le variabili da analizzare',
                    feature_columns,
                    default=hydro_features
                )

                if stat_features:
                    # Statistiche descrittive
                    stats_df = filtered_df[stat_features].describe().T
                    st.dataframe(stats_df.round(3))

                    # Download statistiche
                    st.markdown(get_table_download_link(stats_df.reset_index().rename(columns={'index': 'Sensore'})), unsafe_allow_html=True)

                    # Boxplot
                    fig, ax = plt.subplots(figsize=(12, 6))
                    filtered_df[stat_features].boxplot(ax=ax)
                    plt.title('Boxplot delle variabili selezionate')
                    plt.xticks(rotation=90)
                    plt.tight_layout()

                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig, "boxplot.png", "questo boxplot"), unsafe_allow_html=True)
        else:
            st.warning('Nessun dato disponibile nel periodo selezionato')

elif page == 'Allenamento Modello':
    st.header('Allenamento Modello')

    if df is None:
        st.warning("Attenzione: File dati non caricato correttamente. L'allenamento del modello potrebbe non funzionare correttamente.")
    else:

        # Istruzioni iniziali
        st.info('Questa sezione permette di addestrare o riaddestrare il modello di previsione idrologica.')

        # Opzioni per l'upload dei dati
        data_option = st.radio(
            "Dati di addestramento",
            ['Usa dati caricati', 'Carica nuovo file CSV']
        )

        if data_option == 'Carica nuovo file CSV':
            training_data = st.file_uploader('Carica il CSV per l\'addestramento', type=['csv'])
            if training_data:
                try:
                    train_df = pd.read_csv(training_data, sep=';', parse_dates=['Data e Ora'], decimal=',')
                    train_df['Data e Ora'] = pd.to_datetime(train_df['Data e Ora'], format='%d/%m/%Y %H:%M')
                    st.success(f'File caricato: {len(train_df)} righe')
                except Exception as e:
                    st.error(f'Errore nel caricamento del file: {e}')
                    train_df = None
            else:
                st.warning('Carica un file CSV per procedere')
                train_df = None
        else:
            train_df = df.copy()
            st.success(f'Utilizzo dei dati già caricati: {len(train_df)} righe')

        if train_df is not None:
            # Selezione dell'intervallo temporale per l'addestramento
            st.subheader('Seleziona l\'intervallo per l\'addestramento')

            min_date_train = train_df['Data e Ora'].min().date()
            max_date_train = train_df['Data e Ora'].max().date()

            col1, col2 = st.columns(2)
            with col1:
                train_start_date = st.date_input('Data inizio addestramento', min_date_train, min_value=min_date_train, max_value=max_date_train)
            with col2:
                train_end_date = st.date_input('Data fine addestramento', max_date_train, min_value=min_date_train, max_value=max_date_train)

            # Filtraggio dei dati per l'addestramento
            train_mask = (train_df['Data e Ora'].dt.date >= train_start_date) & (train_df['Data e Ora'].dt.date <= train_end_date)
            train_filtered_df = train_df.loc[train_mask]

            if len(train_filtered_df) < INPUT_WINDOW + OUTPUT_WINDOW: # USA INPUT_WINDOW MODIFICATO
                st.error(f'Servono almeno {INPUT_WINDOW + OUTPUT_WINDOW} righe di dati per l\'addestramento. Hai selezionato solo {len(train_filtered_df)} righe.') # USA INPUT_WINDOW MODIFICATO
            else:
                st.success(f'Dati selezionati per l\'addestramento: {len(train_filtered_df)} righe')

                # Divisione train/validation
                st.subheader('Parametri di suddivisione train/validation')
                val_split = st.slider('Percentuale di dati per la validazione', 5, 30, 20)

                # Parametri del modello
                st.subheader('Parametri del modello')

                col1, col2 = st.columns(2)
                with col1:
                    hidden_size = st.number_input('Dimensione hidden layer', 32, 512, 128, 32)
                    num_layers = st.number_input('Numero di layer LSTM', 1, 5, 2, 1)
                with col2:
                    dropout = st.slider('Dropout', 0.0, 0.5, 0.2, 0.05)
                    learning_rate = st.select_slider('Learning rate', options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)

                # Parametri di addestramento
                st.subheader('Parametri di addestramento')

                col1, col2 = st.columns(2)
                with col1:
                    batch_size = st.select_slider('Batch size', options=[8, 16, 32, 64, 128], value=32)
                    epochs = st.number_input('Numero di epoche', 5, 200, 50, 5)
                with col2:
                    patience = st.number_input('Patience per early stopping', 3, 30, 10, 1)
                    use_scheduler = st.checkbox('Usa learning rate scheduler', value=True)

                # Visualizza informazioni sulla finestra di input modificata
                st.info(f"Il modello utilizzerà {INPUT_WINDOW} ore di dati in input per prevedere le successive {OUTPUT_WINDOW} ore") # USA INPUT_WINDOW MODIFICATO

                # Sezione per il Fine Tuning del modello
                st.subheader('Fine Tuning del Modello')
                do_fine_tuning = st.checkbox('Abilita Fine Tuning', value=False)

                if do_fine_tuning:
                    st.info("Il fine tuning permette di addestrare ulteriormente un modello esistente con nuovi dati")

                    # Carica un modello esistente per il fine tuning
                    uploaded_model = st.file_uploader("Carica un modello salvato (.pt)", type=["pt"])

                    if uploaded_model is not None:
                        # Parametri specifici per il fine tuning
                        col1, col2 = st.columns(2)
                        with col1:
                            ft_learning_rate = st.select_slider('Learning rate per fine tuning',
                                                            options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
                                                            value=0.0001)
                            ft_epochs = st.number_input('Numero di epoche per fine tuning', 5, 100, 20, 5)
                        with col2:
                            ft_batch_size = st.select_slider('Batch size per fine tuning',
                                                          options=[4, 8, 16, 32, 64],
                                                          value=16)
                            ft_freeze_layers = st.checkbox('Congela i primi layer', value=True)
                    else:
                        st.warning("Carica un modello salvato per abilitare il fine tuning")


                # Funzione per preparare i dati di addestramento
                def prepare_training_data(df_train, features, targets, input_window, output_window, val_split):
                    # 1. Creazione delle sequenze di input (X) e output (y).
                    X, y = [], []
                    for i in range(len(df_train) - input_window - output_window + 1):
                        X.append(df_train.iloc[i:i+input_window][features].values)
                        # Correction: Target y should only be the output window
                        y.append(df_train.iloc[i+input_window:i+input_window+output_window][targets].values)
                    X = np.array(X)
                    y = np.array(y)

                    # 2. Normalizzazione dei dati (MinMaxScaler).
                    scaler_features_train = MinMaxScaler()
                    scaler_targets_train = MinMaxScaler()
                    X_flat = X.reshape(-1, X.shape[-1])
                    y_flat = y.reshape(-1, y.shape[-1])
                    X_scaled_flat = scaler_features_train.fit_transform(X_flat)
                    y_scaled_flat = scaler_targets_train.fit_transform(y_flat)
                    X_scaled = X_scaled_flat.reshape(X.shape)
                    y_scaled = y_scaled_flat.reshape(y.shape)

                    # 3. Divisione in set di addestramento e validazione.
                    split_idx = int(len(X_scaled) * (1 - val_split/100))
                    X_train = X_scaled[:split_idx]
                    y_train = y_scaled[:split_idx]
                    X_val = X_scaled[split_idx:]
                    y_val = y_scaled[split_idx:]

                    return X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train


                # Funzione di addestramento
                def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window, hidden_size, num_layers, dropout, batch_size, epochs, learning_rate, patience, use_scheduler, fine_tuning=False, loaded_model=None, freeze_layers=False):
                    # 1. Conversione dei dati in tensori PyTorch.
                    X_train_tensor = torch.FloatTensor(X_train)
                    y_train_tensor = torch.FloatTensor(y_train)
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.FloatTensor(y_val)

                    # 2. Creazione dei DataLoader per gestire i batch.
                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

                    # 3. Definizione del dispositivo (CPU o GPU).
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # 4. Creazione dell'istanza del modello (HydroLSTM) o caricamento del modello esistente per fine tuning
                    if fine_tuning and loaded_model is not None:
                        st.info("Effettuando fine tuning sul modello caricato")
                        model_train = loaded_model

                        # Opzionalmente congela i primi layer del modello
                        if freeze_layers:
                            layers_to_freeze = num_layers - 1  # Congela tutti i layer tranne l'ultimo
                            for i, param in enumerate(model_train.parameters()):
                                if i < layers_to_freeze * 4:  # Ogni layer LSTM ha 4 set di parametri
                                    param.requires_grad = False
                            st.info(f"Primi {layers_to_freeze} layer congelati per il fine tuning")
                    else:
                        model_train = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

                    # 5. Definizione dell'ottimizzatore (Adam).
                    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model_train.parameters()),
                                                 lr=learning_rate)

                    # 6. (Opzionale) Definizione dello scheduler per il learning rate.
                    scheduler = None
                    if use_scheduler:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

                    # 7. Definizione della funzione di perdita (MSELoss).
                    criterion = nn.MSELoss()

                    # 8. Implementazione dell'early stopping.
                    best_val_loss = float('inf')
                    early_stopping_counter = 0
                    best_model_state = None

                    # 9. Ciclo di addestramento (per ogni epoca).
                    train_losses = []
                    val_losses = []
                    progress_bar = st.progress(0)  # Barra di progresso
                    status_text = st.empty()      # Testo di stato
                    loss_chart = st.empty()       # Grafico delle perdite

                    for epoch in range(epochs):
                        model_train.train()  # Imposta il modello in modalità addestramento
                        train_loss = 0

                        for batch_X, batch_y in train_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device) # Sposta i dati sul dispositivo
                            outputs = model_train(batch_X)       # Forward pass
                            loss = criterion(outputs, batch_y)    # Calcolo della perdita
                            optimizer.zero_grad()                # Azzeramento dei gradienti
                            loss.backward()                     # Backward pass (calcolo dei gradienti)
                            optimizer.step()                    # Aggiornamento dei pesi
                            train_loss += loss.item()           # Accumulo della perdita

                        # Fase di validazione (dopo ogni epoca)
                        model_train.eval()  # Imposta il modello in modalità valutazione
                        val_loss = 0
                        with torch.no_grad(): # Disabilita il calcolo dei gradienti
                            for batch_X, batch_y in val_loader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                                outputs = model_train(batch_X)
                                loss = criterion(outputs, batch_y)
                                val_loss += loss.item()

                        # Calcolo della perdita media per epoca
                        train_loss /= len(train_loader)
                        val_loss /= len(val_loader)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

                        # Aggiornamento dello scheduler (se utilizzato)
                        if scheduler:
                            scheduler.step(val_loss)

                        # Visualizzazione del progresso, grafico delle perdite, early stopping (come spiegato sopra)
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

                        # Grafico delle perdite (creazione e visualizzazione)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(train_losses, label='Train Loss')
                        ax.plot(val_losses, label='Validation Loss')
                        ax.set_xlabel('Epoca')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True)
                        loss_chart.pyplot(fig)
                        plt.close(fig)

                        # Early stopping (controllo e salvataggio del modello migliore)
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stopping_counter = 0
                            best_model_state = model_train.state_dict()  # Salva lo stato del modello
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= patience:
                                status_text.text(f'Early stopping attivato all\'epoca {epoch+1}')
                                break

                    # Carica lo stato del modello migliore (quello con la validation loss minima)
                    model_train.load_state_dict(best_model_state)
                    return model_train, train_losses, val_losses, best_val_loss


                # Funzione per caricare un modello salvato
                def load_model_for_fine_tuning(uploaded_model, input_size, output_size, output_window, hidden_size, num_layers, dropout):
                    # Leggi il file caricato
                    model_bytes = uploaded_model.read()

                    # Crea una nuova istanza del modello
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

                    # Carica lo stato del modello
                    model.load_state_dict(torch.load(io.BytesIO(model_bytes), map_location=device))

                    return model

                # Logica per avviare l'addestramento e salvare i risultati/modello.
                if 'training_completed' not in st.session_state:
                    st.session_state['training_completed'] = False

                if st.button('Avvia Addestramento') or st.session_state['training_completed']:
                    if not st.session_state['training_completed']: # Controlla se l'addestramento è già stato completato
                        with st.spinner('Preparazione dei dati in corso...'):
                            # Verifica della presenza di tutti i dati necessari
                            missing_columns = [col for col in feature_columns if col not in train_filtered_df.columns] # DEFINISCO missing_columns QUI PRIMA DELL'USO
                            if missing_columns:
                              st.error(f'Mancano le seguenti colonne nei dati: {missing_columns}')
                            else:
                              # Preparazione dei dati
                              X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train = prepare_training_data(
                                  train_filtered_df,
                                  feature_columns,
                                  hydro_features,
                                  INPUT_WINDOW,
                                  OUTPUT_WINDOW,
                                  val_split
                              )

                              st.session_state['X_val'] = X_val # Salva nello stato della sessione
                              st.session_state['y_val'] = y_val # Salva nello stato della sessione
                              st.session_state['scaler_targets_train'] = scaler_targets_train # Salva nello stato della sessione
                              st.session_state['hydro_features'] = hydro_features # Salva nello stato della sessione

                              st.success(f'Dati preparati: {X_train.shape[0]} esempi di addestramento, {X_val.shape[0]} esempi di validazione')

                              with st.spinner('Addestramento in corso...'):


                                # Gestione del fine tuning se abilitato
                                loaded_model = None
                                if do_fine_tuning and uploaded_model is not None:
                                    loaded_model = load_model_for_fine_tuning(
                                        uploaded_model,
                                        len(feature_columns), len(hydro_features), OUTPUT_WINDOW,
                                        hidden_size, num_layers, dropout
                                    )

                                    # Chiamata alla funzione train_model con parametri di fine tuning
                                    trained_model, train_losses, val_losses, best_val_loss = train_model(
                                        X_train, y_train, X_val, y_val,
                                        len(feature_columns), len(hydro_features), OUTPUT_WINDOW,
                                        hidden_size, num_layers, dropout, ft_batch_size, ft_epochs,
                                        ft_learning_rate, patience, use_scheduler,
                                        fine_tuning=True, loaded_model=loaded_model, freeze_layers=ft_freeze_layers)
                                else:
                                    # Chiamata alla funzione train_model per addestramento normale
                                    trained_model, train_losses, val_losses, best_val_loss = train_model(
                                        X_train, y_train, X_val, y_val,
                                        len(feature_columns), len(hydro_features), OUTPUT_WINDOW,
                                        hidden_size, num_layers, dropout, batch_size, epochs,
                                        learning_rate, patience, use_scheduler)

                                # Salvataggio dei risultati e del modello nello stato della sessione.
                                st.session_state['trained_model_state'] = trained_model.state_dict() # Salva i pesi del modello
                                st.session_state['train_losses'] = train_losses
                                st.session_state['val_losses'] = val_losses
                                st.session_state['best_val_loss'] = best_val_loss
                                st.session_state['feature_columns'] = feature_columns # Salva feature columns per caricare il modello
                                st.session_state['output_window'] = OUTPUT_WINDOW # Salva output_window per caricare il modello
                                st.session_state['input_window'] = INPUT_WINDOW # Salva anche input_window per riferimento
                                st.session_state['hydro_features'] = hydro_features # Salva hydro_features per caricare il modello
                                st.session_state['scaler_features_train'] = scaler_features_train # Salva scaler features
                                st.session_state['training_completed'] = True # Imposta il flag a True

                                # Indicazione del tipo di addestramento completato
                                if do_fine_tuning and uploaded_model is not None:
                                    st.session_state['training_type'] = "fine_tuning"
                                else:
                                    st.session_state['training_type'] = "standard"


                    # Visualizzazione dei risultati, salvataggio del modello addestrato e dei suoi parametri, test sul validation set.
                    if st.session_state['training_completed']:
                      # ... (visualizzazione grafico perdite, salvataggio modello e scaler, ...)
                      if st.session_state.get('training_type') == "fine_tuning":
                          st.success(f'Fine Tuning completato! Miglior loss di validazione: {st.session_state["best_val_loss"]:.6f}')
                      else:
                          st.success(f'Addestramento completato! Miglior loss di validazione: {st.session_state["best_val_loss"]:.6f}')

                      # Salvataggio automatico del modello
                      st.subheader('Salvataggio automatico del modello completato')
                      model_name = 'hydro_model_new.pth'
                      scaler_features_name = 'scaler_features_new.joblib'
                      scaler_targets_name = 'scaler_targets_new.joblib'

                      st.write("Salvataggio modello e scaler in corso...")
                      from datetime import datetime
                      import os

                      # Creazione directory per i modelli se non esiste
                      os.makedirs('models', exist_ok=True)

                      # Timestamp per il nome del file
                      timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

                      # Percorsi dei file
                      model_path = os.path.join('models', f'{timestamp}_{model_name}')
                      scaler_features_path = os.path.join('models', f'{timestamp}_{scaler_features_name}')
                      scaler_targets_path = os.path.join('models', f'{timestamp}_{scaler_targets_name}')


                      try:
                          # Imposta device per il salvataggio
                          device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Definisci device qui

                          # Salvataggio del modello
                          trained_model_instance = HydroLSTM(len(st.session_state['feature_columns']), hidden_size, len(st.session_state['hydro_features']), st.session_state['output_window'], num_layers, dropout).to(device)
                          trained_model_instance.load_state_dict(st.session_state['trained_model_state'])
                          torch.save(st.session_state['trained_model_state'], model_path)
                          st.success(f'Modello salvato con successo in: {model_path}') # Success message con percorso

                          # Salvataggio degli scaler CORRETTI (quelli di training)
                          joblib.dump(st.session_state['scaler_features_train'], scaler_features_path)
                          joblib.dump(st.session_state['scaler_targets_train'], scaler_targets_path)

                          st.success(f'Scaler salvati con successo in cartella "models"!')
                      except Exception as e:
                          st.error(f'Errore durante il salvataggio: {e}')
                          st.error(f'Dettagli errore: {e}') # Print error details


                      # Salvataggio dei parametri del modello
                      params_path = os.path.join('models', f'{timestamp}_model_params.txt')
                      with open(params_path, 'w') as f:
                          f.write(f'Data e ora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                          f.write(f'Periodo di addestramento: {train_start_date} - {train_end_date}\n')
                          f.write(f'Numero di esempi: {len(st.session_state["X_val"]) + len(st.session_state["y_val"])}\n')
                          f.write(f'Split validazione: {val_split}%\n')
                          f.write(f'Hidden size: {hidden_size}\n')
                          f.write(f'Num layers: {num_layers}\n')
                          f.write(f'Dropout: {dropout}\n')
                          f.write(f'Learning rate: {learning_rate}\n')
                          f.write(f'Batch size: {batch_size}\n')
                          f.write(f'Epochs: {epochs}\n')
                          f.write(f'Best validation loss: {st.session_state["best_val_loss"]:.6f}\n')

                      # Link di download
                      st.markdown(f'### Link per il download')

                      def get_file_download_link(file_path, link_text):
                          with open(file_path, 'rb') as f:
                              data = f.read()
                          b64 = base64.b64encode(data).decode()
                          href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
                          return href

                      st.markdown(get_file_download_link(model_path, 'Scarica il modello'), unsafe_allow_html=True)
                      st.markdown(get_file_download_link(scaler_features_path, 'Scarica lo scaler features'), unsafe_allow_html=True)
                      st.markdown(get_file_download_link(scaler_targets_path, 'Scarica lo scaler targets'), unsafe_allow_html=True)

                      # Test delle prestazioni
                      st.subheader('Test delle prestazioni del modello')

                      if st.button('Esegui test sul set di validazione'):
                        if 'X_val' in st.session_state and st.session_state['X_val'] is not None: # Check if X_val exists
                          with st.spinner('Test in corso...'):
                              # Recupera dati dal session state
                              X_val = st.session_state['X_val']
                              y_val = st.session_state['y_val']
                              scaler_targets_train = st.session_state['scaler_targets_train']
                              hydro_features = st.session_state['hydro_features']
                              trained_model_state = st.session_state['trained_model_state']
                              feature_columns_test = st.session_state['feature_columns']
                              output_window_test = st.session_state['output_window']
                              input_window_test = st.session_state['input_window'] # RECUPERA INPUT_WINDOW DAL SESSION STATE


                              # Conversione in tensori
                              X_val_tensor = torch.FloatTensor(X_val)
                              y_val_tensor = torch.FloatTensor(y_val)

                              # Valutazione sul set di validazione
                              device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                              trained_model_test = HydroLSTM(len(feature_columns_test), hidden_size, len(hydro_features), OUTPUT_WINDOW, num_layers, dropout).to(device) # Re-inizializza modello
                              trained_model_test.load_state_dict(trained_model_state) # Carica i pesi
                              trained_model_test.eval()


                              # Previsioni
                              with torch.no_grad():
                                  y_pred = trained_model_test(X_val_tensor.to(device))
                                  y_pred = y_pred.cpu().numpy()

                              # Reshape per la denormalizzazione
                              y_pred_flat = y_pred.reshape(-1, y_pred.shape[-1])
                              y_val_flat = y_val.reshape(-1, y_val.shape[-1])

                              # Denormalizzazione - USA SCALER DI TRAINING!
                              y_pred_denorm = scaler_targets_train.inverse_transform(y_pred_flat).reshape(y_pred.shape)
                              y_val_denorm = scaler_targets_train.inverse_transform(y_val_flat).reshape(y_val.shape)

                              # Calcolo degli errori
                              mae = np.mean(np.abs(y_pred_denorm - y_val_denorm), axis=(0, 1))
                              rmse = np.sqrt(np.mean((y_pred_denorm - y_val_denorm)**2, axis=(0, 1)))

                              # Visualizzazione degli errori
                              error_df = pd.DataFrame({
                                  'Sensore': hydro_features,
                                  'MAE [m]': mae,
                                  'RMSE [m]': rmse
                              })

                              st.dataframe(error_df.round(3))

                              # Visualizzazione di alcune previsioni di esempio
                              st.subheader('Esempio di previsioni')
                              # ...
                        else:
                          st.error("Dati di validazione non disponibili. Esegui prima l'addestramento.")

# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
