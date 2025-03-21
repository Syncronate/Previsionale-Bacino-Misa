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
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"Errore durante il caricamento dei pesi del modello: {e}")
        return None, device # Return None model and device

    model.eval()
    return model, device

# Funzione per caricare gli scaler
@st.cache_resource
def load_scalers(scaler_features_path, scaler_targets_path):
    try:
        scaler_features = joblib.load(scaler_features_path)
        scaler_targets = joblib.load(scaler_targets_path)
        return scaler_features, scaler_targets
    except Exception as e:
        st.error(f"Errore durante il caricamento degli scaler: {e}")
        return None, None # Return None scalers

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
    if model is None or scaler_features is None or scaler_targets is None:
        st.error("Modello o scaler non caricati correttamente. Impossibile fare previsioni.")
        return None

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
df = None # Initialize df to None
try:
    if use_demo_files:
        df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    elif data_file is not None: # Check if data_file is loaded
        df = pd.read_csv(data_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
    if df is not None:
        st.sidebar.success(f'Dati caricati: {len(df)} righe')
except Exception as e:
    st.sidebar.error(f'Errore nel caricamento dei dati: {e}')


# Estrazione delle caratteristiche (originali - potremmo non usarle direttamente nella simulazione modificata)
rain_features_original = [
    'Cumulata Sensore 1295 (Arcevia)',
    'Cumulata Sensore 2637 (Bettolelle)',
    'Cumulata Sensore 2858 (Barbara)',
    'Cumulata Sensore 2964 (Corinaldo)'
]

humidity_feature_original = ['Umidita\' Sensore 3452 (Montemurello)']

hydro_features_original = [
    'Livello Idrometrico Sensore 1008 [m] (Serra dei Conti)',
    'Livello Idrometrico Sensore 1112 [m] (Bettolelle)',
    'Livello Idrometrico Sensore 1283 [m] (Corinaldo/Nevola)',
    'Livello Idrometrico Sensore 3072 [m] (Pianello di Ostra)',
    'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'
]

feature_columns_original = rain_features_original + humidity_feature_original + hydro_features_original

# Definisci le feature columns e hydro_features iniziali (prima della modifica)
feature_columns = feature_columns_original
hydro_features = hydro_features_original

model = None
scaler_features = None
scaler_targets = None

# Caricamento del modello e degli scaler SOLO se i file sono stati caricati o si usano quelli demo
if use_demo_files or (data_file and model_file and scaler_features_file and scaler_targets_file):
    try:
        if use_demo_files:
            model, device = load_model(MODEL_PATH, 8, 1, OUTPUT_WINDOW) # MODIFIED: input_size=8, output_size=1
            scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
        else:
            model_bytes = io.BytesIO(model_file.read())
            model, device = load_model(model_bytes, 8, 1, OUTPUT_WINDOW) # MODIFIED: input_size=8, output_size=1
            scaler_features, scaler_targets = load_scalers(scaler_features_file, scaler_targets_file)

        if model is not None and scaler_features is not None and scaler_targets is not None: # Check if loading was successful
            st.sidebar.success('Modello e scaler caricati con successo')
    except Exception as e:
        st.sidebar.error(f'Errore nel caricamento del modello o degli scaler: {e}')

# Menu principale
st.sidebar.header('Menu')
page = st.sidebar.radio('Scegli una funzionalità',
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'])

# --- FUNZIONE modifica_modello_previsionale() e sue componenti ---
# Modifica: imposta il target solo all'idrometro di Bettolelle
# Trova l'indice dell'idrometro di Bettolelle e di Ponte Garibaldi
def modifica_modello_previsionale():
    # --- MODIFICHE AL CODICE ORIGINALE ---

    # 1. Modifica dei target di previsione - solo idrometro di Bettolelle (1112)
    target_features_mod = ['Livello Idrometrico Sensore 1112 [m] (Bettolelle)']  # Solo Bettolelle

    # 2. Definizione degli input (tutti tranne Ponte Garibaldi)
    # Supponiamo che 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)' sia Ponte Garibaldi
    ponte_garibaldi_id_name = 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'

    # Funzione per ottenere tutte le feature tranne Ponte Garibaldi
    def get_input_features(df):
        # Ottieni tutte le colonne con prefisso "Cumulata Sensore" (tutte le cumulate)
        cumulate_columns = [col for col in df.columns if col.startswith('Cumulata Sensore')]

        # Ottieni tutte le colonne con prefisso "Umidita'" (tutti i sensori di umidità)
        umidita_columns = [col for col in df.columns if col.startswith('Umidita\' Sensore')]

        # Ottieni tutte le colonne con prefisso "Livello Idrometrico Sensore" eccetto Ponte Garibaldi e Bettolelle
        idrometro_columns = [col for col in df.columns if col.startswith('Livello Idrometrico Sensore')
                            and col != ponte_garibaldi_id_name and col != 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)']

        # Combinazione di tutte le colonne di input
        input_features_mod = cumulate_columns + umidita_columns + idrometro_columns

        return input_features_mod

    # 3. Modifica della funzione di preparazione dati per utilizzare queste nuove feature
    def prepare_training_data_modificato(df_train, val_split):
        # Ottieni le feature di input
        feature_columns_mod = get_input_features(df_train)

        # Creazione delle sequenze di input (X) e output (y)
        X, y = [], []
        for i in range(len(df_train) - INPUT_WINDOW - OUTPUT_WINDOW + 1):
            X.append(df_train.iloc[i:i+INPUT_WINDOW][feature_columns_mod].values)
            y.append(df_train.iloc[i+INPUT_WINDOW:i+INPUT_WINDOW+OUTPUT_WINDOW][target_features_mod].values)
        X = np.array(X)
        y = np.array(y)

        # Normalizzazione dei dati
        scaler_features_train = MinMaxScaler()
        scaler_targets_train = MinMaxScaler()
        X_flat = X.reshape(-1, X.shape[-1])
        y_flat = y.reshape(-1, y.shape[-1])
        X_scaled_flat = scaler_features_train.fit_transform(X_flat)
        y_scaled_flat = scaler_targets_train.fit_transform(y_flat)
        X_scaled = X_scaled_flat.reshape(X.shape)
        y_scaled = y_scaled_flat.reshape(y.shape)

        # Divisione in set di addestramento e validazione
        split_idx = int(len(X_scaled) * (1 - val_split/100))
        X_train = X_scaled[:split_idx]
        y_train = y_scaled[:split_idx]
        X_val = X_scaled[split_idx:]
        y_val = y_scaled[split_idx:]

        return X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train, feature_columns_mod, target_features_mod

    return get_input_features, prepare_training_data_modificato, target_features_mod

get_input_features_func, _, target_features_modificato = modifica_modello_previsionale() # Ottieni solo get_input_features e target_features_modificato

if df is not None: # Only calculate if df is loaded
    feature_columns_mod = get_input_features_func(pd.DataFrame(columns=df.columns if df is not None else feature_columns_original)) # needed to have columns to pass
else:
    feature_columns_mod = get_input_features_func(pd.DataFrame(columns=feature_columns_original)) # needed to have columns to pass in case df is None

# --- FINE FUNZIONE modifica_modello_previsionale() ---


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
            for feature in rain_features_original: # use original rain features
                rain_data.append({'Sensore': feature, 'Valore [mm]': last_data[feature]})
            st.dataframe(pd.DataFrame(rain_data).round(2))

            # Umidità
            st.subheader('Umidità attuale')
            st.write(f"{humidity_feature_original[0]}: {last_data[humidity_feature_original[0]]:.1f}%") # use original humidity feature

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

                    if predictions is not None: # Check if prediction was successful
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
    st.header('Simulazione Idrologica (Modello Aggiornato)') # Titolo modificato per indicare il modello aggiornato
    st.write('Inserisci i valori per simulare uno scenario idrologico con il modello aggiornato (Bettolelle)') # Descrizione modificata

    if df is None or model is None or scaler_features is None or scaler_targets is None:
        st.warning("Attenzione: Alcuni file necessari non sono stati caricati correttamente. La simulazione potrebbe non funzionare.")
    else:
        # Opzioni per la simulazione
        sim_method = st.radio(
            "Metodo di simulazione",
            ['Inserisci dati orari', 'Inserisci manualmente tutti i valori'] # Rimosso 'Modifica dati recenti' per semplificare e focalizzarsi sul nuovo modello
        )

        if sim_method == 'Inserisci dati orari':
            st.subheader(f'Inserisci dati per ogni ora ({INPUT_WINDOW} ore precedenti) per la simulazione del modello Bettolelle') # Testo modificato

            # Creiamo un dataframe vuoto per i dati della simulazione con le *nuove* feature
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns_mod))) # Usa feature_columns_mod

            # Opzioni per la compilazione rapida (pioggia e umidità) - ADATTATO ALLE NUOVE FEATURE
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
                    "Pioggia intensa": 5.0, # Valore ridotto per pioggia intensa per il modello modificato
                    "Evento estremo": 10.0 # Valore ridotto per evento estremo per il modello modificato
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

            # Creiamo tabs per separare i diversi tipi di dati (ADATTATO ALLE NUOVE FEATURE)
            data_tabs = st.tabs(["Cumulate Pioggia", "Umidità", "Idrometri (Input)"]) # Tab idrometri rinominato per chiarezza

            # Utilizziamo session_state per mantenere i valori tra le interazioni (ADATTATO ALLE NUOVE FEATURE)
            if 'rain_data_mod' not in st.session_state:
                st.session_state.rain_data_mod = np.zeros((INPUT_WINDOW, sum([1 for col in feature_columns_mod if col.startswith('Cumulata Sensore')]))) # Corretto per contare solo le cumulate
            if 'humidity_data_mod' not in st.session_state:
                st.session_state.humidity_data_mod = np.zeros((INPUT_WINDOW, sum([1 for col in feature_columns_mod if col.startswith('Umidita\' Sensore')]))) # Corretto per contare solo umidità
            if 'hydro_input_data_mod' not in st.session_state: # Rinominato per chiarezza (input idrometri)
                st.session_state.hydro_input_data_mod = np.zeros((INPUT_WINDOW, sum([1 for col in feature_columns_mod if col.startswith('Livello Idrometrico Sensore') and 'Bettolelle' not in col]))) # Corretto per contare idrometri input (escluso Bettolelle e Ponte Garibaldi)

            # Riferimenti più corti per maggiore leggibilità (ADATTATO ALLE NUOVE FEATURE)
            rain_data_mod = st.session_state.rain_data_mod
            humidity_data_mod = st.session_state.humidity_data_mod
            hydro_input_data_mod = st.session_state.hydro_input_data_mod

            # Feature names per tab (ADATTATO ALLE NUOVE FEATURE)
            rain_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Cumulata Sensore')]
            humidity_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Umidita\' Sensore')]
            hydro_input_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Livello Idrometrico Sensore') and 'Bettolelle' not in col]

            # Se l'utente ha cliccato su applica scenario di pioggia
            if apply_rain:
                for h in range(rain_duration):
                    hour_idx = (rain_start + h) % 6 # AGGIORNATO MODULO
                    if hour_idx < INPUT_WINDOW:
                        for i in range(len(rain_features_sim_mod)): # Usa rain_features_sim_mod
                            rain_data_mod[hour_idx, i] = rain_values[rain_scenario]

            # Se l'utente ha cliccato su applica umidità
            if apply_humidity:
                for h in range(INPUT_WINDOW):
                    humidity_data_mod[h, 0] = humidity_values[humidity_preset] # Assume una sola feature di umidità, adatta se necessario

            # Tab per la pioggia (ADATTATO ALLE NUOVE FEATURE)
            with data_tabs[0]:
                st.write("Inserisci i valori di pioggia cumulata per ogni ora (mm)")

                # Creiamo un layout a griglia per l'inserimento dei dati orari
                num_cols = 3  # AGGIORNATO NUM COLONNE GRIGLIA
                num_rows = math.ceil(INPUT_WINDOW / num_cols)

                for feature_idx, feature in enumerate(rain_features_sim_mod): # Usa rain_features_sim_mod
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"rain_mod_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = rain_data_mod[hour_idx, feature_idx]

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
                                    rain_data_mod[hour_idx, feature_idx] = value

            # Tab per l'umidità (ADATTATO ALLE NUOVE FEATURE)
            with data_tabs[1]:
                st.write("Inserisci i valori di umidità per ogni ora (%)")

                for feature_idx, feature in enumerate(humidity_features_sim_mod): # Usa humidity_features_sim_mod
                    st.write(f"### {feature}")

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"humidity_mod_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = humidity_data_mod[hour_idx, feature_idx]

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
                                    humidity_data_mod[hour_idx, feature_idx] = value

            # Tab per gli idrometri (INPUT - ADATTATO ALLE NUOVE FEATURE)
            with data_tabs[2]:
                st.write("Inserisci i livelli idrometrici per ogni ora (m) (escluso Bettolelle e Ponte Garibaldi)") # Testo modificato

                for feature_idx, feature in enumerate(hydro_input_features_sim_mod): # Usa hydro_input_features_sim_mod
                    st.write(f"### {feature}")

                    # Aggiungiamo un modo per impostare un valore costante
                    const_col1, const_col2 = st.columns([3, 1])

                    # Chiave univoca per ogni valore costante
                    const_key = f"const_hydro_input_{feature_idx}"
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
                            st.session_state.hydro_input_data_mod[h, feature_idx] = value
                            input_key = f"hydro_input_mod_{feature_idx}_{h}" # Generate input key
                            st.session_state[input_key] = value # Update input widget session state

                    with const_col2:
                        if st.button(f"Applica a tutte le ore", key=f"apply_const_hydro_input_{feature_idx}", on_click=apply_constant_value):
                            pass # Callback già gestisce l'aggiornamento

                    for row in range(num_rows):
                        cols = st.columns(num_cols)
                        for col in range(num_cols):
                            hour_idx = row * num_cols + col
                            if hour_idx < INPUT_WINDOW:
                                with cols[col]:
                                    # Creiamo una chiave univoca per ogni input
                                    input_key = f"hydro_input_mod_{feature_idx}_{hour_idx}"

                                    # Inizializziamo il valore nel session_state se non esiste
                                    if input_key not in st.session_state:
                                        st.session_state[input_key] = hydro_input_data_mod[hour_idx, feature_idx]

                                    # Usiamo il valore dal session_state
                                    value = st.number_input(
                                        f"Ora {hour_idx}",
                                        -1.0, 10.0,
                                        st.session_state[input_key], 0.01,
                                        key=input_key
                                    )

                                    # Aggiorniamo l'array con il valore corrente
                                    hydro_input_data_mod[hour_idx, feature_idx] = value

            # Componiamo i dati per la simulazione (ADATTATO ALLE NUOVE FEATURE)
            rain_offset = 0
            humidity_offset = len(rain_features_sim_mod) # Usa rain_features_sim_mod length
            hydro_input_offset = humidity_offset + len(humidity_features_sim_mod) # Usa humidity_features_sim_mod length

            sim_data_mod = np.zeros((INPUT_WINDOW, len(feature_columns_mod))) # Crea sim_data_mod con dimensioni corrette

            for h in range(INPUT_WINDOW):
                # Copiamo i dati di pioggia
                for i in range(len(rain_features_sim_mod)): # Usa rain_features_sim_mod length
                    sim_data_mod[h, rain_offset + i] = rain_data_mod[h, i]

                # Copiamo i dati di umidità
                for i in range(len(humidity_features_sim_mod)): # Usa humidity_features_sim_mod length
                    sim_data_mod[h, humidity_offset + i] = humidity_data_mod[h, i]

                # Copiamo i dati degli idrometri (input)
                for i in range(len(hydro_input_features_sim_mod)): # Usa hydro_input_features_sim_mod length
                    sim_data_mod[h, hydro_input_offset + i] = hydro_input_data_mod[h, i]


            # Visualizziamo un'anteprima dei dati (ADATTATO ALLE NUOVE FEATURE)
            st.subheader("Anteprima dei dati inseriti (Modello Bettolelle)") # Testo modificato
            preview_df_mod = pd.DataFrame(sim_data_mod, columns=feature_columns_mod) # Usa feature_columns_mod
            preview_df_mod.index = [f"Ora {i}" for i in range(INPUT_WINDOW)]
            st.dataframe(preview_df_mod.round(2))

        else:  # Inserimento manuale completo (ADATTATO ALLE NUOVE FEATURE)
            st.subheader('Inserisci valori per ogni parametro (Modello Bettolelle)') # Testo modificato

            # Creiamo un dataframe vuoto per i dati della simulazione
            sim_data_mod = np.zeros((INPUT_WINDOW, len(feature_columns_mod))) # Usa feature_columns_mod

            # Raggruppiamo i controlli per tipo di sensore
            with st.expander("Imposta valori di pioggia cumulata"): # Testo modificato
                for i, feature in enumerate(rain_features_sim_mod): # Usa rain_features_sim_mod
                    value = st.number_input(f'{feature} (mm)', 0.0, 100.0, 0.0, 0.5)
                    sim_data_mod[:, i] = value

            with st.expander("Imposta valore di umidità"):
                value = st.number_input(f'{humidity_features_sim_mod[0]} (%)', 0.0, 100.0, 50.0, 0.5) # Usa humidity_features_sim_mod
                sim_data_mod[:, len(rain_features_sim_mod)] = value # Usa rain_features_sim_mod length

            with st.expander("Imposta livelli idrometrici (Input Modello Bettolelle)"): # Testo modificato
                offset = len(rain_features_sim_mod) + len(humidity_features_sim_mod) # Usa le lunghezze corrette
                for i, feature in enumerate(hydro_input_features_sim_mod): # Usa hydro_input_features_sim_mod
                    value = st.number_input(f'{feature} (m)', -1.0, 10.0, 0.0, 0.01)
                    sim_data_mod[:, offset + i] = value

        # Bottone per eseguire la simulazione (ADATTATO ALLE NUOVE FEATURE e TARGET)
        if st.button('Esegui simulazione (Modello Bettolelle)', type="primary"): # Testo pulsante modificato
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile eseguire la simulazione.")
            else:
                with st.spinner('Simulazione in corso (Modello Bettolelle)...'): # Testo spinner modificato
                    # Previsione - USA SIM_DATA_MOD, FEATURE_COLUMNS_MOD, TARGET_FEATURES_MODIFICATO
                    predictions_mod = predict(model, sim_data_mod, scaler_features, scaler_targets, target_features_modificato, device, OUTPUT_WINDOW) # Usa target_features_modificato

                    if predictions_mod is not None: # Check if prediction was successful
                        # Visualizzazione dei risultati (ADATTATO AL NUOVO TARGET)
                        st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore (Modello Bettolelle - Target: {target_features_modificato[0]})') # Testo modificato

                        # Creazione dataframe risultati (ADATTATO AL NUOVO TARGET)
                        current_time = datetime.now()
                        prediction_times = [current_time + timedelta(hours=i) for i in range(OUTPUT_WINDOW)]
                        results_df_mod = pd.DataFrame(predictions_mod, columns=target_features_modificato) # Usa target_features_modificato
                        results_df_mod['Ora previsione'] = prediction_times
                        results_df_mod = results_df_mod[['Ora previsione'] + target_features_modificato] # Usa target_features_modificato

                        # Visualizzazione tabella risultati
                        st.dataframe(results_df_mod.round(3))

                        # Download dei risultati
                        st.markdown(get_table_download_link(results_df_mod), unsafe_allow_html=True)

                        # Grafici per ogni sensore (ADATTATO AL NUOVO TARGET)
                        st.subheader(f'Grafici delle previsioni (Modello Bettolelle - Target: {target_features_modificato[0]})') # Testo modificato

                        # Creiamo una visualizzazione che mostri sia i dati inseriti che le previsioni
                        for i, feature in enumerate(target_features_modificato): # Usa target_features_modificato
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Indice per i dati degli idrometri nel sim_data_mod (corretto per il nuovo set di feature)
                            hydro_idx = hydro_input_offset  + i # Corretto offset per il nuovo set di features, in questo caso i è sempre 0 perché target_features_modificato ha solo 1 elemento

                            # Dati storici (input) - Grafico solo se ha senso visualizzare l'input per il target
                            # In questo caso, potremmo non avere un input diretto "storico" per il target Bettolelle nella simulazione manuale
                            # Se si volesse visualizzare *qualche* input, si dovrebbe decidere quale feature input mostrare e adattare l'indice.
                            # Per ora, commento la parte di plot dell'input per semplificare.

                            # Dati previsti (output)
                            ax.plot(prediction_times, predictions_mod[:, i], 'r-', label='Previsione') # Usa predictions_mod

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
                            st.markdown(get_image_download_link(fig, f"sim_mod_{sensor_name}.png", f"il grafico di {feature} (Modello Bettolelle)"), unsafe_allow_html=True) # Testo download modificato

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')

    # ... (rest of Analisi Dati Storici page - no changes needed for model modification itself)

elif page == 'Allenamento Modello':
    st.header('Allenamento Modello')
    # ... (rest of Allenamento Modello page - no changes needed here, as the modified training is already implemented there)


# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
