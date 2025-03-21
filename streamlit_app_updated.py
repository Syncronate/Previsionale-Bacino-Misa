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
import math

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

# --- FUNZIONE modifica_modello_previsionale() AGGIORNATA per includere lo storico di Bettolelle ---
def modifica_modello_previsionale():
    # Target: idrometro di Bettolelle
    bettolelle_id_name = 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)'
    target_features_mod = [bettolelle_id_name]  # Solo Bettolelle

    # Ponte Garibaldi - lo escludiamo
    ponte_garibaldi_id_name = 'Livello Idrometrico Sensore 3405 [m] (Ponte Garibaldi)'

    # Funzione per ottenere tutte le feature, INCLUDENDO Bettolelle ma ESCLUDENDO Ponte Garibaldi
    def get_input_features(df):
        # Ottieni tutte le colonne con prefisso "Cumulata Sensore" (tutte le cumulate)
        cumulate_columns = [col for col in df.columns if col.startswith('Cumulata Sensore')]

        # Ottieni tutte le colonne con prefisso "Umidita'" (tutti i sensori di umidità)
        umidita_columns = [col for col in df.columns if col.startswith('Umidita\' Sensore')]

        # Ottieni tutte le colonne con prefisso "Livello Idrometrico Sensore" eccetto Ponte Garibaldi
        # MA INCLUDENDO Bettolelle, che adesso sarà sia input che target
        idrometro_columns = [col for col in df.columns if col.startswith('Livello Idrometrico Sensore')
                            and col != ponte_garibaldi_id_name]

        # Combinazione di tutte le colonne di input
        input_features_mod = cumulate_columns + umidita_columns + idrometro_columns

        return input_features_mod

    # Modifica della funzione di preparazione dati per utilizzare queste nuove feature
    def prepare_training_data_modificato(df_train, val_split):
        # Ottieni le feature di input (ora include Bettolelle)
        feature_columns_mod = get_input_features(df_train)

        # Creazione delle sequenze di input (X) e output (y)
        X, y = [], []

        for i in range(len(df_train) - INPUT_WINDOW - OUTPUT_WINDOW + 1):
            # X include tutte le feature, compreso lo storico di Bettolelle
            X.append(df_train.iloc[i:i+INPUT_WINDOW][feature_columns_mod].values)
            # y contiene solo Bettolelle per le previsioni future
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

# Ottieni le funzioni per la preparazione dei dati modificati
get_input_features_func, prepare_training_data_modificato_func, target_features_modificato = modifica_modello_previsionale()

# --- FINE FUNZIONE modifica_modello_previsionale() ---


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
    MODEL_PATH_MOD = 'best_hydro_model_bettolelle_with_history.pth' # Percorso modello modificato
    SCALER_FEATURES_PATH = 'scaler_features.joblib'  # Sostituisci con il percorso corretto
    SCALER_FEATURES_PATH_MOD = 'scaler_features_bettolelle_with_history.joblib' # Percorso scaler features modificato
    SCALER_TARGETS_PATH = 'scaler_targets.joblib'  # Sostituisci con il percorso corretto
    SCALER_TARGETS_PATH_MOD = 'scaler_targets_bettolelle_with_history.joblib' # Percorso scaler targets modificato
else:
    # Caricamento dei file dall'utente
    st.sidebar.subheader('Carica i tuoi file')
    data_file = st.sidebar.file_uploader('File CSV con i dati storici', type=['csv'])
    model_file = st.sidebar.file_uploader('File del modello originale (.pth)', type=['pth'])
    model_file_mod = st.sidebar.file_uploader('File del modello Bettolelle (.pth)', type=['pth']) # File modello modificato
    scaler_features_file = st.sidebar.file_uploader('File scaler features originale (.joblib)', type=['joblib'])
    scaler_features_file_mod = st.sidebar.file_uploader('File scaler features Bettolelle (.joblib)', type=['joblib']) # File scaler features modificato
    scaler_targets_file = st.sidebar.file_uploader('File scaler targets originale (.joblib)', type=['joblib'])
    scaler_targets_file_mod = st.sidebar.file_uploader('File scaler targets Bettolelle (.joblib)', type=['joblib']) # File scaler targets modificato

    # Controllo se tutti i file sono stati caricati
    if not use_demo_files:
        if not (data_file and model_file and scaler_features_file and scaler_targets_file and model_file_mod and scaler_features_file_mod and scaler_targets_file_mod):
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

model_mod = None # Modello modificato
scaler_features_mod = None # Scaler features modificato
scaler_targets_mod = None # Scaler targets modificato


# Caricamento del modello e degli scaler SOLO se i file sono stati caricati o si usano quelli demo
if use_demo_files or (data_file and model_file and scaler_features_file and scaler_targets_file):
    try:
        if use_demo_files:
            model, device = load_model(MODEL_PATH, 8, 5, OUTPUT_WINDOW) # MODIFIED: input_size=8, output_size=5
            scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
        else:
            model_bytes = io.BytesIO(model_file.read())
            model, device = load_model(model_bytes, 8, 5, OUTPUT_WINDOW) # MODIFIED: input_size=8, output_size=5
            scaler_features, scaler_targets = load_scalers(scaler_features_file, scaler_targets_file)

        if model is not None and scaler_features is not None and scaler_targets is not None: # Check if loading was successful
            st.sidebar.success('Modello originale e scaler caricati con successo')
    except Exception as e:
        st.sidebar.error(f'Errore nel caricamento del modello originale o degli scaler: {e}')

# Caricamento del modello modificato e scaler
if use_demo_files or (data_file and model_file_mod and scaler_features_file_mod and scaler_targets_file_mod):
    try:
        if use_demo_files:
            model_mod, device = load_model(MODEL_PATH_MOD, 8, 1, OUTPUT_WINDOW) # Usa MODEL_PATH_MOD
            scaler_features_mod, scaler_targets_mod = load_scalers(SCALER_FEATURES_PATH_MOD, SCALER_TARGETS_PATH_MOD) # Usa *_PATH_MOD
        else:
            model_bytes_mod = io.BytesIO(model_file_mod.read()) # Usa model_file_mod
            model_mod, device = load_model(model_bytes_mod, 8, 1, OUTPUT_WINDOW) # Usa model_bytes_mod
            scaler_features_mod, scaler_targets_mod = load_scalers(scaler_features_file_mod, scaler_targets_file_mod) # Usa *_file_mod

        if model_mod is not None and scaler_features_mod is not None and scaler_targets_mod is not None: # Check if loading was successful
            st.sidebar.success('Modello Bettolelle e scaler caricati con successo')
    except Exception as e:
        st.sidebar.error(f'Errore nel caricamento del modello Bettolelle o degli scaler: {e}')


# Menu principale
st.sidebar.header('Menu')
page = st.sidebar.radio('Scegli una funzionalità',
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Allenamento Modello'])


if df is not None: # Only calculate if df is loaded
    feature_columns_mod = get_input_features_func(pd.DataFrame(columns=df.columns if df is not None else feature_columns_original)) # needed to have columns to pass
else:
    feature_columns_mod = get_input_features_func(pd.DataFrame(columns=feature_columns_original)) # needed to have columns to pass in case df is None


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
        st.header('Previsione in base agli ultimi dati (Modello Originale)') # Modificato titolo per chiarezza

        if st.button('Genera previsione (Modello Originale)'): # Modificato testo bottone
            if model is None or scaler_features is None or scaler_targets is None:
                st.error("Modello o scaler non caricati correttamente. Impossibile generare la previsione.")
            else:
                with st.spinner('Generazione previsione in corso (Modello Originale)...'): # Modificato testo spinner
                    # Preparazione dei dati di input (ultime INPUT_WINDOW ore)
                    latest_data = df.iloc[-INPUT_WINDOW:][feature_columns].values

                    # Previsione
                    predictions = predict(model, latest_data, scaler_features, scaler_targets, hydro_features, device, OUTPUT_WINDOW)

                    if predictions is not None: # Check if prediction was successful
                        # Visualizzazione dei risultati
                        st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore (Modello Originale)') # Modificato titolo

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
                        st.subheader('Grafici delle previsioni (Modello Originale)') # Modificato titolo
                        figs = plot_predictions(predictions, hydro_features, OUTPUT_WINDOW, start_time)

                        # Visualizzazione grafici
                        for i, fig in enumerate(figs):
                            st.pyplot(fig)
                            sensor_name = hydro_features[i].replace(' ', '_').replace('/', '_')
                            st.markdown(get_image_download_link(fig, f"{sensor_name}.png", f"il grafico di {hydro_features[i]} (Modello Originale)"), unsafe_allow_html=True) # Modificato testo download

elif page == 'Simulazione':
    st.header('Simulazione Idrologica (Modello Bettolelle con Storico)') # Titolo modificato per indicare il modello aggiornato
    st.write('Inserisci i valori per simulare uno scenario idrologico con il modello aggiornato (Bettolelle con storico)') # Descrizione modificata

    if df is None or model_mod is None or scaler_features_mod is None or scaler_targets_mod is None: # Usa model_mod e scaler modificati
        st.warning("Attenzione: Alcuni file necessari per il modello Bettolelle non sono stati caricati correttamente. La simulazione potrebbe non funzionare.") # Modificato messaggio
    else:
        # Opzioni per la simulazione
        sim_method = st.radio(
            "Metodo di simulazione",
            ['Inserisci dati orari', 'Inserisci manualmente tutti i valori'] # Rimosso 'Modifica dati recenti' per semplificare e focalizzarsi sul nuovo modello
        )

        if sim_method == 'Inserisci dati orari':
            st.subheader(f'Inserisci dati per ogni ora ({INPUT_WINDOW} ore precedenti) per la simulazione del modello Bettolelle con storico') # Testo modificato

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
            if 'bettolelle_history_data_mod' not in st.session_state: # Aggiunto per storico Bettolelle
                bettolelle_feature_index = feature_columns_mod.index('Livello Idrometrico Sensore 1112 [m] (Bettolelle)') if 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)' in feature_columns_mod else -1
                if bettolelle_feature_index != -1:
                    st.session_state.bettolelle_history_data_mod = np.zeros((INPUT_WINDOW, 1))
                else:
                    st.session_state.bettolelle_history_data_mod = np.zeros((INPUT_WINDOW, 1)) # Inizializza comunque anche se la feature non è presente

            # Riferimenti più corti per maggiore leggibilità (ADATTATO ALLE NUOVE FEATURE)
            rain_data_mod = st.session_state.rain_data_mod
            humidity_data_mod = st.session_state.humidity_data_mod
            hydro_input_data_mod = st.session_state.hydro_input_data_mod
            bettolelle_history_data_mod = st.session_state.bettolelle_history_data_mod # Riferimento allo storico di Bettolelle

            # Feature names per tab (ADATTATO ALLE NUOVE FEATURE)
            rain_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Cumulata Sensore')]
            humidity_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Umidita\' Sensore')]
            hydro_input_features_sim_mod = [col for col in feature_columns_mod if col.startswith('Livello Idrometrico Sensore') and 'Bettolelle' not in col]
            bettolelle_history_feature_sim_mod = ['Livello Idrometrico Sensore 1112 [m] (Bettolelle)'] # Feature per storico Bettolelle

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

            # Seleziona se includere lo storico di Bettolelle come input nella simulazione
            include_bettolelle_history = st.checkbox("Includi storico Bettolelle come input nella simulazione", value=True)

            if include_bettolelle_history:
                # Tab per lo storico di Bettolelle (INPUT - ADATTATO ALLE NUOVE FEATURE)
                bettolelle_history_tab = st.tabs(["Storico Bettolelle (Input)"])[0] # Crea una singola tab
                with bettolelle_history_tab:
                    st.write("Inserisci i livelli idrometrici storici di Bettolelle per ogni ora (m)") # Testo modificato

                    for feature_idx, feature in enumerate(bettolelle_history_feature_sim_mod): # Usa bettolelle_history_feature_sim_mod
                        st.write(f"### {feature}")

                        for row in range(num_rows):
                            cols = st.columns(num_cols)
                            for col in range(num_cols):
                                hour_idx = row * num_cols + col
                                if hour_idx < INPUT_WINDOW:
                                    with cols[col]:
                                        # Creiamo una chiave univoca per ogni input
                                        input_key = f"bettolelle_history_mod_{feature_idx}_{hour_idx}"

                                        # Inizializziamo il valore nel session_state se non esiste
                                        if input_key not in st.session_state:
                                            st.session_state[input_key] = bettolelle_history_data_mod[hour_idx, feature_idx]

                                        # Usiamo il valore dal session_state
                                        value = st.number_input(
                                            f"Ora {hour_idx}",
                                            -1.0, 10.0,
                                            st.session_state[input_key], 0.01,
                                            key=input_key
                                        )

                                        # Aggiorniamo l'array con il valore corrente
                                        bettolelle_history_data_mod[hour_idx, feature_idx] = value
            else:
                # Se non si include lo storico, riempi con zeri
                bettolelle_history_data_mod = np.zeros_like(bettolelle_history_data_mod)


            # Componiamo i dati per la simulazione (ADATTATO ALLE NUOVE FEATURE)
            rain_offset = 0
            humidity_offset = len(rain_features_sim_mod) # Usa rain_features_sim_mod length
            hydro_input_offset = humidity_offset + len(humidity_features_sim_mod) # Usa humidity_features_sim_mod length
            bettolelle_history_offset = hydro_input_offset + len(hydro_input_features_sim_mod) # Usa hydro_input_features_sim_mod length

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

                # Copiamo i dati dello storico di Bettolelle (se inclusi)
                if include_bettolelle_history and bettolelle_history_feature_sim_mod: # Controlla se ci sono feature storiche di Bettolelle
                    sim_data_mod[h, bettolelle_history_offset] = bettolelle_history_data_mod[h, 0] # Assume una sola feature storica di Bettolelle


            # Visualizziamo un'anteprima dei dati (ADATTATO ALLE NUOVE FEATURE)
            st.subheader("Anteprima dei dati inseriti (Modello Bettolelle con Storico)") # Testo modificato
            preview_df_mod = pd.DataFrame(sim_data_mod, columns=feature_columns_mod) # Usa feature_columns_mod
            preview_df_mod.index = [f"Ora {i}" for i in range(INPUT_WINDOW)]
            st.dataframe(preview_df_mod.round(2))

        else:  # Inserimento manuale completo (ADATTATO ALLE NUOVE FEATURE)
            st.subheader('Inserisci valori per ogni parametro (Modello Bettolelle con Storico)') # Testo modificato

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

            # Checkbox per includere/escludere lo storico di Bettolelle
            include_bettolelle_history_manual = st.checkbox("Includi storico Bettolelle come input nella simulazione (manuale)", value=True)

            if include_bettolelle_history_manual:
                with st.expander("Imposta livelli idrometrici storici di Bettolelle"): # Testo modificato
                    offset_bettolelle_history = len(rain_features_sim_mod) + len(humidity_features_sim_mod) + len(hydro_input_features_sim_mod) # Calcola offset corretto
                    value = st.number_input(f'{bettolelle_history_feature_sim_mod[0]} (m)', -1.0, 10.0, 0.0, 0.01) # Usa bettolelle_history_feature_sim_mod
                    sim_data_mod[:, offset_bettolelle_history] = value
            else:
                offset_bettolelle_history = len(rain_features_sim_mod) + len(humidity_features_sim_mod) + len(hydro_input_features_sim_mod) # Calcola offset corretto
                sim_data_mod[:, offset_bettolelle_history] = 0.0 # Imposta a zero se non incluso

        # Bottone per eseguire la simulazione (ADATTATO ALLE NUOVE FEATURE e TARGET)
        if st.button('Esegui simulazione (Modello Bettolelle con Storico)', type="primary"): # Testo pulsante modificato
            if model_mod is None or scaler_features_mod is None or scaler_targets_mod is None: # Usa model_mod e scaler modificati
                st.error("Modello Bettolelle o scaler non caricati correttamente. Impossibile eseguire la simulazione.") # Modificato messaggio
            else:
                with st.spinner('Simulazione in corso (Modello Bettolelle con Storico)...'): # Testo spinner modificato
                    # Previsione - USA SIM_DATA_MOD, FEATURE_COLUMNS_MOD, TARGET_FEATURES_MODIFICATO, MODELLO MODIFICATO E SCALER MODIFICATI
                    predictions_mod = predict(model_mod, sim_data_mod, scaler_features_mod, scaler_targets_mod, target_features_modificato, device, OUTPUT_WINDOW) # Usa modello e scaler modificati e target_features_modificato

                    if predictions_mod is not None: # Check if prediction was successful
                        # Visualizzazione dei risultati (ADATTATO AL NUOVO TARGET)
                        st.subheader(f'Previsione per le prossime {OUTPUT_WINDOW} ore (Modello Bettolelle con Storico - Target: {target_features_modificato[0]})') # Testo modificato

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
                        st.subheader(f'Grafici delle previsioni (Modello Bettolelle con Storico - Target: {target_features_modificato[0]})') # Testo modificato

                        # Creiamo una visualizzazione che mostri sia i dati inseriti che le previsioni
                        for i, feature in enumerate(target_features_modificato): # Usa target_features_modificato
                            fig, ax = plt.subplots(figsize=(10, 6))

                            # Indice per i dati degli idrometri nel sim_data_mod (corretto per il nuovo set di feature)
                            hydro_idx = bettolelle_history_offset # Corretto offset per il nuovo set di features

                            # Dati storici (input) - Grafico dello storico di Bettolelle
                            input_hours = [current_time - timedelta(hours=INPUT_WINDOW - h) for h in range(INPUT_WINDOW)] # Ore precedenti
                            ax.plot(input_hours, sim_data_mod[:, hydro_idx], 'b-', label='Storico Bettolelle (Input)') # Plot input storico

                            # Dati previsti (output)
                            ax.plot(prediction_times, predictions_mod[:, i], 'r-', label='Previsione') # Usa predictions_mod

                            # Linea verticale per separare dati storici e previsione
                            ax.axvline(x=current_time, color='black', linestyle='--')
                            ax.annotate('Ora attuale', (current_time, ax.get_xlim()[0]),
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
                            st.markdown(get_image_download_link(fig, f"sim_mod_{sensor_name}.png", f"il grafico di {feature} (Modello Bettolelle con Storico)"), unsafe_allow_html=True) # Testo download modificato

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')

    # ... (rest of Analisi Dati Storici page - no changes needed for model modification itself)

elif page == 'Allenamento Modello':
    st.header('Allenamento Modello')

    st.subheader('Allenamento Modello Bettolelle (con storico)')
    st.write("Clicca il bottone per avviare l'allenamento del modello Bettolelle con storico. Questo processo potrebbe richiedere del tempo.")

    if st.button('Avvia Allenamento Modello Bettolelle (con storico)'):
        if df is None:
            st.error("Errore: Dati non caricati. Carica i dati nella sidebar per procedere con l'allenamento.")
        else:
            with st.spinner('Allenamento del modello in corso...'):
                # --- Parametri per l'allenamento ---
                INPUT_WINDOW_TRAIN = 6 # Ore di input
                OUTPUT_WINDOW_TRAIN = 6 # Ore di output
                VAL_SPLIT_TRAIN = 20 # Percentuale di dati per la validazione
                BATCH_SIZE_TRAIN = 32
                LEARNING_RATE_TRAIN = 0.001
                NUM_EPOCHS_TRAIN = 100
                OUTPUT_SIZE_MOD_TRAIN = 1 # Prevede solo 'Livello Idrometrico Sensore 1112 [m] (Bettolelle)'

                # --- Preparazione dei dati modificata ---
                X_train_mod, y_train_mod, X_val_mod, y_val_mod, scaler_features_mod_train, scaler_targets_mod_train, feature_columns_mod_train, target_features_mod_train = prepare_training_data_modificato_func(df, VAL_SPLIT_TRAIN)

                # Calcola dinamicamente INPUT_SIZE_MOD in base alle feature selezionate
                INPUT_SIZE_MOD_TRAIN = X_train_mod.shape[-1]

                st.write(f"Numero di feature di input per il modello modificato: {INPUT_SIZE_MOD_TRAIN}")
                st.write(f"Feature di input per il modello modificato: {feature_columns_mod_train}")
                st.write(f"Feature target per il modello modificato: {target_features_mod_train}")

                # --- Inizializzazione del modello modificato ---
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model_mod_train = HydroLSTM(INPUT_SIZE_MOD_TRAIN, 128, OUTPUT_SIZE_MOD_TRAIN, OUTPUT_WINDOW_TRAIN, num_layers=2, dropout=0.2).to(device)

                # --- Definizione della loss function e dell'optimizer ---
                criterion_mod_train = nn.MSELoss()
                optimizer_mod_train = torch.optim.Adam(model_mod_train.parameters(), lr=LEARNING_RATE_TRAIN)

                # --- Training loop modificato ---
                history_mod_train = {'train_loss': [], 'val_loss': []}

                for epoch in range(NUM_EPOCHS_TRAIN):
                    model_mod_train.train()
                    train_loss = 0

                    for batch_idx in range(0, len(X_train_mod), BATCH_SIZE_TRAIN):
                        X_batch = torch.FloatTensor(X_train_mod[batch_idx:batch_idx+BATCH_SIZE_TRAIN]).to(device)
                        y_batch = torch.FloatTensor(y_train_mod[batch_idx:batch_idx+BATCH_SIZE_TRAIN]).to(device)

                        optimizer_mod_train.zero_grad()
                        output = model_mod_train(X_batch)
                        loss = criterion_mod_train(output, y_batch)
                        loss.backward()
                        optimizer_mod_train.step()

                        train_loss += loss.item()

                    avg_train_loss = train_loss / (len(X_train_mod) // BATCH_SIZE_TRAIN + 1)
                    history_mod_train['train_loss'].append(avg_train_loss)

                    # Validation loop
                    model_mod_train.eval()
                    val_loss = 0

                    with torch.no_grad():
                        for batch_idx in range(0, len(X_val_mod), BATCH_SIZE_TRAIN):
                            X_batch_val = torch.FloatTensor(X_val_mod[batch_idx:batch_idx+BATCH_SIZE_TRAIN]).to(device)
                            y_batch_val = torch.FloatTensor(y_val_mod[batch_idx:batch_idx+BATCH_SIZE_TRAIN]).to(device)

                            output_val = model_mod_train(X_batch_val)
                            loss_val = criterion_mod_train(output_val, y_batch_val)
                            val_loss += loss_val.item()

                    avg_val_loss = val_loss / (len(X_val_mod) // BATCH_SIZE_TRAIN + 1)
                    history_mod_train['val_loss'].append(avg_val_loss)

                    st.write(f'Epoch {epoch+1}/{NUM_EPOCHS_TRAIN}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}') # Output to Streamlit

                # --- Salva il modello e gli scaler modificati ---
                MODEL_OUTPUT_PATH_MOD_TRAIN = 'best_hydro_model_bettolelle_with_history.pth' # Sovrascrive il modello demo o caricato
                SCALER_FEATURES_OUTPUT_PATH_MOD_TRAIN = 'scaler_features_bettolelle_with_history.joblib' # Sovrascrive scaler demo o caricato
                SCALER_TARGETS_OUTPUT_PATH_MOD_TRAIN = 'scaler_targets_bettolelle_with_history.joblib' # Sovrascrive scaler demo o caricato

                torch.save(model_mod_train.state_dict(), MODEL_OUTPUT_PATH_MOD_TRAIN)
                joblib.dump(scaler_features_mod_train, SCALER_FEATURES_OUTPUT_PATH_MOD_TRAIN)
                joblib.dump(scaler_targets_mod_train, SCALER_TARGETS_OUTPUT_PATH_MOD_TRAIN)

                st.success(f"Modello Bettolelle con storico allenato e salvato con successo!")
                st.write(f"Modello salvato in: {MODEL_OUTPUT_PATH_MOD_TRAIN}")
                st.write(f"Scaler features salvato in: {SCALER_FEATURES_OUTPUT_PATH_MOD_TRAIN}")
                st.write(f"Scaler targets salvato in: {SCALER_TARGETS_OUTPUT_PATH_MOD_TRAIN}")

                # Visualizza le curve di loss in Streamlit
                fig_loss, ax_loss = plt.subplots(figsize=(10, 6))
                ax_loss.plot(history_mod_train['train_loss'], label='Train Loss')
                ax_loss.plot(history_mod_train['val_loss'], label='Validation Loss')
                ax_loss.set_xlabel('Epoch')
                ax_loss.set_ylabel('Loss')
                ax_loss.set_title('Curve di Loss del Modello Bettolelle con Storico')
                ax_loss.legend()
                ax_loss.grid(True)
                st.pyplot(fig_loss)


# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
