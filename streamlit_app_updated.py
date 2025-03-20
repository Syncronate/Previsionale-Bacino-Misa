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
INPUT_WINDOW = 24
OUTPUT_WINDOW = 12

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
            ['Modifica dati recenti', 'Inserisci manualmente tutti i valori', 'Inserisci manualmente i valori per ogni ora']
        )

        if sim_method == 'Modifica dati recenti':
            # Prendiamo i dati recenti come base
            recent_data = df.iloc[-INPUT_WINDOW:][feature_columns].copy()

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

        elif sim_method == 'Inserisci manualmente tutti i valori':  # Inserimento manuale completo (valore singolo ripetuto)
            st.subheader('Inserisci valori per ogni parametro (valore singolo per tutte le ore)')

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

        elif sim_method == 'Inserisci manualmente i valori per ogni ora': # Nuova modalità di simulazione
            st.subheader('Inserisci valori per ogni ora (per tutte le 24 ore)')
            sim_data = np.zeros((INPUT_WINDOW, len(feature_columns)))

            # Input widgets for Hour 1 (used as defaults)
            hour1_values = {}
            with st.expander(f"Ora 1"):
                col_rain, col_humidity, col_hydro = st.columns(3)
                with col_rain:
                    st.markdown("**Pioggia**")
                    for i, feature in enumerate(rain_features):
                        hour1_values[f'rain_{0}_{i}'] = st.number_input(f'{feature} (mm)', 0.0, 100.0, 0.0, 0.5, key=f'rain_{0}_{i}')
                        sim_data[0, i] = hour1_values[f'rain_{0}_{i}']
                with col_humidity:
                    st.markdown("**Umidità**")
                    hour1_values[f'humidity_{0}_0'] = st.number_input(f'{humidity_feature[0]} (%)', 0.0, 100.0, 50.0, 0.5, key=f'humidity_{0}_0')
                    sim_data[0, len(rain_features)] = hour1_values[f'humidity_{0}_0']
                with col_hydro:
                    st.markdown("**Livelli Idrometrici**")
                    offset = len(rain_features) + len(humidity_feature)
                    for i, feature in enumerate(hydro_features):
                        hour1_values[f'hydro_{0}_{i}'] = st.number_input(f'{feature} (m)', -1.0, 10.0, 0.0, 0.01, key=f'hydro_{0}_{i}')
                        sim_data[0, offset + i] = hour1_values[f'hydro_{0}_{i}']

            if st.button("Popola tutte le ore con i valori dell'Ora 1"):
                for hour in range(1, INPUT_WINDOW): # Start from hour 2
                    for i, feature in enumerate(rain_features):
                        sim_data[hour, i] = hour1_values[f'rain_{0}_{i}']
                    sim_data[hour, len(rain_features)] = hour1_values[f'humidity_{0}_0']
                    offset = len(rain_features) + len(humidity_feature)
                    for i, feature in enumerate(hydro_features):
                        sim_data[hour, offset + i] = hour1_values[f'hydro_{0}_{i}']

                st.success("Valori di Ora 1 copiati a tutte le altre ore.") # Optional success message

            for hour in range(1, INPUT_WINDOW): # Start from hour 2 - Hour 1 is already done above
                with st.expander(f"Ora {hour+1}"):
                    col_rain, col_humidity, col_hydro = st.columns(3)
                    with col_rain:
                        st.markdown("**Pioggia**")
                        for i, feature in enumerate(rain_features):
                            st.number_input(f'{feature} (mm)', 0.0, 100.0, value=sim_data[hour, i], step=0.5, key=f'rain_{hour}_{i}') # Use value=sim_data
                    with col_humidity:
                        st.markdown("**Umidità**")
                        st.number_input(f'{humidity_feature[0]} (%)', 0.0, 100.0, value=sim_data[hour, len(rain_features)], step=0.5, key=f'humidity_{hour}_0') # Use value=sim_data
                    with col_hydro:
                        st.markdown("**Livelli Idrometrici**")
                        offset = len(rain_features) + len(humidity_feature)
                        for i, feature in enumerate(hydro_features):
                            st.number_input(f'{feature} (m)', -1.0, 10.0, value=sim_data[hour, offset + i], step=0.01, key=f'hydro_{hour}_{i}') # Use value=sim_data


        # Bottone per eseguire la simulazione
        if st.button('Esegui simulazione'):
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
                    figs = plot_predictions(predictions, hydro_features, OUTPUT_WINDOW, current_time)

                    # Visualizzazione grafici
                    for i, fig in enumerate(figs):
                        st.pyplot(fig)
                        sensor_name = hydro_features[i].replace(' ', '_').replace('/', '_')
                        st.markdown(get_image_download_link(fig, f"sim_{sensor_name}.png", f"il grafico di {hydro_features[i]}"), unsafe_allow_html=True)

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

            if len(train_filtered_df) < INPUT_WINDOW + OUTPUT_WINDOW:
                st.error(f'Servono almeno {INPUT_WINDOW + OUTPUT_WINDOW} righe di dati per l\'addestramento. Hai selezionato solo {len(train_filtered_df)} righe.')
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

                # Funzione per preparare i dati di addestramento
                def prepare_training_data(df_train, features, targets, input_window, output_window, val_split): # Renamed df to df_train
                    # Prepara i dati per l'addestramento creando sequenze di input e output
                    X, y = [], []

                    for i in range(len(df_train) - input_window - output_window + 1): # Use df_train
                        # Sequenza di input (input_window timesteps)
                        X.append(df_train.iloc[i:i+input_window][features].values) # Use df_train

                        # Sequenza di output (output_window timesteps)
                        y.append(df_train.iloc[i+input_window:i+input_window+output_window][targets].values) # Use df_train

                    X = np.array(X)
                    y = np.array(y)

                    # Normalizzazione dei dati
                    scaler_features_train = MinMaxScaler()
                    scaler_targets_train = MinMaxScaler()

                    # Reshape per la normalizzazione
                    X_flat = X.reshape(-1, X.shape[-1])
                    y_flat = y.reshape(-1, y.shape[-1])

                    X_scaled_flat = scaler_features_train.fit_transform(X_flat)
                    y_scaled_flat = scaler_targets_train.fit_transform(y_flat)

                    # Reshape indietro alla forma originale
                    X_scaled = X_scaled_flat.reshape(X.shape)
                    y_scaled = y_scaled_flat.reshape(y.shape)

                    # Divisione train/validation
                    split_idx = int(len(X_scaled) * (1 - val_split/100))

                    X_train = X_scaled[:split_idx]
                    y_train = y_scaled[:split_idx]
                    X_val = X_scaled[split_idx:]
                    y_val = X_scaled[split_idx:]

                    return X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train

                # Funzione di addestramento
                def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window, hidden_size, num_layers, dropout, batch_size, epochs, learning_rate, patience, use_scheduler):
                    # Conversione in tensori PyTorch
                    X_train_tensor = torch.FloatTensor(X_train)
                    y_train_tensor = torch.FloatTensor(y_train)
                    X_val_tensor = torch.FloatTensor(X_val)
                    y_val_tensor = torch.FloatTensor(y_val)

                    # Dataset e dataloader
                    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
                    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)

                    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)

                    # Device
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

                    # Modello
                    model_train = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers, dropout).to(device)

                    # Ottimizzatore
                    optimizer = torch.optim.Adam(model_train.parameters(), lr=learning_rate)

                    # Scheduler
                    scheduler = None
                    if use_scheduler:
                        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

                    # Loss function
                    criterion = nn.MSELoss()

                    # Early stopping
                    best_val_loss = float('inf')
                    early_stopping_counter = 0
                    best_model_state = None

                    # Storico delle perdite
                    train_losses = []
                    val_losses = []

                    # Addestramento
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    loss_chart = st.empty()

                    for epoch in range(epochs):
                        model_train.train()
                        train_loss = 0

                        for batch_X, batch_y in train_loader:
                            batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                            # Forward pass
                            outputs = model_train(batch_X)
                            loss = criterion(outputs, batch_y)
                            # print("Shape of outputs:", outputs.shape) # Debugging shapes
                            # print("Shape of batch_y:", batch_y.shape) # Debugging shapes

                            # Backward pass e ottimizzazione
                            optimizer.zero_grad()
                            loss.backward()
                            optimizer.step()

                            train_loss += loss.item()

                        # Validazione
                        model_train.eval()
                        val_loss = 0

                        with torch.no_grad():
                            for batch_X, batch_y in val_loader:
                                batch_X, batch_y = batch_X.to(device), batch_y.to(device)

                                outputs = model_train(batch_X)
                                loss = criterion(outputs, batch_y)
                                # print("Shape of outputs (val):", outputs.shape) # Debugging shapes
                                # print("Shape of batch_y (val):", batch_y.shape) # Debugging shapes

                                val_loss += loss.item()

                        # Normalizzazione delle perdite
                        train_loss /= len(train_loader)
                        val_loss /= len(val_loader)

                        train_losses.append(train_loss)
                        val_losses.append(val_loss)

                        # Aggiornamento dello scheduler
                        if scheduler:
                            scheduler.step(val_loss)

                        # Visualizzazione progresso
                        progress_bar.progress((epoch + 1) / epochs)
                        status_text.text(f'Epoca {epoch+1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

                        # Grafico delle perdite
                        fig, ax = plt.subplots(figsize=(10, 4))
                        ax.plot(train_losses, label='Train Loss')
                        ax.plot(val_losses, label='Validation Loss')
                        ax.set_xlabel('Epoca')
                        ax.set_ylabel('Loss')
                        ax.legend()
                        ax.grid(True)
                        loss_chart.pyplot(fig)
                        plt.close(fig)

                        # Early stopping
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            early_stopping_counter = 0
                            best_model_state = model_train.state_dict()
                        else:
                            early_stopping_counter += 1
                            if early_stopping_counter >= patience:
                                status_text.text(f'Early stopping attivato all\'epoca {epoch+1}')
                                break

                    # Caricamento del miglior modello
                    model_train.load_state_dict(best_model_state)

                    return model_train, train_losses, val_losses, best_val_loss

                # Bottone per avviare l'addestramento
                if st.button('Avvia Addestramento'):
                    with st.spinner('Preparazione dei dati in corso...'):
                        # Verifica della presenza di tutti i dati necessari
                        missing_columns = [col for col in feature_columns if col not in train_filtered_df.columns]
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

                            st.success(f'Dati preparati: {X_train.shape[0]} esempi di addestramento, {X_val.shape[0]} esempi di validazione')

                            with st.spinner('Addestramento in corso...'):
                                # Addestramento del modello
                                trained_model, train_losses, val_losses, best_val_loss = train_model(
                                    X_train,
                                    y_train,
                                    X_val,
                                    y_val,
                                    len(feature_columns),
                                    len(hydro_features),
                                    OUTPUT_WINDOW,
                                    hidden_size,
                                    num_layers,
                                    dropout,
                                    batch_size,
                                    epochs,
                                    learning_rate,
                                    patience,
                                    use_scheduler
                                )

                                st.success(f'Addestramento completato! Miglior loss di validazione: {best_val_loss:.6f}')

                                # Visualizzazione dei risultati finali
                                st.subheader('Risultati dell\'addestramento')

                                # Grafico delle perdite
                                fig, ax = plt.subplots(figsize=(10, 4))
                                ax.plot(train_losses, label='Train Loss')
                                ax.plot(val_losses, label='Validation Loss')
                                ax.set_xlabel('Epoca')
                                ax.set_ylabel('Loss')
                                ax.legend()
                                ax.grid(True)
                                st.pyplot(fig)
                                st.markdown(get_image_download_link(fig, "training_loss.png", "il grafico delle perdite"), unsafe_allow_html=True)

                                # Salvataggio automatico del modello
                                st.subheader('Salvataggio automatico del modello completato') # Modified subheader to reflect automatic saving
                                model_name = 'hydro_model_new.pth' # Fixed names, no input needed now, or provide default
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
                                model_path = os.path.join('models', f'{timestamp}_{model_name}') # Usando os.path.join
                                scaler_features_path = os.path.join('models', f'{timestamp}_{scaler_features_name}') # Usando os.path.join
                                scaler_targets_path = os.path.join('models', f'{timestamp}_{scaler_targets_name}') # Usando os.path.join


                                try:
                                    # Salvataggio del modello
                                    torch.save(trained_model.state_dict(), model_path)

                                    # Salvataggio degli scaler CORRETTI (quelli di training)
                                    joblib.dump(scaler_features_train, scaler_features_path)
                                    joblib.dump(scaler_targets_train, scaler_targets_path)

                                    st.success(f'Modello e scaler salvati con successo in cartella "models"!')
                                except Exception as e:
                                    st.error(f'Errore durante il salvataggio: {e}')


                                # Salvataggio dei parametri del modello
                                params_path = os.path.join('models', f'{timestamp}_model_params.txt') # Usando os.path.join
                                with open(params_path, 'w') as f:
                                    f.write(f'Data e ora: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
                                    f.write(f'Periodo di addestramento: {train_start_date} - {train_end_date}\n')
                                    f.write(f'Numero di esempi: {len(X_train) + len(X_val)}\n')
                                    f.write(f'Split validazione: {val_split}%\n')
                                    f.write(f'Hidden size: {hidden_size}\n')
                                    f.write(f'Num layers: {num_layers}\n')
                                    f.write(f'Dropout: {dropout}\n')
                                    f.write(f'Learning rate: {learning_rate}\n')
                                    f.write(f'Batch size: {batch_size}\n')
                                    f.write(f'Epochs: {epochs}\n')
                                    f.write(f'Best validation loss: {best_val_loss:.6f}\n')

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
                                    with st.spinner('Test in corso...'):
                                        # Conversione in tensori
                                        X_val_tensor = torch.FloatTensor(X_val)
                                        y_val_tensor = torch.FloatTensor(y_val)

                                        # Valutazione sul set di validazione
                                        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                                        trained_model.eval()

                                        # Previsioni
                                        with torch.no_grad():
                                            y_pred = trained_model(X_val_tensor.to(device))
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

                                        # Selezione di un esempio casuale
                                        example_idx = np.random.randint(0, len(X_val))

                                        # Input, previsione e ground truth
                                        example_input = X_val[example_idx]
                                        example_pred = y_pred_denorm[example_idx]
                                        example_truth = y_val_denorm[example_idx]

                                        # Visualizzazione
                                        for i, sensor in enumerate(hydro_features):
                                            fig, ax = plt.subplots(figsize=(10, 5))

                                            hours = range(OUTPUT_WINDOW)
                                            ax.plot(hours, example_truth[:, i], marker='o', linestyle='-', label='Ground Truth')
                                            ax.plot(hours, example_pred[:, i], marker='x', linestyle='--', label='Previsione')

                                            ax.set_title(f'Previsione vs. Ground Truth - {sensor}')
                                            ax.set_xlabel('Ore future')
                                            ax.set_ylabel('Livello idrometrico [m]')
                                            ax.legend()
                                            ax.grid(True)

                                            st.pyplot(fig)

# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
