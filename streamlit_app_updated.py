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
    import joblib
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
    if data_file and model_file and scaler_features_file and scaler_targets_file:
        # Salvataggio temporaneo dei file caricati
        DATA_PATH = 'temp_data.csv'
        MODEL_PATH = 'temp_model.pth'
        SCALER_FEATURES_PATH = 'temp_scaler_features.joblib'
        SCALER_TARGETS_PATH = 'temp_scaler_targets.joblib'
        
        with open(DATA_PATH, 'wb') as f:
            f.write(data_file.getbuffer())
        with open(MODEL_PATH, 'wb') as f:
            f.write(model_file.getbuffer())
        with open(SCALER_FEATURES_PATH, 'wb') as f:
            f.write(scaler_features_file.getbuffer())
        with open(SCALER_TARGETS_PATH, 'wb') as f:
            f.write(scaler_targets_file.getbuffer())
    else:
        st.sidebar.warning('Carica tutti i file necessari per procedere')
        st.stop()

# Definizione delle costanti
INPUT_WINDOW = 24  # 24 ore di dati storici
OUTPUT_WINDOW = 12  # 12 ore di previsione

# Caricamento dei dati storici
try:
    df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Data e Ora'])
    st.sidebar.success(f'Dati caricati: {len(df)} righe')
except Exception as e:
    st.error(f'Errore nel caricamento dei dati: {e}')
    st.stop()

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

# Caricamento del modello e degli scaler
try:
    model, device = load_model(MODEL_PATH, len(feature_columns), len(hydro_features), OUTPUT_WINDOW)
    scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
    st.sidebar.success('Modello e scaler caricati con successo')
except Exception as e:
    st.error(f'Errore nel caricamento del modello o degli scaler: {e}')
    st.stop()

# Menu principale
st.sidebar.header('Menu')
page = st.sidebar.radio('Scegli una funzionalità', 
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici'])

if page == 'Dashboard':
    st.header('Dashboard Idrologica')
    
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
    
    # Opzioni per la simulazione
    sim_method = st.radio(
        "Metodo di simulazione",
        ['Modifica dati recenti', 'Inserisci manualmente tutti i valori']
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
    if st.button('Esegui simulazione'):
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
    
    # Selezione dell'intervallo di date
    st.subheader('Seleziona l\'intervallo di date')
    min_date = df['Data e Ora'].min().date()
    max_date = df['Data e Ora'].max().date()
    
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input('Data inizio', min_date, min_value=min_date, max_value=max_date)
    with col2:
        end_date = st.date_input('Data fine', max_date, min_value=min_date, max_value=max_date)
    
    # Filtraggio dei dati
    mask = (df['Data e Ora'].dt.date >= start_date) & (df['Data e Ora'].dt.date <= end_date)
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
                ax.set_title(f'Andamento temporale {"".join(sensors)} - {start_date} a {end_date}')
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

# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
