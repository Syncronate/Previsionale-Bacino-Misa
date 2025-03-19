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
from sklearn.model_selection import train_test_split

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

# Funzione per scaricare modello e scaler
def get_binary_file_downloader_html(bin_file, file_label):
    with open(bin_file, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(bin_file)}">{file_label}</a>'

# Funzione per preparare i dati per l'addestramento
def prepare_training_data(df, feature_columns, hydro_features, input_window, output_window):
    """
    Prepara i dati per l'addestramento, creando sequenze di input e output.

    Args:
        df: DataFrame con i dati
        feature_columns: Colonne da usare come features
        hydro_features: Colonne dei sensori idrometrici (target)
        input_window: Numero di timestep per l'input
        output_window: Numero di timestep per l'output

    Returns:
        X: Sequenze di input
        y: Sequenze di output
    """
    X = []
    y = []

    # Otteniamo i dati normalizzati
    features = df[feature_columns].values
    targets = df[hydro_features].values

    # Creiamo le sequenze
    for i in range(len(df) - input_window - output_window + 1):
        X.append(features[i:i+input_window])
        y.append(targets[i+input_window:i+input_window+output_window])

    return np.array(X), np.array(y)

# Funzione per addestrare il modello
def train_model(X_train, y_train, X_val, y_val, input_size, output_size, output_window,
                hidden_size, num_layers, learning_rate, batch_size, epochs, device):
    """
    Addestra il modello LSTM.

    Args:
        X_train, y_train: Dati di addestramento
        X_val, y_val: Dati di validazione
        input_size: Dimensione dell'input (numero di features)
        output_size: Dimensione dell'output (numero di sensori idrometrici)
        output_window: Numero di timestep per l'output
        hidden_size: Dimensione dello stato nascosto della LSTM
        num_layers: Numero di layer LSTM
        learning_rate: Learning rate per l'ottimizzatore
        batch_size: Dimensione del batch
        epochs: Numero di epoche
        device: Device per l'addestramento (CPU/GPU)

    Returns:
        model: Modello addestrato
        train_losses: Lista delle loss di addestramento
        val_losses: Lista delle loss di validazione
    """
    # Creazione del modello
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers).to(device)

    # Definizione del loss e dell'ottimizzatore
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Preparazione dei dati
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Liste per memorizzare le loss
    train_losses = []
    val_losses = []

    # Addestramento
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        # Creazione dei batch
        for i in range(0, len(X_train), batch_size):
            # Otteniamo il batch
            batch_X = X_train_tensor[i:i+batch_size]
            batch_y = y_train_tensor[i:i+batch_size]

            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)

            # Backward pass e ottimizzazione
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validazione
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()

        # Calcolo delle loss medie
        train_loss = train_loss / (len(X_train) / batch_size)

        # Aggiungiamo le loss alle liste
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Stampa delle loss
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

    return model, train_losses, val_losses

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
                        ['Dashboard', 'Simulazione', 'Analisi Dati Storici', 'Addestramento Modello'])

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
                    ax.set_title(f'Scatterplot: {corr_features[0]} vs {corr_features[1]}')
                    plt.grid(True)
                    plt.tight_layout()
                    st.pyplot(fig)
                    st.markdown(get_image_download_link(fig, "scatterplot.png", "questo scatterplot"), unsafe_allow_html=True)

        elif analysis_type == 'Statistiche descrittive':
            st.subheader('Statistiche descrittive')
            stats_desc = filtered_df[feature_columns].describe()
            st.dataframe(stats_desc)

elif page == 'Addestramento Modello':
    st.header('Addestramento Modello')
    st.write('In questa sezione puoi addestrare un nuovo modello con i tuoi dati.')

    # **1. Caricamento dei dati di addestramento**
    st.subheader('Carica i dati di addestramento')
    training_data_file = st.file_uploader('File CSV per l\'addestramento', type=['csv'])

    if training_data_file is not None:
        try:
            train_df = pd.read_csv(training_data_file, sep=';', parse_dates=['Data e Ora'])
            st.success(f'Dati di addestramento caricati: {len(train_df)} righe')

            # Mostra le prime righe dei dati caricati
            st.dataframe(train_df.head())

            # **2. Configurazione parametri modello**
            st.subheader('Configura i parametri del modello')

            col1, col2 = st.columns(2)
            with col1:
                train_input_window = st.number_input('Input Window', min_value=1, value=INPUT_WINDOW, step=1, help='Numero di timestep di input')
                train_output_window = st.number_input('Output Window', min_value=1, value=OUTPUT_WINDOW, step=1, help='Numero di timestep di output')
                train_hidden_size = st.number_input('Hidden Size LSTM', min_value=32, value=128, step=32, help='Dimensione dello stato nascosto LSTM')
                train_num_layers = st.number_input('Numero di Layers LSTM', min_value=1, value=2, step=1, help='Numero di livelli LSTM')
            with col2:
                train_learning_rate = st.number_input('Learning Rate', min_value=0.0001, value=0.001, step=0.0001, format="%.4f", help='Tasso di apprendimento')
                train_batch_size = st.number_input('Batch Size', min_value=16, value=32, step=16, help='Dimensione del batch')
                train_epochs = st.number_input('Numero di Epoche', min_value=10, value=100, step=10, help='Numero di epoche di addestramento')
                validation_split = st.slider('Validation Split (%)', min_value=10, max_value=50, value=20, step=5, help='Percentuale di dati per la validazione')

            # **3. Esecuzione Addestramento**
            if st.button('Avvia Addestramento'):
                with st.spinner('Addestramento del modello in corso...'):
                    try:
                        # Preparazione dati
                        train_features_df = train_df[feature_columns].copy()

                        # Scalers
                        scaler_train_features = MinMaxScaler()
                        scaler_train_targets = MinMaxScaler()

                        train_features_scaled = scaler_train_features.fit_transform(train_features_df[feature_columns])
                        train_targets_scaled = scaler_train_targets.fit_transform(train_df[hydro_features])

                        train_scaled_df = pd.DataFrame(train_features_scaled, columns=feature_columns)
                        for i, hydro_feature in enumerate(hydro_features):
                            train_scaled_df[hydro_feature] = train_targets_scaled[:,i]
                        train_scaled_df['Data e Ora'] = train_df['Data e Ora'].values # Mantieni la colonna Data e Ora per comodità se serve

                        X, y = prepare_training_data(train_scaled_df, feature_columns, hydro_features, train_input_window, train_output_window)

                        # Split training/validation
                        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split/100, shuffle=False)

                        input_size = X_train.shape[2]
                        output_size = y_train.shape[2]

                        # Addestramento modello
                        trained_model, train_losses, val_losses = train_model(
                            X_train, y_train, X_val, y_val,
                            input_size, output_size, train_output_window,
                            train_hidden_size, train_num_layers,
                            train_learning_rate, train_batch_size,
                            train_epochs, device
                        )

                        st.success('Modello addestrato con successo!')

                        # **4. Visualizzazione Loss**
                        st.subheader('Loss di Addestramento e Validazione')
                        fig_loss, ax_loss = plt.subplots()
                        ax_loss.plot(train_losses, label='Train Loss')
                        ax_loss.plot(val_losses, label='Validation Loss')
                        ax_loss.set_xlabel('Epoca')
                        ax_loss.set_ylabel('Loss (MSE)')
                        ax_loss.set_title('Andamento della Loss durante l\'addestramento')
                        ax_loss.legend()
                        st.pyplot(fig_loss)

                        # **5. Salvataggio Modello e Scaler**
                        st.subheader('Salva Modello e Scaler')

                        # Salva modello
                        model_save_path = 'trained_hydro_model.pth'
                        torch.save(trained_model.state_dict(), model_save_path)

                        # Salva scalers
                        scaler_features_save_path = 'trained_scaler_features.joblib'
                        scaler_targets_save_path = 'trained_scaler_targets.joblib'
                        joblib.dump(scaler_train_features, scaler_features_save_path)
                        joblib.dump(scaler_train_targets, scaler_targets_save_path)

                        st.markdown(get_binary_file_downloader_html(model_save_path, 'Scarica Modello Addestrato (.pth)'), unsafe_allow_html=True)
                        st.markdown(get_binary_file_downloader_html(scaler_features_save_path, 'Scarica Scaler Features (.joblib)'), unsafe_allow_html=True)
                        st.markdown(get_binary_file_downloader_html(scaler_targets_save_path, 'Scarica Scaler Targets (.joblib)'), unsafe_allow_html=True)


                    except Exception as e:
                        st.error(f'Errore durante l\'addestramento: {e}')

        except Exception as e:
            st.error(f'Errore nel caricamento dei dati di addestramento: {e}')
