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

# Model class with corrected reshape in forward method
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

        # Fully connected layer for hydrometric level prediction
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)

        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        # LSTM forward pass
        # out shape: (batch_size, seq_len, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Take only the output of the last timestep
        out = out[:, -1, :]

        # Fully connected layer
        # out shape: (batch_size, output_size * output_window)
        out = self.fc(out)

        # Reshaping to get output sequence - FIXED RESHAPE
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


# Support functions for data preparation and training - INTEGRATED NEW TRAINING SCRIPT

def prepare_training_data(df, input_window, output_window, val_split, feature_columns, target_features):
    """
    Prepares data for training the LSTM model.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        input_window (int): Length of input time window.
        output_window (int): Length of output time window.
        val_split (float): Percentage of data to use for validation.
        feature_columns (list): Names of columns to use as input features.
        target_features (list): Names of columns to predict (targets).

    Returns:
        tuple: X_train, y_train, X_val, y_val, scaler_features, scaler_targets, feature_columns, target_features
    """
    df_train = df.copy()

    # Creating input (X) and output (y) sequences
    X, y = [], []
    for i in range(len(df_train) - input_window - output_window + 1):
        X.append(df_train.iloc[i:i+input_window][feature_columns].values)
        y.append(df_train.iloc[i+input_window:i+input_window+output_window][target_features].values)
    X = np.array(X)
    y = np.array(y)

    # Data normalization
    scaler_features_train = MinMaxScaler()
    scaler_targets_train = MinMaxScaler()
    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])
    X_scaled_flat = scaler_features_train.fit_transform(X_flat)
    y_scaled_flat = scaler_targets_train.fit_transform(y_flat)
    X_scaled = X_scaled_flat.reshape(X.shape)
    y_scaled = y_scaled_flat.reshape(y.shape)

    # Splitting into training and validation sets
    split_idx = int(len(X_scaled) * (1 - val_split/100))
    X_train = X_scaled[:split_idx]
    y_train = y_scaled[:split_idx]
    X_val = X_scaled[split_idx:]
    y_val = y_scaled[split_idx:]

    return X_train, y_train, X_val, y_val, scaler_features_train, scaler_targets_train, feature_columns, target_features


def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, learning_rate, device, model_path, scaler_features_path, scaler_targets_path, feature_columns, target_features, loss_chart_placeholder): # Aggiunto loss_chart_placeholder
    """
    Trains the LSTM model and updates the loss chart in Streamlit.

    Args:
        model (nn.Module): LSTM model to train.
        X_train (np.array): Training data (features).
        y_train (np.array): Training data (targets).
        X_val (np.array): Validation data (features).
        y_val (np.array): Validation data (targets).
        epochs (int): Number of training epochs.
        batch_size (int): Batch size.
        learning_rate (float): Learning rate.
        device (torch.device): Device (CPU/GPU).
        model_path (str): Path to save the trained model.
        scaler_features_path (str): Path to save the features scaler.
        scaler_targets_path (str): Path to save the targets scaler.
        feature_columns (list): Names of feature columns.
        target_features (list): Names of target columns.
        loss_chart_placeholder (st.empty): Placeholder per il grafico delle loss in Streamlit. # Aggiunto loss_chart_placeholder
    Returns:
        tuple: model, training_loss_history, validation_loss_history
    """

    # Optimizer and Loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    # Convert numpy arrays to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val).to(device)

    # Loss history for plots
    training_loss_history = []
    validation_loss_history = []

    # Training loop
    for epoch in range(epochs):
        model.train()  # Set model to training mode
        total_train_loss = 0.0
        for i in range(0, len(X_train_tensor), batch_size):
            X_batch = X_train_tensor[i:i+batch_size]
            y_batch = y_train_tensor[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / (len(X_train_tensor) // batch_size + (1 if len(X_train_tensor) % batch_size != 0 else 0))
        training_loss_history.append(avg_train_loss)

        # Validation loop
        model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            total_val_loss = 0.0
            for i in range(0, len(X_val_tensor), batch_size):
                X_batch_val = X_val_tensor[i:i+batch_size]
                y_batch_val = y_val_tensor[i:i+batch_size]
                outputs_val = model(X_batch_val)
                val_loss = criterion(outputs_val, y_batch_val)
                total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / (len(X_val_tensor) // batch_size + (1 if len(X_val_tensor) % batch_size != 0 else 0))
            validation_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}')

        # Aggiorna il grafico delle loss nel placeholder
        loss_fig = plot_loss_curves(training_loss_history, validation_loss_history)
        loss_chart_placeholder.pyplot(loss_fig) # Aggiorna il placeholder con il nuovo grafico


    # Save the model, scalers, and feature columns
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_features_train, scaler_features_path)
    joblib.dump(scaler_targets_train, scaler_targets_path)

    print(f"Model saved at: {model_path}")
    print(f"Features scaler saved at: {scaler_features_path}")
    print(f"Targets scaler saved at: {scaler_targets_path}")

    return model, training_loss_history, validation_loss_history


def plot_loss_curves(training_loss_history, validation_loss_history):
    """
    Plots training and validation loss curves.

    Args:
        training_loss_history (list): List of training losses for each epoch.
        validation_loss_history (list): List of validation losses for each epoch.

    Returns:
        matplotlib.figure.Figure: Matplotlib figure containing the plot.
    """
    epochs_range = range(1, len(training_loss_history) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_range, training_loss_history, 'b-', label='Training Loss')
    plt.plot(epochs_range, validation_loss_history, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    return plt.gcf()  # Get current figure


def evaluate_model_on_test(model, test_df, scaler_features, scaler_targets, input_window, output_window, feature_columns, target_features, device, batch_size):
    """
    Evaluates the model on the test set.

    Args:
        model (nn.Module): Trained LSTM model.
        test_df (pd.DataFrame): DataFrame containing test data.
        scaler_features (sklearn.preprocessing.MinMaxScaler): Scaler for features.
        scaler_targets (sklearn.preprocessing.MinMaxScaler): Scaler for targets.
        input_window (int): Length of input time window.
        output_window (int): Length of output time window.
        feature_columns (list): Names of feature columns.
        target_features (list): Names of target columns.
        device (torch.device): Device (CPU/GPU).
        batch_size (int): Batch size for evaluation.

    Returns:
        tuple: test_loss, predictions, actuals
    """
    model.eval()
    criterion = nn.MSELoss()
    total_test_loss = 0.0
    predictions_list = []
    actuals_list = []

    # Prepare test data in the same way as training data
    X_test, y_test = [], []
    for i in range(len(test_df) - input_window - output_window + 1):
        X_test.append(test_df.iloc[i:i+input_window][feature_columns].values)
        y_test.append(test_df.iloc[i+input_window:i+input_window+output_window][target_features].values)

    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Normalize test data using the training scalers
    X_test_flat = X_test.reshape(-1, X_test.shape[-1])
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])
    X_test_scaled_flat = scaler_features.transform(X_test_flat)
    y_test_scaled_flat = scaler_targets.transform(y_test_flat)
    X_test_scaled = X_test_scaled_flat.reshape(X_test.shape)
    y_test_scaled = y_test_scaled_flat.reshape(y_test.shape)

    X_test_tensor = torch.FloatTensor(X_test_scaled).to(device)
    y_test_tensor = torch.FloatTensor(y_test_scaled).to(device)

    with torch.no_grad():  # No gradient calculation during testing
        for i in range(0, len(X_test_tensor), batch_size):
            X_batch_test = X_test_tensor[i:i+batch_size]
            y_batch_test = y_test_tensor[i:i+batch_size]

            outputs_test = model(X_batch_test)
            test_loss_batch = criterion(outputs_test, y_batch_test)
            total_test_loss += test_loss_batch.item()

            # Store predictions and actuals for denormalization and further analysis
            predictions_batch = outputs_test.cpu().numpy()
            actuals_batch = y_batch_test.cpu().numpy()
            predictions_list.append(predictions_batch)
            actuals_list.append(actuals_batch)

    avg_test_loss = total_test_loss / (len(X_test_tensor) // batch_size + (1 if len(X_test_tensor) % batch_size != 0 else 0))
    print(f'Test Loss: {avg_test_loss:.4f}')

    # Concatenate all predictions and actuals
    predictions_np = np.concatenate(predictions_list, axis=0)
    actuals_np = np.concatenate(actuals_list, axis=0)

    # Denormalize predictions and actuals
    predictions_denormalized = scaler_targets.inverse_transform(predictions_np.reshape(-1, len(target_features))).reshape(predictions_np.shape)
    actuals_denormalized = scaler_targets.inverse_transform(actuals_np.reshape(-1, len(target_features))).reshape(actuals_np.shape)

    return avg_test_loss, predictions_denormalized, actuals_denormalized


def plot_test_predictions(predictions, actuals, target_features, output_window):
    """
    Plots test set predictions compared to actual values.

    Args:
        predictions (np.array): Model predictions on test set (denormalized).
        actuals (np.array): Actual values from test set (denormalized).
        target_features (list): Names of target columns.
        output_window (int): Length of output time window.

    Returns:
        list[matplotlib.figure.Figure]: List of matplotlib figures, one for each target feature.
    """
    figures = []
    num_samples_to_plot = min(len(predictions), 10)  # Plot only first 10 samples or less if fewer available

    for i, sensor_name in enumerate(target_features):
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot a limited number of prediction samples to avoid overcrowding
        for sample_idx in range(num_samples_to_plot):
            hours = np.arange(output_window)
            ax.plot(hours, actuals[sample_idx, :, i], marker='o', linestyle='-', label=f'Actual (Sample {sample_idx+1})' if sample_idx == 0 else "", color='blue')  # Actuals
            ax.plot(hours, predictions[sample_idx, :, i], marker='x', linestyle='--', label=f'Prediction (Sample {sample_idx+1})' if sample_idx == 0 else "", color='red')  # Predictions

        ax.set_title(f'Predictions vs Actual - {sensor_name}')
        ax.set_xlabel('Future Hours')
        ax.set_ylabel('Hydrometric Level [m]')
        ax.legend()
        ax.grid(True)
        plt.tight_layout()
        figures.append(fig)
    return figures


# Titolo dell'app
st.title('Dashboard Modello Predittivo Idrologico')

# Sidebar per le opzioni
st.sidebar.header('Impostazioni')

# Opzione per caricare i propri file o usare quelli demo
use_demo_files = st.sidebar.checkbox('Usa file di esempio', value=True)

if use_demo_files:
    # Qui dovresti fornire percorsi ai file di esempio
    DATA_PATH = 'dati_idro.csv'  # Sostituisci con il percorso corretto
    TEST_DATA_PATH = 'dati_idro_test.csv' # Sostituisci con il percorso corretto
    MODEL_PATH = 'best_hydro_model.pth'  # Sostituisci con il percorso corretto
    SCALER_FEATURES_PATH = 'scaler_features.joblib'  # Sostituisci con il percorso corretto
    SCALER_TARGETS_PATH = 'scaler_targets.joblib'  # Sostituisci con il percorso corretto
else:
    # Caricamento dei file dall'utente
    st.sidebar.subheader('Carica i tuoi file')
    data_file = st.sidebar.file_uploader('File CSV con i dati storici', type=['csv'])
    test_data_file = st.sidebar.file_uploader('File CSV con i dati di test (opzionale per allenamento)', type=['csv'])
    model_file = st.sidebar.file_uploader('File del modello (.pth)', type=['pth'])
    scaler_features_file = st.sidebar.file_uploader('File scaler features (.joblib)', type=['joblib'])
    scaler_targets_file = st.sidebar.file_uploader('File scaler targets (.joblib)', type=['joblib'])

    # Controllo se tutti i file sono stati caricati
    if not (data_file and model_file and scaler_features_file and scaler_targets_file):
        st.sidebar.warning('Carica tutti i file necessari per procedere')
    else:
        pass

# Definizione delle costanti - MODIFIED INPUT AND OUTPUT WINDOW
INPUT_WINDOW = 24 # MODIFICATO INPUT_WINDOW A 24 ORE
OUTPUT_WINDOW = 12 # MODIFICATO OUTPUT_WINDOW A 12 ORE
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 50
BATCH_SIZE = 32
LEARNING_RATE = 0.001
VALIDATION_SPLIT = 20

# Caricamento dei dati storici
df = None # Initialize df to None
test_df = None # Initialize test_df to None
try:
    if use_demo_files:
        df = pd.read_csv(DATA_PATH, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
        test_df = pd.read_csv(TEST_DATA_PATH, sep=';', parse_dates=['Data e Ora'], decimal=',')
        test_df['Data e Ora'] = pd.to_datetime(test_df['Data e Ora'], format='%d/%m/%Y %H:%M')

    elif data_file is not None: # Check if data_file is loaded
        df = pd.read_csv(data_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
        df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
        if test_data_file:
            test_df = pd.read_csv(test_data_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
            test_df['Data e Ora'] = pd.to_datetime(test_df['Data e Ora'], format='%d/%m/%Y %H:%M')

    if df is not None:
        st.sidebar.success(f'Dati caricati: {len(df)} righe')
    if test_df is not None:
        st.sidebar.success(f'Dati di test caricati: {len(test_df)} righe')

except Exception as e:
    st.sidebar.error(f'Errore nel caricamento dei dati: {e}')


# Estrazione delle caratteristiche
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
target_features_original = hydro_features_original

feature_columns = feature_columns_original
hydro_features = hydro_features_original
target_features = target_features_original


input_size = len(feature_columns)
output_size = len(target_features)


model = None
scaler_features = None
scaler_targets = None

# Caricamento del modello e degli scaler SOLO se i file sono stati caricati o si usano quelli demo
if use_demo_files or (data_file and model_file and scaler_features_file and scaler_targets_file):
    try:
        if use_demo_files:
            model, device = load_model(MODEL_PATH, input_size, output_size, OUTPUT_WINDOW)
            scaler_features, scaler_targets = load_scalers(SCALER_FEATURES_PATH, SCALER_TARGETS_PATH)
        else:
            model_bytes = io.BytesIO(model_file.read())
            model, device = load_model(model_bytes, input_size, output_size, OUTPUT_WINDOW)
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
    st.write(f'Inserisci i valori per simulare uno scenario idrologico con il modello aggiornato (Bettolelle). Il modello richiede **{INPUT_WINDOW} ore** di dati di input.') # Descrizione modificata

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

                rain_duration = st.slider("Durata pioggia (ore)", 0, INPUT_WINDOW, 3) # AGGIORNATO RANGE ORE
                rain_start = st.slider("Ora di inizio pioggia", 0, INPUT_WINDOW-1, 0) # AGGIORNATO RANGE ORE

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
                    hour_idx = (rain_start + h) % INPUT_WINDOW # AGGIORNATO MODULO
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
                                        f"Ora {hour_idx+1}", # Hour index + 1 for display
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
                                        f"Ora {hour_idx+1}", # Hour index + 1 for display
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
                                        f"Ora {hour_idx+1}", # Hour index + 1 for display
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
            preview_df_mod.index = [f"Ora {i+1}" for i in range(INPUT_WINDOW)] # Hour index + 1 for display
            st.dataframe(preview_df_mod.round(2))

        else:  # Inserimento manuale completo (ADATTATO ALLE NUOVE FEATURE)
            st.subheader('Inserisci valori per ogni parametro (Modello Bettolelle)') # Testo modificato
            st.write(f"Inserisci i valori medi per le ultime **{INPUT_WINDOW} ore**.") # Instruction updated for 24 hours input

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
                            st.markdown(get_image_download_link(fig, f"sim_mod_{sensor_name}.png", f"il grafico di {feature} (Modello Bettolelle)"), unsafe_allow_html=True)

elif page == 'Analisi Dati Storici':
    st.header('Analisi Dati Storici')

    # ... (rest of Analisi Dati Storici page - no changes needed for model modification itself)

elif page == 'Allenamento Modello':
    st.header('Allenamento Modello')

    # --- Configurazione Allenamento nella UI ---
    st.subheader('Configurazione Allenamento')
    col1, col2 = st.columns(2)
    with col1:
        epochs = st.number_input('Epoche', min_value=1, value=EPOCHS)
        batch_size = st.number_input('Batch Size', min_value=8, value=BATCH_SIZE)
        learning_rate = st.number_input('Learning Rate', step=0.0001, format="%.4f", value=LEARNING_RATE)
    with col2:
        validation_split = st.slider('Validation Split (%)', min_value=10, max_value=50, value=VALIDATION_SPLIT)
        hidden_size = st.number_input('Hidden Size LSTM', min_value=32, value=HIDDEN_SIZE)
        num_layers = st.number_input('Numero di Livelli LSTM', min_value=1, value=NUM_LAYERS)
        dropout_rate = st.slider('Dropout Rate', min_value=0.0, max_value=0.5, step=0.05, value=DROPOUT)

    # --- Scelta features e targets ---
    st.subheader('Selezione Features e Targets')

    # Features selection - using original features for now, can be expanded
    st.write("Features selezionate per l'allenamento:")
    st.dataframe(pd.DataFrame({'Features': feature_columns_original}))

    # Targets selection - using original targets for now, can be expanded
    st.write("Targets selezionati per l'allenamento:")
    st.dataframe(pd.DataFrame({'Targets': target_features_original}))

    # --- Caricamento dati di training e test (opzionale) ---
    st.subheader('Carica dati per Allenamento e Test (opzionale)')
    train_data_file_upload = st.file_uploader("Carica file CSV per dati di training", type=['csv'])
    test_data_file_upload = st.file_uploader("Carica file CSV per dati di test (opzionale)", type=['csv'])

    train_df_for_training = df if use_demo_files else None
    test_df_for_training = test_df if use_demo_files else None

    if train_data_file_upload is not None:
        try:
            train_df_for_training = pd.read_csv(train_data_file_upload, sep=';', parse_dates=['Data e Ora'], decimal=',')
            train_df_for_training['Data e Ora'] = pd.to_datetime(train_df_for_training['Data e Ora'], format='%d/%m/%Y %H:%M')
            st.success("File dati di training caricato con successo.")
        except Exception as e:
            st.error(f"Errore nel caricamento del file di training: {e}")
            train_df_for_training = None

    if test_data_file_upload is not None:
        try:
            test_df_for_training = pd.read_csv(test_data_file_upload, sep=';', parse_dates=['Data e Ora'], decimal=',')
            test_df_for_training['Data e Ora'] = pd.to_datetime(test_df_for_training['Data e Ora'], format='%d/%m/%Y %H:%M')
            st.success("File dati di test caricato con successo.")
        except Exception as e:
            st.error(f"Errore nel caricamento del file di test: {e}")
            test_df_for_training = None


    # --- Bottone per avviare l'allenamento ---
    if st.button('Avvia Allenamento Modello', type="primary"):
        if train_df_for_training is None:
            st.error("Nessun dato di training disponibile. Carica un file CSV o usa i file di esempio.")
        else:
            with st.spinner('Allenamento del modello in corso...'):
                try:
                    # Preparazione dati
                    X_train_val, y_train_val, X_val, y_val, scaler_features_train, scaler_targets_train, feature_columns_trained, target_features_trained = prepare_training_data(
                        train_df_for_training, INPUT_WINDOW, OUTPUT_WINDOW, validation_split, feature_columns_original, target_features_original
                    )

                    # Definizione e training del modello
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model_trained = HydroLSTM(input_size, hidden_size, output_size, OUTPUT_WINDOW, num_layers, dropout_rate).to(device)

                    # Creazione segnaposto per il grafico delle loss
                    loss_chart_placeholder = st.empty() # Aggiunto placeholder qui

                    trained_model, training_loss_history, validation_loss_history = train_model(
                        model_trained, X_train_val, y_train_val, X_val, y_val, epochs, batch_size, learning_rate, device,
                        'trained_model.pth', 'scaler_features_trained.joblib', 'scaler_targets_trained.joblib', feature_columns_trained, target_features_trained,
                        loss_chart_placeholder # Passa il placeholder alla funzione train_model
                    )

                    st.success('Modello allenato con successo!')

                    # Plot delle loss curves FINALE (per assicurarsi che sia visualizzato correttamente)
                    loss_fig = plot_loss_curves(training_loss_history, validation_loss_history)
                    st.pyplot(loss_fig)
                    st.markdown(get_image_download_link(loss_fig, "loss_curves.png", "Scarica grafico Loss Curves"), unsafe_allow_html=True)

                    # --- Evaluation on Test Set (if test data is provided) ---
                    if test_df_for_training is not None:
                        st.subheader('Valutazione sul Test Set')
                        test_loss, test_predictions_denorm, test_actuals_denorm = evaluate_model_on_test(
                            trained_model, test_df_for_training, scaler_features_train, scaler_targets_train, INPUT_WINDOW, OUTPUT_WINDOW, feature_columns_trained, target_features_trained, device, batch_size
                        )
                        st.write(f"**Test Loss:** {test_loss:.4f}")

                        test_predictions_figs = plot_test_predictions(test_predictions_denorm, test_actuals_denorm, target_features_trained, OUTPUT_WINDOW)
                        for i, fig in enumerate(test_predictions_figs):
                            st.pyplot(fig)
                            sensor_name = target_features_trained[i].replace(' ', '_').replace('/', '_')
                            st.markdown(get_image_download_link(fig, f"test_predictions_{sensor_name}.png", f"Scarica grafico Previsioni Test Set per {target_features_trained[i]}"), unsafe_allow_html=True)


                    # --- Download Link for Trained Model ---
                    st.subheader('Download Modello Allenato')
                    model_bytes_buffer = io.BytesIO()
                    torch.save(trained_model.state_dict(), model_bytes_buffer)
                    model_bytes = model_bytes_buffer.getvalue()

                    st.download_button(
                        label="Download Modello (.pth)",
                        data=model_bytes,
                        file_name="trained_hydro_model.pth",
                        mime="application/octet-stream"
                    )
                    st.download_button(
                        label="Download Scaler Features (.joblib)",
                        data=joblib.dumps(scaler_features_train),
                        file_name="scaler_features_trained.joblib",
                        mime="application/octet-stream"
                    )
                    st.download_button(
                        label="Download Scaler Targets (.joblib)",
                        data=joblib.dumps(scaler_targets_train),
                        file_name="scaler_targets_trained.joblib",
                        mime="application/octet-stream"
                    )


                except Exception as e:
                    st.error(f"Errore durante l'allenamento del modello: {e}")
                    st.exception(e) # Mostra i dettagli completi dell'eccezione


# Footer della dashboard
st.sidebar.markdown('---')
st.sidebar.info('Dashboard per modello predittivo idrologico')
