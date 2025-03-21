import streamlit as st
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import io
import base64
from datetime import datetime, timedelta
import joblib
import math
import tempfile
import zipfile
import time
import altair as alt

# Impostare la pagina
st.set_page_config(page_title="Modello previsionale idrologico", layout="wide")

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
def load_model(model_path, input_size, output_size, output_window, hidden_size, num_layers):
    # Impostazione del device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Creazione del modello
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers).to(device)
    # Caricamento dei pesi del modello
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except Exception as e:
        st.error(f"Errore durante il caricamento dei pesi del modello: {e}")
        return None, device
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
        return None, None

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

    return get_input_features, target_features_mod

# Funzione per preparare i dati (modificata)
def prepare_training_data(df, input_window, output_window, val_split, test_split):
    # Ottieni le funzioni per la preparazione dei dati modificati
    get_input_features_func, target_features_mod = modifica_modello_previsionale()
    # Feature di input (include Bettolelle)
    feature_columns = get_input_features_func(df)
    # Creazione delle sequenze di input (X) e output (y)
    X, y = [], []
    for i in range(len(df) - input_window - output_window + 1):
        # X include tutte le feature, compreso lo storico di Bettolelle
        X.append(df.iloc[i:i+input_window][feature_columns].values)
        # y contiene solo Bettolelle per le previsioni future
        y.append(df.iloc[i+input_window:i+input_window+output_window][target_features_mod].values)
    X = np.array(X)
    y = np.array(y)

    # Normalizzazione dei dati
    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    X_flat = X.reshape(-1, X.shape[-1])
    y_flat = y.reshape(-1, y.shape[-1])

    X_scaled_flat = scaler_features.fit_transform(X_flat)
    y_scaled_flat = scaler_targets.fit_transform(y_flat)

    X_scaled = X_scaled_flat.reshape(X.shape)
    y_scaled = y_scaled_flat.reshape(y.shape)

    # Divisione in set di addestramento, validazione e test
    train_end_idx = int(len(X_scaled) * (1 - (val_split + test_split)/100))
    val_end_idx = int(len(X_scaled) * (1 - test_split/100))

    X_train = X_scaled[:train_end_idx]
    y_train = y_scaled[:train_end_idx]
    X_val = X_scaled[train_end_idx:val_end_idx]
    y_val = y_scaled[train_end_idx:val_end_idx]
    X_test = X_scaled[val_end_idx:]
    y_test = y_scaled[val_end_idx:]

    # Manteniamo anche i dati non scalati per il test set
    X_test_unscaled = X[val_end_idx:]
    y_test_unscaled = y[val_end_idx:]

    return (X_train, y_train, X_val, y_val, X_test, y_test, X_test_unscaled, y_test_unscaled,
            scaler_features, scaler_targets, feature_columns, target_features_mod)

# Funzione per creare link di download per un file
def get_download_link(file_path, file_name, text):
    with open(file_path, "rb") as file:
        bytes_data = file.read()
    b64 = base64.b64encode(bytes_data).decode()
    href = f'<a href="data:file/zip;base64,{b64}" download="{file_name}">{text}</a>'
    return href

# Funzione per creare un file zip con tutti i file del modello
def create_model_zip(model_path, scaler_features_path, scaler_targets_path, metadata_path=None):
    # Creiamo un file temporaneo
    temp_zip = tempfile.NamedTemporaryFile(delete=False, suffix='.zip')
    temp_zip.close()

    # Creiamo lo zip
    with zipfile.ZipFile(temp_zip.name, 'w') as zipf:
        zipf.write(model_path, arcname=os.path.basename(model_path))
        zipf.write(scaler_features_path, arcname=os.path.basename(scaler_features_path))
        zipf.write(scaler_targets_path, arcname=os.path.basename(scaler_targets_path))
        if metadata_path:
            zipf.write(metadata_path, arcname=os.path.basename(metadata_path))

    return temp_zip.name

# Funzione per valutazione e visualizzazione delle previsioni sul test set
def evaluate_model(model, X_test, y_test, X_test_unscaled, y_test_unscaled, scaler_targets, target_features, device):
    model.eval()
    y_pred_scaled = []

    with torch.no_grad():
        for i in range(len(X_test)):
            input_tensor = torch.FloatTensor(X_test[i:i+1]).to(device)
            output = model(input_tensor)
            y_pred_scaled.append(output.cpu().numpy().squeeze())

    y_pred_scaled = np.array(y_pred_scaled)

    # Reshape per la denormalizzazione
    y_pred_scaled_flat = y_pred_scaled.reshape(-1, y_pred_scaled.shape[-1])
    y_test_flat = y_test.reshape(-1, y_test.shape[-1])

    # Denormalizzazione
    y_pred = scaler_targets.inverse_transform(y_pred_scaled_flat).reshape(y_pred_scaled.shape)
    y_true = scaler_targets.inverse_transform(y_test_flat).reshape(y_test.shape)

    # Calcolo metriche (MSE, MAE, R2)
    mse = mean_squared_error(y_true.flatten(), y_pred.flatten())
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true.flatten(), y_pred.flatten())
    r2 = r2_score(y_true.flatten(), y_pred.flatten())

    # Preparazione dei dati per la visualizzazione
    results = {
        "metriche": {
            "MSE": mse,
            "RMSE": rmse,
            "MAE": mae,
            "R²": r2
        },
        "previsioni": y_pred,
        "valori_reali": y_true
    }

    return results

# Funzione per visualizzare le previsioni sul test set
def plot_test_predictions(results, target_features, output_window, test_size=10):
    previsioni = results["previsioni"]
    valori_reali = results["valori_reali"]

    # Scegliamo un sottoinsieme di esempi da visualizzare
    num_examples = min(test_size, len(previsioni))
    indices = np.random.choice(len(previsioni), num_examples, replace=False)

    fig, axes = plt.subplots(num_examples, 1, figsize=(12, num_examples * 4), sharex=True)
    if num_examples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        timesteps = np.arange(output_window)
        axes[i].plot(timesteps, valori_reali[idx, :, 0], 'b-', label='Valore reale')
        axes[i].plot(timesteps, previsioni[idx, :, 0], 'r--', label='Previsione')
        axes[i].set_title(f'Esempio {idx+1}: {target_features[0]}')
        axes[i].set_ylabel('Livello (m)')
        axes[i].set_xlabel('Ore future')
        axes[i].legend()
        axes[i].grid(True)

    plt.tight_layout()
    return fig

# Funzione principale per l'allenamento del modello
def train_model(df, params, train_progress_bar, train_progress_text, train_chart_placeholder):
    # Estrazione dei parametri
    input_window = params["input_window"]
    output_window = params["output_window"]
    val_split = params["val_split"]
    test_split = params["test_split"]
    hidden_size = params["hidden_size"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    batch_size = params["batch_size"]
    learning_rate = params["learning_rate"]
    num_epochs = params["num_epochs"]
    patience = params["patience"]
    model_name = params["model_name"]

    # Preparazione dei dati
    (X_train, y_train, X_val, y_val, X_test, y_test, X_test_unscaled, y_test_unscaled,
     scaler_features, scaler_targets, feature_columns, target_features) = prepare_training_data(
        df, input_window, output_window, val_split, test_split)

    # Calcolo dimensioni di input e output
    input_size = X_train.shape[-1]
    output_size = y_train.shape[-1]

    # Inizializzazione del modello
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = HydroLSTM(input_size, hidden_size, output_size, output_window, num_layers=num_layers, dropout=dropout).to(device)

    # Definizione della loss function e dell'optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop
    history = {'train_loss': [], 'val_loss': [], 'epoch': []}
    train_progress_text.text(f"Inizializzazione modello su {device}...")

    # Inizializzazione del grafico per monitoraggio real-time
    train_chart_data = pd.DataFrame({
        'Epoca': [],
        'Loss': [],
        'Tipo': []
    })

    # Creazione del grafico iniziale
    chart = alt.Chart(train_chart_data).mark_line().encode(
        x='Epoca',
        y='Loss',
        color='Tipo',
        tooltip=['Epoca', 'Loss', 'Tipo']
    ).interactive()

    train_chart = train_chart_placeholder.altair_chart(chart, use_container_width=True)

    # Ciclo di training
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        train_progress_text.text(f"Epoch {epoch+1}/{num_epochs} - Training...")
        num_train_batches = (len(X_train) // batch_size) + (1 if len(X_train) % batch_size != 0 else 0)
        batch_progress = st.progress(0.0)

        for batch_idx in range(0, len(X_train), batch_size):
            batch_progress.progress(batch_idx / len(X_train))
            X_batch = torch.FloatTensor(X_train[batch_idx:batch_idx+batch_size]).to(device)
            y_batch = torch.FloatTensor(y_train[batch_idx:batch_size+batch_size]).to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / num_train_batches

        # Validazione
        model.eval()
        val_loss = 0
        train_progress_text.text(f"Epoch {epoch+1}/{num_epochs} - Validazione...")
        num_val_batches = (len(X_val) // batch_size) + (1 if len(X_val) % batch_size != 0 else 0)

        with torch.no_grad():
            for batch_idx in range(0, len(X_val), batch_size):
                X_batch_val = torch.FloatTensor(X_val[batch_idx:batch_idx+batch_size]).to(device)
                y_batch_val = torch.FloatTensor(y_val[batch_idx:batch_batch+batch_size]).to(device)
                output_val = model(X_batch_val)
                loss_val = criterion(output_val, y_batch_val)
                val_loss += loss_val.item()

        avg_val_loss = val_loss / num_val_batches

        # Aggiornamento dello storico
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['epoch'].append(epoch+1)

        # Aggiornamento del grafico in tempo reale
        new_data = pd.DataFrame({
            'Epoca': [epoch+1, epoch+1],
            'Loss': [avg_train_loss, avg_val_loss],
            'Tipo': ['Train', 'Validation']
        })
        train_chart_data = pd.concat([train_chart_data, new_data])
        chart = alt.Chart(train_chart_data).mark_line().encode(
            x='Epoca',
            y=alt.Y('Loss', scale=alt.Scale(type='log')),
            color='Tipo',
            tooltip=['Epoca', 'Loss', 'Tipo']
        ).properties(
            title='Andamento Training e Validation Loss'
        ).interactive()
        train_chart.altair_chart(chart, use_container_width=True)

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                train_progress_text.text(f"Early stopping all'epoca {epoch+1}")
                break

        train_progress_bar.progress((epoch + 1) / num_epochs)
        train_progress_text.text(f"Epoca {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    # Ripristino del miglior modello
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Salvataggio del modello e degli scaler
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'{model_name}_{timestamp}.pth'
    scaler_features_path = f'scaler_features_{model_name}_{timestamp}.joblib'
    scaler_targets_path = f'scaler_targets_{model_name}_{timestamp}.joblib'

    # Salvataggio metadati
    metadata = {
        "input_window": input_window,
        "output_window": output_window,
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "input_size": input_size,
        "output_size": output_size,
        "feature_columns": feature_columns,
        "target_features": target_features,
        "val_loss": best_val_loss,
        "training_date": timestamp
    }
    metadata_path = f'metadata_{model_name}_{timestamp}.joblib'

    # Salvataggio dei file
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler_features, scaler_features_path)
    joblib.dump(scaler_targets, scaler_targets_path)
    joblib.dump(metadata, metadata_path)

    train_progress_text.text(f"Modello salvato in: {model_path}")

    # Valutazione sul test set
    test_results = evaluate_model(model, X_test, y_test, X_test_unscaled, y_test_unscaled,
                                 scaler_targets, target_features, device)

    # Creazione dello zip per download
    zip_path = create_model_zip(model_path, scaler_features_path, scaler_targets_path, metadata_path)

    return {
        'model': model,
        'device': device,
        'scaler_features': scaler_features,
        'scaler_targets': scaler_targets,
        'feature_columns': feature_columns,
        'target_features': target_features,
        'history': history,
        'test_results': test_results,
        'model_path': model_path,
        'scaler_features_path': scaler_features_path,
        'scaler_targets_path': scaler_targets_path,
        'metadata_path': metadata_path,
        'zip_path': zip_path,
        'metadata': metadata
    }

# UI Principale
def main():
    st.title("App per Allenamento e Test di Modelli Previsionali Idrologici")

    # Creazione di tabs
    tab1, tab2, tab3 = st.tabs(["Allenamento Modello", "Test Modello", "Simulazione"])

    with tab1:
        st.header("Allenamento di un Nuovo Modello")
        # Caricamento dati
        uploaded_file = st.file_uploader("Carica il file CSV con i dati idrologici", type=["csv"])
        if uploaded_file is not None:
            try:
                df = pd.read_csv(uploaded_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
                df['Data e Ora'] = pd.to_datetime(df['Data e Ora'], format='%d/%m/%Y %H:%M')
                st.success(f"File caricato con successo. Righe: {len(df)}, Colonne: {len(df.columns)}")
                st.subheader("Anteprima dei dati")
                st.dataframe(df.head())

                st.subheader("Parametri di Allenamento")
                col1, col2 = st.columns(2)
                with col1:
                    input_window = st.slider("Finestra di input (ore)", 1, 24, 6, 1)
                    output_window = st.slider("Finestra di output (ore)", 1, 24, 6, 1)
                    val_split = st.slider("Percentuale validazione", 5, 30, 20, 1)
                    test_split = st.slider("Percentuale test", 5, 30, 10, 1)
                    hidden_size = st.select_slider("Dimensione hidden layer", options=[32, 64, 128, 256, 512], value=128)
                with col2:
                    num_layers = st.slider("Numero di layer LSTM", 1, 4, 2, 1)
                    dropout = st.slider("Dropout", 0.0, 0.5, 0.2, 0.1)
                    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64, 128], value=32)
                    learning_rate = st.select_slider("Learning rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01], value=0.001)
                    num_epochs = st.slider("Numero di epoche", 10, 500, 100, 10)

                patience = st.slider("Patience per early stopping", 5, 50, 15, 1)
                model_name = st.text_input("Nome del modello", "hydro_model")

                if st.button("Avvia Allenamento"):
                    st.subheader("Progresso Allenamento")
                    train_progress_bar = st.progress(0.0)
                    train_progress_text = st.empty()
                    st.subheader("Curve di Loss")
                    train_chart_placeholder = st.empty()

                    params = {
                        "input_window": input_window,
                        "output_window": output_window,
                        "val_split": val_split,
                        "test_split": test_split,
                        "hidden_size": hidden_size,
                        "num_layers": num_layers,
                        "dropout": dropout,
                        "batch_size": batch_size,
                        "learning_rate": learning_rate,
                        "num_epochs": num_epochs,
                        "patience": patience,
                        "model_name": model_name
                    }

                    results = train_model(df, params, train_progress_bar, train_progress_text, train_chart_placeholder)

                    st.subheader("Risultati dell'Allenamento")
                    st.write("Metriche sul Test Set:")
                    metrics = results['test_results']['metriche']
                    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                    col_m1.metric("MSE", f"{metrics['MSE']:.6f}")
                    col_m2.metric("RMSE", f"{metrics['RMSE']:.6f}")
                    col_m3.metric("MAE", f"{metrics['MAE']:.6f}")
                    col_m4.metric("R²", f"{metrics['R²']:.6f}")

                    st.subheader("Visualizzazione Previsioni sul Test Set")
                    fig = plot_test_predictions(results['test_results'], results['target_features'], output_window)
                    st.pyplot(fig)

                    st.subheader("Download del Modello")
                    download_link = get_download_link(results['zip_path'], os.path.basename(results['zip_path']), "Clicca qui per scaricare il modello")
                    st.markdown(download_link, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Errore nel caricamento o nella lettura del file: {e}")

    with tab2:
        st.header("Test del Modello Previsionale")
        st.write("Carica il modello e gli scaler per eseguire delle previsioni su nuovi dati.")
        model_file = st.file_uploader("Carica il file del modello (.pth)", type=["pth"])
        scaler_features_file = st.file_uploader("Carica lo scaler delle feature (.joblib)", type=["joblib"])
        scaler_targets_file = st.file_uploader("Carica lo scaler dei target (.joblib)", type=["joblib"])

        if model_file is not None and scaler_features_file is not None and scaler_targets_file is not None:
            # Salviamo temporaneamente i file caricati
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_model:
                tmp_model.write(model_file.read())
                model_path = tmp_model.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_scaler_features:
                tmp_scaler_features.write(scaler_features_file.read())
                scaler_features_path = tmp_scaler_features.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_scaler_targets:
                tmp_scaler_targets.write(scaler_targets_file.read())
                scaler_targets_path = tmp_scaler_targets.name

            # Parametri del modello (modifica in base al tuo training)
            input_size = st.number_input("Input size", value=10, min_value=1)
            hidden_size = st.number_input("Hidden size", value=128, min_value=1)
            num_layers = st.number_input("Numero di layers", value=2, min_value=1)
            output_window = st.number_input("Finestra di output (ore)", value=6, min_value=1)
            output_size = 1  # Poiché il target è Bettolelle

            model, device = load_model(model_path, input_size, output_size, output_window, hidden_size, num_layers)
            scaler_features, scaler_targets = load_scalers(scaler_features_path, scaler_targets_path)

            st.success("Modello e scaler caricati correttamente!")

            st.subheader("Esempio di Previsione")
            st.write("Inserisci i dati di input per ottenere la previsione per le prossime ore.")
            # Creazione di un form per l'inserimento dei dati
            input_data = []
            for i in range(st.number_input("Numero di ore da simulare", value=6, min_value=1)):
                row = st.text_input(f"Valori separati da virgola per ora {i}", "0,0,0,0,0,0,0,0,0,0")
                try:
                    valori = [float(x.strip()) for x in row.split(",")]
                    input_data.append(valori)
                except Exception as e:
                    st.error(f"Errore nella conversione dei valori: {e}")
            if st.button("Esegui Previsione"):
                if len(input_data) == 0:
                    st.error("Inserisci i dati correttamente.")
                else:
                    input_data = np.array(input_data)
                    # Si assume che il numero di feature usato in training corrisponda a quello inserito
                    hydro_features = ["Bettolelle"]  # Solo target per la previsione
                    prediction = predict(model, input_data, scaler_features, scaler_targets, hydro_features, device, output_window)
                    st.write("Previsione per le prossime ore:")
                    st.dataframe(prediction)

    with tab3:
        st.header("Simulazione")
        st.write("In questa sezione puoi simulare le previsioni a partire da dati d'ingresso personalizzati.")
        # Esempio: utilizzo del modello addestrato in tab1 (se presente) oppure carica nuovamente modello e scaler
        st.write("Utilizza un file CSV di esempio per simulare le previsioni.")
        sim_file = st.file_uploader("Carica il file CSV per la simulazione", type=["csv"], key="sim_file")
        if sim_file is not None:
            try:
                sim_df = pd.read_csv(sim_file, sep=';', parse_dates=['Data e Ora'], decimal=',')
                sim_df['Data e Ora'] = pd.to_datetime(sim_df['Data e Ora'], format='%d/%m/%Y %H:%M')
                st.success(f"File simulazione caricato. Righe: {len(sim_df)}")
                st.dataframe(sim_df.head())
                st.write("Seleziona le ore da utilizzare per la simulazione:")
                input_window = st.number_input("Finestra di input (ore)", value=6, min_value=1)
                # Otteniamo le feature modificate
                get_input_features_func, target_features_mod = modifica_modello_previsionale()
                feature_columns = get_input_features_func(sim_df)
                st.write("Feature utilizzate:", feature_columns)
                # Selezione delle ultime ore per la simulazione
                sim_data = sim_df[feature_columns].tail(input_window).values
                st.write("Anteprima dati di simulazione:")
                st.dataframe(pd.DataFrame(sim_data, columns=feature_columns))

                # Caricamento modello e scaler (se già addestrato in tab1 oppure da file)
                st.write("Carica il modello e gli scaler per la simulazione:")
                model_file_sim = st.file_uploader("Carica il file del modello (.pth)", type=["pth"], key="model_sim")
                scaler_features_file_sim = st.file_uploader("Carica lo scaler delle feature (.joblib)", type=["joblib"], key="scaler_feat_sim")
                scaler_targets_file_sim = st.file_uploader("Carica lo scaler dei target (.joblib)", type=["joblib"], key="scaler_targ_sim")
                if model_file_sim is not None and scaler_features_file_sim is not None and scaler_targets_file_sim is not None:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pth") as tmp_model_sim:
                        tmp_model_sim.write(model_file_sim.read())
                        model_path_sim = tmp_model_sim.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_scaler_feat:
                        tmp_scaler_feat.write(scaler_features_file_sim.read())
                        scaler_features_path_sim = tmp_scaler_feat.name
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".joblib") as tmp_scaler_targ:
                        tmp_scaler_targ.write(scaler_targets_file_sim.read())
                        scaler_targets_path_sim = tmp_scaler_targ.name

                    input_size = len(feature_columns)
                    hidden_size = st.number_input("Hidden size", value=128, min_value=1, key="hidden_sim")
                    num_layers = st.number_input("Numero di layers", value=2, min_value=1, key="layers_sim")
                    output_window = st.number_input("Finestra di output (ore)", value=6, min_value=1, key="output_sim")
                    output_size = 1  # Poiché il target è Bettolelle

                    model_sim, device_sim = load_model(model_path_sim, input_size, output_size, output_window, hidden_size, num_layers)
                    scaler_features_sim, scaler_targets_sim = load_scalers(scaler_features_path_sim, scaler_targets_path_sim)

                    st.success("Modello e scaler per simulazione caricati correttamente!")
                    if st.button("Esegui Simulazione"):
                        prediction_sim = predict(model_sim, sim_data, scaler_features_sim, scaler_targets_sim, target_features_mod, device_sim, output_window)
                        st.subheader("Previsione simulata per le prossime ore:")
                        st.dataframe(prediction_sim)
                        fig, ax = plt.subplots(figsize=(10, 5))
                        hours = [sim_df['Data e Ora'].iloc[-1] + timedelta(hours=i+1) for i in range(output_window)]
                        ax.plot(hours, prediction_sim[:, 0], marker='o', linestyle='-', color='r', label='Previsione')
                        ax.set_title(f'Previsione per {target_features_mod[0]}')
                        ax.set_xlabel('Data/Ora')
                        ax.set_ylabel('Livello (m)')
                        ax.legend()
                        ax.grid(True)
                        plt.xticks(rotation=45)
                        st.pyplot(fig)
            except Exception as e:
                st.error(f"Errore nella simulazione: {e}")

if __name__ == '__main__':
    main()
