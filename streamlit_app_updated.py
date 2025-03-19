import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import io
import os

# Impostazioni dei parametri (possono essere modificati nella dashboard)
INPUT_WINDOW = 24
OUTPUT_WINDOW = 12
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.2
DEFAULT_DATA_PATH = 'dati_idro.csv' # Percorso predefinito per i dati

# Preparazione dei dati (funzioni dal codice originale)
class HydroDataset(Dataset):
    def __init__(self, features, targets):
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

def prepare_data(data_path, input_window=INPUT_WINDOW, output_window=OUTPUT_WINDOW):
    try:
        df = pd.read_csv(data_path, sep=';', parse_dates=['Data e Ora'])
    except FileNotFoundError:
        st.error(f"File non trovato: {data_path}. Assicurati che il file esista o carica un file CSV.")
        return None, None, None, None, None, None, None

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

    st.write("Tipi di dati iniziali delle colonne:")
    st.write(df[feature_columns].dtypes)

    # Conversione forzata a numerico e gestione degli errori
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    st.write("Tipi di dati dopo la conversione a numerico:")
    st.write(df[feature_columns].dtypes)

    # Riempimento dei valori mancanti con la media
    df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())

    scaler_features = MinMaxScaler()
    scaler_targets = MinMaxScaler()

    features_normalized = scaler_features.fit_transform(df[feature_columns])
    targets_normalized = scaler_targets.fit_transform(df[hydro_features])

    X, y = [], []
    for i in range(len(df) - input_window - output_window + 1):
        X.append(features_normalized[i:i+input_window])
        y.append(targets_normalized[i+input_window:i+input_window+output_window])

    X = np.array(X)
    y = np.array(y)

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train)
    X_val = torch.FloatTensor(X_val)
    y_val = torch.FloatTensor(y_val)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test)

    train_dataset = HydroDataset(X_train, y_train)
    val_dataset = HydroDataset(X_val, y_val)
    test_dataset = HydroDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, scaler_features, scaler_targets, feature_columns, hydro_features

# Definizione del modello LSTM (funzione dal codice originale)
class HydroLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, output_window, num_layers=NUM_LAYERS, dropout=DROPOUT):
        super(HydroLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_window = output_window
        self.output_size = output_size

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, output_size * output_window)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = out.view(out.size(0), self.output_window, self.output_size)
        return out

# Addestramento del modello (funzione modificata per Streamlit)
def train_model(train_loader, val_loader, input_size, output_size, output_window, device, epochs=EPOCHS, learning_rate=LEARNING_RATE):
    model = HydroLSTM(input_size, HIDDEN_SIZE, output_size, output_window).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    progress_bar = st.progress(0)
    status_text = st.empty()

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        status_text.text(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        progress_bar.progress((epoch + 1) / epochs)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_hydro_model.pth')

    model.load_state_dict(torch.load('best_hydro_model.pth'))
    status_text.success("Training completato!")
    return model, train_losses, val_losses

# Valutazione del modello (funzione dal codice originale)
def evaluate_model(model, test_loader, device, scaler_targets, hydro_features):
    model.eval()
    test_loss = 0.0
    criterion = nn.MSELoss()

    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item()

            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    test_loss /= len(test_loader)
    st.write(f'Test Loss: {test_loss:.4f}')

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    pred_shape = all_predictions.shape
    all_predictions_reshaped = all_predictions.reshape(-1, len(hydro_features))
    all_targets_reshaped = all_targets.reshape(-1, len(hydro_features))

    all_predictions_original = scaler_targets.inverse_transform(all_predictions_reshaped)
    all_targets_original = scaler_targets.inverse_transform(all_targets_reshaped)

    all_predictions_original = all_predictions_original.reshape(pred_shape)
    all_targets_original = all_targets_original.reshape(pred_shape)

    return all_predictions_original, all_targets_original, test_loss

# Funzione per la visualizzazione dei risultati (modificata per Streamlit)
def plot_results(predictions, targets, hydro_features, output_window):
    st.subheader("Visualizzazione Previsioni vs Valori Reali")
    sample_idx = np.random.randint(0, len(predictions))

    for i, sensor_name in enumerate(hydro_features):
        fig, ax = plt.subplots(figsize=(10, 4))
        hours = np.arange(output_window)
        ax.plot(hours, predictions[sample_idx, :, i], label='Previsione', marker='o')
        ax.plot(hours, targets[sample_idx, :, i], label='Valore Reale', marker='x')
        ax.set_title(f'Previsione vs Valore Reale - {sensor_name}')
        ax.set_xlabel('Ore future')
        ax.set_ylabel('Livello idrometrico [m]')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
        plt.close(fig) # Close figure to prevent display issues

    st.subheader("Errore Quadratico Medio (MSE) per Sensore")
    mse_per_sensor = np.mean((predictions - targets)**2, axis=(0, 1))
    fig_mse, ax_mse = plt.subplots(figsize=(10, 4))
    ax_mse.bar(hydro_features, mse_per_sensor)
    ax_mse.set_title('MSE per Sensore Idrometrico')
    ax_mse.set_ylabel('MSE')
    ax_mse.tick_params(axis='x', rotation=45)
    st.pyplot(fig_mse)
    plt.close(fig_mse)


# Funzione per fare una previsione (funzione dal codice originale con adattamenti)
def predict(model, input_data, scaler_features, scaler_targets, hydro_features, device):
    model.eval()
    input_normalized = scaler_features.transform(input_data)
    input_tensor = torch.FloatTensor(input_normalized).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
    output_np = output.cpu().numpy().reshape(-1, len(hydro_features))
    predictions = scaler_targets.inverse_transform(output_np)
    predictions = predictions.reshape(OUTPUT_WINDOW, len(hydro_features))
    return predictions

# Streamlit app
def main():
    st.title("Dashboard Previsionale Livelli Idrometrici")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    st.write(f"Dispositivo utilizzato: {device}")

    uploaded_file = st.file_uploader("Carica il tuo file CSV di dati", type=["csv"])
    data_path = DEFAULT_DATA_PATH # Usa il percorso predefinito se non viene caricato un file
    if uploaded_file is not None:
        # Salva il file caricato temporaneamente per essere letto da pandas
        with open("temp.csv", "wb") as f:
            f.write(uploaded_file.getbuffer())
        data_path = "temp.csv"
    elif not os.path.exists(DEFAULT_DATA_PATH):
        st.warning(f"File dati predefinito '{DEFAULT_DATA_PATH}' non trovato. Carica un file CSV per iniziare.")
        return

    if data_path:
        train_dataset, val_dataset, test_dataset, scaler_features, scaler_targets, feature_columns, hydro_features = prepare_data(data_path)

        if train_dataset is None: # Gestisci il caso in cui prepare_data fallisce (e.g., file non trovato)
            return

        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

        input_size = len(feature_columns)
        output_size = len(hydro_features)

        if st.checkbox("Addestra il Modello"):
            epochs = st.slider("Numero di Epoche", 10, 300, EPOCHS)
            learning_rate = st.number_input("Learning Rate", value=LEARNING_RATE, format="%.5f")

            if st.button("Avvia Addestramento"):
                with st.spinner("Addestramento del modello in corso..."):
                    model, train_losses, val_losses = train_model(train_loader, val_loader, input_size, output_size, OUTPUT_WINDOW, device, epochs, learning_rate)
                    st.success("Modello addestrato con successo!")

                    # Visualizza le curve di perdita dopo l'addestramento
                    st.subheader("Curve di Perdita durante l'Addestramento")
                    fig_loss, ax_loss = plt.subplots(figsize=(10, 4))
                    ax_loss.plot(train_losses, label='Train Loss')
                    ax_loss.plot(val_losses, label='Validation Loss')
                    ax_loss.set_title('Curve di Perdita')
                    ax_loss.set_xlabel('Epoca')
                    ax_loss.set_ylabel('Loss')
                    ax_loss.legend()
                    ax_loss.grid(True)
                    st.pyplot(fig_loss)
                    plt.close(fig_loss) # Close figure to prevent display issues

                    # Valutazione e visualizzazione solo dopo l'addestramento
                    st.subheader("Valutazione del Modello")
                    predictions, targets, test_loss = evaluate_model(model, test_loader, device, scaler_targets, hydro_features)
                    plot_results(predictions, targets, hydro_features, OUTPUT_WINDOW)

                    # Download del modello
                    model_state = io.BytesIO()
                    torch.save(model.state_dict(), model_state)
                    model_state.seek(0)
                    st.download_button(
                        label="Scarica Modello Addestrato",
                        data=model_state,
                        file_name="hydro_model.pth",
                        mime="application/octet-stream"
                    )
            else:
                model = None # Modello non addestrato

        else: # Se "Addestra il Modello" non è selezionato, prova a caricare un modello pre-addestrato per la previsione
            model = HydroLSTM(input_size, HIDDEN_SIZE, output_size, OUTPUT_WINDOW).to(device)
            try:
                model.load_state_dict(torch.load('best_hydro_model.pth', map_location=device))
                st.success("Modello pre-addestrato caricato.")
            except FileNotFoundError:
                st.warning("Modello pre-addestrato non trovato. Addestra il modello o carica un modello addestrato.")
                model = None

        if model: # Esegui la previsione solo se c'è un modello (addestrato o caricato)
            st.subheader("Previsione Livelli Idrometrici")
            # Input per la previsione (ultime INPUT_WINDOW ore di dati)
            st.write("Inserisci i dati delle ultime", INPUT_WINDOW, "ore per effettuare la previsione.")

            # Crea un DataFrame vuoto con le colonne delle features per l'input
            input_df = pd.DataFrame(columns=feature_columns, index=range(INPUT_WINDOW))

            # Usa st.data_editor per inserire i dati
            edited_df = st.data_editor(input_df, num_rows="dynamic")

            if st.button("Effettua Previsione"):
                # Converti il DataFrame editato in numpy array e gestisci i valori mancanti
                input_prediction_data = edited_df.fillna(edited_df.mean()).values # Riempi i NaN con la media, potresti voler usare un altro metodo
                if input_prediction_data.shape == (INPUT_WINDOW, len(feature_columns)):
                    with st.spinner("Effettuando la previsione..."):
                        predictions_output = predict(model, input_prediction_data, scaler_features, scaler_targets, hydro_features, device)

                    st.success("Previsione completata!")
                    st.subheader("Previsioni Livelli Idrometrici per le prossime ore:")
                    prediction_df = pd.DataFrame(predictions_output, columns=hydro_features, index=[f"Ora +{h+1}" for h in range(OUTPUT_WINDOW)])
                    st.dataframe(prediction_df)

                    # Visualizza le previsioni con un grafico
                    st.subheader("Visualizzazione Previsioni")
                    for i, sensor_name in enumerate(hydro_features):
                        fig_pred, ax_pred = plt.subplots(figsize=(10, 4))
                        hours_pred = np.arange(OUTPUT_WINDOW)
                        ax_pred.plot(hours_pred, predictions_output[:, i], label='Previsione', marker='o')
                        ax_pred.set_title(f'Previsione Livello Idrometrico - {sensor_name}')
                        ax_pred.set_xlabel('Ore future')
                        ax_pred.set_ylabel('Livello idrometrico [m]')
                        ax_pred.legend()
                        ax_pred.grid(True)
                        st.pyplot(fig_pred)
                        plt.close(fig_pred) # Close figure to prevent display issues
                else:
                    st.error("Inserisci dati completi per le ultime 24 ore.")

    if uploaded_file is not None and os.path.exists("temp.csv"):
        os.remove("temp.csv") # Pulisci il file temporaneo

if __name__ == "__main__":
    main()
