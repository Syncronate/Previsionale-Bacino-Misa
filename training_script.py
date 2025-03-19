import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import os
import matplotlib.pyplot as plt

# Definisci i nomi delle colonne e dei sensori - **AGGIORNATO**
feature_columns = [
    'cumulata_sensore_1295_arcevia',
    'cumulata_sensore_2637_bettolelle',
    'cumulata_sensore_2858_barbara',
    'cumulata_sensore_2964_corinaldo',
    'umidita_sensore_3452_montemurello',
    'livello_idrometrico_sensore_1008_serra_dei_conti',
    'livello_idrometrico_sensore_1112_bettolelle',
    'livello_idrometrico_sensore_1283_corinaldo_nevola',
    'livello_idrometrico_sensore_3072_pianello_di_ostra',
    'livello_idrometrico_sensore_3405_ponte_garibaldi'
]
target_sensors = [
    'livello_idrometrico_sensore_1008_serra_dei_conti',
    'livello_idrometrico_sensore_1112_bettolelle',
    'livello_idrometrico_sensore_1283_corinaldo_nevola',
    'livello_idrometrico_sensore_3072_pianello_di_ostra',
    'livello_idrometrico_sensore_3405_ponte_garibaldi'
]

# Parametri per la creazione delle sequenze (rimangono invariati)
input_window = 48 # 24 ore a intervalli di 30 minuti
output_horizon = 24 # 12 ore a intervalli di 30 minuti
num_sensors_out = len(target_sensors)
num_features_in = len(feature_columns)

# Funzione per preparare le sequenze di input e target (rimane invariata)
def create_sequences(data, input_window, output_horizon, target_sensors, feature_columns):
    X, y = [], []
    num_samples = len(data) - input_window - output_horizon + 1
    target_sensor_indices = [data.columns.get_loc(sensor) for sensor in target_sensors]
    feature_indices = [data.columns.get_loc(feature) for feature in feature_columns]

    for i in range(num_samples):
        input_seq = data.iloc[i:i+input_window, feature_indices].values
        target_seq = data.iloc[i+input_window:i+input_window+output_horizon, target_sensor_indices].values
        if len(target_seq) == output_horizon: # Assicurati che ci siano abbastanza dati per la finestra target
            X.append(input_seq)
            y.append(target_seq)
    return np.array(X), np.array(y)


# Definizione del modello LSTM (rimane invariata)
class HydrologicalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HydrologicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # Inizializza hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device) # Inizializza cell state
        out, _ = self.lstm(x, (h0, c0)) # out: tensor di shape (batch_size, seq_length, hidden_size)
        # Prendi solo l'output all'ultimo timestep per la previsione sequenziale
        out = self.fc(out[:, -1, :]) # Prendi l'ultimo timestep e applica fully connected layer
        return out.view(x.size(0), output_horizon, num_sensors_out) # Reshape per multi-output multi-step


# Funzioni di training e valutazione (rimangono invariate)
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 10  # Numero di epoche di patience per l'early stopping
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth') # Salva il modello migliore
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print(f'Early stopping triggered after epoch {epoch+1}!')
            break

    model.load_state_dict(torch.load('best_model.pth')) # Carica i pesi del modello migliore
    return history, model


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_loss += loss.item()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(batch_y.cpu().numpy())

    avg_test_loss = test_loss / len(test_loader)
    return avg_test_loss, np.array(predictions), np.array(actuals)


# Funzione per il plot della loss curve (rimane invariata)
def plot_loss_curve(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('loss_curve.png')  # Salva il grafico come immagine
    plt.show()


if __name__ == '__main__':
    # Caricamento dati - **MODIFICATO per gestire Data e Ora e nomi colonne**
    data = pd.read_csv('dataset_idrologico_timeseries.csv', sep='\t')
    data['timestamp'] = pd.to_datetime(data['Data'] + ' ' + data['Ora'], format='%d/%m/%Y %H:%M') # Combina Data e Ora in timestamp
    data = data.sort_values(by='timestamp')
    data = data.drop(columns=['Data', 'Ora', 'timestamp']) # Rimuovi Data, Ora e timestamp dopo l'ordinamento

    # Rinomina colonne per coerenza con feature_columns e target_sensors - **NUOVO**
    data.columns = [
        'cumulata_sensore_1295_arcevia',
        'cumulata_sensore_2637_bettolelle',
        'cumulata_sensore_2858_barbara',
        'cumulata_sensore_2964_corinaldo',
        'umidita_sensore_3452_montemurello',
        'livello_idrometrico_sensore_1008_serra_dei_conti',
        'livello_idrometrico_sensore_1112_bettolelle',
        'livello_idrometrico_sensore_1283_corinaldo_nevola',
        'livello_idrometrico_sensore_3072_pianello_di_ostra',
        'livello_idrometrico_sensore_3405_ponte_garibaldi'
    ]

    # Gestione valori mancanti (imputazione semplice con la media) - (rimane invariata)
    for col in data.columns:
        if data[col].isnull().any():
            data[col] = data[col].fillna(data[col].mean())

    # Normalizzazione dei dati (rimane invariata)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    scaled_data = pd.DataFrame(scaled_data, columns=data.columns)

    # Creazione sequenze (rimane invariata)
    X, y = create_sequences(scaled_data, input_window, output_horizon, target_sensors, feature_columns)

    # Divisione in training, validation e test set (rimane invariata)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, shuffle=False) # shuffle=False per dati time-series
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False) # shuffle=False per dati time-series

    # Conversione in tensori PyTorch e creazione DataLoader (rimane invariata)
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

    batch_size = 32
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False) # shuffle=False per dati time-series
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False) # shuffle=False per dati time-series
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False) # shuffle=False per dati time-series

    # Inizializzazione modello, loss function e optimizer (rimane invariata)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = X_train_tensor.shape[2] # Numero di features in input
    hidden_size = 64
    num_layers = 2
    output_size = output_horizon * num_sensors_out # Output per 12 ore per 5 sensori

    model = HydrologicalLSTM(input_size, hidden_size, num_layers, output_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100

    # Training del modello (rimane invariata)
    history, trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)

    # Valutazione sul test set (rimane invariata)
    test_loss, predictions, actuals = evaluate_model(trained_model, test_loader, criterion, device)
    print(f'Test Loss: {test_loss:.4f}')

    # Plot della loss curve (rimane invariata)
    plot_loss_curve(history)

    # Salvataggio del modello e dello scaler (rimane invariata)
    torch.save(trained_model.state_dict(), 'hydrological_lstm_model.pth')
    torch.save(scaler, 'scaler.pkl')
    print("Modello e scaler salvati con successo!")
