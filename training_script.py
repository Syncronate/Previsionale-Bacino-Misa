import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os

# Definisci il modello LSTM
class HydrologicalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HydrologicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        # Prendi solo l'output dell'ultimo timestep per la previsione multi-step
        out = self.fc(out[:, -1, :])
        return out

# Funzioni per la preparazione dei dati in sequenze
def create_sequences(input_data, output_data, seq_length_in, seq_length_out):
    Xs, ys = [], []
    for i in range(len(input_data) - seq_length_in - seq_length_out + 1):
        Xs.append(input_data[i:(i + seq_length_in)])
        ys.append(output_data[(i + seq_length_in):(i + seq_length_in + seq_length_out)])
    return np.array(Xs), np.array(ys)

def prepare_data(csv_path, input_cols, output_cols, seq_length_in, seq_length_out, test_size=0.2, validation_size=0.2):
    df = pd.read_csv(csv_path, sep='\t')

    # Conversione delle colonne al tipo corretto e gestione degli errori
    for col in df.columns:
        if col != 'data': # Mantieni la colonna 'data' come stringa per ora
            try:
                df[col] = pd.to_numeric(df[col].str.replace(',', '.'), errors='coerce')
            except AttributeError: # Gestisci il caso in cui la colonna è già numerica
                pass

    df = df.dropna() # Gestione dei valori mancanti: rimozione righe con NaN

    input_data = df[input_cols].values
    output_data = df[output_cols].values

    # Normalizzazione Min-Max Scaler per input e output separatamente
    input_scaler = MinMaxScaler()
    output_scaler = MinMaxScaler()

    input_scaled = input_scaler.fit_transform(input_data)
    output_scaled = output_scaler.fit_transform(output_data)

    X, y = create_sequences(input_scaled, output_scaled, seq_length_in, seq_length_out)

    # Divisione in training, validation e test set
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size, random_state=42, shuffle=False)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, shuffle=False) # validation_size = test_size = 0.2 of original data

    # Conversione in tensori PyTorch
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    return X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, input_scaler, output_scaler


def train_model(model, X_train, y_train, X_val, y_val, learning_rate, num_epochs, batch_size=32, patience=10):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), 'best_model.pth') # Salva il modello migliore
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f'Early stopping triggered after epoch {epoch+1}!')
            break

    model.load_state_dict(torch.load('best_model.pth')) # Ricarica i pesi del modello migliore
    return model


if __name__ == '__main__':
    # Parametri
    csv_path = 'dataset_idrologico.csv'
    input_cols = [
        'cumulata_sensore_1295', 'cumulata_sensore_2637', 'cumulata_sensore_2858', 'cumulata_sensore_2964',
        'umidita_sensore_3452',
        'livello_idrometrico_sensore_1008', 'livello_idrometrico_sensore_1112', 'livello_idrometrico_sensore_1283',
        'livello_idrometrico_sensore_3072', 'livello_idrometrico_sensore_3405'
    ]
    output_cols = [
        'livello_idrometrico_sensore_1008', 'livello_idrometrico_sensore_1112', 'livello_idrometrico_sensore_1283',
        'livello_idrometrico_sensore_3072', 'livello_idrometrico_sensore_3405'
    ] # Previsione per tutti i sensori
    sequence_length_in = 24 # 24 ore di dati in input (se dati orari)
    sequence_length_out = 12 # Previsione per le successive 12 ore

    input_size = len(input_cols)
    hidden_size = 64
    num_layers = 2
    output_size = len(output_cols) # Output per tutti i sensori
    learning_rate = 0.001
    num_epochs = 200
    patience = 10 # Early stopping patience

    # Preparazione dati
    X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor, input_scaler, output_scaler = prepare_data(
        csv_path, input_cols, output_cols, sequence_length_in, sequence_length_out
    )

    # Inizializzazione e training del modello
    model = HydrologicalLSTM(input_size, hidden_size, num_layers, output_size)
    trained_model = train_model(model, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, learning_rate, num_epochs, patience=patience)

    # Salvataggio del modello e degli scalers
    torch.save(trained_model.state_dict(), 'hydrological_model.pth')
    print('Modello salvato in hydrological_model.pth')
    torch.save(input_scaler, 'input_scaler.pth')
    torch.save(output_scaler, 'output_scaler.pth')
    print('Scalers salvati.')

    # Valutazione finale sul test set (opzionale)
    trained_model.eval()
    with torch.no_grad():
        test_outputs = trained_model(X_test_tensor)
        criterion = nn.MSELoss()
        test_loss = criterion(test_outputs, y_test_tensor)
        print(f'Loss sul Test Set: {test_loss.item():.4f}')