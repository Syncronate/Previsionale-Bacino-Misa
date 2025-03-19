import streamlit as st
import pandas as pd
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

# Nomi colonne e sensori - **AGGIORNATO**
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
sensor_names_mapping = { # Mappa nomi colonna -> nomi visualizzazione - **AGGIORNATO**
    'livello_idrometrico_sensore_1008_serra_dei_conti': 'Sensore 1008 (Serra dei Conti)',
    'livello_idrometrico_sensore_1112_bettolelle': 'Sensore 1112 (Bettolelle)',
    'livello_idrometrico_sensore_1283_corinaldo_nevola': 'Sensore 1283 (Corinaldo/Nevola)',
    'livello_idrometrico_sensore_3072_pianello_di_ostra': 'Sensore 3072 (Pianello di Ostra)',
    'livello_idrometrico_sensore_3405_ponte_garibaldi': 'Sensore 3405 (Ponte Garibaldi)'
}

input_window = 48 # Deve corrispondere allo script di training
output_horizon = 24 # Deve corrispondere allo script di training
num_sensors_out = len(target_sensors)
num_features_in = len(feature_columns)


####################################################PARTE 1: DATA ENTRY (Streamlit) - Maschera per inserimento dati nel CSV####################################################
def data_entry_form(df, file_path="dataset_idrologico.csv"):
    """Streamlit form for data entry to add new data to the hydrological dataset."""
    st.header("Inserimento Nuovo Evento Idrologico")

    next_event_id = get_next_event_id(df)

    fields_info = { # **AGGIORNATO - Nomi campi e label**
        "data": {"label": "Data Evento", "type": "date", "default": datetime.now(), "streamlit_type": st.date_input},
        "ora": {"label": "Ora Evento", "type": "text", "default": "00:00", "streamlit_type": st.text_input, "kwargs": {"placeholder": "HH:MM"}}, # Campo per l'ora
        "evento": {"label": "ID Evento", "type": "int", "default": next_event_id, "streamlit_type": st.number_input, "kwargs": {"format": "%d", "disabled": True}},
        "cumulata_sensore_1295_arcevia": {"label": "Cumulata Sensore 1295 (Arcevia)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "cumulata_sensore_2637_bettolelle": {"label": "Cumulata Sensore 2637 (Bettolelle)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "cumulata_sensore_2858_barbara": {"label": "Cumulata Sensore 2858 (Barbara)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "cumulata_sensore_2964_corinaldo": {"label": "Cumulata Sensore 2964 (Corinaldo)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "umidita_sensore_3452_montemurello": {"label": "Umidità Sensore 3452 (Montemurello)", "type": "float", "default": 70.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "livello_idrometrico_sensore_1008_serra_dei_conti": {"label": "Livello Idrometrico 1008 (Serra dei Conti)", "type": "float", "default": 0.5, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "livello_idrometrico_sensore_1112_bettolelle": {"label": "Livello Idrometrico 1112 (Bettolelle)", "type": "float", "default": 0.8, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "livello_idrometrico_sensore_1283_corinaldo_nevola": {"label": "Livello Idrometrico 1283 (Corinaldo/Nevola)", "type": "float", "default": 1.2, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "livello_idrometrico_sensore_3072_pianello_di_ostra": {"label": "Livello Idrometrico 3072 (Pianello di Ostra)", "type": "float", "default": 0.7, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "livello_idrometrico_sensore_3405_ponte_garibaldi": {"label": "Livello Idrometrico 3405 (Ponte Garibaldi)", "type": "float", "default": 0.7, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}}
    }

    input_values = {}
    with st.form("data_entry_form"):
        col1, col2, col3 = st.columns(3)
        field_items = list(fields_info.items())
        for i, (field_name, field_info) in enumerate(field_items):
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                kwargs = {"label": field_info["label"], "value": field_info["default"]}
                kwargs.update(field_info.get("kwargs", {}))
                input_values[field_name] = field_info["streamlit_type"](**kwargs)

        col1, col2, col3 = st.columns(3)
        with col1:
            save_button = st.form_submit_button("Salva Dati", use_container_width=True)
        with col2:
            clear_button = st.form_submit_button("Cancella Campi", use_container_width=True)
        with col3:
            view_button = st.form_submit_button("Visualizza Dataset", use_container_width=True)

        if save_button:
            save_data(input_values, df, file_path, fields_info)
        if clear_button:
            st.session_state['fields_cleared'] = True
        if view_button:
            st.session_state['dataset_view'] = True

    if st.session_state.get('dataset_view', False):
        view_dataset_streamlit(df)
        st.session_state['dataset_view'] = False # Reset flag after viewing

    if st.session_state.get('fields_cleared', False):
        clear_fields_streamlit(fields_info)
        st.session_state['fields_cleared'] = False # Reset flag after clearing


def get_next_event_id(df):
    if df.empty:
        return 1
    return int(df['evento'].max()) + 1

def validate_inputs(input_values, fields_info):
    valid_data = {}
    for field_name, field_info in fields_info.items():
        try:
            value = input_values[field_name]
            if field_info["type"] == "int":
                valid_data[field_name] = int(value)
            elif field_info["type"] == "float":
                valid_data[field_name] = float(value)
            elif field_info["type"] == "date": # Handle date type
                valid_data['data'] = value.strftime('%Y-%m-%d') # Format date to string YYYY-MM-DD
            elif field_info["type"] == "text": # Handle time type
                valid_data['ora'] = value # Keep time as string
            else:
                valid_data[field_name] = value
        except ValueError:
            st.error(f"Il valore per {field_info['label']} non è valido.")
            return None
    return valid_data

def save_data(input_values, df, file_path, fields_info):
    valid_data = validate_inputs(input_values, fields_info)
    if valid_data is None:
        return

    new_row = pd.DataFrame([valid_data])
    updated_df = pd.concat([df, new_row], ignore_index=True)

    try:
        updated_df.to_csv(file_path, index=False, sep='\t')
        st.success("Dati salvati con successo nel file CSV!")
        st.session_state['dataset'] = updated_df # Update session state dataset

        next_id = get_next_event_id(updated_df)
        fields_info["evento"]["default"] = next_id # Update default for next entry

    except Exception as e:
        st.error(f"Errore durante il salvataggio: {str(e)}")

def clear_fields_streamlit(fields_info):
    for field_name, field_info in fields_info.items():
        if field_name != "evento":
            fields_info[field_name]["default"] = field_info["default"] # Reset to original default
    st.session_state['fields_cleared'] = True # Set flag to re-render form with cleared fields

def view_dataset_streamlit(df):
    if not df.empty:
        # Aggiungi funzionalità di ordinamento e filtro
        st.subheader("Ultimi 10 eventi del dataset")
        st.dataframe(
            df.tail(10),
            column_config={
                "evento": st.column_config.NumberColumn(format="%d"),
                "livello_idrometrico_sensore_1112_bettolelle": st.column_config.NumberColumn( # **AGGIORNATO - Nome colonna**
                    "Livello Idrometrico 1112 (Bettolelle) (m)",
                    help="Valore massimo registrato",
                    format="%.2f",
                    step=0.01
                ),
                "data": st.column_config.DateColumn("Data Evento", format="YYYY-MM-DD"), # Display Data column as Date
                "ora": st.column_config.TextColumn("Ora Evento") # Display Ora column as Text - **NUOVO**
            },
            use_container_width=True,
            hide_index=True
        )

        # Visualizzazione statistica di base
        if len(df) > 5:  # Solo se abbiamo abbastanza dati
            st.subheader("Statistiche di base")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Numero totale eventi", len(df))
                st.metric("Livello Idrometrico 1112 Max Medio", f"{df['livello_idrometrico_sensore_1112_bettolelle'].mean():.2f} m") # **AGGIORNATO - Nome colonna**
            with col2:
                # Le statistiche sulla cumulata totale e intensità media non sono più direttamente applicabili qui
                # Puoi adattare queste metriche se hai nuove colonne pertinenti per le statistiche
                pass # Rimozione metriche non pertinenti
    else:
        st.info("Dataset vuoto.")
    st.session_state['dataset_view'] = True # Set flag to show dataset view

def prepare_initial_dataset(output_file="dataset_idrologico.csv"):
    """
    Carica il dataset dal CSV se esiste.
    Se il file non esiste, restituisce un DataFrame vuoto.
    """
    if os.path.exists(output_file):
        try:
            df = pd.read_csv(output_file, sep='\t')
            if 'data' in df.columns: # Try to parse 'data' column as datetime if exists
                try:
                    df['data'] = pd.to_datetime(df['data'])
                except (ValueError, TypeError):
                    st.warning("Impossibile convertire la colonna 'data' in formato data. Verificare il formato nel CSV.")

            st.success(f"File {output_file} caricato con successo.")
            return df
        except Exception as e:
            st.error(f"Errore nel caricamento del file esistente: {str(e)}")
            return pd.DataFrame() # Restituisci DataFrame vuoto in caso di errore
    else:
        st.info(f"File {output_file} non trovato. Inizializzando un dataset vuoto.")
        return pd.DataFrame(columns=['data', 'ora', 'evento', 'cumulata_sensore_1295_arcevia', 'cumulata_sensore_2637_bettolelle', 'cumulata_sensore_2858_barbara', 'cumulata_sensore_2964_corinaldo', 'umidita_sensore_3452_montemurello', 'livello_idrometrico_sensore_1008_serra_dei_conti', 'livello_idrometrico_sensore_1112_bettolelle', 'livello_idrometrico_sensore_1283_corinaldo_nevola', 'livello_idrometrico_sensore_3072_pianello_di_ostra', 'livello_idrometrico_sensore_3405_ponte_garibaldi']) # Return empty DataFrame with columns including 'data'

####################################################PARTE 2: SIMULAZIONE - Maschera per inserimento dati di simulazione (Streamlit)####################################################
def simulation_data_entry_form(feature_defaults, on_submit):
    """
    Streamlit form for simulation data entry.

    Args:
        feature_defaults: Dictionary with names and default values for each feature.
        on_submit: Callback function that receives the entered values when "Avvia Simulazione" is pressed.
    """
    st.header("Inserisci i dati per la simulazione")

    input_values = {}
    with st.form("simulation_form"):
        for field, default in feature_defaults.items():
            input_values[field] = st.number_input(field, value=default, format="%.2f")

        if st.form_submit_button("Avvia Simulazione"):
            try:
                values = [float(str(input_values[field]).replace(',', '.')) for field in feature_defaults] # Ensure comma is handled correctly
                on_submit(values)
            except ValueError:
                st.error("Verifica di aver inserito correttamente i valori numerici.")

####################################################Funzioni per salvare e caricare il modello####################################################
def carica_modello_e_scaler(model_path="hydrological_lstm_model.pth", scaler_path="scaler.pkl"):
    """Carica modello LSTM e scaler."""
    device = torch.device('cpu') # Carica sempre su CPU per compatibilità Streamlit Cloud

    try:
        scaler = torch.load(scaler_path) # Carica scaler salvato come oggetto PyTorch
    except Exception as e:
        st.error(f"Errore caricando lo scaler: {e}")
        return None, None

    try:
        # Inizializza il modello con gli stessi parametri usati in training
        input_size = num_features_in
        hidden_size = 64
        num_layers = 2
        output_size = output_horizon * num_sensors_out
        model = HydrologicalLSTM(input_size, hidden_size, num_layers, output_size)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval() # Mettere in modalità valutazione
        return model, scaler
    except Exception as e:
        st.error(f"Errore caricando il modello: {e}")
        return None, None


####################################################PARTE 4: INTERFACCIA PER SIMULAZIONI MULTIPLE (Streamlit)####################################################
def multiple_simulations_interface(model, scaler, features_cols, df):
    """Streamlit interface to run multiple simulations with LSTM model."""
    st.header("Simulazioni Multiple - Modello LSTM Addestrato")

    sim_defaults = { # **AGGIORNATO - Nomi colonne e valori di default**
        'cumulata_sensore_1295_arcevia': 0.0,
        'cumulata_sensore_2637_bettolelle': 0.0,
        'cumulata_sensore_2858_barbara': 0.0,
        'cumulata_sensore_2964_corinaldo': 0.0,
        'umidita_sensore_3452_montemurello': 70.0,
        'livello_idrometrico_sensore_1008_serra_dei_conti': 0.54,
        'livello_idrometrico_sensore_1112_bettolelle': 1.18,
        'livello_idrometrico_sensore_1283_corinaldo_nevola': 1.18,
        'livello_idrometrico_sensore_3072_pianello_di_ostra': 0.89,
        'livello_idrometrico_sensore_3405_ponte_garibaldi': 0.70,
    }
    input_feature_names = list(sim_defaults.keys()) # Ordine deve corrispondere a feature_columns

    # Container per input e grafico
    col1, col2 = st.columns([1, 2])

    with col1:
        with st.form("simulation_parameters_form"):
            st.subheader("Parametri simulazione")
            input_values = {}
            for field in input_feature_names: # Usa input_feature_names per l'ordine corretto
                input_values[field] = st.number_input(
                    field,
                    label=field.replace('_', ' ').title(), # Label più leggibile
                    value=sim_defaults[field],
                    format="%.2f",
                    step=0.01,
                    label_visibility="visible"
                )

            simulate_button = st.form_submit_button("Esegui Simulazione", use_container_width=True)
            clear_button = st.form_submit_button("Pulisci Campi", use_container_width=True)

            if simulate_button:
                st.session_state['simulation_run'] = True
                st.session_state['simulation_input_values'] = input_values

            if clear_button:
                st.session_state['simulation_fields_cleared'] = True

    # Visualizzazione risultati simulazione
    if st.session_state.get('simulation_run', False):
        input_values = st.session_state.get('simulation_input_values', sim_defaults)
        values = get_simulation_input_values(input_values, input_feature_names) # Passa anche i nomi delle features

        if values is not None:
            # Esegui la previsione LSTM
            previsioni_lstm_scaled = simula_previsione_lstm(model, scaler, values, input_feature_names)
            previsioni_lstm = scaler.inverse_transform(np.concatenate([np.zeros((1, len(feature_columns)-num_sensors_out)), previsioni_lstm_scaled.reshape(1, num_sensors_out)], axis=1))[:, -num_sensors_out:].flatten() # Inverse transform solo le idrometrie

            # Memorizza risultati
            st.session_state['previsioni_lstm'] = previsioni_lstm
            st.session_state['valori_input'] = values

            # Visualizza grafico Plotly
            with col2:
                fig_plotly = visualizza_previsione_plotly_lstm(previsioni_lstm, target_sensors, sensor_names_mapping)
                st.plotly_chart(fig_plotly, use_container_width=True)

    elif col2.container():
        with col2:
            st.info("Inserisci i parametri e clicca 'Esegui Simulazione' per vedere i risultati.")

    if st.session_state.get('simulation_fields_cleared', False):
        clear_simulation_fields_streamlit(sim_defaults)
        st.session_state['simulation_fields_cleared'] = False


def clear_simulation_fields_streamlit(sim_defaults):
    """Pulisce i campi di input simulazione e li resetta ai valori predefiniti."""
    for field in sim_defaults.keys():
        sim_defaults[field] = sim_defaults[field] # Streamlit form gestirà il reset
    st.session_state['simulation_fields_cleared'] = True

def get_simulation_input_values(input_values, input_feature_names):
    """Ottiene i valori di input dai campi simulazione nell'ordine corretto."""
    try:
        values = []
        for field in input_feature_names: # Itera secondo l'ordine dei nomi delle features
            value = float(str(input_values[field]).replace(',', '.'))
            values.append(value)
        return np.array(values)
    except ValueError:
        st.error("Inserisci valori numerici validi in tutti i campi di simulazione!")
        return None


####################################################
# Funzione di simulazione della previsione LSTM
####################################################
def simula_previsione_lstm(modello, scaler, input_features_values, input_feature_names):
    """Esegue la previsione con il modello LSTM."""
    modello.eval()
    device = torch.device('cpu') # Assicura che sia su CPU per Streamlit Cloud

    # 1. Prepara input: Assicurati che l'ordine delle features corrisponda al training
    input_dict = {name: value for name, value in zip(input_feature_names, input_features_values)}
    ordered_input_values = np.array([input_dict[feature] for feature in feature_columns]) # Ordina secondo feature_columns
    input_features_values_reshaped = ordered_input_values.reshape(1, -1) # Reshape a 2D array (sample, features)

    # 2. Scaling dell'input
    input_scaled = scaler.transform(input_features_values_reshaped)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32).unsqueeze(0).to(device) # Aggiungi dimensione sequenza (lunghezza 1)

    # 3. Previsione con LSTM
    with torch.no_grad():
        prediction_scaled = modello(input_tensor) # Ottieni previsione scalata

    return prediction_scaled.cpu().numpy().reshape(output_horizon, num_sensors_out) # Reshape e converti in numpy


####################################################
# NUOVA FUNZIONE: Visualizzazione Plotly interattiva per LSTM (Multi-output, Multi-step)
####################################################
def visualizza_previsione_plotly_lstm(previsioni, target_sensors, sensor_names_mapping):
    """Visualizza previsioni LSTM multi-step e multi-output con Plotly."""
    fig = go.Figure()
    time_steps = np.arange(1, output_horizon + 1) # 12 ore di previsione (24 passi a 30 min)

    for i, sensor_col in enumerate(target_sensors):
        sensor_name = sensor_names_mapping.get(sensor_col, sensor_col) # Usa nome visualizzazione se disponibile
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=previsioni[:, i],
            mode='lines',
            name=sensor_name
        ))

    fig.update_layout(
        title='Previsioni Livelli Idrometrici (LSTM) - Prossime 12 Ore',
        xaxis_title='Passo Temporale (ogni 30 minuti)',
        yaxis_title='Livello Idrometrico (m)',
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


####################################################
# MAIN: Esecuzione sequenziale Streamlit
####################################################
if __name__ == "__main__":
    st.title("Dashboard Idrologico - LSTM")

    # Inizializza session state (rimane invariato)
    if 'dataset' not in st.session_state:
        st.session_state['dataset'] = prepare_initial_dataset()
    if 'dataset_view' not in st.session_state:
        st.session_state['dataset_view'] = False
    if 'fields_cleared' not in st.session_state:
        st.session_state['fields_cleared'] = False
    if 'simulation_run' not in st.session_state:
        st.session_state['simulation_run'] = False
    if 'simulation_fields_cleared' not in st.session_state:
        st.session_state['simulation_fields_cleared'] = False
    if 'simulation_input_values' not in st.session_state:
        st.session_state['simulation_input_values'] = None
    if 'simulation_plot' not in st.session_state:
        st.session_state['simulation_plot'] = None
    if 'previsioni_lstm' not in st.session_state:
        st.session_state['previsioni_lstm'] = None
    if 'valori_input' not in st.session_state:
        st.session_state['valori_input'] = None
    if 'retrain_model' not in st.session_state:
        st.session_state['retrain_model'] = False

    dataset_csv = "dataset_idrologico.csv"
    df = st.session_state['dataset']

    # PARTE 1: Data Entry Form (rimane invariata, ma usa nuovi nomi colonne)
    with st.expander("Inserimento Dati", expanded=False):
        data_entry_form(df, file_path=dataset_csv)
    df = pd.read_csv(dataset_csv, sep='\t', dtype=str) # Ricarica dataset dopo salvataggio
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    for col in df.columns:
        if col != 'evento' and col != 'data' and col != 'ora': # Exclude 'ora' too
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    df['evento'] = df['evento'].astype(int)
    if 'data' in df.columns:
        try:
            df['data'] = pd.to_datetime(df['data'])
        except (ValueError, TypeError):
            st.warning("Impossibile convertire colonna 'data' in formato data.")
    st.session_state['dataset'] = df

    # Carica modello e scaler addestrati (rimane invariata)
    model_lstm, scaler_lstm = carica_modello_e_scaler()

    # PARTE 4: Interfaccia Simulazioni Multiple (LSTM) (rimane invariata, ma usa nuovi nomi colonne)
    with st.expander("Simulazioni Multiple LSTM", expanded=True):
        if model_lstm and scaler_lstm:
            multiple_simulations_interface(model_lstm, scaler_lstm, feature_columns, df)
        else:
            st.error("Modello LSTM o scaler non caricati. Assicurati che i file 'hydrological_lstm_model.pth' e 'scaler.pkl' siano presenti nella directory.")

    st.sidebar.header("Informazioni")
    st.sidebar.info("Dashboard per inserimento dati idrologici, visualizzazione dataset e simulazioni previsionali con modello LSTM. La parte grafica è integrata sotto il form di simulazione.")
