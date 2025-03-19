import streamlit as st
import pandas as pd
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler # Importa anche MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
from io import StringIO
import matplotlib.pyplot as plt
import pickle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

####################################################PARTE 1: DATA ENTRY (Streamlit) - Maschera per inserimento dati nel CSV####################################################
def data_entry_form(df, file_path="dataset_idrologico.csv"):
    """
    Streamlit form for data entry to add new data to the hydrological dataset.
    """
    st.header("Inserimento Nuovo Evento Idrologico")

    next_event_id = get_next_event_id(df)

    fields_info = {
        "data": {"label": "Data Evento", "type": "date", "default": datetime.now(), "streamlit_type": st.date_input}, # ADDED Data field
        "evento": {"label": "ID Evento", "type": "int", "default": next_event_id, "streamlit_type": st.number_input, "kwargs": {"format": "%d", "disabled": True}},
        "saturazione_terreno": {"label": "Saturazione Terreno (%)", "type": "float", "default": 35.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "ore_pioggia_totali": {"label": "Ore pioggia totali", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "cumulata_totale": {"label": "Cumulata Totale (mm)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "pioggia_gg_precedenti": {"label": "Pioggia gg Precedenti (mm)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "intensità_media": {"label": "Intensità Media (mm/h)", "type": "float", "default": 0.0, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1008_inizio": {"label": "Idrometria 1008 Inizio (m)", "type": "float", "default": 0.5, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1112_inizio": {"label": "Idrometria 1112 Inizio (m)", "type": "float", "default": 0.8, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1112_max": {"label": "Idrometria 1112 Max (m)", "type": "float", "default": 0.8, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_1283_inizio": {"label": "Idrometria 1283 Inizio (m)", "type": "float", "default": 1.2, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}},
        "idrometria_3072_inizio": {"label": "Idrometria 3072 Inizio (m)", "type": "float", "default": 0.7, "streamlit_type": st.number_input, "kwargs": {"format": "%.2f"}}
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
                valid_data[field_name] = value.strftime('%Y-%m-%d') # Format date to string YYYY-MM-DD
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
                "idrometria_1112_max": st.column_config.NumberColumn(
                    "Idrometria 1112 Max (m)",
                    help="Valore massimo registrato",
                    format="%.2f",
                    step=0.01
                ),
                "data": st.column_config.DateColumn("Data Evento", format="YYYY-MM-DD") # Display Data column as Date
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
                st.metric("Idrometria 1112 Max Media", f"{df['idrometria_1112_max'].mean():.2f} m")
            with col2:
                st.metric("Cumulata totale media", f"{df['cumulata_totale'].mean():.2f} mm")
                st.metric("Intensità media", f"{df['intensità_media'].mean():.2f} mm/h")
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
        return pd.DataFrame(columns=['data', 'evento', 'saturazione_terreno', 'ore_pioggia_totali', 'cumulata_totale', 'pioggia_gg_precedenti', 'intensità_media', 'idrometria_1008_inizio', 'idrometria_1112_inizio', 'idrometria_1112_max', 'idrometria_1283_inizio', 'idrometria_3072_inizio']) # Return empty DataFrame with columns including 'data'


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
def salva_modello(model, scaler, output_scaler, model_path="hydrological_model.pth", input_scaler_path="input_scaler.pth", output_scaler_path="output_scaler.pth"):
    """
    Salva il modello addestrato e gli scalers per un uso futuro.
    """
    # Salva il modello PyTorch
    torch.save(model.state_dict(), model_path)
    st.success(f"Modello salvato in {model_path}")

    # Salva l'input scaler
    torch.save(scaler, input_scaler_path)
    st.success(f"Input Scaler salvato in {input_scaler_path}")

    # Salva l'output scaler
    torch.save(output_scaler, output_scaler_path)
    st.success(f"Output Scaler salvato in {output_scaler_path}")


def carica_modello(input_size, hidden_size, num_layers, output_size, model_path="hydrological_model.pth", input_scaler_path="input_scaler.pth", output_scaler_path="output_scaler.pth"):
    """
    Carica un modello precedentemente addestrato e gli scalers.
    """
    import os

    if not os.path.exists(model_path) or not os.path.exists(input_scaler_path) or not os.path.exists(output_scaler_path):
        return None, None, None

    try:
        # Inizializza un nuovo modello con la stessa architettura
        model = HydrologicalLSTM(input_size, hidden_size, num_layers, output_size)
        # Carica i parametri del modello salvato
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'))) # Load on CPU to avoid GPU issues
        model.eval()  # Imposta il modello in modalità valutazione

        # Carica l'input scaler
        input_scaler = torch.load(input_scaler_path, map_location=torch.device('cpu'))

        # Carica l'output scaler
        output_scaler = torch.load(output_scaler_path, map_location=torch.device('cpu'))


        st.success("Modello e scalers caricati con successo!")
        return model, input_scaler, output_scaler
    except Exception as e:
        st.error(f"Errore durante il caricamento del modello: {str(e)}")
        return None, None, None

####################################################PARTE 4: INTERFACCIA PER SIMULAZIONI MULTIPLE (Streamlit)####################################################
def multiple_simulations_interface(model, input_scaler, output_scaler, features_cols, output_sensors, df):
    """
    Streamlit interface to run multiple simulations without retraining the model.
    """
    st.header("Simulazioni Multiple - Modello Addestrato")

    sim_defaults = {
        "cumulata_sensore_1295": 0.0,
        "cumulata_sensore_2637": 0.0,
        "cumulata_sensore_2858": 0.0,
        "cumulata_sensore_2964": 0.0,
        "umidita_sensore_3452": 50.0,
        "livello_idrometrico_sensore_1008": 0.5,
        "livello_idrometrico_sensore_1112": 0.8,
        "livello_idrometrico_sensore_1283": 1.2,
        "livello_idrometrico_sensore_3072": 0.7,
        "livello_idrometrico_sensore_3405": 0.7
    }

    # Aggiungi la possibilità di precompilare i campi da un evento esistente
    if not df.empty:
        st.subheader("Precompila campi da evento esistente")
        eventi_disponibili = ["Nessuno (usa valori predefiniti)"] # Initialize with default option
        for index, row in df.iterrows():
            event_label = f"Evento ID: {row['evento']}, Data: {row['data'].strftime('%Y-%m-%d') if isinstance(row['data'], pd.Timestamp) else row['data']}, Cumulata: {row['cumulata_totale']:.2f}, Idro Max 1112: {row['idrometria_1112_max']:.2f}" # Formatted label
            eventi_disponibili.append(event_label)

        selected_event_label = st.selectbox("Seleziona evento", eventi_disponibili)

        if selected_event_label != "Nessuno (usa valori predefiniti)":
            selected_event_id = int(selected_event_label.split("Evento ID: ")[1].split(",")[0]) # Extract event ID from label
            event_data = df[df['evento'] == selected_event_id].iloc[0]
            for field in sim_defaults.keys():
                if field in event_data:
                    sim_defaults[field] = float(event_data[field])
            st.success(f"Campi precompilati con evento {selected_event_id}")

    # Container per il modulo di input e il grafico
    col1, col2 = st.columns([1, 2])

    with col1:
        # Modulo di input più compatto
        with st.form("simulation_parameters_form"):
            st.subheader("Parametri simulazione")
            input_values = {}

            for field, default in sim_defaults.items():
                input_values[field] = st.number_input(
                    field,
                    label=field.replace('_', ' ').title(), # Migliora la label
                    value=default,
                    format="%.2f",
                    step=0.01,
                    label_visibility="visible" # or "collapsed" to hide labels above inputs
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
        input_list = [input_values[feature] for feature in sim_defaults] # Ordine coerente con features_cols
        values = get_simulation_input_values(input_list) # Ora accetta una lista

        if values is not None:
            # Esegui la previsione multi-step e multi-output
            predizioni_scalate = simula_previsione(
                model, input_scaler, values
            )

            # De-normalizza le previsioni
            predizioni_np = output_scaler.inverse_transform(predizioni_scalate)
            predizioni_df = pd.DataFrame(predizioni_np, columns=output_sensors) # DataFrame per facilità

            # Memorizza i risultati della previsione nella session state
            st.session_state['predizioni_df'] = predizioni_df
            st.session_state['valori_input'] = values

            # Genera e visualizza il grafico Plotly per le previsioni multi-sensore
            with col2:
                fig_plotly = visualizza_previsione_multi_sensore_plotly(predizioni_df, output_sensors)
                st.plotly_chart(fig_plotly, use_container_width=True)

        st.session_state['simulation_run'] = False
        st.session_state['simulation_input_values'] = None

    # Se è la prima visualizzazione, mostra un placeholder
    elif col2.container():
        with col2:
            st.info("Inserisci i parametri e clicca 'Esegui Simulazione' per vedere i risultati.")

    if st.session_state.get('simulation_fields_cleared', False):
        clear_simulation_fields_streamlit(sim_defaults)
        st.session_state['simulation_fields_cleared'] = False


def clear_simulation_fields_streamlit(sim_defaults):
    """Pulisce i campi di input simulazione e li resetta ai valori predefiniti."""
    for field, default in sim_defaults.items():
        sim_defaults[field] = default # No need to set value directly, Streamlit form will handle on re-run
    st.session_state['simulation_fields_cleared'] = True # Set flag to re-render form with cleared fields


def get_simulation_input_values(input_list): # Ora accetta una lista di valori
    """Ottiene i valori di input dai campi simulazione."""
    try:
        values = [float(str(val).replace(',', '.')) for val in input_list] # Converte la lista
        return values
    except ValueError:
        st.error("Inserisci valori numerici validi in tutti i campi di simulazione!")
        return None

def run_simulation_streamlit(model, input_scaler, output_scaler, input_values): # Scalers aggiunti
    """Esegui la simulazione con i parametri correnti."""
    values = get_simulation_input_values(input_values)
    if values is None:
        return

    # Esegui la previsione
    predizioni_scalate = simula_previsione(
        model, input_scaler, values
    )
    # De-normalizza (esempio - potresti aver bisogno di un output_scaler)
    predizioni_np = output_scaler.inverse_transform(predizioni_scalate)
    predizioni_df = pd.DataFrame(predizioni_np, columns=output_sensors) # Assumi output_sensors definito globalmente

    # Crea e visualizza il grafico
    fig = visualizza_previsione_multi_sensore_plotly(predizioni_df, output_sensors) # Funzione per multi-sensore
    st.plotly_chart(fig)


####################################################
# Funzione di simulazione della previsione (MODIFICATA per LSTM multi-step)
####################################################
def simula_previsione(modello, input_scaler, input_features_values):
    modello.eval()
    with torch.no_grad():
        nuovi_dati_input_np = np.array(input_features_values).reshape(1, -1) # Reshape per una singola sequenza
        nuovi_dati_input_scaled = input_scaler.transform(nuovi_dati_input_np)
        nuovi_dati_input_tensor = torch.tensor(nuovi_dati_input_scaled, dtype=torch.float32).unsqueeze(1) # Aggiungi dimensione sequenza (lunghezza 1 per input iniziale)

        # Estendi l'input per la lunghezza della sequenza di input richiesta dal modello
        sequence_length_in = 24 # Assumi lunghezza sequenza input definita (o passala come parametro)
        input_sequence = nuovi_dati_input_tensor.repeat(1, sequence_length_in, 1) # Duplica l'input iniziale per formare una sequenza fittizia

        predizioni_tensor = modello(input_sequence) # Passa la sequenza al modello LSTM
        predizioni_scalate = predizioni_tensor.cpu().numpy() # Prendi le previsioni e converti in numpy array

    return predizioni_scalate # Restituisce le previsioni scalate per tutti i sensori e timestep


####################################################
# NUOVA FUNZIONE: Visualizzazione Plotly interattiva per multi-sensore e multi-step
####################################################
def visualizza_previsione_multi_sensore_plotly(predizioni_df, output_sensors):
    """
    Crea una visualizzazione interattiva delle previsioni multi-sensore usando Plotly.
    """
    fig = go.Figure()

    time_steps = list(range(1, len(predizioni_df) + 1)) # Assi X come timesteps (ore future)

    for sensor in output_sensors:
        fig.add_trace(go.Scatter(
            x=time_steps,
            y=predizioni_df[sensor],
            mode='lines+markers',
            name=sensor.replace('_', ' ').title()
        ))

    fig.update_layout(
        title='Previsioni Idrometriche Multi-Sensore (Prossime 12 Ore)',
        xaxis_title='Ore Successive',
        yaxis_title='Livello Idrometrico (m)',
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    return fig


####################################################
# Definizione del Modello LSTM PyTorch (Deve corrispondere all'architettura usata per l'addestramento)
####################################################
class HydrologicalLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(HydrologicalLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size) # Output layer per output_size (n. sensori)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # Prendi solo l'output dell'ultimo timestep per la previsione multi-step (ora corretto per multi-output)
        out = self.fc(out[:, -1, :]) # Applica FC layer all'ultimo timestep
        return out


####################################################MAIN: Esecuzione sequenziale: Data Entry -> Inserimento dati simulazione -> Simulazione (Streamlit Main)####################################################
if __name__ == "__main__":
    st.title("Dashboard Idrologico")

    # Definisci le colonne di input e output (devono corrispondere allo script di training)
    input_features_cols = [
        'cumulata_sensore_1295', 'cumulata_sensore_2637', 'cumulata_sensore_2858', 'cumulata_sensore_2964',
        'umidita_sensore_3452',
        'livello_idrometrico_sensore_1008', 'livello_idrometrico_sensore_1112', 'livello_idrometrico_sensore_1283',
        'livello_idrometrico_sensore_3072', 'livello_idrometrico_sensore_3405'
    ]
    output_sensors_cols = [
        'livello_idrometrico_sensore_1008', 'livello_idrometrico_sensore_1112', 'livello_idrometrico_sensore_1283',
        'livello_idrometrico_sensore_3072', 'livello_idrometrico_sensore_3405'
    ]

    # Initialize session state for dataset and flags
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
    if 'predizioni_df' not in st.session_state:
        st.session_state['predizioni_df'] = None
    if 'valori_input' not in st.session_state:
        st.session_state['valori_input'] = None
    if 'retrain_model' not in st.session_state:
        st.session_state['retrain_model'] = False # Inizializza la variabile per il riallenamento


    dataset_csv = "dataset_idrologico.csv"
    df = st.session_state['dataset']

    # PART 1: Data Entry Form
    with st.expander("Inserimento Dati", expanded=False):
        data_entry_form(df, file_path=dataset_csv)
    df = pd.read_csv(dataset_csv, sep='\t', dtype=str) # Reload dataset after potential save operation
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    for col in df.columns:
        if col != 'evento' and col != 'data': # Exclude 'data' column from numeric conversion
            df[col] = df[col].str.replace(',', '.', regex=False).astype(float)
    df['evento'] = df['evento'].astype(int)
    if 'data' in df.columns: # Convert 'data' column to datetime after reload
        try:
            df['data'] = pd.to_datetime(df['data'])
        except (ValueError, TypeError):
            st.warning("Impossibile convertire la colonna 'data' in formato data. Verificare il formato nel CSV.")
    st.session_state['dataset'] = df # Update dataset in session state


    # Caricamento modello e scalers (parametri devono corrispondere allo script di training)
    input_size = len(input_features_cols)
    hidden_size = 64
    num_layers = 2
    output_size = len(output_sensors_cols)
    model, input_scaler, output_scaler = carica_modello(input_size, hidden_size, num_layers, output_size)


    if model is None or input_scaler is None or output_scaler is None or st.session_state.get('retrain_model'): # Condizione MODIFICATA
        st.warning("Modello non trovato o richiesto riallenamento. Esegui lo script di training `training_script.py` per addestrare e salvare il modello, poi riavvia l'app.")
        if st.session_state.get('retrain_model'):
            st.session_state['retrain_model'] = False # Resetta flag
            st.sidebar.info("Riallenamento richiesto. Esegui lo script `training_script.py`.")
    else:
        # PART 4: Multiple Simulations Interface
        with st.expander("Simulazioni Multiple", expanded=True):
            multiple_simulations_interface(model, input_scaler, output_scaler, input_features_cols, output_sensors_cols, df)

        st.sidebar.header("Informazioni")
        st.sidebar.info("Questa dashboard permette l'inserimento di dati idrologici, la visualizzazione del dataset, e l'esecuzione di simulazioni previsionali multi-step e multi-output sui livelli idrometrici. La parte grafica della simulazione è integrata direttamente nella dashboard sotto il form di simulazione.")

        # Aggiungi il pulsante per riallenare il modello (AGGIUNTO)
        if st.sidebar.button("Riallena Modello"):
            st.session_state['retrain_model'] = True
            st.sidebar.info("Per riallenare il modello, esegui lo script `training_script.py` separatamente.") # Istruzioni chiare


# (INCOLLA QUI TUTTO IL CODICE DEL TUO VECCHIO SIMULATORE)