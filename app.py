import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3 Ultra", page_icon="🎧", layout="wide")

# --- INITIALISATION DE L'HISTORIQUE ---
if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    h1 { font-family: 'Segoe UI', sans-serif; color: #1A1A1A; text-align: center; font-weight: 800; }
    .stMetric { background-color: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .alert-box { padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; background-color: #FFEBEB; color: #B30000; font-weight: bold; margin-bottom: 20px; }
    .success-box { padding: 15px; border-radius: 10px; border-left: 5px solid #28A745; background-color: #E8F5E9; color: #1B5E20; font-weight: bold; margin-bottom: 20px; }
    .history-section { background-color: #FFFFFF; padding: 20px; border-radius: 15px; border: 1px solid #DDD; margin-top: 30px; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {
    'Ab': '1A', 'G#': '1A', 'Eb': '2A', 'D#': '2A', 'Bb': '3A', 'A#': '3A',
    'F': '4A', 'C': '5A', 'G': '6A', 'D': '7A', 'A': '8A', 'E': '9A',
    'B': '10A', 'Cb': '10A', 'F#': '11A', 'Gb': '11A', 'Db': '12A', 'C#': '12A'
}
BASE_CAMELOT_MAJOR = {
    'B': '1B', 'Cb': '1B', 'F#': '2B', 'Gb': '2B', 'Db': '3B', 'C#': '3B',
    'Ab': '4B', 'G#': '4B', 'Eb': '5B', 'D#': '5B', 'Bb': '6B', 'A#': '6B',
    'F': '7B', 'C': '8B', 'G': '9B', 'D': '10B', 'A': '11B', 'E': '12B'
}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        if mode in ['minor', 'dorian']:
            return BASE_CAMELOT_MINOR.get(key, "??")
        return BASE_CAMELOT_MAJOR.get(key, "??")
    except: return "??"

def calculate_energy(y, sr):
    rms = np.mean(librosa.feature.rms(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy_score = (rms * 28) + (rolloff / 1100) + (float(tempo) / 160)
    return int(np.clip(energy_score, 1, 10))

def analyze_segment(y, sr):
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    y_harm, _ = librosa.effects.hpss(y, margin=(3.0, 1.0))
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning, fmin=librosa.note_to_hz('C2'))
    chroma_avg = np.mean(chroma, axis=1)
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
    }
    best_s, res_k, res_m = -1, "", ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_s:
                best_s, res_k, res_m = score, NOTES[i], mode
    return f"{res_k} {res_m}", best_s

@st.cache_data(show_spinner="Analyse ultra-précise en cours...")
def get_single_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = calculate_energy(y, sr)
    timeline_data = []
    votes = []
    for start_t in range(0, int(duration) - 15, 10):
        seg, score = analyze_segment(y[int(start_t*sr):int((start_t+15)*sr)], sr)
        if score > 0.45:
            votes.append(seg)
            timeline_data.append({"Temps": start_t, "Note_Mode": seg, "Confiance": score})
    return {"dominante": Counter(votes).most_common(1)[0][0] if votes else "Inconnue",
            "timeline": timeline_data, "tempo": int(float(tempo)), "energy": energy}

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 | ANALYSEUR V3 ULTRA (MULTI)</h1>", unsafe_allow_html=True)

# MODIFICATION 1 : Activation de la sélection multiple
files = st.file_uploader("Importer des fichiers audio", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

if files:
    # On traite les fichiers dans l'ordre inverse pour que le dernier ajouté soit en haut
    for file in reversed(files):
        # MODIFICATION 2 : Vérification pour éviter de re-analyser un fichier déjà en session
        if any(h['Fichier'] == file.name for h in st.session_state.history):
            continue 
            
        res = get_single_analysis(file)
        timeline_data = res["timeline"]
        dominante = res["dominante"]
        
        note_weights = {}
        conf_scores = []
        for d in timeline_data:
            n = d["Note_Mode"]
            note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
            if n == dominante:
                conf_scores.append(d["Confiance"])
        
        if note_weights:
            tonique_synth = max(note_weights, key=note_weights.get)
            camelot = get_camelot_pro(tonique_synth)
            confidence_pct = int(np.mean(conf_scores) * 100) if conf_scores else 0
            
            history_entry = {
                "Heure": datetime.datetime.now().strftime("%H:%M:%S"),
                "Fichier": file.name,
                "Key": tonique_synth,
                "Camelot": camelot,
                "BPM": res['tempo'],
                "Energie": f"{res['energy']}/10",
                "Confiance": f"{confidence_pct}%"
            }
            # Ajout systématique en haut de l'historique
            st.session_state.history.insert(0, history_entry)

    # --- AFFICHAGE DU DERNIER FICHIER ANALYSÉ (FOCUS) ---
    if st.session_state.history:
        last = st.session_state.history[0]
        st.subheader(f"Dernière analyse : {last['Fichier']}")
        cols = st.columns(5)
        cols[0].metric("KEY", last['Key'])
        cols[1].metric("CAMELOT", last['Camelot'])
        cols[2].metric("CONFIANCE", last['Confiance'])
        cols[3].metric("BPM", last['BPM'])
        cols[4].metric("ÉNERGIE", last['Energie'])

# --- SECTION HISTORIQUE ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Historique de Session (Multi-fichiers)")
    df_hist = pd.DataFrame(st.session_state.history)
    st.table(df_hist)
    
    col_h1, col_h2 = st.columns(2)
    csv_data = df_hist.to_csv(index=False).encode('utf-8')
    col_h1.download_button("📂 Exporter l'historique (CSV)", csv_data, "Historique_Ricardo_DJ.csv", "text/csv")
    
    if col_h2.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
