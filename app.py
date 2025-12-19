import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3 Ultra", page_icon="🎧", layout="wide")

# --- DESIGN & CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    h1 { font-family: 'Segoe UI', sans-serif; color: #1A1A1A; text-align: center; font-weight: 800; }
    .stMetric { background-color: #FFFFFF !important; border: 1px solid #E0E0E0 !important; border-radius: 12px; padding: 15px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
    .alert-box { padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; background-color: #FFEBEB; color: #B30000; font-weight: bold; margin-bottom: 20px; }
    .success-box { padding: 15px; border-radius: 10px; border-left: 5px solid #28A745; background-color: #E8F5E9; color: #1B5E20; font-weight: bold; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT (F#m=11A, D#m=2A) ---
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

# --- MOTEURS DE CALCUL ---
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

# --- FONCTION D'ANALYSE ---
@st.cache_data(show_spinner="Analyse harmonique en cours...")
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
st.markdown("<h1>RICARDO_DJ228 | ANALYSEUR V3</h1>", unsafe_allow_html=True)

# Accept_multiple_files mis à False
file = st.file_uploader("Choisissez un morceau (MP3, WAV, FLAC)", type=['mp3', 'wav', 'flac'], accept_multiple_files=False)

if file:
    res = get_single_analysis(file)
    
    timeline_data = res["timeline"]
    dominante = res["dominante"]
    
    # --- SYNTHÈSE DE LA TONIQUE ---
    note_weights = {}
    for d in timeline_data:
        n = d["Note_Mode"]
        note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
    
    if note_weights:
        tonique_synth = max(note_weights, key=note_weights.get)
        
        # --- ALERTES ---
        if dominante != tonique_synth:
            st.markdown(f'<div class="alert-box">⚠️ MODULATION DÉTECTÉE : Dominante ({dominante}) vs Tonique Synthèse ({tonique_synth})</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ STABILITÉ HARMONIQUE : Tonique confirmée.</div>', unsafe_allow_html=True)

        # --- MÉTRIQUES ---
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("DOMINANTE (Vote)", dominante)
        with c2:
            st.metric("TONIQUE (Synthèse)", tonique_synth)
        with c3:
            st.metric("CODE CAMELOT", get_camelot_pro(tonique_synth))
        with c4:
            st.metric("BPM / ÉNERGIE", f"{res['tempo']} / {res['energy']}")

        # --- GRAPHIQUE ---
        df = pd.DataFrame(timeline_data)
        fig = px.scatter(df, x="Temps", y="Note_Mode", size="Confiance", color="Note_Mode",
                         title=f"Analyse structurelle : {file.name}")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.error("Analyse impossible : le signal audio est trop complexe ou trop court.")
