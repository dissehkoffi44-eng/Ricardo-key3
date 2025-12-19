import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3.5 Ultra", page_icon="🎧", layout="wide")

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
    # 1. Estimation fine de l'accordage
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 2. Nettoyage : On garde uniquement les harmoniques (on vire les percussions sèches)
    y_harm, _ = librosa.effects.hpss(y, margin=(3.0, 1.0))
    
    # 3. Filtrage Spectral : Focus sur la zone harmonique (C2 à C7)
    # On utilise 24 bins par octave pour une précision chirurgicale
    chroma = librosa.feature.chroma_cqt(
        y=y_harm, 
        sr=sr, 
        tuning=tuning, 
        fmin=librosa.note_to_hz('C2'), 
        bins_per_octave=24
    )
    chroma_avg = np.mean(chroma, axis=1)
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
    }
    
    best_s, res_k, res_m = -1, "", ""
    for mode, profile in PROFILES.items():
        # Normalisation du profil pour une corrélation robuste
        prof_norm = (profile - np.mean(profile)) / np.std(profile)
        for i in range(12):
            shifted_prof = np.roll(prof_norm, i)
            chroma_norm = (chroma_avg - np.mean(chroma_avg)) / np.std(chroma_avg)
            score = np.corrcoef(chroma_norm, shifted_prof)[0, 1]
            if score > best_s:
                best_s, res_k, res_m = score, NOTES[i], mode
    return f"{res_k} {res_m}", best_s

# --- FONCTION D'ANALYSE PRINCIPALE ---
@st.cache_data(show_spinner="Analyse spectrale haute précision...")
def get_single_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = calculate_energy(y, sr)
    
    timeline_data = []
    votes = []
    
    # Analyse par fenêtres glissantes de 15s toutes les 10s
    for start_t in range(0, int(duration) - 15, 10):
        start_sample = int(start_t * sr)
        end_sample = int((start_t + 15) * sr)
        seg, score = analyze_segment(y[start_sample:end_sample], sr)
        
        if score > 0.40: # Seuil de confiance
            votes.append(seg)
            timeline_data.append({"Temps": start_t, "Note_Mode": seg, "Confiance": score})
            
    return {
        "dominante": Counter(votes).most_common(1)[0][0] if votes else "Inconnue",
        "timeline": timeline_data, 
        "tempo": int(float(tempo)), 
        "energy": energy
    }

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 | ANALYSEUR V3.5 PRO</h1>", unsafe_allow_html=True)

file = st.file_uploader("Importer un fichier audio", type=['mp3', 'wav', 'flac'], accept_multiple_files=False)

if file:
    res = get_single_analysis(file)
    timeline_data = res["timeline"]
    
    if timeline_data:
        # Synthèse pondérée par la confiance
        note_weights = {}
        for d in timeline_data:
            n = d["Note_Mode"]
            note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
        
        tonique_synth = max(note_weights, key=note_weights.get)
        camelot = get_camelot_pro(tonique_synth)
        
        # Historique
        history_entry = {
            "Heure": datetime.datetime.now().strftime("%H:%M:%S"),
            "Fichier": file.name,
            "Key": tonique_synth,
            "Camelot": camelot,
            "BPM": int(res['tempo']),
            "Energie": res['energy']
        }
        
        if not st.session_state.history or st.session_state.history[-1]["Fichier"] != file.name:
            st.session_state.history.append(history_entry)

        # Affichage Alertes
        if res["dominante"] != tonique_synth:
            st.markdown(f'<div class="alert-box">⚠️ COMPLEXITÉ DÉTECTÉE : Variations harmoniques trouvées. Tonique suggérée : {tonique_synth}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ STABILITÉ HARMONIQUE : Signature sonore claire et précise.</div>', unsafe_allow_html=True)

        # Métriques
        cols = st.columns(4)
        cols[0].metric("TONALITÉ", tonique_synth)
        cols[1].metric("CODE CAMELOT", camelot)
        cols[2].metric("BPM", res['tempo'])
        cols[3].metric("ÉNERGIE", f"{res['energy']}/10")

        # Graphique
        df = pd.DataFrame(timeline_data)
        fig = px.line(df, x="Temps", y="Confiance", color="Note_Mode", title="Stabilité du signal par segment")
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.error("Signal trop faible ou bruit excessif pour une analyse fiable.")

# --- SECTION HISTORIQUE ---
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕒 Historique des Analyses")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
