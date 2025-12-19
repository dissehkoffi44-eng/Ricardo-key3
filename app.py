import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Pro Analyzer", page_icon="🎧", layout="wide")

# --- DESIGN PERSONNALISÉ (THÈME DARK PRO & GOLD) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    h1 { 
        font-family: 'Helvetica Neue', sans-serif; 
        color: #FFD700; text-align: center; font-weight: 900;
        text-transform: uppercase; letter-spacing: 2px;
        text-shadow: 2px 2px 10px rgba(255, 215, 0, 0.3);
    }
    /* Correction de l'affichage des métriques pour qu'elles soient bien visibles */
    [data-testid="stMetric"] {
        background-color: #1A1C24 !important;
        border: 1px solid #30333D !important;
        border-radius: 15px;
        padding: 15px;
        text-align: center;
    }
    [data-testid="stMetricValue"] { color: #FFD700 !important; }
    [data-testid="stMetricLabel"] { color: #A0A0A0 !important; }
    
    .alert-box { 
        padding: 15px; border-radius: 10px; border-left: 5px solid #FF4B4B; 
        background-color: #2D1B1B; color: #FF8080; font-weight: bold; margin-bottom: 20px; 
    }
    .success-box { 
        padding: 15px; border-radius: 10px; border-left: 5px solid #00C853; 
        background-color: #1B2D1B; color: #80FFAD; font-weight: bold; margin-bottom: 20px; 
    }
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
st.markdown("<h1>RICARDO_DJ228 | V3 ULTRA ANALYZER</h1>", unsafe_allow_html=True)

file = st.file_uploader("", type=['mp3', 'wav', 'flac'], accept_multiple_files=False)

if file:
    res = get_single_analysis(file)
    timeline_data = res["timeline"]
    dominante = res["dominante"]
    
    # Calcul de la Synthèse (Poids mélodique réel)
    note_weights = {}
    for d in timeline_data:
        n = d["Note_Mode"]
        note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
    
    if note_weights:
        tonique_synth = max(note_weights, key=note_weights.get)
        camelot = get_camelot_pro(tonique_synth)
        
        # --- ALERTES ---
        if dominante != tonique_synth:
            st.markdown(f'<div class="alert-box">⚠️ MODULATION : Dominante ({dominante}) vs Tonique Synthèse ({tonique_synth}).</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">✅ ANALYSE STABLE : Tonalité confirmée par synthèse globale.</div>', unsafe_allow_html=True)

        # --- MÉTRIQUES (Vérifie bien que ces 5 lignes s'affichent) ---
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("VOTE (MAJORITÉ)", dominante)
        c2.metric("TONIQUE (SYNTHÈSE)", tonique_synth)
        c3.metric("CODE CAMELOT", camelot)
        c4.metric("BPM", f"{int(res['tempo'])}")
        c5.metric("ÉNERGIE", f"{res['energy']}/10")

        # --- ESPACE POUR LE BOUTON DE TÉLÉCHARGEMENT ---
        st.markdown("### 💾 Exportation")
        report_text = f"RAPPORT RICARDO_DJ228\nMorceau: {file.name}\nDominante (Vote): {dominante}\nTonique (Synthèse): {tonique_synth}\nCamelot: {camelot}\nBPM: {int(res['tempo'])}\nEnergie: {res['energy']}/10"
        
        st.download_button(
            label="📥 TÉLÉCHARGER LE RAPPORT COMPLET",
            data=report_text,
            file_name=f"Analyse_RicardoDJ_{file.name}.txt",
            mime="text/plain"
        )

        # --- GRAPHIQUE ---
        st.markdown("---")
        df = pd.DataFrame(timeline_data)
        fig = px.scatter(df, x="Temps", y="Note_Mode", size="Confiance", color="Note_Mode",
                         template="plotly_dark", title=f"Visualisation : {file.name}")
        st.plotly_chart(fig, use_container_width=True)
