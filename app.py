import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3 Ultra", page_icon="🎧", layout="wide")

# Initialisation de l'historique de session
if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN : INTERFACE CLAIRE ET LISIBLE ---
st.markdown("""
    <style>
    /* Fond de page clair et moderne */
    .stApp { background-color: #F8F9FA; color: #212529; }
    
    /* Titre principal */
    h1 { font-family: 'Segoe UI', Roboto, Helvetica, sans-serif; color: #1A1A1A; text-align: center; font-weight: 800; padding-bottom: 10px; }
    
    /* Cartes de métriques (Clé, BPM, Énergie) */
    div[data-testid="stMetricValue"] { color: #D4AF37 !important; font-weight: bold; }
    .stMetric { 
        background-color: #FFFFFF !important; 
        border: 1px solid #E0E0E0 !important; 
        border-radius: 12px; 
        padding: 15px; 
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Cartes de l'historique */
    .history-card { 
        background-color: #FFFFFF; 
        padding: 15px; 
        border-radius: 10px; 
        border-left: 5px solid #D4AF37; 
        margin-bottom: 10px; 
        color: #333;
        box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        font-size: 1.05rem;
    }
    
    /* Zone d'upload et expanders */
    .stExpander { border: 1px solid #E0E0E0 !important; background-color: #FFFFFF !important; border-radius: 10px !important; }
    p { font-size: 1.1rem; color: #444; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE HARMONIQUE & MAPPING ---
BASE_CAMELOT = {
    'B': '1', 'Cb': '1', 'F#': '2', 'Gb': '2', 'Db': '3', 'C#': '3', 
    'Ab': '4', 'G#': '4', 'Eb': '5', 'D#': '5', 'Bb': '6', 'A#': '6', 
    'F': '7', 'C': '8', 'G': '9', 'D': '10', 'A': '11', 'E': '12'
}

FREQS = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 
         'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}

def get_camelot_pro(key, mode):
    # Règle utilisateur : F# MINOR = 11A
    if key == 'F#' and mode in ['minor', 'dorian']: return "11A"
    if key == 'B' and mode in ['minor', 'dorian']: return "10A"
    number = BASE_CAMELOT.get(key, "1")
    letter = "A" if mode in ['minor', 'dorian'] else "B"
    return f"{number}{letter}"

# --- CALCUL DE L'ÉNERGIE (1-10) ---
def calculate_energy(y, sr):
    rms = np.mean(librosa.feature.rms(y=y))
    rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # Pondération : Loudness (50%), Brillance (30%), Tempo (20%)
    energy_score = (rms * 28) + (rolloff / 1100) + (float(tempo) / 160)
    return int(np.clip(energy_score, 1, 10))

# --- MOTEUR D'ANALYSE ULTRA-PRÉCISION ---
def analyze_ultra_precision(y, sr):
    # 1. Compensation de Tuning (décalage par rapport au 440Hz)
    tuning = librosa.estimate_tuning(y=y, sr=sr)
    
    # 2. Filtrage Percussif (HPSS) : Isole la mélodie des percussions
    y_harmonic, _ = librosa.effects.hpss(y, margin=(3.0, 1.0))
    
    # 3. Calcul du Chroma CQT (Précision fréquentielle + filtrage basses C2)
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24, 
                                       tuning=tuning, fmin=librosa.note_to_hz('C2'))
    chroma_avg = np.mean(chroma, axis=1)
    
    NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    PROFILES = {
        "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
        "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
        "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
    }
    
    best_score, res_key, res_mode = -1, "", ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return res_key, res_mode, best_score, tuning

# --- INTERFACE UTILISATEUR ---
st.markdown("<h1>RICARDO_DJ228 | PRECISION V3 ULTRA</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Filtrage Percussif • Correction Tuning • Analyse Dorian • Énergie 1-10</p>", unsafe_allow_html=True)

files = st.file_uploader("", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

if files:
    for file in files:
        with st.expander(f"📂 ANALYSE : {file.name}", expanded=True):
            with st.spinner("Analyse chirurgicale du spectre..."):
                y_full, sr = librosa.load(file)
                duration = librosa.get_duration(y=y_full, sr=sr)
                tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
                energy = calculate_energy(y_full, sr)
                
                votes, timeline_data, tunings = [], [], []
                
                # Analyse segmentée (toutes les 10s sur 15s de fenêtre)
                for start_t in range(0, int(duration) - 15, 10):
                    start_s, end_s = int(start_t * sr), int((start_t + 15) * sr)
                    key, mode, score, t_shift = analyze_ultra_precision(y_full[start_s:end_s], sr)
                    
                    if score > 0.45:
                        m_label = "m" if mode == "minor" else ("-Dor" if mode == "dorian" else "")
                        votes.append(f"{key} {mode}")
                        timeline_data.append({
                            "Temps": start_t, 
                            "Note": f"{key}{m_label}", # Graphique détaillé (Am, A, A-Dor)
                            "Mode": mode, 
                            "Confiance": score
                        })
                        tunings.append(t_shift)

                if votes:
                    final_decision = Counter(votes).most_common(1)[0][0]
                    f_key, f_mode = final_decision.split(" ")
                    f_camelot = get_camelot_pro(f_key, f_mode)
                    avg_tuning = np.mean(tunings) if tunings else 0
                    
                    # Ajout à l'historique
                    st.session_state.history.insert(0, {
                        "Nom": file.name, "Cle": f"{f_key} {f_mode.upper()}",
                        "Camelot": f_camelot, "BPM": int(float(tempo)), "Energy": energy
                    })

                    # Métriques principales
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("TONALITÉ", f"{f_key} {f_mode.upper()}")
                    c2.metric("CODE CAMELOT", f_camelot)
                    c3.metric("NIVEAU ÉNERGIE", f"{energy}/10")
                    c4.metric("TEMPO", f"{int(float(tempo))} BPM")

                    # Graphique détaillé (Stabilité Temporelle)
                    st.markdown("### 📊 Stabilité Harmonique (Détail par segment)")
                    df_plot = pd.DataFrame(timeline_data)
                    fig = px.scatter(df_plot, x="Temps", y="Note", color="Mode", size="Confiance",
                                     color_discrete_map={"major": "#D4AF37", "minor": "#4A90E2", "dorian": "#A259FF"})
                    fig.update_layout(plot_bgcolor='#FDFDFD', paper_bgcolor='white', font_color='#333')
                    st.plotly_chart(fig, use_container_width=True)

                    # Vérification sonore
                    st.markdown("### 🔊 Vérification Auditive")
                    v1, v2 = st.columns(2)
                    with v1: st.audio(file)
                    with v2:
                        target_f = FREQS.get(f_key, 440.0)
                        tone = 0.4 * np.sin(2 * np.pi * target_f * np.linspace(0, 3, int(22050 * 3)))
                        st.audio(tone, sample_rate=22050)
                else:
                    st.warning("Signal trop complexe pour une détection fiable.")

# --- SECTION HISTORIQUE ---
if st.session_state.history:
    st.divider()
    st.subheader("📜 Historique de la session")
    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-card">
            <b>{item['Nom']}</b> | 
            <span style="color:#D4AF37; font-weight:bold;">{item['Cle']} ({item['Camelot']})</span> | 
            {item['BPM']} BPM | <b>Énergie: {item['Energy']}/10</b>
        </div>
        """, unsafe_allow_html=True)
