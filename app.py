import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Pro Analyzer", page_icon="🎧", layout="wide")

if 'history' not in st.session_state:
    st.session_state.history = []

# --- DESIGN HAUTE LISIBILITÉ (CSS) ---
st.markdown("""
    <style>
    /* Fond et texte principal */
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Titre principal */
    h1 { font-family: 'Helvetica Neue', sans-serif; color: #D4AF37; text-align: center; 
         font-size: 3rem !important; font-weight: 800; padding-bottom: 20px; }
    
    /* Sous-titres */
    h2, h3 { color: #D4AF37 !important; font-weight: 600 !important; }

    /* Métriques (Clé, Camelot, BPM) */
    div[data-testid="stMetricValue"] { font-size: 2.5rem !important; font-weight: 700 !important; color: #FFFFFF !important; }
    div[data-testid="stMetricLabel"] { font-size: 1.1rem !important; color: #D4AF37 !important; text-transform: uppercase; letter-spacing: 1px; }
    div[data-testid="stMetric"] { background-color: #1E2129; border: 2px solid #30363D; border-radius: 15px; padding: 20px !important; }

    /* Cartes Historique */
    .history-card { background-color: #1E2129; padding: 15px; border-radius: 10px; 
                     border: 1px solid #30363D; margin-bottom: 10px; font-size: 1.1rem; line-height: 1.6; }
    .history-card b { color: #D4AF37; font-size: 1.2rem; }

    /* Audio et Upload */
    .stAudio { margin-top: 10px; }
    
    /* Texte d'aide */
    .stMarkdown p { font-size: 1.1rem !important; line-height: 1.7; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
BASE_CAMELOT = {
    'B': '1', 'Cb': '1', 'F#': '2', 'Gb': '2', 'Db': '3', 'C#': '3', 
    'Ab': '4', 'G#': '4', 'Eb': '5', 'D#': '5', 'Bb': '6', 'A#': '6', 
    'F': '7', 'C': '8', 'G': '9', 'D': '10', 'A': '11', 'E': '12'
}

FREQS = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 
         'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}

def get_camelot_pro(key, mode):
    if key == 'B' and mode in ['minor', 'dorian']: return "10A"
    number = BASE_CAMELOT.get(key, "1")
    letter = "A" if mode in ['minor', 'dorian'] else "B"
    return f"{number}{letter}"

# --- MOTEUR D'ANALYSE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
}

def analyze_ultra_precision(y, sr):
    y_harm = librosa.effects.harmonic(y, margin=4.0)
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, bins_per_octave=24)
    chroma_avg = np.mean(chroma, axis=1)
    best_score = -1
    res_key, res_mode = "", ""
    for mode, profile in PROFILES.items():
        for i in range(12):
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return res_key, res_mode, best_score

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228 | PRO ANALYZER</h1>", unsafe_allow_html=True)

files = st.file_uploader("", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

if files:
    for file in files:
        with st.expander(f"🎵 ANALYSE EN COURS : {file.name.upper()}", expanded=True):
            y_full, sr = librosa.load(file)
            duration = librosa.get_duration(y=y_full, sr=sr)
            tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
            
            votes, timeline_data = [], []
            for start_t in range(0, int(duration) - 15, 10):
                res_key, res_mode, score = analyze_ultra_precision(y_full[start_t*sr : (start_t+15)*sr], sr)
                if score > 0.5:
                    votes.append(f"{res_key} {res_mode}")
                    timeline_data.append({"Temps": start_t, "Note": res_key, "Mode": res_mode, "Confiance": score})

            if votes:
                f_key, f_mode = Counter(votes).most_common(1)[0][0].split(" ")
                f_camelot = get_camelot_pro(f_key, f_mode)
                
                st.session_state.history.insert(0, {
                    "Heure": datetime.now().strftime("%H:%M"),
                    "Nom": file.name, "Cle": f"{f_key} {f_mode.upper()}",
                    "Camelot": f_camelot, "BPM": int(float(tempo))
                })

                # Affichage des résultats
                c1, c2, c3 = st.columns(3)
                c1.metric("Clé Détectée", f"{f_key} {f_mode.upper()}")
                c2.metric("Code Camelot", f_camelot)
                c3.metric("Tempo", f"{int(float(tempo))} BPM")

                st.markdown("---")
                
                # Vérification auditive
                st.subheader("🔊 TEST D'ACCORDAGE")
                v1, v2 = st.columns(2)
                with v1:
                    st.markdown("**Morceau Original**")
                    st.audio(file)
                with v2:
                    st.markdown(f"**Référence : {f_key}**")
                    t = np.linspace(0, 3.0, int(22050 * 3.0), False)
                    tone = 0.4 * np.sin(2 * np.pi * FREQS[f_key] * t) + 0.2 * np.sin(2 * np.pi * (FREQS[f_key] * 2) * t)
                    st.audio(tone, sample_rate=22050)

                # Graphique
                df_plot = pd.DataFrame(timeline_data)
                fig = px.scatter(df_plot, x="Temps", y="Note", color="Mode", size="Confiance",
                                 title="STABILITÉ HARMONIQUE (Nuage de confiance)",
                                 color_discrete_sequence=["#D4AF37", "#4A90E2"], category_orders={"Note": NOTES})
                fig.update_layout(template="plotly_dark", font=dict(size=14))
                st.plotly_chart(fig, use_container_width=True)

# --- HISTORIQUE ---
st.markdown("## 📜 HISTORIQUE DES SESSIONS")
if st.session_state.history:
    h_col1, h_col2 = st.columns([1, 4])
    with h_col1:
        csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode('utf-8')
        st.download_button("📥 EXPORTER CSV", csv, "sessions_ricardo.csv", "text/csv")
    with h_col2:
        if st.button("🗑️ EFFACER TOUT"):
            st.session_state.history = []
            st.rerun()

    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-card">
            🕒 <b>{item['Heure']}</b> | 📄 {item['Nom']} | 
            🔑 <b>{item['Cle']} ({item['Camelot']})</b> | 🥁 {item['BPM']} BPM
        </div>
        """, unsafe_allow_html=True)
