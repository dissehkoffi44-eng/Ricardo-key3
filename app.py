import streamlit as st
import librosa
import numpy as np
import pandas as pd
import plotly.express as px
from collections import Counter
from datetime import datetime
import io

# --- CONFIGURATION ---
st.set_page_config(page_title="Ricardo_DJ228 | Precision V3", page_icon="🎧", layout="wide")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state.history = []

# CSS Thème Studio Dark/Wood
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    h1 { font-family: 'serif'; color: #D4AF37; text-align: center; text-shadow: 2px 2px 4px #000; }
    .stMetric { background-color: #1E1E1E !important; border-left: 5px solid #D4AF37 !important; border-radius: 10px; padding: 15px; }
    .history-card { background-color: #1E1E1E; padding: 12px; border-radius: 8px; border-bottom: 1px solid #333; margin-bottom: 8px; color: #BBB; }
    .stExpander { border: 1px solid #333 !important; background-color: #181818 !important; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
BASE_CAMELOT = {
    'B': '1', 'Cb': '1', 'F#': '2', 'Gb': '2', 'Db': '3', 'C#': '3', 
    'Ab': '4', 'G#': '4', 'Eb': '5', 'D#': '5', 'Bb': '6', 'A#': '6', 
    'F': '7', 'C': '8', 'G': '9', 'D': '10', 'A': '11', 'E': '12'
}

# Fréquences pour vérification auditive
FREQS = {'C': 261.63, 'C#': 277.18, 'D': 293.66, 'D#': 311.13, 'E': 329.63, 'F': 349.23, 
         'F#': 369.99, 'G': 392.00, 'G#': 415.30, 'A': 440.00, 'A#': 466.16, 'B': 493.88}

def get_camelot_pro(key, mode):
    # Correction spécifique basée sur tes instructions : F# MINOR = 11A
    if key == 'F#' and mode in ['minor', 'dorian']: return "11A"
    if key == 'B' and mode in ['minor', 'dorian']: return "10A"
    number = BASE_CAMELOT.get(key, "1")
    letter = "A" if mode in ['minor', 'dorian'] else "B"
    return f"{number}{letter}"

# --- MOTEUR D'ANALYSE HAUTE PRÉCISION ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
PROFILES = {
    "major": [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    "minor": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    "dorian": [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 2.69, 3.98, 3.34, 3.17]
}

def analyze_ultra_precision(y, sr):
    # AJOUT DU FILTRAGE PERCUSSIF : On isole la composante harmonique
    # Le paramètre margin=3.0 permet une séparation plus stricte
    y_harmonic, y_percussive = librosa.effects.hpss(y, margin=(3.0, 1.0))
    
    # Utilisation de la composante harmonique pour le calcul du chroma
    chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, bins_per_octave=24)
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
st.markdown("<h1>RICARDO_DJ228 PRECISION ANALYZER V3</h1>", unsafe_allow_html=True)
files = st.file_uploader("Déposez vos morceaux (Analyse haute précision)", type=['mp3', 'wav', 'flac'], accept_multiple_files=True)

if files:
    for file in files:
        with st.expander(f"🎼 Étude harmonique : {file.name}", expanded=True):
            with st.spinner("Analyse spectrale avec filtrage percussif..."):
                y_full, sr = librosa.load(file)
                duration = librosa.get_duration(y=y_full, sr=sr)
                tempo, _ = librosa.beat.beat_track(y=y_full, sr=sr)
                
                votes = []
                timeline_data = []
                
                for start_t in range(0, int(duration) - 15, 10):
                    start_s = int(start_t * sr)
                    end_s = int((start_t + 15) * sr)
                    key, mode, score = analyze_ultra_precision(y_full[start_s:end_s], sr)
                    if score > 0.5:
                        votes.append(f"{key} {mode}")
                        timeline_data.append({"Temps": start_t, "Note": key, "Mode": mode, "Confiance": score})

                if votes:
                    final_decision = Counter(votes).most_common(1)[0][0]
                    f_key, f_mode = final_decision.split(" ")
                    f_camelot = get_camelot_pro(f_key, f_mode)
                    
                    st.session_state.history.insert(0, {
                        "Heure": datetime.now().strftime("%H:%M"),
                        "Nom": file.name,
                        "Cle": f"{f_key} {f_mode.upper()}",
                        "Camelot": f_camelot,
                        "BPM": int(float(tempo))
                    })

                    # Affichage Métriques
                    c1, c2, c3 = st.columns(3)
                    c1.metric("CLÉ STABLE", f"{f_key} {f_mode.upper()}")
                    c2.metric("NOTATION CAMELOT", f_camelot)
                    c3.metric("TEMPO BPM", f"{int(float(tempo))}")

                    # --- VÉRIFICATION AUDITIVE ---
                    st.markdown("### 🔊 Vérification à l'oreille")
                    v1, v2 = st.columns(2)
                    with v1:
                        st.write("Fichier original :")
                        st.audio(file)
                    with v2:
                        st.write(f"Ton de référence ({f_key}) :")
                        target_freq = FREQS.get(f_key, 440.0)
                        t = np.linspace(0, 3.0, int(22050 * 3.0), False)
                        tone = 0.4 * np.sin(2 * np.pi * target_freq * t) + 0.2 * np.sin(2 * np.pi * (target_freq * 2) * t)
                        st.audio(tone, sample_rate=22050)

                    # Graphique
                    df_plot = pd.DataFrame(timeline_data)
                    fig = px.scatter(df_plot, x="Temps", y="Note", color="Mode", size="Confiance",
                                     title=f"Nuage de stabilité harmonique - {file.name}",
                                     color_discrete_sequence=["#D4AF37", "#4A90E2"],
                                     category_orders={"Note": NOTES})
                    fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.warning("Analyse instable : signal trop complexe.")

# --- HISTORIQUE ---
st.divider()
if st.session_state.history:
    col_h1, col_h2 = st.columns([1, 4])
    with col_h1:
        csv = pd.DataFrame(st.session_state.history).to_csv(index=False).encode('utf-8')
        st.download_button("📥 EXPORTER CSV", csv, "export_dj_set.csv", "text/csv")
    with col_h2:
        if st.button("🗑️ VIDER L'HISTORIQUE"):
            st.session_state.history = []
            st.rerun()

    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-card">
            <b>{item['Heure']}</b> | {item['Nom']} | <span style="color:#D4AF37">{item['Cle']} ({item['Camelot']})</span> | {item['BPM']} BPM
        </div>
        """, unsafe_allow_html=True)
