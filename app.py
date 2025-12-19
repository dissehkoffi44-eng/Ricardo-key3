import streamlit as st
import librosa
import numpy as np
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(page_title="RICARDO_DJ228 | Global Analyzer", page_icon="🎧", layout="wide")

# Initialisation de l'historique
if 'history' not in st.session_state:
    st.session_state.history = []

# --- CSS HAUTE LISIBILITÉ ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;700;900&display=swap');
    .stApp { background-color: #0F0F0F; font-family: 'Inter', sans-serif; }
    h1 { color: #FFFFFF; font-weight: 900; text-align: center; font-size: 3.5rem !important; }
    .sub-text { color: #BB86FC; text-align: center; font-size: 1.3rem; margin-bottom: 2rem; }

    /* ZONE UPLOAD */
    .stFileUploader section { background-color: #1A1A1A !important; border: 2px dashed #BB86FC !important; border-radius: 20px !important; padding: 3rem !important; }
    .stFileUploader section [data-testid="stMarkdownContainer"] p { color: #FFFFFF !important; font-size: 1.5rem !important; font-weight: 700 !important; }
    .stFileUploader button { background-color: #FFFFFF !important; color: #000000 !important; font-weight: 900 !important; font-size: 1.1rem !important; border-radius: 10px !important; }

    /* CARTES RÉSULTATS */
    div[data-testid="stMetric"] { background-color: #1A1A1A; border: 1px solid #333333; border-radius: 16px; padding: 25px !important; }
    div[data-testid="stMetricLabel"] { color: #BB86FC !important; font-size: 1.1rem !important; font-weight: 600; }
    div[data-testid="stMetricValue"] { color: #FFFFFF !important; font-size: 2.5rem !important; font-weight: 800; }

    /* TABLEAU */
    .stTable { background-color: #1A1A1A; border-radius: 12px; }
    thead tr th { background-color: #252525 !important; color: #BB86FC !important; }
    </style>
    """, unsafe_allow_html=True)

# --- LOGIQUE MUSICALE ---
NOTES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

def get_camelot(key, mode):
    camelot_map = {
        'G# minor': '1A', 'Ab minor': '1A', 'B major': '1B', 'Cb major': '1B',
        'D# minor': '2A', 'Eb minor': '2A', 'F# major': '2B', 'Gb major': '2B',
        'Bb minor': '3A', 'A# minor': '3A', 'Db major': '3B', 'C# major': '3B',
        'F minor': '4A', 'Ab major': '4B', 'C minor': '5A', 'Eb major': '5B',
        'G minor': '6A', 'Bb major': '6B', 'D minor': '7A', 'F major': '7B',
        'A minor': '8A', 'C major': '8B', 'E minor': '9A', 'G major': '9B',
        'B minor': '10A', 'D major': '10B', 'F# minor': '11A', 'Gb minor': '11A', 'A major': '11B',
        'C# minor': '12A', 'Db minor': '12A', 'E major': '12B'
    }
    return camelot_map.get(f"{key} {mode}", "1A")

def analyze_global(y, sr):
    # Séparation harmonique (pour isoler les mélodies)
    y_harm = librosa.effects.hpss(y)[0]
    # Calcul du chromagramme global sur tout le fichier
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr)
    chroma_avg = np.mean(chroma, axis=1)
    
    best_score = -1
    res_key, res_mode = "", ""
    for i in range(12):
        for mode, profile in [("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)]:
            score = np.corrcoef(chroma_avg, np.roll(profile, i))[0, 1]
            if score > best_score:
                best_score, res_key, res_mode = score, NOTES[i], mode
    return res_key, res_mode

# --- INTERFACE ---
st.markdown("<h1>RICARDO_DJ228</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-text'>Analyseur de Tonalité Globale Dominante</p>", unsafe_allow_html=True)

file = st.file_uploader(" ", type=['mp3', 'wav', 'flac'])

if file:
    with st.spinner("🚀 Analyse globale du signal..."):
        y, sr = librosa.load(file)
        
        # Analyse unique sur l'ensemble du fichier
        final_key, final_mode = analyze_global(y, sr)
        
        # Calcul Tempo & Camelot
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        bpm = int(round(float(tempo)))
        camelot = get_camelot(final_key, final_mode)

        # Historique
        entry = {"Fichier": file.name, "Key": f"{final_key} {final_mode.upper()}", "Camelot": camelot, "BPM": bpm}
        if not st.session_state.history or st.session_state.history[0]['Fichier'] != file.name:
            st.session_state.history.insert(0, entry)

        # Résultats
        st.markdown("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tonalité Dominante", entry["Key"])
        c2.metric("Code Camelot", camelot)
        c3.metric("Tempo", f"{bpm} BPM")
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.audio(file)

# --- HISTORIQUE ---
st.markdown("<br><hr style='border: 1px solid #333333;'><br>", unsafe_allow_html=True)
st.markdown("### 🕒 Historique des Analyses")

if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.table(df_history)
    if st.button("🗑️ Effacer l'historique"):
        st.session_state.history = []
        st.rerun()
