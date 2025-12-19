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

# --- DESIGN & CSS (V4 STYLE) ---
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; color: #212529; }
    .metric-container { background: white; padding: 20px; border-radius: 15px; border: 1px solid #E0E0E0; text-align: center; height: 100%; transition: transform 0.3s; box-shadow: 0 4px 6px rgba(0,0,0,0.02); }
    .metric-container:hover { transform: translateY(-5px); border-color: #6366F1; }
    .label-custom { color: #666; font-size: 0.85em; font-weight: bold; margin-bottom: 8px; text-transform: uppercase; letter-spacing: 1px; }
    .value-custom { font-size: 1.8em; font-weight: 800; color: #1A1A1A; line-height: 1.2; }
    .camelot-custom { font-size: 1.6em; font-weight: 800; margin-top: 5px; }
    .reliability-bar-bg { background-color: #E0E0E0; border-radius: 10px; height: 18px; width: 100%; margin: 15px 0; overflow: hidden; }
    .reliability-fill { height: 100%; transition: width 0.8s ease-in-out; display: flex; align-items: center; justify-content: center; color: white; font-size: 0.8em; font-weight: bold; }
    .warning-box { background-color: #FFFBEB; border-left: 5px solid #F59E0B; padding: 15px; border-radius: 5px; margin: 15px 0; color: #92400E; }
    .success-box { background-color: #ECFDF5; border-left: 5px solid #10B981; padding: 15px; border-radius: 5px; margin: 15px 0; color: #065F46; }
    </style>
    """, unsafe_allow_html=True)

# --- MAPPING CAMELOT ---
BASE_CAMELOT_MINOR = {'Ab':'1A','G#':'1A','Eb':'2A','D#':'2A','Bb':'3A','A#':'3A','F':'4A','C':'5A','G':'6A','D':'7A','A':'8A','E':'9A','B':'10A','F#':'11A','Gb':'11A','Db':'12A','C#':'12A'}
BASE_CAMELOT_MAJOR = {'B':'1B','F#':'2B','Gb':'2B','Db':'3B','C#':'3B','Ab':'4B','G#':'4B','Eb':'5B','D#':'5B','Bb':'6B','A#':'6B','F':'7B','C':'8B','G':'9B','D':'10B','A':'11B','E':'12B'}

def get_camelot_pro(key_mode_str):
    try:
        parts = key_mode_str.split(" ")
        key, mode = parts[0], parts[1].lower()
        return BASE_CAMELOT_MINOR.get(key, "??") if mode in ['minor', 'dorian'] else BASE_CAMELOT_MAJOR.get(key, "??")
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
    chroma = librosa.feature.chroma_cqt(y=y_harm, sr=sr, tuning=tuning)
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

@st.cache_data(show_spinner="Analyse spectrale haute précision...")
def get_full_analysis(file_buffer):
    y, sr = librosa.load(file_buffer)
    duration = librosa.get_duration(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    energy = calculate_energy(y, sr)
    timeline_data, votes = [], []
    
    for start_t in range(0, int(duration) - 15, 10):
        seg, score = analyze_segment(y[int(start_t*sr):int((start_t+15)*sr)], sr)
        if score > 0.45:
            votes.append(seg)
            timeline_data.append({"Temps": start_t, "Note_Mode": seg, "Confiance": score})
    
    # Détection de stabilité
    dominante = Counter(votes).most_common(1)[0][0] if votes else "Inconnue"
    note_weights = {}
    for d in timeline_data:
        n = d["Note_Mode"]
        note_weights[n] = note_weights.get(n, 0) + d["Confiance"]
    tonique_synth = max(note_weights, key=note_weights.get) if note_weights else "Inconnue"
    
    # Score de fiabilité
    stability = (Counter(votes).most_common(1)[0][1] / len(votes)) if votes else 0
    confidence = int(np.clip((stability * 60) + (max([d['Confiance'] for d in timeline_data]) * 40), 40, 99))
    if dominante == tonique_synth: confidence = min(99, confidence + 10)

    return {
        "dominante": dominante, "synthese": tonique_synth, "confidence": confidence,
        "tempo": int(float(tempo)), "energy": energy, "timeline": timeline_data
    }

# --- INTERFACE ---
st.markdown("<h1 style='text-align: center;'>🎧 RICARDO_DJ228 | V3 ULTRA PRECISION</h1>", unsafe_allow_html=True)

file = st.file_uploader("Glissez-déposez votre track ici", type=['mp3', 'wav', 'flac'])

if file:
    res = get_full_analysis(file)
    conf = res["confidence"]
    color = "#10B981" if conf >= 90 else "#F59E0B" if conf > 70 else "#EF4444"
    
    # Barre de fiabilité
    st.markdown(f"**Indice de Fiabilité Harmonique : {conf}%**")
    st.markdown(f"""<div class="reliability-bar-bg"><div class="reliability-fill" style="width: {conf}%; background-color: {color};">{conf}%</div></div>""", unsafe_allow_html=True)
    
    # Alertes dynamiques
    if res["dominante"] != res["synthese"]:
        st.markdown(f"""<div class="warning-box">⚠️ <b>ANALYSE COMPLEXE :</b> Décalage détecté entre la dominante ({res['dominante']}) et la synthèse globale ({res['synthese']}). Le morceau pourrait changer de ton.</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="success-box">✅ <b>ANALYSE CERTIFIÉE :</b> Signal harmonique stable et cohérent sur toute la durée du fichier.</div>""", unsafe_allow_html=True)

    # --- GRILLE DE MÉTRIQUES ---
    st.markdown("---")
    m1, m2, m3, m4 = st.columns(4)
    
    with m1:
        st.markdown(f"""<div class="metric-container"><div class="label-custom">Camelot Code</div><div class="camelot-custom" style="color:{color};">{get_camelot_pro(res['synthese'])}</div><div style="font-size:0.9em; opacity:0.7;">{res['synthese']}</div></div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""<div class="metric-container"><div class="label-custom">Tempo</div><div class="value-custom">{res['tempo']}</div><div style="font-size:0.9em; opacity:0.7;">BPM</div></div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""<div class="metric-container"><div class="label-custom">Énergie</div><div class="value-custom">{res['energy']}/10</div><div style="font-size:0.9em; opacity:0.7;">Puissance</div></div>""", unsafe_allow_html=True)
    with m4:
        status = "SOLIDE" if conf >= 90 else "À MIXER PRUDEMMENT"
        st.markdown(f"""<div class="metric-container"><div class="label-custom">Statut</div><div class="value-custom" style="font-size:1.2em;">{status}</div></div>""", unsafe_allow_html=True)

    # --- GRAPHIQUE ---
    st.markdown("### 📊 Stabilité Harmonique")
    df = pd.DataFrame(res["timeline"])
    fig = px.scatter(df, x="Temps", y="Note_Mode", size="Confiance", color="Note_Mode", 
                     template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
    fig.update_layout(margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig, use_container_width=True)

    # --- ACTIONS & HISTORIQUE ---
    # Sauvegarde historique
    if not st.session_state.history or st.session_state.history[-1]["Fichier"] != file.name:
        st.session_state.history.append({
            "Heure": datetime.datetime.now().strftime("%H:%M"),
            "Fichier": file.name,
            "Key": res['synthese'],
            "Camelot": get_camelot_pro(res['synthese']),
            "BPM": res['tempo'],
            "Conf.": f"{conf}%"
        })

    c1, c2 = st.columns(2)
    report = f"RAPPORT RICARDO_DJ228\nTrack: {file.name}\nKey: {res['synthese']} ({get_camelot_pro(res['synthese'])})\nBPM: {res['tempo']}\nEnergy: {res['energy']}/10\nFiabilité: {conf}%"
    c1.download_button("📥 Télécharger le Rapport", report, file_name=f"Ricardo_Analysis_{file.name}.txt")
    
    if st.button("🗑️ Effacer l'Historique"):
        st.session_state.history = []
        st.rerun()

if st.session_state.history:
    with st.expander("🕒 Historique des analyses de la session"):
        st.table(pd.DataFrame(st.session_state.history))
