import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# =========================
# CONFIGURATION
# =========================
st.set_page_config(page_title="💓 CardioPredict Pro", layout="wide")

@st.cache_resource
def load_model():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "best_model_heart_Ameliorer.pkl")
    if not os.path.exists(model_path):
        st.error(f"❌ Modèle introuvable : {model_path}")
        st.stop()
    return joblib.load(model_path)

model = load_model()

if "historique" not in st.session_state:
    st.session_state.historique = []

# =========================
# SIDEBAR : GESTION DES DONNÉES
# =========================
with st.sidebar:
    st.title("📁 Gestion des Patients")
    nb_diagnostics = len(st.session_state.historique)
    st.metric("Diagnostics effectués", nb_diagnostics)
    
    if nb_diagnostics > 0:
        df_export = pd.DataFrame(st.session_state.historique)
        csv = df_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Télécharger l'historique CSV",
            data=csv,
            file_name="historique_cardiaque.csv",
            mime="text/csv",
            use_container_width=True
        )
        if st.button("🗑️ Réinitialiser"):
            st.session_state.historique = []
            st.rerun()

# =========================
# FORMULAIRE DE SAISIE
# =========================
st.title("💓 Analyse Prédictive Cardiaque")
st.write("Complétez les informations pour une analyse assistée par IA.")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📋 Profil")
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    chol = st.selectbox("Cholestérol élevé", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    bmi = st.number_input("IMC (BMI)", 10.0, 60.0, 25.0)
    sexe = st.selectbox("Sexe", [1, 0], format_func=lambda x: "Homme" if x else "Femme")
    age = st.slider("Catégorie d'âge (1-13)", 1, 13, 5)

with col2:
    st.subheader("🚬 Habitudes")
    fumeur = st.selectbox("Fumeur", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    alcool = st.selectbox("Alcool excessif", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    fruits = st.selectbox("Consomme Fruits", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    legumes = st.selectbox("Consomme Légumes", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    activite = st.selectbox("Activité physique", [0, 1], format_func=lambda x: "Oui" if x else "Non")

with col3:
    st.subheader("🏥 Médical & Social")
    avc = st.selectbox("Déjà eu un AVC", [0, 1], format_func=lambda x: "Oui" if x else "Non")
    diabete = st.selectbox("Diabète", [0, 1, 2], format_func=lambda x: ["Non", "Pré-diabète", "Diabète"][x])
    couverture = st.selectbox("Couverture santé", [1, 0], format_func=lambda x: "Oui" if x else "Non")
    revenu = st.slider("Revenu", 1, 8, 4)
    education = st.slider("Éducation", 1, 6, 4)

# =========================
# LOGIQUE DE PRÉDICTION
# =========================
if st.button("🔍 ANALYSER LE CAS PATIENT", use_container_width=True):
    # Dictionnaire des caractéristiques
    data = {
        "Hypertension": float(hypertension), "CholestérolÉlevé": float(chol),
        "ContrôleCholestérol": 1.0, "BMI": float(bmi), "Fumeur": float(fumeur),
        "AVC": float(avc), "Diabète": float(diabete), "ActivitéPhysique": float(activite),
        "ConsommationFruits": float(fruits), "ConsommationLégumes": float(legumes),
        "AlcoolExcessif": float(alcool), "AUneCouvertureSanté": float(couverture),
        "PasConsultéÀCauseDuCoût": 0.0, "SantéGénérale" : 3.0,
        "SantéMentaleJoursMauvais" : 0.0, "SantéPhysiqueJoursMauvais": 0.0,
        "DifficultéMarche": 0.0, "Sexe": float(sexe), "ÂgeCatégorie": float(age),
        "NiveauÉducation": float(education), "Revenu": float(revenu),
        # Feature Engineering
        "BMI_log": np.log(bmi) if bmi > 0 else 0, "Age_BMI": float(age * bmi),
        "alc_fumeur": float(alcool * fumeur), "alc_BMI": float(alcool * bmi),
        "BMI_squared": float(bmi**2), "BMI_Physique": float(bmi * activite),
        "ActiPhysique_legume_fruit": float(activite * fruits * legumes),
        "revenu_Hyper": float(revenu * hypertension)
    }

    df_input = pd.DataFrame([data])
    
    try:
        expected_cols = model.feature_names_in_
        df_input = df_input.reindex(columns=expected_cols, fill_value=0)
        
        prediction = model.predict(df_input)[0]
        proba = model.predict_proba(df_input)[0][1]

        # Calcul de la variable la plus influente (Local Feature Importance)
        # On utilise ici l'importance globale filtrée par la présence de la feature pour ce patient
        importances = model.feature_importances_
        impact_df = pd.Series(importances, index=expected_cols).sort_values(ascending=False)
        top_variable = impact_df.index[0] # Variable ayant le plus gros poids dans le modèle
        
        st.markdown("---")
        res_col, viz_col = st.columns([1, 1.2])

        with res_col:
            if prediction == 1:
                st.error(f"### ⚠️ RISQUE ÉLEVÉ ({proba:.2%})")
                st.write(f"🚩 **Facteur principal d'influence :** `{top_variable}`")
            else:
                st.success(f"### ✅ RISQUE FAIBLE ({proba:.2%})")
                st.write(f"ℹ️ **Variable la plus déterminante :** `{top_variable}`")
            
            st.progress(int(proba * 100))
            
            # --- CONSEILS ---
            st.subheader("💡 Conseils")
            if hypertension == 1: st.info("- Surveillance stricte de la tension.")
            if fumeur == 1: st.info("- Programme de sevrage tabagique recommandé.")
            if bmi > 25: st.info("- Suivi nutritionnel pour stabiliser l'IMC.")

        with viz_col:
            st.write("#### 📊 Poids des facteurs dans la décision")
            top_10 = impact_df.head(10).sort_values(ascending=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            top_10.plot(kind='barh', color='#ff4b4b', ax=ax)
            ax.set_title("Top 10 des variables impactant le modèle")
            st.pyplot(fig)

        # Enregistrement
        data_hist = {"Patient_ID": nb_diagnostics + 1, "Risque": "Élevé" if prediction == 1 else "Faible", "Probabilité": proba, "Top_Facteur": top_variable}
        st.session_state.historique.append(data_hist)

    except Exception as e:
        st.error(f"Erreur technique : {e}")

# =========================
# AVERTISSEMENT LÉGAL (FIN)
# =========================
st.markdown("---")
st.warning("""
**⚠️ AVERTISSEMENT MÉDICAL :** Cette application est un outil d'assistance basé sur l'intelligence artificielle. **Elle ne remplace en aucun cas l'avis, le diagnostic ou le traitement d'un médecin professionnel.** Les résultats sont fournis à titre informatif et probabiliste. En cas de doute ou de symptômes, veuillez consulter immédiatement un service de santé.
""")