import streamlit as st
import pandas as pd

# Chargement en local en csv
df_vacances_zoneb_confinement = pd.read_csv("dates_vacances_zoneB_confinement.csv")
df_clean = pd.read_csv("df_clean.csv")
df_meteo = pd.read_csv("df_meteo.csv")
df_meteo_cvl = pd.read_csv("df_meteo_cvl.csv")
df_meteo_hf = pd.read_csv("df_meteo_hf.csv")
df_pro = pd.read_csv("df_pro.csv")
df_pro_res = pd.read_csv("df_pro_res.csv")
df_res = pd.read_csv("df_res.csv")
jours_feries_metropole = pd.read_csv("jours_feries_metropole.csv")
#jours-ouvres-week-end-feries-france-2010-a-2030 = pd.read_csv("jours-ouvres-week-end-feries-france-2010-a-2030.csv")
vacances_confinement = pd.read_csv("vacances_confinement.csv")
df_final_20_22 = pd.read_csv("df_final_20_22.csv")


# Configuration de la page
st.set_page_config(
    page_title="WattMétéo",
    layout="wide",
    page_icon="☀️")
#Mise en forme fond de page
page_bg_img = """
<style>
[data-testid = "stAppViewContainer"] {

background-color: #e5e5f7;
opacity: 0.8;
#background-image: radial-gradient(#444cf7 0.5px, #e5e5f7 0.5px);
background-size: 10px 10px;

}

</style>
"""
st.markdown(page_bg_img, unsafe_allow_html = True)



# Titre
st.title("Bienvenue dans notre application de prédiction de consommation éléctrique")


