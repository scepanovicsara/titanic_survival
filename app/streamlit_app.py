# =============================================================
# streamlit_app.py
# Jednostavan UI za predikciju prezivljavanja putnika Titanika
# =============================================================

import streamlit as st
import joblib
import os
import sys

sys.path.append(os.path.dirname(__file__))
from preprocessing import pripremi_unos

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")

st.title("🚢 Titanic Survival Predictor")
st.write(
    "Unesite podatke o putniku da biste predvideli da li bi "
    "preziveo brodolom Titanika. Model: Random Forest "
    "(tacnost 82.68% na test skupu)."
)

# Ucitavanje istreniranog modela
model = joblib.load(os.path.join(os.path.dirname(__file__), "model.joblib"))

st.divider()

col1, col2 = st.columns(2)

with col1:
    ime = st.text_input(
        "Ime i prezime (opciono)",
        placeholder="npr. Smith, Mr. John"
    )
    pclass = st.selectbox("Klasa karte", [1, 2, 3], index=2)
    sex = st.selectbox("Pol", ["male", "female"])
    age = st.slider("Starost", 0, 90, 28)

with col2:
    sibsp = st.number_input("Broj brace/supruznika na brodu", 0, 10, 0)
    parch = st.number_input("Broj roditelja/dece na brodu", 0, 10, 0)
    fare = st.number_input("Cena karte (GBP)", 0.0, 600.0, 32.0)
    embarked = st.selectbox("Luka ukrcavanja", ["S", "C", "Q"])
    ima_kabinu = st.checkbox("Putnik ima dodeljenu kabinu")

st.divider()

if st.button("Predvidi prezivljavanje", type="primary"):
    X_novi = pripremi_unos(
        pclass, sex, age, sibsp, parch, fare, embarked, ime, ima_kabinu
    )

    predikcija = model.predict(X_novi)[0]
    verovatnoca = model.predict_proba(X_novi)[0]

    if predikcija == 1:
        st.success(
            f"✅ Putnik bi VEROVATNO PREZIVEO "
            f"(verovatnoca preživljavanja: {verovatnoca[1]*100:.1f}%)"
        )
    else:
        st.error(
            f"❌ Putnik VEROVATNO NE BI PREZIVEO "
            f"(verovatnoca preživljavanja: {verovatnoca[1]*100:.1f}%)"
        )

    st.caption(
        "Napomena: predikcija je zasnovana na istorijskim podacima "
        "i koristi se iskljucivo u edukativne svrhe."
    )