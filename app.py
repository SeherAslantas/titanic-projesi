import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

model = load_model("titanic_model.keras")
scaler = joblib.load("titanic_scaler.pkl")
sutunlar = joblib.load("titanic_columns.pkl")

st.title("Titanic Hayatta Kalma Tahmini")

pclass = st.selectbox("Yolcu sınıfı", [1, 2, 3])
sex = st.selectbox("Cinsiyet", ["male", "female"])
age = st.number_input("Yaş", 1, 100, 25)
sibsp = st.number_input("Kardeş/eş sayısı", 0, 10, 0)
parch = st.number_input("Ebeveyn/çocuk sayısı", 0, 10, 0)
fare = st.number_input("Bilet ücreti", min_value=0.0, value=30.0)
embarked = st.selectbox("Biniş limanı", ["S", "C", "Q"])

if st.button("Tahmin Et"):
    veri = pd.DataFrame([{
        "Pclass": pclass,
        "Sex": sex,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Embarked": embarked
    }])

    veri = pd.get_dummies(veri)
    veri = veri.reindex(columns=sutunlar, fill_value=0)

    veri = scaler.transform(veri)
    tahmin = model.predict(veri, verbose=0)[0][0]
    st.write(f"Hayatta kalma olasılığı: %{tahmin*100:.1f}")

    if tahmin >= 0.5:
        st.success("Modelin tahmini: Hayatta kaldı.")
    else:
        st.error("Modelin tahmini: Hayatta kalamadı.")
