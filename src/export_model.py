# =============================================================
# export_model.py
# Treniranje finalnog modela na celom datasetu i export
# za upotrebu u aplikaciji (deployment)
# =============================================================

import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from data_preparation import pripremi_podatke

os.makedirs("app", exist_ok=True)

# =============================================================
# 1. PRIPREMA CELOG DATASETA (bez podele na train/test)
# =============================================================
# Test skup je vec odigrao svoju ulogu - dao nam je realnu
# procenu performansi modela (tacnost 82.68%, vidi rezultate
# u train.py). Za model koji se realno koristi (deployment),
# treniramo na SVIM dostupnim podacima kako bi model video
# sto vise primera i bio sto precizniji u praksi.

X_train, X_test, y_train, y_test, features = pripremi_podatke(skaliraj=False)

X_sve = pd.concat([X_train, X_test])
y_sve = pd.concat([y_train, y_test])

# =============================================================
# 2. TRENIRANJE FINALNOG MODELA
# =============================================================
# Koristimo najbolje hiperparametre pronadjene Grid Search-om
# u train.py (max_depth=5, n_estimators=100, min_samples_split=2)

finalni_model = RandomForestClassifier(
    max_depth=5, min_samples_split=2, n_estimators=100, random_state=42
)
finalni_model.fit(X_sve, y_sve)

print(f"Model istreniran na {len(X_sve)} putnika (ceo dataset).")

# =============================================================
# 3. EXPORT MODELA I METAPODATAKA
# =============================================================
# Cuvamo model (joblib format - standardni nacin za cuvanje
# sklearn modela) i listu kolona (potrebno da bismo znali
# tacan redosled i nazive atributa pri predikciji novih podataka)

joblib.dump(finalni_model, "app/model.joblib")

with open("app/feature_columns.json", "w") as f:
    json.dump(features, f, indent=4)

print("Model sacuvan: app/model.joblib")
print("Kolone sacuvane: app/feature_columns.json")