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


X_train, X_test, y_train, y_test, features = pripremi_podatke(skaliraj=False)

X_sve = pd.concat([X_train, X_test])
y_sve = pd.concat([y_train, y_test])

# =============================================================
# 2. TRENIRANJE FINALNOG MODELA
# =============================================================


finalni_model = RandomForestClassifier(
    max_depth=5, min_samples_split=2, n_estimators=100, random_state=42
)
finalni_model.fit(X_sve, y_sve)

print(f"Model istreniran na {len(X_sve)} putnika (ceo dataset).")

# =============================================================
# 3. EXPORT MODELA I METAPODATAKA
# =============================================================


joblib.dump(finalni_model, "app/model.joblib")

with open("app/feature_columns.json", "w") as f:
    json.dump(features, f, indent=4)

print("Model sacuvan: app/model.joblib")
print("Kolone sacuvane: app/feature_columns.json")