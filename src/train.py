# =============================================================
# train.py
# Treniranje i evaluacija svih modela
#
# Metodologija:
# 1. Za svaki model trazimo najbolje hiperparametre pomocu
#    GridSearchCV, koji interno koristi 5-fold cross-validaciju
#    (test skup se ovde NE koristi)
# 2. Najbolje (vec podesene) verzije svih modela poredimo
#    medjusobno na osnovu cross-validacionog F1-skora
# 3. Test skup se koristi SAMO JEDNOM, na kraju, za finalnu
#    procenu jednog izabranog (najboljeg) modela
# =============================================================

import sys
import os

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from data_preparation import pripremi_podatke
from evaluate import evaluiraj_model

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

RESULTS_DIR = "results/metrics"
FIGURES_DIR = "results/figures"
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# =============================================================
# 1. PRIPREMA PODATAKA
# =============================================================
# Test skup se odvaja odmah i ne dira se sve do samog kraja

print("\nPriprema podataka bez skaliranja...")
X_train, X_test, y_train, y_test, features = pripremi_podatke(skaliraj=False)

print("Priprema podataka sa skaliranjem...")
X_train_s, X_test_s, y_train_s, y_test_s, _ = pripremi_podatke(skaliraj=True)

# =============================================================
# 2. MODELI I PROSTOR HIPERPARAMETARA
# =============================================================
# Svaki model ima definisan prostor parametara koje
# GridSearchCV isprobava. GridSearchCV interno koristi
# 5-fold cross-validaciju da oceni svaku kombinaciju.

konfiguracije = [
    {
        "naziv": "Random Forest",
        "model": RandomForestClassifier(random_state=42),
        "parametri": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        },
        "skalirano": False
    },
    {
        "naziv": "Decision Tree",
        "model": DecisionTreeClassifier(random_state=42),
        "parametri": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "skalirano": False
    },
    {
        "naziv": "Gradient Boosting",
        "model": GradientBoostingClassifier(random_state=42),
        "parametri": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        },
        "skalirano": False
    },
    {
        "naziv": "KNN",
        "model": KNeighborsClassifier(),
        "parametri": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "metric": ["euclidean", "manhattan"]
        },
        "skalirano": True
    },
    {
        "naziv": "SVM",
        "model": SVC(random_state=42),
        "parametri": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"]
        },
        "skalirano": True
    },
    {
        "naziv": "Logistic Regression",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "parametri": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        },
        "skalirano": True
    },
    {
        "naziv": "Naive Bayes",
        "model": GaussianNB(),
        "parametri": {
            "var_smoothing": [1e-9, 1e-8, 1e-7, 1e-6]
        },
        "skalirano": False
    },
]

# =============================================================
# 3. GRID SEARCH + CROSS-VALIDACIJA (samo na trening skupu)
# =============================================================

print("\n" + "="*60)
print("GRID SEARCH + CROSS-VALIDACIJA (5-fold) ZA SVAKI MODEL")
print("="*60)
print("Test skup NIJE koriscen u ovom koraku.\n")

rezultati = []
istrenirani_modeli = {}

for config in konfiguracije:
    print(f"Trazim najbolje parametre: {config['naziv']}...")

    X_tr = X_train_s if config["skalirano"] else X_train
    y_tr = y_train_s if config["skalirano"] else y_train

    gs = GridSearchCV(
        estimator=config["model"],
        param_grid=config["parametri"],
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    gs.fit(X_tr, y_tr)

    print(f"   Najbolji parametri: {gs.best_params_}")
    print(f"   Najbolji CV F1: {gs.best_score_:.4f}")

    rezultati.append({
        "naziv": config["naziv"],
        "najbolji_parametri": gs.best_params_,
        "cv_f1": round(gs.best_score_, 4)
    })

    istrenirani_modeli[config["naziv"]] = {
        "model": gs.best_estimator_,
        "skalirano": config["skalirano"]
    }

# =============================================================
# 4. TABELA POREĐENJA (na osnovu cross-validacije)
# =============================================================

rezultati_sortirani = sorted(rezultati, key=lambda x: x["cv_f1"], reverse=True)

print("\n" + "="*65)
print(f"{'POREĐENJE MODELA (Grid Search + 5-fold CV)':^65}")
print("="*65)
print(f"{'Model':<22}{'CV F1-skor':>15}{'Najbolji parametri'}")
print("-"*65)
for r in rezultati_sortirani:
    print(f"{r['naziv']:<22}{r['cv_f1']:>15.4f}   {r['najbolji_parametri']}")
print("="*65)

with open(f"{RESULTS_DIR}/grid_search_rezultati.json", "w") as f:
    json.dump(rezultati_sortirani, f, indent=4, default=str)

df_cv = pd.DataFrame(rezultati_sortirani)
plt.figure(figsize=(11, 6))
sns.barplot(data=df_cv, x="naziv", y="cv_f1", hue="naziv",
            palette="viridis", legend=False)
plt.title("Poređenje modela (Grid Search + cross-validacija, F1-skor)")
plt.xlabel("Model")
plt.ylabel("Najbolji prosečan F1-skor (5-fold CV)")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{FIGURES_DIR}/cross_validation_poredjenje.png")
plt.close()

# =============================================================
# 4.5 DIJAGNOSTICKI GRAFICI ZA SVE MODELE
# =============================================================
# Koristimo cross_val_predict sa vec podesenim (najboljim)
# hiperparametrima da generisemo matricu konfuzije za svaki
# model bez koriscenja test skupa.

print("\n" + "="*60)
print("GENERISANJE DIJAGNOSTICKIH GRAFIKA ZA SVE MODELE")
print("="*60)

for naziv, info in istrenirani_modeli.items():
    print(f"Graficka analiza: {naziv}...")
    naziv_fajla = naziv.lower().replace(" ", "_")
    model = info["model"]
    skalirano = info["skalirano"]

    X_cv = X_train_s if skalirano else X_train
    y_cv = y_train_s if skalirano else y_train

    # --- Matrica konfuzije (na osnovu CV predikcija) ---
    y_pred_cv = cross_val_predict(model, X_cv, y_cv, cv=5)
    cm = confusion_matrix(y_cv, y_pred_cv)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Nije preziveo", "Preziveo"]
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title(f"Matrica konfuzije (CV) - {naziv}")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/matrica_konfuzije_{naziv_fajla}.png")
    plt.close()

    # --- Feature importance (samo za modele koji ga podrzavaju) ---
    if hasattr(model, "feature_importances_"):
        fi_df = pd.DataFrame({
            "Feature": features,
            "Importance": model.feature_importances_
        }).sort_values("Importance", ascending=False)

        plt.figure(figsize=(8, 6))
        sns.barplot(data=fi_df, x="Importance", y="Feature",
                    hue="Feature", palette="viridis", legend=False)
        plt.title(f"Vaznost atributa - {naziv}")
        plt.tight_layout()
        plt.savefig(f"{FIGURES_DIR}/feature_importance_{naziv_fajla}.png")
        plt.close()

print("Svi dijagnosticki grafici sacuvani!")

# =============================================================
# 5. IZBOR NAJBOLJEG MODELA
# =============================================================

najbolji_naziv = rezultati_sortirani[0]["naziv"]
print(f"\nNajbolji model (Grid Search + CV): {najbolji_naziv}")
print(f"Najbolji parametri: {rezultati_sortirani[0]['najbolji_parametri']}")

najbolji_model = istrenirani_modeli[najbolji_naziv]["model"]
skalirano = istrenirani_modeli[najbolji_naziv]["skalirano"]

# =============================================================
# 6. FINALNA PROCENA - test skup se koristi SAMO OVDE, JEDNOM
# =============================================================
# GridSearchCV automatski trenira best_estimator_ na celom
# trening skupu (refit=True je podrazumevano), pa model
# ovde samo evaluiramo na test skupu

print(f"\nFinalna procena modela ({najbolji_naziv}) na test skupu...")

if skalirano:
    finalni_rezultat = evaluiraj_model(
        f"{najbolji_naziv} (FINALNI - test skup)",
        najbolji_model, X_test_s, y_test_s, features
    )
else:
    finalni_rezultat = evaluiraj_model(
        f"{najbolji_naziv} (FINALNI - test skup)",
        najbolji_model, X_test, y_test, features
    )

print("\n" + "="*60)
print("ZAVRSNA NAPOMENA")
print("="*60)
print(f"Model {najbolji_naziv} je izabran na osnovu Grid Search-a")
print(f"i cross-validacije (test skup nije koriscen za izbor).")
print(f"Test skup je koriscen SAMO JEDNOM, za finalnu procenu")
print(f"vec izabranog i vec podesenog modela.")