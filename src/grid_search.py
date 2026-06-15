# =============================================================
# grid_search.py
# Optimizacija hiperparametara pomoću Grid Search-a
# =============================================================

import sys
import os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_preparation import pripremi_podatke
from evaluate import evaluiraj_model, uporedi_modele

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# =============================================================
# 1. PRIPREMA PODATAKA
# =============================================================

X_train, X_test, y_train, y_test, features = pripremi_podatke(skaliraj=False)
X_train_s, X_test_s, y_train_s, y_test_s, _ = pripremi_podatke(skaliraj=True)

# =============================================================
# 2. DEFINISANJE MODELA I PROSTORA PRETRAGE
# =============================================================

# Svaki model ima svoj prostor hiperparametara koji pretražujemo
# Grid Search isprobava SVE kombinacije i bira najbolju

konfiguracije = [
    {
        "naziv": "Random Forest (GS)",
        "model": RandomForestClassifier(random_state=42),
        "parametri": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 5, 10],
            "min_samples_split": [2, 5]
        },
        "skalirano": False
    },
    {
        "naziv": "Decision Tree (GS)",
        "model": DecisionTreeClassifier(random_state=42),
        "parametri": {
            "max_depth": [3, 5, 7, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4]
        },
        "skalirano": False
    },
    {
        "naziv": "Gradient Boosting (GS)",
        "model": GradientBoostingClassifier(random_state=42),
        "parametri": {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 5]
        },
        "skalirano": False
    },
    {
        "naziv": "KNN (GS)",
        "model": KNeighborsClassifier(),
        "parametri": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "metric": ["euclidean", "manhattan"]
        },
        "skalirano": True
    },
    {
        "naziv": "SVM (GS)",
        "model": SVC(random_state=42),
        "parametri": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf", "poly"]
        },
        "skalirano": True
    },
    {
        "naziv": "Logistic Regression (GS)",
        "model": LogisticRegression(random_state=42, max_iter=1000),
        "parametri": {
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "liblinear"]
        },
        "skalirano": True
    },
]

# =============================================================
# 3. GRID SEARCH ZA SVAKI MODEL
# =============================================================

print("\n" + "="*50)
print("GRID SEARCH - OPTIMIZACIJA HIPERPARAMETARA")
print("="*50)
print("Ovo moze potrajati nekoliko minuta...\n")

for config in konfiguracije:
    print(f"Pretrazujem parametre za: {config['naziv']}...")

    # Izaberi skalirane ili neskalirane podatke
    if config["skalirano"]:
        X_tr, X_te, y_tr, y_te = X_train_s, X_test_s, y_train_s, y_test_s
    else:
        X_tr, X_te, y_tr, y_te = X_train, X_test, y_train, y_test

    # Grid Search sa 5-fold cross validacijom
    gs = GridSearchCV(
        estimator=config["model"],
        param_grid=config["parametri"],
        cv=5,
        scoring="f1",
        n_jobs=-1  # koristi sve dostupne procesore
    )
    gs.fit(X_tr, y_tr)

    print(f"Najbolji parametri: {gs.best_params_}")
    print(f"Najbolji F1 (cross-val): {gs.best_score_:.4f}")

    # Evaluiraj najbolji model na test skupu
    evaluiraj_model(
        config["naziv"],
        gs.best_estimator_,
        X_te, y_te,
        features
    )

# =============================================================
# 4. UPOREDNO POREĐENJE
# =============================================================

print("\n")
uporedi_modele()

print("\nGrid Search završen!")