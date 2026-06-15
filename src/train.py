# =============================================================
# train.py
# Treniranje i evaluacija svih modela
# =============================================================

import sys
import os

# Dodajemo putanje da Python moze da nadje nase module
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from data_preparation import pripremi_podatke
from evaluate import evaluiraj_model, uporedi_modele

# Uvoz svih modela
from models.random_forest import kreiraj_model as rf
from models.decision_tree import kreiraj_model as dt
from models.knn import kreiraj_model as knn
from models.svm import kreiraj_model as svm
from models.gradient_boosting import kreiraj_model as gb
from models.logistic_regression import kreiraj_model as lr
from models.naive_bayes import kreiraj_model as nb

# =============================================================
# 1. PRIPREMA PODATAKA
# =============================================================

# Podaci bez skaliranja - za modele zasnovana na stablima
print("\nPriprema podataka bez skaliranja...")
X_train, X_test, y_train, y_test, features = pripremi_podatke(
    skaliraj=False
)

# Podaci sa skaliranjem - za modele zasnovane na rastojanju
print("\nPriprema podataka sa skaliranjem...")
X_train_s, X_test_s, y_train_s, y_test_s, _ = pripremi_podatke(
    skaliraj=True
)

# =============================================================
# 2. DEFINISANJE MODELA
# =============================================================

# (naziv, model, skalirani podaci?)
modeli = [
    ("Random Forest",        rf(),  False),
    ("Decision Tree",        dt(),  False),
    ("Gradient Boosting",    gb(),  False),
    ("KNN",                  knn(), True),
    ("SVM",                  svm(), True),
    ("Logistic Regression",  lr(),  True),
    ("Naive Bayes",          nb(),  False),
]

# =============================================================
# 3. TRENIRANJE I EVALUACIJA
# =============================================================

print("\n" + "="*50)
print("TRENIRANJE I EVALUACIJA SVIH MODELA")
print("="*50)

for naziv, model, skalirano in modeli:
    print(f"\nTreniranje: {naziv}...")
    
    # Izaberi skalirane ili neskalirane podatke
    if skalirano:
        model.fit(X_train_s, y_train_s)
        evaluiraj_model(naziv, model, X_test_s, y_test_s, features)
    else:
        model.fit(X_train, y_train)
        evaluiraj_model(naziv, model, X_test, y_test, features)

# =============================================================
# 4. UPOREDNO POREĐENJE
# =============================================================

print("\n")
uporedi_modele()

print("\nSvi modeli su istrenirani i evaluirani!")
print("Rezultati su sacuvani u folderu 'results/'")