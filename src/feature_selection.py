# =============================================================
# feature_selection.py
# Odabir najznacajnijih atributa i poredjenje performansi
# modela sa svim atributima naspram samo najbitnijih
# =============================================================

import sys, os
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import cross_validate

from data_preparation import pripremi_podatke

os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/metrics", exist_ok=True)

# =============================================================
# 1. PRIPREMA PODATAKA
# =============================================================
X_train, X_test, y_train, y_test, features = pripremi_podatke(skaliraj=False)

# Najbolji model i hiperparametri pronadjeni u train.py (Grid Search)
najbolji_model = RandomForestClassifier(
    max_depth=5, min_samples_split=2, n_estimators=100, random_state=42
)

# =============================================================
# 2. METODA 1 - Feature importance (Random Forest)
# =============================================================
# Model sam tokom treniranja racuna koliko je svaki atribut
# doprineo smanjenju neuredjenosti (Gini) u stablima

najbolji_model.fit(X_train, y_train)
fi_df = pd.DataFrame({
    "Feature": features,
    "RF_Importance": najbolji_model.feature_importances_
}).sort_values("RF_Importance", ascending=False)

print("="*50)
print("METODA 1: RANDOM FOREST FEATURE IMPORTANCE")
print("="*50)
print(fi_df.to_string(index=False))

# =============================================================
# 3. METODA 2 - Mutual Information (statisticka metoda)
# =============================================================
# Mutual information meri koliko informacije atribut nosi
# o ciljnoj promenljivoj, nezavisno od bilo kog modela

mi = mutual_info_classif(X_train, y_train, random_state=42)
mi_df = pd.DataFrame({
    "Feature": features,
    "Mutual_Info": mi
}).sort_values("Mutual_Info", ascending=False)

print("\n" + "="*50)
print("METODA 2: MUTUAL INFORMATION")
print("="*50)
print(mi_df.to_string(index=False))

# Grafik poredjenja obe metode
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
sns.barplot(data=fi_df, x="RF_Importance", y="Feature", ax=axes[0],
            hue="Feature", palette="viridis", legend=False)
axes[0].set_title("Random Forest Feature Importance")
sns.barplot(data=mi_df, x="Mutual_Info", y="Feature", ax=axes[1],
            hue="Feature", palette="magma", legend=False)
axes[1].set_title("Mutual Information")
plt.tight_layout()
plt.savefig("results/figures/feature_selection_metode.png")
plt.close()

# =============================================================
# 4. ODABIR TOP N ATRIBUTA
# =============================================================
TOP_N = 6
top_features = fi_df.head(TOP_N)["Feature"].tolist()
print(f"\nTop {TOP_N} atributa (na osnovu RF importance): {top_features}")

# =============================================================
# 5. POREDJENJE: SVI ATRIBUTI vs TOP N ATRIBUTA
# =============================================================
# Koristimo isti model (vec podesene hiperparametre) i
# 5-fold cross-validaciju, samo menjamo skup atributa

metrike = ["accuracy", "precision", "recall", "f1"]

cv_svi = cross_validate(najbolji_model, X_train, y_train, cv=5, scoring=metrike)
cv_top = cross_validate(najbolji_model, X_train[top_features], y_train, cv=5, scoring=metrike)

rezultati = pd.DataFrame({
    "Skup atributa": [f"Svi atributi ({len(features)})", f"Top {TOP_N} atributa"],
    "Tacnost": [cv_svi["test_accuracy"].mean(), cv_top["test_accuracy"].mean()],
    "Preciznost": [cv_svi["test_precision"].mean(), cv_top["test_precision"].mean()],
    "Odziv": [cv_svi["test_recall"].mean(), cv_top["test_recall"].mean()],
    "F1": [cv_svi["test_f1"].mean(), cv_top["test_f1"].mean()],
})

print("\n" + "="*60)
print("POREDJENJE: SVI ATRIBUTI vs TOP ATRIBUTI (5-fold CV)")
print("="*60)
print(rezultati.to_string(index=False))

rezultati.to_json("results/metrics/feature_selection_poredjenje.json", orient="records", indent=4)

# Grafik poredjenja
df_plot = rezultati.melt(
    id_vars="Skup atributa",
    value_vars=["Tacnost", "Preciznost", "Odziv", "F1"],
    var_name="Metrika", value_name="Vrednost"
)
plt.figure(figsize=(9, 6))
sns.barplot(data=df_plot, x="Metrika", y="Vrednost", hue="Skup atributa", palette="Set2")
plt.title("Poredjenje: svi atributi vs top atributi")
plt.ylim(0, 1)
plt.tight_layout()
plt.savefig("results/figures/feature_selection_poredjenje.png")
plt.close()

print("\nGrafici sacuvani u results/figures/")