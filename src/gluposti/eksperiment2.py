# =============================================================
# Eksperiment 2 - Ekstrakcija deka iz kabine
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
import sys

sys.path.append(os.path.dirname(__file__))
from gluposti.experiment_tracker import sacuvaj_rezultate, prikazi_rezultate

# =============================================================
# 1. UCITAVANJE
# =============================================================

df = pd.read_csv("data/train.csv")

# =============================================================
# 2. OBRADA - ista kao pre + ekstrakcija deka
# =============================================================

# Ekstrakcija titule
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].str.strip()
česte_titule = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].apply(lambda x: x if x in česte_titule else "Other")

# Popunjavanje Age
df["Age"] = df.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# *** NOVO: Ekstrakcija deka iz kabine ***
# Kabina izgleda ovako: "C85", "B96", "A10"
# Prvo slovo = dek (A, B, C, D, E, F, G)
# Ako nema kabine -> "Unknown"
df["Deck"] = df["Cabin"].str[0]
df["Deck"] = df["Deck"].fillna("Unknown")

print("Raspodela putnika po dekovima:")
print(df["Deck"].value_counts())

# Prikaz preživljavanja po deku
print("\nPreživljavanje po deku (prosek):")
print(df.groupby("Deck")["Survived"].mean().sort_values(ascending=False))

# Family Size i Is_Alone
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# Uklanjanje kolona
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

# Enkodovanje
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df = pd.get_dummies(df, columns=["Embarked", "Title", "Deck"], drop_first=True)

# =============================================================
# 3. MODEL
# =============================================================

X = df.drop(columns=["Survived"])
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

tacnost = accuracy_score(y_test, y_pred)
preciznost = precision_score(y_test, y_pred)
odziv = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("EVALUACIJA MODELA - Eksperiment 2")
print("="*50)
print(f"Tacnost (Accuracy):     {tacnost:.4f}")
print(f"Preciznost (Precision): {preciznost:.4f}")
print(f"Odziv (Recall):         {odziv:.4f}")
print(f"F1-skor:                {f1:.4f}")

# =============================================================
# 4. GRAFICI
# =============================================================

os.makedirs("outputs", exist_ok=True)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- Grafik 1: Feature Importance ---
importances = model.feature_importances_
fi_df = pd.DataFrame({"Feature": X.columns, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 8))
sns.barplot(data=fi_df, x="Importance", y="Feature", hue="Feature",
            palette="viridis", legend=False)
plt.title("Važnost atributa - Eksperiment 2 (sa dekovima)")
plt.tight_layout()
plt.savefig("outputs/feature_importance_exp2.png")
plt.close()

# --- Grafik 2: Preživljavanje po polu ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po polu (0=muški, 1=ženski)")
plt.xlabel("Pol")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("outputs/prezivljavanje_po_polu_exp2.png")
plt.close()

# --- Grafik 3: Preživljavanje po klasi ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po klasi karte")
plt.xlabel("Klasa")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("outputs/prezivljavanje_po_klasi_exp2.png")
plt.close()

# --- Grafik 4: Raspodela starosti ---
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
             palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Raspodela starosti putnika")
plt.xlabel("Starost")
plt.ylabel("Broj putnika")
plt.tight_layout()
plt.savefig("outputs/raspodela_starosti_exp2.png")
plt.close()

# --- Grafik 5: Matrica konfuzije ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Nije preživeo", "Preživeo"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Matrica konfuzije - Eksperiment 2")
plt.tight_layout()
plt.savefig("outputs/matrica_konfuzije_exp2.png")
plt.close()

# --- Grafik 6: Preživljavanje po deku (NOVO) ---
deck_col = [col for col in df.columns if col.startswith("Deck_")]
if deck_col:
    # Rekonstruišemo dek kolonu za vizuelizaciju
    df_temp = pd.read_csv("data/train.csv")
    df_temp["Deck"] = df_temp["Cabin"].str[0].fillna("Unknown")
    
    plt.figure(figsize=(8, 4))
    sns.countplot(data=df_temp, x="Deck", hue="Survived",
                  order=sorted(df_temp["Deck"].unique()),
                  palette={0: "#e74c3c", 1: "#2ecc71"})
    plt.title("Preživljavanje po deku")
    plt.xlabel("Dek")
    plt.ylabel("Broj putnika")
    plt.legend(["Nije preživeo", "Preživeo"])
    plt.tight_layout()
    plt.savefig("outputs/prezivljavanje_po_deku_exp2.png")
    plt.close()

print("\nSvi grafici sacuvani!")

# =============================================================
# 5. CUVANJE REZULTATA
# =============================================================

sacuvaj_rezultate("Eksperiment 2 - Sa dekovima iz kabine", {
    "tacnost": tacnost,
    "preciznost": preciznost,
    "odziv": odziv,
    "f1": f1
})

prikazi_rezultate()