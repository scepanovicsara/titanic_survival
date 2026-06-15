# =============================================================
# Titanic Survival Classification
# Predmetni projekat - SAUS
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

import os

# =============================================================
# 1. UCITAVANJE PODATAKA
# =============================================================

df = pd.read_csv("data/train.csv")

print("="*50)
print("PREGLED DATASETA")
print("="*50)
print(f"Broj redova: {df.shape[0]}, Broj kolona: {df.shape[1]}")
print("\nPrvih 5 redova:")
print(df.head())
print("\nNedostajuce vrednosti po kolonama:")
print(df.isnull().sum())

# =============================================================
# 2. OBRADA PODATAKA
# =============================================================

# --- Ekstrakcija titule iz imena ---
# Iz kolone Name izvlacimo titulu (Mr., Mrs., Miss. itd.)
# jer titula nosi informaciju o polu, godinama i statusu
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].str.strip()

# Grupišemo retke titule u kategoriju "Other"
česte_titule = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].apply(lambda x: x if x in česte_titule else "Other")

# --- Popunjavanje nedostajucih vrednosti ---

# Age: popunjavamo medijanom po tituli
# (Mr. ima drugaciju prosecnu starost od Master. ili Miss.)
df["Age"] = df.groupby("Title")["Age"].transform(
    lambda x: x.fillna(x.median())
)

# Embarked: samo 2 nedostajuce vrednosti, popunjavamo modom (najcescom vrednoscu)
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Cabin: veliki broj nedostajucih vrednosti (~77%)
# Umesto da pokusamo da popunimo, pravimo binarnu kolonu:
# 1 = ima kabinu, 0 = nema kabinu
df["Has_Cabin"] = df["Cabin"].notna().astype(int)

# --- Kreiranje novih atributa ---

# Velicina porodice = SibSp + Parch + sam putnik
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

# Da li putuje sam?
df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

# --- Uklanjanje nepotrebnih kolona ---
# Name, Ticket, Cabin - ne koristimo direktno
# PassengerId - samo identifikator, nema prediktivnu vrednost
df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

# --- Enkodovanje kategorijskih atributa ---
# Sex: male -> 0, female -> 1
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

# Embarked i Title: one-hot encoding
df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

print("\nDataset nakon obrade:")
print(df.head())
print(f"\nKolone: {list(df.columns)}")

# =============================================================
# 3. PRIPREMA ZA MODEL
# =============================================================

X = df.drop(columns=["Survived"])
y = df["Survived"]

# Podela na trening (80%) i test (20%) skup
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =============================================================
# 4. TRENIRANJE MODELA
# =============================================================

# Koristimo Random Forest - ansambl metoda koja kombinuje vise
# stabala odlucivanja, robusna je i daje feature importance
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# =============================================================
# 5. EVALUACIJA MODELA
# =============================================================

y_pred = model.predict(X_test)

tacnost = accuracy_score(y_test, y_pred)
preciznost = precision_score(y_test, y_pred)
odziv = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*50)
print("EVALUACIJA MODELA")
print("="*50)
print(f"Tacnost (Accuracy):   {tacnost:.4f}")
print(f"Preciznost (Precision): {preciznost:.4f}")
print(f"Odziv (Recall):       {odziv:.4f}")
print(f"F1-skor:              {f1:.4f}")

# Sacuvaj rezultate eksperimenta
from gluposti.experiment_tracker import sacuvaj_rezultate, prikazi_rezultate

sacuvaj_rezultate("Eksperiment 1b - Baseline (bez kabine) ali 200 stabala", {
    "tacnost": tacnost,
    "preciznost": preciznost,
    "odziv": odziv,
    "f1": f1
})

prikazi_rezultate()

# =============================================================
# 6. VIZUELIZACIJE
# =============================================================

os.makedirs("outputs", exist_ok=True)

# --- Grafik 1: Preživljavanje po polu ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Sex", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po polu (0=muški, 1=ženski)")
plt.xlabel("Pol")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("outputs/prezivljavanje_po_polu.png")
plt.close()

# --- Grafik 2: Preživljavanje po klasi ---
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Pclass", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po klasi karte")
plt.xlabel("Klasa")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("outputs/prezivljavanje_po_klasi.png")
plt.close()

# --- Grafik 3: Raspodela starosti ---
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
             palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Raspodela starosti putnika")
plt.xlabel("Starost")
plt.ylabel("Broj putnika")
plt.tight_layout()
plt.savefig("outputs/raspodela_starosti.png")
plt.close()

# --- Grafik 4: Feature Importance ---
importances = model.feature_importances_
feature_names = X.columns
fi_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
fi_df = fi_df.sort_values("Importance", ascending=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=fi_df, x="Importance", y="Feature", palette="viridis")
plt.title("Važnost atributa (Feature Importance)")
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()

# --- Grafik 5: Matrica konfuzije ---
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["Nije preživeo", "Preživeo"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, colorbar=False, cmap="Blues")
plt.title("Matrica konfuzije")
plt.tight_layout()
plt.savefig("outputs/matrica_konfuzije.png")
plt.close()

print("\nSvi grafici su sacuvani u folderu 'outputs/'")
