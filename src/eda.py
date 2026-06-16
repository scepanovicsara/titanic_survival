# =============================================================
# eda.py
# Eksplorativna analiza podataka (EDA)
# Graficki prikazi odnosa izmedju atributa i prezivljavanja
# =============================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("data/train.csv")

# Ekstrakcija titule za analizu
df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
df["Title"] = df["Title"].str.strip()
ceste_titule = ["Mr", "Miss", "Mrs", "Master"]
df["Title"] = df["Title"].apply(lambda x: x if x in ceste_titule else "Other")
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

# =============================================================
# 1. Prezivljavanje po polu
# =============================================================
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Sex", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po polu")
plt.xlabel("Pol")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("results/figures/eda_prezivljavanje_po_polu.png")
plt.close()

# =============================================================
# 2. Prezivljavanje po klasi
# =============================================================
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Pclass", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po klasi karte")
plt.xlabel("Klasa")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("results/figures/eda_prezivljavanje_po_klasi.png")
plt.close()

# =============================================================
# 3. Raspodela starosti
# =============================================================
plt.figure(figsize=(9, 5))
sns.histplot(data=df, x="Age", hue="Survived", bins=30, kde=True,
             palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Raspodela starosti putnika")
plt.xlabel("Starost")
plt.ylabel("Broj putnika")
plt.tight_layout()
plt.savefig("results/figures/eda_raspodela_starosti.png")
plt.close()

# =============================================================
# 4. Prezivljavanje po luci ukrcavanja
# =============================================================
plt.figure(figsize=(7, 5))
sns.countplot(data=df, x="Embarked", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po luci ukrcavanja")
plt.xlabel("Luka (C=Cherbourg, Q=Queenstown, S=Southampton)")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("results/figures/eda_prezivljavanje_po_luci.png")
plt.close()

# =============================================================
# 5. Prezivljavanje po velicini porodice
# =============================================================
plt.figure(figsize=(9, 5))
sns.countplot(data=df, x="Family_Size", hue="Survived",
              palette={0: "#e74c3c", 1: "#2ecc71"})
plt.title("Preživljavanje po veličini porodice")
plt.xlabel("Veličina porodice")
plt.ylabel("Broj putnika")
plt.legend(["Nije preživeo", "Preživeo"])
plt.tight_layout()
plt.savefig("results/figures/eda_prezivljavanje_po_porodici.png")
plt.close()

# =============================================================
# 6. Stopa prezivljavanja po polu i klasi
# =============================================================
plt.figure(figsize=(8, 5))
survival_rate = df.groupby(["Pclass", "Sex"])["Survived"].mean().unstack()
survival_rate.plot(kind="bar", color=["#3498db", "#e91e8c"], figsize=(8, 5))
plt.title("Stopa preživljavanja po klasi i polu")
plt.xlabel("Klasa karte")
plt.ylabel("Stopa preživljavanja")
plt.xticks(rotation=0)
plt.legend(["Muški", "Ženski"])
plt.tight_layout()
plt.savefig("results/figures/eda_stopa_po_klasi_i_polu.png")
plt.close()

# =============================================================
# 7. Korelaciona mapa
# =============================================================
df_num = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare", "Family_Size"]].copy()
df_num["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df_num["Age"] = df_num["Age"].fillna(df_num["Age"].median())

plt.figure(figsize=(9, 7))
sns.heatmap(df_num.corr(), annot=True, fmt=".2f",
            cmap="coolwarm", center=0, square=True)
plt.title("Korelaciona mapa atributa")
plt.tight_layout()
plt.savefig("results/figures/eda_korelaciona_mapa.png")
plt.close()

print("Svi EDA grafici sacuvani u results/figures/")