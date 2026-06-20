# =============================================================
# anomaly_detection.py
# Detekcija anomalija u podacima
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
import os

os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("data/train.csv")
df["Family_Size"] = df["SibSp"] + df["Parch"] + 1

# =============================================================
# 1. IQR METODA - statisticka detekcija
# =============================================================
# IQR (Interquartile Range) metoda obelezava kao anomaliju
# svaku vrednost koja je daleko ispod Q1 ili iznad Q3
# (van "brkova" na boxplotu)

def iqr_anomalije(kolona):
    Q1 = df[kolona].quantile(0.25)
    Q3 = df[kolona].quantile(0.75)
    IQR = Q3 - Q1
    donja = Q1 - 1.5 * IQR
    gornja = Q3 + 1.5 * IQR
    broj = ((df[kolona] < donja) | (df[kolona] > gornja)).sum()
    print(f"{kolona}: normalan opseg [{donja:.2f}, {gornja:.2f}] -> {broj} anomalija")
    return donja, gornja

print("="*50)
print("IQR METODA - STATISTICKA DETEKCIJA ANOMALIJA")
print("="*50)
iqr_anomalije("Age")
iqr_anomalije("Fare")
iqr_anomalije("Family_Size")

# =============================================================
# 2. ISOLATION FOREST - algoritamska detekcija
# =============================================================
# Isolation Forest izoluje tacke kroz nasumicne podele po
# atributima. Tacke koje se izoluju u malom broju podela
# (lakse ih je "odvojiti" od ostatka) smatraju se anomalijama.

numericke_kolone = ["Age", "Fare", "SibSp", "Parch", "Family_Size"]
df_iso = df[numericke_kolone].copy()
df_iso["Age"] = df_iso["Age"].fillna(df_iso["Age"].median())

iso = IsolationForest(contamination=0.05, random_state=42)
df["Anomaly"] = iso.fit_predict(df_iso)
# -1 = anomalija, 1 = normalna tacka

broj_anomalija = (df["Anomaly"] == -1).sum()
print(f"\nIsolation Forest: {broj_anomalija} anomalija od {len(df)} putnika "
      f"({broj_anomalija/len(df)*100:.1f}%)")

# Grafik - Age vs Fare, anomalije oznacene crvenom bojom
plt.figure(figsize=(9, 6))
sns.scatterplot(data=df, x="Age", y="Fare", hue="Anomaly",
                 palette={1: "#2ecc71", -1: "#e74c3c"}, alpha=0.7)
plt.title("Detekcija anomalija (Isolation Forest)")
plt.xlabel("Starost")
plt.ylabel("Cena karte")
plt.legend(title="", labels=["Anomalija", "Normalna vrednost"])
plt.tight_layout()
plt.savefig("results/figures/anomalije_isolation_forest.png")
plt.close()

print("\nPrimer otkrivenih anomalija (prvih 10):")
print(df[df["Anomaly"] == -1][["Name", "Age", "Fare", "Pclass", "SibSp", "Parch"]].head(10))

# =============================================================
# 3. ODLUKA - sta radimo sa otkrivenim anomalijama
# =============================================================

print("\n" + "="*50)
print("ODLUKA")
print("="*50)
print("""
Anomalije se NE uklanjaju iz dataseta.

Razlog: Ekstremne vrednosti u koloni Fare najcesce odgovaraju
putnicima prve klase visokog imovnog stanja (npr. cele porodice
koje su kupile vise skupih karata odjednom), sto je stvaran i
koristan signal za predikciju prezivljavanja, a ne greska u
podacima. Uklanjanje ovih putnika bi smanjilo dataset i moglo
bi da ukloni bas one primere koji najjasnije pokazuju vezu
izmedju imovnog stanja/klase i prezivljavanja.
""")