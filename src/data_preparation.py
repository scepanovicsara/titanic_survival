# =============================================================
# data_preparation.py
# Obrada dataseta - koristi se za sve modele
# =============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

def pripremi_podatke(filepath="data/train.csv", skaliraj=False):
    """
    Učitava i priprema dataset za treniranje modela.
    
    Parametri:
    - filepath: putanja do dataseta
    - skaliraj: True za KNN i SVM jer rade na osnovu rastojanja,
                False za stabla (RF, DT, GB) jer nisu osetljiva na skalu
    
    Vraća:
    - X_train, X_test, y_train, y_test, feature_names
    """

    from sklearn.model_selection import train_test_split

    # =============================================================
    # 1. UCITAVANJE
    # =============================================================
    df = pd.read_csv(filepath)

    print("="*50)
    print("PRIPREMA PODATAKA")
    print("="*50)
    print(f"Ucitano {df.shape[0]} redova, {df.shape[1]} kolona")
    print("\nNedostajuce vrednosti:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    # =============================================================
    # 2. EKSTRAKCIJA TITULE IZ IMENA
    # =============================================================
    # Titula nosi informaciju o polu, godinama i društvenom statusu
    df["Title"] = df["Name"].str.extract(r",\s*([^\.]+)\.")
    df["Title"] = df["Title"].str.strip()
    ceste_titule = ["Mr", "Miss", "Mrs", "Master"]
    df["Title"] = df["Title"].apply(
        lambda x: x if x in ceste_titule else "Other"
    )

    # =============================================================
    # 3. OBRADA NEDOSTAJUCIH VREDNOSTI
    # =============================================================

    # Age: medijanom po tituli - preciznija procena
    # jer Master (dečaci) imaju drugačiju starost od Mr.
    df["Age"] = df.groupby("Title")["Age"].transform(
        lambda x: x.fillna(x.median())
    )

    # Embarked: samo 2 prazne vrednosti, popunjavamo modom
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    # Cabin: 77% praznih vrednosti - previše za pouzdano popunjavanje
    # Kreiramo binarnu kolonu: ima/nema kabinu
    df["Has_Cabin"] = df["Cabin"].notna().astype(int)

    # =============================================================
    # 4. FEATURE ENGINEERING - novi atributi
    # =============================================================
    df["Family_Size"] = df["SibSp"] + df["Parch"] + 1
    df["Is_Alone"] = (df["Family_Size"] == 1).astype(int)

    # =============================================================
    # 5. UKLANJANJE NEPOTREBNIH KOLONA
    # =============================================================
    # Name - iskoristili smo je za titulu
    # Ticket - nema jasnu prediktivnu vrednost
    # Cabin - zamenjena sa Has_Cabin
    # PassengerId - samo identifikator
    df = df.drop(columns=["Name", "Ticket", "Cabin", "PassengerId"])

    # =============================================================
    # 6. ENKODOVANJE KATEGORIJSKIH ATRIBUTA
    # =============================================================
    # Sex: direktno mapiranje (redosled ima smisao)
    df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

    # Embarked i Title: one-hot encoding jer ne postoji
    # prirodni redosled između vrednosti
    df = pd.get_dummies(df, columns=["Embarked", "Title"], drop_first=True)

    # Pclass ostavljamo kao broj (1, 2, 3) jer redosled ima smisao

    print("\nDataset nakon obrade:")
    print(f"Kolone: {list(df.columns)}")
    print(f"Oblik: {df.shape}")

    # =============================================================
    # 7. PODELA NA TRAIN I TEST
    # =============================================================
    X = df.drop(columns=["Survived"])
    y = df["Survived"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =============================================================
    # 8. SKALIRANJE (samo za modele zasnovane na rastojanju)
    # =============================================================
    # KNN i SVM su osetljivi na skalu atributa jer rade na osnovu
    # rastojanja - atribut sa većim vrednostima bi dominirao
    # RF, DT, GB nisu osetljivi na skalu pa skaliranje nije potrebno
    if skaliraj:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        print("\nPodaci su skalirani (StandardScaler)")
    else:
        print("\nPodaci nisu skalirani")

    print(f"\nTrening skup: {X_train.shape[0]} primeraka")
    print(f"Test skup: {X_test.shape[0]} primeraka")

    return X_train, X_test, y_train, y_test, list(X.columns)


if __name__ == "__main__":
    # Test - pokreni direktno da proveriš da li radi
    X_train, X_test, y_train, y_test, features = pripremi_podatke()
    print("\nPriprema podataka uspesna!")