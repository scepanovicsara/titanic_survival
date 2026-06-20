# =============================================================
# preprocessing.py
# Priprema jednog novog unosa (putnika) za predikciju,
# na isti nacin kao sto su pripremani trening podaci
# =============================================================

import pandas as pd
import json
import os
import re

def ucitaj_kolone():
    putanja = os.path.join(os.path.dirname(__file__), "feature_columns.json")
    with open(putanja) as f:
        return json.load(f)

def izvuci_titulu(ime):
    """Izvlaci titulu iz imena, isto kao u data_preparation.py"""
    if not ime:
        return "Other"
    match = re.search(r",\s*([^\.]+)\.", ime)
    if not match:
        return "Other"
    titula = match.group(1).strip()
    return titula if titula in ["Mr", "Miss", "Mrs", "Master"] else "Other"

def pripremi_unos(pclass, sex, age, sibsp, parch, fare, embarked, ime, ima_kabinu):
    """
    Prima sirove podatke o putniku i vraca DataFrame sa jednim
    redom, formatiran tacno onako kako model ocekuje.
    """
    feature_columns = ucitaj_kolone()

    titula = izvuci_titulu(ime)
    family_size = sibsp + parch + 1
    is_alone = 1 if family_size == 1 else 0
    sex_num = 1 if sex == "female" else 0
    has_cabin = 1 if ima_kabinu else 0

    # Title_Master ne postoji kao kolona (drop_first=True ga je
    # izbacio kao bazu) - ako je titula Master, sve Title_* kolone
    # ostaju 0, sto model interpretira kao "Master"
    red = {
        "Pclass": pclass,
        "Sex": sex_num,
        "Age": age,
        "SibSp": sibsp,
        "Parch": parch,
        "Fare": fare,
        "Has_Cabin": has_cabin,
        "Family_Size": family_size,
        "Is_Alone": is_alone,
        "Embarked_Q": 1 if embarked == "Q" else 0,
        "Embarked_S": 1 if embarked == "S" else 0,
        "Title_Miss": 1 if titula == "Miss" else 0,
        "Title_Mr": 1 if titula == "Mr" else 0,
        "Title_Mrs": 1 if titula == "Mrs" else 0,
        "Title_Other": 1 if titula == "Other" else 0,
    }

    df = pd.DataFrame([red])
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df