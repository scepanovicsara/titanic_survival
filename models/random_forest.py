# =============================================================
# random_forest.py
# Random Forest klasifikator
# =============================================================

from sklearn.ensemble import RandomForestClassifier

def kreiraj_model():
    """
    Kreira i vraća Random Forest model.
    
    Random Forest je ansambl metoda zasnovana na većem broju
    stabala odlučivanja. Svako stablo se trenira na različitom
    podskupu podataka, a konačna predikcija se dobija glasanjem.
    
    Hiperparametri:
    - n_estimators: broj stabala u šumi
    - random_state: osigurava reproduktivnost rezultata
    """
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
    return model