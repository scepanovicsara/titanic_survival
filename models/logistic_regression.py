# =============================================================
# logistic_regression.py
# Logistic Regression klasifikator
# =============================================================

from sklearn.linear_model import LogisticRegression

def kreiraj_model():
    """
    Kreira i vraća Logistic Regression model.
    
    Uprkos nazivu, Logistic Regression se koristi za klasifikaciju.
    Model računa verovatnoću pripadnosti klasi koristeći logističku
    funkciju (sigmoid), koja vrednosti preslikava u opseg [0, 1].
    Ako je verovatnoća veća od 0.5, predikcija je klasa 1
    (preživeo), inače klasa 0 (nije preživeo).
    
    Logistic Regression je jedan od najjednostavnijih klasifikacionih
    modela i često se koristi kao baseline - polazna tačka sa kojom
    se porede složeniji modeli.
    
    Važno: osetljiv je na skalu atributa, pa se koristi
    skaliranje podataka (StandardScaler).
    
    Hiperparametri:
    - max_iter: maksimalan broj iteracija za konvergenciju
    - random_state: osigurava reproduktivnost rezultata
    """
    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    return model