# =============================================================
# gradient_boosting.py
# Gradient Boosting klasifikator
# =============================================================

from sklearn.ensemble import GradientBoostingClassifier

def kreiraj_model():
    """
    Kreira i vraća Gradient Boosting model.
    
    Gradient Boosting sekvencijalno gradi model dodavanjem stabala
    odlučivanja, gde svako sledeće stablo ispravlja greške
    prethodnih. Za razliku od Random Foresta gde stabla nastaju
    paralelno i nezavisno, ovde svako stablo zavisi od prethodnog.
    
    Hiperparametri:
    - n_estimators: broj stabala koja se dodaju sekvencijalno
    - learning_rate: koliko svako novo stablo utiče na konačnu
      predikciju - manji learning_rate zahteva više stabala
      ali često daje bolje rezultate
    - random_state: osigurava reproduktivnost rezultata
    """
    model = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )
    return model