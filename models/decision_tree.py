# =============================================================
# decision_tree.py
# Stablo odlučivanja klasifikator
# =============================================================

from sklearn.tree import DecisionTreeClassifier

def kreiraj_model():
    """
    Kreira i vraća Decision Tree model.
    
    Stablo odlučivanja deli podatke kroz niz pitanja nad atributima.
    Svaki unutrašnji čvor predstavlja pitanje, svaka grana ishod,
    a svaki list konačnu predikciju.
    
    Hiperparametri:
    - max_depth: maksimalna dubina stabla - ograničava overfitting
    - random_state: osigurava reproduktivnost rezultata
    """
    model = DecisionTreeClassifier(
        max_depth=5,
        random_state=42
    )
    return model